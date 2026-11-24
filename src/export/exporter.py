from __future__ import annotations

import gc
import inspect
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnx
import tensorrt as trt
import torch
import yaml
from modelopt.onnx.autocast import convert_to_mixed_precision
from modelopt.onnx.quantization import quantize as modelopt_quantize
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(path_str: str | Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

def _letterbox_preprocess(img_bgr: np.ndarray, target_size: int) -> np.ndarray:
    """Resize and pad an image using Ultralytics-compatible letterbox preprocessing."""
    h, w = img_bgr.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_size - new_w) / 2.0
    pad_h = (target_size - new_h) / 2.0
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw)


def _resolve_yaml_root(yaml_path: Path, raw_root: str | None) -> Path:
    """Resolve dataset root directory declared in a YOLO data.yaml file."""
    if not raw_root:
        return yaml_path.parent

    root_candidate = Path(raw_root)
    if root_candidate.is_absolute() and root_candidate.exists():
        return root_candidate

    for base in (Path.cwd(), yaml_path.parent):
        resolved = (base / root_candidate).resolve()
        if resolved.exists():
            return resolved

    return root_candidate


def _gather_images_from_entry(entry_path: Path) -> list[Path]:
    """Collect image paths from either a directory or a newline-delimited .txt list."""
    if entry_path.is_dir():
        return sorted(
            p for p in entry_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    if entry_path.is_file() and entry_path.suffix.lower() == ".txt":
        images: list[Path] = []
        for line in entry_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            img_path = Path(line)
            if not img_path.is_absolute():
                img_path = (entry_path.parent / img_path).resolve()
            if img_path.exists():
                images.append(img_path)
        return sorted(images)

    return []


def _collect_calibration_tensor(
    data_yaml: str | None,
    img_size: int,
    max_images: int = 160,
    seed: int = 42,
) -> np.ndarray:
    """Build a calibrated NCHW float32 tensor from representative dataset frames."""
    images: list[Path] = []

    if data_yaml:
        yaml_path = Path(data_yaml).resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"Calibration data.yaml not found: {yaml_path}")

        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        dataset_root = _resolve_yaml_root(yaml_path, cfg.get("path"))

        for split_key in ("val", "train"):
            split_val = cfg.get(split_key)
            if not split_val:
                continue
            entries = split_val if isinstance(split_val, list) else [split_val]
            for entry in entries:
                entry_path = Path(entry)
                if not entry_path.is_absolute():
                    entry_path = (dataset_root / entry_path).resolve()
                images.extend(_gather_images_from_entry(entry_path))
            if images:
                break

    if not images:
        fallback_dir = Path("data")
        if fallback_dir.exists():
            images = sorted(
                p for p in fallback_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )

    if not images:
        raise FileNotFoundError(
            "No calibration images found. Pass a valid 'data_yaml' path or place frames under 'data/'."
        )

    step = max(1, len(images) // max_images)
    sampled = images[::step][:max_images]
    rng = random.Random(seed)
    rng.shuffle(sampled)

    tensors: list[np.ndarray] = []
    for img_path in sampled:
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read calibration image: {img_path}")
        tensors.append(_letterbox_preprocess(img, img_size))

    return np.stack(tensors, axis=0)


def _convert_onnx_fp16_autocast(
    fp32_onnx_path: Path,
    fp16_onnx_path: Path,
    calib_npz_path: Path,
) -> None:
    """Convert FP32 ONNX to strongly-typed FP16 mixed-precision ONNX using ModelOpt AutoCast."""
    sig_params = inspect.signature(convert_to_mixed_precision).parameters
    candidate_kwargs = {
        "onnx_path": str(fp32_onnx_path),
        "low_precision_type": "fp16",
        "calibration_data": str(calib_npz_path),
        "op_types_to_exclude": ["Range"],
        "keep_io_types": True,
        "output_path": str(fp16_onnx_path),
    }

    if "onnx_path" in sig_params:
        filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig_params}
        result = convert_to_mixed_precision(**filtered_kwargs)
    else:
        candidate_kwargs.pop("onnx_path")
        filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig_params}
        result = convert_to_mixed_precision(str(fp32_onnx_path), **filtered_kwargs)

    if not fp16_onnx_path.exists() and result is not None:
        onnx.save(result, str(fp16_onnx_path))


def _quantize_onnx_explicit_int8(
    fp32_onnx_path: Path,
    qdq_onnx_path: Path,
    input_name: str,
    calib_tensor: np.ndarray,
    head_prefix: str = "model.22",
) -> None:
    """Generate an explicit QDQ INT8 ONNX graph via ModelOpt while keeping the detection head in FP16."""
    onnx_model = onnx.load(str(fp32_onnx_path))
    head_nodes = [
        node.name
        for node in onnx_model.graph.node
        if node.name and (head_prefix in node.name or "decoder" in node.name.lower())
    ]
    del onnx_model

    print(
        f"Calibrating INT8 on {len(calib_tensor)} frames "
        f"(excluding {len(head_nodes)} detection head nodes matching '{head_prefix}')..."
    )

    sig_params = inspect.signature(modelopt_quantize).parameters
    candidate_kwargs = {
        "onnx_path": str(fp32_onnx_path),
        "quantize_mode": "int8",
        "calibration_data": {input_name: calib_tensor},
        "calibration_method": "entropy",
        "calibration_eps": ["cuda:0", "cpu"],
        "nodes_to_exclude": head_nodes,
        "high_precision_dtype": "fp16",
        "output_path": str(qdq_onnx_path),
        "simplify": False,
    }

    if "onnx_path" in sig_params:
        filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig_params}
        modelopt_quantize(**filtered_kwargs)
    else:
        candidate_kwargs.pop("onnx_path")
        filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig_params}
        modelopt_quantize(str(fp32_onnx_path), **filtered_kwargs)

    gc.collect()
    torch.cuda.empty_cache()


def _build_strongly_typed_trt_engine(
    onnx_path: Path,
    engine_path: Path,
    workspace_gb: int,
    metadata: dict,
) -> None:
    """Compile a strongly-typed TensorRT 11+ engine from a pre-typed ONNX graph."""
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, trt_logger)

    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parsing failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build engine from {onnx_path.name}.")

    # Prepend Ultralytics JSON metadata header so YOLO() loads task, imgsz, and class names automatically
    meta_bytes = json.dumps(metadata).encode("utf-8")
    with open(engine_path, "wb") as f:
        f.write(len(meta_bytes).to_bytes(4, byteorder="little", signed=True))
        f.write(meta_bytes)
        f.write(serialized)

    del serialized, builder, network, config
    gc.collect()
    torch.cuda.empty_cache()


def export_model_suite(
    model_path: str,
    output_dir: str = "weights/exported",
    precisions: tuple[str, ...] = ("fp16", "int8"),
    imgsz: int = 800,
    data_yaml: str | None = None,
    workspace_gb: int = 4,
    batch_size: int = 1,
    cleanup_artifacts: bool = True,
) -> dict[str, str]:
    src_path = _resolve_project_path(model_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Source model not found: {src_path}")

    out_dir_path = _resolve_project_path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    if data_yaml:
        data_yaml = str(_resolve_project_path(data_yaml))
    stem = src_path.stem
    exported_paths: dict[str, str] = {}

    temp_fp32_onnx = out_dir_path / f"{stem}_{imgsz}_fp32_temp.onnx"
    temp_fp16_onnx = out_dir_path / f"{stem}_{imgsz}_fp16_temp.onnx"
    temp_qdq_onnx = out_dir_path / f"{stem}_{imgsz}_qdq_temp.onnx"
    temp_calib_npz = out_dir_path / f"{stem}_{imgsz}_calib_temp.npz"

    # Export clean FP32 ONNX graph once
    gc.collect()
    torch.cuda.empty_cache()
    model = YOLO(str(src_path))
    raw_stride = getattr(model.model, "stride", None)
    stride = int(max(raw_stride)) if raw_stride is not None else 32

    # Automatically resolve final detection head prefix (model.22 for YOLOv8, model.23 for YOLO11/10, model.28 for RT-DETR)
    layers_seq = getattr(model.model, "model", None)
    head_prefix = f"model.{len(layers_seq) - 1}" if layers_seq is not None else "model.22"

    metadata = {
        "task": model.task,
        "names": {int(k): str(v) for k, v in model.names.items()},
        "imgsz": [imgsz, imgsz],
        "batch": batch_size,
        "stride": stride,
        "channels": 3,
    }

    raw_onnx = model.export(
        format="onnx",
        imgsz=imgsz,
        batch=batch_size,
        opset=17,
        simplify=True,
        device=0,
    )
    shutil.move(str(raw_onnx), str(temp_fp32_onnx))
    del model
    gc.collect()
    torch.cuda.empty_cache()

    onnx_meta = onnx.load(str(temp_fp32_onnx), load_external_data=False)
    input_name = onnx_meta.graph.input[0].name
    del onnx_meta

    calib_tensor = _collect_calibration_tensor(data_yaml=data_yaml, img_size=imgsz, max_images=160)
    np.savez(temp_calib_npz, **{input_name: calib_tensor[:8]})

    try:
        for prec in precisions:
            prec_clean = prec.lower().strip()
            if prec_clean not in {"fp16", "int8"}:
                raise ValueError(f"Unknown format '{prec}'.")

            print(f"\nExporting {src_path.name} -> {prec_clean.upper()} (imgsz={imgsz})...")
            gc.collect()
            torch.cuda.empty_cache()

            target_file = out_dir_path / f"{stem}_{prec_clean}_{imgsz}.engine"

            if prec_clean == "fp16":
                _convert_onnx_fp16_autocast(
                    fp32_onnx_path=temp_fp32_onnx,
                    fp16_onnx_path=temp_fp16_onnx,
                    calib_npz_path=temp_calib_npz,
                )
                source_onnx = temp_fp16_onnx
            else:
                _quantize_onnx_explicit_int8(
                    fp32_onnx_path=temp_fp32_onnx,
                    qdq_onnx_path=temp_qdq_onnx,
                    input_name=input_name,
                    calib_tensor=calib_tensor,
                    head_prefix=head_prefix,
                )
                source_onnx = temp_qdq_onnx

            _build_strongly_typed_trt_engine(
                onnx_path=source_onnx,
                engine_path=target_file,
                workspace_gb=workspace_gb,
                metadata=metadata,
            )

            exported_paths[prec_clean] = str(target_file).replace("\\", "/")
            print(f"Done: {exported_paths[prec_clean]}")
    finally:
        del calib_tensor
        if cleanup_artifacts:
            for artifact in (
                temp_fp32_onnx,
                temp_fp16_onnx,
                temp_qdq_onnx,
                temp_calib_npz,
                src_path.with_suffix(".onnx"),
                src_path.with_suffix(".cache"),
            ):
                if artifact.exists():
                    artifact.unlink()

    gc.collect()
    torch.cuda.empty_cache()
    return exported_paths