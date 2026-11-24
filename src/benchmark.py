import os
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms
import supervision as sv
from ultralytics import YOLO
from tqdm.auto import tqdm
from src.trackers.factory import create_tracker
from src.metrics import evaluate_dataset_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NAME_TO_VISDRONE_GT = {
    "person": 1,
    "pedestrian": 1,
    "people": 2,
    "car": 4,
    "van": 5,
    "truck": 6,
    "bus": 9,
    "motor": 10,
    "motorcycle": 10,
}


def _resolve_project_path(path_str: str | Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_ignored_regions(gt_path: str | Path) -> dict[int, np.ndarray]:
    cols = ['frame', 'id', 'left', 'top', 'width', 'height', 'score', 'class', 'trunc', 'occ']
    df = pd.read_csv(gt_path, header=None, names=cols)
    ignored = df[(df['score'] == 0) | (df['class'] == 0)]
    regions = {}
    for frame_id, group in ignored.groupby('frame'):
        boxes = group[['left', 'top', 'width', 'height']].values.astype(float).copy()
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        regions[int(frame_id)] = boxes
    return regions


def _gpu_letterbox(frame_bgr: np.ndarray, imgsz: int, device: str, half: bool = False) -> torch.Tensor:
    """Resizes and pads a BGR frame on GPU to square letterbox format."""
    orig_h, orig_w = frame_bgr.shape[:2]
    dtype = torch.float16 if half else torch.float32

    tensor = torch.from_numpy(frame_bgr).to(device=device, non_blocking=True)
    tensor = tensor.permute(2, 0, 1).flip(0).unsqueeze(0).to(dtype=dtype).div_(255.0)

    scale = min(imgsz / orig_h, imgsz / orig_w)
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    if (new_w, new_h) != (orig_w, orig_h):
        tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    pad_w = (imgsz - new_w) / 2.0
    pad_h = (imgsz - new_h) / 2.0
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

    return F.pad(tensor, (left, right, top, bottom), mode="constant", value=114.0 / 255.0).contiguous()


def _rescale_boxes(boxes_xyxy: np.ndarray, orig_shape: tuple[int, int], imgsz: int) -> np.ndarray:
    if len(boxes_xyxy) == 0:
        return boxes_xyxy
    orig_h, orig_w = orig_shape
    gain = min(imgsz / orig_h, imgsz / orig_w)
    new_w, new_h = int(round(orig_w * gain)), int(round(orig_h * gain))
    pad_x = int(round((imgsz - new_w) / 2.0 - 0.1))
    pad_y = int(round((imgsz - new_h) / 2.0 - 0.1))

    scaled = boxes_xyxy.copy()
    scaled[:, [0, 2]] = np.clip((scaled[:, [0, 2]] - pad_x) / gain, 0, orig_w)
    scaled[:, [1, 3]] = np.clip((scaled[:, [1, 3]] - pad_y) / gain, 0, orig_h)
    return scaled


def _fast_engine_infer(
    model: YOLO,
    frame_bgr: np.ndarray,
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    device: str,
    is_rtdetr: bool = False,
    max_det: int = 300,
) -> sv.Detections:
    """Runs GPU-accelerated tensor inference and post-processing for YOLO and RT-DETR models."""
    backend = model.predictor.model
    half = bool(getattr(backend, "fp16", False))
    orig_h, orig_w = frame_bgr.shape[:2]

    if is_rtdetr:
        dtype = torch.float16 if half else torch.float32
        resized_bgr = cv2.resize(frame_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized_bgr).to(device=device, non_blocking=True)
        tensor = tensor.permute(2, 0, 1).flip(0).unsqueeze(0).to(dtype=dtype).div_(255.0)
    else:
        tensor = _gpu_letterbox(frame_bgr, imgsz=imgsz, device=device, half=half)

    with torch.inference_mode():
        raw_out = backend.forward(tensor)

    preds = raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out
    preds = preds.squeeze(0).float()

    if is_rtdetr:
        if preds.ndim == 2 and preds.shape[1] == 6:
            scores = preds[:, 4]
            class_ids = preds[:, 5].long()
        else:
            scores, class_ids = preds[:, 4:].max(dim=1)

        mask = scores > conf_thres
        if not torch.any(mask):
            return sv.Detections.empty()

        boxes_cxcywh = preds[mask, :4][:max_det]
        scores = scores[mask][:max_det]
        class_ids = class_ids[mask][:max_det]

        boxes_xyxy = torch.empty_like(boxes_cxcywh)
        boxes_xyxy[:, 0] = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] * 0.5) * orig_w
        boxes_xyxy[:, 1] = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] * 0.5) * orig_h
        boxes_xyxy[:, 2] = (boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] * 0.5) * orig_w
        boxes_xyxy[:, 3] = (boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] * 0.5) * orig_h

        boxes_np = boxes_xyxy.cpu().numpy()
        boxes_np[:, [0, 2]] = np.clip(boxes_np[:, [0, 2]], 0, orig_w)
        boxes_np[:, [1, 3]] = np.clip(boxes_np[:, [1, 3]], 0, orig_h)
        return sv.Detections(
            xyxy=boxes_np,
            confidence=scores.cpu().numpy(),
            class_id=class_ids.cpu().numpy().astype(int),
        )

    if preds.ndim == 2 and preds.shape[1] == 6:
        mask = preds[:, 4] > conf_thres
        if not torch.any(mask):
            return sv.Detections.empty()
        filtered = preds[mask][:max_det]
        boxes_np = _rescale_boxes(filtered[:, :4].cpu().numpy(), (orig_h, orig_w), imgsz)
        return sv.Detections(
            xyxy=boxes_np,
            confidence=filtered[:, 4].cpu().numpy(),
            class_id=filtered[:, 5].cpu().numpy().astype(int),
        )

    preds = preds.transpose(0, 1)
    scores, class_ids = preds[:, 4:].max(dim=1)
    mask = scores > conf_thres
    if not torch.any(mask):
        return sv.Detections.empty()

    boxes_cxcywh = preds[mask, :4]
    scores = scores[mask]
    class_ids = class_ids[mask]

    boxes_xyxy = torch.empty_like(boxes_cxcywh)
    half_w = boxes_cxcywh[:, 2] * 0.5
    half_h = boxes_cxcywh[:, 3] * 0.5
    boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - half_w
    boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - half_h
    boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + half_w
    boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + half_h

    keep = batched_nms(boxes_xyxy, scores, class_ids, iou_thres)[:max_det]
    boxes_np = _rescale_boxes(boxes_xyxy[keep].cpu().numpy(), (orig_h, orig_w), imgsz)

    return sv.Detections(
        xyxy=boxes_np,
        confidence=scores[keep].cpu().numpy(),
        class_id=class_ids[keep].cpu().numpy().astype(int),
    )


def _measure_pure_gpu_ms(model: YOLO, imgsz: int, device: str, runs: int = 100) -> float:
    """Measures isolated neural network forward pass latency in milliseconds using CUDA events."""
    if not device.startswith("cuda") or getattr(model, "predictor", None) is None:
        return 0.0
    backend = model.predictor.model
    half = bool(getattr(backend, "fp16", False))
    dtype = torch.float16 if half else torch.float32
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device, dtype=dtype)

    with torch.inference_mode():
        for _ in range(20):
            _ = backend.forward(dummy)
        torch.cuda.synchronize()

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(runs):
            _ = backend.forward(dummy)
        end_evt.record()
        torch.cuda.synchronize()

    return round(start_evt.elapsed_time(end_evt) / runs, 2)


def _init_detector(model_path: Path, imgsz: int, conf_thres: float, iou_thres: float, gpu_preproc: bool, device: str):
    is_engine = model_path.suffix == ".engine"
    model = YOLO(str(model_path), task="detect")
    if not is_engine:
        model.to(device)

    if gpu_preproc and device.startswith('cuda'):
        _ = model.predict(
            source=np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
            imgsz=imgsz,
            conf=conf_thres,
            iou=iou_thres,
            device=0,
            verbose=False,
        )
        model.predictor.model.eval()
    return model


def _process_sequence(
    model: YOLO,
    model_names: dict,
    tracker_type: str,
    tracker_params: dict,
    frames_dir: Path,
    gt_path: Path,
    pred_txt_path: Path,
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    gpu_preproc: bool,
    device: str,
    is_rtdetr: bool,
    timings: dict[str, list[float]],
    show_progress: bool = False,
) -> None:
    """Processes all frames in a sequence, updating the tracker, logging latencies, and saving MOT predictions."""
    ignored_regions = load_ignored_regions(gt_path)
    tracker = create_tracker(tracker_type=tracker_type, params=tracker_params)
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
    iterator = tqdm(frame_files, desc=frames_dir.name, unit="frame", leave=False) if show_progress else frame_files

    with open(pred_txt_path, "w") as f:
        for frame_idx, frame_name in enumerate(iterator, start=1):
            t0 = time.perf_counter()
            frame = cv2.imread(str(frames_dir / frame_name))
            if frame is None:
                continue
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            if gpu_preproc and device.startswith('cuda'):
                detections = _fast_engine_infer(
                    model=model,
                    frame_bgr=frame,
                    imgsz=imgsz,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                    device=device,
                    is_rtdetr=is_rtdetr,
                )
            else:
                results = model.predict(
                    source=frame,
                    imgsz=imgsz,
                    conf=conf_thres,
                    iou=iou_thres,
                    device=0 if device.startswith('cuda') else 'cpu',
                    verbose=False,
                )[0]
                detections = sv.Detections.from_ultralytics(results)

            if device.startswith('cuda'):
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            valid_indices, mapped_classes = [], []
            for i, cls_id in enumerate(detections.class_id):
                cls_name = str(model_names.get(int(cls_id), "")).lower().strip()
                if cls_name in NAME_TO_VISDRONE_GT:
                    valid_indices.append(i)
                    mapped_classes.append(NAME_TO_VISDRONE_GT[cls_name])

            if valid_indices:
                detections = detections[valid_indices]
                detections.class_id = np.array(mapped_classes)
            else:
                detections = sv.Detections.empty()

            detections = tracker.update(detections, frame)
            t3 = time.perf_counter()

            if frame_idx > 5:
                timings["disk"].append((t1 - t0) * 1000)
                timings["infer"].append((t2 - t1) * 1000)
                timings["track"].append((t3 - t2) * 1000)
                timings["total"].append((t3 - t0) * 1000)

            if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                ign_boxes = ignored_regions.get(frame_idx, np.empty((0, 4)))
                for xyxy, conf, cls, trk_id in zip(
                    detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id
                ):
                    if int(trk_id) < 0:
                        continue
                    x1, y1, x2, y2 = xyxy
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    if len(ign_boxes) > 0 and np.any(
                        (cx >= ign_boxes[:, 0]) & (cx <= ign_boxes[:, 2]) &
                        (cy >= ign_boxes[:, 1]) & (cy <= ign_boxes[:, 3])
                    ):
                        continue
                    w, h = x2 - x1, y2 - y1
                    f.write(f"{frame_idx},{int(trk_id)},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{conf:.3f},{int(cls)},-1,-1\n")


def _save_and_print_record(
    exp_name: str,
    model_name: str,
    tracker_type: str,
    imgsz: int,
    quality_metrics: dict,
    timings: dict[str, list[float]],
    pure_gpu_ms: float,
    vram_mb: float,
    csv_path: Path,
) -> dict:
    avg_disk = float(np.mean(timings["disk"])) if timings["disk"] else 0.0
    avg_infer = float(np.mean(timings["infer"])) if timings["infer"] else 0.0
    avg_track = float(np.mean(timings["track"])) if timings["track"] else 0.0
    avg_total = float(np.mean(timings["total"])) if timings["total"] else 0.0
    jitter = float(np.std(timings["total"])) if timings["total"] else 0.0
    fps = 1000.0 / avg_total if avg_total > 0 else 0.0

    record = {
        "Experiment": exp_name,
        "Model": model_name,
        "Tracker": tracker_type,
        "IMGSZ": imgsz,
        "MOTA (%)": round(quality_metrics["mota"] * 100, 2),
        "IDF1 (%)": round(quality_metrics["idf1"] * 100, 2),
        "HOTA (%)": round(quality_metrics["hota"] * 100, 2),
        "ID Switches": int(quality_metrics["num_switches"]),
        "MT (>=80%)": int(quality_metrics["mostly_tracked"]),
        "ML (<=20%)": int(quality_metrics["mostly_lost"]),
        "Matched (TP)": int(quality_metrics["num_matches"]),
        "Missed (FN)": int(quality_metrics["num_misses"]),
        "False Pos": int(quality_metrics["num_false_positives"]),
        "Infer Latency (ms)": round(avg_infer, 2),
        "Track Latency (ms)": round(avg_track, 2),
        "Total Latency (ms)": round(avg_total, 2),
        "Jitter (ms)": round(jitter, 2),
        "FPS": round(fps, 1),
        "Peak VRAM (MB)": round(vram_mb, 1),
        "Pure GPU (ms)": pure_gpu_ms,
        "Disk Read (ms)": round(avg_disk, 2),
    }

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([record])
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        df_old = df_old[df_old["Experiment"] != exp_name]
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.to_csv(csv_path, index=False)
    print(
        f"[{exp_name}] MOTA: {record['MOTA (%)']}% | IDF1: {record['IDF1 (%)']}% | HOTA: {record['HOTA (%)']}% | "
        f"ID Sw: {record['ID Switches']} | FPS: {record['FPS']} | Pure GPU: {record['Pure GPU (ms)']}ms"
    )
    return record


def run_experiment(
    exp_name: str,
    model_path: str,
    tracker_type: str,
    tracker_params: dict,
    frames_dir: str,
    gt_path: str,
    imgsz: int = 800,
    conf_thres: float = 0.35,
    iou_thres: float = 0.45,
    gpu_preproc: bool = True,
    results_csv: str = "outputs/benchmark_results.csv",
) -> dict:
    """Runs detection, tracking, and performance profiling on a single video sequence."""
    resolved_model = _resolve_project_path(model_path)
    resolved_frames = _resolve_project_path(frames_dir)
    resolved_gt = _resolve_project_path(gt_path)
    resolved_csv = _resolve_project_path(results_csv)

    pred_dir = PROJECT_ROOT / "outputs" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_txt = pred_dir / f"pred_{exp_name}.txt"

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    is_rtdetr = "rtdetr" in resolved_model.name.lower()
    model = _init_detector(resolved_model, imgsz, conf_thres, iou_thres, gpu_preproc, device)

    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    timings = {"disk": [], "infer": [], "track": [], "total": []}
    _process_sequence(
        model, model.names, tracker_type, tracker_params,
        resolved_frames, resolved_gt, pred_txt,
        imgsz, conf_thres, iou_thres, gpu_preproc, device, is_rtdetr, timings, show_progress=True
    )

    pure_gpu_ms = _measure_pure_gpu_ms(model, imgsz=imgsz, device=device)
    vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.startswith('cuda') else 0.0
    quality_metrics = evaluate_dataset_dict([(str(resolved_gt), str(pred_txt))])

    return _save_and_print_record(
        exp_name, resolved_model.name, tracker_type, imgsz,
        quality_metrics, timings, pure_gpu_ms, vram_mb, resolved_csv
    )


def run_dataset_experiment(
    exp_name: str,
    model_path: str,
    tracker_type: str,
    tracker_params: dict,
    sequences_dir: str = "data/VisDrone2019-MOT-test-dev/sequences",
    annotations_dir: str = "data/VisDrone2019-MOT-test-dev/annotations",
    imgsz: int = 800,
    conf_thres: float = 0.35,
    iou_thres: float = 0.45,
    gpu_preproc: bool = True,
    results_csv: str = "outputs/table_results.csv",
) -> dict:
    """Runs evaluation across all sequences in a MOT dataset and logs overall average metrics."""
    resolved_model = _resolve_project_path(model_path)
    resolved_seq_root = _resolve_project_path(sequences_dir)
    resolved_ann_root = _resolve_project_path(annotations_dir)
    resolved_csv = _resolve_project_path(results_csv)

    pred_dir = PROJECT_ROOT / "outputs" / "predictions" / exp_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted([d for d in resolved_seq_root.iterdir() if d.is_dir()])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    is_rtdetr = "rtdetr" in resolved_model.name.lower()
    model = _init_detector(resolved_model, imgsz, conf_thres, iou_thres, gpu_preproc, device)

    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    timings = {"disk": [], "infer": [], "track": [], "total": []}
    gt_pred_pairs = []

    for seq_dir in tqdm(seq_dirs, desc=exp_name, unit="seq"):
        gt_path = resolved_ann_root / f"{seq_dir.name}.txt"
        if not gt_path.exists():
            continue
        pred_txt = pred_dir / f"{seq_dir.name}.txt"
        _process_sequence(
            model, model.names, tracker_type, tracker_params,
            seq_dir, gt_path, pred_txt,
            imgsz, conf_thres, iou_thres, gpu_preproc, device, is_rtdetr, timings, show_progress=False
        )
        gt_pred_pairs.append((str(gt_path), str(pred_txt)))

    pure_gpu_ms = _measure_pure_gpu_ms(model, imgsz=imgsz, device=device)
    vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.startswith('cuda') else 0.0
    quality_metrics = evaluate_dataset_dict(gt_pred_pairs)

    return _save_and_print_record(
        exp_name, resolved_model.name, tracker_type, imgsz,
        quality_metrics, timings, pure_gpu_ms, vram_mb, resolved_csv
    )