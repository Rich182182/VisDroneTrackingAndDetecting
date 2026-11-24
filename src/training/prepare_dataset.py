import shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


# Resolve project root dynamically: .../VisDroneTrackingAndDetecting/src/training/prepare_dataset.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# UAV target classes mapping: VisDrone MOT category ID -> contiguous YOLO class ID (0..6)
# 1:pedestrian->0, 2:people->1, 4:car->2, 5:van->3, 6:truck->4, 9:bus->5, 10:motor->6
CLASS_MAPPING = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 9: 5, 10: 6}


def resolve_project_path(path_str: str | Path) -> Path:
    """Resolves a path relative to PROJECT_ROOT unless it is already absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def parse_mot_annotations(ann_file: Path) -> dict[int, list[tuple]]:
    """
    Reads a VisDrone-MOT sequence .txt file and groups valid target boxes by frame index.
    Format: <frame_idx>,<target_id>,<left>,<top>,<width>,<height>,<score>,<category>,<trunc>,<occ>
    """
    frame_annotations = defaultdict(list)
    if not ann_file.exists():
        return frame_annotations

    with open(ann_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue

            frame_idx = int(parts[0])
            left = float(parts[2])
            top = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            score = int(parts[6])
            category = int(parts[7])

            # Skip ignored regions (score == 0) and non-target classes
            if score == 0 or category not in CLASS_MAPPING:
                continue

            yolo_cls = CLASS_MAPPING[category]
            frame_annotations[frame_idx].append((yolo_cls, left, top, width, height))

    return frame_annotations


def convert_split(raw_dir: Path, output_dir: Path, split: str, frame_step: int = 5):
    """
    Converts a single VisDrone-MOT split (train or val) into flat YOLO images/ and labels/ directories.
    """
    sequences_root = raw_dir / "sequences"
    annotations_root = raw_dir / "annotations"

    if not sequences_root.exists() or not annotations_root.exists():
        raise FileNotFoundError(
            f"Expected 'sequences' and 'annotations' inside '{raw_dir}', but one or both are missing."
        )

    out_img_dir = output_dir / "images" / split
    out_lbl_dir = output_dir / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    sequences = sorted([d.name for d in sequences_root.iterdir() if d.is_dir()])

    total_images = 0
    total_boxes = 0

    for seq_name in tqdm(sequences, desc=f"Converting [{split}]", unit="seq"):
        seq_dir = sequences_root / seq_name
        ann_file = annotations_root / f"{seq_name}.txt"

        frame_anns = parse_mot_annotations(ann_file)
        frame_files = sorted([f.name for f in seq_dir.iterdir() if f.suffix.lower() in (".jpg", ".png")])

        # Subsample every N-th frame to prevent near-duplicate 30-FPS redundancy
        sampled_frames = frame_files[::frame_step]

        for fname in sampled_frames:
            frame_stem = Path(fname).stem
            frame_idx = int(frame_stem)
            src_img_path = seq_dir / fname

            with Image.open(src_img_path) as img:
                img_w, img_h = img.size

            # Prefix with sequence name to avoid filename collisions across sequences
            unique_name = f"{seq_name}_{frame_stem}"
            dst_img_path = out_img_dir / f"{unique_name}.jpg"
            dst_lbl_path = out_lbl_dir / f"{unique_name}.txt"

            shutil.copy2(src_img_path, dst_img_path)

            yolo_lines = []
            for yolo_cls, left, top, w, h in frame_anns.get(frame_idx, []):
                if w <= 0 or h <= 0:
                    continue

                # Convert top-left pixels (left, top, w, h) to normalized YOLO center (xc, yc, w, h)
                x_center = float(np.clip((left + w / 2.0) / img_w, 0.0, 1.0))
                y_center = float(np.clip((top + h / 2.0) / img_h, 0.0, 1.0))
                w_norm = float(np.clip(w / img_w, 0.0, 1.0))
                h_norm = float(np.clip(h / img_h, 0.0, 1.0))

                if w_norm > 0 and h_norm > 0:
                    yolo_lines.append(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                    total_boxes += 1

            with open(dst_lbl_path, "w", encoding="utf-8") as out_f:
                out_f.writelines(yolo_lines)

            total_images += 1

    print(f"[{split.upper()}] Saved {total_images} images and {total_boxes} bounding boxes to {output_dir}")


def write_dataset_yaml(dataset_dir: Path, yaml_path: Path):
    """Generates data/VisDrone_UAV.yaml with the exact resolved dataset path for Ultralytics."""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_content = (
        f"path: {dataset_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n"
        f"  0: pedestrian\n"
        f"  1: people\n"
        f"  2: car\n"
        f"  3: van\n"
        f"  4: truck\n"
        f"  5: bus\n"
        f"  6: motor\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"Dataset YAML config generated at: {yaml_path}")


def build_yolo_dataset(
    train_raw_dir: str = "data/VisDrone2019-MOT-train",
    val_raw_dir: str = "data/VisDrone2019-MOT-val",
    output_dir: str = "data/VisDrone_UAV_Clean",
    yaml_output: str = "data/VisDrone_UAV.yaml",
    frame_step: int = 5
):
    """Builds clean train/val YOLO detection splits and writes the dataset YAML config."""
    train_path = resolve_project_path(train_raw_dir)
    val_path = resolve_project_path(val_raw_dir)
    out_path = resolve_project_path(output_dir)
    yaml_path = resolve_project_path(yaml_output)

    print(f"Project root resolved to: {PROJECT_ROOT}")
    convert_split(raw_dir=train_path, output_dir=out_path, split="train", frame_step=frame_step)
    convert_split(raw_dir=val_path, output_dir=out_path, split="val", frame_step=frame_step)
    write_dataset_yaml(dataset_dir=out_path, yaml_path=yaml_path)