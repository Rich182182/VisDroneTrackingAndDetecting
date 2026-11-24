import shutil
from pathlib import Path
import torch
from torch.backends.mkldnn import deterministic
from ultralytics import YOLO


# Dynamically resolve project root: .../VisDroneTrackingAndDetecting/src/training/train.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path_str: str | Path) -> Path:
    """Resolves a path relative to PROJECT_ROOT unless it is already absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def train_uav_model(
    base_weights: str = "weights/yolov8s.pt",
    exp_name: str = "yolov8s_visdrone",
    data_yaml: str = "data/VisDrone_UAV.yaml",
    epochs: int = 50,
    imgsz: int = 800,
    batch: int = 8,
    lr0: float = 0.001,
    patience: int = 15,
    workers: int = 2
):
    """
    Fine-tunes a detector (YOLOv8, YOLOv10, YOLO11, or RT-DETR) on the cleaned VisDrone UAV subset
    and copies the best checkpoint to the weights/ directory.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Reinstall PyTorch with CUDA support.")
    device = 0

    weights_path = resolve_project_path(base_weights)
    yaml_path = resolve_project_path(data_yaml)
    runs_project_dir = PROJECT_ROOT / "runs" / "uav_training"
    final_weights_dst = PROJECT_ROOT / "weights" / f"{exp_name}.pt"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found at: {yaml_path}. Run prepare_dataset.py first.")

    device = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"TRAINING EXPERIMENT: {exp_name}")
    print(f"Base weights: {weights_path.name} | IMGSZ: {imgsz} | Batch: {batch} | Epochs: {epochs}")
    print(f"{'=' * 60}\n")

    # Ultralytics YOLO wrapper natively supports YOLOv8, YOLOv9, YOLOv10, YOLO11, and RT-DETR weights
    model = YOLO(str(weights_path))

    model.train(
        data=str(yaml_path),
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        cache=False,

        # Fine-tuning optimizer & learning rate schedule
        optimizer="AdamW",
        lr0=lr0,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,

        # UAV-specific augmentations for small, dense aerial objects
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.1,
        copy_paste=0.1,
        degrees=15.0,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,

        # Performance & output management
        amp=True,
        project=str(runs_project_dir),
        name=exp_name,
        deterministic=False,
        exist_ok=True
    )

    best_weights_src = runs_project_dir / exp_name / "weights" / "best.pt"
    if best_weights_src.exists():
        final_weights_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights_src, final_weights_dst)
        print(f"\nTraining complete! Best weights saved to: {final_weights_dst}")
        print(f"Training plots and metrics saved in: {runs_project_dir / exp_name}")
    else:
        print(f"\nWarning: best.pt was not found in {best_weights_src}")

if __name__ == "__main__":
    train_uav_model(
        base_weights="weights/yolov8s.pt",
        exp_name="yolov8s_visdrone",
        epochs=50,
        imgsz=800,
        batch=8,
        workers=2
    )