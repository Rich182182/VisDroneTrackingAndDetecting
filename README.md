# Aerial Multi-Object Detection and Tracking Benchmark (VisDrone2019-MOT)

End-to-end computer vision benchmarking and deployment pipeline for UAV aerial surveillance, evaluating convolutional and transformer-based detectors (**YOLOv8s**, **YOLOv10s**, **YOLO11n**, **RT-DETR-L**) paired with **ByteTrack** and **Sparse Optical Flow Global Motion Compensation (GMC)** under **PyTorch FP32** and **NVIDIA TensorRT (FP16 / INT8)** quantization.

**Live Interactive Demo:** [Hugging Face Spaces - `Rich182/visdrone-uav-tracking`](https://huggingface.co/spaces/Rich182/visdrone-uav-tracking)  
**Pre-Trained Checkpoints:** [GitHub Release v1.0.0 (`800x800` Fine-Tuned Weights)](https://github.com/Rich182182/VisDroneTrackingAndDetecting/releases/tag/v1.0.0)

---

## Visual Comparison 
> **Overlay Legend:** **Green** = True Positive (`TP`) | **Magenta** = False Positive (`FP`) | **Red** = Missed Ground Truth (`FN`) | **Cyan Vectors** = Sparse Optical Flow GMC Background Inliers

| **1. Baseline: YOLOv8s (COCO) + Vanilla ByteTrack** | **2. YOLOv8s (Fine-Tuned, TRT INT8) + ByteTrack-GMC** |
| :---: | :---: |
| ![Baseline COCO](outputs/gifs/01_baseline_coco_vanilla.gif) | ![YOLOv8s INT8 GMC](outputs/gifs/02_yolov8s_int8_gmc.gif) |
| **33.90% HOTA** \| 45.87% IDF1 \| 1,075 ID Sw \| 27.35 ms GPU | **42.12% HOTA** \| 61.13% IDF1 \| 1,514 ID Sw \| **3.01 ms GPU** |

| **3. YOLO11n (Fine-Tuned, TRT INT8) + ByteTrack-GMC** | **4. RT-DETR-L (Fine-Tuned, TRT FP16) + ByteTrack-GMC** |
| :---: | :---: |
| ![YOLO11n INT8 GMC](outputs/gifs/03_yolo11n_int8_gmc.gif) | ![RT-DETR-L FP16 GMC](outputs/gifs/05_rtdetr_l_fp16_gmc.gif) |
| **37.30% HOTA** \| 48.66% IDF1 \| 782 ID Sw \| **2.16 ms GPU** | **40.46% HOTA** \| 61.95% IDF1 \| 6,632 ID Sw \| **7.47 ms GPU** |

*Note on visual grid selection: The 2x2 grid contrasts four distinct architectural and precision regimes (unadapted COCO baseline, top-HOTA CNN `YOLOv8s INT8`, compact edge CNN `YOLO11n INT8`, and Vision Transformer `RT-DETR-L FP16`). The NMS-free `YOLOv10s` model exhibits visually similar tracking behavior to `YOLOv8s` (`41.47%` vs. `42.12%` HOTA in INT8) and is fully reported in the quantitative tables and Pareto plots below.*

---

## Executive Summary & Key Engineering Findings

* **Primary Metric Focus (HOTA vs. MOTA):** Because aerial tracking suffers from frequent camera motion and small-object occlusions, **HOTA** (*Higher Order Tracking Accuracy*) serves as the primary evaluation metric, balancing detection accuracy ($\text{DetA}$) with trajectory association stability ($\text{AssA}$).
* **Optimal Production Balance (`YOLOv8s TRT INT8 + GMC`):** Achieves the highest overall tracking consistency at **42.12% HOTA** (`42.13%` in FP16) and **46.40% MOTA**, compressing isolated neural network forward-pass latency by **2.94x** (`8.86 ms` to `3.01 ms`) with zero accuracy degradation.
* **NMS-Free Architecture Stability (`YOLOv10s`):** Eliminating Non-Maximum Suppression via dual-label assignments yields **41.78% HOTA** (`41.47%` in INT8) and the lowest ID switch count among medium-scale models (`1,188` ID switches in FP32 vs. `1,393` for YOLOv8s and `6,407` for RT-DETR-L), running at **3.06 ms** isolated GPU latency in INT8.
* **Domain Adaptation & Optical Flow GMC Impact:** Fine-tuning on VisDrone aerial imagery boosts YOLOv8s HOTA from **33.90%** to **40.80%** (`+6.90 pp`), while adding sparse Lucas-Kanade optical flow GMC with RANSAC affine estimation raises HOTA to **42.08%** (`+8.18 pp` total), cuts identity switches by **37.3%** (`2,222` down to `1,393`), and reduces the relative ID switch rate per matched object to **0.99%** (outperforming the `1.06%` rate of the COCO baseline despite tracking 39% more targets).
* **CNN vs. Vision Transformer Quantization Behavior:** All three CNN architectures (`YOLOv8s`, `YOLOv10s`, `YOLO11n`) retain full HOTA accuracy under Post-Training Quantization (PTQ) to **INT8** (`2.16–3.06 ms` pure GPU latency). Conversely, **RT-DETR-L** achieves the highest raw detection recall (`52.23% MOTA`, `40.56% HOTA`) and accelerates **5.30x** in **TRT FP16** (`39.57 ms` to `7.47 ms`), but undergoes *PTQ Quantization Collapse* (`0.00%` at standard confidence threshold) in 8-bit integer precision due to activation outliers in self-attention and layer normalization blocks without Quantization-Aware Training (QAT).
* **Edge Deployment & Bottleneck Analysis (Amdahl's Law):** In offline file-based evaluation on an NVIDIA RTX 3060 Laptop GPU, pure neural network execution accounts for only ~7% of total frame time (`2.16–3.06 ms` out of `39.85–41.82 ms`), while synchronous disk decoding (`~18.3 ms`) and Python/PCIe tensor overhead (`~15 ms`) dominate. Replacing file I/O and Python wrappers with a streaming C++ / NVIDIA DeepStream pipeline eliminates this host bottleneck for real-time deployment on embedded edge hardware (e.g., NVIDIA Jetson Orin Nano / NX).

---

## Tech Stack & Pipeline Architecture

* **Deep Learning & Inference:** PyTorch, Ultralytics, NVIDIA TensorRT (`FP16` / `INT8` PTQ calibration), Torchvision (`batched_nms`)
* **Multi-Object Tracking & Computer Vision:** OpenCV (Pyramidal Lucas-Kanade Optical Flow, RANSAC Affine Estimation), ByteTrack (two-stage Kalman + IoU cascade), Supervision
* **Evaluation & Profiling:** Custom vectorized 19-threshold HOTA engine, `motmetrics` (CLEAR MOT, IDF1), CUDA Event GPU latency profiler
* **Deployment:** Gradio Web Application hosted on Hugging Face Spaces

```mermaid
flowchart LR
    A[UAV Video Stream / Frame] --> B[GPU Letterbox & Normalization]
    B --> C[TensorRT Engine<br/>FP16 / INT8]
    C --> D[Unified Decoder & Batched NMS]
    A --> E[Sparse Optical Flow GMC<br/>Shi-Tomasi + PyrLK + RANSAC]
    E --> F[Kalman State Camera<br/>Motion Compensation]
    D --> G[ByteTrack Two-Stage<br/>Cascade Association]
    F --> G
    G --> H[19-Alpha HOTA & CLEAR MOT<br/>Evaluator / GT Visualizer]
```

---

## Quantitative Benchmarks & Analysis

All models were fine-tuned on the VisDrone2019-MOT training split and evaluated across all **17 sequences** (`VisDrone2019-MOT-test-dev`) at `800x800` resolution on an **NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM)**. Metrics follow the official VisDrone-MOT protocol across 7 target categories (`pedestrian`, `people`, `car`, `van`, `truck`, `bus`, `motor`), filtering out ignored regions (`score == 0` or `class == 0`).

### 1. Domain Adaptation & Camera Motion Compensation Ablation

Progressive evaluation isolating the impact of VisDrone fine-tuning and sparse optical-flow Global Motion Compensation (`GMC`) on `YOLOv8s` (`imgsz=800`, `FP32`):

| Experiment | Model | Tracker | HOTA (%) | IDF1 (%) | MOTA (%) | ID Switches | ID Sw / TP (%) | Pure GPU (ms) | Total Latency (ms) | FPS |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `01_YOLOv8s_COCO_Vanilla` | `yolov8s.pt` | `bytetrack_vanilla` | 33.90 | 45.87 | 32.65 | 1,075 | 1.06% | 27.35 | 51.21 | 19.5 |
| `02_YOLOv8s_VisDrone_FP32_Vanilla` | `yolov8s_visdrone.pt` | `bytetrack_vanilla` | 40.80 | 59.24 | 47.72 | 2,222 | 1.49% | 8.79 | 39.05 | 25.6 |
| `03_YOLOv8s_VisDrone_FP32_GMC` | `yolov8s_visdrone.pt` | `bytetrack_gmc` | **42.08** | **60.44** | 45.60 | **1,393** | **0.99%** | 8.80 | 42.28 | 23.7 |

*Note: While raw dataset-wide ID switches increase from `1,075` (COCO) to `1,393` (VisDrone + GMC), the fine-tuned detector tracks **39.0% more true-positive targets** (`140,693` vs. `101,230`), including heavily occluded small vehicles. Normalized by matched detections, GMC reduces relative ID switch frequency to `0.99%`—improving identity stability over both the unadapted COCO baseline (`1.06%`) and non-GMC fine-tuned model (`1.49%`).*

![Ablation Study Impact](outputs/figures/03_ablation_gmc_impact.png)

### 2. Cross-Architecture Comparison (COCO Zero-Shot vs. VisDrone Fine-Tuned)

Comparison across four detector families paired with `ByteTrack-GMC` (`imgsz=800`, `FP32`):

| Experiment | Architecture | Params / `.pt` Size | HOTA (%) | IDF1 (%) | MOTA (%) | ID Switches | Matched (TP) | False Pos | Pure GPU (ms) | Peak VRAM (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `YOLOv8s_COCO_GMC` | CNN (Anchor-Free) | 11.2M / 22.5 MB | 34.62 | 45.93 | 30.86 | 642 | 95,388 | 4,928 | 8.91 | 265.5 |
| `YOLOv8s_VisDrone_FP32_GMC` | CNN (Anchor-Free) | 11.2M / 22.5 MB | **42.08** | 60.44 | 45.60 | 1,393 | 140,693 | 7,036 | 8.80 | 265.3 |
| `YOLO11n_COCO_GMC` | CNN (Edge Nano) | 2.6M / 5.5 MB | 30.10 | 38.23 | 24.24 | 481 | 75,128 | 4,092 | 8.78 | 230.2 |
| `YOLO11n_VisDrone_FP32_GMC` | CNN (Edge Nano) | 2.6M / 5.5 MB | 37.34 | 48.22 | 32.80 | **700** | 100,257 | **4,110** | **8.64** | **230.1** |
| `YOLOv10s_COCO_GMC` | CNN (NMS-Free) | 7.2M / 16.5 MB | 34.08 | 43.79 | 29.04 | 632 | 90,314 | 5,203 | 8.68 | 270.2 |
| `YOLOv10s_VisDrone_FP32_GMC` | CNN (NMS-Free) | 7.2M / 16.5 MB | 41.78 | 57.50 | 42.60 | 1,188 | 131,697 | 6,850 | 8.71 | 269.9 |
| `RTDETR_L_COCO_GMC` | Vision Transformer | 32.0M / 66.2 MB | 38.30 | 55.37 | 40.46 | 1,843 | 134,291 | 15,706 | 40.34 | 418.2 |
| `RTDETR_L_VisDrone_FP32_GMC` | Vision Transformer | 32.0M / 66.2 MB | 40.56 | **62.01** | **52.23** | 6,407 | **184,517** | 31,444 | 39.54 | 414.6 |

### 3. TensorRT Optimization & Hardware Latency Profiling

Post-training conversion from PyTorch `FP32` to TensorRT `FP16` and `INT8` engines (`workspace=4 GB`, calibrated on VisDrone validation imagery):

| Experiment | Precision | HOTA (%) | IDF1 (%) | MOTA (%) | ID Switches | Pure GPU (ms) | Infer Latency (ms) | Total Latency (ms) | FPS | Peak VRAM (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `YOLOv8s_FP32` | PyTorch FP32 | 42.08 | 60.44 | 45.60 | 1,393 | 8.86 | 19.69 | 43.45 | 23.0 | 297.3 |
| `YOLOv8s_TRT_FP16` | TensorRT FP16 | **42.13** | 60.42 | 45.61 | 1,403 | 2.72 | 18.90 | 42.27 | 23.7 | 217.7 |
| `YOLOv8s_TRT_INT8` | TensorRT INT8 | 42.12 | 61.13 | 46.40 | 1,514 | 3.01 | 18.32 | 41.82 | 23.9 | 217.7 |
| `YOLO11n_FP32` | PyTorch FP32 | 37.34 | 48.22 | 32.80 | 700 | 8.67 | 18.98 | 41.84 | 23.9 | 230.1 |
| `YOLO11n_TRT_FP16` | TensorRT FP16 | 37.36 | 48.29 | 32.92 | **693** | 2.55 | **17.06** | **39.85** | **25.1** | 217.7 |
| `YOLO11n_TRT_INT8` | TensorRT INT8 | 37.30 | 48.66 | 33.24 | 782 | **2.16** | 17.82 | 41.05 | 24.4 | 217.7 |
| `YOLOv10s_FP32` | PyTorch FP32 | 41.78 | 57.50 | 42.60 | 1,188 | 8.62 | 19.07 | 42.21 | 23.7 | 269.9 |
| `YOLOv10s_TRT_FP16` | TensorRT FP16 | 41.69 | 57.41 | 42.55 | 1,252 | 2.82 | 18.35 | 41.17 | 24.3 | 217.7 |
| `YOLOv10s_TRT_INT8` | TensorRT INT8 | 41.47 | 57.41 | 42.70 | 1,277 | 3.06 | 17.82 | 40.72 | 24.6 | 217.7 |
| `RTDETR_L_FP32` | PyTorch FP32 | 40.56 | **62.01** | **52.23** | 6,407 | 39.57 | 45.01 | 69.46 | 14.4 | 414.6 |
| `RTDETR_L_TRT_FP16` | TensorRT FP16 | 40.46 | 61.95 | 52.17 | 6,632 | 7.47 | 20.86 | 45.25 | 22.1 | **51.0** |
| `RTDETR_L_TRT_INT8`* | TensorRT INT8 | 0.00 | 0.00 | 0.00 | 0 | 5.30 | 17.69 | 39.68 | 25.2 | 65.0 |

*\*`RTDETR_L_TRT_INT8` exhibits PTQ Quantization Collapse at the standard `0.35` confidence threshold: 8-bit uniform quantization in self-attention and deformable decoder layers compresses output logits below `0.15` due to activation outliers, requiring either `TRT_FP16` (`40.46% HOTA` @ `7.47 ms`) or Quantization-Aware Training (QAT).*

![Accuracy-Latency Pareto Frontier](outputs/figures/02_pareto_hota_vs_speed.png)

![End-to-End Pipeline Latency Breakdown](outputs/figures/01_latency_breakdown.png)

---

## Honest Limitations & Failure Modes

* **Small-Scale Target Recall vs. Input Resolution (`imgsz=800`):** At high UAV altitudes, pedestrians, motorcycles, and distant vehicles occupy fewer than `12x12` pixels after downscaling 1080p/2.7K frames to `800x800`, resulting in higher false-negative rates on `pedestrian` and `motor` classes compared to passenger vehicles. Scaling inference resolution to `1280x1280` substantially improves small-object recall, but increases GPU forward-pass latency by roughly `2.5x` and raises peak VRAM consumption.
* **Ground-Truth Annotation Gaps (Apparent False Positives):** Visual inspection of diagnostic overlays shows that a notable fraction of counted `False Positives` (`FP`) correspond to real stationary or partially occluded vehicles in parking lots and side streets that were omitted from VisDrone human annotations and not masked as ignored regions (`class == 0`).
* **Dense-Traffic ID Swaps under Aggressive Occlusion Recovery:** To prevent trajectory fragmentation during rapid UAV yaw/pitch maneuvers, ByteTrack is configured with a long track retention buffer (`lost_track_buffer=120` frames), a `0.50` high-confidence track initialization threshold (`0.35` low-confidence rescue), and a permissive `0.05` IoU matching floor. While this preserves IDs across transient occlusions, closely spaced vehicles moving in parallel lanes can trigger localized ID swaps.
* **Transformer PTQ INT8 Sensitivity:** Unlike CNN backbones, `RT-DETR-L` cannot be deployed with standard Post-Training Quantization to `INT8` without Quantization-Aware Training (QAT) due to severe logit compression in attention layers.

---

## Repository Structure

```text
VisDroneTrackingAndDetecting/
├── data/
│   └── VisDrone_UAV.yaml              # 7-class VisDrone dataset configuration
├── notebooks/
│   ├── 01_training.ipynb              # Fine-tuning workflow for YOLOv8s, YOLOv10s, YOLO11n, RT-DETR-L
│   └── 02_quantization.ipynb          # TensorRT FP16/INT8 export and calibration workflow
├── outputs/
│   ├── figures/                       # Summary plots (Latency Breakdown, Pareto Frontier, Ablation Impact)
│   ├── gifs/                          # Visual comparison demos embedded in README
│   ├── table1_tracker_ablation.csv    # Tracker and domain adaptation ablation results
│   ├── table2_architectures.csv       # Cross-architecture benchmark results
│   └── table3_tensorrt.csv            # TensorRT precision and latency profiling results
├── runs/uav_training/                 # Training curves, PR plots, confusion matrices, and hyperparams
├── src/
│   ├── export/
│   │   └── exporter.py                # ONNX and TensorRT engine builder
│   ├── trackers/
│   │   ├── base.py                    # Abstract tracker interface
│   │   ├── cmc.py                     # Sparse Optical Flow Global Motion Compensation
│   │   └── factory.py                 # ByteTrack & ByteTrack-GMC tracker factory
│   ├── training/
│   │   ├── prepare_dataset.py         # VisDrone-MOT to YOLO detection format converter
│   │   └── train.py                   # Training entrypoint
│   ├── benchmark.py                   # GPU inference loop, CUDA event profiler, and dataset runner
│   ├── metrics.py                     # HOTA, IDF1, and CLEAR MOT evaluation suite
│   ├── utils.py                       # Video assembly and I/O helpers
│   └── visualizer.py                  # GT-matched diagnostic video renderer
├── experiments.ipynb                  # Master evaluation and visualization notebook
└── requirements.txt                   # Locked Python dependencies
```

---

## Reproducibility & Quickstart

### 1. Environment Setup

```bash
git clone [https://github.com/Rich182182/VisDroneTrackingAndDetecting.git](https://github.com/Rich182182/VisDroneTrackingAndDetecting.git)
cd VisDroneTrackingAndDetecting
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Fine-Tuned Weights

Download the fine-tuned `.pt` checkpoints from the **[v1.0.0 GitHub Release](https://github.com/Rich182182/VisDroneTrackingAndDetecting/releases/tag/v1.0.0)** and place them into `weights/`:

* `weights/rtdetr_l_visdrone.pt`
* `weights/yolov8s_visdrone.pt`
* `weights/yolov10s_visdrone.pt`
* `weights/yolo11n_visdrone.pt`

### 3. Run Evaluation & Visualization

```python
from src.benchmark import run_dataset_experiment
from src.visualizer import run_visualization

tracker_params = {
    "lost_track_buffer": 120,
    "track_activation_threshold": 0.35,
    "minimum_consecutive_frames": 1,
    "minimum_iou_threshold": 0.05,
    "high_conf_det_threshold": 0.5,
}

# Evaluate across all 17 VisDrone2019-MOT-test-dev sequences
run_dataset_experiment(
    exp_name="YOLOv8s_VisDrone_FP32_GMC",
    model_path="weights/yolov8s_visdrone.pt",
    tracker_type="bytetrack_gmc",
    tracker_params=tracker_params,
    results_csv="outputs/table2_architectures.csv",
)

# Render diagnostic video with GT matching and optical-flow GMC overlay
run_visualization(
    model_path="weights/yolov8s_visdrone.pt",
    video_path="data/videos/uav0000249_00001_v.mp4",
    output_path="outputs/videos/demo_yolov8s_gmc.mp4",
    tracker_type="bytetrack_gmc",
    tracker_params=tracker_params,
    gt_path="data/VisDrone2019-MOT-test-dev/annotations/uav0000249_00001_v.txt",
    show_gmc_flow=True,
)
```

---

## Dataset, Privacy & Ethics

* **Data Source:** All experiments use the publicly available academic **[VisDrone2019-MOT](https://github.com/VisDrone/VisDrone-Dataset)** benchmark dataset collected by the AISKYEYE team at Tianjin University for non-commercial computer vision research.
* **Privacy & Anonymization:** Aerial footage is captured at high altitude and resolution where individual facial features and license plate characters are not identifiable; no personally identifiable information (PII) is stored or processed by the tracking pipeline.
