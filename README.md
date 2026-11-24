## Visual Comparison (`VisDrone2019-MOT-test-dev` — `uav0000249_00001_v`)

> **Legend:** 🟩 **Green** = True Positive (`TP`) | 🟪 **Magenta** = False Positive (`FP`) | 🟥 **Red** = Missed Ground Truth (`FN`) | 🟦 **Cyan Vectors** = Sparse Optical Flow GMC Inliers

| **1. Baseline: YOLOv8s (COCO) + Vanilla ByteTrack** | **2. YOLOv8s (Fine-Tuned, TRT INT8) + ByteTrack-GMC** |
| :---: | :---: |
| ![Baseline COCO](outputs/gifs/01_baseline_coco_vanilla2.gif) | ![YOLOv8s INT8 GMC](outputs/gifs/02_yolov8s_int8_gmc.gif) |
| *High miss rate on top-down aerial targets (`32.65% MOTA`)* | *Balanced real-time CNN (`46.40% MOTA`, `3.01 ms` GPU)* |

| **3. YOLO11n (Fine-Tuned, TRT INT8) + ByteTrack-GMC** | **4. RT-DETR-L (Fine-Tuned, TRT FP16) + ByteTrack-GMC** |
| :---: | :---: |
| ![YOLO11n INT8 GMC](outputs/gifs/03_yolo11n_int8_gmc.gif) | ![RT-DETR-L FP16 GMC](outputs/gifs/05_rtdetr_l_fp16_gmc.gif) |
| *Ultra-lightweight edge baseline (`33.24% MOTA`, `2.16 ms` GPU)* | *Highest tracking accuracy (`52.17% MOTA`, `7.47 ms` GPU)* |
