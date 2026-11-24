import cv2
import torch
import numpy as np
import pandas as pd
import supervision as sv
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm
from src.trackers.factory import create_tracker
from src.benchmark import (
    NAME_TO_VISDRONE_GT,
    _fast_engine_infer,
    _init_detector,
    _resolve_project_path,
)

VISDRONE_NAMES = {
    1: "pedestrian",
    2: "people",
    4: "car",
    5: "van",
    6: "truck",
    9: "bus",
    10: "motor"
}


def _load_gt_and_ignored(gt_path: str) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, np.ndarray]]:
    """Loads valid GT boxes/classes and ignored regions per 1-based frame index."""
    cols = ['frame', 'id', 'left', 'top', 'width', 'height', 'score', 'class', 'trunc', 'occ']
    df = pd.read_csv(gt_path, header=None, names=cols)

    ignored_df = df[(df['score'] == 0) | (df['class'] == 0)]
    ignored_regions = {}
    for frame_id, group in ignored_df.groupby('frame'):
        boxes = group[['left', 'top', 'width', 'height']].values.astype(float).copy()
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        ignored_regions[int(frame_id)] = boxes

    target_classes = list(VISDRONE_NAMES.keys())
    valid_df = df[(df['score'] > 0) & (df['class'].isin(target_classes))]
    gt_by_frame = {}
    for frame_id, group in valid_df.groupby('frame'):
        boxes = group[['left', 'top', 'width', 'height']].values.astype(float).copy()
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        classes = group['class'].values.astype(int)
        gt_by_frame[int(frame_id)] = (boxes, classes)

    return gt_by_frame, ignored_regions


def _match_boxes_iou(gt_boxes: np.ndarray, pred_boxes: np.ndarray, iou_thresh: float = 0.5) -> tuple[set[int], set[int]]:
    """Matches GT boxes and predicted boxes using Hungarian algorithm (1:1 with motmetrics)."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return set(), set()

    x1 = np.maximum(gt_boxes[:, None, 0], pred_boxes[None, :, 0])
    y1 = np.maximum(gt_boxes[:, None, 1], pred_boxes[None, :, 1])
    x2 = np.minimum(gt_boxes[:, None, 2], pred_boxes[None, :, 2])
    y2 = np.minimum(gt_boxes[:, None, 3], pred_boxes[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    union_area = gt_area[:, None] + pred_area[None, :] - inter_area

    iou_matrix = np.where(union_area > 0, inter_area / union_area, 0.0)
    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)

    matched_gt = set()
    matched_pred = set()
    for g_idx, p_idx in zip(gt_indices, pred_indices):
        if iou_matrix[g_idx, p_idx] >= iou_thresh:
            matched_gt.add(int(g_idx))
            matched_pred.add(int(p_idx))

    return matched_gt, matched_pred


def _draw_gmc_flow(frame: np.ndarray, debug_data: dict, arrow_scale: float = 3.0) -> np.ndarray:
    """Draws optical flow vectors (Cyan = background inliers, Orange = moving outliers)."""
    prev_pts = debug_data["prev_pts"]
    curr_pts = debug_data["curr_pts"]
    inliers = debug_data["inliers"]

    for p1, p2, is_inlier in zip(prev_pts, curr_pts, inliers):
        end_pt = p1 + (p2 - p1) * arrow_scale
        color = (255, 229, 0) if is_inlier else (0, 140, 255)
        cv2.arrowedLine(
            frame,
            (int(p1[0]), int(p1[1])),
            (int(end_pt[0]), int(end_pt[1])),
            color,
            thickness=1,
            tipLength=0.3
        )

    hud_text = (
        f"GMC Shift: dx={debug_data['dx']:+.2f}px, dy={debug_data['dy']:+.2f}px | "
        f"Inliers: {int(inliers.sum())}/{len(inliers)}"
    )
    cv2.rectangle(frame, (10, 10), (540, 38), (20, 20, 20), -1)
    cv2.putText(frame, hud_text, (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 229, 0), 1, cv2.LINE_AA)
    return frame


def run_visualization(
        model_path: str,
        video_path: str,
        output_path: str,
        tracker_type: str,
        tracker_params: dict,
        gt_path: str | None = None,
        show_traces: bool = True,
        show_gmc_flow: bool = False,
        imgsz: int = 800,
        conf_thres: float = 0.35,
        iou_thres: float = 0.45
):
    resolved_model = _resolve_project_path(model_path)
    resolved_video = _resolve_project_path(video_path)
    resolved_output = _resolve_project_path(output_path)
    resolved_gt = _resolve_project_path(gt_path) if gt_path else None

    print(f"Starting visualization. Model: {resolved_model.name} | Tracker: {tracker_type}")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    use_fast_infer = device.startswith('cuda')
    is_rtdetr = "rtdetr" in resolved_model.name.lower()

    model = _init_detector(
        resolved_model,
        imgsz=imgsz,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        gpu_preproc=use_fast_infer,
        device=device,
    )

    model_names = model.names
    tracker = create_tracker(tracker_type=tracker_type, params=tracker_params)

    has_cmc = hasattr(tracker, "cmc")
    if show_gmc_flow and not has_cmc:
        print(f"[INFO] show_gmc_flow=True skipped: tracker '{tracker_type}' does not use CMC.")

    gt_by_frame, ignored_regions = _load_gt_and_ignored(str(resolved_gt)) if resolved_gt else ({}, {})

    tp_box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#00E676"), thickness=2)
    tp_label_annotator = sv.LabelAnnotator(
        color=sv.Color.from_hex("#00E676"), text_color=sv.Color.BLACK, text_scale=0.45, text_thickness=1
    )

    fp_box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#FF00FF"), thickness=2)
    fp_label_annotator = sv.LabelAnnotator(
        color=sv.Color.from_hex("#FF00FF"), text_color=sv.Color.WHITE, text_scale=0.45, text_thickness=1
    )

    fn_box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#FF1744"), thickness=2)
    fn_label_annotator = sv.LabelAnnotator(
        color=sv.Color.from_hex("#FF1744"), text_color=sv.Color.WHITE, text_scale=0.45, text_thickness=1
    )

    trace_annotator = sv.TraceAnnotator(color=sv.Color.from_hex("#00E676"), trace_length=30, thickness=2)

    tracked_unique_ids = set()

    def process_frame(frame: np.ndarray, index: int) -> np.ndarray:
        frame_idx = index + 1

        if use_fast_infer:
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
                device='cpu',
                verbose=False
            )[0]
            detections = sv.Detections.from_ultralytics(results)

        valid_indices = []
        mapped_classes = []
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

        confirmed_mask = []
        ign_boxes = ignored_regions.get(frame_idx, np.empty((0, 4)))

        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            for xyxy, trk_id in zip(detections.xyxy, detections.tracker_id):
                if int(trk_id) < 0:
                    confirmed_mask.append(False)
                    continue
                if len(ign_boxes) > 0:
                    cx = (xyxy[0] + xyxy[2]) / 2.0
                    cy = (xyxy[1] + xyxy[3]) / 2.0
                    in_ignored = np.any(
                        (cx >= ign_boxes[:, 0]) & (cx <= ign_boxes[:, 2]) &
                        (cy >= ign_boxes[:, 1]) & (cy <= ign_boxes[:, 3])
                    )
                    if in_ignored:
                        confirmed_mask.append(False)
                        continue
                confirmed_mask.append(True)

        detections = detections[np.array(confirmed_mask, dtype=bool)] if len(confirmed_mask) > 0 else sv.Detections.empty()

        for trk_id in (detections.tracker_id if detections.tracker_id is not None else []):
            tracked_unique_ids.add(int(trk_id))

        annotated = frame.copy()

        if show_gmc_flow and has_cmc and getattr(tracker.cmc, "last_debug_data", None) is not None:
            annotated = _draw_gmc_flow(annotated, tracker.cmc.last_debug_data, arrow_scale=3.0)

        if show_traces and len(detections) > 0:
            annotated = trace_annotator.annotate(scene=annotated, detections=detections)

        if resolved_gt is not None:
            gt_boxes, gt_classes = gt_by_frame.get(frame_idx, (np.empty((0, 4)), np.empty((0,), dtype=int)))
            matched_gt, matched_pred = _match_boxes_iou(gt_boxes, detections.xyxy, iou_thresh=0.5)

            tp_idx = [i for i in range(len(detections)) if i in matched_pred]
            fp_idx = [i for i in range(len(detections)) if i not in matched_pred]
            fn_idx = [i for i in range(len(gt_boxes)) if i not in matched_gt]

            if tp_idx:
                tp_det = detections[tp_idx]
                tp_labels = [
                    f"#{int(tid)} {VISDRONE_NAMES.get(int(cid), cid)} {conf:.2f}"
                    for conf, cid, tid in zip(tp_det.confidence, tp_det.class_id, tp_det.tracker_id)
                ]
                annotated = tp_box_annotator.annotate(scene=annotated, detections=tp_det)
                annotated = tp_label_annotator.annotate(scene=annotated, detections=tp_det, labels=tp_labels)

            if fp_idx:
                fp_det = detections[fp_idx]
                fp_labels = [
                    f"FP | #{int(tid)} {VISDRONE_NAMES.get(int(cid), cid)} {conf:.2f}"
                    for conf, cid, tid in zip(fp_det.confidence, fp_det.class_id, fp_det.tracker_id)
                ]
                annotated = fp_box_annotator.annotate(scene=annotated, detections=fp_det)
                annotated = fp_label_annotator.annotate(scene=annotated, detections=fp_det, labels=fp_labels)

            if fn_idx:
                fn_det = sv.Detections(
                    xyxy=gt_boxes[fn_idx].astype(np.float32),
                    class_id=gt_classes[fn_idx]
                )
                fn_labels = [
                    f"MISSED: {VISDRONE_NAMES.get(int(cid), cid)}"
                    for cid in fn_det.class_id
                ]
                annotated = fn_box_annotator.annotate(scene=annotated, detections=fn_det)
                annotated = fn_label_annotator.annotate(scene=annotated, detections=fn_det, labels=fn_labels)

        else:
            if len(detections) > 0:
                labels = [
                    f"#{int(tid)} {VISDRONE_NAMES.get(int(cid), cid)} {conf:.2f}"
                    for conf, cid, tid in zip(detections.confidence, detections.class_id, detections.tracker_id)
                ]
                annotated = tp_box_annotator.annotate(scene=annotated, detections=detections)
                annotated = tp_label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        return annotated

    video_info = sv.VideoInfo.from_video_path(str(resolved_video))
    frames_generator = sv.get_video_frames_generator(str(resolved_video))

    with sv.VideoSink(target_path=str(resolved_output), video_info=video_info) as sink:
        for index, frame in enumerate(tqdm(frames_generator, total=video_info.total_frames, desc="Rendering video")):
            annotated_frame = process_frame(frame, index)
            sink.write_frame(annotated_frame)

    print(f"Done! Video saved to: {resolved_output}")
    print(f"Unique objects (Track IDs): {len(tracked_unique_ids)}")