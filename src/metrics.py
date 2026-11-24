import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

# Patch for NumPy 2.0+ compatibility with motmetrics
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, **kwargs: np.asarray(a, dtype=float, **kwargs)

import motmetrics as mm

mm.lap.default_solver = 'lap'


def load_mot_file(filepath: str, is_gt: bool = False) -> pd.DataFrame:
    cols = ['frame', 'id', 'left', 'top', 'width', 'height', 'score', 'class', 'trunc', 'occ']
    try:
        df = pd.read_csv(filepath, header=None, names=cols)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=cols)

    if df.empty:
        return df

    if is_gt:
        df = df[df['score'] > 0]
        if 'class' in df.columns:
            # UAV Targets: pedestrian(1), people(2), car(4), van(5), truck(6), bus(9), motor(10)
            target_classes = [1, 2, 4, 5, 6, 9, 10]
            df = df[df['class'].isin(target_classes)]

    df['frame'] = df['frame'].astype(int)
    df['id'] = df['id'].astype(int)
    return df


def _xywh_to_xyxy(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.empty((0, 4), dtype=float)
    boxes = df[['left', 'top', 'width', 'height']].values.astype(float).copy()
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    return boxes


def _compute_iou_matrix(gt_boxes: np.ndarray, pr_boxes: np.ndarray) -> np.ndarray:
    if len(gt_boxes) == 0 or len(pr_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pr_boxes)), dtype=float)

    x1 = np.maximum(gt_boxes[:, None, 0], pr_boxes[None, :, 0])
    y1 = np.maximum(gt_boxes[:, None, 1], pr_boxes[None, :, 1])
    x2 = np.minimum(gt_boxes[:, None, 2], pr_boxes[None, :, 2])
    y2 = np.minimum(gt_boxes[:, None, 3], pr_boxes[None, :, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    area_pr = (pr_boxes[:, 2] - pr_boxes[:, 0]) * (pr_boxes[:, 3] - pr_boxes[:, 1])
    union = area_gt[:, None] + area_pr[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _accumulate_hota_sequence(gt_df: pd.DataFrame, pred_df: pd.DataFrame, alphas: np.ndarray) -> dict:
    """Computes HOTA TP, FN, FP, and association sums across 19 IoU thresholds for one sequence."""
    n_a = len(alphas)
    res = {
        "tp": np.zeros(n_a, dtype=float),
        "fn": np.zeros(n_a, dtype=float),
        "fp": np.zeros(n_a, dtype=float),
        "assa_sum": np.zeros(n_a, dtype=float),
    }

    gt_unique = gt_df['id'].unique()
    pr_unique = pred_df['id'].unique()
    gt_map = {gid: i for i, gid in enumerate(gt_unique)}
    pr_map = {pid: i for i, pid in enumerate(pr_unique)}
    n_gt_ids, n_pr_ids = len(gt_unique), len(pr_unique)

    all_frames = sorted(set(gt_df['frame'].unique()) | set(pred_df['frame'].unique()))
    frame_cache = []
    gt_counts = np.zeros(n_gt_ids, dtype=float)
    pr_counts = np.zeros(n_pr_ids, dtype=float)
    pot_matches = np.zeros((n_a, n_gt_ids, n_pr_ids), dtype=float)

    for fr in all_frames:
        gt_fr = gt_df[gt_df['frame'] == fr]
        pr_fr = pred_df[pred_df['frame'] == fr]
        g_idx = np.array([gt_map[x] for x in gt_fr['id']], dtype=int)
        p_idx = np.array([pr_map[x] for x in pr_fr['id']], dtype=int)
        iou_mat = _compute_iou_matrix(_xywh_to_xyxy(gt_fr), _xywh_to_xyxy(pr_fr))
        frame_cache.append((g_idx, p_idx, iou_mat))

        if len(g_idx) > 0:
            gt_counts[g_idx] += 1
        if len(p_idx) > 0:
            pr_counts[p_idx] += 1
        if len(g_idx) > 0 and len(p_idx) > 0:
            for a_i, alpha in enumerate(alphas):
                valid = iou_mat >= alpha
                if np.any(valid):
                    gi, pi = np.where(valid)
                    pot_matches[a_i, g_idx[gi], p_idx[pi]] += 1

    for a_i, alpha in enumerate(alphas):
        denom = gt_counts[:, None] + pr_counts[None, :] - pot_matches[a_i]
        a_gp = np.where(denom > 0, pot_matches[a_i] / denom, 0.0)

        for g_idx, p_idx, iou_mat in frame_cache:
            n_g, n_p = len(g_idx), len(p_idx)
            if n_g == 0 or n_p == 0:
                res["fn"][a_i] += n_g
                res["fp"][a_i] += n_p
                continue

            valid = iou_mat >= alpha
            if not np.any(valid):
                res["fn"][a_i] += n_g
                res["fp"][a_i] += n_p
                continue

            sub_a = a_gp[g_idx[:, None], p_idx[None, :]]
            score_mat = np.where(valid, sub_a + 1e-5 * iou_mat, 0.0)
            r_ind, c_ind = linear_sum_assignment(-score_mat)
            kept = valid[r_ind, c_ind]
            r_kept, c_kept = r_ind[kept], c_ind[kept]

            tp = len(r_kept)
            res["tp"][a_i] += tp
            res["fn"][a_i] += n_g - tp
            res["fp"][a_i] += n_p - tp
            if tp > 0:
                res["assa_sum"][a_i] += np.sum(sub_a[r_kept, c_kept])

    return res


def _finalize_hota(totals: dict) -> float:
    tp, fn, fp, assa_sum = totals["tp"], totals["fn"], totals["fp"], totals["assa_sum"]
    det_denom = tp + fn + fp
    deta = np.where(det_denom > 0, tp / det_denom, 0.0)
    assa = np.divide(assa_sum, tp, out=np.zeros_like(tp), where=tp > 0)
    hota_alpha = np.sqrt(deta * assa)
    return float(np.mean(hota_alpha))


def _build_mot_accumulator(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> mm.MOTAccumulator:
    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(gt_df['frame'].unique()) | set(pred_df['frame'].unique()))

    for frame in frames:
        gt_frame = gt_df[gt_df['frame'] == frame]
        pred_frame = pred_df[pred_df['frame'] == frame]

        distances = mm.distances.iou_matrix(
            _xywh_to_xyxy(gt_frame),
            _xywh_to_xyxy(pred_frame),
            max_iou=0.5,
        )
        acc.update(gt_frame['id'].tolist(), pred_frame['id'].tolist(), distances)

    return acc


def evaluate_tracking_dict(gt_path: str, pred_path: str) -> dict:
    """Evaluates tracking performance (CLEAR MOT, IDF1, HOTA) for a single sequence."""
    return evaluate_dataset_dict([(gt_path, pred_path)])


def evaluate_dataset_dict(gt_pred_pairs: list[tuple[str, str]]) -> dict:
    """Evaluates overall tracking performance across one or multiple (gt_path, pred_path) pairs."""
    alphas = np.arange(0.05, 0.96, 0.05)
    hota_totals = {
        "tp": np.zeros(len(alphas), dtype=float),
        "fn": np.zeros(len(alphas), dtype=float),
        "fp": np.zeros(len(alphas), dtype=float),
        "assa_sum": np.zeros(len(alphas), dtype=float),
    }

    accumulators = []
    names = []

    for idx, (gt_path, pred_path) in enumerate(gt_pred_pairs):
        gt_df = load_mot_file(str(gt_path), is_gt=True)
        pred_df = load_mot_file(str(pred_path), is_gt=False)

        accumulators.append(_build_mot_accumulator(gt_df, pred_df))
        names.append(f"seq_{idx}")

        seq_hota = _accumulate_hota_sequence(gt_df, pred_df, alphas)
        for k in hota_totals:
            hota_totals[k] += seq_hota[k]

    mh = mm.metrics.create()
    metrics_to_compute = [
        'num_objects', 'num_matches', 'num_misses',
        'num_false_positives', 'num_switches',
        'mostly_tracked', 'mostly_lost',
        'mota', 'motp', 'idf1'
    ]

    summary = mh.compute_many(accumulators, metrics=metrics_to_compute, names=names, generate_overall=True)
    result = summary.loc['OVERALL'].to_dict()
    result['hota'] = _finalize_hota(hota_totals)
    return result