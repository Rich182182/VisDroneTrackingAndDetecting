import cv2
import numpy as np


class CameraMotionCompensator:
    """Computes 2x3 affine camera shift matrix using downscaled Sparse Optical Flow."""

    def __init__(self, max_corners: int = 120, quality_level: float = 0.01, min_distance: int = 15, downscale: int = 6):
        self.prev_gray = None
        self.downscale = downscale
        self.last_debug_data = None
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=7
        )

    def compute_affine_matrix(self, frame: np.ndarray) -> np.ndarray:
        """Returns 2x3 affine matrix mapping points from previous frame to current frame (in full-res pixels)."""
        if self.downscale > 1:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (w // self.downscale, h // self.downscale), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.last_debug_data = None
            return np.eye(2, 3, dtype=np.float32)

        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
        if prev_pts is None or len(prev_pts) < 10:
            self.prev_gray = gray
            self.last_debug_data = None
            return np.eye(2, 3, dtype=np.float32)

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts, None)

        valid_prev = prev_pts[status == 1].reshape(-1, 2)
        valid_curr = curr_pts[status == 1].reshape(-1, 2)
        self.prev_gray = gray

        if len(valid_prev) < 10:
            self.last_debug_data = None
            return np.eye(2, 3, dtype=np.float32)

        affine_matrix, inliers = cv2.estimateAffinePartial2D(
            valid_prev, valid_curr, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )

        if affine_matrix is None or inliers is None:
            self.last_debug_data = None
            return np.eye(2, 3, dtype=np.float32)

        affine_matrix = affine_matrix.astype(np.float32)
        affine_matrix[:, 2] *= self.downscale

        self.last_debug_data = {
            "prev_pts": valid_prev * self.downscale,
            "curr_pts": valid_curr * self.downscale,
            "inliers": inliers.ravel().astype(bool),
            "dx": float(affine_matrix[0, 2]),
            "dy": float(affine_matrix[1, 2]),
        }

        return affine_matrix