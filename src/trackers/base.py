from abc import ABC, abstractmethod
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker
from .cmc import CameraMotionCompensator


class BaseTrackerAdapter(ABC):
    """Unified interface for all trackers in the benchmark."""

    @abstractmethod
    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        pass


class VanillaByteTrackAdapter(BaseTrackerAdapter):
    """Wrapper for standard ByteTrack without camera motion compensation."""

    def __init__(self, **kwargs):
        self.tracker = ByteTrackTracker(**kwargs)

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        return self.tracker.update(detections)


class CMCByteTrackAdapter(BaseTrackerAdapter):
    """ByteTrack enhanced with cumulative Global Motion Compensation (GMC) via Sparse Optical Flow."""

    def __init__(self, **kwargs):
        self.tracker = ByteTrackTracker(**kwargs)
        self.cmc = CameraMotionCompensator(downscale=4)
        # Cumulative 3x3 homography mapping stationary Frame-0 world -> Current Frame
        self.cum_transform = np.eye(3, dtype=np.float32)

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        # 2x3 matrix mapping Frame (t-1) -> Frame (t)
        step_affine = self.cmc.compute_affine_matrix(frame)
        step_3x3 = np.vstack([step_affine, [0.0, 0.0, 1.0]]).astype(np.float32)

        # Accumulate global transform: Frame 0 -> Frame t
        self.cum_transform = step_3x3 @ self.cum_transform

        if len(detections) > 0:
            # Project current detections into stationary Frame-0 coordinate space
            inv_cum = np.linalg.inv(self.cum_transform)[:2, :]
            warped_xyxy = self._warp_boxes(detections.xyxy, inv_cum)

            stabilized_dets = sv.Detections(
                xyxy=warped_xyxy,
                confidence=detections.confidence,
                class_id=detections.class_id
            )
            tracked = self.tracker.update(stabilized_dets)

            # Project tracked boxes back to current screen coordinates
            if len(tracked) > 0:
                fwd_cum = self.cum_transform[:2, :]
                tracked.xyxy = self._warp_boxes(tracked.xyxy, fwd_cum)
            return tracked

        return self.tracker.update(detections)

    @staticmethod
    def _warp_boxes(boxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return boxes
        pts1 = np.hstack([boxes[:, :2], np.ones((len(boxes), 1), dtype=np.float32)])
        pts2 = np.hstack([boxes[:, 2:4], np.ones((len(boxes), 1), dtype=np.float32)])
        return np.hstack([pts1 @ matrix.T, pts2 @ matrix.T])