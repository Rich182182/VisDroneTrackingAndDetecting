from .base import VanillaByteTrackAdapter, CMCByteTrackAdapter, BaseTrackerAdapter


def create_tracker(tracker_type: str, params: dict = None) -> BaseTrackerAdapter:
    """Creates the requested tracker instance based on experiment configuration."""
    params = params or {}

    default_kwargs = {
        "lost_track_buffer": params.get("lost_track_buffer", 120),
        "track_activation_threshold": params.get("track_activation_threshold", 0.6),
        "minimum_consecutive_frames": params.get("minimum_consecutive_frames", 1),
        "minimum_iou_threshold": params.get("minimum_iou_threshold", 0.05),
        "high_conf_det_threshold": params.get("high_conf_det_threshold", 0.4),
    }

    if tracker_type == "bytetrack_vanilla":
        return VanillaByteTrackAdapter(**default_kwargs)

    elif tracker_type == "bytetrack_gmc":
        return CMCByteTrackAdapter(**default_kwargs)

    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")