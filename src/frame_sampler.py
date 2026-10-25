"""
Frame Sampler Module.
Extracts video frames at configurable sampling intervals/FPS,
calculates exact video timeline timestamps, and guarantees low memory usage
using generator streams.
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Generator, Optional, Tuple

@dataclass
class VideoFrame:
    frame_index: int
    video_timestamp_sec: float
    timecode_str: str          # Format HH:MM:SS or MM:SS
    image: np.ndarray
    original_fps: float
    is_keyframe: bool = True

class FrameSampler:
    """
    Samples frames from video streams at configurable frequencies (e.g. 5 FPS, 10 FPS, or original),
    yielding VideoFrame objects with exact video-timeline timestamps.
    """

    def __init__(self, target_sampling_fps: float = 5.0):
        self.target_sampling_fps = max(0.5, float(target_sampling_fps))

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Formats seconds into HH:MM:SS or MM:SS string."""
        total_sec = max(0.0, seconds)
        hours = int(total_sec // 3600)
        minutes = int((total_sec % 3600) // 60)
        secs = int(total_sec % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def sample_video(
        self,
        video_path_or_cap,
        max_duration_sec: Optional[float] = None,
        start_time_sec: float = 0.0
    ) -> Generator[VideoFrame, None, None]:
        """
        Memory-efficient generator yielding sampled VideoFrame instances.
        Never loads entire video into RAM.
        """
        if isinstance(video_path_or_cap, str):
            cap = cv2.VideoCapture(video_path_or_cap)
            should_close = True
        else:
            cap = video_path_or_cap
            should_close = False

        if not cap.isOpened():
            return

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine frame skip interval
        if self.target_sampling_fps >= original_fps:
            frame_step = 1
        else:
            frame_step = max(1, int(round(original_fps / self.target_sampling_fps)))

        # Seek to start time if requested
        if start_time_sec > 0:
            start_frame = int(start_time_sec * original_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame_idx = start_frame
        else:
            current_frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Only yield if on the sampling interval step
                if current_frame_idx % frame_step == 0:
                    video_sec = current_frame_idx / original_fps
                    if max_duration_sec is not None and video_sec > max_duration_sec:
                        break

                    timecode = self.format_timestamp(video_sec)
                    yield VideoFrame(
                        frame_index=current_frame_idx,
                        video_timestamp_sec=round(video_sec, 3),
                        timecode_str=timecode,
                        image=frame,
                        original_fps=original_fps
                    )

                current_frame_idx += 1
        finally:
            if should_close:
                cap.release()
