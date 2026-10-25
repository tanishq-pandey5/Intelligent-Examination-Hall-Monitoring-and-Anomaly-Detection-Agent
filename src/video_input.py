"""
Video Input and Validation Module.
Handles video file inspection, metadata extraction, format verification,
and stream acquisition without loading the entire video into RAM.
"""
import os
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class VideoMetadata:
    file_path: str
    file_name: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    aspect_ratio: str
    is_valid: bool
    error_message: Optional[str] = None

class VideoInputHandler:
    """
    Validates and inspects input video streams.
    Ensures safe reading without excessive memory footprint.
    """

    SUPPORTED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

    @classmethod
    def inspect_video(cls, file_path: str) -> VideoMetadata:
        """
        Extracts metadata from a video file and validates readability.
        """
        if not file_path or not os.path.exists(file_path):
            return VideoMetadata(
                file_path=file_path,
                file_name=os.path.basename(file_path) if file_path else "",
                width=0, height=0, fps=0.0, total_frames=0, duration_seconds=0.0,
                aspect_ratio="N/A", is_valid=False, error_message=f"File does not exist: {file_path}"
            )

        _, ext = os.path.splitext(file_path)
        if ext.lower() not in cls.SUPPORTED_EXTENSIONS:
            return VideoMetadata(
                file_path=file_path,
                file_name=os.path.basename(file_path),
                width=0, height=0, fps=0.0, total_frames=0, duration_seconds=0.0,
                aspect_ratio="N/A", is_valid=False,
                error_message=f"Unsupported video format '{ext}'. Supported: {', '.join(cls.SUPPORTED_EXTENSIONS)}"
            )

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return VideoMetadata(
                file_path=file_path,
                file_name=os.path.basename(file_path),
                width=0, height=0, fps=0.0, total_frames=0, duration_seconds=0.0,
                aspect_ratio="N/A", is_valid=False,
                error_message="Corrupted or unreadable video file."
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
            cap.release()
            return VideoMetadata(
                file_path=file_path,
                file_name=os.path.basename(file_path),
                width=width, height=height, fps=fps, total_frames=total_frames,
                duration_seconds=0.0, aspect_ratio="N/A", is_valid=False,
                error_message="Invalid video stream parameters (zero frame count or FPS)."
            )

        duration = total_frames / fps
        aspect = f"{width}:{height}"
        cap.release()

        return VideoMetadata(
            file_path=file_path,
            file_name=os.path.basename(file_path),
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=round(duration, 2),
            aspect_ratio=aspect,
            is_valid=True
        )

    @classmethod
    def get_capture(cls, file_path_or_cam: Any) -> Optional[cv2.VideoCapture]:
        """Returns open VideoCapture object or None."""
        cap = cv2.VideoCapture(file_path_or_cam)
        if not cap.isOpened():
            return None
        return cap
