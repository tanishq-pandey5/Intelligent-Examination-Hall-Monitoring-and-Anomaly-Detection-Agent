"""
Configuration settings, thresholds, production rules, and parameters for
Intelligent Examination Hall Monitoring and Anomaly Detection Agent.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class SeatRegion:
    seat_id: int
    label: str
    row: int
    col: int
    # Normalized coordinates [0.0, 1.0]
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_pixel_box(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        return (
            int(self.x1 * frame_w),
            int(self.y1 * frame_h),
            int(self.x2 * frame_w),
            int(self.y2 * frame_h)
        )

    def contains_point(self, px: float, py: float, frame_w: int, frame_h: int, margin: float = 15.0) -> bool:
        bx1, by1, bx2, by2 = self.to_pixel_box(frame_w, frame_h)
        return (bx1 - margin <= px <= bx2 + margin) and (by1 - margin <= py <= by2 + margin)

@dataclass
class RuleDefinition:
    rule_id: str
    name: str
    event_type: str
    description: str
    thresholds: Dict[str, Any]
    target_state: str
    base_severity: float

@dataclass
class AppSettings:
    # Project paths
    base_dir: str = BASE_DIR
    model_path: str = os.path.join(BASE_DIR, "models", "yolov8n.pt")
    database_path: str = os.path.join(BASE_DIR, "data", "exam_monitor.db")
    evidence_dir: str = os.path.join(BASE_DIR, "data", "evidence")
    sample_videos_dir: str = os.path.join(BASE_DIR, "data", "sample_videos")
    reports_dir: str = os.path.join(BASE_DIR, "outputs", "reports")

    # Video processing defaults
    supported_video_formats: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")
    default_frame_sampling_fps: float = 5.0    # Process 5 frames per second of video
    default_detection_confidence: float = 0.35
    default_iou_threshold: float = 0.45
    device: str = "cpu"

    # Tracking parameters
    max_disappeared_frames: int = 15
    max_matching_distance_px: float = 120.0

    # Behavioral Thresholds
    # A. Leaving Seat
    seat_departure_duration_threshold_sec: float = 5.0
    seat_return_grace_period_sec: float = 2.0
    seat_boundary_margin_px: float = 25.0

    # B. Repeated Head Turning
    head_turn_angle_threshold_deg: float = 26.0
    head_turn_count_threshold: int = 3
    head_turn_window_seconds: float = 25.0
    head_turn_duration_threshold_sec: float = 4.0

    # C. Mobile Phone Detection
    phone_confidence_threshold: float = 0.38
    phone_persistence_frames: int = 4
    phone_workspace_margin_px: float = 45.0

    # D. Multiple Students Gathering
    gathering_proximity_pixels: float = 100.0
    gathering_min_students: int = 2
    gathering_duration_threshold_sec: float = 3.5

    # E. Unusual Movement
    unusual_movement_velocity_px_per_sec: float = 40.0
    unusual_movement_duration_sec: float = 3.0

    # Model-Based Agent State Machine thresholds
    state_observe_threshold: float = 25.0
    state_suspicious_threshold: float = 55.0
    state_alert_threshold: float = 80.0
    risk_decay_per_second: float = 2.5
    evidence_stale_window_seconds: float = 90.0

    # Risk Scoring Weights (sum to 1.0)
    weight_detection_confidence: float = 0.25
    weight_event_duration: float = 0.30
    weight_frequency: float = 0.20
    weight_event_type: float = 0.15
    weight_tracking_reliability: float = 0.10

    # Activity Severity Multipliers
    activity_severity: Dict[str, float] = field(default_factory=lambda: {
        "MOBILE_PHONE_DETECTED": 1.00,
        "LEAVING_SEAT": 0.88,
        "GATHERING_CLUSTER": 0.85,
        "REPEATED_HEAD_TURNING": 0.75,
        "UNUSUAL_MOVEMENT": 0.65,
    })

    # Data retention
    max_evidence_retention_days: int = 30
    auto_delete_resolved_evidence: bool = False

    def generate_default_seating_grid(self, rows: int = 3, cols: int = 6) -> Dict[int, SeatRegion]:
        """Generates standard classroom 3x6 grid with 18 seats."""
        seats = {}
        margin_x = 0.12
        margin_y = 0.18
        grid_w = 1.0 - 2 * margin_x
        grid_h = 1.0 - 2 * margin_y

        col_spacing = grid_w / (cols - 1) if cols > 1 else 0
        row_spacing = grid_h / (rows - 1) if rows > 1 else 0
        seat_w = 0.09
        seat_h = 0.15

        seat_id = 1
        for r in range(rows):
            for c in range(cols):
                cx = margin_x + c * col_spacing
                cy = margin_y + r * row_spacing
                seats[seat_id] = SeatRegion(
                    seat_id=seat_id,
                    label=f"Seat {seat_id}",
                    row=r + 1,
                    col=c + 1,
                    x1=cx - seat_w / 2.0,
                    y1=cy - seat_h / 2.0,
                    x2=cx + seat_w / 2.0,
                    y2=cy + seat_h / 2.0
                )
                seat_id += 1
        return seats

DEFAULT_SETTINGS = AppSettings()
