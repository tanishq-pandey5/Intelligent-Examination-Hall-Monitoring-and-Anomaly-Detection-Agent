"""
Configuration module for Intelligent Examination Hall Monitoring and Anomaly Detection Agent.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os

@dataclass
class SeatConfig:
    seat_id: int
    label: str
    row: int
    col: int
    center_x: float  # Normalized [0, 1]
    center_y: float  # Normalized [0, 1]
    width: float     # Normalized [0, 1]
    height: float    # Normalized [0, 1]

    def to_pixel_box(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """Returns (x1, y1, x2, y2) in absolute pixel coordinates."""
        cx, cy = self.center_x * frame_w, self.center_y * frame_h
        w, h = self.width * frame_w, self.height * frame_h
        return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))

@dataclass
class AnomalyThresholds:
    # 1. Repeated Head Turning
    head_turn_count_threshold: int = 3       # Number of turns in window
    head_turn_duration_threshold: float = 4.0 # Seconds sustained turning
    head_turn_window_seconds: float = 25.0    # Sliding window to track frequency
    head_turn_angle_degrees: float = 28.0     # Degrees off-center to count as turning

    # 2. Leaving Assigned Seat
    leaving_seat_distance_pixels: float = 75.0 # Max allowed distance from desk center (at 720p)
    leaving_seat_duration_threshold: float = 5.0 # Seconds outside seat before Alert

    # 3. Mobile Phone Detection
    phone_confidence_threshold: float = 0.40
    phone_duration_threshold: float = 2.0     # Seconds visible near student desk

    # 4. Gathering / Cluster near one location
    gathering_proximity_pixels: float = 110.0 # Distance between student centers to trigger gathering
    gathering_min_students: int = 2
    gathering_duration_threshold: float = 3.5  # Seconds in close cluster

    # 5. Unusual Movement during examination
    unusual_movement_velocity_pixels: float = 42.0 # Sudden movement speed px/sec
    unusual_movement_duration_threshold: float = 3.0 # Seconds of sustained erratic motion

@dataclass
class ScoringWeights:
    confidence_weight: float = 0.30
    duration_weight: float = 0.40
    frequency_weight: float = 0.30

    # Activity risk multipliers (reflecting severity of offence in exam hall)
    activity_severity: Dict[str, float] = field(default_factory=lambda: {
        "MOBILE_PHONE": 1.00,
        "LEAVING_SEAT": 0.88,
        "GATHERING": 0.85,
        "REPEATED_HEAD_TURNING": 0.75,
        "UNUSUAL_MOVEMENT": 0.65,
    })

@dataclass
class SystemConfig:
    # Model settings
    yolo_model_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yolov8n.pt")
    device: str = "cpu"  # 'cpu', 'mps', or 'cuda'
    detection_confidence: float = 0.35
    iou_threshold: float = 0.45

    # Video defaults
    default_frame_width: int = 1280
    default_frame_height: int = 720
    default_fps: float = 25.0

    # Hall seat grid setup (e.g., 3 rows x 6 columns = 18 seats)
    hall_rows: int = 3
    hall_cols: int = 6
    total_seats: int = 18

    # Paths
    evidence_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "evidence")
    database_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "exam_monitor.db")
    sample_videos_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_videos")

    # Thresholds and weights
    thresholds: AnomalyThresholds = field(default_factory=AnomalyThresholds)
    weights: ScoringWeights = field(default_factory=ScoringWeights)

    # State machine thresholds
    state_observe_threshold: float = 25.0    # 25-54: OBSERVE
    state_suspicious_threshold: float = 55.0 # 55-79: SUSPICIOUS
    state_alert_threshold: float = 80.0      # 80-100: ALERT
    risk_decay_per_second: float = 2.5       # Score decay for calm/normal behaviour

    def generate_default_seats(self) -> Dict[int, SeatConfig]:
        """Generates standard 3x6 classroom seating grid with 18 seats."""
        seats = {}
        margin_x = 0.12
        margin_y = 0.20
        grid_w = 1.0 - 2 * margin_x
        grid_h = 1.0 - 2 * margin_y

        col_spacing = grid_w / (self.hall_cols - 1) if self.hall_cols > 1 else 0
        row_spacing = grid_h / (self.hall_rows - 1) if self.hall_rows > 1 else 0

        seat_id = 1
        for r in range(self.hall_rows):
            for c in range(self.hall_cols):
                cx = margin_x + c * col_spacing
                cy = margin_y + r * row_spacing
                seats[seat_id] = SeatConfig(
                    seat_id=seat_id,
                    label=f"Seat {seat_id}",
                    row=r + 1,
                    col=c + 1,
                    center_x=cx,
                    center_y=cy,
                    width=0.10,
                    height=0.16
                )
                seat_id += 1
        return seats

DEFAULT_CONFIG = SystemConfig()
