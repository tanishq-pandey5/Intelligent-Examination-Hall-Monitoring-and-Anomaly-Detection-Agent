"""
State definitions and student memory representations for Model-Based Agent.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
import time

class StudentState(str, Enum):
    NORMAL = "NORMAL"
    OBSERVE = "OBSERVE"
    SUSPICIOUS = "SUSPICIOUS"
    ALERT = "ALERT"

    @property
    def color_bgr(self) -> Tuple[int, int, int]:
        """Color code for video annotations: Green -> Yellow -> Orange -> Red."""
        if self == StudentState.NORMAL:
            return (50, 205, 50)     # Bright Green
        elif self == StudentState.OBSERVE:
            return (0, 215, 255)     # Amber / Yellow
        elif self == StudentState.SUSPICIOUS:
            return (0, 140, 255)     # Deep Orange
        elif self == StudentState.ALERT:
            return (34, 34, 255)     # Crimson Red
        return (200, 200, 200)

    @property
    def color_hex(self) -> str:
        """Hex color code for web UI dashboards."""
        if self == StudentState.NORMAL:
            return "#2ecc71"
        elif self == StudentState.OBSERVE:
            return "#f1c40f"
        elif self == StudentState.SUSPICIOUS:
            return "#e67e22"
        elif self == StudentState.ALERT:
            return "#e74c3c"
        return "#95a5a6"

class EventType(str, Enum):
    LEAVING_SEAT = "LEAVING_SEAT"
    REPEATED_HEAD_TURNING = "REPEATED_HEAD_TURNING"
    MOBILE_PHONE_DETECTED = "MOBILE_PHONE_DETECTED"
    GATHERING_CLUSTER = "GATHERING_CLUSTER"
    UNUSUAL_MOVEMENT = "UNUSUAL_MOVEMENT"

    @property
    def human_name(self) -> str:
        mapping = {
            EventType.LEAVING_SEAT: "Student leaving assigned seat",
            EventType.REPEATED_HEAD_TURNING: "Repeated turning toward another student",
            EventType.MOBILE_PHONE_DETECTED: "Mobile phone detection",
            EventType.GATHERING_CLUSTER: "Multiple students gathering near one location",
            EventType.UNUSUAL_MOVEMENT: "Unusual movement during examination",
        }
        return mapping.get(self, self.value)

class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        if score < 40.0:
            return cls.LOW
        elif score < 70.0:
            return cls.MEDIUM
        elif score < 90.0:
            return cls.HIGH
        else:
            return cls.CRITICAL

    @property
    def badge_color(self) -> str:
        if self == RiskLevel.LOW:
            return "green"
        elif self == RiskLevel.MEDIUM:
            return "orange"
        elif self == RiskLevel.HIGH:
            return "red"
        return "purple"

@dataclass
class Observation:
    timestamp: float
    event_type: EventType
    confidence: float
    details: Dict = field(default_factory=dict)
    frame_index: int = 0

@dataclass
class StudentMemory:
    """
    Model-Based Agent memory for an individual student.
    Maintains historical observations, spatial trajectory, head turn oscillations,
    active duration timers, and temporal risk score so that isolated single frames
    never trigger premature alerts.
    """
    student_id: int
    assigned_seat_id: int
    current_state: StudentState = StudentState.NORMAL
    current_risk_score: float = 0.0
    current_risk_level: RiskLevel = RiskLevel.LOW

    # Spatial history
    last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    last_centroid: Tuple[float, float] = (0.0, 0.0)
    assigned_seat_center: Tuple[float, float] = (0.0, 0.0)

    # Historical observations buffer (max 100 recent)
    history: Deque[Observation] = field(default_factory=lambda: deque(maxlen=150))
    state_transitions: List[Tuple[float, StudentState, str]] = field(default_factory=list)

    # Active continuous duration timers for ongoing events
    # Maps EventType -> start_timestamp
    active_event_start_times: Dict[EventType, float] = field(default_factory=dict)

    # Head turn oscillation memory
    # Stores tuples: (timestamp, angle_deg, direction: 'LEFT'|'RIGHT')
    head_turn_events: Deque[Tuple[float, float, str]] = field(default_factory=lambda: deque(maxlen=50))
    last_turn_direction: Optional[str] = None
    last_turn_timestamp: float = 0.0

    # Motion history: (timestamp, velocity_px_per_sec)
    motion_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=60))

    # General temporal counters
    last_observed_timestamp: float = 0.0
    consecutive_normal_frames: int = 0
    total_alerts_emitted: int = 0
    last_alert_timestamp: float = 0.0

    # Last active timestamps for grace window smoothing
    last_active_timestamps: Dict[EventType, float] = field(default_factory=dict)

    def record_turn(self, timestamp: float, angle_deg: float, direction: str, angle_threshold: float = 25.0):
        """Records a head turn and detects oscillatory side turning."""
        if abs(angle_deg) >= angle_threshold:
            # Check if this is a new turn or direction switch
            if direction != self.last_turn_direction or (timestamp - self.last_turn_timestamp) > 1.2:
                self.head_turn_events.append((timestamp, angle_deg, direction))
                self.last_turn_direction = direction
                self.last_turn_timestamp = timestamp

    def get_head_turn_count_in_window(self, current_time: float, window_seconds: float = 30.0) -> int:
        """Counts how many distinct head turns occurred in the recent sliding window."""
        cutoff = current_time - window_seconds
        return sum(1 for ts, _, _ in self.head_turn_events if ts >= cutoff)

    def update_continuous_event(
        self,
        event_type: EventType,
        is_active: bool,
        current_time: float,
        grace_window: float = 2.5
    ) -> float:
        """
        Updates ongoing episodic duration for an event type with temporal memory smoothing.
        Permits intermittent brief pauses (grace window) without prematurely resetting
        repeated oscillatory behavior (e.g., student turning, glancing back, turning again).
        """
        if is_active:
            if event_type not in self.active_event_start_times:
                self.active_event_start_times[event_type] = current_time
            self.last_active_timestamps[event_type] = current_time
            duration = max(0.0, current_time - self.active_event_start_times[event_type])
            return duration
        else:
            last_active = self.last_active_timestamps.get(event_type, 0.0)
            if (current_time - last_active) <= grace_window and event_type in self.active_event_start_times:
                duration = max(0.0, current_time - self.active_event_start_times[event_type])
                return duration
            else:
                if event_type in self.active_event_start_times:
                    del self.active_event_start_times[event_type]
                return 0.0

    def decay_risk(self, dt: float, decay_rate: float = 2.5):
        """Gradually decays risk score during normal calm behavior."""
        if self.current_risk_score > 0.0:
            self.current_risk_score = max(0.0, self.current_risk_score - decay_rate * dt)
            self.current_risk_level = RiskLevel.from_score(self.current_risk_score)
            self._update_state_from_score(self.last_observed_timestamp, "Decay during normal behavior")

    def _update_state_from_score(self, timestamp: float, reason: str):
        """Transitions state based on current risk score with hysteresis."""
        old_state = self.current_state
        if self.current_risk_score >= 80.0:
            new_state = StudentState.ALERT
        elif self.current_risk_score >= 55.0:
            new_state = StudentState.SUSPICIOUS
        elif self.current_risk_score >= 25.0:
            new_state = StudentState.OBSERVE
        else:
            new_state = StudentState.NORMAL

        if new_state != old_state:
            self.current_state = new_state
            self.state_transitions.append((timestamp, new_state, reason))
