"""
Model-Based AI Agent Module.
Implements Sections 5 & 6:
  - Finite State Machine: NORMAL -> OBSERVE -> SUSPICIOUS -> ALERT
  - Temporal observation buffering across sliding windows
  - Event duration, start/end times, and recurrence calculations
  - Risk-score decay when behavior stops
  - Human review state tracking (UNREVIEWED, ACKNOWLEDGED, UNDER_REVIEW, RESOLVED, FALSE_POSITIVE)
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque, Any
from collections import deque
import time

from config.settings import AppSettings, DEFAULT_SETTINGS
from src.behavior import BehaviorObservation

class AgentState(str, Enum):
    NORMAL = "NORMAL"
    OBSERVE = "OBSERVE"
    SUSPICIOUS = "SUSPICIOUS"
    ALERT = "ALERT"

    @property
    def color_hex(self) -> str:
        if self == AgentState.NORMAL:
            return "#2ecc71"
        elif self == AgentState.OBSERVE:
            return "#f1c40f"
        elif self == AgentState.SUSPICIOUS:
            return "#e67e22"
        return "#e74c3c"

    @property
    def color_bgr(self) -> Tuple[int, int, int]:
        if self == AgentState.NORMAL:
            return (50, 205, 50)     # Green
        elif self == AgentState.OBSERVE:
            return (0, 215, 255)     # Yellow
        elif self == AgentState.SUSPICIOUS:
            return (0, 140, 255)     # Orange
        return (34, 34, 255)         # Red

class ReviewState(str, Enum):
    UNREVIEWED = "UNREVIEWED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    UNDER_REVIEW = "UNDER_REVIEW"
    RESOLVED = "RESOLVED"
    FALSE_POSITIVE = "FALSE_POSITIVE"

@dataclass
class TimestampedObservation:
    timestamp: float
    event_type: str
    confidence: float
    details: Dict = field(default_factory=dict)

@dataclass
class PersonAgentMemory:
    """
    Internal model memory representation for an individual candidate.
    Maintains temporal event histories, state transitions, review status,
    and decay dynamics.
    """
    tracking_id: int
    assigned_seat_id: Optional[int]
    current_state: AgentState = AgentState.NORMAL
    review_state: ReviewState = ReviewState.UNREVIEWED
    current_risk_score: float = 0.0
    first_seen_timestamp: float = 0.0
    last_seen_timestamp: float = 0.0

    # Temporal observation buffer (sliding window)
    observation_buffer: Deque[TimestampedObservation] = field(default_factory=lambda: deque(maxlen=150))

    # Event episodes: event_type -> (start_time, last_active_time)
    active_episodes: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Head turn oscillatory tracker: (timestamp, angle, direction)
    head_turn_history: Deque[Tuple[float, float, str]] = field(default_factory=lambda: deque(maxlen=60))
    last_turn_dir: Optional[str] = None
    last_turn_ts: float = 0.0

    # Audit history of state changes
    state_transitions: List[Tuple[float, AgentState, str]] = field(default_factory=list)

    # Generated evidence references
    evidence_snapshots: List[str] = field(default_factory=list)

    # Consecutive normal observation counter
    consecutive_normal_ticks: int = 0
    total_alerts_generated: int = 0

    # ML Behavioral Anomaly tracking
    ml_anomaly_score: float = 0.0
    last_ml_explanation: Dict[str, Any] = field(default_factory=dict)

    def update_ml_anomaly(self, anomaly_score: float, explanation: Optional[Dict[str, Any]] = None):
        """Records trained ML anomaly detector output."""
        self.ml_anomaly_score = float(anomaly_score)
        if explanation:
            self.last_ml_explanation = explanation


    def record_head_turn(self, timestamp: float, yaw_deg: float, angle_threshold: float = 25.0):
        """Records a head turn and checks for distinct oscillation episodes."""
        if abs(yaw_deg) >= angle_threshold:
            direction = "RIGHT" if yaw_deg > 0 else "LEFT"
            if direction != self.last_turn_dir or (timestamp - self.last_turn_ts) > 1.2:
                self.head_turn_history.append((timestamp, yaw_deg, direction))
                self.last_turn_dir = direction
                self.last_turn_ts = timestamp

    def get_turn_count_in_window(self, current_time: float, window_sec: float = 25.0) -> int:
        cutoff = current_time - window_sec
        return sum(1 for ts, _, _ in self.head_turn_history if ts >= cutoff)

    def update_event_episode(
        self,
        event_type: str,
        is_active: bool,
        current_time: float,
        grace_window_sec: float = 2.5
    ) -> float:
        """
        Calculates ongoing episodic duration with temporal smoothing.
        Permits brief natural pauses (grace window) without resetting duration.
        """
        if is_active:
            if event_type not in self.active_episodes:
                self.active_episodes[event_type] = (current_time, current_time)
            else:
                start_t, _ = self.active_episodes[event_type]
                self.active_episodes[event_type] = (start_t, current_time)
            start_t, last_t = self.active_episodes[event_type]
            return max(0.0, current_time - start_t)
        else:
            if event_type in self.active_episodes:
                start_t, last_t = self.active_episodes[event_type]
                # If still within grace window, maintain episode
                if (current_time - last_t) <= grace_window_sec:
                    return max(0.0, current_time - start_t)
                else:
                    # Episode ended
                    del self.active_episodes[event_type]
            return 0.0

    def apply_risk_decay(self, dt: float, decay_rate: float = 2.5):
        """Gradually decays risk score when behavior ceases or evidence becomes stale."""
        if self.current_risk_score > 0.0:
            self.current_risk_score = max(0.0, self.current_risk_score - decay_rate * dt)
            self._update_state_from_score(self.last_seen_timestamp, "Temporal decay (calm behavior)")

    def _update_state_from_score(self, current_time: float, reason: str):
        """FSM transition based on current risk score."""
        old_state = self.current_state
        if self.current_risk_score >= 80.0:
            new_state = AgentState.ALERT
        elif self.current_risk_score >= 55.0:
            new_state = AgentState.SUSPICIOUS
        elif self.current_risk_score >= 25.0:
            new_state = AgentState.OBSERVE
        else:
            new_state = AgentState.NORMAL

        if new_state != old_state:
            self.current_state = new_state
            self.state_transitions.append((current_time, new_state, reason))
