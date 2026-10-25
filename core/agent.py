"""
Model-Based Examination Monitoring Agent.
Implements Tasks 5, 6, 7, 8, 9, 10.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import time

from core.config import SystemConfig, SeatConfig, DEFAULT_CONFIG
from core.state import (
    StudentState, EventType, RiskLevel, StudentMemory, Observation
)
from core.reasoning import RuleBasedReasoningEngine, RuleEvaluationResult
from core.risk_scorer import RiskScorer, RiskBreakdown

@dataclass
class RawPerception:
    """Perception data extracted from raw detector and tracker for a single student."""
    student_id: int
    bbox: Tuple[int, int, int, int]           # (x1, y1, x2, y2)
    centroid: Tuple[float, float]             # (cx, cy)
    assigned_seat_id: int
    head_turn_angle_deg: float                # Positive=Right, Negative=Left, ~0=Forward
    is_head_turning: bool
    phone_detected: bool
    phone_confidence: float
    neighbor_distances: List[Tuple[int, float]] # (other_student_id, distance_pixels)
    velocity_px_per_sec: float
    detection_confidence: float = 0.85

@dataclass
class AgentDecision:
    """Agent output and state representation for a single student."""
    student_id: int
    seat_id: int
    previous_state: StudentState
    current_state: StudentState
    risk_score: float
    risk_level: RiskLevel
    risk_breakdown: Optional[RiskBreakdown]
    is_alert_triggered: bool
    explainable_alert: Optional[str]
    matched_rules: List[RuleEvaluationResult]
    state_changed: bool

class ModelBasedExamMonitoringAgent:
    """
    Model-Based Agent that maintains an internal representation of the exam hall,
    tracks per-student behavioral history over time, and executes rule-based
    reasoning with temporal integration.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.config = config
        self.seats = config.generate_default_seats()
        self.reasoning_engine = RuleBasedReasoningEngine(config.thresholds)
        self.risk_scorer = RiskScorer(config.weights)

        # Internal state memory: maps student_id -> StudentMemory
        self.student_memories: Dict[int, StudentMemory] = {}

        # Alert debounce: maps (student_id, rule_id) -> last_alert_time
        self.alert_cooldowns: Dict[Tuple[int, str], float] = {}
        self.alert_cooldown_seconds: float = 3.5

    def get_or_create_memory(
        self,
        student_id: int,
        assigned_seat_id: int,
        seat_center: Tuple[float, float]
    ) -> StudentMemory:
        """Retrieves or initializes internal memory for a student."""
        if student_id not in self.student_memories:
            self.student_memories[student_id] = StudentMemory(
                student_id=student_id,
                assigned_seat_id=assigned_seat_id,
                assigned_seat_center=seat_center,
                current_state=StudentState.NORMAL,
                current_risk_score=0.0
            )
        return self.student_memories[student_id]

    def process_frame_perceptions(
        self,
        perceptions: List[RawPerception],
        current_time: float,
        dt: float
    ) -> List[AgentDecision]:
        """
        Main reasoning cycle of the Model-Based Agent:
        1. Updates internal memory models
        2. Computes continuous durations & frequencies
        3. Evaluates rule-based reasoning system
        4. Calculates multi-factor certainty/risk scores
        5. Updates internal FSM state (NORMAL -> OBSERVE -> SUSPICIOUS -> ALERT)
        6. Emits explainable alerts
        """
        decisions: List[AgentDecision] = []
        active_student_ids = set()

        for p in perceptions:
            active_student_ids.add(p.student_id)
            seat_info = self.seats.get(p.assigned_seat_id)
            seat_center = (0.0, 0.0)
            if seat_info:
                # Approximate pixel center if needed, or normalized
                seat_center = (
                    seat_info.center_x * self.config.default_frame_width,
                    seat_info.center_y * self.config.default_frame_height
                )

            memory = self.get_or_create_memory(p.student_id, p.assigned_seat_id, seat_center)
            old_state = memory.current_state

            # Update memory spatial state
            memory.last_bbox = p.bbox
            memory.last_centroid = p.centroid
            memory.last_observed_timestamp = current_time

            # Record head turn dynamics
            turn_direction = "RIGHT" if p.head_turn_angle_deg > 0 else "LEFT"
            memory.record_turn(
                timestamp=current_time,
                angle_deg=p.head_turn_angle_deg,
                direction=turn_direction,
                angle_threshold=self.config.thresholds.head_turn_angle_degrees
            )

            # Record motion history
            memory.motion_history.append((current_time, p.velocity_px_per_sec))

            # ------------------------------------------------------------------
            # Calculate continuous durations for each event type
            # ------------------------------------------------------------------
            # 1. Head turning continuous duration
            head_turn_dur = memory.update_continuous_event(
                EventType.REPEATED_HEAD_TURNING,
                p.is_head_turning,
                current_time
            )

            # 2. Leaving seat continuous duration
            # Distance from assigned seat center
            dx = p.centroid[0] - seat_center[0]
            dy = p.centroid[1] - seat_center[1]
            dist_from_seat = (dx*dx + dy*dy) ** 0.5
            is_outside_seat = dist_from_seat > self.config.thresholds.leaving_seat_distance_pixels
            leave_dur = memory.update_continuous_event(
                EventType.LEAVING_SEAT,
                is_outside_seat,
                current_time
            )

            # 3. Mobile phone continuous duration
            phone_dur = memory.update_continuous_event(
                EventType.MOBILE_PHONE_DETECTED,
                p.phone_detected,
                current_time
            )

            # 4. Gathering / cluster continuous duration
            close_neighbors = [
                d for s_id, d in p.neighbor_distances
                if d <= self.config.thresholds.gathering_proximity_pixels
            ]
            is_gathering = len(close_neighbors) >= (self.config.thresholds.gathering_min_students - 1)
            min_dist = min(close_neighbors) if close_neighbors else 999.0
            gather_dur = memory.update_continuous_event(
                EventType.GATHERING_CLUSTER,
                is_gathering,
                current_time
            )

            # 5. Unusual movement continuous duration
            is_unusual_motion = p.velocity_px_per_sec >= self.config.thresholds.unusual_movement_velocity_pixels
            motion_dur = memory.update_continuous_event(
                EventType.UNUSUAL_MOVEMENT,
                is_unusual_motion,
                current_time
            )

            # ------------------------------------------------------------------
            # Rule-Based Reasoning Engine Evaluation (Task 7)
            # ------------------------------------------------------------------
            matched_rules: List[RuleEvaluationResult] = []

            # Rule 1: Repeated Head Turning
            r1 = self.reasoning_engine.evaluate_head_turning_rule(memory, current_time, head_turn_dur)
            if r1:
                matched_rules.append(r1)

            # Rule 2: Leaving Assigned Seat
            r2 = self.reasoning_engine.evaluate_leaving_seat_rule(memory, dist_from_seat, leave_dur)
            if r2:
                matched_rules.append(r2)

            # Rule 3: Mobile Phone Detection
            r3 = self.reasoning_engine.evaluate_mobile_phone_rule(memory, p.phone_confidence, phone_dur)
            if r3:
                matched_rules.append(r3)

            # Rule 4: Gathering Near Location
            r4 = self.reasoning_engine.evaluate_gathering_rule(
                memory, len(close_neighbors) + 1, gather_dur, min_dist
            )
            if r4:
                matched_rules.append(r4)

            # Rule 5: Unusual Movement
            r5 = self.reasoning_engine.evaluate_unusual_movement_rule(memory, p.velocity_px_per_sec, motion_dur)
            if r5:
                matched_rules.append(r5)

            # Rule 6: Composite Escalation
            r6 = self.reasoning_engine.evaluate_composite_escalation(memory, matched_rules, current_time)
            if r6:
                matched_rules.append(r6)

            # ------------------------------------------------------------------
            # Multi-Factor Certainty and Risk Scoring (Tasks 8 & 9)
            # ------------------------------------------------------------------
            risk_breakdown: Optional[RiskBreakdown] = None
            alert_triggered = False
            explainable_alert: Optional[str] = None

            if matched_rules:
                # Select highest priority rule
                primary_rule = matched_rules[0]
                # Determine frequency and duration for the primary event
                ev_type = primary_rule.event_type
                if ev_type == EventType.REPEATED_HEAD_TURNING:
                    dur = head_turn_dur
                    freq = memory.get_head_turn_count_in_window(current_time)
                    det_conf = p.detection_confidence
                elif ev_type == EventType.LEAVING_SEAT:
                    dur = leave_dur
                    freq = 1
                    det_conf = p.detection_confidence
                elif ev_type == EventType.MOBILE_PHONE_DETECTED:
                    dur = phone_dur
                    freq = 1
                    det_conf = p.phone_confidence
                elif ev_type == EventType.GATHERING_CLUSTER:
                    dur = gather_dur
                    freq = len(close_neighbors) + 1
                    det_conf = p.detection_confidence
                else:
                    dur = motion_dur
                    freq = 2
                    det_conf = p.detection_confidence

                risk_breakdown = self.risk_scorer.compute_risk(
                    event_type=ev_type,
                    detection_confidence=det_conf,
                    duration_seconds=dur,
                    frequency_count=freq,
                    memory=memory,
                    current_time=current_time
                )

                # Update memory risk score
                memory.current_risk_score = risk_breakdown.total_risk_score
                memory.current_risk_level = risk_breakdown.risk_level
                memory.consecutive_normal_frames = 0

                # Record observation in historical buffer
                obs = Observation(
                    timestamp=current_time,
                    event_type=ev_type,
                    confidence=det_conf,
                    details={"duration": dur, "rule": primary_rule.rule_id}
                )
                memory.history.append(obs)

                # Update FSM state (Task 5: NORMAL -> OBSERVE -> SUSPICIOUS -> ALERT)
                # Ensure target_state from rules also enforces minimum floor
                suggested_state = primary_rule.target_state
                state_order = [StudentState.NORMAL, StudentState.OBSERVE, StudentState.SUSPICIOUS, StudentState.ALERT]
                score_state = (
                    StudentState.ALERT if memory.current_risk_score >= self.config.state_alert_threshold
                    else StudentState.SUSPICIOUS if memory.current_risk_score >= self.config.state_suspicious_threshold
                    else StudentState.OBSERVE if memory.current_risk_score >= self.config.state_observe_threshold
                    else StudentState.NORMAL
                )
                new_state = max([suggested_state, score_state], key=lambda s: state_order.index(s))
                if new_state != memory.current_state:
                    memory.current_state = new_state
                    memory.state_transitions.append((current_time, new_state, primary_rule.explainable_text))

                # Check if we should emit a high-level alert toast/event
                if memory.current_state in [StudentState.SUSPICIOUS, StudentState.ALERT]:
                    cooldown_key = (p.student_id, primary_rule.rule_id)
                    last_alert = self.alert_cooldowns.get(cooldown_key, 0.0)
                    if (current_time - last_alert) >= self.alert_cooldown_seconds:
                        alert_triggered = True
                        explainable_alert = primary_rule.explainable_text
                        self.alert_cooldowns[cooldown_key] = current_time
                        memory.total_alerts_emitted += 1
                        memory.last_alert_timestamp = current_time

            else:
                # Student is behaving normally in this frame
                memory.consecutive_normal_frames += 1
                # Decay risk score gradually (Task 6: Model-based memory decay)
                memory.decay_risk(dt, decay_rate=self.config.risk_decay_per_second)

            decisions.append(AgentDecision(
                student_id=p.student_id,
                seat_id=p.assigned_seat_id,
                previous_state=old_state,
                current_state=memory.current_state,
                risk_score=memory.current_risk_score,
                risk_level=memory.current_risk_level,
                risk_breakdown=risk_breakdown,
                is_alert_triggered=alert_triggered,
                explainable_alert=explainable_alert,
                matched_rules=matched_rules,
                state_changed=(old_state != memory.current_state)
            ))

        return decisions
