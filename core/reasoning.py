"""
Rule-Based Reasoning System and Explainable Alert Generator.
Implements Tasks 7, 9, 10.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from core.state import StudentState, EventType, RiskLevel, StudentMemory, Observation
from core.config import AnomalyThresholds

@dataclass
class RuleEvaluationResult:
    matched: bool
    rule_id: str
    rule_name: str
    event_type: EventType
    target_state: StudentState
    recommended_risk_level: RiskLevel
    explainable_text: str
    metrics: Dict[str, Any] = field(default_factory=dict)

class RuleBasedReasoningEngine:
    """
    Expert rule-based inference engine that executes structured production rules
    over student behavioral metrics and temporal observations.
    """

    def __init__(self, thresholds: AnomalyThresholds = AnomalyThresholds()):
        self.thresholds = thresholds

    def evaluate_head_turning_rule(
        self,
        memory: StudentMemory,
        current_time: float,
        turn_duration: float
    ) -> Optional[RuleEvaluationResult]:
        """
        Rule:
        IF head_turn_count > threshold
        AND duration > T
        THEN Suspicion_Level = Medium / High
        """
        window_sec = self.thresholds.head_turn_window_seconds
        turn_count = memory.get_head_turn_count_in_window(current_time, window_seconds=window_sec)
        threshold_count = self.thresholds.head_turn_count_threshold
        t_duration = self.thresholds.head_turn_duration_threshold

        # Single glance does NOT match (Model-Based Agent memory check)
        if turn_count >= threshold_count and turn_duration >= t_duration:
            # Escalation based on intensity
            if turn_count >= threshold_count + 3 or turn_duration >= t_duration * 2.5:
                rec_state = StudentState.ALERT
                rec_level = RiskLevel.HIGH
            elif turn_count >= threshold_count + 1 or turn_duration >= t_duration * 1.5:
                rec_state = StudentState.SUSPICIOUS
                rec_level = RiskLevel.MEDIUM
            else:
                rec_state = StudentState.OBSERVE
                rec_level = RiskLevel.MEDIUM

            # Generate precise explainable alert (Task 10)
            seat_str = f"Student/Seat {memory.assigned_seat_id}"
            explain = (
                f"{seat_str} → Repeated side movement detected for {turn_duration:.0f} sec "
                f"({turn_count} turns in last {int(window_sec)}s) → Risk: {rec_level.value}"
            )

            return RuleEvaluationResult(
                matched=True,
                rule_id="RULE_01_HEAD_TURNING",
                rule_name="Repeated Head Turning Toward Peer",
                event_type=EventType.REPEATED_HEAD_TURNING,
                target_state=rec_state,
                recommended_risk_level=rec_level,
                explainable_text=explain,
                metrics={
                    "head_turn_count": turn_count,
                    "duration_seconds": round(turn_duration, 1),
                    "window_seconds": window_sec,
                    "threshold_count": threshold_count,
                    "threshold_duration": t_duration
                }
            )
        return None

    def evaluate_leaving_seat_rule(
        self,
        memory: StudentMemory,
        distance_from_seat: float,
        leaving_duration: float
    ) -> Optional[RuleEvaluationResult]:
        """
        Rule:
        IF distance_from_seat > threshold
        AND duration > T_leave
        THEN Suspicion_Level = High / Alert
        """
        dist_thresh = self.thresholds.leaving_seat_distance_pixels
        t_leave = self.thresholds.leaving_seat_duration_threshold

        if distance_from_seat >= dist_thresh and leaving_duration >= t_leave:
            if leaving_duration >= t_leave * 2.0:
                rec_state = StudentState.ALERT
                rec_level = RiskLevel.HIGH
            else:
                rec_state = StudentState.SUSPICIOUS
                rec_level = RiskLevel.HIGH

            seat_str = f"Student/Seat {memory.assigned_seat_id}"
            explain = (
                f"{seat_str} → Vacated assigned seat ({distance_from_seat:.0f}px away) "
                f"for {leaving_duration:.0f} sec → Risk: {rec_level.value}"
            )

            return RuleEvaluationResult(
                matched=True,
                rule_id="RULE_02_LEAVING_SEAT",
                rule_name="Student Leaving Assigned Seat",
                event_type=EventType.LEAVING_SEAT,
                target_state=rec_state,
                recommended_risk_level=rec_level,
                explainable_text=explain,
                metrics={
                    "distance_pixels": round(distance_from_seat, 1),
                    "duration_seconds": round(leaving_duration, 1),
                    "distance_threshold": dist_thresh,
                    "duration_threshold": t_leave
                }
            )
        return None

    def evaluate_mobile_phone_rule(
        self,
        memory: StudentMemory,
        phone_confidence: float,
        phone_duration: float
    ) -> Optional[RuleEvaluationResult]:
        """
        Rule:
        IF mobile_phone_detected
        AND confidence > threshold
        AND duration > T_phone
        THEN Suspicion_Level = Alert
        """
        conf_thresh = self.thresholds.phone_confidence_threshold
        t_phone = self.thresholds.phone_duration_threshold

        if phone_confidence >= conf_thresh and phone_duration >= t_phone:
            # Immediate severe escalation: Mobile phone during exam is critical
            rec_state = StudentState.ALERT
            rec_level = RiskLevel.CRITICAL if phone_duration >= 4.0 else RiskLevel.HIGH

            seat_str = f"Student/Seat {memory.assigned_seat_id}"
            explain = (
                f"{seat_str} → Mobile phone detected in workspace for {phone_duration:.1f} sec "
                f"(confidence: {int(phone_confidence * 100)}%) → Risk: {rec_level.value}"
            )

            return RuleEvaluationResult(
                matched=True,
                rule_id="RULE_03_MOBILE_PHONE",
                rule_name="Unauthorized Mobile Device Detected",
                event_type=EventType.MOBILE_PHONE_DETECTED,
                target_state=rec_state,
                recommended_risk_level=rec_level,
                explainable_text=explain,
                metrics={
                    "detection_confidence": round(phone_confidence, 2),
                    "duration_seconds": round(phone_duration, 1),
                    "confidence_threshold": conf_thresh,
                    "duration_threshold": t_phone
                }
            )
        return None

    def evaluate_gathering_rule(
        self,
        memory: StudentMemory,
        nearby_students_count: int,
        cluster_duration: float,
        min_distance: float
    ) -> Optional[RuleEvaluationResult]:
        """
        Rule:
        IF cluster_count >= min_students
        AND proximity < threshold
        AND duration > T_gather
        THEN Suspicion_Level = High / Alert
        """
        min_students = self.thresholds.gathering_min_students
        t_gather = self.thresholds.gathering_duration_threshold

        if nearby_students_count >= min_students and cluster_duration >= t_gather:
            rec_state = StudentState.ALERT if cluster_duration >= 6.0 else StudentState.SUSPICIOUS
            rec_level = RiskLevel.HIGH

            seat_str = f"Student/Seat {memory.assigned_seat_id}"
            explain = (
                f"{seat_str} → Multiple students ({nearby_students_count} students) gathering near desk "
                f"for {cluster_duration:.0f} sec (proximity: {min_distance:.0f}px) → Risk: {rec_level.value}"
            )

            return RuleEvaluationResult(
                matched=True,
                rule_id="RULE_04_GATHERING",
                rule_name="Unauthorized Group Gathering Near Desk",
                event_type=EventType.GATHERING_CLUSTER,
                target_state=rec_state,
                recommended_risk_level=rec_level,
                explainable_text=explain,
                metrics={
                    "cluster_students": nearby_students_count,
                    "cluster_duration": round(cluster_duration, 1),
                    "min_distance_px": round(min_distance, 1),
                    "duration_threshold": t_gather
                }
            )
        return None

    def evaluate_unusual_movement_rule(
        self,
        memory: StudentMemory,
        movement_velocity: float,
        motion_duration: float
    ) -> Optional[RuleEvaluationResult]:
        """
        Rule:
        IF unusual_movement_speed > threshold
        AND duration > T_motion
        THEN Suspicion_Level = Medium / High
        """
        v_thresh = self.thresholds.unusual_movement_velocity_pixels
        t_motion = self.thresholds.unusual_movement_duration_threshold

        if movement_velocity >= v_thresh and motion_duration >= t_motion:
            rec_state = StudentState.SUSPICIOUS
            rec_level = RiskLevel.MEDIUM

            seat_str = f"Student/Seat {memory.assigned_seat_id}"
            explain = (
                f"{seat_str} → Unusual erratic movement detected for {motion_duration:.0f} sec "
                f"(velocity: {movement_velocity:.1f} px/s) → Risk: {rec_level.value}"
            )

            return RuleEvaluationResult(
                matched=True,
                rule_id="RULE_05_UNUSUAL_MOVEMENT",
                rule_name="Erratic Movement During Exam",
                event_type=EventType.UNUSUAL_MOVEMENT,
                target_state=rec_state,
                recommended_risk_level=rec_level,
                explainable_text=explain,
                metrics={
                    "velocity_px_per_sec": round(movement_velocity, 1),
                    "duration_seconds": round(motion_duration, 1),
                    "velocity_threshold": v_thresh,
                    "duration_threshold": t_motion
                }
            )
        return None

    def evaluate_composite_escalation(
        self,
        memory: StudentMemory,
        matched_results: List[RuleEvaluationResult],
        current_time: float
    ) -> Optional[RuleEvaluationResult]:
        """
        Rule: Composite Escalation
        IF multiple distinct anomaly rules match concurrently OR recent history shows repeated offences
        THEN escalate state to ALERT / CRITICAL
        """
        if len(matched_results) >= 2:
            event_names = [r.rule_name for r in matched_results]
            seat_str = f"Student/Seat {memory.assigned_seat_id}"
            explain = (
                f"{seat_str} → CRITICAL ESCALATION: Simultaneous anomalies detected "
                f"({', '.join(event_names)}) → Risk: Critical"
            )
            return RuleEvaluationResult(
                matched=True,
                rule_id="RULE_06_COMPOSITE_ESCALATION",
                rule_name="Composite Multi-Anomaly Escalation",
                event_type=matched_results[0].event_type,
                target_state=StudentState.ALERT,
                recommended_risk_level=RiskLevel.CRITICAL,
                explainable_text=explain,
                metrics={"co_occurring_rules": [r.rule_id for r in matched_results]}
            )
        return None
