"""
Rule-Based Reasoning Engine Module.
Implements Section 7:
  - Configurable IF-THEN production rules loaded from config/settings.py
  - Evaluates temporal observations and behavioral indicators
  - Escalates review priority upon multiple co-occurring indicators
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from config.settings import AppSettings, DEFAULT_SETTINGS
from src.agent import PersonAgentMemory, AgentState

@dataclass
class RuleMatchResult:
    rule_id: str
    rule_name: str
    event_type: str
    matched: bool
    suggested_state: AgentState
    severity_weight: float
    reason_explanation: str
    metrics: Dict[str, Any]

class RuleBasedReasoningEngine:
    """
    Expert rule inference engine executing production rules over student agent memory.
    """

    def __init__(self, settings: AppSettings = DEFAULT_SETTINGS):
        self.settings = settings

    def evaluate_all_rules(
        self,
        memory: PersonAgentMemory,
        current_time: float,
        active_observations: Dict[str, Any]
    ) -> List[RuleMatchResult]:
        """
        Evaluates production rules against candidate observations.
        """
        matches: List[RuleMatchResult] = []

        # ----------------------------------------------------------------------
        # RULE 1: Seat Departure Rule
        # IF person remains outside assigned seat > duration, THEN create event
        # ----------------------------------------------------------------------
        if "seat_departure" in active_observations:
            obs = active_observations["seat_departure"]
            dur = obs.get("duration_sec", 0.0)
            dist = obs.get("distance_px", 0.0)
            t_leave = self.settings.seat_departure_duration_threshold_sec

            if dur >= t_leave:
                rec_state = AgentState.ALERT if dur >= 8.0 else AgentState.SUSPICIOUS
                matches.append(RuleMatchResult(
                    rule_id="RULE_01_SEAT_DEPARTURE",
                    rule_name="Seat Departure",
                    event_type="LEAVING_SEAT",
                    matched=True,
                    suggested_state=rec_state,
                    severity_weight=self.settings.activity_severity.get("LEAVING_SEAT", 0.88),
                    reason_explanation=(
                        f"Candidate remained outside assigned seat region ({dist:.0f}px departure) "
                        f"for {dur:.1f}s, exceeding threshold ({t_leave:.1f}s)."
                    ),
                    metrics={"duration_sec": dur, "distance_px": dist}
                ))

        # ----------------------------------------------------------------------
        # RULE 2: Repeated Head Turn Rule
        # IF multiple distinct turns occur in window > T, THEN increase priority
        # ----------------------------------------------------------------------
        turn_count = memory.get_turn_count_in_window(current_time, self.settings.head_turn_window_seconds)
        turn_dur = memory.update_event_episode(
            "REPEATED_HEAD_TURNING",
            active_observations.get("is_head_turning", False),
            current_time
        )
        t_count = self.settings.head_turn_count_threshold
        t_dur = self.settings.head_turn_duration_threshold_sec

        if turn_count >= t_count and turn_dur >= t_dur:
            rec_state = AgentState.ALERT if turn_dur >= 12.0 or turn_count >= 6 else AgentState.SUSPICIOUS
            matches.append(RuleMatchResult(
                rule_id="RULE_02_HEAD_TURNING",
                rule_name="Repeated Head Turning",
                event_type="REPEATED_HEAD_TURNING",
                matched=True,
                suggested_state=rec_state,
                severity_weight=self.settings.activity_severity.get("REPEATED_HEAD_TURNING", 0.75),
                reason_explanation=(
                    f"Repeated side movements ({turn_count} distinct turns) observed during "
                    f"analysis window with sustained duration of {turn_dur:.1f}s."
                ),
                metrics={"turn_count": turn_count, "duration_sec": turn_dur}
            ))

        # ----------------------------------------------------------------------
        # RULE 3: Mobile Phone Persistence Rule
        # IF phone detection persists across frames, THEN create phone review event
        # ----------------------------------------------------------------------
        if "phone" in active_observations:
            obs = active_observations["phone"]
            conf = obs.get("confidence", 0.0)
            dur = obs.get("duration_sec", 0.0)
            frames = obs.get("persistence_frames", 0)

            if frames >= self.settings.phone_persistence_frames or dur >= 2.0:
                is_confirmed = obs.get("is_confirmed_use", False)
                matches.append(RuleMatchResult(
                    rule_id="RULE_03_PHONE_DETECTION",
                    rule_name="Mobile Phone Detected",
                    event_type="MOBILE_PHONE_DETECTED",
                    matched=True,
                    suggested_state=AgentState.ALERT,
                    severity_weight=self.settings.activity_severity.get("MOBILE_PHONE_DETECTED", 1.00),
                    reason_explanation=(
                        f"Mobile device {'confirmed in active use' if is_confirmed else 'detected in workspace'} "
                        f"across {frames} frames ({dur:.1f}s) with confidence {int(conf*100)}%."
                    ),
                    metrics={"confidence": conf, "duration_sec": dur, "confirmed_use": is_confirmed}
                ))

        # ----------------------------------------------------------------------
        # RULE 4: Group Gathering Rule
        # IF several people remain close together, THEN create group movement event
        # ----------------------------------------------------------------------
        if "gathering" in active_observations:
            obs = active_observations["gathering"]
            g_size = obs.get("group_size", 0)
            dur = obs.get("duration_sec", 0.0)

            if g_size >= self.settings.gathering_min_students and dur >= self.settings.gathering_duration_threshold_sec:
                matches.append(RuleMatchResult(
                    rule_id="RULE_04_GATHERING",
                    rule_name="Group Gathering",
                    event_type="GATHERING_CLUSTER",
                    matched=True,
                    suggested_state=AgentState.ALERT if g_size >= 3 else AgentState.SUSPICIOUS,
                    severity_weight=self.settings.activity_severity.get("GATHERING_CLUSTER", 0.85),
                    reason_explanation=(
                        f"Group gathering of {g_size} people persisted near candidate workspace "
                        f"for {dur:.1f}s."
                    ),
                    metrics={"group_size": g_size, "duration_sec": dur}
                ))

        # ----------------------------------------------------------------------
        # RULE 5: Unusual Movement Rule
        # ----------------------------------------------------------------------
        if "unusual_movement" in active_observations:
            obs = active_observations["unusual_movement"]
            vel = obs.get("velocity_px_per_sec", 0.0)
            dur = obs.get("duration_sec", 0.0)

            if vel >= self.settings.unusual_movement_velocity_px_per_sec and dur >= self.settings.unusual_movement_duration_sec:
                matches.append(RuleMatchResult(
                    rule_id="RULE_05_UNUSUAL_MOVEMENT",
                    rule_name="Unusual Movement",
                    event_type="UNUSUAL_MOVEMENT",
                    matched=True,
                    suggested_state=AgentState.SUSPICIOUS,
                    severity_weight=self.settings.activity_severity.get("UNUSUAL_MOVEMENT", 0.65),
                    reason_explanation=(
                        f"Sustained unusual movement ({vel:.1f} px/s) observed exceeding "
                        f"calm seating baseline for {dur:.1f}s."
                    ),
                    metrics={"velocity": vel, "duration_sec": dur}
                ))

        # ----------------------------------------------------------------------
        # RULE 6: Co-Occurring Multiple Indicators Escalation Rule
        # IF multiple independent indicators persist, THEN increase review priority
        # ----------------------------------------------------------------------
        if len(matches) >= 2:
            event_names = [m.rule_name for m in matches]
            matches.append(RuleMatchResult(
                rule_id="RULE_06_COMPOSITE_ESCALATION",
                rule_name="Composite Multi-Indicator Escalation",
                event_type="COMPOSITE_ANOMALY",
                matched=True,
                suggested_state=AgentState.ALERT,
                severity_weight=1.00,
                reason_explanation=(
                    f"High Priority Escalation: Multiple independent behavioral anomalies "
                    f"co-occurred simultaneously ({', '.join(event_names)})."
                ),
                metrics={"co_occurring_count": len(matches)}
            ))

        return matches
