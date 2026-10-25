"""
Multi-factor Certainty and Risk Scoring Engine with Temporal Reasoning.
Implements Tasks 8 & 9.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
from core.state import EventType, RiskLevel, StudentMemory
from core.config import ScoringWeights

@dataclass
class RiskBreakdown:
    total_risk_score: float         # 0.0 to 100.0
    risk_level: RiskLevel
    certainty: float                # 0.0 to 1.0
    confidence_component: float     # Contribution from detection confidence
    duration_component: float       # Contribution from duration
    frequency_component: float      # Contribution from frequency
    activity_multiplier: float      # Weight for event type
    temporal_escalation_factor: float # Escalation from past history
    explanation_summary: str

class RiskScorer:
    """
    Assigns certainty and risk score to examination events based on:
      1. Detection confidence (C_det)
      2. Duration of activity (D)
      3. Frequency of occurrence (F)
      4. Type of activity severity (W_type)
      5. Temporal reasoning escalation (M_temporal)
    """

    def __init__(self, weights: ScoringWeights = ScoringWeights()):
        self.weights = weights

        # Targets for normalization
        self.duration_target_seconds: Dict[EventType, float] = {
            EventType.MOBILE_PHONE_DETECTED: 5.0,
            EventType.LEAVING_SEAT: 8.0,
            EventType.GATHERING_CLUSTER: 6.0,
            EventType.REPEATED_HEAD_TURNING: 12.0,
            EventType.UNUSUAL_MOVEMENT: 6.0,
        }

        self.frequency_target_count: Dict[EventType, int] = {
            EventType.MOBILE_PHONE_DETECTED: 2,
            EventType.LEAVING_SEAT: 2,
            EventType.GATHERING_CLUSTER: 3,
            EventType.REPEATED_HEAD_TURNING: 5,
            EventType.UNUSUAL_MOVEMENT: 4,
        }

    def compute_risk(
        self,
        event_type: EventType,
        detection_confidence: float,
        duration_seconds: float,
        frequency_count: int,
        memory: StudentMemory,
        current_time: float
    ) -> RiskBreakdown:
        """
        Computes the full multi-factor certainty and risk score with temporal memory reinforcement.
        """
        # 1. Detection confidence component [0, 1]
        c_det = max(0.0, min(1.0, detection_confidence))

        # 2. Duration component [0, 1]
        d_target = self.duration_target_seconds.get(event_type, 6.0)
        norm_duration = max(0.0, min(1.0, duration_seconds / d_target))

        # 3. Frequency component [0, 1]
        f_target = self.frequency_target_count.get(event_type, 4)
        norm_freq = max(0.0, min(1.0, frequency_count / f_target))

        # Base weighted score
        w_c = self.weights.confidence_weight
        w_d = self.weights.duration_weight
        w_f = self.weights.frequency_weight
        base_score = (w_c * c_det) + (w_d * norm_duration) + (w_f * norm_freq)

        # 4. Activity severity multiplier
        activity_mult = self.weights.activity_severity.get(event_type.value, 0.70)

        # 5. Temporal Reasoning Escalation (Task 9)
        # Repeated suspicious actions in student memory increase risk level exponentially or with decay
        temporal_factor = 1.0
        if memory and memory.history:
            decay_tau = 90.0  # 90-second temporal memory decay window
            recent_relevance = 0.0
            for obs in memory.history:
                dt = current_time - obs.timestamp
                if 0 <= dt <= 180.0:
                    weight = math.exp(-dt / decay_tau)
                    # Cross-activity escalation: if they previously had another type of suspicion
                    activity_boost = 1.3 if obs.event_type != event_type else 1.0
                    recent_relevance += weight * 0.15 * activity_boost
            temporal_factor = 1.0 + min(1.2, recent_relevance)

        # Total combined risk score (0 to 100)
        raw_risk = base_score * activity_mult * temporal_factor * 100.0
        total_risk_score = round(max(0.0, min(100.0, raw_risk)), 1)
        risk_level = RiskLevel.from_score(total_risk_score)

        # Certainty: how confident we are this is a genuine cheating/suspicious anomaly
        # Increases with higher detection confidence and longer observation duration
        certainty = round(max(0.0, min(1.0, c_det * (0.6 + 0.4 * norm_duration))), 2)

        # Explanation summary
        summary = (
            f"Risk: {risk_level.value} ({total_risk_score}/100) | Certainty: {int(certainty*100)}% | "
            f"Factors: Conf={int(c_det*100)}%, Dur={duration_seconds:.1f}s, Freq={frequency_count}, "
            f"Severity={activity_mult:.2f}, TemporalBoost=x{temporal_factor:.2f}"
        )

        return RiskBreakdown(
            total_risk_score=total_risk_score,
            risk_level=risk_level,
            certainty=certainty,
            confidence_component=round(w_c * c_det * 100, 1),
            duration_component=round(w_d * norm_duration * 100, 1),
            frequency_component=round(w_f * norm_freq * 100, 1),
            activity_multiplier=activity_mult,
            temporal_escalation_factor=round(temporal_factor, 2),
            explanation_summary=summary
        )
