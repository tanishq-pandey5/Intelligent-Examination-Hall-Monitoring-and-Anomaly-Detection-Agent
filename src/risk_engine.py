"""
Risk Scoring Engine Module.
Implements Section 8:
  - Interpretable multi-factor risk score calculation (0 to 100)
  - Detailed component breakdown: detection confidence, duration, frequency,
    event type severity, temporal consistency, and tracking reliability.
  - Prioritizes human review; does NOT deliver automated cheating verdicts.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

from config.settings import AppSettings, DEFAULT_SETTINGS

@dataclass
class RiskBreakdown:
    total_score: float                # 0.0 to 100.0
    risk_category: str                # 'Low', 'Medium', 'High', 'Critical'
    confidence_component: float       # Contribution from detection confidence
    duration_component: float         # Contribution from duration
    frequency_component: float        # Contribution from recurrence
    event_type_component: float       # Contribution from activity severity
    temporal_consistency_component: float # Consistency boost across window
    tracking_reliability_component: float # Multiplier based on tracking quality
    explanation_of_components: str
    ml_anomaly_component: float = 0.0 # Contribution from trained behavioral ML model
    non_verdict_disclaimer: str = (
        "Notice: This risk score is an anomaly prioritization index designed solely "
        "to assist authorized human invigilators. It does NOT represent a definitive verdict of cheating."
    )

class RiskEngine:
    """
    Computes an interpretable 0-100 anomaly prioritization risk score
    with mathematical transparency across every constituent factor,
    including trained ML behavioral anomaly detection model outputs.
    """

    def __init__(self, settings: AppSettings = DEFAULT_SETTINGS):
        self.settings = settings

        # Duration and frequency scaling targets
        self.duration_targets: Dict[str, float] = {
            "MOBILE_PHONE_DETECTED": 4.0,
            "LEAVING_SEAT": 8.0,
            "GATHERING_CLUSTER": 6.0,
            "REPEATED_HEAD_TURNING": 12.0,
            "UNUSUAL_MOVEMENT": 5.0,
            "COMPOSITE_ANOMALY": 3.0
        }

        self.frequency_targets: Dict[str, int] = {
            "MOBILE_PHONE_DETECTED": 2,
            "LEAVING_SEAT": 2,
            "GATHERING_CLUSTER": 3,
            "REPEATED_HEAD_TURNING": 5,
            "UNUSUAL_MOVEMENT": 4,
            "COMPOSITE_ANOMALY": 2
        }

    @staticmethod
    def get_category(score: float) -> str:
        if score < 40.0:
            return "Low"
        elif score < 70.0:
            return "Medium"
        elif score < 90.0:
            return "High"
        return "Critical"

    def compute_risk_score(
        self,
        event_type: str,
        detection_confidence: float,
        duration_sec: float,
        frequency_count: int,
        tracking_reliability: float = 0.95,
        past_observations_count: int = 0,
        ml_anomaly_score: Optional[float] = None
    ) -> RiskBreakdown:
        """
        Calculates interpretable risk score from 0 to 100.
        Optional ml_anomaly_score [0.0, 1.0] from trained behavioral model.
        """
        # 1. Detection Confidence Component [0, 1]
        c_det = max(0.0, min(1.0, float(detection_confidence)))

        # 2. Duration Component [0, 1]
        target_d = self.duration_targets.get(event_type, 6.0)
        norm_dur = max(0.0, min(1.0, duration_sec / target_d))

        # 3. Frequency Component [0, 1]
        target_f = self.frequency_targets.get(event_type, 4)
        norm_freq = max(0.0, min(1.0, frequency_count / target_f))

        # 4. Event Type Severity Multiplier
        severity = self.settings.activity_severity.get(event_type, 0.70)

        # 5. Temporal Consistency Component (recurrence over time)
        temporal_consistency = min(1.0, 0.4 + 0.15 * past_observations_count)

        # 6. Tracking Reliability Multiplier
        t_rel = max(0.3, min(1.0, float(tracking_reliability)))

        # Weighted combination
        w_c = self.settings.weight_detection_confidence
        w_d = self.settings.weight_event_duration
        w_f = self.settings.weight_frequency
        w_t = self.settings.weight_event_type
        w_r = self.settings.weight_tracking_reliability

        weighted_sum = (
            w_c * c_det +
            w_d * norm_dur +
            w_f * norm_freq +
            w_t * severity +
            w_r * t_rel
        )

        # Apply temporal consistency scaling
        scaled_score = weighted_sum * (0.85 + 0.15 * temporal_consistency) * 100.0

        # Incorporate trained ML Anomaly Detector score if provided
        ml_component_val = 0.0
        if ml_anomaly_score is not None:
            ml_anomaly_clamped = max(0.0, min(1.0, float(ml_anomaly_score)))
            ml_weight = 0.20
            ml_component_val = round(ml_anomaly_clamped * 100.0, 1)
            scaled_score = (1.0 - ml_weight) * scaled_score + ml_weight * (ml_anomaly_clamped * 100.0)

        final_score = round(max(0.0, min(100.0, scaled_score)), 1)
        category = self.get_category(final_score)

        explanation = (
            f"Risk Score: {final_score}/100 ({category}) | "
            f"Factors: Confidence ({int(c_det*100)}%, w={w_c}), "
            f"Duration ({duration_sec:.1f}s, w={w_d}), "
            f"Frequency ({frequency_count}x, w={w_f}), "
            f"Activity Severity ({severity:.2f}, w={w_t}), "
            f"Tracking Reliability ({int(t_rel*100)}%, w={w_r})"
        )
        if ml_anomaly_score is not None:
            explanation += f", ML Anomaly Model ({int(ml_anomaly_score*100)}%, w=0.20)"

        return RiskBreakdown(
            total_score=final_score,
            risk_category=category,
            confidence_component=round(w_c * c_det * 100, 1),
            duration_component=round(w_d * norm_dur * 100, 1),
            frequency_component=round(w_f * norm_freq * 100, 1),
            event_type_component=round(w_t * severity * 100, 1),
            temporal_consistency_component=round(temporal_consistency, 2),
            tracking_reliability_component=round(t_rel, 2),
            ml_anomaly_component=ml_component_val,
            explanation_of_components=explanation
        )

