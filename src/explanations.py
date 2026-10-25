"""
Explainable Alert Formatter Module.
Implements Section 9:
Generates natural language, structured, human-interpretable alerts
adhering strictly to measured results.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any

from src.frame_sampler import FrameSampler
from src.risk_engine import RiskBreakdown

@dataclass
class ExplainableAlert:
    event_type: str
    event_display_name: str
    tracking_id: int
    seat_id: Optional[int]
    timestamp_sec: float
    timecode_str: str
    duration_seconds: float
    confidence_percent: int
    risk_score: float
    risk_category: str
    reason: str
    evidence_path: str
    metrics: Dict[str, Any]

    def to_formatted_text(self) -> str:
        """
        Formats alert exactly matching Section 9 specification:
        Event: Repeated side movement
        Seat: 18
        Timestamp: 00:12:36
        Duration: 14 seconds
        Detection confidence: 86%
        Risk score: 68/100
        Reason: Repeated side movement was observed during the configured analysis window. Human verification is recommended.
        Evidence: evidence/seat_18_001236.jpg
        """
        seat_str = str(self.seat_id) if self.seat_id is not None else "Unassigned"
        return (
            f"Event: {self.event_display_name}\n"
            f"Seat: {seat_str}\n"
            f"Timestamp: {self.timecode_str}\n"
            f"Duration: {int(round(self.duration_seconds))} seconds\n"
            f"Detection confidence: {self.confidence_percent}%\n"
            f"Risk score: {int(round(self.risk_score))}/100\n"
            f"Reason: {self.reason}\n"
            f"Evidence: {self.evidence_path}"
        )

class AlertExplainer:
    """
    Constructs explainable alerts from measured perception and reasoning data.
    """

    EVENT_NAME_MAP = {
        "REPEATED_HEAD_TURNING": "Repeated side movement",
        "LEAVING_SEAT": "Leaving assigned seat",
        "MOBILE_PHONE_DETECTED": "Mobile phone presence",
        "GATHERING_CLUSTER": "Multiple candidates gathering",
        "UNUSUAL_MOVEMENT": "Unusual erratic movement",
        "COMPOSITE_ANOMALY": "Co-occurring multiple indicators"
    }

    REASON_TEMPLATES = {
        "REPEATED_HEAD_TURNING": (
            "Repeated side movement was observed during the configured analysis window. "
            "Human verification is recommended."
        ),
        "LEAVING_SEAT": (
            "Candidate's estimated foot-point departed from the assigned seat region for a sustained duration. "
            "Human verification is recommended."
        ),
        "MOBILE_PHONE_DETECTED": (
            "A mobile phone device was detected within the candidate's workspace area across multiple frames. "
            "Immediate invigilator inspection is recommended."
        ),
        "GATHERING_CLUSTER": (
            "Multiple individuals were detected in close physical proximity outside standard desk separation. "
            "Invigilator review is recommended."
        ),
        "UNUSUAL_MOVEMENT": (
            "Rapid or sustained motion displacement exceeding typical seating baseline was detected. "
            "Human verification is recommended."
        ),
        "COMPOSITE_ANOMALY": (
            "Multiple distinct behavioral indicators were observed concurrently, escalating the review priority."
        )
    }

    @classmethod
    def generate_alert(
        cls,
        event_type: str,
        tracking_id: int,
        seat_id: Optional[int],
        timestamp_sec: float,
        duration_sec: float,
        detection_confidence: float,
        risk_breakdown: RiskBreakdown,
        evidence_path: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> ExplainableAlert:
        """
        Creates an ExplainableAlert object with verified real measurements.
        """
        timecode = FrameSampler.format_timestamp(timestamp_sec)
        display_name = cls.EVENT_NAME_MAP.get(event_type, event_type.replace("_", " ").title())
        reason = cls.REASON_TEMPLATES.get(event_type, "Unusual behavior pattern observed. Human review advised.")

        return ExplainableAlert(
            event_type=event_type,
            event_display_name=display_name,
            tracking_id=tracking_id,
            seat_id=seat_id,
            timestamp_sec=timestamp_sec,
            timecode_str=timecode,
            duration_seconds=duration_sec,
            confidence_percent=int(round(detection_confidence * 100)),
            risk_score=risk_breakdown.total_score,
            risk_category=risk_breakdown.risk_category,
            reason=reason,
            evidence_path=evidence_path,
            metrics=metrics or {}
        )
