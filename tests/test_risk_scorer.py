"""
Unit tests for Certainty & Risk Scoring and Temporal Reasoning.
Verifies Tasks 8 & 9.
"""
import unittest
from core.risk_scorer import RiskScorer
from core.state import EventType, RiskLevel, StudentMemory, Observation
from core.config import ScoringWeights

class TestRiskScorer(unittest.TestCase):

    def test_risk_scoring_factors(self):
        """Verify that certainty/risk score incorporates confidence, duration, frequency, and activity type."""
        weights = ScoringWeights(confidence_weight=0.30, duration_weight=0.40, frequency_weight=0.30)
        scorer = RiskScorer(weights)
        memory = StudentMemory(student_id=1, assigned_seat_id=1)

        # 1. Low risk scenario: short duration, low frequency
        breakdown_low = scorer.compute_risk(
            event_type=EventType.UNUSUAL_MOVEMENT,
            detection_confidence=0.50,
            duration_seconds=1.0,
            frequency_count=1,
            memory=memory,
            current_time=10.0
        )

        # 2. High risk scenario: phone with high confidence and sustained duration
        breakdown_high = scorer.compute_risk(
            event_type=EventType.MOBILE_PHONE_DETECTED,
            detection_confidence=0.92,
            duration_seconds=5.0,
            frequency_count=2,
            memory=memory,
            current_time=10.0
        )

        self.assertGreater(breakdown_high.total_risk_score, breakdown_low.total_risk_score)
        self.assertGreater(breakdown_high.certainty, breakdown_low.certainty)
        self.assertGreater(breakdown_high.activity_multiplier, breakdown_low.activity_multiplier)

    def test_temporal_reasoning_increases_risk_on_repeated_actions(self):
        """Verify Task 9: Repeated suspicious actions increase the risk level via temporal memory."""
        scorer = RiskScorer()

        # Clean student with no prior history
        memory_clean = StudentMemory(student_id=18, assigned_seat_id=18)
        risk_first = scorer.compute_risk(
            event_type=EventType.REPEATED_HEAD_TURNING,
            detection_confidence=0.80,
            duration_seconds=4.0,
            frequency_count=3,
            memory=memory_clean,
            current_time=10.0
        )

        # Student with recent past suspicious history in temporal memory
        memory_repeat = StudentMemory(student_id=18, assigned_seat_id=18)
        memory_repeat.history.append(Observation(timestamp=2.0, event_type=EventType.REPEATED_HEAD_TURNING, confidence=0.85))
        memory_repeat.history.append(Observation(timestamp=6.0, event_type=EventType.REPEATED_HEAD_TURNING, confidence=0.88))

        risk_repeated = scorer.compute_risk(
            event_type=EventType.REPEATED_HEAD_TURNING,
            detection_confidence=0.80,
            duration_seconds=4.0,
            frequency_count=3,
            memory=memory_repeat,
            current_time=10.0
        )

        # Temporal escalation must strictly increase the risk score
        self.assertGreater(risk_repeated.temporal_escalation_factor, 1.0)
        self.assertGreater(risk_repeated.total_risk_score, risk_first.total_risk_score)

if __name__ == "__main__":
    unittest.main()
