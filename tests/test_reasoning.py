"""
Unit tests for Rule-Based Reasoning Engine and Explainable Alert Generator.
Verifies Tasks 7 & 10.
"""
import unittest
from core.reasoning import RuleBasedReasoningEngine
from core.config import AnomalyThresholds
from core.state import StudentMemory, StudentState, RiskLevel, EventType

class TestReasoningEngine(unittest.TestCase):

    def test_head_turn_rule_evaluation(self):
        """Verify Rule 1: IF head_turn_count > threshold AND duration > T THEN Suspicion_Level = Medium/High."""
        thresholds = AnomalyThresholds(
            head_turn_count_threshold=3,
            head_turn_duration_threshold=4.0,
            head_turn_window_seconds=30.0
        )
        engine = RuleBasedReasoningEngine(thresholds)

        memory = StudentMemory(student_id=18, assigned_seat_id=18)

        # Below threshold: 2 turns, 2.0s duration -> No match
        memory.record_turn(timestamp=1.0, angle_deg=-35.0, direction="LEFT")
        memory.record_turn(timestamp=3.0, angle_deg=-35.0, direction="LEFT")
        result = engine.evaluate_head_turning_rule(memory, current_time=4.0, turn_duration=2.0)
        self.assertIsNone(result)

        # Above threshold: 4 turns, 6.0s duration -> Matched!
        memory.record_turn(timestamp=5.0, angle_deg=-40.0, direction="LEFT")
        memory.record_turn(timestamp=7.0, angle_deg=-45.0, direction="LEFT")
        result = engine.evaluate_head_turning_rule(memory, current_time=8.0, turn_duration=6.0)

        self.assertIsNotNone(result)
        self.assertTrue(result.matched)
        self.assertIn(result.recommended_risk_level, [RiskLevel.MEDIUM, RiskLevel.HIGH])
        # Check explainable alert output
        self.assertIn("Student/Seat 18", result.explainable_text)
        self.assertIn("Repeated side movement detected", result.explainable_text)

    def test_mobile_phone_rule_evaluation(self):
        """Verify Rule 3: Mobile phone detection."""
        engine = RuleBasedReasoningEngine()
        memory = StudentMemory(student_id=5, assigned_seat_id=5)

        # Under duration -> None
        r_short = engine.evaluate_mobile_phone_rule(memory, phone_confidence=0.85, phone_duration=0.5)
        self.assertIsNone(r_short)

        # Sufficient duration -> ALERT
        r_alert = engine.evaluate_mobile_phone_rule(memory, phone_confidence=0.88, phone_duration=2.5)
        self.assertIsNotNone(r_alert)
        self.assertEqual(r_alert.target_state, StudentState.ALERT)
        self.assertIn("Mobile phone detected", r_alert.explainable_text)

    def test_leaving_seat_rule_evaluation(self):
        """Verify Rule 2: Leaving seat."""
        engine = RuleBasedReasoningEngine()
        memory = StudentMemory(student_id=9, assigned_seat_id=9)

        result = engine.evaluate_leaving_seat_rule(memory, distance_from_seat=120.0, leaving_duration=6.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.event_type, EventType.LEAVING_SEAT)
        self.assertIn("Vacated assigned seat", result.explainable_text)

if __name__ == "__main__":
    unittest.main()
