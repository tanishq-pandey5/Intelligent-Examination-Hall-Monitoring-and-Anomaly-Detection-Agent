"""
Unit tests for Rule-Based Reasoning Engine, Risk Scoring, and Explainable Alerts.
Verifies Sections 7, 8, 9 & 12:
  - Production rule matching
  - Risk score calculation (0 to 100) and factor contributions
  - Non-verdict disclaimer
  - Section 9 formatted explainable output
"""
import unittest

from config.settings import AppSettings
from src.agent import PersonAgentMemory
from src.reasoning import RuleBasedReasoningEngine
from src.risk_engine import RiskEngine
from src.explanations import AlertExplainer

class TestReasoningAndRisk(unittest.TestCase):

    def setUp(self):
        self.settings = AppSettings()
        self.reasoning = RuleBasedReasoningEngine(self.settings)
        self.risk_engine = RiskEngine(self.settings)

    def test_seat_departure_rule_match(self):
        """Verifies Section 7 Rule 1: Seat departure rule."""
        mem = PersonAgentMemory(tracking_id=9, assigned_seat_id=9)
        active_obs = {
            "seat_departure": {"duration_sec": 7.0, "distance_px": 120.0}
        }

        matches = self.reasoning.evaluate_all_rules(mem, current_time=10.0, active_observations=active_obs)
        self.assertEqual(len(matches), 1)
        m = matches[0]
        self.assertEqual(m.rule_id, "RULE_01_SEAT_DEPARTURE")
        self.assertEqual(m.event_type, "LEAVING_SEAT")
        self.assertIn("assigned seat region", m.reason_explanation.lower())

    def test_multi_indicator_escalation_rule(self):
        """Verifies Section 7 Rule 6: Composite multi-indicator escalation."""
        mem = PersonAgentMemory(tracking_id=12, assigned_seat_id=12)
        active_obs = {
            "seat_departure": {"duration_sec": 6.0, "distance_px": 90.0},
            "gathering": {"group_size": 2, "duration_sec": 4.0}
        }

        matches = self.reasoning.evaluate_all_rules(mem, current_time=10.0, active_observations=active_obs)
        # Should contain Rule 1, Rule 4, and Rule 6 (composite escalation)
        rule_ids = [m.rule_id for m in matches]
        self.assertIn("RULE_01_SEAT_DEPARTURE", rule_ids)
        self.assertIn("RULE_04_GATHERING", rule_ids)
        self.assertIn("RULE_06_COMPOSITE_ESCALATION", rule_ids)

    def test_risk_scoring_components_and_disclaimer(self):
        """Verifies Section 8: Interpretable risk score and non-verdict disclaimer."""
        breakdown = self.risk_engine.compute_risk_score(
            event_type="MOBILE_PHONE_DETECTED",
            detection_confidence=0.88,
            duration_sec=4.0,
            frequency_count=2,
            tracking_reliability=0.95
        )

        self.assertGreaterEqual(breakdown.total_score, 60.0)
        self.assertLessEqual(breakdown.total_score, 100.0)
        self.assertIn(breakdown.risk_category, ["Medium", "High", "Critical"])
        self.assertIn("Confidence", breakdown.explanation_of_components)
        self.assertIn("Duration", breakdown.explanation_of_components)
        self.assertIn("does NOT represent a definitive verdict of cheating", breakdown.non_verdict_disclaimer)

    def test_explainable_alert_formatting_matches_spec(self):
        """Verifies Section 9: Exact output format match."""
        breakdown = self.risk_engine.compute_risk_score(
            event_type="REPEATED_HEAD_TURNING",
            detection_confidence=0.86,
            duration_sec=14.0,
            frequency_count=4
        )

        alert = AlertExplainer.generate_alert(
            event_type="REPEATED_HEAD_TURNING",
            tracking_id=18,
            seat_id=18,
            timestamp_sec=756.0, # 00:12:36
            duration_sec=14.0,
            detection_confidence=0.86,
            risk_breakdown=breakdown,
            evidence_path="evidence/seat_18_001236.jpg"
        )

        formatted = alert.to_formatted_text()
        self.assertIn("Event: Repeated side movement", formatted)
        self.assertIn("Seat: 18", formatted)
        self.assertIn("Timestamp: 12:36", formatted)
        self.assertIn("Duration: 14 seconds", formatted)
        self.assertIn("Detection confidence: 86%", formatted)
        self.assertIn("Reason: Repeated side movement was observed", formatted)
        self.assertIn("Evidence: evidence/seat_18_001236.jpg", formatted)

if __name__ == "__main__":
    unittest.main()
