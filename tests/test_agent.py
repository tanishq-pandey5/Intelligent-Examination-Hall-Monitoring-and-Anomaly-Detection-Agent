"""
Unit tests for Model-Based AI Agent Module.
Verifies Section 5, 6 & 12:
  - Internal state machine: NORMAL -> OBSERVE -> SUSPICIOUS -> ALERT
  - Single frame damping (single frame never triggers high-priority alert)
  - Risk decay when behavior ceases
  - Human review state tracking
"""
import unittest

from src.agent import PersonAgentMemory, AgentState, ReviewState

class TestModelBasedAgent(unittest.TestCase):

    def test_single_frame_does_not_trigger_alert(self):
        """Verifies Section 5 & 6: a single isolated frame cannot trigger an ALERT."""
        memory = PersonAgentMemory(
            tracking_id=1,
            assigned_seat_id=18,
            current_state=AgentState.NORMAL,
            current_risk_score=0.0
        )

        # Single glance or tick: duration is 0.1s
        dur = memory.update_event_episode("REPEATED_HEAD_TURNING", is_active=True, current_time=1.0)
        self.assertEqual(dur, 0.0)
        self.assertEqual(memory.current_state, AgentState.NORMAL)

    def test_state_transitions_with_increasing_risk(self):
        """Verifies state machine transitions: NORMAL -> OBSERVE -> SUSPICIOUS -> ALERT."""
        memory = PersonAgentMemory(tracking_id=1, assigned_seat_id=18)
        self.assertEqual(memory.current_state, AgentState.NORMAL)

        # Score 30 -> OBSERVE
        memory.current_risk_score = 30.0
        memory._update_state_from_score(1.0, "Score reached observe")
        self.assertEqual(memory.current_state, AgentState.OBSERVE)

        # Score 65 -> SUSPICIOUS
        memory.current_risk_score = 65.0
        memory._update_state_from_score(2.0, "Score reached suspicious")
        self.assertEqual(memory.current_state, AgentState.SUSPICIOUS)

        # Score 85 -> ALERT
        memory.current_risk_score = 85.0
        memory._update_state_from_score(3.0, "Score reached alert")
        self.assertEqual(memory.current_state, AgentState.ALERT)

        self.assertEqual(len(memory.state_transitions), 3)

    def test_risk_decay_during_calm_behavior(self):
        """Verifies risk score decays when behavior ceases and returns state to NORMAL."""
        memory = PersonAgentMemory(
            tracking_id=5,
            assigned_seat_id=5,
            current_state=AgentState.SUSPICIOUS,
            current_risk_score=60.0
        )

        # Run 20 seconds of decay @ 2.5 pts/sec = 50 points decayed
        memory.apply_risk_decay(dt=20.0, decay_rate=2.5)

        self.assertEqual(memory.current_risk_score, 10.0)
        self.assertEqual(memory.current_state, AgentState.NORMAL)

    def test_human_review_states(self):
        """Verifies human review lifecycle states."""
        memory = PersonAgentMemory(tracking_id=9, assigned_seat_id=9)
        self.assertEqual(memory.review_state, ReviewState.UNREVIEWED)

        memory.review_state = ReviewState.UNDER_REVIEW
        self.assertEqual(memory.review_state, ReviewState.UNDER_REVIEW)

        memory.review_state = ReviewState.RESOLVED
        self.assertEqual(memory.review_state, ReviewState.RESOLVED)

if __name__ == "__main__":
    unittest.main()
