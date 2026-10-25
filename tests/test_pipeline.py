"""
Unit tests for Pipeline, Simulator, and SQLite Event Database.
Verifies Tasks 1, 2, 3, 4, 11.
"""
import unittest
import os
import tempfile
import cv2
from core.simulator import ExamHallSimulator
from core.pipeline import ExamMonitoringPipeline
from core.config import SystemConfig
from core.database import ExamEventDatabase

class TestPipelineAndDatabase(unittest.TestCase):

    def test_simulator_and_pipeline_integration(self):
        """Verify simulator generates frames, pipeline processes them, and alerts are stored with evidence."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_db_path = os.path.join(tmp_dir, "test_exam.db")
            test_evidence_dir = os.path.join(tmp_dir, "evidence")

            config = SystemConfig(
                database_path=test_db_path,
                evidence_dir=test_evidence_dir,
                default_fps=10.0
            )
            db = ExamEventDatabase(test_db_path)
            pipeline = ExamMonitoringPipeline(config=config, db=db)
            simulator = ExamHallSimulator(config=config)

            alert_count = 0
            # Run 9 seconds of simulation at 10 fps (90 frames)
            # Seat 18 turns starting at t=3s (crosses 4.0s duration at t=7s)
            # Seat 5 phone starts at t=5s (crosses 2.0s duration at t=7s)
            for frame_i in range(90):
                t_sec = frame_i * 0.1
                frame, hints = simulator.render_frame(t_sec)
                annotated, decisions, emitted = pipeline.process_frame(frame, t_sec, synthetic_hints=hints)
                alert_count += len(emitted)

                # Check visual overlay produced
                self.assertEqual(annotated.shape, frame.shape)

            # Check database records
            events = db.get_recent_events(limit=20)
            self.assertGreater(len(events), 0)

            ev = events[0]
            # Verify all 6 mandatory fields from Task 11 are present and non-empty:
            self.assertIn("timestamp", ev)
            self.assertTrue(ev["timestamp"])
            self.assertIn("seat_id", ev)
            self.assertGreater(ev["seat_id"], 0)
            self.assertIn("student_id", ev)
            self.assertIn("event_type", ev)
            self.assertTrue(ev["event_type"])
            self.assertIn("confidence", ev)
            self.assertGreater(ev["confidence"], 0)
            self.assertIn("risk_level", ev)
            self.assertIn(ev["risk_level"], ["Low", "Medium", "High", "Critical"])
            self.assertIn("evidence_frame_path", ev)
            self.assertTrue(os.path.exists(ev["evidence_frame_path"]))

if __name__ == "__main__":
    unittest.main()
