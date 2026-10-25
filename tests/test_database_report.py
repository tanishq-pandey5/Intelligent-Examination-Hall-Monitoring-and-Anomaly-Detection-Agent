"""
Unit tests for Database Manager, Evidence Manager, and Report Generator.
Verifies Sections 9, 11 & 12:
  - SQLite database session and event storage
  - Event status updating (ACKNOWLEDGED, RESOLVED, FALSE_POSITIVE)
  - CSV report export
  - Human-readable session summary generation
  - Evidence snapshot capturing and deduplication
"""
import unittest
import os
import tempfile
import numpy as np

from config.settings import AppSettings
from src.database import DatabaseManager
from src.evidence import EvidenceManager
from src.report_generator import ReportGenerator

class TestDatabaseAndReport(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_exam.db")
        self.evidence_dir = os.path.join(self.tmp_dir.name, "evidence")
        self.reports_dir = os.path.join(self.tmp_dir.name, "reports")

        self.settings = AppSettings(
            database_path=self.db_path,
            evidence_dir=self.evidence_dir,
            reports_dir=self.reports_dir
        )
        self.db = DatabaseManager(self.db_path)
        self.evidence_mgr = EvidenceManager(self.settings)
        self.reporter = ReportGenerator(self.db, self.settings)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_database_session_and_event_crud(self):
        """Verifies session and event insertion, query filtering, and status updates."""
        # Insert session
        sess_id = "sess_test_101"
        self.db.insert_session(
            session_id=sess_id,
            file_name="cctv_hall.mp4",
            file_path="/videos/cctv_hall.mp4",
            fps=25.0,
            resolution="1280:720",
            duration_seconds=60.0,
            total_frames=1500,
            analyzed_frames=300,
            sampling_fps=5.0
        )

        # Insert event
        ev_id = self.db.insert_event(
            session_id=sess_id,
            video_timestamp_sec=14.2,
            timecode_str="00:14",
            tracking_id=18,
            seat_id=18,
            event_type="REPEATED_HEAD_TURNING",
            event_display_name="Repeated side movement",
            confidence=0.86,
            duration_seconds=14.0,
            risk_score=68.0,
            risk_category="Medium",
            reason="Repeated side movement was observed.",
            evidence_path="data/evidence/seat_18_0014.jpg",
            review_status="UNREVIEWED"
        )
        self.assertGreater(ev_id, 0)

        # Query events
        events = self.db.get_events(session_id=sess_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["seat_id"], 18)
        self.assertEqual(events[0]["review_status"], "UNREVIEWED")

        # Update status
        self.db.update_event_status(ev_id, "RESOLVED", "Verified with proctor")
        updated_events = self.db.get_events(session_id=sess_id)
        self.assertEqual(updated_events[0]["review_status"], "RESOLVED")
        self.assertEqual(updated_events[0]["invigilator_notes"], "Verified with proctor")

    def test_evidence_capture_and_deduplication(self):
        """Verifies evidence image creation and deduplication logic."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # First capture should succeed
        self.assertTrue(self.evidence_mgr.should_save_evidence(18, "REPEATED_HEAD_TURNING", 10.0))
        rel_path = self.evidence_mgr.capture_evidence_frame(
            frame=frame,
            tracking_id=18,
            seat_id=18,
            event_type="REPEATED_HEAD_TURNING",
            timestamp_sec=10.0,
            timecode_str="00:10",
            risk_score=75.0,
            target_bbox=(100, 100, 200, 250),
            reason_text="Head turned left"
        )
        abs_path = os.path.join(self.settings.base_dir, rel_path)
        self.assertTrue(os.path.exists(abs_path) or os.path.exists(os.path.join(self.tmp_dir.name, "evidence", os.path.basename(rel_path))))

        # Immediate repeat should be suppressed by deduplication
        self.assertFalse(self.evidence_mgr.should_save_evidence(18, "REPEATED_HEAD_TURNING", 10.5))

    def test_report_export(self):
        """Verifies CSV export and Markdown session summary generation."""
        # Insert event
        self.db.insert_event(
            session_id="sess_report_test",
            video_timestamp_sec=8.5,
            timecode_str="00:08",
            tracking_id=5,
            seat_id=5,
            event_type="MOBILE_PHONE_DETECTED",
            event_display_name="Mobile phone presence",
            confidence=0.92,
            duration_seconds=4.0,
            risk_score=92.0,
            risk_category="Critical",
            reason="Phone detected on desk.",
            evidence_path="data/evidence/seat_5.jpg"
        )

        csv_path = self.reporter.export_csv_report("sess_report_test")
        self.assertTrue(os.path.exists(csv_path))

        summary = self.reporter.generate_human_readable_summary("sess_report_test")
        self.assertIn("Examination Hall AI Monitoring", summary)
        self.assertIn("Mobile phone presence", summary)
        self.assertIn("Critical", summary)

if __name__ == "__main__":
    unittest.main()
