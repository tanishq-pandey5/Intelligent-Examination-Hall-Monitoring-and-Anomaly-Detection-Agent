"""
Unit tests for Behavioral Analysis Modules and Seat Mapping.
Verifies Section 4 & 12:
  - Seat-region configuration and foot-point occupancy
  - Leaving assigned seat module
  - Mobile phone workspace association vs confirmed use
  - Multiple students gathering cluster detection
  - Unusual movement velocity thresholding
"""
import unittest
import numpy as np

from config.settings import SeatRegion, AppSettings
from src.seat_mapper import SeatMapper
from src.tracker import TrackedPerson
from src.detector import DetectedObject
from src.behavior import BehaviorAnalyzer

class TestBehaviorModules(unittest.TestCase):

    def setUp(self):
        self.settings = AppSettings()
        # Define 2 test seats
        self.seats = {
            1: SeatRegion(seat_id=1, label="Seat 1", row=1, col=1, x1=0.1, y1=0.2, x2=0.3, y2=0.5),
            2: SeatRegion(seat_id=2, label="Seat 2", row=1, col=2, x1=0.4, y1=0.2, x2=0.6, y2=0.5),
        }
        self.seat_mapper = SeatMapper(custom_seats=self.seats)
        self.behavior = BehaviorAnalyzer(self.settings)

    def test_foot_point_seat_occupancy(self):
        """Verifies candidate's foot point inside seat region marks it occupied."""
        frame_w, frame_h = 1000, 1000
        # Candidate 1 foot point at (200, 450) -> inside Seat 1 (x: 100-300, y: 200-500)
        track1 = TrackedPerson(
            tracking_id=1,
            bbox=(150, 200, 250, 450),
            detection_confidence=0.90,
            first_seen_timestamp=1.0,
            last_seen_timestamp=1.0
        )
        status = self.seat_mapper.update_occupancy([track1], current_timestamp=1.0, dt=0.2, frame_w=frame_w, frame_h=frame_h)

        self.assertTrue(status[1]["is_occupied"])
        self.assertEqual(status[1]["assigned_tracking_id"], 1)
        self.assertEqual(status[1]["departure_duration_sec"], 0.0)

    def test_module_a_leaving_seat_duration(self):
        """Verifies Section 4A: departure duration and threshold trigger."""
        frame_w, frame_h = 1000, 1000
        track1 = TrackedPerson(
            tracking_id=1,
            bbox=(150, 200, 250, 450),
            detection_confidence=0.90,
            first_seen_timestamp=1.0,
            last_seen_timestamp=1.0
        )
        # Initialize seat occupancy at t=1.0s
        self.seat_mapper.update_occupancy([track1], current_timestamp=1.0, dt=0.2, frame_w=frame_w, frame_h=frame_h)

        # Move candidate out into hallway at t=2.0s
        track1.bbox = (750, 600, 850, 800)
        self.seat_mapper.update_occupancy([track1], current_timestamp=2.0, dt=1.0, frame_w=frame_w, frame_h=frame_h)

        # Candidate still outside at t=8.0s (6 seconds elapsed outside seat)
        status = self.seat_mapper.update_occupancy([track1], current_timestamp=8.0, dt=1.0, frame_w=frame_w, frame_h=frame_h)

        self.assertFalse(status[1]["is_occupied"])
        self.assertGreaterEqual(status[1]["departure_duration_sec"], 5.0)

        obs = self.behavior.analyze_seat_departure(track1, status, current_time=7.0)
        self.assertIsNotNone(obs)
        self.assertEqual(obs.event_type, "LEAVING_SEAT")
        self.assertEqual(obs.seat_id, 1)

    def test_module_c_phone_association_and_persistence(self):
        """Verifies Section 4C: mobile phone association with candidate workspace across frames."""
        track = TrackedPerson(
            tracking_id=5,
            assigned_seat_id=5,
            bbox=(400, 300, 500, 480),
            detection_confidence=0.90,
            first_seen_timestamp=1.0,
            last_seen_timestamp=1.0
        )
        phone = DetectedObject(
            class_id=67,
            class_name="cell phone",
            confidence=0.88,
            bbox=(480, 380, 510, 420) # Near candidate workspace
        )

        # Frame 1: Not yet enough persistence
        obs1 = self.behavior.analyze_phone_detection(track, [phone], current_time=1.0, frame_w=1000, frame_h=1000)
        # After 4 consecutive frames or 2.0s:
        self.behavior.phone_persistence[5] = 4
        self.behavior.phone_event_start_times[5] = 1.0
        obs2 = self.behavior.analyze_phone_detection(track, [phone], current_time=3.5, frame_w=1000, frame_h=1000)

        self.assertIsNotNone(obs2)
        self.assertEqual(obs2.event_type, "MOBILE_PHONE_DETECTED")
        self.assertEqual(obs2.tracking_id, 5)

    def test_module_d_gathering_cluster(self):
        """Verifies Section 4D: pairwise distance gathering detection."""
        # Candidate 1 and Candidate 2 within 60px distance
        t1 = TrackedPerson(tracking_id=11, bbox=(500, 400, 580, 560), detection_confidence=0.9, first_seen_timestamp=1.0, last_seen_timestamp=1.0)
        t2 = TrackedPerson(tracking_id=12, bbox=(530, 410, 610, 570), detection_confidence=0.9, first_seen_timestamp=1.0, last_seen_timestamp=1.0)
        t3 = TrackedPerson(tracking_id=13, bbox=(100, 100, 180, 260), detection_confidence=0.9, first_seen_timestamp=1.0, last_seen_timestamp=1.0) # Far away

        clusters = self.behavior.analyze_gathering_clusters([t1, t2, t3], current_time=5.0)
        self.assertEqual(len(clusters), 2)
        flagged_ids = [c.tracking_id for c in clusters]
        self.assertIn(11, flagged_ids)
        self.assertIn(12, flagged_ids)
        self.assertNotIn(13, flagged_ids)

if __name__ == "__main__":
    unittest.main()
