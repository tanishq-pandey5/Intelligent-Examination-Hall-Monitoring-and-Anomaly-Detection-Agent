"""
Unit tests for Object Detector and Multi-Person Tracker.
Verifies Sections 3 & 12:
  - Person detection and temporary tracking ID assignment
  - Bounding box and estimated foot-point coordinates
  - Position and movement history
  - First-seen and last-seen timestamps
  - Temporary detection loss handling
"""
import unittest
import numpy as np

from src.detector import ExamHallDetector, DetectedObject
from src.tracker import PersonTracker, TrackedPerson

class TestDetectorAndTracker(unittest.TestCase):

    def test_detected_object_properties(self):
        """Verifies centroid and foot-point properties on detected objects."""
        det = DetectedObject(
            class_id=0,
            class_name="person",
            confidence=0.88,
            bbox=(100, 50, 200, 250)
        )
        self.assertEqual(det.centroid, (150.0, 150.0))
        # Foot point is bottom-center: ((100+200)/2, 250) = (150.0, 250.0)
        self.assertEqual(det.foot_point, (150.0, 250.0))

    def test_tracker_initial_id_assignment(self):
        """Verifies initial track creation with temporary tracking ID."""
        tracker = PersonTracker()
        dets = [
            DetectedObject(class_id=0, class_name="person", confidence=0.90, bbox=(100, 100, 180, 260)),
            DetectedObject(class_id=0, class_name="person", confidence=0.85, bbox=(300, 100, 380, 260))
        ]

        tracks = tracker.update(dets, current_timestamp=1.0)
        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracks[0].tracking_id, 1)
        self.assertEqual(tracks[1].tracking_id, 2)
        self.assertEqual(tracks[0].first_seen_timestamp, 1.0)
        self.assertEqual(tracks[0].last_seen_timestamp, 1.0)

    def test_tracker_trajectory_and_velocity(self):
        """Verifies tracking across frames updates trajectory and calculates velocity."""
        tracker = PersonTracker()

        # Frame 1 at t=1.0s, centroid=(140, 180)
        dets_f1 = [DetectedObject(class_id=0, class_name="person", confidence=0.90, bbox=(100, 100, 180, 260))]
        tracker.update(dets_f1, current_timestamp=1.0)

        # Frame 2 at t=2.0s, moved 50px right: centroid=(190, 180)
        dets_f2 = [DetectedObject(class_id=0, class_name="person", confidence=0.90, bbox=(150, 100, 230, 260))]
        tracks = tracker.update(dets_f2, current_timestamp=2.0)

        self.assertEqual(len(tracks), 1)
        t = tracks[0]
        self.assertEqual(t.tracking_id, 1)
        self.assertEqual(t.last_seen_timestamp, 2.0)
        self.assertGreater(t.velocity_px_per_sec, 25.0) # Moved 50px in 1s
        self.assertEqual(len(t.trajectory), 2)

    def test_temporary_detection_loss_handling(self):
        """Verifies tracks are retained during brief occlusions and deleted after max_disappeared_frames."""
        tracker = PersonTracker(max_disappeared_frames=3)
        dets = [DetectedObject(class_id=0, class_name="person", confidence=0.90, bbox=(100, 100, 180, 260))]
        tracker.update(dets, current_timestamp=1.0)

        # Disappear for 2 frames (within tolerance)
        tracker.update([], current_timestamp=1.5)
        tracker.update([], current_timestamp=2.0)
        self.assertEqual(len(tracker.active_tracks), 1)
        self.assertEqual(tracker.active_tracks[1].frames_since_update, 2)

        # Disappear for 2 more frames (exceeds max_disappeared_frames of 3)
        tracker.update([], current_timestamp=2.5)
        tracker.update([], current_timestamp=3.0)
        self.assertEqual(len(tracker.active_tracks), 0)

if __name__ == "__main__":
    unittest.main()
