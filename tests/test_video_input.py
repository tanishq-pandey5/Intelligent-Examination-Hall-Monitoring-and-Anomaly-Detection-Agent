"""
Unit tests for Video Input Handler and Frame Sampler Modules.
Verifies Section 2 & 12:
  - Video format verification (.mp4, .avi, .mov)
  - Video metadata extraction (FPS, duration, resolution)
  - Frame sampling at configurable intervals
  - Video timeline timestamping (HH:MM:SS)
"""
import unittest
import os
import cv2
import numpy as np
import tempfile

from src.video_input import VideoInputHandler, VideoMetadata
from src.frame_sampler import FrameSampler, VideoFrame

class TestVideoInputAndSampler(unittest.TestCase):

    def setUp(self):
        # Create a small temporary test video (2 seconds @ 20 FPS = 40 frames)
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.test_video_path = os.path.join(self.tmp_dir.name, "test_exam_hall.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.test_video_path, fourcc, 20.0, (640, 480))
        for i in range(40):
            frame = np.full((480, 640, 3), (i * 5) % 255, dtype=np.uint8)
            cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            out.write(frame)
        out.release()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_inspect_valid_video(self):
        """Verifies metadata extraction from a valid video file."""
        meta = VideoInputHandler.inspect_video(self.test_video_path)
        self.assertTrue(meta.is_valid)
        self.assertEqual(meta.width, 640)
        self.assertEqual(meta.height, 480)
        self.assertAlmostEqual(meta.fps, 20.0, delta=0.5)
        self.assertEqual(meta.total_frames, 40)
        self.assertAlmostEqual(meta.duration_seconds, 2.0, delta=0.1)

    def test_inspect_unsupported_format(self):
        """Verifies rejection of unsupported video extensions."""
        bad_path = os.path.join(self.tmp_dir.name, "exam.txt")
        with open(bad_path, "w") as f:
            f.write("not a video")
        meta = VideoInputHandler.inspect_video(bad_path)
        self.assertFalse(meta.is_valid)
        self.assertIn("Unsupported video format", meta.error_message)

    def test_inspect_nonexistent_file(self):
        """Verifies error handling on nonexistent file."""
        meta = VideoInputHandler.inspect_video("non_existent_file.mp4")
        self.assertFalse(meta.is_valid)
        self.assertIn("File does not exist", meta.error_message)

    def test_frame_sampler_sampling_interval(self):
        """Verifies sampling at a target FPS (e.g. 5 FPS from 20 FPS video = 10 frames sampled)."""
        sampler = FrameSampler(target_sampling_fps=5.0)
        frames = list(sampler.sample_video(self.test_video_path))

        # 2 seconds @ 5 FPS = ~10 frames
        self.assertGreaterEqual(len(frames), 8)
        self.assertLessEqual(len(frames), 12)

        # Check timestamp consistency
        first = frames[0]
        self.assertEqual(first.frame_index, 0)
        self.assertEqual(first.video_timestamp_sec, 0.0)
        self.assertEqual(first.timecode_str, "00:00")

    def test_timecode_formatting(self):
        """Verifies timeline timecode formatting (HH:MM:SS or MM:SS)."""
        self.assertEqual(FrameSampler.format_timestamp(0.0), "00:00")
        self.assertEqual(FrameSampler.format_timestamp(756.0), "12:36")
        self.assertEqual(FrameSampler.format_timestamp(3674.0), "01:01:14")

if __name__ == "__main__":
    unittest.main()
