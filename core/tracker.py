"""
Multi-Object Student Tracker with Seat Association and Velocity Estimation.
Implements Task 3.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import time

from core.config import SeatConfig

@dataclass
class TrackedStudent:
    track_id: int
    assigned_seat_id: int
    bbox: Tuple[int, int, int, int]           # (x1, y1, x2, y2)
    centroid: Tuple[float, float]             # (cx, cy)
    confidence: float
    history: List[Tuple[float, Tuple[float, float]]] = field(default_factory=list) # (timestamp, centroid)
    velocity_px_per_sec: float = 0.0
    frames_since_update: int = 0
    total_frames_tracked: int = 0

class StudentTracker:
    """
    Tracks students across consecutive video frames using Centroid + IoU distance matching,
    smooths trajectories, computes instantaneous velocity, and associates each track with
    the nearest designated examination seat.
    """

    def __init__(
        self,
        seats: Dict[int, SeatConfig],
        max_disappeared_frames: int = 15,
        max_distance_pixels: float = 120.0,
        frame_w: int = 1280,
        frame_h: int = 720
    ):
        self.seats = seats
        self.max_disappeared_frames = max_disappeared_frames
        self.max_distance_pixels = max_distance_pixels
        self.frame_w = frame_w
        self.frame_h = frame_h

        # Active tracks: track_id -> TrackedStudent
        self.active_tracks: Dict[int, TrackedStudent] = {}
        self.next_track_id: int = 1

        # Cache seat pixel centers
        self.seat_pixel_centers: Dict[int, Tuple[float, float]] = {}
        for s_id, s in seats.items():
            self.seat_pixel_centers[s_id] = (s.center_x * frame_w, s.center_y * frame_h)

    def _iou(self, boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _associate_to_seat(self, centroid: Tuple[float, float]) -> int:
        """Finds closest designated seat ID to the given centroid."""
        best_seat_id = 1
        min_dist = float("inf")
        cx, cy = centroid
        for s_id, (sx, sy) in self.seat_pixel_centers.items():
            dist = np.hypot(cx - sx, cy - sy)
            if dist < min_dist:
                min_dist = dist
                best_seat_id = s_id
        return best_seat_id

    def update(
        self,
        detections: List[Tuple[Tuple[int, int, int, int], float]],
        timestamp: float
    ) -> List[TrackedStudent]:
        """
        Updates tracks with new detections [(bbox, confidence)].
        Matches existing tracks with greedy centroid-distance and IoU minimization.
        """
        if len(detections) == 0:
            # Mark all as missed
            to_delete = []
            for t_id, track in self.active_tracks.items():
                track.frames_since_update += 1
                if track.frames_since_update > self.max_disappeared_frames:
                    to_delete.append(t_id)
            for t_id in to_delete:
                del self.active_tracks[t_id]
            return list(self.active_tracks.values())

        det_bboxes = [d[0] for d in detections]
        det_confs = [d[1] for d in detections]
        det_centroids = [
            ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
            for b in det_bboxes
        ]

        if len(self.active_tracks) == 0:
            # Initialize all as new tracks
            for bbox, conf, centroid in zip(det_bboxes, det_confs, det_centroids):
                seat_id = self._associate_to_seat(centroid)
                track = TrackedStudent(
                    track_id=self.next_track_id,
                    assigned_seat_id=seat_id,
                    bbox=bbox,
                    centroid=centroid,
                    confidence=conf,
                    history=[(timestamp, centroid)],
                    velocity_px_per_sec=0.0,
                    frames_since_update=0,
                    total_frames_tracked=1
                )
                self.active_tracks[self.next_track_id] = track
                self.next_track_id += 1
            return list(self.active_tracks.values())

        # Match existing tracks to detections using distance matrix
        track_ids = list(self.active_tracks.keys())
        track_centroids = [self.active_tracks[tid].centroid for tid in track_ids]

        # Cost matrix: Euclidean distance between track centroids and det centroids
        cost_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
        for i, (tcx, tcy) in enumerate(track_centroids):
            for j, (dcx, dcy) in enumerate(det_centroids):
                cost_matrix[i, j] = np.hypot(tcx - dcx, tcy - dcy)

        # Greedy matching
        matched_tracks = set()
        matched_dets = set()

        if cost_matrix.size > 0:
            # Sort pairs by distance
            sorted_indices = np.unravel_index(np.argsort(cost_matrix, axis=None), cost_matrix.shape)
            for r, c in zip(sorted_indices[0], sorted_indices[1]):
                if r in matched_tracks or c in matched_dets:
                    continue
                if cost_matrix[r, c] > self.max_distance_pixels:
                    # Too far to be the same student
                    continue

                matched_tracks.add(r)
                matched_dets.add(c)

                tid = track_ids[r]
                track = self.active_tracks[tid]
                new_bbox = det_bboxes[c]
                new_centroid = det_centroids[c]
                new_conf = det_confs[c]

                # Compute instantaneous velocity
                velocity = 0.0
                if track.history:
                    last_time, last_c = track.history[-1]
                    dt = max(1e-4, timestamp - last_time)
                    dist = np.hypot(new_centroid[0] - last_c[0], new_centroid[1] - last_c[1])
                    instant_v = dist / dt
                    # Exponential moving average filter for velocity
                    velocity = 0.7 * instant_v + 0.3 * track.velocity_px_per_sec

                track.bbox = new_bbox
                track.centroid = new_centroid
                track.confidence = new_conf
                track.velocity_px_per_sec = velocity
                track.frames_since_update = 0
                track.total_frames_tracked += 1
                track.history.append((timestamp, new_centroid))
                if len(track.history) > 60:
                    track.history.pop(0)

        # Unmatched tracks
        to_delete = []
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.active_tracks[tid].frames_since_update += 1
                if self.active_tracks[tid].frames_since_update > self.max_disappeared_frames:
                    to_delete.append(tid)

        for tid in to_delete:
            del self.active_tracks[tid]

        # Unmatched detections -> create new tracks
        for j in range(len(detections)):
            if j not in matched_dets:
                bbox = det_bboxes[j]
                conf = det_confs[j]
                centroid = det_centroids[j]
                seat_id = self._associate_to_seat(centroid)

                track = TrackedStudent(
                    track_id=self.next_track_id,
                    assigned_seat_id=seat_id,
                    bbox=bbox,
                    centroid=centroid,
                    confidence=conf,
                    history=[(timestamp, centroid)],
                    velocity_px_per_sec=0.0,
                    frames_since_update=0,
                    total_frames_tracked=1
                )
                self.active_tracks[self.next_track_id] = track
                self.next_track_id += 1

        return list(self.active_tracks.values())
