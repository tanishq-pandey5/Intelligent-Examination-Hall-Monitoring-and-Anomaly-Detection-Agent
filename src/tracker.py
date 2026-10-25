"""
Multi-Person Tracker Module (ByteTrack / MOT).
Tracks individuals across consecutive frames, assigns temporary tracking IDs,
records position/movement histories, calculates foot points and velocities,
and maintains continuity during temporary detection drops.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
import numpy as np

from src.detector import DetectedObject

@dataclass
class TrackedPerson:
    tracking_id: int                           # Temporary tracking ID (NOT verified student identity)
    bbox: Tuple[int, int, int, int]            # (x1, y1, x2, y2)
    detection_confidence: float
    first_seen_timestamp: float
    last_seen_timestamp: float
    assigned_seat_id: Optional[int] = None

    # Position and movement history: deque of (timestamp, centroid_x, centroid_y, foot_x, foot_y)
    trajectory: Deque[Tuple[float, float, float, float, float]] = field(
        default_factory=lambda: deque(maxlen=120)
    )

    velocity_px_per_sec: float = 0.0
    frames_since_update: int = 0
    total_frames_tracked: int = 0
    tracking_reliability: float = 1.0          # Metric based on continuous detection consistency

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def foot_point(self) -> Tuple[float, float]:
        """Estimated ground foot point (center-bottom) for perspective-accurate positioning."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, float(y2))

class PersonTracker:
    """
    ByteTrack / IoU Multi-Object Tracker for examination halls.
    Maintains persistent tracking states, handles temporary occlusions,
    and calculates motion dynamics.
    """

    def __init__(
        self,
        max_disappeared_frames: int = 15,
        max_matching_distance_px: float = 110.0,
        iou_threshold: float = 0.30
    ):
        self.max_disappeared_frames = max_disappeared_frames
        self.max_matching_distance_px = max_matching_distance_px
        self.iou_threshold = iou_threshold

        self.active_tracks: Dict[int, TrackedPerson] = {}
        self.next_tracking_id: int = 1

    @staticmethod
    def _compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        return interArea / float(boxAArea + boxBArea - interArea)

    def update(
        self,
        detected_people: List[DetectedObject],
        current_timestamp: float
    ) -> List[TrackedPerson]:
        """
        Updates tracked people with detections from the current frame.
        """
        # If no detections in current frame
        if len(detected_people) == 0:
            to_remove = []
            for tid, track in self.active_tracks.items():
                track.frames_since_update += 1
                track.tracking_reliability = max(0.2, track.tracking_reliability - 0.05)
                if track.frames_since_update > self.max_disappeared_frames:
                    to_remove.append(tid)
            for tid in to_remove:
                del self.active_tracks[tid]
            return list(self.active_tracks.values())

        det_boxes = [d.bbox for d in detected_people]
        det_confs = [d.confidence for d in detected_people]
        det_centroids = [d.centroid for d in detected_people]
        det_feet = [d.foot_point for d in detected_people]

        # First frame: initialize all detections
        if len(self.active_tracks) == 0:
            for bbox, conf, c_pt, f_pt in zip(det_boxes, det_confs, det_centroids, det_feet):
                track = TrackedPerson(
                    tracking_id=self.next_tracking_id,
                    bbox=bbox,
                    detection_confidence=conf,
                    first_seen_timestamp=current_timestamp,
                    last_seen_timestamp=current_timestamp,
                    total_frames_tracked=1
                )
                track.trajectory.append((current_timestamp, c_pt[0], c_pt[1], f_pt[0], f_pt[1]))
                self.active_tracks[self.next_tracking_id] = track
                self.next_tracking_id += 1
            return list(self.active_tracks.values())

        # Match existing tracks to new detections
        track_ids = list(self.active_tracks.keys())
        track_centroids = [self.active_tracks[tid].centroid for tid in track_ids]

        cost_matrix = np.zeros((len(track_ids), len(detected_people)), dtype=np.float32)
        for i, (tcx, tcy) in enumerate(track_centroids):
            for j, (dcx, dcy) in enumerate(det_centroids):
                dist = np.hypot(tcx - dcx, tcy - dcy)
                iou = self._compute_iou(self.active_tracks[track_ids[i]].bbox, det_boxes[j])
                # Blended cost: lower Euclidean distance and higher IoU produce lower cost
                cost_matrix[i, j] = dist * (1.0 - 0.5 * iou)

        matched_tracks = set()
        matched_dets = set()

        if cost_matrix.size > 0:
            sorted_indices = np.unravel_index(np.argsort(cost_matrix, axis=None), cost_matrix.shape)
            for r, c in zip(sorted_indices[0], sorted_indices[1]):
                if r in matched_tracks or c in matched_dets:
                    continue
                if cost_matrix[r, c] > self.max_matching_distance_px:
                    continue

                matched_tracks.add(r)
                matched_dets.add(c)
                tid = track_ids[r]
                track = self.active_tracks[tid]

                new_bbox = det_boxes[c]
                new_conf = det_confs[c]
                new_c = det_centroids[c]
                new_f = det_feet[c]

                # Compute velocity from trajectory
                velocity = 0.0
                if track.trajectory:
                    last_t, last_cx, last_cy, _, _ = track.trajectory[-1]
                    dt = max(1e-4, current_timestamp - last_t)
                    dist = np.hypot(new_c[0] - last_cx, new_c[1] - last_cy)
                    inst_v = dist / dt
                    velocity = 0.7 * inst_v + 0.3 * track.velocity_px_per_sec

                track.bbox = new_bbox
                track.detection_confidence = new_conf
                track.last_seen_timestamp = current_timestamp
                track.frames_since_update = 0
                track.total_frames_tracked += 1
                track.velocity_px_per_sec = velocity
                track.tracking_reliability = min(1.0, track.tracking_reliability + 0.05)
                track.trajectory.append((current_timestamp, new_c[0], new_c[1], new_f[0], new_f[1]))

        # Handle unmatched existing tracks
        to_remove = []
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                track = self.active_tracks[tid]
                track.frames_since_update += 1
                track.tracking_reliability = max(0.1, track.tracking_reliability - 0.08)
                if track.frames_since_update > self.max_disappeared_frames:
                    to_remove.append(tid)

        for tid in to_remove:
            del self.active_tracks[tid]

        # Handle unmatched detections (create new tracks)
        for j in range(len(detected_people)):
            if j not in matched_dets:
                track = TrackedPerson(
                    tracking_id=self.next_tracking_id,
                    bbox=det_boxes[j],
                    detection_confidence=det_confs[j],
                    first_seen_timestamp=current_timestamp,
                    last_seen_timestamp=current_timestamp,
                    total_frames_tracked=1
                )
                track.trajectory.append((
                    current_timestamp, det_centroids[j][0], det_centroids[j][1],
                    det_feet[j][0], det_feet[j][1]
                ))
                self.active_tracks[self.next_tracking_id] = track
                self.next_tracking_id += 1

        return list(self.active_tracks.values())

# Backward compatibility alias
ByteTracker = PersonTracker

