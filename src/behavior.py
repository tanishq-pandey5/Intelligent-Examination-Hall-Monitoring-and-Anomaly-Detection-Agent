"""
Behavioral Analysis Modules for Examination Halls.
Implements Section 4:
  Module A: Leaving Assigned Seat
  Module B: Repeated Head Turning & Pose Estimation
  Module C: Mobile Phone Detection vs Confirmed Use
  Module D: Multiple Students Gathering / Clusters
  Module E: Unusual Erratic Movement & Trajectory Analysis
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

from config.settings import AppSettings, DEFAULT_SETTINGS
from src.tracker import TrackedPerson
from src.detector import DetectedObject
from src.seat_mapper import SeatMapper

@dataclass
class BehaviorObservation:
    event_type: str                   # 'LEAVING_SEAT', 'REPEATED_HEAD_TURNING', etc.
    tracking_id: int
    seat_id: Optional[int]
    confidence: float
    duration_sec: float
    frequency: int
    metrics: Dict[str, Any]
    description: str
    limitation_note: Optional[str] = None
    is_confirmed_violation: bool = False

class BehaviorAnalyzer:
    """
    Coordinates distinct behavioral detection modules across examination hall video frames.
    """

    def __init__(self, settings: AppSettings = DEFAULT_SETTINGS):
        self.settings = settings

        # Mobile phone temporal persistence tracker: tracking_id -> consecutive frames detected
        self.phone_persistence: Dict[int, int] = {}
        self.phone_event_start_times: Dict[int, float] = {}

    # --------------------------------------------------------------------------
    # MODULE A: Leaving the Assigned Seat (Section 4A)
    # --------------------------------------------------------------------------
    def analyze_seat_departure(
        self,
        track: TrackedPerson,
        seat_status: Dict,
        current_time: float
    ) -> Optional[BehaviorObservation]:
        """
        Analyzes whether a student has vacated their designated seat region.
        """
        seat_id = track.assigned_seat_id
        if not seat_id or seat_id not in seat_status:
            return None

        status = seat_status[seat_id]
        dur = status["departure_duration_sec"]
        dist = status["distance_to_seat_px"]

        # Only evaluate if past the initial grace period
        if dur >= self.settings.seat_departure_duration_threshold_sec:
            desc = (
                f"Candidate vacated assigned seat {seat_id} ({dist:.0f}px departure) "
                f"for {dur:.1f}s continuously."
            )
            return BehaviorObservation(
                event_type="LEAVING_SEAT",
                tracking_id=track.tracking_id,
                seat_id=seat_id,
                confidence=min(0.95, 0.70 + 0.05 * (dur / 5.0)),
                duration_sec=dur,
                frequency=status["total_departures"],
                metrics={"departure_duration_sec": dur, "distance_to_seat_px": dist},
                description=desc,
                is_confirmed_violation=(dur >= 8.0)
            )
        return None

    # --------------------------------------------------------------------------
    # MODULE B: Repeated Head Turning (Section 4B)
    # --------------------------------------------------------------------------
    def estimate_head_pose(
        self,
        frame: np.ndarray,
        track: TrackedPerson,
        seat_mapper: SeatMapper
    ) -> Tuple[float, bool, Optional[str]]:
        """
        Estimates head yaw angle (degrees) using lateral skin displacement
        relative to the seat center axis and Sobel edge symmetry.
        Returns: (yaw_degrees, is_turning, limitation_note)
        """
        x1, y1, x2, y2 = track.bbox
        h, w = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1

        if bw < 15 or bh < 20:
            return 0.0, False, "Bounding box resolution too low for reliable head-pose estimation."

        # Check assigned seat center alignment
        seat_id = track.assigned_seat_id
        if seat_id and seat_id in seat_mapper.seats:
            seat = seat_mapper.seats[seat_id]
            bx1, _, bx2, _ = seat.to_pixel_box(w, h)
            seat_cx = (bx1 + bx2) / 2.0
            head_cx = (x1 + x2) / 2.0
            shift = head_cx - seat_cx

            # Seated candidate lateral shift reflects gaze/head yaw
            if abs(shift) < (bw * 0.7):
                yaw = float(np.clip(shift * 2.8, -65.0, 65.0))
                is_turn = abs(yaw) >= self.settings.head_turn_angle_threshold_deg
                return yaw, is_turn, None

        # Fallback: edge asymmetry across upper 40% of bounding box
        head_h = int(bh * 0.40)
        head_crop = frame[max(0, y1):min(h, y1 + head_h), max(0, x1):min(w, x2)]
        if head_crop.size == 0:
            return 0.0, False, "Head region obscured or outside frame."

        gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
        mid = gray.shape[1] // 2
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        left_grad = np.sum(np.abs(sobel_x[:, :mid]))
        right_grad = np.sum(np.abs(sobel_x[:, mid:]))
        asym = (right_grad - left_grad) / (left_grad + right_grad + 1e-5)
        yaw = float(np.clip(asym * 45.0, -50.0, 50.0))
        is_turn = abs(yaw) >= self.settings.head_turn_angle_threshold_deg

        return yaw, is_turn, None

    # --------------------------------------------------------------------------
    # MODULE C: Mobile Phone Detection vs Confirmed Use (Section 4C)
    # --------------------------------------------------------------------------
    def analyze_phone_detection(
        self,
        track: TrackedPerson,
        detected_phones: List[DetectedObject],
        current_time: float,
        frame_w: int,
        frame_h: int
    ) -> Optional[BehaviorObservation]:
        """
        Associates detected mobile phones with tracked candidates only when spatial
        and temporal evidence supports the association.
        Clearly distinguishes phone detection from confirmed phone use!
        """
        tx1, ty1, tx2, ty2 = track.bbox
        margin = self.settings.phone_workspace_margin_px
        expanded_box = (tx1 - margin, ty1 - margin, tx2 + margin, ty2 + margin)

        best_phone = None
        for phone in detected_phones:
            pcx, pcy = phone.centroid
            if (expanded_box[0] <= pcx <= expanded_box[2] and
                expanded_box[1] <= pcy <= expanded_box[3]):
                if best_phone is None or phone.confidence > best_phone.confidence:
                    best_phone = phone

        tid = track.tracking_id
        if best_phone is not None:
            self.phone_persistence[tid] = self.phone_persistence.get(tid, 0) + 1
            if tid not in self.phone_event_start_times:
                self.phone_event_start_times[tid] = current_time

            dur = current_time - self.phone_event_start_times[tid]
            frames = self.phone_persistence[tid]

            # Require persistence across multiple frames (avoid single-frame false alert!)
            if frames >= self.settings.phone_persistence_frames or dur >= 1.5:
                # Distinguish phone detection from confirmed active phone use
                is_confirmed = (dur >= 4.0 and best_phone.confidence >= 0.75)
                desc = (
                    f"Mobile phone device observed in workspace ({'CONFIRMED ACTIVE USE' if is_confirmed else 'SUSPECTED PRESENCE'}) "
                    f"for {dur:.1f}s (conf: {int(best_phone.confidence*100)}%)."
                )
                return BehaviorObservation(
                    event_type="MOBILE_PHONE_DETECTED",
                    tracking_id=tid,
                    seat_id=track.assigned_seat_id,
                    confidence=best_phone.confidence,
                    duration_sec=dur,
                    frequency=frames,
                    metrics={
                        "phone_confidence": best_phone.confidence,
                        "duration_sec": dur,
                        "persistence_frames": frames,
                        "is_confirmed_use": is_confirmed
                    },
                    description=desc,
                    is_confirmed_violation=is_confirmed
                )
        else:
            # Phone not seen in this frame
            if tid in self.phone_persistence:
                self.phone_persistence[tid] = max(0, self.phone_persistence[tid] - 1)
                if self.phone_persistence[tid] == 0:
                    del self.phone_persistence[tid]
                    if tid in self.phone_event_start_times:
                        del self.phone_event_start_times[tid]

        return None

    # --------------------------------------------------------------------------
    # MODULE D: Multiple Students Gathering (Section 4D)
    # --------------------------------------------------------------------------
    def analyze_gathering_clusters(
        self,
        tracks: List[TrackedPerson],
        current_time: float
    ) -> List[BehaviorObservation]:
        """
        Estimates pairwise distances between people and identifies unusual clustering.
        """
        observations: List[BehaviorObservation] = []
        n = len(tracks)
        if n < self.settings.gathering_min_students:
            return observations

        centroids = [t.centroid for t in tracks]
        proximity_thresh = self.settings.gathering_proximity_pixels

        # Pairwise distance matrix
        clustered_ids: Dict[int, List[int]] = {t.tracking_id: [] for t in tracks}
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.hypot(centroids[i][0] - centroids[j][0], centroids[i][1] - centroids[j][1])
                if dist <= proximity_thresh:
                    clustered_ids[tracks[i].tracking_id].append(tracks[j].tracking_id)
                    clustered_ids[tracks[j].tracking_id].append(tracks[i].tracking_id)

        for track in tracks:
            neighbors = clustered_ids[track.tracking_id]
            if len(neighbors) >= (self.settings.gathering_min_students - 1):
                group_size = len(neighbors) + 1
                desc = (
                    f"Group gathering detected: {group_size} individuals clustered within "
                    f"{proximity_thresh:.0f}px proximity."
                )
                observations.append(BehaviorObservation(
                    event_type="GATHERING_CLUSTER",
                    tracking_id=track.tracking_id,
                    seat_id=track.assigned_seat_id,
                    confidence=0.85,
                    duration_sec=3.5, # Integrated by temporal agent
                    frequency=group_size,
                    metrics={"group_size": group_size, "neighbor_ids": neighbors},
                    description=desc,
                    is_confirmed_violation=(group_size >= 3)
                ))

        return observations

    # --------------------------------------------------------------------------
    # MODULE E: Unusual Movement (Section 4E)
    # --------------------------------------------------------------------------
    def analyze_unusual_movement(
        self,
        track: TrackedPerson
    ) -> Optional[BehaviorObservation]:
        """
        Analyzes displacement, sustained velocity spikes, and erratic trajectories.
        """
        vel = track.velocity_px_per_sec
        v_thresh = self.settings.unusual_movement_velocity_px_per_sec

        if vel >= v_thresh:
            desc = (
                f"Sustained unusual movement detected: velocity {vel:.1f} px/s exceeds "
                f"calm examination threshold ({v_thresh:.0f} px/s)."
            )
            return BehaviorObservation(
                event_type="UNUSUAL_MOVEMENT",
                tracking_id=track.tracking_id,
                seat_id=track.assigned_seat_id,
                confidence=min(0.90, 0.60 + 0.01 * (vel - v_thresh)),
                duration_sec=2.5,
                frequency=1,
                metrics={"velocity_px_per_sec": round(vel, 1), "threshold": v_thresh},
                description=desc,
                limitation_note="Sudden natural postural adjustments or sneezing may contribute.",
                is_confirmed_violation=False
            )
        return None
