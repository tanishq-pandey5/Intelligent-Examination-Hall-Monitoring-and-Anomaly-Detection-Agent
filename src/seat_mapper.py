"""
Seat Mapper and Foot-Point Spatial Occupancy Module.
Implements manual seat region configuration, foot-point projection,
seat assignment, occupancy tracking, and departure/return monitoring.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

from config.settings import SeatRegion, DEFAULT_SETTINGS
from src.tracker import TrackedPerson

@dataclass
class SeatOccupancyRecord:
    seat_id: int
    current_occupant_id: Optional[int] = None
    is_occupied: bool = True
    assigned_occupant_id: Optional[int] = None
    time_spent_in_seat_sec: float = 0.0
    departure_timestamp: Optional[float] = None
    return_timestamp: Optional[float] = None
    total_departures_count: int = 0
    current_departure_duration_sec: float = 0.0

class SeatMapper:
    """
    Manages examination hall seating geometry, foot-point assignment,
    and seat departure / return lifecycle.
    """

    def __init__(
        self,
        custom_seats: Optional[Dict[int, SeatRegion]] = None,
        boundary_margin_px: float = 25.0,
        departure_grace_period_sec: float = 2.0
    ):
        self.seats = custom_seats or DEFAULT_SETTINGS.generate_default_seating_grid()
        self.boundary_margin_px = boundary_margin_px
        self.departure_grace_period_sec = departure_grace_period_sec

        # Seat records: seat_id -> SeatOccupancyRecord
        self.seat_records: Dict[int, SeatOccupancyRecord] = {
            sid: SeatOccupancyRecord(seat_id=sid) for sid in self.seats
        }

        # Track to assigned seat lookup: tracking_id -> seat_id
        self.track_assigned_seats: Dict[int, int] = {}

    def auto_assign_initial_seats(self, tracks: List[TrackedPerson], frame_w: int, frame_h: int):
        """
        Associates initial tracked people to their closest designated seats
        based on foot points.
        """
        for track in tracks:
            if track.tracking_id in self.track_assigned_seats:
                track.assigned_seat_id = self.track_assigned_seats[track.tracking_id]
                continue

            fx, fy = track.foot_point
            best_seat_id = None
            min_dist = float("inf")

            for sid, seat in self.seats.items():
                bx1, by1, bx2, by2 = seat.to_pixel_box(frame_w, frame_h)
                scx = (bx1 + bx2) / 2.0
                scy = (by1 + by2) / 2.0
                dist = np.hypot(fx - scx, fy - scy)

                if dist < min_dist:
                    min_dist = dist
                    best_seat_id = sid

            if best_seat_id is not None:
                self.track_assigned_seats[track.tracking_id] = best_seat_id
                track.assigned_seat_id = best_seat_id
                rec = self.seat_records[best_seat_id]
                rec.assigned_occupant_id = track.tracking_id
                rec.current_occupant_id = track.tracking_id
                rec.is_occupied = True

    def update_occupancy(
        self,
        tracks: List[TrackedPerson],
        current_timestamp: float,
        dt: float,
        frame_w: int,
        frame_h: int
    ) -> Dict[int, Dict]:
        """
        Evaluates foot-point locations against assigned seats.
        Returns departure statuses and duration metrics for each seat.
        """
        # Ensure tracks have assignments
        self.auto_assign_initial_seats(tracks, frame_w, frame_h)

        track_map = {t.tracking_id: t for t in tracks}
        seat_status_report: Dict[int, Dict] = {}

        for sid, seat in self.seats.items():
            rec = self.seat_records[sid]
            assigned_tid = rec.assigned_occupant_id
            assigned_track = track_map.get(assigned_tid) if assigned_tid else None

            is_inside_seat = False
            distance_to_seat = 0.0

            if assigned_track is not None:
                fx, fy = assigned_track.foot_point
                is_inside_seat = seat.contains_point(fx, fy, frame_w, frame_h, margin=self.boundary_margin_px)

                # Pixel distance from foot point to seat center
                bx1, by1, bx2, by2 = seat.to_pixel_box(frame_w, frame_h)
                scx = (bx1 + bx2) / 2.0
                scy = (by1 + by2) / 2.0
                distance_to_seat = float(np.hypot(fx - scx, fy - scy))

            # Departure logic with grace period
            if assigned_track is None or not is_inside_seat:
                # Student is absent from seat
                if rec.departure_timestamp is None:
                    rec.departure_timestamp = current_timestamp
                    rec.total_departures_count += 1

                rec.current_departure_duration_sec = current_timestamp - rec.departure_timestamp
                rec.is_occupied = False
            else:
                # Student is inside seat
                if rec.departure_timestamp is not None:
                    # Student returned!
                    rec.return_timestamp = current_timestamp
                    rec.departure_timestamp = None
                    rec.current_departure_duration_sec = 0.0

                rec.time_spent_in_seat_sec += dt
                rec.is_occupied = True

            seat_status_report[sid] = {
                "seat_id": sid,
                "assigned_tracking_id": assigned_tid,
                "is_occupied": rec.is_occupied,
                "departure_duration_sec": round(rec.current_departure_duration_sec, 1),
                "distance_to_seat_px": round(distance_to_seat, 1),
                "total_departures": rec.total_departures_count,
                "time_in_seat_sec": round(rec.time_spent_in_seat_sec, 1)
            }

        return seat_status_report
