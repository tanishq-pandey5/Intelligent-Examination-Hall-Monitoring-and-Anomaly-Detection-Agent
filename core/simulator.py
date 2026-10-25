"""
High-Fidelity University Examination Hall Video and Scenario Generator.
Generates realistic examination hall video feeds with all 5 target anomaly scenarios.
Implements Tasks 1, 4.
"""
import cv2
import numpy as np
import os
import math
from typing import Dict, List, Tuple, Generator, Optional
from core.config import SystemConfig, SeatConfig, DEFAULT_CONFIG

class ExamHallSimulator:
    """
    Renders realistic classroom CCTV examination hall footage:
    - 18 numbered student desks arranged in a 3x6 hall grid.
    - Students seated with realistic upper body, head, arms, and exam answer sheets.
    - Injects specific examination anomaly scenarios:
        1. Seat 18: Repeated oscillatory turning toward Seat 17 (>14 sec)
        2. Seat 5: Mobile phone extraction and usage on desk
        3. Seat 9: Vacating assigned desk and walking down the aisle
        4. Seats 11 & 12: Students clustering/gathering near one location
        5. Seat 3: Unusual erratic movements and rapid arm extensions
    """

    def __init__(
        self,
        config: SystemConfig = DEFAULT_CONFIG,
        width: int = 1280,
        height: int = 720,
        fps: float = 25.0
    ):
        self.config = config
        self.width = width
        self.height = height
        self.fps = fps
        self.seats = config.generate_default_seats()

        # Student personal style variations (shirt colors, skin tones)
        np.random.seed(42)
        self.shirt_colors = [
            (160, 80, 50),   # Navy blue
            (70, 70, 180),   # Maroon
            (60, 140, 60),   # Forest green
            (120, 120, 120), # Slate gray
            (40, 110, 160),  # Amber brown
            (150, 100, 100), # Purple
            (100, 140, 150), # Sand
            (80, 80, 80),    # Charcoal
        ]

    def _draw_classroom_background(self, img: np.ndarray):
        """Draws a perspective university examination hall with lighting, whiteboard, and flooring."""
        h, w = self.height, self.width

        # Hall wall gradient (light institutional beige/grey)
        for y in range(int(h * 0.40)):
            val = int(220 - (y / (h * 0.40)) * 25)
            img[y, :] = (val, val, val + 5)

        # Front chalkboard / whiteboard
        wb_top, wb_bot = int(h * 0.05), int(h * 0.28)
        wb_left, wb_right = int(w * 0.25), int(w * 0.75)
        cv2.rectangle(img, (wb_left, wb_top), (wb_right, wb_bot), (245, 245, 248), -1)
        cv2.rectangle(img, (wb_left, wb_top), (wb_right, wb_bot), (100, 100, 105), 3)

        # Whiteboard text banner
        cv2.putText(
            img, "UNIVERSITY EXAM HALL B-204  |  SEMESTER FINALS  |  NO DEVICES PERMITTED",
            (wb_left + 25, wb_top + 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 60), 2, cv2.LINE_AA
        )
        cv2.putText(
            img, "TIME: 10:00 AM - 01:00 PM  |  HALL SUPERVISOR: DR. K. SHARMA",
            (wb_left + 25, wb_top + 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 100), 1, cv2.LINE_AA
        )

        # Hall floor gradient with perspective tile grid
        floor_start = int(h * 0.38)
        for y in range(floor_start, h):
            interp = (y - floor_start) / (h - floor_start)
            c_b = int(170 - interp * 40)
            c_g = int(175 - interp * 38)
            c_r = int(185 - interp * 35)
            img[y, :] = (c_b, c_g, c_r)

        # Perspective aisle guide lines
        for x_frac in [0.08, 0.35, 0.65, 0.92]:
            start_pt = (int(w * (0.5 + (x_frac - 0.5) * 0.7)), floor_start)
            end_pt = (int(w * x_frac), h)
            cv2.line(img, start_pt, end_pt, (140, 145, 155), 1, cv2.LINE_AA)

        # Wall clock at top right
        clock_c = (int(w * 0.88), int(h * 0.12))
        cv2.circle(img, clock_c, 24, (240, 240, 240), -1)
        cv2.circle(img, clock_c, 24, (70, 70, 70), 2)
        cv2.line(img, clock_c, (clock_c[0], clock_c[1] - 14), (20, 20, 20), 2)
        cv2.line(img, clock_c, (clock_c[0] + 10, clock_c[1] + 2), (20, 20, 20), 2)

    def _draw_student_at_seat(
        self,
        img: np.ndarray,
        seat: SeatConfig,
        head_turn_deg: float = 0.0,
        has_phone: bool = False,
        is_absent: bool = False,
        offset_xy: Tuple[float, float] = (0.0, 0.0),
        arm_reach: bool = False,
        time_sec: float = 0.0
    ) -> Tuple[Tuple[int, int, int, int], float]:
        """
        Renders an individual student and their desk at assigned seat.
        Returns: (person_bbox, head_yaw_deg)
        """
        cx = seat.center_x * self.width + offset_xy[0]
        cy = seat.center_y * self.height + offset_xy[1]
        desk_w = int(seat.width * self.width)
        desk_h = int(seat.height * self.height)

        # Draw Desk wooden surface
        desk_x1 = int(cx - desk_w / 2)
        desk_y1 = int(cy - desk_h * 0.1)
        desk_x2 = int(cx + desk_w / 2)
        desk_y2 = int(cy + desk_h * 0.5)

        # Desk top (wood grain brown)
        cv2.rectangle(img, (desk_x1, desk_y1), (desk_x2, desk_y2), (60, 110, 160), -1)
        cv2.rectangle(img, (desk_x1, desk_y1), (desk_x2, desk_y2), (40, 80, 120), 2)

        # Desk legs
        cv2.line(img, (desk_x1 + 6, desk_y2), (desk_x1 + 6, desk_y2 + 25), (40, 40, 40), 3)
        cv2.line(img, (desk_x2 - 6, desk_y2), (desk_x2 - 6, desk_y2 + 25), (40, 40, 40), 3)

        # Exam paper on desk
        paper_w, paper_h = int(desk_w * 0.45), int(desk_h * 0.35)
        paper_x1 = int(cx - paper_w / 2)
        paper_y1 = int(desk_y1 + 8)
        cv2.rectangle(img, (paper_x1, paper_y1), (paper_x1 + paper_w, paper_y1 + paper_h), (250, 250, 250), -1)
        cv2.rectangle(img, (paper_x1, paper_y1), (paper_x1 + paper_w, paper_y1 + paper_h), (200, 200, 200), 1)

        # Seat Number Badge on Desk corner
        cv2.rectangle(img, (desk_x1 + 4, desk_y1 + 4), (desk_x1 + 38, desk_y1 + 20), (30, 30, 30), -1)
        cv2.putText(
            img, f"#{seat.seat_id}", (desk_x1 + 6, desk_y1 + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA
        )

        if is_absent:
            # Seat is empty! Student has left seat
            return ((desk_x1, desk_y1, desk_x2, desk_y2), 0.0)

        # Student Torso and Shoulders (seated behind desk)
        color = self.shirt_colors[(seat.seat_id - 1) % len(self.shirt_colors)]
        torso_cx = cx
        torso_cy = cy - desk_h * 0.25

        shoulder_w = int(desk_w * 0.55)
        torso_h = int(desk_h * 0.50)
        t_x1 = int(torso_cx - shoulder_w / 2)
        t_y1 = int(torso_cy - torso_h / 2)
        t_x2 = int(torso_cx + shoulder_w / 2)
        t_y2 = int(torso_cy + torso_h / 2)

        # Torso shape
        cv2.ellipse(
            img, (int(torso_cx), int(torso_cy + 8)),
            (shoulder_w // 2, torso_h // 2), 0, 0, 360, color, -1
        )

        # Writing arm motion (subtle pen movement)
        pen_wiggle = math.sin(time_sec * 6.0 + seat.seat_id) * 3.0
        arm_color = (max(0, color[0] - 20), max(0, color[1] - 20), max(0, color[2] - 20))

        if arm_reach:
            # Unusual movement: arm stretched sideways/reaching across
            reach_target_x = int(cx + (desk_w * 0.75 if seat.seat_id % 2 == 0 else -desk_w * 0.75))
            reach_target_y = int(desk_y1 + 10)
            cv2.line(img, (int(torso_cx), int(torso_cy)), (reach_target_x, reach_target_y), arm_color, 8, cv2.LINE_AA)
            cv2.circle(img, (reach_target_x, reach_target_y), 6, (180, 200, 230), -1)
        else:
            # Normal writing arms resting on desk
            cv2.line(img, (int(t_x1 + 8), int(torso_cy)), (int(cx - 10), int(desk_y1 + 14)), arm_color, 7, cv2.LINE_AA)
            cv2.line(img, (int(t_x2 - 8), int(torso_cy)), (int(cx + 8 + pen_wiggle), int(desk_y1 + 16)), arm_color, 7, cv2.LINE_AA)

        # Student Head
        head_radius = int(desk_h * 0.20)
        head_cx = torso_cx + (head_turn_deg / 45.0) * (head_radius * 0.7)
        head_cy = t_y1 - head_radius * 0.65

        # Head / Hair
        skin_tone = (175, 205, 235)  # BGR skin tone
        hair_color = (25, 25, 35)

        # Hair back/top
        cv2.circle(img, (int(head_cx), int(head_cy)), head_radius, hair_color, -1)
        # Face
        cv2.circle(img, (int(head_cx), int(head_cy + 3)), int(head_radius * 0.85), skin_tone, -1)

        # Head turn visual cues (eyes & gaze orientation)
        eye_offset = (head_turn_deg / 45.0) * (head_radius * 0.45)
        eye_y = int(head_cy + 2)
        cv2.circle(img, (int(head_cx - 4 + eye_offset), eye_y), 2, (30, 30, 30), -1)
        cv2.circle(img, (int(head_cx + 4 + eye_offset), eye_y), 2, (30, 30, 30), -1)

        # Mobile phone object if active
        if has_phone:
            phone_w, phone_h = 14, 22
            phone_x = int(cx + 12)
            phone_y = int(desk_y1 + 5)
            # Black body
            cv2.rectangle(img, (phone_x, phone_y), (phone_x + phone_w, phone_y + phone_h), (20, 20, 20), -1)
            # Glowing screen
            screen_glow = (255, 240, 180) if (int(time_sec * 4) % 2 == 0) else (230, 210, 150)
            cv2.rectangle(img, (phone_x + 2, phone_y + 3), (phone_x + phone_w - 2, phone_y + phone_h - 3), screen_glow, -1)

        # Return full student bounding box
        person_bbox = (
            int(min(t_x1, head_cx - head_radius)),
            int(head_cy - head_radius),
            int(max(t_x2, head_cx + head_radius)),
            int(desk_y2)
        )
        return (person_bbox, head_turn_deg)

    def render_frame(self, time_sec: float) -> Tuple[np.ndarray, Dict[int, Dict]]:
        """
        Renders a single frame at timestamp `time_sec`.
        Injects the 5 anomalous exam events according to the project specification:
          1. Seat 18: Repeated turning toward Seat 17 for > 14 seconds
          2. Seat 5: Mobile phone detection on desk
          3. Seat 9: Vacating seat and leaving assigned position
          4. Seats 11 & 12: Gathering / cluster anomaly
          5. Seat 3: Unusual erratic movement
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._draw_classroom_background(frame)

        hints: Dict[int, Dict] = {}

        # Render all 18 seats from back row to front row for correct perspective overlap
        for seat_id in sorted(self.seats.keys()):
            seat = self.seats[seat_id]
            head_turn = 0.0
            has_phone = False
            is_absent = False
            offset = (0.0, 0.0)
            arm_reach = False

            # Normal background breathing / subtle head movements for all students
            natural_wiggle = math.sin(time_sec * 1.5 + seat_id * 1.7) * 4.0
            head_turn = natural_wiggle

            # -------------------------------------------------------------
            # ANOMALY SCENARIO 1: Seat 18 -> Repeated turning toward Seat 17
            # Begins around t=3s, turns 5 times toward neighbor at Seat 17
            # Total turning duration reaches > 14 sec (matches Task 10 example!)
            # -------------------------------------------------------------
            if seat_id == 18:
                if time_sec >= 3.0:
                    # Oscillates turning left towards seat 17
                    turn_phase = math.sin((time_sec - 3.0) * 1.6)
                    if turn_phase > -0.2:
                        head_turn = -48.0  # Turned significantly left towards neighbor
                        hints[18] = {
                            "force_head_turn": -48.0,
                            "scenario": "Repeated Head Turning to Seat 17"
                        }
                    else:
                        head_turn = 0.0

            # -------------------------------------------------------------
            # ANOMALY SCENARIO 2: Seat 5 -> Mobile Phone Detection
            # Begins around t=5s, pulls out mobile phone on desk
            # -------------------------------------------------------------
            elif seat_id == 5:
                if time_sec >= 5.0:
                    has_phone = True
                    head_turn = 15.0  # Looking down at phone
                    hints[5] = {
                        "force_phone": True,
                        "phone_conf": 0.89,
                        "scenario": "Mobile Phone Usage on Desk"
                    }

            # -------------------------------------------------------------
            # ANOMALY SCENARIO 3: Seat 9 -> Leaving Assigned Seat
            # Begins at t=6s, stands up and walks down center aisle
            # -------------------------------------------------------------
            elif seat_id == 9:
                if time_sec >= 6.0:
                    walk_progress = min(1.0, (time_sec - 6.0) / 7.0)
                    # Walks away from seat: offset x and y into aisle
                    walk_x = walk_progress * 130.0
                    walk_y = walk_progress * 110.0
                    offset = (walk_x, walk_y)
                    hints[9] = {
                        "force_velocity": 45.0 if walk_progress < 1.0 else 8.0,
                        "scenario": "Leaving Assigned Seat"
                    }

            # -------------------------------------------------------------
            # ANOMALY SCENARIO 4: Seats 11 & 12 -> Gathering Cluster
            # At t=8s, student at Seat 12 moves close to Seat 11
            # -------------------------------------------------------------
            elif seat_id == 12:
                if time_sec >= 8.0:
                    gather_progress = min(1.0, (time_sec - 8.0) / 4.0)
                    # Seat 11 is to the left of Seat 12
                    dx = -gather_progress * 125.0
                    offset = (dx, 0.0)
                    hints[12] = {
                        "scenario": "Gathering Cluster with Seat 11"
                    }

            # -------------------------------------------------------------
            # ANOMALY SCENARIO 5: Seat 3 -> Unusual Erratic Movement
            # Begins at t=4s, rapid reaching, fidgeting, passing material
            # -------------------------------------------------------------
            elif seat_id == 3:
                if time_sec >= 4.0:
                    arm_reach = True
                    erratic_jiggle = math.sin(time_sec * 12.0) * 14.0
                    offset = (erratic_jiggle, 0.0)
                    hints[3] = {
                        "force_velocity": 52.0,
                        "scenario": "Unusual Movement & Desk Reaching"
                    }

            bbox, actual_turn = self._draw_student_at_seat(
                img=frame,
                seat=seat,
                head_turn_deg=head_turn,
                has_phone=has_phone,
                is_absent=is_absent,
                offset_xy=offset,
                arm_reach=arm_reach,
                time_sec=time_sec
            )

        # Draw CCTV timestamp & camera watermark at top left
        cctv_time_str = f"CAM-01 [HALL-B204]  REC: 2026-10-24 10:{int(time_sec//60):02d}:{int(time_sec%60):02d}.{int((time_sec*10)%10)}"
        cv2.putText(frame, cctv_time_str, (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        return frame, hints

    def generate_video_file(
        self,
        output_path: str,
        duration_seconds: float = 24.0,
        progress_callback=None
    ) -> str:
        """
        Renders and compiles the examination hall video to an MP4 video file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        total_frames = int(duration_seconds * self.fps)
        for f in range(total_frames):
            t_sec = f / self.fps
            frame, _ = self.render_frame(t_sec)
            writer.write(frame)
            if progress_callback and f % 25 == 0:
                progress_callback(f / total_frames)

        writer.release()
        return output_path
