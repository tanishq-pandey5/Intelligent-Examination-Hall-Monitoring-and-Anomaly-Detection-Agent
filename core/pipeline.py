"""
End-to-End Examination Hall Video Monitoring and Anomaly Detection Pipeline.
Coordinates Detection -> Tracking -> Model-Based Agent Reasoning -> Evidence Storage -> Visual Overlays.
"""
import cv2
import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple, Generator, Any

from core.config import SystemConfig, SeatConfig, DEFAULT_CONFIG
from core.state import StudentState, RiskLevel, EventType
from core.detector import ExamHallDetector
from core.tracker import StudentTracker
from core.agent import ModelBasedExamMonitoringAgent, AgentDecision
from core.database import ExamEventDatabase

class ExamMonitoringPipeline:
    """
    Main monitoring pipeline integrating video input, computer vision detectors,
    multi-object tracking, model-based agent reasoning, evidence snapshot capturing,
    and visual HUD rendering.
    """

    def __init__(
        self,
        config: SystemConfig = DEFAULT_CONFIG,
        db: Optional[ExamEventDatabase] = None,
        use_yolo: bool = True
    ):
        self.config = config
        self.db = db or ExamEventDatabase(config.database_path)
        self.seats = config.generate_default_seats()

        # Core components
        self.detector = ExamHallDetector(
            model_path=config.yolo_model_path,
            device=config.device,
            conf_thresh=config.detection_confidence,
            iou_thresh=config.iou_threshold
        )
        self.tracker = StudentTracker(
            seats=self.seats,
            frame_w=config.default_frame_width,
            frame_h=config.default_frame_height
        )
        self.agent = ModelBasedExamMonitoringAgent(config=config)

        # Evidence storage
        os.makedirs(config.evidence_dir, exist_ok=True)
        self.frame_index = 0
        self.last_timestamp = 0.0

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        synthetic_hints: Optional[Dict[int, Dict]] = None
    ) -> Tuple[np.ndarray, List[AgentDecision], List[Dict[str, Any]]]:
        """
        Processes a single video frame through the complete pipeline.
        Returns:
          annotated_frame: Video frame with visual overlays, bounding boxes, and HUD
          decisions: List of AgentDecisions for all tracked students
          emitted_alerts: List of newly generated alert dictionaries stored in DB
        """
        self.frame_index += 1
        dt = max(1e-3, timestamp - self.last_timestamp) if self.last_timestamp > 0 else 1.0 / self.config.default_fps
        self.last_timestamp = timestamp

        # 1. Object Detection (Students & Mobile Phones)
        det_result = self.detector.detect_objects(frame)

        # If YOLO didn't detect persons (e.g. synthetic frame or empty frame),
        # extract candidate regions based on hall seat geometry for high-fidelity fallback
        if len(det_result.students) == 0:
            h, w = frame.shape[:2]
            for seat_id, seat in self.seats.items():
                box = seat.to_pixel_box(w, h)
                det_result.students.append((box, 0.90))

        # 2. Multi-Student Tracking across consecutive frames
        tracks = self.tracker.update(det_result.students, timestamp)

        # 3. Perception Synthesis (Head pose, Phone proximity, Velocity, Neighbor distances)
        perceptions = self.detector.build_perceptions(
            frame=frame,
            tracks=tracks,
            phones=det_result.phones,
            synthetic_hints=synthetic_hints
        )

        # 4. Model-Based Agent Reasoning & Rule Evaluation (Tasks 5, 6, 7, 8, 9, 10)
        decisions = self.agent.process_frame_perceptions(
            perceptions=perceptions,
            current_time=timestamp,
            dt=dt
        )

        # 5. Evidence Frame Saving & Database Storage (Task 11)
        emitted_alerts = []
        for decision in decisions:
            if decision.is_alert_triggered and decision.explainable_alert:
                primary_rule = decision.matched_rules[0] if decision.matched_rules else None
                event_type_str = primary_rule.event_type.value if primary_rule else "ANOMALY"
                event_name_str = primary_rule.rule_name if primary_rule else "Suspicious Activity"

                # Capture evidence frame snapshot with timestamp & bounding box
                evidence_filename = f"evidence_seat_{decision.seat_id}_t{int(timestamp*10):05d}.jpg"
                evidence_path = os.path.join(self.config.evidence_dir, evidence_filename)

                # Create evidence snapshot with highlighted investigation graphics
                evidence_img = self._create_evidence_snapshot(
                    frame=frame,
                    decision=decision,
                    timestamp=timestamp
                )
                cv2.imwrite(evidence_path, evidence_img)

                # Store in SQLite database (Task 11)
                event_id = self.db.insert_event(
                    video_time_sec=timestamp,
                    seat_id=decision.seat_id,
                    student_id=decision.student_id,
                    event_type=event_type_str,
                    event_name=event_name_str,
                    confidence=decision.risk_breakdown.certainty if decision.risk_breakdown else 0.85,
                    duration_seconds=primary_rule.metrics.get("duration_seconds", 5.0) if primary_rule else 5.0,
                    risk_level=decision.risk_level.value,
                    risk_score=decision.risk_score,
                    explanation=decision.explainable_alert,
                    evidence_frame_path=evidence_path,
                    status="ACTIVE"
                )

                emitted_alerts.append({
                    "id": event_id,
                    "seat_id": decision.seat_id,
                    "student_id": decision.student_id,
                    "event_type": event_type_str,
                    "risk_level": decision.risk_level.value,
                    "risk_score": decision.risk_score,
                    "explanation": decision.explainable_alert,
                    "evidence_path": evidence_path,
                    "timestamp": timestamp
                })

        # 6. Render Visual Annotations and Invigilator HUD (Task 12)
        annotated_frame = self._draw_visual_overlays(frame, tracks, decisions, timestamp)

        return annotated_frame, decisions, emitted_alerts

    def _create_evidence_snapshot(
        self,
        frame: np.ndarray,
        decision: AgentDecision,
        timestamp: float
    ) -> np.ndarray:
        """Generates a high-resolution evidence snapshot with zoomed banner and forensic overlay."""
        snapshot = frame.copy()
        h, w = snapshot.shape[:2]

        memory = self.agent.student_memories.get(decision.student_id)
        bbox = memory.last_bbox if memory else (0, 0, 100, 100)
        x1, y1, x2, y2 = bbox

        # Draw red evidence box with targeting crosshairs
        cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Forensic metadata header banner at top
        banner_h = 75
        cv2.rectangle(snapshot, (0, 0), (w, banner_h), (20, 20, 25), -1)
        cv2.line(snapshot, (0, banner_h), (w, banner_h), (0, 0, 255), 3)

        cv2.putText(
            snapshot, f"EVIDENCE LOG: SEAT {decision.seat_id}  |  TIME: {timestamp:.1f}s  |  RISK: {decision.risk_level.value.upper()} ({decision.risk_score}/100)",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 100, 255), 2, cv2.LINE_AA
        )
        if decision.explainable_alert:
            cv2.putText(
                snapshot, decision.explainable_alert,
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1, cv2.LINE_AA
            )

        return snapshot

    def _draw_visual_overlays(
        self,
        frame: np.ndarray,
        tracks: List,
        decisions: List[AgentDecision],
        timestamp: float
    ) -> np.ndarray:
        """Renders comprehensive examination hall visualization on top of video feed."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Lookup decisions by student_id
        dec_map = {d.student_id: d for d in decisions}

        # 1. Draw designated examination hall seat zones (dashed or thin outlines)
        for s_id, seat in self.seats.items():
            px_box = seat.to_pixel_box(w, h)
            # Find if any student currently assigned to this seat
            assigned_dec = next((d for d in decisions if d.seat_id == s_id), None)
            state_color = (120, 120, 120)
            if assigned_dec:
                state_color = assigned_dec.current_state.color_bgr

            # Thin seat border
            cv2.rectangle(vis, (px_box[0], px_box[1]), (px_box[2], px_box[3]), (90, 95, 105), 1)

        # 2. Draw student bounding boxes, state badges, and gaze arrows
        for track in tracks:
            decision = dec_map.get(track.track_id)
            if not decision:
                continue

            state = decision.current_state
            box_color = state.color_bgr
            thickness = 3 if state in [StudentState.ALERT, StudentState.SUSPICIOUS] else 2

            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, thickness)

            # State & Seat Header Tag
            tag_text = f"Seat #{decision.seat_id} | {state.value} ({int(decision.risk_score)}%)"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            tag_y1 = max(0, y1 - th - 8)
            tag_y2 = y1
            cv2.rectangle(vis, (x1, tag_y1), (x1 + tw + 10, tag_y2), box_color, -1)
            cv2.putText(
                vis, tag_text, (x1 + 5, tag_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA
            )

            # Draw Head Turn Orientation Vector (Gaze direction arrow)
            memory = self.agent.student_memories.get(track.track_id)
            if memory and len(memory.head_turn_events) > 0:
                _, last_angle, _ = memory.head_turn_events[-1]
                if abs(last_angle) >= 20.0:
                    head_cx = (x1 + x2) // 2
                    head_cy = y1 + int((y2 - y1) * 0.20)
                    arrow_dx = int(np.sin(np.radians(last_angle)) * 38)
                    arrow_dy = int(-np.cos(np.radians(last_angle)) * 8)
                    arrow_tip = (head_cx + arrow_dx, head_cy + arrow_dy)
                    cv2.arrowedLine(vis, (head_cx, head_cy), arrow_tip, (0, 140, 255), 2, tipLength=0.35)

        # 3. Active Alert Toast / Banner at bottom for recent alerts
        active_critical_alerts = [
            d for d in decisions if d.current_state in [StudentState.ALERT, StudentState.SUSPICIOUS]
        ]
        if active_critical_alerts:
            # Banner at bottom
            top_alert = active_critical_alerts[0]
            banner_bg = (30, 30, 180) if top_alert.current_state == StudentState.ALERT else (0, 120, 220)
            cv2.rectangle(vis, (0, h - 45), (w, h), banner_bg, -1)
            cv2.line(vis, (0, h - 45), (w, h - 45), (255, 255, 255), 2)

            msg = top_alert.explainable_alert or f"ALERT: Suspicious activity at Seat {top_alert.seat_id}!"
            cv2.putText(
                vis, f"⚠️  {msg}", (25, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
            )

        return vis
