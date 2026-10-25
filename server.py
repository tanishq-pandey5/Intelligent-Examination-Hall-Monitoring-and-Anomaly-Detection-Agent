"""
==============================================================================
FastAPI High-Performance Backend Server
==============================================================================
Replaces Streamlit with a modern, asynchronous, zero-lag REST & streaming server
powering the custom examination monitoring frontend.
==============================================================================
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import cv2
import numpy as np
import pandas as pd

from config.settings import AppSettings, DEFAULT_SETTINGS
from src.video_input import VideoInputHandler
from src.frame_sampler import FrameSampler
from src.detector import ExamHallDetector, DetectedObject
from src.tracker import PersonTracker, TrackedPerson
from src.seat_mapper import SeatMapper
from src.behavior import BehaviorAnalyzer
from src.agent import PersonAgentMemory, AgentState, ReviewState
from src.reasoning import RuleBasedReasoningEngine
from src.risk_engine import RiskEngine
from src.explanations import AlertExplainer
from src.evidence import EvidenceManager
from src.database import DatabaseManager
from src.report_generator import ReportGenerator
from src.ml_anomaly_model import ExamAnomalyDetector
from src.trainer import ExamModelTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FastAPIServer")

app = FastAPI(title="ExamGuard AI - High Performance Exam Hall Monitor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core engine singletons
settings = AppSettings()
db = DatabaseManager(settings.database_path)
evidence_mgr = EvidenceManager(settings)
reporter = ReportGenerator(db, settings)
seat_mapper = SeatMapper()
behavior_analyzer = BehaviorAnalyzer(settings)
reasoning_engine = RuleBasedReasoningEngine(settings)
risk_engine = RiskEngine(settings)

# Detector initialization
detector = ExamHallDetector()

# ML Anomaly model
anomaly_detector: Optional[ExamAnomalyDetector] = None
anom_model_path = os.path.join("models", "behavior_anomaly_detector.joblib")
if os.path.exists(anom_model_path):
    try:
        anomaly_detector = ExamAnomalyDetector.load(anom_model_path)
        logger.info("Loaded trained Behavioral ML Anomaly Detector.")
    except Exception as e:
        logger.warning(f"Could not load ML anomaly model: {e}")

# Global runtime session state
active_tracker = PersonTracker()
agent_memories: Dict[int, PersonAgentMemory] = {}
current_session_id = f"session_{int(time.time())}"
analysis_progress = {"status": "IDLE", "percent": 0.0, "current_time": 0.0, "total_time": 0.0, "video": ""}

# Cache video captures to avoid reopening files repeatedly
video_caps: Dict[str, cv2.VideoCapture] = {}

def get_video_cap(video_path: str) -> cv2.VideoCapture:
    if video_path not in video_caps or not video_caps[video_path].isOpened():
        cap = cv2.VideoCapture(video_path)
        video_caps[video_path] = cap
    return video_caps[video_path]

def resolve_video_path(video_name: str) -> str:
    # Check videos/
    p1 = os.path.join("videos", video_name)
    if os.path.exists(p1):
        return p1
    # Check sample_videos/
    p2 = os.path.join(settings.sample_videos_dir, video_name)
    if os.path.exists(p2):
        return p2
    # Check direct path
    if os.path.exists(video_name):
        return video_name
    raise HTTPException(status_code=404, detail=f"Video file '{video_name}' not found.")


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/api/status")
def get_system_status():
    """Returns engine status, active models, and statistics."""
    events = db.get_events(limit=500)
    high_risk_count = sum(1 for e in events if e.get("risk_category") in ["High", "Critical"])
    
    return {
        "status": "ONLINE",
        "models": {
            "yolo_detector": {
                "path": detector.model_path,
                "is_fine_tuned": detector.is_fine_tuned,
                "name": "Fine-Tuned Exam YOLOv8" if detector.is_fine_tuned else "Base YOLOv8n"
            },
            "ml_anomaly_detector": {
                "is_loaded": anomaly_detector is not None,
                "samples_trained": anomaly_detector.training_sample_count if anomaly_detector else 0,
                "algorithm": "IsolationForest + MinCovDet Ensemble"
            }
        },
        "stats": {
            "total_alerts": len(events),
            "high_risk_alerts": high_risk_count,
            "active_candidates": len(active_tracker.active_tracks),
            "total_desks": 18,
            "session_id": current_session_id
        }
    }


@app.get("/api/videos")
def list_available_videos():
    """Lists all available videos with metadata."""
    video_list = []
    
    # 1. Classroom videos
    v_dir = "videos"
    if os.path.exists(v_dir):
        for f in sorted(os.listdir(v_dir)):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                p = os.path.join(v_dir, f)
                meta = VideoInputHandler.inspect_video(p)
                video_list.append({
                    "name": f,
                    "category": "Classroom CCTV",
                    "path": p,
                    "resolution": f"{meta.width}x{meta.height}",
                    "fps": round(meta.fps, 1),
                    "duration_sec": round(meta.duration_seconds, 1),
                    "duration_str": FrameSampler.format_timestamp(meta.duration_seconds),
                    "size_mb": round(os.path.getsize(p) / (1024 * 1024), 2)
                })

    # 2. Benchmark video
    bm_path = os.path.join(settings.sample_videos_dir, "exam_hall_benchmark.mp4")
    if os.path.exists(bm_path):
        meta = VideoInputHandler.inspect_video(bm_path)
        video_list.append({
            "name": "exam_hall_benchmark.mp4",
            "category": "Synthetic Benchmark",
            "path": bm_path,
            "resolution": f"{meta.width}x{meta.height}",
            "fps": round(meta.fps, 1),
            "duration_sec": round(meta.duration_seconds, 1),
            "duration_str": FrameSampler.format_timestamp(meta.duration_seconds),
            "size_mb": round(os.path.getsize(bm_path) / (1024 * 1024), 2)
        })

    return {"videos": video_list}


class FrameAnalysisRequest(BaseModel):
    video_name: str
    timestamp_sec: float
    conf_threshold: Optional[float] = 0.30
    enable_ml: Optional[bool] = True


@app.post("/api/analyze-frame")
def analyze_single_frame(req: FrameAnalysisRequest):
    """
    Analyzes a specific frame at timestamp_sec from the video.
    Returns tracking bounding boxes, candidate risk states, and detected anomalies.
    """
    video_path = resolve_video_path(req.video_name)
    cap = get_video_cap(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = int(req.timestamp_sec * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        raise HTTPException(status_code=400, detail=f"Could not read frame at {req.timestamp_sec}s.")

    h, w = frame.shape[:2]
    current_t = float(req.timestamp_sec)
    timecode_str = FrameSampler.format_timestamp(current_t)

    # 1. Detection
    people_dets, phone_dets = detector.detect(frame, conf_override=req.conf_threshold)

    # 2. Tracking
    tracked_people = active_tracker.update(people_dets, current_t)

    # 3. Gathering clusters
    gathering_obs = behavior_analyzer.analyze_gathering_clusters(tracked_people, current_t)
    gathering_tids = {obs.tracking_id: obs for obs in gathering_obs}


    # 4. Seat mappings
    seat_statuses = seat_mapper.update_occupancy(tracked_people, current_t, 0.2, w, h)


    detections_output = []
    new_alerts = []
    max_risk_on_frame = 0.0

    all_positions = [t.foot_point for t in tracked_people]

    for track in tracked_people:
        tid = track.tracking_id
        if tid not in agent_memories:
            agent_memories[tid] = PersonAgentMemory(
                tracking_id=tid,
                assigned_seat_id=track.assigned_seat_id,
                first_seen_timestamp=current_t,
                last_seen_timestamp=current_t
            )

        mem = agent_memories[tid]
        mem.last_seen_timestamp = current_t
        sid = track.assigned_seat_id

        # Behaviors
        seat_status = seat_statuses.get(sid) if sid else None
        obs_seat = behavior_analyzer.analyze_seat_departure(track, seat_status, current_t)
        yaw_deg, is_turn, note = behavior_analyzer.estimate_head_pose(frame, track, seat_mapper)
        mem.record_head_turn(current_t, yaw_deg)
        obs_phone = behavior_analyzer.analyze_phone_detection(track, phone_dets, current_t, w, h)
        obs_gather = gathering_tids.get(tid)
        obs_move = behavior_analyzer.analyze_unusual_movement(track)

        active_obs = {"is_head_turning": is_turn}
        if obs_seat: active_obs["seat_departure"] = obs_seat.metrics
        if obs_phone: active_obs["phone"] = obs_phone.metrics
        if obs_gather: active_obs["gathering"] = obs_gather.metrics
        if obs_move: active_obs["unusual_movement"] = obs_move.metrics

        # Rules
        matched_rules = reasoning_engine.evaluate_all_rules(mem, current_t, active_obs)

        ml_score = None
        ml_explanation = None
        if req.enable_ml and anomaly_detector is not None and len(track.trajectory) >= 3:
            try:
                traj_pts = [(pt[3], pt[4]) for pt in track.trajectory]
                ts_list = [pt[0] for pt in track.trajectory]
                box_list = [track.bbox] * len(traj_pts)
                feat_vec = anomaly_detector.extract_feature_vector(
                    trajectory=traj_pts,
                    timestamps=ts_list,
                    head_poses=[0.0] * len(traj_pts),
                    bboxes=box_list,
                    all_student_positions=all_positions
                )
                ml_score = float(anomaly_detector.score_samples(feat_vec)[0])
                ml_explanation = anomaly_detector.explain_anomaly(feat_vec)
                mem.update_ml_anomaly(ml_score, ml_explanation)
            except Exception:
                ml_score = None

        if matched_rules:
            primary = matched_rules[0]
            ev_dur = primary.metrics.get("duration_sec", 4.0)
            ev_freq = primary.metrics.get("turn_count", primary.metrics.get("group_size", 1))

            risk_breakdown = risk_engine.compute_risk_score(
                event_type=primary.event_type,
                detection_confidence=track.detection_confidence,
                duration_sec=ev_dur,
                frequency_count=ev_freq,
                tracking_reliability=track.tracking_reliability,
                past_observations_count=len(mem.observation_buffer),
                ml_anomaly_score=ml_score
            )

            mem.current_risk_score = risk_breakdown.total_score
            mem._update_state_from_score(current_t, primary.reason_explanation)
            mem.consecutive_normal_ticks = 0

            if mem.current_state in [AgentState.SUSPICIOUS, AgentState.ALERT]:
                if evidence_mgr.should_save_evidence(tid, primary.event_type, current_t):
                    ev_rel_path = evidence_mgr.capture_evidence_frame(
                        frame=frame,
                        tracking_id=tid,
                        seat_id=sid,
                        event_type=primary.event_type,
                        timestamp_sec=current_t,
                        timecode_str=timecode_str,
                        risk_score=risk_breakdown.total_score,
                        target_bbox=track.bbox,
                        reason_text=primary.reason_explanation
                    )
                    alert = AlertExplainer.generate_alert(
                        event_type=primary.event_type,
                        tracking_id=tid,
                        seat_id=sid,
                        timestamp_sec=current_t,
                        duration_sec=ev_dur,
                        detection_confidence=track.detection_confidence,
                        risk_breakdown=risk_breakdown,
                        evidence_path=ev_rel_path,
                        metrics=primary.metrics
                    )
                    db.insert_event(
                        session_id=current_session_id,
                        video_timestamp_sec=current_t,
                        timecode_str=timecode_str,
                        tracking_id=tid,
                        seat_id=sid,
                        event_type=alert.event_type,
                        event_display_name=alert.event_display_name,
                        confidence=track.detection_confidence,
                        duration_seconds=ev_dur,
                        risk_score=alert.risk_score,
                        risk_category=alert.risk_category,
                        reason=alert.human_readable_explanation,
                        evidence_path=ev_rel_path
                    )
                    new_alerts.append(alert.to_dict())

        else:
            mem.consecutive_normal_ticks += 1
            if mem.consecutive_normal_ticks > 5:
                mem.apply_risk_decay(0.5, settings.risk_decay_per_second)

        max_risk_on_frame = max(max_risk_on_frame, mem.current_risk_score)

        detections_output.append({
            "tracking_id": tid,
            "seat_id": sid,
            "bbox": list(track.bbox),
            "confidence": round(track.detection_confidence, 2),
            "foot_point": [round(track.foot_point[0], 1), round(track.foot_point[1], 1)],
            "velocity": round(track.velocity_px_per_sec, 1),
            "state": mem.current_state.value,
            "risk_score": round(mem.current_risk_score, 1),
            "ml_anomaly_score": round(ml_score, 2) if ml_score is not None else None,
            "ml_reasons": ml_explanation.get("primary_reasons", []) if ml_explanation else []
        })

    # Seat regions for overlay
    seat_regions_out = []
    for sid, sreg in seat_mapper.seats.items():
        px_box = sreg.to_pixel_box(w, h)
        seat_regions_out.append({
            "seat_id": sid,
            "label": sreg.label,
            "box": list(px_box),
            "is_occupied": sid in seat_statuses and seat_statuses[sid].get("is_occupied", True)
        })


    return {
        "timestamp_sec": current_t,
        "timecode": timecode_str,
        "frame_dimensions": {"width": w, "height": h},
        "detections": detections_output,
        "phones": [{"bbox": list(p.bbox), "conf": round(p.confidence, 2)} for p in phone_dets],
        "seats": seat_regions_out,
        "new_alerts": new_alerts,
        "stats": {
            "total_tracked": len(tracked_people),
            "max_risk_score": round(max_risk_on_frame, 1),
            "active_alerts_count": len(new_alerts)
        }
    }


@app.get("/api/alerts")
def get_alerts(
    seat_id: Optional[int] = None,
    risk_category: Optional[str] = None,
    event_type: Optional[str] = None,
    review_status: Optional[str] = None,
    limit: int = 50
):
    """Retrieves persisted anomaly alerts with filters."""
    events = db.get_events(
        session_id=None,
        risk_category=risk_category,
        event_type=event_type,
        seat_id=seat_id,
        review_status=review_status,
        limit=limit
    )
    return {"alerts": events}


class AlertReviewRequest(BaseModel):
    event_id: int
    review_status: str  # ACKNOWLEDGED, UNDER_REVIEW, FALSE_POSITIVE, RESOLVED
    notes: Optional[str] = ""


@app.post("/api/alerts/review")
def review_alert(req: AlertReviewRequest):
    """Updates the human review status for an event."""
    success = db.update_review_status(req.event_id, req.review_status, req.notes)
    if not success:
        raise HTTPException(status_code=404, detail="Alert record not found.")
    return {"status": "SUCCESS", "event_id": req.event_id, "new_status": req.review_status}


@app.get("/api/chart-data")
def get_chart_data():
    """Returns aggregated timeline datapoints for the smooth portfolio-style chart."""
    events = db.get_events(limit=300)
    
    # Generate continuous timeline intervals (every 2 seconds up to max timestamp)
    max_ts = 20.0
    if events:
        max_ts = max(max(e["video_timestamp_sec"] for e in events), 20.0)

    time_steps = np.arange(0.0, max_ts + 2.0, 2.0)
    datapoints = []

    for t in time_steps:
        # Find events around this window
        window_events = [e for e in events if abs(e["video_timestamp_sec"] - t) <= 2.0]
        max_risk = max([e["risk_score"] for e in window_events], default=15.0)
        # Add smooth baseline fluctuation
        display_val = max(12.0, min(95.0, max_risk + np.sin(t * 0.4) * 3.0))
        
        datapoints.append({
            "timestamp_sec": float(round(t, 1)),
            "timecode": FrameSampler.format_timestamp(t),
            "risk_score": float(round(display_val, 1)),
            "alert_count": len(window_events),
            "label": f"{FrameSampler.format_timestamp(t)}"
        })

    return {
        "datapoints": datapoints,
        "max_risk": max([p["risk_score"] for p in datapoints], default=0.0),
        "total_events": len(events)
    }


@app.get("/api/evidence/{image_name}")
def get_evidence_image(image_name: str):
    """Serves high-resolution forensic evidence image files."""
    path = os.path.join(settings.evidence_dir, image_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Evidence image '{image_name}' not found.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/export-csv")
def export_csv_report():
    """Generates and downloads the official CSV examination audit report."""
    df = db.get_events_dataframe()
    if df.empty:
        # Create empty template
        df = pd.DataFrame(columns=[
            "event_id", "session_id", "video_timestamp_sec", "timecode_str",
            "seat_id", "tracking_id", "event_type", "confidence", "risk_score",
            "risk_category", "review_status", "explanation"
        ])
    
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=exam_audit_report_{int(time.time())}.csv"}
    )


class AIQueryRequest(BaseModel):
    query: str


@app.post("/api/ask-ai")
def ask_ai_assistant(req: AIQueryRequest):
    """
    NLP assistant that answers invigilator questions about current hall status,
    candidate behaviors, highest risk seats, or rules.
    """
    q = req.query.lower().strip()
    events = db.get_events(limit=100)

    if "highest risk" in q or "worst" in q or "critical" in q:
        if not events:
            return {"answer": "Currently, no high-risk anomalies have been logged in the examination hall."}
        sorted_e = sorted(events, key=lambda x: x["risk_score"], reverse=True)
        top = sorted_e[0]
        return {
            "answer": (
                f"Candidate at Seat #{top.get('seat_id') or top.get('tracking_id')} registered the highest risk score: "
                f"{top.get('risk_score', 0)}/100 ({top.get('risk_category', 'Review')}) due to '{top.get('event_display_name', 'Event')}' at {top.get('timecode_str', '00:00')}. "
                f"Reason: {top.get('reason') or top.get('explanation') or 'Human verification recommended.'}"
            )
        }


    if "phone" in q or "mobile" in q:
        phone_events = [e for e in events if "phone" in e["event_type"].lower()]
        if phone_events:
            return {
                "answer": (
                    f"Found {len(phone_events)} mobile phone related anomaly incidents. "
                    f"Earliest detection at {phone_events[0]['timecode_str']} at Seat #{phone_events[0]['seat_id']}. "
                    "All incidents are flagged for human invigilator verification."
                )
            }
        return {"answer": "No mobile phone devices have been detected in candidate workspaces so far."}

    if "seat" in q:
        # Check if a specific seat number was mentioned
        import re
        match = re.search(r"seat\s*#?\s*(\d+)", q)
        if match:
            s_num = int(match.group(1))
            seat_evs = [e for e in events if e.get("seat_id") == s_num]
            if seat_evs:
                latest = seat_evs[0]
                return {
                    "answer": (
                        f"Seat #{s_num} has {len(seat_evs)} logged anomaly events. "
                        f"Most recent: '{latest['event_display_name']}' at {latest['timecode_str']} "
                        f"with Risk {latest['risk_score']}/100 ({latest['risk_category']}). Review status: {latest['review_status']}."
                    )
                }
            return {"answer": f"Seat #{s_num} has remained normal with no behavioral anomaly violations recorded."}

    # Default overview
    high_count = sum(1 for e in events if e.get("risk_category") in ["High", "Critical"])
    return {
        "answer": (
            f"ExamGuard AI Monitoring Status: {len(events)} total incidents logged ({high_count} requiring priority review). "
            f"Active models: {('Fine-Tuned YOLOv8' if detector.is_fine_tuned else 'Base YOLOv8')} and "
            f"{('Behavioral ML Anomaly Detector' if anomaly_detector else 'Rule-Based Engine')}. "
            "All findings prioritize authorized human verification and do not constitute automated cheating verdicts."
        )
    }


class SwitchModelRequest(BaseModel):
    use_fine_tuned: bool
    enable_ml_anomaly: bool


@app.post("/api/switch-model")
def switch_model_configuration(req: SwitchModelRequest):
    """Switches active object detector and ML anomaly model at runtime."""
    global detector, anomaly_detector
    
    desired_weights = "models/yolov8n_exam.pt" if req.use_fine_tuned else "models/yolov8n.pt"
    if not os.path.exists(desired_weights):
        desired_weights = "yolov8n.pt"

    detector = ExamHallDetector(model_path=desired_weights)
    
    if req.enable_ml_anomaly and os.path.exists("models/behavior_anomaly_detector.joblib"):
        anomaly_detector = ExamAnomalyDetector.load("models/behavior_anomaly_detector.joblib")
    else:
        anomaly_detector = None

    return {
        "status": "UPDATED",
        "detector_path": detector.model_path,
        "is_fine_tuned": detector.is_fine_tuned,
        "ml_anomaly_active": anomaly_detector is not None
    }


# Serve video files for direct HTML5 range streaming
@app.get("/api/video-file/{folder}/{filename}")
def stream_video_file(folder: str, filename: str):
    """Direct video streaming endpoint for HTML5 video elements."""
    if folder == "videos":
        p = os.path.join("videos", filename)
    elif folder == "samples":
        p = os.path.join(settings.sample_videos_dir, filename)
    else:
        raise HTTPException(status_code=400, detail="Invalid folder")
    
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(p, media_type="video/mp4")


# Serve the Web UI static files
app.mount("/static", StaticFiles(directory="web"), name="static")

@app.get("/")
def serve_index_html():
    return FileResponse("web/index.html")


if __name__ == "__main__":
    import argparse
    import socket
    import uvicorn

    parser = argparse.ArgumentParser(description="ExamGuard AI Ultra-Fast Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on (default: 8000)")
    args = parser.parse_args()

    def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) == 0

    target_port = args.port
    if is_port_in_use(target_port):
        print(f"\n⚠️  [NOTICE] Port {target_port} is currently in use by another process.")
        print(f"👉 To terminate any existing process on port {target_port}:")
        print(f"   lsof -ti:{target_port} | xargs kill -9\n")
        # Find next free port
        for p in range(target_port + 1, target_port + 50):
            if not is_port_in_use(p):
                print(f"✅ Auto-switched to available port: {p}")
                target_port = p
                break

    print(f"\n{'='*65}")
    print(f"🚀 ExamGuard AI Ultra-Fast Server is launching!")
    print(f"🔗 Open your browser at: http://localhost:{target_port}")
    print(f"🔗 Network URL:          http://127.0.0.1:{target_port}")
    print(f"{'='*65}\n")

    uvicorn.run("server:app", host=args.host, port=target_port, reload=False, workers=1)

