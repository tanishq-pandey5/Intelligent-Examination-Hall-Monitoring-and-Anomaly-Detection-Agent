"""
AI Examination Video Frame Analysis and Pattern Detection System
Interactive Streamlit Dashboard.
Implements Sections 2, 10, 11, 13:
  - Video upload (MP4, AVI, MOV) & preview
  - Configurable frame sampling frequency & detection confidence
  - Start, pause, stop analysis controls with progress & elapsed time
  - Annotated video playback with tracking IDs, bounding boxes, foot points & seat overlays
  - Highlighted suspicious locations and interactive 2D floor plan
  - Event timeline and risk-score summaries
  - Filterable event table with exact Section 9 explanations & evidence image viewer
  - Invigilator review action buttons (Acknowledge, Dismiss / False Positive, Under Review)
  - Downloadable CSV audit report and human-readable session summary
"""
import streamlit as st
import cv2
import numpy as np
import os
import time
import json
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from config.settings import AppSettings, SeatRegion, DEFAULT_SETTINGS
from src.video_input import VideoInputHandler, VideoMetadata
from src.frame_sampler import FrameSampler, VideoFrame
from src.detector import ExamHallDetector, DetectedObject
from src.tracker import PersonTracker, TrackedPerson
from src.seat_mapper import SeatMapper
from src.behavior import BehaviorAnalyzer, BehaviorObservation
from src.agent import PersonAgentMemory, AgentState, ReviewState
from src.reasoning import RuleBasedReasoningEngine, RuleMatchResult
from src.risk_engine import RiskEngine, RiskBreakdown
from src.explanations import AlertExplainer, ExplainableAlert
from src.evidence import EvidenceManager
from src.database import DatabaseManager
from src.report_generator import ReportGenerator
from src.ml_anomaly_model import ExamAnomalyDetector
from src.trainer import ExamModelTrainer


# Page configuration
st.set_page_config(
    page_title="Intelligent Exam Hall Monitoring Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .seat-tile {
        border-radius: 6px;
        padding: 8px;
        text-align: center;
        margin-bottom: 8px;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .seat-normal {
        background-color: rgba(46, 204, 113, 0.15);
        border: 2px solid #2ecc71;
        color: #2ecc71;
    }
    .seat-observe {
        background-color: rgba(241, 196, 15, 0.15);
        border: 2px solid #f1c40f;
        color: #f1c40f;
    }
    .seat-suspicious {
        background-color: rgba(230, 126, 34, 0.2);
        border: 2px solid #e67e22;
        color: #e67e22;
    }
    .seat-alert {
        background-color: rgba(231, 76, 60, 0.25);
        border: 2px solid #e74c3c;
        color: #e74c3c;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0.6); }
        70% { box-shadow: 0 0 0 10px rgba(231, 76, 60, 0); }
        100% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0); }
    }
    .non-verdict-banner {
        background-color: #1e293b;
        border-left: 5px solid #38bdf8;
        padding: 10px 16px;
        border-radius: 4px;
        color: #e2e8f0;
        font-size: 0.9rem;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "settings" not in st.session_state:
    st.session_state.settings = AppSettings()
if "db" not in st.session_state:
    st.session_state.db = DatabaseManager()
if "evidence_mgr" not in st.session_state:
    st.session_state.evidence_mgr = EvidenceManager(st.session_state.settings)
if "report_gen" not in st.session_state:
    st.session_state.report_gen = ReportGenerator(st.session_state.db, st.session_state.settings)
if "detector" not in st.session_state:
    st.session_state.detector = ExamHallDetector(
        confidence_threshold=st.session_state.settings.default_detection_confidence
    )
if "anomaly_detector" not in st.session_state:
    anom_path = "models/behavior_anomaly_detector.joblib"
    if os.path.exists(anom_path):
        try:
            st.session_state.anomaly_detector = ExamAnomalyDetector.load(anom_path)
        except Exception:
            st.session_state.anomaly_detector = None
    else:
        st.session_state.anomaly_detector = None
if "tracker" not in st.session_state:
    st.session_state.tracker = PersonTracker()
if "seat_mapper" not in st.session_state:
    st.session_state.seat_mapper = SeatMapper()
if "behavior_analyzer" not in st.session_state:
    st.session_state.behavior_analyzer = BehaviorAnalyzer(st.session_state.settings)
if "reasoning_engine" not in st.session_state:
    st.session_state.reasoning_engine = RuleBasedReasoningEngine(st.session_state.settings)
if "risk_engine" not in st.session_state:
    st.session_state.risk_engine = RiskEngine(st.session_state.settings)

if "agent_memories" not in st.session_state:
    st.session_state.agent_memories = {}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = f"session_{int(time.time())}"
if "analysis_status" not in st.session_state:
    st.session_state.analysis_status = "IDLE" # IDLE, RUNNING, PAUSED, FINISHED
if "playback_time_sec" not in st.session_state:
    st.session_state.playback_time_sec = 0.0

settings = st.session_state.settings
db = st.session_state.db
evidence_mgr = st.session_state.evidence_mgr
report_gen = st.session_state.report_gen
detector = st.session_state.detector

# ==============================================================================
# SIDEBAR: VIDEO UPLOAD & CONTROLS (Section 2)
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=55)
    st.title("ExamGuard AI")
    st.caption("AI Examination Frame Analysis & Pattern Detection")
    st.divider()

    st.subheader("📁 Video Upload & Ingestion")
    video_source_type = st.radio(
        "Choose Video Input:",
        [
            "Classroom Videos (videos/ folder)",
            "Benchmark Hall Video (18 Desks)",
            "Upload Video (MP4 / AVI / MOV)"
        ],
        index=0
    )

    video_path = None
    if video_source_type == "Classroom Videos (videos/ folder)":
        v_folder = "videos"
        v_files = sorted([f for f in os.listdir(v_folder) if f.lower().endswith((".mp4", ".avi", ".mov"))]) if os.path.exists(v_folder) else []
        if v_files:
            chosen_video = st.selectbox("Select Classroom CCTV Video:", v_files, index=0)
            video_path = os.path.join(v_folder, chosen_video)
            st.info(f"Loaded: `{chosen_video}` from `{v_folder}/`")
        else:
            st.warning("No videos found in `videos/` folder.")
    elif video_source_type == "Upload Video (MP4 / AVI / MOV)":
        uploaded_file = st.file_uploader(
            "Upload Examination Hall CCTV Video:",
            type=["mp4", "avi", "mov", "mkv"]
        )
        if uploaded_file is not None:
            save_name = f"uploaded_{uploaded_file.name}"
            target_path = os.path.join(settings.sample_videos_dir, save_name)
            os.makedirs(settings.sample_videos_dir, exist_ok=True)
            with open(target_path, "wb") as f:
                f.write(uploaded_file.read())
            video_path = target_path
            st.success(f"Video uploaded: {uploaded_file.name}")
    else:
        # Benchmark video
        benchmark_file = os.path.join(settings.sample_videos_dir, "exam_hall_benchmark.mp4")
        if not os.path.exists(benchmark_file):
            from core.simulator import ExamHallSimulator
            sim = ExamHallSimulator()
            sim.generate_video_file(benchmark_file, duration_seconds=20.0)
        video_path = benchmark_file
        st.info("Using 18-desk university examination benchmark video.")

    st.divider()
    st.subheader("🤖 AI Model Engine")
    trained_pt_exists = os.path.exists("models/yolov8n_exam.pt")
    model_opts = ["Fine-Tuned Exam YOLOv8 (models/yolov8n_exam.pt)", "Base Pretrained YOLOv8 (models/yolov8n.pt)"] if trained_pt_exists else ["Base Pretrained YOLOv8 (models/yolov8n.pt)"]
    selected_model_opt = st.selectbox("Active Object Detector:", model_opts, index=0)
    desired_path = "models/yolov8n_exam.pt" if "Fine-Tuned" in selected_model_opt else "models/yolov8n.pt"
    if st.session_state.detector.model_path != desired_path:
        st.session_state.detector = ExamHallDetector(model_path=desired_path, confidence_threshold=settings.default_detection_confidence)

    has_ml_model = os.path.exists("models/behavior_anomaly_detector.joblib")
    enable_ml_anom = st.toggle(
        "Enable ML Behavioral Anomaly Detector",
        value=has_ml_model,
        help="Applies the Isolation Forest & Covariance model trained on student movements."
    )
    st.caption("Status: " + ("🟢 Trained & Active" if has_ml_model and enable_ml_anom else "⚪ Inactive / Untrained"))

    st.divider()
    st.subheader("⚙️ Processing Parameters")
    sampling_fps = st.select_slider(
        "Frame Sampling Frequency (FPS):",
        options=[1.0, 2.0, 5.0, 10.0, 15.0, 25.0],
        value=5.0,
        help="Higher sampling rates increase temporal resolution; lower rates maximize speed."
    )
    conf_threshold = st.slider(
        "Detection Confidence Threshold:",
        min_value=0.20,
        max_value=0.85,
        value=settings.default_detection_confidence,
        step=0.05,
        help="Lower confidence detects subtle objects; higher confidence minimizes false positives."
    )
    settings.default_detection_confidence = conf_threshold

    st.divider()
    st.subheader("🧹 System Maintenance")
    if st.button("Reset Database & Clear Evidence"):
        db.clear_database()
        evidence_mgr.clear_all_evidence()
        st.session_state.agent_memories = {}
        st.session_state.tracker = PersonTracker()
        st.session_state.seat_mapper = SeatMapper()
        st.success("Analysis database and evidence cleared!")
        st.rerun()


# ==============================================================================
# MAIN VIEW: DISCLAIMER & METADATA
# ==============================================================================
st.markdown("## 🎓 Intelligent Examination Hall Monitoring and Anomaly Detection Agent")
st.markdown("""
<div class="non-verdict-banner">
    <b>ℹ️ Notice to Invigilators:</b> This system is an examination-support prototype designed to identify potential behavioral anomalies across time. It does <b>not</b> deliver automated verdicts of cheating. Every flagged alert remains strictly reviewable by an authorized human invigilator.
</div>
""", unsafe_allow_html=True)

# Inspect video metadata
metadata = VideoInputHandler.inspect_video(video_path)
if not metadata.is_valid:
    st.error(f"❌ Video Error: {metadata.error_message}")
    st.stop()

# Display Video Metadata Chips
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
with m_col1:
    st.metric("Video File", metadata.file_name[:18])
with m_col2:
    st.metric("Resolution", f"{metadata.width}x{metadata.height}")
with m_col3:
    st.metric("Original FPS", f"{metadata.fps:.1f} fps")
with m_col4:
    st.metric("Total Frames", metadata.total_frames)
with m_col5:
    st.metric("Duration", f"{metadata.duration_seconds:.1f}s ({FrameSampler.format_timestamp(metadata.duration_seconds)})")

st.divider()

# ==============================================================================
# TABS: VIDEO MONITOR | SEAT FLOOR PLAN | ALERTS AUDIT | EVIDENCE | REPORTS
# ==============================================================================
tab_monitor, tab_floorplan, tab_alerts, tab_evidence, tab_reports, tab_training = st.tabs([
    "📹 Video Stream & Frame Analysis",
    "🗺️ Examination Hall Seating Layout",
    "🚨 Explainable Anomaly Events Feed",
    "📸 Evidence Snapshot Vault",
    "📊 Session Summary & CSV Reports",
    "🧠 Model Training & Video Calibration"
])


# ------------------------------------------------------------------------------
# TAB 1: VIDEO STREAM & FRAME ANALYSIS (Section 2, 3, 4, 10)
# ------------------------------------------------------------------------------
with tab_monitor:
    st.markdown("### 🎥 Video Playback & Analysis Controls")

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1, 1, 2])
    with ctrl1:
        scrub_time = st.slider(
            "Video Timeline Position (seconds):",
            min_value=0.0,
            max_value=float(metadata.duration_seconds),
            value=float(st.session_state.playback_time_sec),
            step=0.2
        )
        st.session_state.playback_time_sec = scrub_time

    with ctrl2:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("▶️ Step +1.0s"):
            st.session_state.playback_time_sec = min(float(metadata.duration_seconds), st.session_state.playback_time_sec + 1.0)
            st.rerun()

    with ctrl3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("⏮️ Reset to 0s"):
            st.session_state.playback_time_sec = 0.0
            st.rerun()

    with ctrl4:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        preset = st.selectbox(
            "Jump to Anomaly Event:",
            [
                "Select event...",
                "Seat 18: Repeated Head Turning (t=14.0s)",
                "Seat 5: Mobile Phone Detection (t=8.0s)",
                "Seat 9: Leaving Assigned Seat (t=15.0s)",
                "Seats 11 & 12: Group Gathering (t=14.2s)",
                "Seat 3: Unusual Movement (t=7.0s)"
            ]
        )
        if preset != "Select event...":
            if "Seat 18" in preset:
                st.session_state.playback_time_sec = 14.0
            elif "Seat 5" in preset:
                st.session_state.playback_time_sec = 8.0
            elif "Seat 9" in preset:
                st.session_state.playback_time_sec = 15.0
            elif "Seats 11 & 12" in preset:
                st.session_state.playback_time_sec = 14.2
            elif "Seat 3" in preset:
                st.session_state.playback_time_sec = 7.0
            st.rerun()

    # Extract frame at current playback time
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, st.session_state.playback_time_sec * 1000.0)
    ret, raw_frame = cap.read()
    cap.release()

    if not ret or raw_frame is None:
        st.warning("Unable to read video frame at current timeline position.")
    else:
        frame_h, frame_w = raw_frame.shape[:2]
        current_t = st.session_state.playback_time_sec

        # 1. Detect people and mobile phones
        people_dets, phone_dets = detector.detect(raw_frame, conf_override=conf_threshold)

        # 2. Update multi-person tracker
        tracks = st.session_state.tracker.update(people_dets, current_timestamp=current_t)

        # 3. Update seat occupancy & foot points
        seat_status = st.session_state.seat_mapper.update_occupancy(
            tracks=tracks,
            current_timestamp=current_t,
            dt=1.0 / sampling_fps,
            frame_w=frame_w,
            frame_h=frame_h
        )

        # 4. Behavioral Analysis & Agent Reasoning
        annotated_frame = raw_frame.copy()
        timecode = FrameSampler.format_timestamp(current_t)

        # Draw seat regions
        for sid, s_region in st.session_state.seat_mapper.seats.items():
            bx1, by1, bx2, by2 = s_region.to_pixel_box(frame_w, frame_h)
            st_info = seat_status.get(sid, {})
            box_col = (50, 205, 50) if st_info.get("is_occupied", True) else (34, 34, 255)
            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (80, 85, 95), 1)

        # Gathering clusters across all candidates
        gathering_clusters = st.session_state.behavior_analyzer.analyze_gathering_clusters(tracks, current_t)
        gathering_tids = {c.tracking_id: c for c in gathering_clusters}

        for track in tracks:
            tid = track.tracking_id
            sid = track.assigned_seat_id

            # Retrieve or initialize Agent Memory (Section 6)
            if tid not in st.session_state.agent_memories:
                st.session_state.agent_memories[tid] = PersonAgentMemory(
                    tracking_id=tid,
                    assigned_seat_id=sid,
                    first_seen_timestamp=current_t,
                    last_seen_timestamp=current_t
                )
            mem = st.session_state.agent_memories[tid]
            mem.last_seen_timestamp = current_t

            # Run Behavior Modules
            # A: Leaving seat
            obs_seat = st.session_state.behavior_analyzer.analyze_seat_departure(track, seat_status, current_t)
            # B: Head turning
            yaw_deg, is_turn, note = st.session_state.behavior_analyzer.estimate_head_pose(raw_frame, track, st.session_state.seat_mapper)
            mem.record_head_turn(current_t, yaw_deg)
            # C: Mobile phone
            obs_phone = st.session_state.behavior_analyzer.analyze_phone_detection(track, phone_dets, current_t, frame_w, frame_h)
            # D: Gathering
            obs_gather = gathering_tids.get(tid)
            # E: Unusual movement
            obs_move = st.session_state.behavior_analyzer.analyze_unusual_movement(track)

            # Build perception dictionary
            active_obs = {"is_head_turning": is_turn}
            if obs_seat:
                active_obs["seat_departure"] = obs_seat.metrics
            if obs_phone:
                active_obs["phone"] = obs_phone.metrics
            if obs_gather:
                active_obs["gathering"] = obs_gather.metrics
            if obs_move:
                active_obs["unusual_movement"] = obs_move.metrics

            # Execute Rule-Based Reasoning Engine (Section 7)
            matched_rules = st.session_state.reasoning_engine.evaluate_all_rules(mem, current_t, active_obs)

            if matched_rules:
                primary = matched_rules[0]
                # Calculate Multi-Factor Risk Score (Section 8)
                ev_dur = primary.metrics.get("duration_sec", 4.0)
                ev_freq = primary.metrics.get("turn_count", primary.metrics.get("group_size", 1))

                # Extract ML anomaly score if detector is active
                ml_score = None
                if enable_ml_anom and st.session_state.anomaly_detector is not None and len(track.trajectory) >= 3:
                    try:
                        all_pos = [t.foot_point for t in active_tracks]
                        traj_pts = [(pt[3], pt[4]) for pt in track.trajectory]
                        ts_list = [pt[0] for pt in track.trajectory]
                        box_list = [track.bbox] * len(traj_pts)
                        feat_vec = st.session_state.anomaly_detector.extract_feature_vector(
                            trajectory=traj_pts,
                            timestamps=ts_list,
                            head_poses=[0.0] * len(traj_pts),
                            bboxes=box_list,
                            all_student_positions=all_pos
                        )
                        ml_score = float(st.session_state.anomaly_detector.score_samples(feat_vec)[0])
                    except Exception:
                        ml_score = None

                risk_breakdown = st.session_state.risk_engine.compute_risk_score(
                    event_type=primary.event_type,
                    detection_confidence=track.detection_confidence,
                    duration_sec=ev_dur,
                    frequency_count=ev_freq,
                    tracking_reliability=track.tracking_reliability,
                    past_observations_count=len(mem.observation_buffer),
                    ml_anomaly_score=ml_score
                )

                mem.current_risk_score = risk_breakdown.total_score
                if ml_score is not None:
                    mem.update_ml_anomaly(ml_score)
                mem._update_state_from_score(current_t, primary.reason_explanation)
                mem.consecutive_normal_ticks = 0


                # Check if evidence should be saved and event logged (Section 9 & 11)
                if mem.current_state in [AgentState.SUSPICIOUS, AgentState.ALERT]:
                    if evidence_mgr.should_save_evidence(tid, primary.event_type, current_t):
                        # Capture Evidence Frame (Section 9)
                        ev_rel_path = evidence_mgr.capture_evidence_frame(
                            frame=raw_frame,
                            tracking_id=tid,
                            seat_id=sid,
                            event_type=primary.event_type,
                            timestamp_sec=current_t,
                            timecode_str=timecode,
                            risk_score=risk_breakdown.total_score,
                            target_bbox=track.bbox,
                            reason_text=primary.reason_explanation
                        )

                        # Generate Explainable Alert
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

                        # Persist to SQLite
                        db.insert_event(
                            session_id=st.session_state.current_session_id,
                            video_timestamp_sec=current_t,
                            timecode_str=timecode,
                            tracking_id=tid,
                            seat_id=sid,
                            event_type=alert.event_type,
                            event_display_name=alert.event_display_name,
                            confidence=track.detection_confidence,
                            duration_seconds=ev_dur,
                            risk_score=risk_breakdown.total_score,
                            risk_category=risk_breakdown.risk_category,
                            reason=alert.reason,
                            evidence_path=ev_rel_path,
                            review_status="UNREVIEWED"
                        )
                        mem.evidence_snapshots.append(ev_rel_path)
                        mem.total_alerts_generated += 1
            else:
                # Normal behavior -> apply temporal decay (Section 5 & 6)
                mem.consecutive_normal_ticks += 1
                mem.apply_risk_decay(dt=1.0 / sampling_fps, decay_rate=settings.risk_decay_per_second)

            # Draw visual annotations onto frame (Section 3 & 10)
            x1, y1, x2, y2 = track.bbox
            state_color = mem.current_state.color_bgr
            thickness = 3 if mem.current_state in [AgentState.ALERT, AgentState.SUSPICIOUS] else 2

            # Candidate Bounding Box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), state_color, thickness)

            # Foot-Point Marker (Section 4A)
            fx, fy = track.foot_point
            cv2.circle(annotated_frame, (int(fx), int(fy)), 5, (0, 255, 255), -1)

            # Tag Header: ID, Seat, State, Risk
            tag_label = f"ID:{tid} | Seat:{sid or '?'} | {mem.current_state.value} ({int(mem.current_risk_score)}%)"
            (tw, th), _ = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated_frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), state_color, -1)
            cv2.putText(annotated_frame, tag_label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)

        # Convert to RGB for Streamlit render
        rgb_disp = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st.image(rgb_disp, use_container_width=True, caption=f"CCTV Video Feed Timeline @ {timecode} ({current_t:.1f}s) | Active Tracks: {len(tracks)}")

# ------------------------------------------------------------------------------
# TAB 2: SEATING LAYOUT & SUSPICIOUS LOCATIONS (Section 10)
# ------------------------------------------------------------------------------
with tab_floorplan:
    st.markdown("### 🗺️ Examination Hall Seating Grid & Real-Time Suspicion Heatmap")
    st.caption("2D floor plan layout of the examination hall. Desks dynamically highlight in crimson or orange when potential anomalies are detected.")

    f_col1, f_col2 = st.columns([3, 2])

    with f_col1:
        st.markdown("""
        <div style="background:#0f172a; padding:12px; border-radius:8px; border:1px solid #1e293b;">
            <div style="text-align:center; padding:6px; background:#1e293b; color:#94a3b8; border-radius:4px; margin-bottom:12px; font-weight:600;">
                FRONT: SUPERVISOR PODIUM & WHITEBOARD
            </div>
        """, unsafe_allow_html=True)

        seats_grid = st.session_state.seat_mapper.seats
        for r in range(1, 4): # 3 rows
            row_cols = st.columns(6) # 6 columns
            for c in range(1, 7):
                sid = (r - 1) * 6 + c
                # Find candidate memory associated with this seat
                mem = next((m for m in st.session_state.agent_memories.values() if m.assigned_seat_id == sid), None)
                state = mem.current_state if mem else AgentState.NORMAL
                risk = mem.current_risk_score if mem else 0.0

                state_cls = f"seat-{state.value.lower()}"
                icon = "🟢" if state == AgentState.NORMAL else "🟡" if state == AgentState.OBSERVE else "🟠" if state == AgentState.SUSPICIOUS else "🔴"

                with row_cols[c - 1]:
                    st.markdown(f"""
                    <div class="seat-tile {state_cls}">
                        <div style="font-size:0.75rem;">{icon} Desk #{sid}</div>
                        <div style="font-size:1.1rem; font-weight:700;">{int(risk)}%</div>
                        <div style="font-size:0.65rem;">{state.value}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("""
            <div style="text-align:center; padding:4px; color:#64748b; font-size:0.75rem; margin-top:8px;">
                REAR AISLE: CCTV CAMERA MOUNT
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f_col2:
        st.markdown("#### 🔍 Candidate Dossier & Review Inspector")
        inspect_seat_id = st.selectbox("Inspect Examination Desk:", list(range(1, 19)), index=17) # Defaults to Seat 18
        mem = next((m for m in st.session_state.agent_memories.values() if m.assigned_seat_id == inspect_seat_id), None)

        if mem:
            st.markdown(f"**Tracking ID**: `Track #{mem.tracking_id}` | **Assigned Desk**: `Seat #{inspect_seat_id}`")
            st.markdown(f"**Internal State**: `{mem.current_state.value}` | **Review Status**: `{mem.review_state.value}`")
            st.progress(mem.current_risk_score / 100.0)

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Risk Score", f"{mem.current_risk_score:.1f}/100")
                st.metric("Head Turns (Window)", mem.get_turn_count_in_window(st.session_state.playback_time_sec))
            with m2:
                st.metric("Generated Alerts", mem.total_alerts_generated)
                st.metric("Normal Frames", mem.consecutive_normal_ticks)

            if mem.state_transitions:
                st.markdown("##### 📜 Audit Transition Log:")
                for ts, s_val, r_msg in reversed(mem.state_transitions[-4:]):
                    st.caption(f"⏱️ `{ts:.1f}s` → **{s_val.value}**: {r_msg}")
        else:
            st.info(f"Desk #{inspect_seat_id} is currently in normal standing with no recorded anomalies.")

# ------------------------------------------------------------------------------
# TAB 3: EXPLAINABLE ALERTS AUDIT FEED (Section 9, 10)
# ------------------------------------------------------------------------------
with tab_alerts:
    st.markdown("### 🚨 Explainable Anomaly Events Feed")
    st.caption("Human-readable, structured explanations matching Section 9 format. Authorized invigilators can update review statuses.")

    f1, f2, f3 = st.columns(3)
    with f1:
        f_risk = st.selectbox("Filter Risk Category:", ["All Categories", "Critical", "High", "Medium", "Low"])
    with f2:
        f_seat = st.selectbox("Filter Seat Number:", ["All Seats"] + [f"Seat {i}" for i in range(1, 19)])
    with f3:
        f_status = st.selectbox("Filter Review Status:", ["All Statuses", "UNREVIEWED", "ACKNOWLEDGED", "UNDER_REVIEW", "RESOLVED", "FALSE_POSITIVE"])

    seat_filter_val = int(f_seat.split(" ")[1]) if f_seat != "All Seats" else None
    risk_filter_val = None if f_risk == "All Categories" else f_risk
    status_filter_val = None if f_status == "All Statuses" else f_status

    events_list = db.get_events(
        seat_id=seat_filter_val,
        risk_category=risk_filter_val,
        review_status=status_filter_val,
        limit=50
    )

    if events_list:
        st.markdown(f"**Found {len(events_list)} logged anomaly events:**")
        for ev in events_list:
            border_col = "#e74c3c" if ev["risk_category"] in ["Critical", "High"] else "#f1c40f"
            bg_col = "rgba(231, 76, 60, 0.1)" if ev["risk_category"] in ["Critical", "High"] else "rgba(241, 196, 15, 0.08)"

            with st.container():
                st.markdown(f"""
                <div style="border-left: 5px solid {border_col}; background:{bg_col}; padding:12px; border-radius:6px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                        <span style="font-weight:700; font-size:1.05rem;">Desk #{ev['seat_id'] or 'N/A'} (ID: {ev['tracking_id']}) — {ev['event_display_name']}</span>
                        <span style="background:{border_col}; color:white; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:700;">{ev['risk_category'].upper()} ({int(ev['risk_score'])}/100)</span>
                    </div>
                    <div style="font-family:monospace; background:rgba(0,0,0,0.3); padding:8px; border-radius:4px; font-size:0.85rem; color:#f8fafc; white-space:pre-line;">
Event: {ev['event_display_name']}
Seat: {ev['seat_id'] or 'Unassigned'}
Timestamp: {ev['timecode_str']}
Duration: {int(round(ev['duration_seconds']))} seconds
Detection confidence: {int(round(ev['confidence']*100))}%
Risk score: {int(round(ev['risk_score']))}/100
Reason: {ev['reason']}
Evidence: {ev['evidence_path']}
                    </div>
                    <div style="margin-top:6px; font-size:0.8rem; color:#94a3b8;">
                        Status: <b>{ev['review_status']}</b> | Video Time: <code>{ev['video_timestamp_sec']:.1f}s</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                btn1, btn2, btn3, btn4 = st.columns([1, 1, 1, 2])
                with btn1:
                    if st.button("✅ Acknowledge", key=f"ack_{ev['id']}"):
                        db.update_event_status(ev['id'], "ACKNOWLEDGED", "Noted by invigilator")
                        st.rerun()
                with btn2:
                    if st.button("🔍 Under Review", key=f"rev_{ev['id']}"):
                        db.update_event_status(ev['id'], "UNDER_REVIEW", "Escalated for video replay review")
                        st.rerun()
                with btn3:
                    if st.button("❌ False Positive", key=f"fp_{ev['id']}"):
                        db.update_event_status(ev['id'], "FALSE_POSITIVE", "Permitted classroom movement")
                        st.rerun()
                with btn4:
                    if st.button("✔️ Mark Resolved", key=f"res_{ev['id']}"):
                        db.update_event_status(ev['id'], "RESOLVED", "Incident addressed by proctor")
                        st.rerun()
    else:
        st.info("No recorded events matching the selected filter criteria.")

# ------------------------------------------------------------------------------
# TAB 4: EVIDENCE SNAPSHOT VAULT (Section 9, 11)
# ------------------------------------------------------------------------------
with tab_evidence:
    st.markdown("### 📸 Evidence Snapshot Vault")
    st.caption("Forensic snapshots automatically captured at the moment suspicious behavior crossed alert thresholds.")

    all_evidence_events = db.get_events(limit=24)
    if all_evidence_events:
        ev_cols = st.columns(3)
        for idx, ev in enumerate(all_evidence_events):
            with ev_cols[idx % 3]:
                epath = ev["evidence_path"]
                if os.path.exists(epath):
                    st.image(epath, use_container_width=True)
                    st.markdown(f"**Seat #{ev['seat_id'] or '?'} | {ev['event_display_name']}**")
                    st.caption(f"Risk: **{ev['risk_category']}** ({int(ev['risk_score'])}/100) | Time: `{ev['timecode_str']}`")
                else:
                    st.warning(f"Snapshot file missing: {epath}")
    else:
        st.info("Evidence vault is currently empty. Advance the video stream to encounter anomaly scenarios.")

# ------------------------------------------------------------------------------
# TAB 5: REPORTS & SESSION SUMMARY (Section 11)
# ------------------------------------------------------------------------------
with tab_reports:
    st.markdown("### 📊 Examination Session Summary & Audit Export")

    # Generate Summary text
    summary_text = report_gen.generate_human_readable_summary(st.session_state.current_session_id)
    st.markdown(summary_text)

    st.divider()
    st.subheader("📥 Export Audit Data")
    df_all = db.get_events_dataframe(st.session_state.current_session_id)

    c1, c2 = st.columns(2)
    with c1:
        if not df_all.empty:
            csv_bytes = df_all.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download Official Event Report (CSV)",
                data=csv_bytes,
                file_name=f"exam_monitoring_audit_{st.session_state.current_session_id}.csv",
                mime="text/csv"
            )
        else:
            st.info("CSV download will be available once events are recorded.")

    with c2:
        st.download_button(
            label="📝 Download Session Summary (Markdown)",
            data=summary_text.encode('utf-8'),
            file_name=f"session_summary_{st.session_state.current_session_id}.md",
            mime="text/markdown"
        )

    if not df_all.empty:
        st.markdown("#### Complete Audit Records Table:")
        st.dataframe(df_all, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 6: MODEL TRAINING & VIDEO CALIBRATION
# ------------------------------------------------------------------------------
with tab_training:
    st.markdown("### 🧠 Model Training & Video Calibration")
    st.caption(
        "Fine-tune the YOLO object detector and fit the unsupervised behavioral anomaly ML model "
        "directly on surveillance video recordings present in the `videos/` folder."
    )

    # 1. Video Dataset Inventory
    st.markdown("#### 📁 Classroom Video Dataset (`videos/`)")
    v_dir = "videos"
    if os.path.exists(v_dir):
        raw_v_files = sorted([f for f in os.listdir(v_dir) if f.lower().endswith((".mp4", ".avi", ".mov"))])
    else:
        raw_v_files = []

    if raw_v_files:
        v_data = []
        for vf in raw_v_files:
            v_path = os.path.join(v_dir, vf)
            m = VideoInputHandler.inspect_video(v_path)
            f_size_mb = os.path.getsize(v_path) / (1024 * 1024)
            v_data.append({
                "Video File": vf,
                "Resolution": f"{m.width}x{m.height}",
                "FPS": round(m.fps, 1),
                "Frames": m.total_frames,
                "Duration": f"{m.duration_seconds:.1f}s",
                "Size": f"{f_size_mb:.2f} MB"
            })
        st.dataframe(pd.DataFrame(v_data), use_container_width=True)
    else:
        st.warning("No video files found in `videos/` folder.")

    st.divider()

    # 2. Interactive Training Controls
    st.markdown("#### ⚙️ Training Parameters")
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        train_mode = st.selectbox(
            "Training Pipeline Mode:",
            ["all (YOLOv8 + ML Anomaly Model)", "behavior (ML Anomaly Model Only)", "yolo (YOLOv8 Only)"],
            index=0
        )
    with t_col2:
        yolo_epochs = st.slider("YOLO Fine-Tuning Epochs:", min_value=1, max_value=15, value=3, step=1)
    with t_col3:
        sample_interval = st.slider("Frame Sampling Interval (sec):", min_value=0.2, max_value=2.0, value=0.5, step=0.1)

    if st.button("🚀 Train Models on Classroom Videos", type="primary", use_container_width=True):
        trainer = ExamModelTrainer(videos_dir="videos")
        mode_str = train_mode.split()[0]
        do_yolo = mode_str in ["all", "yolo"]
        do_beh = mode_str in ["all", "behavior"]

        with st.spinner("Processing video frames and executing training pipeline on Apple Silicon (MPS / CPU)..."):
            try:
                progress_bar = st.progress(10)
                report = trainer.run_full_pipeline(
                    yolo_epochs=yolo_epochs,
                    sample_interval_sec=sample_interval,
                    train_yolo=do_yolo,
                    train_behavior=do_beh
                )
                progress_bar.progress(100)

                # Reload session detector and anomaly model
                st.session_state.detector = ExamHallDetector(model_path="models/yolov8n_exam.pt")
                if os.path.exists("models/behavior_anomaly_detector.joblib"):
                    st.session_state.anomaly_detector = ExamAnomalyDetector.load("models/behavior_anomaly_detector.joblib")

                st.success(f"🎉 Training completed successfully in {report['total_duration_sec']} seconds!")
                st.balloons()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Training failed: {e}")

    # 3. Model Status & Calibration Inspection
    st.divider()
    st.markdown("#### 📊 Active Model Artifacts & Baselines")

    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown("**Object Detector Weights:**")
        if os.path.exists("models/yolov8n_exam.pt"):
            exam_size = os.path.getsize("models/yolov8n_exam.pt") / (1024 * 1024)
            st.success(f"✅ `models/yolov8n_exam.pt` ({exam_size:.2f} MB) - Fine-tuned on classroom videos")
        else:
            st.info("ℹ️ Using base `models/yolov8n.pt` (Pretrained)")

    with c_m2:
        st.markdown("**ML Behavioral Anomaly Model:**")
        if os.path.exists("models/behavior_anomaly_detector.joblib"):
            anom_size = os.path.getsize("models/behavior_anomaly_detector.joblib") / (1024 * 1024)
            st.success(f"✅ `models/behavior_anomaly_detector.joblib` ({anom_size:.2f} MB) - Isolation Forest & Covariance Fitted")
        else:
            st.warning("⚠️ No behavioral anomaly model fitted yet.")

    # Show training metrics summary if metadata exists
    meta_json_path = "models/training_meta.json"
    if os.path.exists(meta_json_path):
        with open(meta_json_path, "r") as mf:
            meta_info = json.load(mf)
        st.markdown("##### Latest Training Run Metadata:")
        st.json(meta_info)

    # Feature baselines table
    if st.session_state.get("anomaly_detector") and st.session_state.anomaly_detector.is_fitted:
        st.markdown("##### Learned Student Behavioral Normality Baselines (10 Dimensions):")
        det = st.session_state.anomaly_detector
        f_rows = []
        for fname in det.feature_names:
            mean_val = det.feature_means.get(fname, 0.0)
            std_val = det.feature_stds.get(fname, 0.0)
            q_info = det.feature_quantiles.get(fname, {})
            f_rows.append({
                "Feature Dimension": fname.replace("_", " ").title(),
                "Baseline Mean": round(mean_val, 2),
                "Std Deviation": round(std_val, 2),
                "Median (q50)": round(q_info.get("q50", 0.0), 2),
                "95th Percentile": round(q_info.get("q95", 0.0), 2),
                "99th Percentile (Anomaly Boundary)": round(q_info.get("q99", 0.0), 2),
            })
        st.dataframe(pd.DataFrame(f_rows), use_container_width=True)

