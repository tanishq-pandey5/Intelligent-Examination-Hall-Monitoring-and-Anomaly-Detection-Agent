"""
Command Line Interface Demo Runner for Examination Hall Monitoring Agent.
Uses src/ modules to process video files, stream explainable alerts,
and export session reports.
"""
import argparse
import os
import sys
import time
import cv2

from config.settings import AppSettings, DEFAULT_SETTINGS
from src.video_input import VideoInputHandler
from src.frame_sampler import FrameSampler
from src.detector import ExamHallDetector
from src.tracker import PersonTracker
from src.seat_mapper import SeatMapper
from src.behavior import BehaviorAnalyzer
from src.agent import PersonAgentMemory, AgentState
from src.reasoning import RuleBasedReasoningEngine
from src.risk_engine import RiskEngine
from src.explanations import AlertExplainer
from src.evidence import EvidenceManager
from src.database import DatabaseManager
from src.report_generator import ReportGenerator
from src.ml_anomaly_model import ExamAnomalyDetector
from core.simulator import ExamHallSimulator


def generate_benchmark_video(output_path: str = "data/sample_videos/exam_hall_benchmark.mp4", duration: float = 20.0):
    print(f"🎬 Generating Examination Hall benchmark video ({duration}s, 18 desks, 5 anomaly scenarios)...")
    simulator = ExamHallSimulator()

    def on_progress(p):
        sys.stdout.write(f"\rRendering: {int(p * 100)}% complete")
        sys.stdout.flush()

    simulator.generate_video_file(output_path, duration_seconds=duration, progress_callback=on_progress)
    print(f"\n✅ Video generated successfully: {output_path}")

def run_analysis_on_video(video_path: str, sampling_fps: float = 5.0, max_duration: float = 30.0):
    meta = VideoInputHandler.inspect_video(video_path)
    if not meta.is_valid:
        print(f"❌ Error: {meta.error_message}")
        return

    print("=" * 70)
    print("🎓 AI EXAMINATION VIDEO FRAME ANALYSIS & PATTERN DETECTION")
    print("=" * 70)
    print(f"📹 Input File: {meta.file_name} ({meta.width}x{meta.height} @ {meta.fps:.1f} fps, {meta.duration_seconds}s)")
    print(f"⚙️  Sampling: {sampling_fps} FPS | Pretrained YOLOv8 & ByteTracker active")
    print("=" * 70)

    settings = AppSettings()
    db = DatabaseManager(settings.database_path)
    detector = ExamHallDetector()
    tracker = PersonTracker()
    seat_mapper = SeatMapper()
    behavior = BehaviorAnalyzer(settings)
    reasoning = RuleBasedReasoningEngine(settings)
    risk_engine = RiskEngine(settings)
    evidence_mgr = EvidenceManager(settings)
    reporter = ReportGenerator(db, settings)

    # Load trained ML anomaly model if present
    anomaly_detector = None
    anom_path = os.path.join("models", "behavior_anomaly_detector.joblib")
    if os.path.exists(anom_path):
        try:
            anomaly_detector = ExamAnomalyDetector.load(anom_path)
        except Exception:
            anomaly_detector = None

    model_label = "Fine-Tuned Exam YOLOv8" if detector.is_fine_tuned else "Base YOLOv8"
    ml_label = "Active (IsolationForest)" if anomaly_detector else "Inactive"
    print(f"🤖 Active Models: {model_label} ({detector.model_path}) | ML Behavioral Anomaly: {ml_label}")
    print("=" * 70)


    agent_memories = {}
    session_id = f"cli_{int(time.time())}"

    sampler = FrameSampler(target_sampling_fps=sampling_fps)
    frame_count = 0
    alerts_count = 0
    start_wall_time = time.time()

    for v_frame in sampler.sample_video(video_path, max_duration_sec=max_duration):
        frame = v_frame.image
        current_t = v_frame.video_timestamp_sec
        h, w = frame.shape[:2]
        frame_count += 1

        # 1. Detection
        people_dets, phone_dets = detector.detect(frame)

        # 2. Tracking
        tracks = tracker.update(people_dets, current_timestamp=current_t)

        # 3. Seat mapping & foot points
        seat_status = seat_mapper.update_occupancy(
            tracks=tracks,
            current_timestamp=current_t,
            dt=1.0 / sampling_fps,
            frame_w=w,
            frame_h=h
        )

        # 4. Gathering
        gathering_clusters = behavior.analyze_gathering_clusters(tracks, current_t)
        gathering_tids = {c.tracking_id: c for c in gathering_clusters}

        for track in tracks:
            tid = track.tracking_id
            sid = track.assigned_seat_id

            if tid not in agent_memories:
                agent_memories[tid] = PersonAgentMemory(
                    tracking_id=tid, assigned_seat_id=sid,
                    first_seen_timestamp=current_t, last_seen_timestamp=current_t
                )
            mem = agent_memories[tid]
            mem.last_seen_timestamp = current_t

            # Behavior modules
            obs_seat = behavior.analyze_seat_departure(track, seat_status, current_t)
            yaw, is_turn, _ = behavior.estimate_head_pose(frame, track, seat_mapper)
            mem.record_head_turn(current_t, yaw)
            obs_phone = behavior.analyze_phone_detection(track, phone_dets, current_t, w, h)
            obs_gather = gathering_tids.get(tid)
            obs_move = behavior.analyze_unusual_movement(track)

            active_obs = {"is_head_turning": is_turn}
            if obs_seat:
                active_obs["seat_departure"] = obs_seat.metrics
            if obs_phone:
                active_obs["phone"] = obs_phone.metrics
            if obs_gather:
                active_obs["gathering"] = obs_gather.metrics
            if obs_move:
                active_obs["unusual_movement"] = obs_move.metrics

            matched_rules = reasoning.evaluate_all_rules(mem, current_t, active_obs)

            if matched_rules:
                primary = matched_rules[0]
                ev_dur = primary.metrics.get("duration_sec", 4.0)
                ev_freq = primary.metrics.get("turn_count", primary.metrics.get("group_size", 1))

                # Extract ML anomaly score if detector is active
                ml_score = None
                if anomaly_detector is not None and len(track.trajectory) >= 3:
                    try:
                        all_pos = [t.foot_point for t in tracked_people]
                        traj_pts = [(pt[3], pt[4]) for pt in track.trajectory]
                        ts_list = [pt[0] for pt in track.trajectory]
                        box_list = [track.bbox] * len(traj_pts)
                        feat_vec = anomaly_detector.extract_feature_vector(
                            trajectory=traj_pts,
                            timestamps=ts_list,
                            head_poses=[0.0] * len(traj_pts),
                            bboxes=box_list,
                            all_student_positions=all_pos
                        )
                        ml_score = float(anomaly_detector.score_samples(feat_vec)[0])
                    except Exception:
                        ml_score = None

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
                if ml_score is not None:
                    mem.update_ml_anomaly(ml_score)
                mem._update_state_from_score(current_t, primary.reason_explanation)


                if mem.current_state in [AgentState.SUSPICIOUS, AgentState.ALERT]:
                    if evidence_mgr.should_save_evidence(tid, primary.event_type, current_t):
                        alerts_count += 1
                        ev_rel_path = evidence_mgr.capture_evidence_frame(
                            frame=frame,
                            tracking_id=tid,
                            seat_id=sid,
                            event_type=primary.event_type,
                            timestamp_sec=current_t,
                            timecode_str=v_frame.timecode_str,
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
                            session_id=session_id,
                            video_timestamp_sec=current_t,
                            timecode_str=v_frame.timecode_str,
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

                        # Print exactly formatted alert (Section 9)
                        print("\n--------------------------------------------------")
                        print(alert.to_formatted_text())
                        print("--------------------------------------------------")
            else:
                mem.apply_risk_decay(dt=1.0 / sampling_fps, decay_rate=settings.risk_decay_per_second)

    elapsed_wall = time.time() - start_wall_time
    proc_fps = frame_count / max(1e-3, elapsed_wall)

    # Save session
    db.insert_session(
        session_id=session_id,
        file_name=meta.file_name,
        file_path=meta.file_path,
        fps=meta.fps,
        resolution=f"{meta.width}x{meta.height}",
        duration_seconds=meta.duration_seconds,
        total_frames=meta.total_frames,
        analyzed_frames=frame_count,
        sampling_fps=sampling_fps
    )

    csv_path = reporter.export_csv_report(session_id)
    print("\n" + "=" * 70)
    print(f"🏁 Analysis Finished: Processed {frame_count} sampled frames in {elapsed_wall:.1f}s ({proc_fps:.1f} FPS)")
    print(f"📊 Total alerts recorded: {alerts_count}")
    print(f"📄 Audit Report (CSV): {csv_path}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="ExamGuard AI Video Analysis Agent")
    parser.add_argument("--generate-video", action="store_true", help="Generate benchmark video")
    parser.add_argument("--duration", type=float, default=20.0, help="Benchmark video duration")
    parser.add_argument("--input", type=str, default="data/sample_videos/exam_hall_benchmark.mp4", help="Video path")
    parser.add_argument("--sampling-fps", type=float, default=5.0, help="Frame sampling frequency")

    args = parser.parse_args()

    if args.generate_video:
        generate_benchmark_video("data/sample_videos/exam_hall_benchmark.mp4", duration=args.duration)
        return

    if not os.path.exists(args.input):
        print(f"Video '{args.input}' not found. Generating benchmark...")
        generate_benchmark_video(args.input, duration=args.duration)

    run_analysis_on_video(args.input, sampling_fps=args.sampling_fps)

if __name__ == "__main__":
    main()
