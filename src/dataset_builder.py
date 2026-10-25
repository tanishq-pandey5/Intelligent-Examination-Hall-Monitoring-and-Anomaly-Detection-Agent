"""
==============================================================================
Exam Hall Dataset Builder & Feature Extractor
==============================================================================
Extracts frames and behavioral sequences from classroom/examination videos.
Prepares:
1. YOLO object detection training & validation datasets (images + .txt labels).
2. Behavioral spatial-temporal feature matrices for training ML Anomaly Detectors.
==============================================================================
"""

import os
import cv2
import yaml
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from ultralytics import YOLO

from src.detector import DetectedObject
from src.tracker import PersonTracker
from src.ml_anomaly_model import ExamAnomalyDetector


logger = logging.getLogger("DatasetBuilder")


class ExamDatasetBuilder:
    """
    Processes video files to generate:
    - Computer Vision detection dataset in YOLO format
    - Spatial-temporal behavioral matrices for ML Anomaly Detection
    """

    def __init__(
        self,
        videos_dir: str = "videos",
        output_dir: str = "data/dataset",
        sample_interval_sec: float = 0.5,
        val_split: float = 0.2,
        base_model_path: str = "models/yolov8n.pt"
    ):
        self.videos_dir = videos_dir
        self.output_dir = output_dir
        self.sample_interval_sec = sample_interval_sec
        self.val_split = val_split
        self.base_model_path = base_model_path
        
        self.yolo_model = YOLO(base_model_path)
        self.anomaly_detector = ExamAnomalyDetector()

    def build_all(self) -> Dict[str, Any]:
        """
        Builds both the YOLO visual dataset and the behavioral feature matrix.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        img_train_dir = os.path.join(self.output_dir, "images", "train")
        img_val_dir = os.path.join(self.output_dir, "images", "val")
        lbl_train_dir = os.path.join(self.output_dir, "labels", "train")
        lbl_val_dir = os.path.join(self.output_dir, "labels", "val")

        for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
            os.makedirs(d, exist_ok=True)

        video_files = sorted([
            f for f in os.listdir(self.videos_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ])

        if not video_files:
            raise FileNotFoundError(f"No valid video files found in directory: {self.videos_dir}")

        total_frames_extracted = 0
        train_count = 0
        val_count = 0
        all_feature_vectors: List[np.ndarray] = []

        logger.info(f"Processing {len(video_files)} videos from '{self.videos_dir}'...")

        for v_idx, v_name in enumerate(video_files):
            video_path = os.path.join(self.videos_dir, v_name)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Could not open video file: {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_step = max(int(fps * self.sample_interval_sec), 1)

            # Split logic: use later videos or frame hash for validation
            is_val_video = (v_idx >= int(len(video_files) * (1.0 - self.val_split)))

            tracker = PersonTracker(max_disappeared_frames=int(fps * 2), iou_threshold=0.30)
            
            # Local buffers for behavioral extraction
            track_trajectories: Dict[int, List[Tuple[float, float]]] = {}
            track_timestamps: Dict[int, List[float]] = {}
            track_bboxes: Dict[int, List[Tuple[int, int, int, int]]] = {}
            track_head_poses: Dict[int, List[float]] = {}

            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % frame_step == 0:
                    curr_timestamp = float(frame_num / fps)
                    h, w = frame.shape[:2]

                    # Predict detections with base YOLO
                    results = self.yolo_model.predict(frame, conf=0.25, verbose=False)[0]
                    boxes = results.boxes

                    target_img_dir = img_val_dir if is_val_video else img_train_dir
                    target_lbl_dir = lbl_val_dir if is_val_video else lbl_train_dir

                    sample_id = f"{os.path.splitext(v_name)[0]}_f{frame_num:06d}"
                    img_filename = f"{sample_id}.jpg"
                    lbl_filename = f"{sample_id}.txt"

                    img_path = os.path.join(target_img_dir, img_filename)
                    lbl_path = os.path.join(target_lbl_dir, lbl_filename)

                    # Save sampled frame
                    cv2.imwrite(img_path, frame)

                    # Write YOLO format labels
                    person_dets: List[DetectedObject] = []
                    yolo_lines = []
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        # Map person (0) and cell phone (67)
                        mapped_cls = -1
                        if cls_id == 0:
                            mapped_cls = 0  # person
                        elif cls_id == 67:
                            mapped_cls = 1  # cell phone

                        if mapped_cls != -1:
                            xyxy = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, xyxy)

                            if mapped_cls == 0:
                                person_dets.append(DetectedObject(
                                    class_id=0,
                                    class_name="person",
                                    confidence=conf,
                                    bbox=(x1, y1, x2, y2)
                                ))

                            # Normalized coordinates
                            bx = ((x1 + x2) / 2.0) / w
                            by = ((y1 + y2) / 2.0) / h
                            bw = (x2 - x1) / w
                            bh = (y2 - y1) / h
                            yolo_lines.append(f"{mapped_cls} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")

                    with open(lbl_path, "w") as lf:
                        lf.writelines(yolo_lines)

                    if is_val_video:
                        val_count += 1
                    else:
                        train_count += 1
                    total_frames_extracted += 1

                    # Update tracker for behavioral features
                    active_tracks = tracker.update(person_dets, curr_timestamp)
                    current_positions = [t.foot_point for t in active_tracks]

                    for t in active_tracks:
                        tid = t.tracking_id
                        if tid not in track_trajectories:
                            track_trajectories[tid] = []
                            track_timestamps[tid] = []
                            track_bboxes[tid] = []
                            track_head_poses[tid] = []

                        track_trajectories[tid].append(t.foot_point)
                        track_timestamps[tid].append(curr_timestamp)
                        track_bboxes[tid].append(t.bbox)
                        
                        # Head pose proxy from upper 25% of bbox
                        x1, y1, x2, y2 = t.bbox
                        head_crop = frame[max(0, y1):max(0, y1 + (y2 - y1) // 4), max(0, x1):max(0, x2)]
                        yaw_proxy = 0.0
                        if head_crop.size > 0:
                            gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
                            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                            yaw_proxy = float(np.mean(gx) / 128.0)
                        track_head_poses[tid].append(yaw_proxy)

                        # Extract feature vector if track length >= 3
                        if len(track_trajectories[tid]) >= 3:
                            feat_vec = self.anomaly_detector.extract_feature_vector(
                                trajectory=track_trajectories[tid],
                                timestamps=track_timestamps[tid],
                                head_poses=track_head_poses[tid],
                                bboxes=track_bboxes[tid],
                                all_student_positions=current_positions
                            )
                            all_feature_vectors.append(feat_vec)

                frame_num += 1
            cap.release()

        # Write YOLO dataset.yaml
        yaml_content = {
            "path": os.path.abspath(self.output_dir),
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "person",
                1: "cell phone"
            }
        }
        yaml_path = os.path.join(self.output_dir, "exam_data.yaml")
        with open(yaml_path, "w") as yf:
            yaml.dump(yaml_content, yf, default_flow_style=False)

        # Save behavioral feature matrix
        feature_matrix = np.array(all_feature_vectors, dtype=np.float32) if all_feature_vectors else np.empty((0, 10))
        feat_path = os.path.join(self.output_dir, "behavioral_features.npy")
        np.save(feat_path, feature_matrix)

        summary = {
            "total_frames": total_frames_extracted,
            "train_images": train_count,
            "val_images": val_count,
            "yaml_path": yaml_path,
            "behavioral_samples": len(feature_matrix),
            "behavioral_feature_path": feat_path,
            "videos_processed": len(video_files)
        }

        logger.info(f"Dataset build complete: {summary}")
        return summary
