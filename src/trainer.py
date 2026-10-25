"""
==============================================================================
Exam Monitoring & Anomaly Model Trainer
==============================================================================
Orchestrates:
1. Fine-tuning Ultralytics YOLOv8 on university classroom surveillance videos.
2. Fitting unsupervised behavioral ML anomaly detector (IsolationForest + Covariance).
3. Exporting trained model artifacts, metrics, and calibration reports.
==============================================================================
"""

import os
import json
import time
import shutil
import logging
from typing import Dict, Any, Optional
import numpy as np
import torch
from ultralytics import YOLO

from src.dataset_builder import ExamDatasetBuilder
from src.ml_anomaly_model import ExamAnomalyDetector

logger = logging.getLogger("ExamTrainer")


class ExamModelTrainer:
    """
    Coordinates training of the visual detector and the behavioral anomaly detector.
    """

    def __init__(
        self,
        videos_dir: str = "videos",
        dataset_dir: str = "data/dataset",
        models_dir: str = "models",
        outputs_dir: str = "outputs/training",
        device: Optional[str] = None
    ):
        self.videos_dir = videos_dir
        self.dataset_dir = dataset_dir
        self.models_dir = models_dir
        self.outputs_dir = outputs_dir

        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

    def train_behavior_anomaly_detector(
        self,
        feature_matrix: Optional[np.ndarray] = None,
        contamination: float = 0.05
    ) -> Dict[str, Any]:
        """
        Trains the unsupervised ML anomaly detector on behavioral feature vectors.
        """
        logger.info("Starting Behavioral Anomaly ML Model Training...")
        start_time = time.time()

        if feature_matrix is None or len(feature_matrix) == 0:
            feat_file = os.path.join(self.dataset_dir, "behavioral_features.npy")
            if os.path.exists(feat_file):
                feature_matrix = np.load(feat_file)
            else:
                raise FileNotFoundError(f"No behavioral features found at {feat_file}.")

        if len(feature_matrix) < 10:
            raise ValueError(f"Need at least 10 behavioral samples to train, got {len(feature_matrix)}.")

        detector = ExamAnomalyDetector(contamination=contamination)
        detector.fit(feature_matrix)

        # Compute training scores distribution
        train_scores = detector.score_samples(feature_matrix)
        q25, q50, q75, q95 = np.percentile(train_scores, [25, 50, 75, 95])

        # Save model artifact
        save_path = os.path.join(self.models_dir, "behavior_anomaly_detector.joblib")
        detector.save(save_path)

        train_duration = round(time.time() - start_time, 2)
        metrics = {
            "training_samples": int(len(feature_matrix)),
            "features_dimension": int(feature_matrix.shape[1]),
            "duration_seconds": train_duration,
            "anomaly_score_quantiles": {
                "q25": round(float(q25), 4),
                "q50_median": round(float(q50), 4),
                "q75": round(float(q75), 4),
                "q95": round(float(q95), 4)
            },
            "model_path": save_path,
            "feature_stats": detector.feature_means
        }

        # Save metrics JSON
        metrics_path = os.path.join(self.outputs_dir, "behavior_training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Behavioral Anomaly Detector trained successfully in {train_duration}s.")
        return metrics

    def train_yolo_detector(
        self,
        yaml_path: Optional[str] = None,
        base_weights: str = "models/yolov8n.pt",
        epochs: int = 5,
        imgsz: int = 640,
        batch_size: int = 8
    ) -> Dict[str, Any]:
        """
        Fine-tunes YOLOv8 on the extracted classroom dataset.
        """
        logger.info(f"Starting YOLO Fine-Tuning ({epochs} epochs, device={self.device})...")
        start_time = time.time()

        if yaml_path is None:
            yaml_path = os.path.join(self.dataset_dir, "exam_data.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YOLO data YAML not found at: {yaml_path}")

        # Check if base weights exist
        if not os.path.exists(base_weights):
            base_weights = "yolov8n.pt"

        model = YOLO(base_weights)

        # Run training
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=self.device,
            project=self.outputs_dir,
            name="yolo_exam_run",
            exist_ok=True,
            verbose=True,
            workers=2,
            plots=True
        )

        # Locate best weights from results.save_dir or fallback directories
        potential_dirs = [
            str(getattr(results, "save_dir", "")),
            os.path.join("runs", "detect", self.outputs_dir, "yolo_exam_run"),
            os.path.join(self.outputs_dir, "yolo_exam_run"),
        ]
        
        chosen_weights = None
        for pdir in potential_dirs:
            if not pdir:
                continue
            cand_best = os.path.join(pdir, "weights", "best.pt")
            cand_last = os.path.join(pdir, "weights", "last.pt")
            if os.path.exists(cand_best):
                chosen_weights = cand_best
                break
            elif os.path.exists(cand_last):
                chosen_weights = cand_last
                break

        target_pt = os.path.join(self.models_dir, "yolov8n_exam.pt")
        if chosen_weights and os.path.exists(chosen_weights):
            shutil.copy2(chosen_weights, target_pt)
            logger.info(f"Exported fine-tuned weights from {chosen_weights} to: {target_pt}")
        else:
            logger.warning(f"Could not find weights in candidate dirs, preserving base weights.")
            target_pt = base_weights


        train_duration = round(time.time() - start_time, 2)
        yolo_summary = {
            "epochs": epochs,
            "device": self.device,
            "duration_seconds": train_duration,
            "trained_weights_path": target_pt,
            "training_run_dir": run_dir
        }

        # Save summary JSON
        yolo_metrics_path = os.path.join(self.outputs_dir, "yolo_training_metrics.json")
        with open(yolo_metrics_path, "w") as f:
            json.dump(yolo_summary, f, indent=2)

        return yolo_summary

    def run_full_pipeline(
        self,
        yolo_epochs: int = 5,
        sample_interval_sec: float = 0.5,
        train_yolo: bool = True,
        train_behavior: bool = True
    ) -> Dict[str, Any]:
        """
        Executes end-to-end dataset extraction, YOLO fine-tuning, and ML anomaly detector fitting.
        """
        logger.info("=== Running Full Exam Model Training Pipeline ===")
        overall_start = time.time()

        # Step 1: Build dataset
        builder = ExamDatasetBuilder(
            videos_dir=self.videos_dir,
            output_dir=self.dataset_dir,
            sample_interval_sec=sample_interval_sec,
            base_model_path="models/yolov8n.pt"
        )
        data_summary = builder.build_all()

        yolo_result = None
        if train_yolo:
            yolo_result = self.train_yolo_detector(
                yaml_path=data_summary["yaml_path"],
                epochs=yolo_epochs
            )

        behavior_result = None
        if train_behavior:
            feat_matrix = np.load(data_summary["behavioral_feature_path"])
            behavior_result = self.train_behavior_anomaly_detector(feat_matrix)

        total_duration = round(time.time() - overall_start, 2)
        full_report = {
            "status": "COMPLETED",
            "total_duration_sec": total_duration,
            "dataset_summary": data_summary,
            "yolo_training": yolo_result,
            "behavior_anomaly_training": behavior_result,
            "active_models": {
                "detector": os.path.join(self.models_dir, "yolov8n_exam.pt") if train_yolo else "models/yolov8n.pt",
                "anomaly_classifier": os.path.join(self.models_dir, "behavior_anomaly_detector.joblib") if train_behavior else None
            }
        }

        # Save metadata
        meta_path = os.path.join(self.models_dir, "training_meta.json")
        with open(meta_path, "w") as mf:
            json.dump(full_report, mf, indent=2)

        logger.info(f"Full training pipeline completed in {total_duration}s.")
        return full_report
