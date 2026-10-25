"""
==============================================================================
Exam Behavioral Anomaly Detection ML Model
==============================================================================
Implements an unsupervised and semi-supervised Machine Learning Anomaly Detector
that learns the multivariate normal baseline distribution of student behavior
(velocities, trajectories, head orientations, neighbor proximity, desk displacement)
from classroom CCTV videos.

Provides:
- Temporal feature extraction per student trajectory
- Isolation Forest + Robust Mahalanobis/Covariance statistical ensemble
- Anomaly score computation in [0.0, 1.0] and percentile calibration
- Feature contribution / explainability attribution
- Model persistence via joblib
==============================================================================
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import MinCovDet
import joblib

logger = logging.getLogger("ExamAnomalyML")

# Standard feature order for tabular representation
FEATURE_NAMES = [
    "velocity",               # Instantaneous foot velocity (px/s)
    "accel",                  # Acceleration (px/s^2)
    "disp_from_anchor",       # Euclidean distance from initial seat anchor (px)
    "disp_window_rate",       # Net displacement over sliding window (px/s)
    "head_yaw_proxy",         # Head yaw angle proxy deviation (-1.0 to 1.0)
    "head_jitter_rate",       # Rapid orientation change frequency in window
    "bbox_aspect_ratio",      # h / w ratio of student bounding box
    "bbox_area_change_rate",  # Relative rate of bounding box area variation
    "neighbor_min_dist",      # Distance to nearest student (px)
    "neighbor_density_count"  # Number of students within close cluster radius (e.g. 180px)
]


class ExamAnomalyDetector:
    """
    Learns normal student behavioral distributions from video sequences
    and detects statistical outliers (suspicious behavioral anomalies).
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.feature_names = FEATURE_NAMES
        
        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=random_state,
            bootstrap=True,
            n_jobs=-1
        )
        self.robust_cov = MinCovDet(random_state=random_state)
        
        self.is_fitted = False
        self.feature_means = {}
        self.feature_stds = {}
        self.feature_quantiles = {}
        self.training_sample_count = 0

    def extract_feature_vector(
        self,
        trajectory: List[Tuple[float, float]],
        timestamps: List[float],
        head_poses: List[float],
        bboxes: List[Tuple[int, int, int, int]],
        all_student_positions: Optional[List[Tuple[float, float]]] = None,
        anchor_seat_pos: Optional[Tuple[float, float]] = None,
        cluster_radius: float = 180.0
    ) -> np.ndarray:
        """
        Extracts a 10-dimensional spatial-temporal feature vector from a tracked student.
        """
        if not trajectory or len(trajectory) < 2:
            return np.zeros(len(self.feature_names), dtype=np.float32)

        curr_pt = trajectory[-1]
        prev_pt = trajectory[-2]
        curr_t = timestamps[-1]
        prev_t = timestamps[-2]
        dt = max(curr_t - prev_t, 1e-3)

        # 1. Velocity (px/s)
        step_disp = np.hypot(curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1])
        velocity = float(step_disp / dt)

        # 2. Acceleration (px/s^2)
        if len(trajectory) >= 3:
            prev_prev_pt = trajectory[-3]
            dt_prev = max(prev_t - timestamps[-3], 1e-3)
            prev_vel = np.hypot(prev_pt[0] - prev_prev_pt[0], prev_pt[1] - prev_prev_pt[1]) / dt_prev
            accel = float(abs(velocity - prev_vel) / dt)
        else:
            accel = 0.0

        # 3. Displacement from initial anchor seat (px)
        anchor = anchor_seat_pos if anchor_seat_pos else trajectory[0]
        disp_from_anchor = float(np.hypot(curr_pt[0] - anchor[0], curr_pt[1] - anchor[1]))

        # 4. Displacement window rate over last 3 seconds
        window_start_idx = 0
        for i in range(len(timestamps) - 1, -1, -1):
            if curr_t - timestamps[i] >= 3.0:
                window_start_idx = i
                break
        window_dt = max(curr_t - timestamps[window_start_idx], 1e-3)
        win_disp = np.hypot(
            curr_pt[0] - trajectory[window_start_idx][0],
            curr_pt[1] - trajectory[window_start_idx][1]
        )
        disp_window_rate = float(win_disp / window_dt)

        # 5. Head yaw proxy
        head_yaw = float(head_poses[-1]) if head_poses else 0.0

        # 6. Head jitter / turn rate in window
        if head_poses and len(head_poses) > 1:
            recent_poses = head_poses[window_start_idx:]
            diffs = [abs(recent_poses[k] - recent_poses[k - 1]) for k in range(1, len(recent_poses))]
            head_jitter = float(sum(diffs) / window_dt) if diffs else 0.0
        else:
            head_jitter = 0.0

        # 7. Bounding box aspect ratio
        curr_box = bboxes[-1]
        bw = max(curr_box[2] - curr_box[0], 1)
        bh = max(curr_box[3] - curr_box[1], 1)
        aspect_ratio = float(bh / bw)

        # 8. Bbox area change rate
        init_box = bboxes[0]
        init_area = max((init_box[2] - init_box[0]) * (init_box[3] - init_box[1]), 1)
        curr_area = bw * bh
        area_change_rate = float(abs(curr_area - init_area) / (init_area * max(curr_t - timestamps[0], 1.0)))

        # 9 & 10. Neighbor proximity and density
        if all_student_positions and len(all_student_positions) > 1:
            dists = []
            cluster_count = 0
            for other_pt in all_student_positions:
                if other_pt == curr_pt:
                    continue
                d = float(np.hypot(curr_pt[0] - other_pt[0], curr_pt[1] - other_pt[1]))
                dists.append(d)
                if d <= cluster_radius:
                    cluster_count += 1
            min_dist = min(dists) if dists else 300.0
        else:
            min_dist = 250.0
            cluster_count = 0

        vec = np.array([
            velocity,
            accel,
            disp_from_anchor,
            disp_window_rate,
            abs(head_yaw),
            head_jitter,
            aspect_ratio,
            area_change_rate,
            min_dist,
            float(cluster_count)
        ], dtype=np.float32)

        return vec

    def fit(self, X: np.ndarray) -> "ExamAnomalyDetector":
        """
        Trains the unsupervised anomaly detector on a feature matrix X (N x 10).
        """
        if len(X) < 10:
            raise ValueError(f"Insufficient training samples: got {len(X)}, need at least 10.")

        X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)
        self.training_sample_count = len(X)

        # Fit Scaler
        X_scaled = self.scaler.fit_transform(X)

        # Compute summary stats per feature for interpretability
        for idx, name in enumerate(self.feature_names):
            vals = X[:, idx]
            self.feature_means[name] = float(np.mean(vals))
            self.feature_stds[name] = float(np.std(vals) + 1e-6)
            self.feature_quantiles[name] = {
                "q25": float(np.percentile(vals, 25)),
                "q50": float(np.percentile(vals, 50)),
                "q75": float(np.percentile(vals, 75)),
                "q95": float(np.percentile(vals, 95)),
                "q99": float(np.percentile(vals, 99))
            }

        # Fit Isolation Forest
        self.iso_forest.fit(X_scaled)

        # Fit Robust Covariance if dimensions permit
        try:
            self.robust_cov.fit(X_scaled)
        except Exception as e:
            logger.warning(f"Could not fit MinCovDet ({e}), falling back to standard covariance.")
            self.robust_cov = None

        self.is_fitted = True
        logger.info(f"ExamAnomalyDetector successfully trained on {len(X)} behavioral vectors.")
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Returns continuous anomaly score in [0.0, 1.0].
        Higher value indicates greater anomaly/suspicion.
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before scoring.")

        X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.scaler.transform(X)

        # Isolation Forest returns raw score where lower = more anomalous
        raw_iso = self.iso_forest.score_samples(X_scaled)
        # Normalize to [0, 1] anomaly score: standard iso scores ~ [-0.7, 0.2]
        iso_anomaly = np.clip((0.2 - raw_iso) / 0.6, 0.0, 1.0)

        # Robust Mahalanobis distance if available
        if self.robust_cov is not None:
            try:
                mahal = self.robust_cov.mahalanobis(X_scaled)
                # Map chi2-like mahal distance to [0, 1]
                mahal_anomaly = np.clip(mahal / 35.0, 0.0, 1.0)
                # Ensemble blend
                anomaly_score = 0.65 * iso_anomaly + 0.35 * mahal_anomaly
            except Exception:
                anomaly_score = iso_anomaly
        else:
            anomaly_score = iso_anomaly

        return np.clip(anomaly_score, 0.0, 1.0)

    def explain_anomaly(self, feature_vec: np.ndarray) -> Dict[str, Any]:
        """
        Provides human-interpretable feature contribution to the anomaly.
        """
        if not self.is_fitted:
            return {"score": 0.0, "is_anomaly": False, "primary_reasons": []}

        if feature_vec.ndim == 1:
            vec = feature_vec
        else:
            vec = feature_vec[0]

        score = float(self.score_samples(vec.reshape(1, -1))[0])
        is_anomaly = score >= 0.60

        deviations = []
        for idx, name in enumerate(self.feature_names):
            val = float(vec[idx])
            mean = self.feature_means.get(name, 0.0)
            std = self.feature_stds.get(name, 1.0)
            z_score = (val - mean) / std

            # In neighbor_min_dist, smaller is more anomalous (getting too close)
            if name == "neighbor_min_dist":
                z_score = -z_score

            deviations.append({
                "feature": name,
                "value": round(val, 2),
                "baseline_mean": round(mean, 2),
                "z_score": round(z_score, 2),
                "is_outlier": abs(z_score) > 2.0
            })

        # Sort features by absolute z-score
        deviations.sort(key=lambda d: abs(d["z_score"]), reverse=True)

        primary_reasons = []
        for dev in deviations[:3]:
            if dev["z_score"] > 1.8:
                feat = dev["feature"].replace("_", " ").title()
                primary_reasons.append(
                    f"{feat} ({dev['value']}) significantly exceeded baseline (mean: {dev['baseline_mean']}, z={dev['z_score']})"
                )
            elif dev["feature"] == "neighbor_min_dist" and dev["z_score"] < -1.8:
                primary_reasons.append(
                    f"Student proximity ({dev['value']}px) abnormally close to neighbor (mean: {dev['baseline_mean']}px)"
                )

        return {
            "score": round(score, 4),
            "score_pct": int(round(score * 100)),
            "is_anomaly": is_anomaly,
            "primary_reasons": primary_reasons,
            "top_deviations": deviations[:4]
        }

    def save(self, filepath: str) -> None:
        """Persists trained model and metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        artifact = {
            "scaler": self.scaler,
            "iso_forest": self.iso_forest,
            "robust_cov": self.robust_cov,
            "contamination": self.contamination,
            "feature_names": self.feature_names,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "feature_quantiles": self.feature_quantiles,
            "training_sample_count": self.training_sample_count,
            "is_fitted": self.is_fitted
        }
        joblib.dump(artifact, filepath)
        logger.info(f"Model successfully saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "ExamAnomalyDetector":
        """Loads a persisted model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        artifact = joblib.load(filepath)
        instance = cls(contamination=artifact.get("contamination", 0.05))
        instance.scaler = artifact["scaler"]
        instance.iso_forest = artifact["iso_forest"]
        instance.robust_cov = artifact["robust_cov"]
        instance.feature_names = artifact.get("feature_names", FEATURE_NAMES)
        instance.feature_means = artifact.get("feature_means", {})
        instance.feature_stds = artifact.get("feature_stds", {})
        instance.feature_quantiles = artifact.get("feature_quantiles", {})
        instance.training_sample_count = artifact.get("training_sample_count", 0)
        instance.is_fitted = artifact.get("is_fitted", True)
        logger.info(f"ExamAnomalyDetector successfully loaded from {filepath}")
        return instance
