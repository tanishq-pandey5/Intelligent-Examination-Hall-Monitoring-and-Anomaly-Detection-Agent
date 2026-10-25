"""
==============================================================================
Unit Tests for Model Training Pipeline and ML Anomaly Detection
==============================================================================
Verifies:
- Behavioral temporal feature extraction
- ExamAnomalyDetector fitting, scoring, and explainability
- Anomaly score calibration and thresholds
- Saving and loading model artifacts via joblib
- Integration of ML anomaly scoring with RiskEngine
==============================================================================
"""

import os
import unittest
import numpy as np
import tempfile
import shutil

from src.ml_anomaly_model import ExamAnomalyDetector, FEATURE_NAMES
from src.risk_engine import RiskEngine
from src.dataset_builder import ExamDatasetBuilder
from src.trainer import ExamModelTrainer


class TestMLAnomalyModel(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.detector = ExamAnomalyDetector(contamination=0.05, random_state=42)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_feature_extraction(self):
        """Verifies 10-D feature vector extraction from tracked trajectory."""
        trajectory = [(100.0, 200.0), (102.0, 201.0), (105.0, 203.0)]
        timestamps = [1.0, 2.0, 3.0]
        head_poses = [0.1, 0.15, -0.2]
        bboxes = [(80, 150, 120, 210), (81, 151, 121, 211), (82, 152, 122, 212)]
        all_positions = [(105.0, 203.0), (300.0, 400.0)]

        feat_vec = self.detector.extract_feature_vector(
            trajectory=trajectory,
            timestamps=timestamps,
            head_poses=head_poses,
            bboxes=bboxes,
            all_student_positions=all_positions
        )

        self.assertEqual(len(feat_vec), len(FEATURE_NAMES))
        self.assertIsInstance(feat_vec, np.ndarray)
        self.assertGreater(feat_vec[0], 0.0) # velocity > 0
        self.assertGreater(feat_vec[8], 100.0) # neighbor min dist

    def test_fit_and_score(self):
        """Tests fitting the ML model on synthetic normal exam data and detecting an outlier."""
        np.random.seed(42)
        # 100 normal student feature vectors (low speed, regular spacing, low turn rate)
        n_samples = 80
        X_normal = np.zeros((n_samples, 10), dtype=np.float32)
        X_normal[:, 0] = np.random.uniform(1.0, 8.0, n_samples)      # velocity
        X_normal[:, 1] = np.random.uniform(0.1, 2.0, n_samples)      # accel
        X_normal[:, 2] = np.random.uniform(5.0, 20.0, n_samples)     # disp from anchor
        X_normal[:, 3] = np.random.uniform(0.5, 3.0, n_samples)      # disp window rate
        X_normal[:, 4] = np.random.uniform(0.0, 0.3, n_samples)      # head yaw
        X_normal[:, 5] = np.random.uniform(0.0, 0.5, n_samples)      # head jitter
        X_normal[:, 6] = np.random.uniform(1.2, 1.6, n_samples)      # aspect ratio
        X_normal[:, 7] = np.random.uniform(0.0, 0.1, n_samples)      # area change
        X_normal[:, 8] = np.random.uniform(150.0, 250.0, n_samples)  # neighbor dist
        X_normal[:, 9] = np.random.choice([0.0, 1.0], n_samples)     # neighbor count

        self.detector.fit(X_normal)
        self.assertTrue(self.detector.is_fitted)

        # Score normal sample
        normal_sample = X_normal[0:1]
        normal_score = float(self.detector.score_samples(normal_sample)[0])

        # Create anomalous sample (very high velocity, huge displacement, tight cluster)
        anomalous_sample = np.array([[75.0, 25.0, 350.0, 40.0, 1.8, 4.5, 3.2, 1.2, 15.0, 5.0]], dtype=np.float32)
        anom_score = float(self.detector.score_samples(anomalous_sample)[0])

        self.assertGreater(anom_score, normal_score)
        self.assertGreaterEqual(anom_score, 0.5)

        # Test explanation
        explanation = self.detector.explain_anomaly(anomalous_sample)
        self.assertTrue(explanation["is_anomaly"])
        self.assertGreaterEqual(len(explanation["primary_reasons"]), 1)

    def test_save_and_load(self):
        """Verifies model serialization and deserialization via joblib."""
        np.random.seed(42)
        X = np.random.randn(30, 10).astype(np.float32)
        self.detector.fit(X)

        save_path = os.path.join(self.temp_dir, "test_detector.joblib")
        self.detector.save(save_path)
        self.assertTrue(os.path.exists(save_path))

        loaded = ExamAnomalyDetector.load(save_path)
        self.assertTrue(loaded.is_fitted)
        self.assertEqual(len(loaded.feature_names), 10)

        # Verify predictions match exactly
        orig_scores = self.detector.score_samples(X[:5])
        loaded_scores = loaded.score_samples(X[:5])
        np.testing.assert_allclose(orig_scores, loaded_scores, rtol=1e-5)

    def test_risk_engine_integration_with_ml(self):
        """Verifies RiskEngine incorporates ml_anomaly_score."""
        engine = RiskEngine()
        
        # Standard score without ML
        risk_base = engine.compute_risk_score(
            event_type="UNUSUAL_MOVEMENT",
            detection_confidence=0.85,
            duration_sec=4.0,
            frequency_count=2,
            ml_anomaly_score=None
        )

        # High ML anomaly score should boost risk
        risk_boosted = engine.compute_risk_score(
            event_type="UNUSUAL_MOVEMENT",
            detection_confidence=0.85,
            duration_sec=4.0,
            frequency_count=2,
            ml_anomaly_score=0.92
        )

        self.assertGreater(risk_boosted.total_score, risk_base.total_score)
        self.assertGreater(risk_boosted.ml_anomaly_component, 0.0)
        self.assertIn("ML Anomaly Model", risk_boosted.explanation_of_components)


if __name__ == "__main__":
    unittest.main()
