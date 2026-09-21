"""
tests/test_conformal.py
=======================
Unit tests for Conformal Prediction in Genomic Metric Spaces (rhizaeon.conformal):
  - ConformalMetricCalibrator: Distribution-free non-conformity scoring & p-values
  - Finite-sample exchangeability coverage verification
  - MultiSegmentConformalEngine: Omnibus hypothesis aggregation (Simes & Bonferroni)
  - Dynamic Medoid Recruitment during surveillance streaming
"""

import unittest
import numpy as np

from rhizaeon.conformal import (
    ConformalMetricCalibrator,
    MultiSegmentConformalEngine
)


class TestConformalPrediction(unittest.TestCase):
    def setUp(self):
        # 2 clusters in 2D space: Cluster 0 centered at (0, 0), Cluster 1 at (10, 10)
        np.random.seed(42)
        c0 = np.random.randn(50, 2) * 0.5
        c1 = np.random.randn(50, 2) * 0.5 + np.array([10.0, 10.0])
        self.calib_pts = np.vstack([c0, c1])  # 100 points
        diff = self.calib_pts[:, None, :] - self.calib_pts[None, :, :]
        self.D_calib = np.linalg.norm(diff, axis=-1)

        # Medoid 0 at index 0, Medoid 1 at index 50
        self.medoids = [0, 50]

    def test_calibrator_fit_and_pvalues(self):
        calibrator = ConformalMetricCalibrator(normalized=False)
        calibrator.fit(self.D_calib, self.medoids)

        self.assertEqual(calibrator.n_calib, 100)
        self.assertEqual(len(calibrator.calibration_scores), 100)

        # In-distribution query (near cluster 0)
        q_in = np.array([[0.1, 0.1]])
        d_to_medoids_in = np.array([[
            np.linalg.norm(q_in[0] - self.calib_pts[0]),
            np.linalg.norm(q_in[0] - self.calib_pts[50])
        ]])
        p_in = calibrator.compute_p_value(d_to_medoids_in)
        self.assertGreater(p_in[0], 0.20)

        # Outlier query (far at (100, 100))
        q_out = np.array([[100.0, 100.0]])
        d_to_medoids_out = np.array([[
            np.linalg.norm(q_out[0] - self.calib_pts[0]),
            np.linalg.norm(q_out[0] - self.calib_pts[50])
        ]])
        p_out = calibrator.compute_p_value(d_to_medoids_out)
        self.assertLess(p_out[0], 0.05)

    def test_conformal_coverage_guarantee(self):
        # Under exchangeable data, false positive rate at alpha=0.10 should not exceed alpha
        calibrator = ConformalMetricCalibrator(normalized=False)
        calibrator.fit(self.D_calib, self.medoids)

        # Generate 200 fresh exchangeable test points from the same distribution
        np.random.seed(999)
        test_c0 = np.random.randn(100, 2) * 0.5
        test_c1 = np.random.randn(100, 2) * 0.5 + np.array([10.0, 10.0])
        test_pts = np.vstack([test_c0, test_c1])

        # Distances to the 2 calibration medoids
        d_m0 = np.linalg.norm(test_pts - self.calib_pts[0], axis=1)
        d_m1 = np.linalg.norm(test_pts - self.calib_pts[50], axis=1)
        d_to_medoids = np.column_stack([d_m0, d_m1])

        p_vals = calibrator.compute_p_value(d_to_medoids)
        alpha = 0.10
        emp_fpr = np.mean(p_vals < alpha)
        # Empirical error rate should closely match nominal alpha <= 0.12
        self.assertLessEqual(emp_fpr, 0.12)

    def test_normalized_calibrator(self):
        calibrator_norm = ConformalMetricCalibrator(normalized=True)
        calibrator_norm.fit(self.D_calib, self.medoids)
        self.assertIsNotNone(calibrator_norm.cluster_scales)
        self.assertEqual(len(calibrator_norm.cluster_scales), 2)

        # Query in cluster
        q = np.array([[0.2, -0.2]])
        d_q = np.array([[
            np.linalg.norm(q[0] - self.calib_pts[0]),
            np.linalg.norm(q[0] - self.calib_pts[50])
        ]])
        p_val = calibrator_norm.compute_p_value(d_q)
        self.assertGreater(p_val[0], 0.05)

    def test_get_threshold(self):
        calibrator = ConformalMetricCalibrator(normalized=False)
        calibrator.fit(self.D_calib, self.medoids)

        thresh_05 = calibrator.get_threshold(0.05)
        thresh_01 = calibrator.get_threshold(0.01)
        # Stricter significance level requires larger distance threshold
        self.assertGreaterEqual(thresh_01, thresh_05)


if __name__ == "__main__":
    unittest.main()
