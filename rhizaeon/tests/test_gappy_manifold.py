"""
tests/test_gappy_manifold.py
============================
Comprehensive unit tests for robust gappy sequence handling and continuous
manifold geometry in RhizAeon:
  1. Prefix Distance Engine Coverage and Valid Overlap Queries
  2. Empirical Bayes Distance Shrinkage (preventing zero-distance collapse)
  3. Gower / Nyström Out-of-Sample MDS Landmark Projection
  4. Proper Kabsch SO(k) Alignment with NaN-Tolerance (reflection prevention)
  5. End-to-end Segment-Clamped Manifold Flow on Incomplete Genomes
"""

import unittest
import numpy as np

from rhizaeon.tensor import PrefixDistanceEngine
from rhizaeon.manifold import (
    compute_classical_mds,
    nystrom_out_of_sample_mds,
    align_procrustes,
    compute_ghost_node_zscores,
    trace_continuous_manifold_flow
)


class TestGappySequenceTensorEngine(unittest.TestCase):
    """Tests for low-level coverage and shrinkage distance queries in PrefixDistanceEngine."""

    def setUp(self):
        # 3 taxa, 100 nt
        # Taxon 0: 100% valid (0..3)
        # Taxon 1: 40 nt valid (pos 0..39), 60 nt gaps (pos 40..99, code 4)
        # Taxon 2: 100% gaps (code 4)
        np.random.seed(42)
        self.seq_mat = np.random.randint(0, 4, size=(3, 100), dtype=np.int8)
        self.seq_mat[1, 40:] = 4  # Gaps
        self.seq_mat[2, :] = 4    # 100% Gaps

        self.engine = PrefixDistanceEngine(self.seq_mat, codon_aligned=False)

    def test_query_coverage_full_and_partial(self):
        # Full span [0, 100)
        cov_full = self.engine.query_coverage(0, 100)
        self.assertAlmostEqual(cov_full[0], 1.0, places=5)
        self.assertAlmostEqual(cov_full[1], 0.4, places=5)
        self.assertAlmostEqual(cov_full[2], 0.0, places=5)

        # First half [0, 40)
        cov_first = self.engine.query_coverage(0, 40)
        self.assertAlmostEqual(cov_first[0], 1.0, places=5)
        self.assertAlmostEqual(cov_first[1], 1.0, places=5)
        self.assertAlmostEqual(cov_first[2], 0.0, places=5)

        # Second half [50, 100)
        cov_second = self.engine.query_coverage(50, 100)
        self.assertAlmostEqual(cov_second[0], 1.0, places=5)
        self.assertAlmostEqual(cov_second[1], 0.0, places=5)
        self.assertAlmostEqual(cov_second[2], 0.0, places=5)

    def test_query_coverage_boundary_conditions(self):
        # Out-of-bounds or inverted intervals
        cov_inv = self.engine.query_coverage(80, 50)
        np.testing.assert_array_equal(cov_inv, np.zeros(3))

        cov_clamp = self.engine.query_coverage(-20, 150)
        cov_normal = self.engine.query_coverage(0, 100)
        np.testing.assert_allclose(cov_clamp, cov_normal)

    def test_query_valid_matrix(self):
        # Over [0, 40): Taxon 0 and Taxon 1 both valid -> overlap = 40
        V_first = self.engine.query_valid_matrix(0, 40)
        self.assertEqual(V_first[0, 1], 40)
        self.assertEqual(V_first[1, 0], 40)
        self.assertEqual(V_first[0, 2], 0)

        # Over [40, 100): Taxon 1 has 0 valid overlap with Taxon 0
        V_second = self.engine.query_valid_matrix(40, 100)
        self.assertEqual(V_second[0, 1], 0)
        self.assertEqual(V_second[1, 0], 0)

    def test_empirical_bayes_shrinkage(self):
        # Window [40, 100) where Taxon 1 and 2 have 0 valid sites
        # With missing_strategy="raw", D[0, 1] would collapse to 0.0 (catastrophic bug)
        D_raw = self.engine.query_distance_matrix(40, 100, missing_strategy="raw")
        self.assertEqual(D_raw[0, 1], 0.0)

        # With missing_strategy="shrinkage", D[0, 1] must smoothly shrink toward prior
        prior_val = 0.25
        prior_mat = np.full((3, 3), prior_val, dtype=np.float64)
        np.fill_diagonal(prior_mat, 0.0)

        D_shrunk = self.engine.query_distance_matrix(
            40, 100,
            missing_strategy="shrinkage",
            shrinkage_kappa=20.0,
            prior_distance=prior_mat
        )
        # Because overlap is 0, weight = 0 / (0 + 20) = 0.0 -> exact prior distance
        self.assertAlmostEqual(D_shrunk[0, 1], prior_val, places=5)
        self.assertAlmostEqual(D_shrunk[0, 2], prior_val, places=5)
        # Diagonal must strictly remain 0.0
        self.assertEqual(D_shrunk[0, 0], 0.0)
        self.assertEqual(D_shrunk[1, 1], 0.0)
        self.assertEqual(D_shrunk[2, 2], 0.0)

    def test_nan_missing_strategy(self):
        D_nan = self.engine.query_distance_matrix(40, 100, missing_strategy="nan", min_overlap=5)
        self.assertTrue(np.isnan(D_nan[0, 1]))
        self.assertTrue(np.isnan(D_nan[0, 2]))
        # Diagonal should still be 0.0
        self.assertEqual(D_nan[0, 0], 0.0)


class TestNystromOutOfSampleMDS(unittest.TestCase):
    """Tests for Gower / Nyström out-of-sample projection onto landmark core."""

    def setUp(self):
        # Construct 5 core points in 2D Euclidean space
        np.random.seed(101)
        self.pts_core = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 1.0]
        ], dtype=np.float64)
        # Center core points
        self.pts_core -= np.mean(self.pts_core, axis=0)

        diff = self.pts_core[:, None, :] - self.pts_core[None, :, :]
        self.D_core = np.linalg.norm(diff, axis=-1)

        # MDS on core
        self.Z_core = compute_classical_mds(self.D_core, k=2)

    def test_exact_nystrom_reconstruction(self):
        # Test sample located at (0.5, 0.5) relative to original frame
        x_sample = np.array([0.5, 0.5]) - np.mean(self.pts_core, axis=0)
        d_sample = np.linalg.norm(self.pts_core - x_sample, axis=-1)

        z_proj = nystrom_out_of_sample_mds(self.Z_core, self.D_core, d_sample)

        # Distances from projected point to core in MDS space must match d_sample
        d_proj_to_core = np.linalg.norm(self.Z_core - z_proj, axis=-1)
        np.testing.assert_allclose(d_proj_to_core, d_sample, atol=1e-5)

    def test_nystrom_with_partial_missing_core_contacts(self):
        x_sample = np.array([1.2, 0.8]) - np.mean(self.pts_core, axis=0)
        d_sample = np.linalg.norm(self.pts_core - x_sample, axis=-1)

        # Mask 2 of the 5 core contacts as NaN
        d_sample_partial = d_sample.copy()
        d_sample_partial[0] = np.nan
        d_sample_partial[3] = np.nan

        z_proj = nystrom_out_of_sample_mds(self.Z_core, self.D_core, d_sample_partial)
        self.assertFalse(np.isnan(z_proj[0]))
        self.assertFalse(np.isnan(z_proj[1]))

        # Should match distances to observed core points
        valid_idx = [1, 2, 4]
        d_proj_valid = np.linalg.norm(self.Z_core[valid_idx] - z_proj, axis=-1)
        np.testing.assert_allclose(d_proj_valid, d_sample[valid_idx], atol=1e-5)

    def test_nystrom_insufficient_contacts(self):
        # If fewer than k valid contacts, must return NaNs
        d_insufficient = np.array([np.nan, 1.0, np.nan, np.nan, np.nan])
        z_proj = nystrom_out_of_sample_mds(self.Z_core, self.D_core, d_insufficient)
        self.assertTrue(np.all(np.isnan(z_proj)))


class TestKabschSOAlignment(unittest.TestCase):
    """Tests for NaN-tolerant Procrustes alignment and Kabsch SO(k) reflection ban."""

    def test_procrustes_reflection_ban(self):
        # Target: 4 points on unit square in 2D
        Z_target = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ], dtype=np.float64)

        # Source is a pure reflection along the y-axis: [x, y] -> [x, -y]
        Z_source = Z_target.copy()
        Z_source[:, 1] *= -1.0

        # With enforce_so=False, Procrustes will use a reflection matrix with det(R) = -1
        Z_aligned_unconstrained, _ = align_procrustes(Z_target, Z_source, enforce_so=False)
        np.testing.assert_allclose(Z_aligned_unconstrained, Z_target, atol=1e-6)

        # With enforce_so=True, reflection is strictly forbidden (det(R) = +1)
        Z_aligned_so, residuals = align_procrustes(Z_target, Z_source, enforce_so=True)
        # Because reflection is forbidden, it cannot achieve 0 residual on a mirrored set
        self.assertGreater(float(np.max(residuals)), 0.1)

    def test_procrustes_nan_tolerance(self):
        # 5 points, one is completely NaN in source
        Z_target = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5]
        ], dtype=np.float64)

        # Rotate source by 90 degrees
        R_90 = np.array([[0.0, -1.0], [1.0, 0.0]])
        Z_source = Z_target @ R_90

        # Taxon 4 is missing in source
        Z_source[4, :] = np.nan

        Z_aligned, residuals = align_procrustes(Z_target, Z_source, enforce_so=True)

        # The 4 valid points must align with near-zero residual
        np.testing.assert_allclose(Z_aligned[:4], Z_target[:4], atol=1e-6)
        np.testing.assert_allclose(residuals[:4], 0.0, atol=1e-6)

        # Missing point must remain NaN
        self.assertTrue(np.all(np.isnan(Z_aligned[4])))
        self.assertTrue(np.isnan(residuals[4]))


class TestEndToEndGappyManifoldFlow(unittest.TestCase):
    """Tests end-to-end continuous flow tracing with partial genomes and segments."""

    def test_continuous_flow_with_partial_taxa(self):
        # Construct synthetic alignment: 6 taxa, 200 nt (two 100 nt segments)
        # Taxa 0..3: Complete reference genomes
        # Taxon 4: Partial genome (present in seg 1 [0..100), missing in seg 2 [100..200))
        # Taxon 5: Partial genome (missing in seg 1, present in seg 2)
        np.random.seed(999)
        seq_mat = np.random.randint(0, 4, size=(6, 200), dtype=np.int8)
        seq_mat[4, 100:] = 4  # Missing in seg 2
        seq_mat[5, :100] = 4  # Missing in seg 1

        engine = PrefixDistanceEngine(seq_mat, codon_aligned=False)

        segment_bounds = [0, 100, 200]
        flow = trace_continuous_manifold_flow(
            engine,
            window_units=20,
            step=10,
            k_dims=2,
            min_core_coverage=0.30,
            min_sample_coverage=0.05,
            segment_bounds=segment_bounds
        )

        cutpoints = flow["cutpoints"]
        trajectories = flow["trajectories"]
        coverage = flow["coverage"]

        self.assertGreater(len(cutpoints), 4)
        self.assertEqual(trajectories.shape, (len(cutpoints), 6, 2))

        # Check Taxon 4 in segment 1 vs segment 2
        seg1_cut_mask = cutpoints < 100
        seg2_cut_mask = cutpoints >= 100

        # In segment 1, Taxon 4 should have valid coverage and finite trajectory coordinates
        self.assertTrue(np.all(coverage[seg1_cut_mask, 4] > 0.8))
        self.assertTrue(np.all(~np.isnan(trajectories[seg1_cut_mask, 4, :])))

        # In segment 2, Taxon 4 should be completely missing (coverage = 0, trajectories = NaN)
        self.assertTrue(np.all(coverage[seg2_cut_mask, 4] == 0.0))
        self.assertTrue(np.all(np.isnan(trajectories[seg2_cut_mask, 4, :])))

        # Conversely, Taxon 5 should be NaN in segment 1 and valid in segment 2
        self.assertTrue(np.all(np.isnan(trajectories[seg1_cut_mask, 5, :])))
        self.assertTrue(np.all(~np.isnan(trajectories[seg2_cut_mask, 5, :])))

        # Reference taxa (0..3) must remain finite throughout both segments
        for ref_i in range(4):
            self.assertTrue(np.all(~np.isnan(trajectories[:, ref_i, :])))


if __name__ == "__main__":
    unittest.main()
