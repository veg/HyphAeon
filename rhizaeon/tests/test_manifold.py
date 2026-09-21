"""
tests/test_manifold.py
======================
Unit tests for continuous sequence manifold geometry:
  - Classical Multidimensional Scaling (compute_classical_mds)
  - Graph Laplacian / Diffusion Map embedding (compute_laplacian_eigenmaps)
  - Orthogonal Procrustes alignment (align_procrustes)
  - Studentized Ghost Node Z-scores (compute_ghost_node_zscores)
  - Continuous manifold river flow tracing (trace_continuous_manifold_flow)
"""

import unittest
import numpy as np

from rhizaeon.manifold import (
    compute_classical_mds,
    compute_laplacian_eigenmaps,
    align_procrustes,
    compute_ghost_node_zscores,
    trace_continuous_manifold_flow
)
from rhizaeon.tensor import PrefixDistanceEngine


class TestManifoldGeometry(unittest.TestCase):
    def setUp(self):
        # Construct known 2D Euclidean configuration (4 points on unit square)
        self.pts_true = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ], dtype=np.float64)
        diff = self.pts_true[:, None, :] - self.pts_true[None, :, :]
        self.D_square = np.linalg.norm(diff, axis=-1)

    def test_classical_mds_isometry(self):
        # MDS into 2 dimensions should preserve pairwise distances exactly
        Z = compute_classical_mds(self.D_square, k=2)
        self.assertEqual(Z.shape, (4, 2))

        # Reconstructed pairwise distances
        diff = Z[:, None, :] - Z[None, :, :]
        D_rec = np.linalg.norm(diff, axis=-1)
        np.testing.assert_allclose(D_rec, self.D_square, atol=1e-5)

        # Centered at origin
        np.testing.assert_allclose(np.mean(Z, axis=0), 0.0, atol=1e-5)

    def test_classical_mds_small_n(self):
        # Requesting k >= N should be capped safely without crash
        D_small = self.D_square[:3, :3]
        Z = compute_classical_mds(D_small, k=5)
        self.assertEqual(Z.shape, (3, 2))

    def test_laplacian_eigenmaps(self):
        # 6 taxa distance matrix
        np.random.seed(42)
        pts = np.random.randn(6, 3)
        diff = pts[:, None, :] - pts[None, :, :]
        D = np.linalg.norm(diff, axis=-1)

        Z_lap = compute_laplacian_eigenmaps(D, k=2, k_nn=2)
        self.assertEqual(Z_lap.shape, (6, 2))
        # Non-trivial eigenvectors should not be constant
        self.assertGreater(float(np.std(Z_lap[:, 0])), 1e-4)

    def test_procrustes_identity(self):
        # Identical coordinates should align with zero residual
        Z = self.pts_true
        Z_aligned, residuals = align_procrustes(Z, Z)
        np.testing.assert_allclose(Z_aligned, Z, atol=1e-7)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-7)

    def test_procrustes_rigid_transform(self):
        # Apply known 90-degree rotation and translation
        theta = np.pi / 2
        R_true = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        t_true = np.array([[2.5, -1.5]])
        Z_source = self.pts_true @ R_true + t_true

        Z_aligned, residuals = align_procrustes(self.pts_true, Z_source)
        np.testing.assert_allclose(Z_aligned, self.pts_true, atol=1e-6)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-6)

    def test_ghost_node_zscores(self):
        # Base residuals are tightly clustered near 0.1
        residuals = np.array([0.10, 0.11, 0.09, 0.10, 0.12, 1.50])  # Taxon 5 is an outlier
        z_scores = compute_ghost_node_zscores(residuals)

        # Taxon 5 should detach with significant Ghost Node Z > 3.0
        self.assertGreater(z_scores[5], 5.0)
        # Normal taxa should remain near zero
        for i in range(5):
            self.assertLess(abs(z_scores[i]), 2.5)

    def test_trace_continuous_manifold_flow(self):
        # Create a simple synthetic sequence matrix (4 taxa, 150 nt)
        np.random.seed(123)
        seq_mat = np.random.randint(0, 4, size=(4, 150), dtype=np.int8)
        engine = PrefixDistanceEngine(seq_mat, codon_aligned=False)

        flow = trace_continuous_manifold_flow(engine, window_units=20, step=10, k_dims=3)
        self.assertIn("cutpoints", flow)
        self.assertIn("trajectories", flow)
        self.assertIn("velocities", flow)

        num_cut = len(flow["cutpoints"])
        self.assertGreater(num_cut, 0)
        self.assertEqual(flow["trajectories"].shape, (num_cut, 4, 3))
        self.assertEqual(flow["velocities"].shape, (num_cut - 1, 4))
        # Velocities should be non-negative
        self.assertTrue(np.all(flow["velocities"] >= 0.0))


if __name__ == "__main__":
    unittest.main()
