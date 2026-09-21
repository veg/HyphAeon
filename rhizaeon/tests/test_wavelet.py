"""
tests/test_wavelet.py
=====================
Unit tests for continuous multiresolution wavelet and scalogram analysis (rhizaeon.wavelet):
  - generate_dyadic_scales: Geometrically spaced scale octaves
  - compute_haar_scalogram: Integral Haar-Procrustes scale-space surface
  - trace_modulus_maxima_ridges: WTMM cone of influence ridge tracing
  - run_wavelet_recombination_screen: End-to-end multiscale screening
"""

import unittest
import numpy as np

from rhizaeon.tensor import PrefixDistanceEngine
from rhizaeon.wavelet import (
    generate_dyadic_scales,
    compute_haar_scalogram,
    trace_modulus_maxima_ridges,
    run_wavelet_recombination_screen,
    ScalogramResult,
    WaveletRidge
)


class TestWaveletScalogram(unittest.TestCase):
    def setUp(self):
        # 4 taxa, 600 nucleotides
        # P1: all A (0)
        # P2: all G (2)
        # Rec: 0..300 A, 300..600 G (crossover at nt 300)
        # Out: all C (1)
        N = 4
        L = 600
        seq_mat = np.zeros((N, L), dtype=np.int8)
        seq_mat[1, :] = 0  # P1
        seq_mat[2, :] = 2  # P2
        seq_mat[3, :] = 1  # Out
        seq_mat[0, :300] = 0
        seq_mat[0, 300:] = 2

        self.seq_mat = seq_mat
        self.taxa = ["Rec", "P1", "P2", "Out"]
        self.engine = PrefixDistanceEngine(seq_mat, codon_aligned=False)

    def test_generate_dyadic_scales(self):
        scales = generate_dyadic_scales(min_scale=32, max_scale=256, num_per_octave=2)
        self.assertGreater(len(scales), 3)
        self.assertEqual(scales[0], 32)
        self.assertEqual(scales[-1], 256)
        # Must be strictly increasing
        self.assertTrue(np.all(np.diff(scales) > 0))

    def test_compute_haar_scalogram(self):
        scales = np.array([64, 128, 256])
        scalo = compute_haar_scalogram(
            self.engine,
            self.taxa,
            scales=scales,
            step=25,
            k_mds=3
        )
        self.assertIsInstance(scalo, ScalogramResult)
        self.assertEqual(len(scalo.scales), 3)
        self.assertEqual(scalo.z_surface.shape[0], 4)
        self.assertEqual(scalo.z_surface.shape[1], 3)
        self.assertEqual(scalo.z_surface.shape[2], len(scalo.positions))

        # Peak Ghost Node leverage should be highest around nt 300
        peak_pos_idx = np.unravel_index(np.argmax(scalo.max_z_surface), scalo.max_z_surface.shape)[1]
        peak_nt = scalo.positions[peak_pos_idx]
        self.assertLess(abs(peak_nt - 300), 50)

    def test_trace_modulus_maxima_ridges(self):
        scales = np.array([64, 128, 256])
        scalo = compute_haar_scalogram(
            self.engine,
            self.taxa,
            scales=scales,
            step=20,
            k_mds=3
        )
        ridges = trace_modulus_maxima_ridges(
            scalo,
            self.engine,
            min_prominence=0.2,
            min_z_threshold=1.5
        )
        self.assertGreater(len(ridges), 0)
        top_ridge = ridges[0]
        self.assertIsInstance(top_ridge, WaveletRidge)
        self.assertEqual(top_ridge.taxon_name, "Rec")
        # Converged singularity should be very close to 300
        self.assertLess(abs(top_ridge.singularity_nt - 300), 30)
        self.assertGreater(top_ridge.max_z_score, 2.0)
        self.assertIn(top_ridge.scale_regime, ["micro", "meso", "macro"])

    def test_run_wavelet_recombination_screen(self):
        scalo, ridges = run_wavelet_recombination_screen(
            self.engine,
            self.taxa,
            min_scale=64,
            max_scale=256,
            step=20,
            min_prominence=0.2,
            min_z_threshold=1.5
        )
        self.assertGreater(len(ridges), 0)
        rec_ridge = [r for r in ridges if r.taxon_name == "Rec"][0]
        self.assertLess(abs(rec_ridge.singularity_nt - 300), 30)
        self.assertIn(rec_ridge.parent_1, ["P1", "P2"])
        self.assertIn(rec_ridge.parent_2, ["P1", "P2"])


if __name__ == "__main__":
    unittest.main()
