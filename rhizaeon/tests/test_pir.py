"""
tests/test_pir.py
=================
Unit tests for Latent Parental Incongruence Ratio (L-PIR) and parental attribution:
  - compute_pir: Bounded PIR calculation and outgroup exclusion
  - evaluate_triplets_for_taxon: Vectorized parental pair search and divergence weighting
  - refine_breakpoint_codon: Single-codon local boundary refinement
"""

import unittest
import numpy as np

from rhizaeon.pir import (
    compute_pir,
    evaluate_triplets_for_taxon,
    refine_breakpoint_codon
)
from rhizaeon.tensor import PrefixDistanceEngine


class TestPIRMetrics(unittest.TestCase):
    def test_compute_pir_clean_recombinant(self):
        # 3 taxa: R (0), P1 (1), P2 (2)
        # Left segment D1: R is close to P1, distant from P2
        D1 = np.array([
            [0.00, 0.01, 0.10],
            [0.01, 0.00, 0.10],
            [0.10, 0.10, 0.00]
        ])
        # Right segment D2: R is close to P2, distant from P1
        D2 = np.array([
            [0.00, 0.10, 0.01],
            [0.10, 0.00, 0.10],
            [0.01, 0.10, 0.00]
        ])

        pir, t1, t2, is_valid = compute_pir(D1, D2, r_idx=0, p1_idx=1, p2_idx=2)
        self.assertTrue(is_valid)
        self.assertAlmostEqual(t1, 0.90, places=5)
        self.assertAlmostEqual(t2, 0.90, places=5)
        self.assertAlmostEqual(pir, 0.81, places=5)

    def test_compute_pir_outgroup_bounding(self):
        # Taxon 0 is an outgroup far from both P1 and P2
        D1 = np.array([
            [0.00, 0.50, 0.55],
            [0.50, 0.00, 0.10],
            [0.55, 0.10, 0.00]
        ])
        D2 = np.array([
            [0.00, 0.55, 0.50],
            [0.55, 0.00, 0.10],
            [0.50, 0.10, 0.00]
        ])

        # D1(R, P1) = 0.50 > 1.25 * 0.10 = 0.125 -> Must be rejected
        pir, _, _, is_valid = compute_pir(D1, D2, r_idx=0, p1_idx=1, p2_idx=2, bound_factor=1.25)
        self.assertFalse(is_valid)
        self.assertEqual(pir, 0.0)

    def test_compute_pir_insufficient_divergence(self):
        # Parents are nearly identical (D(P1, P2) = 0.005 < min_parent_dist 0.025)
        D1 = np.array([
            [0.000, 0.001, 0.005],
            [0.001, 0.000, 0.005],
            [0.005, 0.005, 0.000]
        ])
        D2 = D1.copy()
        pir, _, _, is_valid = compute_pir(D1, D2, r_idx=0, p1_idx=1, p2_idx=2, min_parent_dist=0.025)
        self.assertFalse(is_valid)
        self.assertEqual(pir, 0.0)

    def test_evaluate_triplets_for_taxon(self):
        # 4 taxa: 0=Rec, 1=P1, 2=P2, 3=Outgroup
        D1 = np.array([
            [0.00, 0.01, 0.12, 0.30],
            [0.01, 0.00, 0.12, 0.30],
            [0.12, 0.12, 0.00, 0.30],
            [0.30, 0.30, 0.30, 0.00]
        ])
        D2 = np.array([
            [0.00, 0.12, 0.01, 0.30],
            [0.12, 0.00, 0.12, 0.30],
            [0.01, 0.12, 0.00, 0.30],
            [0.30, 0.30, 0.30, 0.00]
        ])

        best = evaluate_triplets_for_taxon(D1, D2, r_idx=0, bound_factor=1.25, min_parent_dist=0.025)
        self.assertIsNotNone(best)
        self.assertEqual(best["recombinant_idx"], 0)
        self.assertEqual(best["parent_left_idx"], 1)
        self.assertEqual(best["parent_right_idx"], 2)
        self.assertGreater(best["pir"], 0.7)

        # Non-recombinant taxon (taxon 1 is pure P1)
        non_rec = evaluate_triplets_for_taxon(D1, D2, r_idx=1)
        self.assertIsNone(non_rec)

        # Multi-candidate top_k test
        cands = evaluate_triplets_for_taxon(D1, D2, r_idx=0, top_k=3)
        self.assertIsInstance(cands, list)
        self.assertGreater(len(cands), 0)
        self.assertEqual(cands[0]["parent_left_idx"], 1)
        self.assertEqual(cands[0]["parent_right_idx"], 2)

    def test_refine_breakpoint_codon(self):
        # Synthetic alignment with 4 taxa, 100 codons (300 nt)
        # P1 (1): AAA x 100
        # P2 (2): GGG x 100
        # Rec (0): AAA x 50, GGG x 50 (true breakpoint at codon 50)
        # Out (3): CCC x 100
        seq_mat = np.zeros((4, 300), dtype=np.int8)
        # P1: A (0)
        seq_mat[1, :] = 0
        # P2: G (2)
        seq_mat[2, :] = 2
        # Rec: A for 0..150, G for 150..300
        seq_mat[0, :150] = 0
        seq_mat[0, 150:] = 2
        # Out: C (1)
        seq_mat[3, :] = 1

        engine = PrefixDistanceEngine(seq_mat, codon_aligned=True)

        # Coarse breakpoint estimate at codon 44 (within search radius of 15)
        refined_bp, refined_pir = refine_breakpoint_codon(
            engine,
            bp=44,
            r_idx=0,
            p1_idx=1,
            p2_idx=2,
            flank_len=30,
            search_radius=15
        )

        # True crossover boundary is exactly at codon 50
        self.assertEqual(refined_bp, 50)
        self.assertAlmostEqual(refined_pir, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
