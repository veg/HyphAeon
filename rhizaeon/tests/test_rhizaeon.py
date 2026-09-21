"""
tests/test_rhizaeon.py
======================
Unit and integration tests for RhizAeon core modules:
  - PrefixDistanceEngine and SNPCompressedPrefixEngine
  - Manifold embedding and Procrustes alignment
  - ML Breakpoint Polisher and plateau bounding
  - Recursive Partitioning FDA (RP-FDA) with Frobenius triage
"""

import unittest
import numpy as np

from rhizaeon.tensor import PrefixDistanceEngine, SNPCompressedPrefixEngine
from rhizaeon.polisher import polish_breakpoint_ml, PolishedBreakpoint
from rhizaeon.fda import run_recursive_partition_fda_screen, FDABreakpoint
from rhizaeon.manifold import compute_classical_mds, align_procrustes, compute_ghost_node_zscores
from rhizaeon.pir import compute_pir, evaluate_triplets_for_taxon


class TestTensorEngines(unittest.TestCase):
    def setUp(self):
        # 4 taxa, 100 nt
        # Taxon 0: all A (0)
        # Taxon 1: first 50 A (0), next 50 C (1)
        # Taxon 2: all C (1)
        # Taxon 3: recombinant: 0..40 A (matches 0), 40..100 C (matches 2)
        L = 100
        mat = np.zeros((4, L), dtype=np.int8)
        mat[0, :] = 0
        mat[1, :50] = 0
        mat[1, 50:] = 1
        mat[2, :] = 1
        mat[3, :40] = 0
        mat[3, 40:] = 1
        self.seq_mat = mat
        self.taxa = ["T0", "T1", "T2", "T3"]

    def test_prefix_distance_engine(self):
        engine = PrefixDistanceEngine(self.seq_mat, codon_aligned=False)
        self.assertEqual(engine.N, 4)
        self.assertEqual(engine.L, 100)

        # Distance over [0, 40) between T0 and T3 should be 0.0
        d_left = engine.query_distance_matrix(0, 40)
        self.assertAlmostEqual(d_left[0, 3], 0.0)
        # Distance over [40, 100) between T2 and T3 should be 0.0
        d_right = engine.query_distance_matrix(40, 100)
        self.assertAlmostEqual(d_right[2, 3], 0.0)

    def test_snp_compressed_engine(self):
        engine = SNPCompressedPrefixEngine(self.seq_mat, codon_aligned=False)
        self.assertEqual(engine.N, 4)
        self.assertEqual(engine.full_L, 100)
        # Should have access to full_seq_matrix
        self.assertTrue(hasattr(engine, "full_seq_matrix"))
        self.assertEqual(engine.full_seq_matrix.shape, (4, 100))

        # Query distances should match exact uncompressed distances
        std_engine = PrefixDistanceEngine(self.seq_mat, codon_aligned=False)
        d_std = std_engine.query_distance_matrix(0, 50)
        d_cmp = engine.query_distance_matrix(0, 50)
        np.testing.assert_allclose(d_std, d_cmp, atol=1e-5)


class TestMLPolisher(unittest.TestCase):
    def test_synthetic_crossover_polisher(self):
        # Create clear 3-taxon setup:
        # P1: 0 (A) everywhere
        # P2: 1 (C) everywhere
        # Rec: P1 for 0..500, P2 for 500..1000
        # Coarse breakpoint estimated at 480
        L = 1000
        mat = np.zeros((3, L), dtype=np.int8)
        mat[0, :] = 0  # P1
        mat[1, :] = 1  # P2
        mat[2, :500] = 0  # Rec left
        mat[2, 500:] = 1  # Rec right
        taxa = ["P1", "P2", "Rec"]

        pol = polish_breakpoint_ml(
            seq_matrix=mat,
            coarse_bp=480,
            r_idx=2,
            p1_idx=0,
            p2_idx=1,
            search_window=100,
            taxa_names=taxa
        )

        self.assertIsInstance(pol, PolishedBreakpoint)
        self.assertEqual(pol.coarse_bp, 480)
        # Informative site at 499 (P1) and 500 (P2)
        # ML plateau should be bounded tightly around 500
        self.assertIn(pol.polished_bp, [499, 500])
        self.assertTrue(pol.log_likelihood_gain > 0)
        self.assertEqual(pol.recombinant_taxon, "Rec")
        self.assertEqual(pol.parent_1, "P1")
        self.assertEqual(pol.parent_2, "P2")


class TestRPFDAScreen(unittest.TestCase):
    def test_rp_fda_detection(self):
        # Construct alignment with an obvious recombinant
        # N=5 taxa, L=600 nt
        np.random.seed(42)
        L = 600
        N = 5
        mat = np.random.randint(0, 4, size=(N, L), dtype=np.int8)
        taxa = [f"taxon_{i}" for i in range(N)]

        # Make taxon 0 the recombinant:
        # 0..300 identical to taxon 1
        # 300..600 identical to taxon 2
        mat[0, :300] = mat[1, :300]
        mat[0, 300:] = mat[2, 300:]
        
        # Introduce distinct mutations between taxon 1 and 2 to ensure high divergence
        mat[2, :300] = (mat[1, :300] + 1) % 4
        mat[1, 300:] = (mat[2, 300:] + 1) % 4

        engine = PrefixDistanceEngine(mat, codon_aligned=False)
        bps = run_recursive_partition_fda_screen(
            engine,
            taxa,
            min_len=50,
            max_depth=3,
            min_z=1.5,
            min_pir=0.05,
            frobenius_triage=True,
            polish_ml=True
        )

        self.assertTrue(len(bps) > 0)
        top_bp = bps[0]
        self.assertEqual(top_bp.recombinant_taxon, "taxon_0")
        # Breakpoint should be polished very close to 300
        self.assertTrue(abs(top_bp.breakpoint_nt - 300) <= 2)
        self.assertIsNotNone(top_bp.ci_left)
        self.assertIsNotNone(top_bp.ci_right)
        self.assertIsNotNone(top_bp.log_likelihood_gain)


if __name__ == "__main__":
    unittest.main()
