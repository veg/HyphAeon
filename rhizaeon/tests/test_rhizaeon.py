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

    def test_prefix_tensor_overflow_large_codon_alignment(self):
        # 2 taxa, 66,000 nt = 22,000 codons
        # Total valid sites = 66,000 > 65,535 (exceeds uint16 max)
        L = 66000
        mat = np.zeros((2, L), dtype=np.int8)
        engine = PrefixDistanceEngine(mat, codon_aligned=True)
        self.assertEqual(engine.prefix_valid.dtype, np.uint32)
        self.assertEqual(int(engine.prefix_valid[0, 1, -1]), L)

    def test_snp_compressed_rejects_codon_aligned(self):
        with self.assertRaises(ValueError):
            SNPCompressedPrefixEngine(self.seq_mat, codon_aligned=True)


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

    def test_synthetic_crossover_polisher_inverted_orientation(self):
        # Rec transitions P2 -> P1 (inverted from caller's p1_idx, p2_idx)
        L = 1000
        mat = np.zeros((3, L), dtype=np.int8)
        mat[0, :] = 0  # P1
        mat[1, :] = 1  # P2
        mat[2, :500] = 1  # Rec left matches P2!
        mat[2, 500:] = 0  # Rec right matches P1!
        taxa = ["P1", "P2", "Rec"]

        # Caller inadvertently passes p1_idx=0 (P1), p2_idx=1 (P2)
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
        # Should auto-detect that P2 is the left donor and P1 is the right donor
        self.assertEqual(pol.parent_1, "P2")
        self.assertEqual(pol.parent_2, "P1")
        # ML midpoint should still pinpoint 499 or 500
        self.assertIn(pol.polished_bp, [499, 500])
        self.assertTrue(pol.log_likelihood_gain > 0)

    def test_adaptive_flank_information_sieve(self):
        # Alignment with sparse informative sites: 6 informative sites between P1 and P2
        # N_informative = 6 -> LL_min = max(0.5, min(2.5, 0.25 * 6)) = 1.5
        L = 200
        mat = np.zeros((3, L), dtype=np.int8)
        snp_positions = [30, 50, 70, 130, 150, 170]
        for pos in snp_positions:
            mat[0, pos] = 0  # P1
            mat[1, pos] = 1  # P2
            mat[2, pos] = 0 if pos < 100 else 1

        pol = polish_breakpoint_ml(
            seq_matrix=mat,
            coarse_bp=100,
            r_idx=2,
            p1_idx=0,
            p2_idx=1,
            search_window=80,
            taxa_names=["P1", "P2", "Rec"]
        )
        self.assertEqual(pol.num_informative_sites, 6)
        self.assertAlmostEqual(pol.min_ll_threshold, 1.5)
        self.assertIn(pol.polished_bp, range(60, 141))

        # Test explicit override threshold
        pol2 = polish_breakpoint_ml(
            seq_matrix=mat,
            coarse_bp=90,
            r_idx=2,
            p1_idx=0,
            p2_idx=1,
            search_window=80,
            min_ll_threshold=2.0
        )
        self.assertEqual(pol2.min_ll_threshold, 2.0)


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

    def test_rp_fda_quartet_procrustes(self):
        # N=4 quartet: Classical MDS + Procrustes manifold tracking without ad-hoc heuristic
        np.random.seed(42)
        L = 300
        N = 4
        mat = np.zeros((N, L), dtype=np.int8)
        taxa = ["Rec", "P1", "P2", "Out"]

        mat[1, :] = 0  # P1
        mat[2, :] = 1  # P2
        mat[3, :] = 2  # Out
        mat[0, :150] = mat[1, :150]
        mat[0, 150:] = mat[2, 150:]

        engine = PrefixDistanceEngine(mat, codon_aligned=False)
        bps = run_recursive_partition_fda_screen(
            engine,
            taxa,
            min_len=40,
            max_depth=2,
            min_z=1.0,
            min_pir=0.10,
            frobenius_triage=False,
            polish_ml=True
        )

        self.assertGreater(len(bps), 0)
        self.assertEqual(bps[0].recombinant_taxon, "Rec")
        self.assertLessEqual(abs(bps[0].breakpoint_nt - 150), 3)

    def test_rp_fda_codon_aligned_coordinates(self):
        # Alignment in codon units: 300 codons = 900 nt
        # Breakpoint at codon 150 (nt 450)
        L_codons = 300
        L_nt = L_codons * 3
        mat = np.zeros((4, L_nt), dtype=np.int8)
        taxa = ["Rec", "P1", "P2", "Out"]

        mat[1, :] = 0  # P1
        mat[2, :] = 1  # P2
        mat[3, :] = 2  # Out
        mat[0, :450] = mat[1, :450]
        mat[0, 450:] = mat[2, 450:]

        engine = PrefixDistanceEngine(mat, codon_aligned=True)
        self.assertTrue(engine.codon_aligned)
        self.assertEqual(engine.num_units, 300)

        bps = run_recursive_partition_fda_screen(
            engine,
            taxa,
            min_len=30,
            max_depth=2,
            min_z=1.0,
            min_pir=0.10,
            frobenius_triage=False,
            polish_ml=True
        )

        self.assertGreater(len(bps), 0)
        top = bps[0]
        self.assertEqual(top.recombinant_taxon, "Rec")
        # Codon coordinate around 150
        self.assertLessEqual(abs(top.breakpoint_nt - 150), 2)
        # Nucleotide coordinate around 450
        self.assertIsNotNone(top.nt_bp)
        self.assertLessEqual(abs(top.nt_bp - 450), 6)


class TestRobustnessRegression(unittest.TestCase):
    def test_cross_lineage_hotspot_preservation(self):
        # Test that deduplication does not collapse breakpoints from different taxa at the same hotspot
        validated = [
            {"breakpoint": 500, "refined_pir": 0.45, "recombinant_idx": 1, "recombinant": "Taxon_1"},
        ]
        ev2 = {"breakpoint": 510, "refined_pir": 0.50, "recombinant_idx": 2, "recombinant": "Taxon_2"}

        duplicate = False
        for v in validated:
            if v["recombinant_idx"] == ev2["recombinant_idx"] and abs(v["breakpoint"] - ev2["breakpoint"]) <= 20:
                duplicate = True
                if ev2["refined_pir"] > v["refined_pir"]:
                    v.update(ev2)
                break
        if not duplicate:
            validated.append(ev2)

        self.assertEqual(len(validated), 2)
        self.assertEqual(validated[0]["recombinant"], "Taxon_1")
        self.assertEqual(validated[1]["recombinant"], "Taxon_2")

    def test_discordant_mosaic_architecture(self):
        from rhizaeon.reporting import BreakpointRecord, construct_mosaic_architecture

        tbps = [
            BreakpointRecord(
                idx=1, coord=2000, coord_nt=2000, unit_type="nt",
                recombinant="R", recombinant_short="R",
                parent_left="Parent_A", parent_right="Parent_B",
                parent_left_short="A", parent_right_short="B",
                z_score=3.5, pir=0.35, tier="tier1", bp_type="T1-Cross",
                support="HIGH", stars="★★★"
            ),
            BreakpointRecord(
                idx=2, coord=4000, coord_nt=4000, unit_type="nt",
                recombinant="R", recombinant_short="R",
                parent_left="Parent_C", parent_right="Parent_A",
                parent_left_short="C", parent_right_short="A",
                z_score=3.8, pir=0.40, tier="tier1", bp_type="T1-Cross",
                support="HIGH", stars="★★★"
            )
        ]
        mosaic_map, props = construct_mosaic_architecture(tbps, alignment_len=6000, unit_type="nt")
        self.assertIn("[B/C]", mosaic_map)
        self.assertIn("B", props)
        self.assertIn("C", props)
        self.assertAlmostEqual(props["B"], 16.66, delta=0.5)
        self.assertAlmostEqual(props["C"], 16.66, delta=0.5)


if __name__ == "__main__":
    unittest.main()
