"""
tests/test_adaptive.py
======================
Unit tests for the 3-Tier Adaptive Hybrid Recombination Engine (rhizaeon.adaptive):
  - AdaptiveBreakpoint dataclass integrity
  - DualArchitectureConfig and dispatch triggers
  - evaluate_tier2_trigger edge conditions
  - run_adaptive_hybrid_screen on clonal null vs recombinant mosaic alignments
  - dispatch_tier2_transformer graceful fallback
"""

import unittest
import tempfile
import numpy as np

from rhizaeon.tensor import PrefixDistanceEngine
from rhizaeon.adaptive import (
    AdaptiveBreakpoint,
    DualArchitectureConfig,
    Tier2TransformerResult,
    evaluate_tier2_trigger,
    dispatch_tier2_transformer,
    run_adaptive_hybrid_screen
)


class TestAdaptiveHybridEngine(unittest.TestCase):
    def test_adaptive_breakpoint_dataclass(self):
        bp = AdaptiveBreakpoint(
            breakpoint_nt=300,
            uncertainty_interval_nt=(280, 320),
            recombinant_taxon="Rec",
            taxon_idx=0,
            parent_1="P1",
            parent_2="P2",
            l_pir=0.85,
            ghost_z_score=3.2,
            kinetic_z_score=4.1,
            characteristic_scale_nt=180,
            scale_regime="micro",
            holder_alpha=0.25,
            tv_jump_magnitude=1.1
        )
        self.assertEqual(bp.breakpoint_nt, 300)
        self.assertEqual(bp.uncertainty_interval_nt, (280, 320))
        self.assertEqual(bp.recombinant_taxon, "Rec")
        self.assertEqual(bp.parent_1, "P1")
        self.assertEqual(bp.scale_regime, "micro")

    def test_dual_architecture_config(self):
        cfg = DualArchitectureConfig()
        self.assertEqual(cfg.max_plateau_nt, 50)
        self.assertEqual(cfg.min_flank_snps, 4)
        self.assertEqual(cfg.min_fiedler_div, 0.40)
        self.assertEqual(cfg.min_taxon_drift, 0.035)

    def test_evaluate_tier2_trigger(self):
        cfg = DualArchitectureConfig(max_plateau_nt=50, min_flank_snps=4)

        # 1. Wide plateau -> Trigger
        should, reason = evaluate_tier2_trigger(60, 10, 0.5, cfg)
        self.assertTrue(should)
        self.assertIn("wide_plateau", reason)

        # 2. Sparse SNPs -> Trigger
        should, reason = evaluate_tier2_trigger(20, 2, 0.5, cfg)
        self.assertTrue(should)
        self.assertIn("sparse_snps", reason)

        # 3. Borderline PIR -> Trigger
        should, reason = evaluate_tier2_trigger(20, 10, 0.10, cfg)
        self.assertTrue(should)
        self.assertIn("borderline_pir", reason)

        # 4. Confident Tier 1 -> No Trigger
        should, reason = evaluate_tier2_trigger(20, 10, 0.50, cfg)
        self.assertFalse(should)
        self.assertEqual(reason, "tier1_confident")

    def test_dispatch_tier2_fallback_on_invalid_path(self):
        cfg = DualArchitectureConfig(weights_path="/non/existent/checkpoint.pt")
        res = dispatch_tier2_transformer(
            fasta_path="/non/existent/alignment.fasta",
            candidate_nt=150,
            uncertainty_window_nt=(130, 170),
            config=cfg
        )
        self.assertIsNone(res)

    def test_run_adaptive_hybrid_screen_clonal_null(self):
        # 6 identical taxa, 600 nucleotides -> Clonal null
        N = 6
        L = 600
        seq_mat = np.zeros((N, L), dtype=np.int8)
        taxa = [f"Taxon_{i}" for i in range(N)]
        engine = PrefixDistanceEngine(seq_mat, codon_aligned=False)

        breakpoints = run_adaptive_hybrid_screen(
            engine,
            taxa,
            min_kinetic_z=2.0,
            min_pir=0.08
        )
        self.assertEqual(len(breakpoints), 0)

    def test_run_adaptive_hybrid_screen_recombinant(self):
        # 6 taxa, 600 nucleotides
        # P1: all A (0)
        # P2: all G (2)
        # Out1: all C (1)
        # Out2: all T (3)
        # Out3: alternating A/C
        # Rec: 0..300 A, 300..600 G
        N = 6
        L = 600
        seq_mat = np.zeros((N, L), dtype=np.int8)
        seq_mat[1, :] = 0  # P1
        seq_mat[2, :] = 2  # P2
        seq_mat[3, :] = 1  # Out1
        seq_mat[4, :] = 3  # Out2
        seq_mat[5, :] = np.tile([0, 1], L // 2)  # Out3

        seq_mat[0, :300] = 0  # Rec
        seq_mat[0, 300:] = 2

        taxa = ["Rec", "P1", "P2", "Out1", "Out2", "Out3"]
        engine = PrefixDistanceEngine(seq_mat, codon_aligned=False)

        breakpoints = run_adaptive_hybrid_screen(
            engine,
            taxa,
            k_global=4,
            fda_bin_size=150,
            fda_step=20,
            min_kinetic_z=1.5,
            min_pir=0.05
        )

        self.assertGreater(len(breakpoints), 0)
        rec_bp = [b for b in breakpoints if b.recombinant_taxon == "Rec"][0]
        self.assertLess(abs(rec_bp.breakpoint_nt - 300), 40)
        self.assertIn(rec_bp.parent_1, ["P1", "P2"])
        self.assertIn(rec_bp.parent_2, ["P1", "P2"])
        self.assertGreater(rec_bp.l_pir, 0.5)


if __name__ == "__main__":
    unittest.main()
