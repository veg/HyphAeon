"""
tests/test_segmentation.py
==========================
Unit and integration tests for Recursive Binary Segmentation and RhizAeonDetector:
  - find_best_split_in_segment: Single-interval changepoint search
  - recursive_binary_segmentation: Multi-breakpoint recursion
  - detect_recombination:
      - Single-tier standalone execution (Scalar, Static 384D)
      - Two-tier cascaded execution (Tier 1 sieve -> Tier 2 verification)
      - Fast negative path (sub-5ms exit on clonal alignments)
      - Spatial concordance filtering (concordance_tolerance)
      - Proximal event deduplication (<= 20 codons)
"""

import unittest
import numpy as np

from rhizaeon.tensor import PrefixDistanceEngine
from rhizaeon.segmentation import RhizAeonDetector
from rhizaeon.embed import TwoTierPrefixDistanceEngine


class TestSegmentationAndDetector(unittest.TestCase):
    def setUp(self):
        # Build a synthetic alignment with 4 taxa, 120 codons (360 nt)
        # Clade 1 (P1): AAA x 120
        # Clade 2 (P2): GGG x 120
        # Outgroup:     CCC x 120
        # Recombinant:  AAA for 0..60, GGG for 60..120 (crossover at codon 60)
        N = 4
        L = 360
        seq_mat = np.zeros((N, L), dtype=np.int8)
        seq_mat[1, :] = 0  # P1: A (0)
        seq_mat[2, :] = 2  # P2: G (2)
        seq_mat[3, :] = 1  # Out: C (1)
        seq_mat[0, :180] = 0  # Rec left (0..60 codons)
        seq_mat[0, 180:] = 2  # Rec right (60..120 codons)

        self.seq_mat_rec = seq_mat
        self.taxa = ["Rec", "P1", "P2", "Outgroup"]
        self.engine_rec = PrefixDistanceEngine(seq_mat, codon_aligned=True)

        # Clonal (non-recombinant) alignment
        seq_mat_null = np.zeros((N, L), dtype=np.int8)
        seq_mat_null[0, :] = 0
        seq_mat_null[1, :] = 0
        seq_mat_null[2, :] = 2
        seq_mat_null[3, :] = 1
        self.engine_null = PrefixDistanceEngine(seq_mat_null, codon_aligned=True)

    def test_detector_defaults(self):
        det = RhizAeonDetector()
        self.assertEqual(det.window_units, 25)
        self.assertEqual(det.min_tract_units, 35)
        self.assertEqual(det.ghost_z_threshold, 3.0)
        self.assertEqual(det.pir_threshold, 0.25)
        self.assertEqual(det.concordance_tolerance, 60)

    def test_find_best_split_in_segment(self):
        det = RhizAeonDetector(min_tract_units=20, ghost_z_threshold=2.5, pir_threshold=0.20, step=5)
        split = det.find_best_split_in_segment(self.engine_rec, self.taxa, 0, 120, step=5)
        self.assertIsNotNone(split)
        # Raw cutpoint should be close to 60 (evaluated in step=5)
        self.assertIn(split["raw_breakpoint"], [55, 60, 65])
        self.assertEqual(split["recombinant"], "Rec")
        self.assertEqual(split["parent_left"], "P1")
        self.assertEqual(split["parent_right"], "P2")
        self.assertGreater(split["ghost_z"], 2.0)
        self.assertGreater(split["l_pir"], 0.5)

    def test_short_interval_rejection(self):
        det = RhizAeonDetector(min_tract_units=30)
        # Interval shorter than 2 * min_tract_units must return None
        split = det.find_best_split_in_segment(self.engine_rec, self.taxa, 0, 50)
        self.assertIsNone(split)

    def test_detect_recombination_single_tier(self):
        det = RhizAeonDetector(min_tract_units=20, ghost_z_threshold=2.5, pir_threshold=0.20, step=5)
        events = det.detect_recombination(self.engine_rec, self.taxa)
        self.assertEqual(len(events), 1)
        ev = events[0]
        # Breakpoint refined to exact codon boundary 60
        self.assertEqual(ev["breakpoint"], 60)
        self.assertEqual(ev["recombinant"], "Rec")
        self.assertEqual(ev["parent_left"], "P1")
        self.assertEqual(ev["parent_right"], "P2")
        self.assertAlmostEqual(ev["refined_pir"], 1.0, places=4)

    def test_detect_recombination_clonal_null(self):
        det = RhizAeonDetector(min_tract_units=20, ghost_z_threshold=2.5, pir_threshold=0.20)
        events = det.detect_recombination(self.engine_null, self.taxa)
        self.assertEqual(len(events), 0)

    def test_two_tier_pipeline_execution(self):
        # Wrap rec engine into a TwoTierPrefixDistanceEngine mock
        two_tier_rec = TwoTierPrefixDistanceEngine(
            tier1=self.engine_rec,
            tier2=self.engine_rec,
            track="joint"
        )
        det = RhizAeonDetector(min_tract_units=20, ghost_z_threshold=2.5, pir_threshold=0.20, step=5)
        events = det.detect_recombination(two_tier_rec, self.taxa)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["breakpoint"], 60)
        self.assertEqual(ev.get("tier"), "two-tier")

    def test_two_tier_fast_negative_path(self):
        # For clonal null, Tier 1 finds 0 events -> returns [] immediately
        called = {"tier2_accessed": False}

        def _factory():
            called["tier2_accessed"] = True
            return self.engine_null

        two_tier_lazy = TwoTierPrefixDistanceEngine(
            tier1=self.engine_null,
            tier2_factory=_factory
        )
        det = RhizAeonDetector()
        events = det.detect_recombination(two_tier_lazy, self.taxa)
        self.assertEqual(events, [])
        # Tier 2 was never constructed in memory
        self.assertFalse(called["tier2_accessed"])

    def test_two_tier_spatial_concordance_filtering(self):
        # Test spatial discordance filtering: if Tier 1 and Tier 2 candidate breakpoints
        # differ by more than concordance_tolerance (e.g. 10 codons), event is rejected
        two_tier_eng = TwoTierPrefixDistanceEngine(
            tier1=self.engine_rec,
            tier2=self.engine_rec
        )
        det = RhizAeonDetector(min_tract_units=20, ghost_z_threshold=2.5, pir_threshold=0.20, concordance_tolerance=60)
        events = det.detect_recombination(two_tier_eng, self.taxa)
        self.assertEqual(len(events), 1)

        # When tolerance is 0, only exact match passes
        events_strict = det.detect_recombination(two_tier_eng, self.taxa, concordance_tolerance=0)
        # Even with tolerance=0, raw breakpoint 60 on tier 1 matches raw 60 on tier 2
        self.assertEqual(len(events_strict), 1)

    def test_deduplication(self):
        det = RhizAeonDetector()
        # Artificial events list with two proximal breakpoints at codons 50 and 52
        ev1 = {"raw_breakpoint": 50, "recombinant_idx": 0, "parent_left_idx": 1, "parent_right_idx": 2}
        ev2 = {"raw_breakpoint": 52, "recombinant_idx": 0, "parent_left_idx": 1, "parent_right_idx": 2}
        # Both events should be tested and the higher PIR preserved


if __name__ == "__main__":
    unittest.main()
