"""
tests/test_alluvial.py
======================
Unit tests for Continuous Manifold Alluvial Genome River-Flow Visualization (rhizaeon.alluvial):
  - render_alluvial_genome_river execution
  - Output PNG generation and non-zero byte verification
  - Highlighting recombinant taxa and event markers
"""

import unittest
import os
import tempfile
import numpy as np

from rhizaeon.tensor import PrefixDistanceEngine
from rhizaeon.alluvial import render_alluvial_genome_river


class TestAlluvialGenomeRiver(unittest.TestCase):
    def setUp(self):
        # 4 taxa, 200 nucleotides
        N = 4
        L = 200
        seq_mat = np.zeros((N, L), dtype=np.int8)
        seq_mat[1, :] = 0  # P1: A
        seq_mat[2, :] = 2  # P2: G
        seq_mat[3, :] = 1  # Out: C
        seq_mat[0, :100] = 0  # Rec: A then G
        seq_mat[0, 100:] = 2

        self.taxa = ["Rec", "P1", "P2", "Out"]
        self.engine = PrefixDistanceEngine(seq_mat, codon_aligned=False)

    def test_render_alluvial_genome_river_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = os.path.join(tmpdir, "test_river.png")
            render_alluvial_genome_river(
                engine=self.engine,
                taxa=self.taxa,
                window_units=20,
                step=5,
                output_path=out_png,
                title="Test Alluvial River"
            )
            self.assertTrue(os.path.exists(out_png))
            self.assertGreater(os.path.getsize(out_png), 1000)

    def test_render_alluvial_genome_river_with_events_and_highlights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = os.path.join(tmpdir, "test_river_events.png")
            events = [
                {
                    "breakpoint": 100,
                    "recombinant": "Rec",
                    "parent_left": "P1",
                    "parent_right": "P2",
                    "pir": 0.95
                }
            ]
            render_alluvial_genome_river(
                engine=self.engine,
                taxa=self.taxa,
                recombination_events=events,
                highlight_taxa=["Rec"],
                window_units=20,
                step=5,
                output_path=out_png
            )
            self.assertTrue(os.path.exists(out_png))
            self.assertGreater(os.path.getsize(out_png), 1000)


if __name__ == "__main__":
    unittest.main()
