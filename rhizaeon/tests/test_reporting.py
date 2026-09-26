"""Unit tests for rhizaeon.reporting module."""

import unittest
from rhizaeon.reporting import (
    shorten_taxon_name,
    shorten_parent_name,
    classify_breakpoint,
    construct_mosaic_architecture,
    synthesize_inference_report,
    format_humanized_report,
    BreakpointRecord
)


class TestReporting(unittest.TestCase):
    def test_shorten_taxon_name(self):
        # LANL HIV format
        self.assertEqual(shorten_taxon_name("SN01_AE.TH.90.CM240.AF069670"), "CRF01_AE (CM240)")
        self.assertEqual(shorten_taxon_name("B.FR.83.HXB2_LAI_III"), "B (HXB2_LAI_III)")
        self.assertEqual(shorten_taxon_name("A1.UG.92.92UG037.AB2"), "A1 (92UG037)")
        
        # Generic / short
        self.assertEqual(shorten_taxon_name("SimpleTaxon"), "SimpleTaxon")

    def test_shorten_parent_name(self):
        self.assertEqual(shorten_parent_name("SN01_AE.TH.90.CM240"), "CRF01")
        self.assertEqual(shorten_parent_name("A1.UG.92.92UG037.AB2"), "A1")
        self.assertEqual(shorten_parent_name("B.FR.83.HXB2"), "B")

    def test_classify_breakpoint(self):
        # High confidence T1 crossover
        t, sup, stars = classify_breakpoint(z_score=3.5, pir=0.30, plateau_width=20)
        self.assertEqual(t, "T1-Cross")
        self.assertEqual(sup, "HIGH")
        self.assertEqual(stars, "★★★")

        # Tier 2 attention handoff
        t, sup, stars = classify_breakpoint(z_score=3.5, pir=0.30, plateau_width=20, tier="two-tier")
        self.assertEqual(t, "T2-Attn")

        # Ghost donor
        t, sup, stars = classify_breakpoint(z_score=3.5, pir=0.10, plateau_width=20)
        self.assertEqual(t, "Ghost")

        # Micro tract
        t, sup, stars = classify_breakpoint(z_score=2.8, pir=0.22, plateau_width=10, is_micro=True)
        self.assertEqual(t, "Micro")
        self.assertEqual(sup, "MOD")

    def test_construct_mosaic_architecture(self):
        tbps = [
            BreakpointRecord(
                idx=1,
                coord=2000,
                coord_nt=2000,
                unit_type="nt",
                recombinant="R",
                recombinant_short="R",
                parent_left="Parent_A",
                parent_right="Parent_B",
                parent_left_short="A",
                parent_right_short="B",
                z_score=3.5,
                pir=0.35,
                tier="tier1",
                bp_type="T1-Cross",
                support="HIGH",
                stars="★★★"
            ),
            BreakpointRecord(
                idx=2,
                coord=4000,
                coord_nt=4000,
                unit_type="nt",
                recombinant="R",
                recombinant_short="R",
                parent_left="Parent_B",
                parent_right="Parent_A",
                parent_left_short="B",
                parent_right_short="A",
                z_score=3.8,
                pir=0.40,
                tier="tier1",
                bp_type="T1-Cross",
                support="HIGH",
                stars="★★★"
            )
        ]
        mosaic_map, props = construct_mosaic_architecture(tbps, alignment_len=6000, unit_type="nt")
        self.assertIn("[A]──2,000──[B]──4,000──[A]", mosaic_map)
        self.assertAlmostEqual(props["A"], 66.66, delta=0.5)
        self.assertAlmostEqual(props["B"], 33.33, delta=0.5)

    def test_synthesize_and_format_humanized_report(self):
        taxa = ["R_mosaic", "P1_pure", "P2_pure", "Out_pure"]
        events = [
            {
                "breakpoint": 500,
                "breakpoint_nt": 500,
                "recombinant": "R_mosaic",
                "parent_left": "P1_pure",
                "parent_right": "P2_pure",
                "ghost_z": 4.2,
                "refined_pir": 0.45,
                "ci_left": 495,
                "ci_right": 505,
                "plateau_width": 10,
                "log_likelihood_gain": 25.0,
                "tier": "tier1"
            }
        ]
        rep = synthesize_inference_report(
            events_or_bps=events,
            taxa=taxa,
            alignment_path="sample.fasta",
            alignment_len=1000,
            unit_type="nt",
            elapsed_sec=0.123
        )
        self.assertEqual(rep.total_breakpoints, 1)
        self.assertEqual(len(rep.primary_recombinants), 1)
        self.assertEqual(len(rep.clonal_taxa), 3)

        out_str = format_humanized_report(rep)
        self.assertIn("RHIZAEON INFERENCE REPORT", out_str)
        self.assertIn("RECOMBINANT LINEAGES & MOSAIC ARCHITECTURE", out_str)
        self.assertIn("BREAKPOINT CATALOG", out_str)
        self.assertIn("R_mosaic", out_str)

        # Summary only
        summary_str = format_humanized_report(rep, summary_only=True)
        self.assertIn("RHIZAEON INFERENCE REPORT", summary_str)
        self.assertNotIn("BREAKPOINT CATALOG", summary_str)


if __name__ == "__main__":
    unittest.main()
