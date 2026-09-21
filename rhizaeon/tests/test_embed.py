"""
tests/test_embed.py
===================
Unit and integration tests for:
  - Approach A: EmbeddingPrefixDistanceEngine (Static 384D token embeddings)
  - Approach B: ContextualPrefixDistanceEngine (Contextual 384D transformer hidden states)
  - Factory: build_prefix_engine
  - CLI execution with --engine {scalar, embed-static, embed-contextual}
"""

import os
import tempfile
import unittest
import numpy as np

from rhizaeon.embed import (
    load_embedding_matrices,
    TwoTierPrefixDistanceEngine,
    EmbeddingPrefixDistanceEngine,
    ContextualPrefixDistanceEngine,
    build_prefix_engine,
    resolve_checkpoint_path
)


class TestEmbeddingEngines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a small synthetic recombinant FASTA file
        # 4 taxa, 90 nt (30 codons)
        # Taxon 0: AAA (Lys) x 15, GGG (Gly) x 15
        # Taxon 1: AAA (Lys) x 30
        # Taxon 2: GGG (Gly) x 30
        # Taxon 3: CCC (Pro) x 30 (outgroup)
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.fasta_path = os.path.join(cls.temp_dir.name, "test_toy.fasta")
        with open(cls.fasta_path, "w") as f:
            f.write(">Taxon_Rec\n" + ("AAA" * 15 + "GGG" * 15) + "\n")
            f.write(">Taxon_P1\n" + ("AAA" * 30) + "\n")
            f.write(">Taxon_P2\n" + ("GGG" * 30) + "\n")
            f.write(">Taxon_Out\n" + ("CCC" * 30) + "\n")

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_load_embedding_matrices(self):
        d_joint, d_ds, d_dn = load_embedding_matrices()
        self.assertEqual(d_joint.shape, (66, 66))
        self.assertEqual(d_ds.shape, (66, 66))
        self.assertEqual(d_dn.shape, (66, 66))
        # Zero diagonal
        np.testing.assert_allclose(np.diag(d_joint), 0.0)
        np.testing.assert_allclose(np.diag(d_ds), 0.0)
        np.testing.assert_allclose(np.diag(d_dn), 0.0)

    def test_embedding_prefix_distance_engine_approach_a(self):
        engine = EmbeddingPrefixDistanceEngine(self.fasta_path, track="joint")
        self.assertEqual(engine.N, 4)
        self.assertEqual(engine.num_units, 30)
        self.assertEqual(engine.taxa, ["Taxon_Rec", "Taxon_P1", "Taxon_P2", "Taxon_Out"])

        # Left segment [0, 15): Rec should match P1
        d_left = engine.query_distance_matrix(0, 15)
        self.assertAlmostEqual(d_left[0, 1], 0.0)
        self.assertGreater(d_left[0, 2], 0.0)

        # Right segment [15, 30): Rec should match P2
        d_right = engine.query_distance_matrix(15, 30)
        self.assertAlmostEqual(d_right[0, 2], 0.0)
        self.assertGreater(d_right[0, 1], 0.0)

        # Test dS and dN track initialization
        eng_ds = EmbeddingPrefixDistanceEngine(self.fasta_path, track="ds")
        self.assertEqual(eng_ds.num_units, 30)
        eng_dn = EmbeddingPrefixDistanceEngine(self.fasta_path, track="dn")
        self.assertEqual(eng_dn.num_units, 30)

    def test_contextual_prefix_distance_engine_approach_b(self):
        engine = ContextualPrefixDistanceEngine(self.fasta_path, device="cpu")
        self.assertEqual(engine.N, 4)
        self.assertEqual(engine.num_units, 30)
        self.assertEqual(len(engine.taxa), 4)

        # Left segment [0, 15): Rec distance to P1 should be much smaller than to P2
        d_left = engine.query_distance_matrix(0, 15)
        self.assertLess(d_left[0, 1], d_left[0, 2])

        # Right segment [15, 30): Rec distance to P2 should be much smaller than to P1
        d_right = engine.query_distance_matrix(15, 30)
        self.assertLess(d_right[0, 2], d_right[0, 1])

    def test_build_prefix_engine_factory(self):
        # Scalar
        eng_scalar = build_prefix_engine(self.fasta_path, engine="scalar")
        self.assertEqual(eng_scalar.num_units, 30)
        self.assertEqual(len(eng_scalar.taxa), 4)

        # Static / Approach A
        eng_static = build_prefix_engine(self.fasta_path, engine="embed-static")
        self.assertIsInstance(eng_static, EmbeddingPrefixDistanceEngine)
        self.assertEqual(eng_static.num_units, 30)

        # Contextual / Approach B
        eng_ctx = build_prefix_engine(self.fasta_path, engine="embed-contextual", device="cpu")
        self.assertIsInstance(eng_ctx, ContextualPrefixDistanceEngine)
        self.assertEqual(eng_ctx.num_units, 30)

        # Default Engine: Two-Tier (Tier 1 Scalar -> Tier 2 Static 384D)
        eng_def = build_prefix_engine(self.fasta_path)
        self.assertIsInstance(eng_def, TwoTierPrefixDistanceEngine)
        self.assertEqual(eng_def.num_units, 30)
        self.assertEqual(eng_def.N, 4)
        self.assertIsNone(eng_def._tier2)  # Lazy evaluation: not built yet
        # Accessing tier2 triggers on-demand factory
        self.assertIsInstance(eng_def.tier2, EmbeddingPrefixDistanceEngine)
        self.assertIsNotNone(eng_def._tier2)

        # Two-Tier Contextual
        eng_two_ctx = build_prefix_engine(self.fasta_path, engine="two-tier-contextual", device="cpu")
        self.assertIsInstance(eng_two_ctx, TwoTierPrefixDistanceEngine)
        self.assertIsInstance(eng_two_ctx.tier2, ContextualPrefixDistanceEngine)

    def test_two_tier_query_distance_matrix(self):
        eng = build_prefix_engine(self.fasta_path, engine="two-tier")
        d_mat = eng.query_distance_matrix(0, 15)
        self.assertEqual(d_mat.shape, (4, 4))
        self.assertAlmostEqual(d_mat[0, 1], 0.0)
        self.assertGreater(d_mat[0, 2], 0.0)


class TestCLIEngineSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_dataset = (
            "/Users/sergei/Projects/TOGA_MEME/axomeme_repo/benchmarks/"
            "simulated_recombination/single_deep_mid/rep_0/alignment.fasta"
        )

    def test_cli_default_scan(self):
        import subprocess
        # Invoking scan without --engine should default to two-tier
        cmd = [
            "python3", "-m", "rhizaeon.cli", "scan",
            self.manifest_dataset
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertIn("RHIZAEON INFERENCE REPORT", res.stdout)
        self.assertIn("Engine: Two-Tier (Tier 1 Scalar Sieve -> Tier 2 Static 384D", res.stdout)
        self.assertIn("Detected 1 Recombination Breakpoint", res.stdout)
        self.assertIn("[Two-Tier Verified]", res.stdout)

    def test_cli_explicit_two_tier_scan(self):
        import subprocess
        cmd = [
            "python3", "-m", "rhizaeon.cli", "scan",
            self.manifest_dataset,
            "--engine", "two-tier"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertIn("RHIZAEON INFERENCE REPORT", res.stdout)
        self.assertIn("Engine: Two-Tier (Tier 1 Scalar Sieve -> Tier 2 Static 384D", res.stdout)
        self.assertIn("Detected 1 Recombination Breakpoint", res.stdout)

    def test_cli_two_tier_contextual_scan(self):
        import subprocess
        cmd = [
            "python3", "-m", "rhizaeon.cli", "scan",
            self.manifest_dataset,
            "--engine", "two-tier-contextual",
            "--device", "cpu"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertIn("RHIZAEON INFERENCE REPORT", res.stdout)
        self.assertIn("Engine: Two-Tier (Tier 1 Scalar Sieve -> Tier 2 Contextual 384D Transformer)", res.stdout)
        self.assertIn("Detected 1 Recombination Breakpoint", res.stdout)

    def test_cli_scalar_scan(self):
        import subprocess
        cmd = [
            "python3", "-m", "rhizaeon.cli", "scan",
            self.manifest_dataset,
            "--engine", "scalar"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertIn("RHIZAEON INFERENCE REPORT", res.stdout)
        self.assertIn("Engine: scalar", res.stdout)
        self.assertIn("Detected 1 Recombination Breakpoint", res.stdout)

    def test_cli_static_scan(self):
        import subprocess
        cmd = [
            "python3", "-m", "rhizaeon.cli", "scan",
            self.manifest_dataset,
            "--engine", "embed-static"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertIn("RHIZAEON INFERENCE REPORT", res.stdout)
        self.assertIn("Engine: embed-static", res.stdout)
        self.assertIn("Detected 1 Recombination Breakpoint", res.stdout)

    def test_cli_contextual_scan(self):
        import subprocess
        cmd = [
            "python3", "-m", "rhizaeon.cli", "scan",
            self.manifest_dataset,
            "--engine", "embed-contextual",
            "--device", "cpu"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertIn("RHIZAEON INFERENCE REPORT", res.stdout)
        self.assertIn("Engine: embed-contextual", res.stdout)
        self.assertIn("Detected 1 Recombination Breakpoint", res.stdout)


if __name__ == "__main__":
    unittest.main()
