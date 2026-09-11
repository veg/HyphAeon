"""
tests/test_splits.py
--------------------
Unit and integration tests for Spectral Graph Bisection via Cross-Taxa Attention Maps (hyphaeon.splits).
"""

import io
import os
import subprocess
import sys
import numpy as np
import pytest
from Bio import Phylo

from hyphaeon.splits import (
    run_spectral_splits,
    spectral_bisection,
    compute_fused_affinity_matrix,
    tree_dict_to_newick,
    get_all_clade_taxa,
)

try:
    import tn93  # noqa: F401
    _HAS_TN93 = True
except ImportError:
    _HAS_TN93 = False


@pytest.fixture
def examples_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples"))


@pytest.fixture
def real_weights_path():
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights", "hyphaeon_v1.pt")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights", "axomeme_v1.pt")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model.safetensors")),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class TestFusedAffinityMatrix:
    """Unit tests for the multiplicative affinity tensor fusion."""

    def test_symmetry_and_diagonal_zeroing(self):
        n = 8
        rng = np.random.RandomState(42)
        attn = rng.uniform(0.01, 0.5, size=(n, n))
        mds = rng.normal(0, 1, size=(n, 4))
        taxa_repr = rng.normal(0, 1, size=(n, 16))

        A = compute_fused_affinity_matrix(attn, mds, taxa_repr)

        assert A.shape == (n, n)
        # Symmetry: A = A^T
        assert np.allclose(A, A.T, atol=1e-6)
        # Zero diagonal
        assert np.all(np.diag(A) == 0.0)
        # Non-negativity and finite
        assert np.all(A >= 0.0)
        assert np.all(np.isfinite(A))

    def test_without_taxa_repr(self):
        n = 6
        rng = np.random.RandomState(42)
        attn = rng.uniform(0.01, 0.5, size=(n, n))
        mds = rng.normal(0, 1, size=(n, 4))

        A = compute_fused_affinity_matrix(attn, mds, taxa_repr=None)
        assert A.shape == (n, n)
        assert np.allclose(A, A.T, atol=1e-6)
        assert np.all(np.diag(A) == 0.0)
        assert np.all(A >= 0.0)

    def test_degenerate_zero_distance_mds(self):
        n = 5
        attn = np.ones((n, n), dtype=np.float64)
        # All MDS coordinates identical -> distances are 0
        mds = np.zeros((n, 4), dtype=np.float64)

        A = compute_fused_affinity_matrix(attn, mds)
        assert A.shape == (n, n)
        assert np.all(np.isfinite(A))
        assert np.all(np.diag(A) == 0.0)


class TestSpectralBisection:
    """Unit tests for recursive spectral graph bisection and Normalized Cuts."""

    def test_synthetic_two_clusters(self):
        taxa = [f"sp_{i}" for i in range(10)]
        # Block diagonal: cluster 1 (0..4), cluster 2 (5..9)
        A = np.zeros((10, 10))
        A[:5, :5] = 1.0
        A[5:, 5:] = 1.0
        # Add small inter-cluster noise
        A[:5, 5:] = 0.02
        A[5:, :5] = 0.02
        np.fill_diagonal(A, 0.0)

        res = spectral_bisection(A, taxa, min_clade_size=2)
        assert res["type"] == "node"
        assert res["eigengap"] > 0.5  # Strong spectral gap
        assert res["fiedler_val"] > 0.0

        left = get_all_clade_taxa(res["left"])
        right = get_all_clade_taxa(res["right"])

        c1 = set(taxa[:5])
        c2 = set(taxa[5:])
        assert (set(left) == c1 and set(right) == c2) or (set(left) == c2 and set(right) == c1)

    def test_min_clade_size_termination(self):
        taxa = ["t1", "t2", "t3"]
        A = np.ones((3, 3))
        np.fill_diagonal(A, 0.0)

        # n <= min_clade_size -> leaf immediately
        res = spectral_bisection(A, taxa, min_clade_size=3)
        assert res["type"] == "leaf"
        assert res["taxa"] == taxa

    def test_max_depth_termination(self):
        taxa = [f"sp_{i}" for i in range(8)]
        A = np.ones((8, 8))
        np.fill_diagonal(A, 0.0)

        # max_depth=0 -> returns leaf at root
        res0 = spectral_bisection(A, taxa, max_depth=0)
        assert res0["type"] == "leaf"

        # max_depth=1 -> splits once, children are leaves
        res1 = spectral_bisection(A, taxa, max_depth=1)
        assert res1["type"] == "node"
        assert res1["left"]["type"] == "leaf"
        assert res1["right"]["type"] == "leaf"

    def test_disconnected_components(self):
        taxa = ["a1", "a2", "b1", "b2"]
        A = np.zeros((4, 4))
        A[0, 1] = A[1, 0] = 1.0
        A[2, 3] = A[3, 2] = 1.0

        res = spectral_bisection(A, taxa, min_clade_size=2)
        assert res["type"] == "node"
        # For disconnected graph, algebraic connectivity (lambda_2) is 0
        assert np.isclose(res["fiedler_val"], 0.0, atol=1e-5)
        assert np.isclose(res["cut_weight"], 0.0, atol=1e-5)

        left = set(get_all_clade_taxa(res["left"]))
        right = set(get_all_clade_taxa(res["right"]))
        assert (left == {"a1", "a2"} and right == {"b1", "b2"}) or (left == {"b1", "b2"} and right == {"a1", "a2"})

    def test_uniform_indicator_fallback(self):
        """Tests that when indicator signs are degenerate, median cut prevents empty clades."""
        taxa = ["t1", "t2", "t3", "t4"]
        A = np.ones((4, 4))
        np.fill_diagonal(A, 0.0)

        res = spectral_bisection(A, taxa, min_clade_size=2)
        assert res["type"] == "node"
        left = get_all_clade_taxa(res["left"])
        right = get_all_clade_taxa(res["right"])
        assert len(left) > 0
        assert len(right) > 0
        assert len(left) + len(right) == 4


class TestTreeSerialization:
    """Unit tests for converting split dictionaries to formatted Newick strings."""

    def test_tree_dict_to_newick_single_leaf(self):
        leaf = {"type": "leaf", "taxa": ["taxonA"]}
        assert tree_dict_to_newick(leaf) == "taxonA"

    def test_tree_dict_to_newick_multi_taxa_leaf(self):
        leaf = {"type": "leaf", "taxa": ["taxonA", "taxonB"]}
        assert tree_dict_to_newick(leaf) == "(taxonA,taxonB)"

    def test_tree_dict_to_newick_nested_and_valid(self):
        tree = {
            "type": "node",
            "eigengap": 0.25,
            "left": {
                "type": "node",
                "eigengap": 0.15,
                "left": {"type": "leaf", "taxa": ["A"]},
                "right": {"type": "leaf", "taxa": ["B"]},
            },
            "right": {
                "type": "leaf",
                "taxa": ["C", "D"],
            },
        }
        nwk = tree_dict_to_newick(tree) + ";"
        assert nwk.startswith("(")
        assert nwk.endswith(";")

        # Validate with Bio.Phylo
        parsed = Phylo.read(io.StringIO(nwk), "newick")
        terminal_names = {t.name for t in parsed.get_terminals()}
        assert terminal_names == {"A", "B", "C", "D"}

    def test_get_all_clade_taxa(self):
        tree = {
            "type": "node",
            "left": {
                "type": "node",
                "left": {"type": "leaf", "taxa": ["t1"]},
                "right": {"type": "leaf", "taxa": ["t2", "t3"]},
            },
            "right": {"type": "leaf", "taxa": ["t4"]},
        }
        all_taxa = get_all_clade_taxa(tree)
        assert set(all_taxa) == {"t1", "t2", "t3", "t4"}
        assert len(all_taxa) == 4


class TestRunSpectralSplits:
    """End-to-end tests for run_spectral_splits with dummy and pretrained weights."""

    def test_run_with_dummy_weights(self, fasta_file, newick_file, dummy_weights):
        res = run_spectral_splits(
            alignment_path=fasta_file,
            tree_path=newick_file,
            weights_path=dummy_weights,
            device="cpu"
        )
        assert isinstance(res, dict)
        assert "newick" in res
        assert res["newick"].endswith(";")
        assert "fiedler_val" in res
        assert "eigengap" in res
        assert "root_split" in res
        assert "taxa" in res
        assert len(res["taxa"]) == 3
        assert res["root_split"]["left_count"] + res["root_split"]["right_count"] == 3

        # Validate generated Newick tree
        parsed = Phylo.read(io.StringIO(res["newick"]), "newick")
        assert len(parsed.get_terminals()) == 3

    @pytest.mark.skipif(not _HAS_TN93, reason="tn93 package not installed")
    def test_run_tree_free_dummy(self, fasta_file, dummy_weights):
        res = run_spectral_splits(
            alignment_path=fasta_file,
            use_tn93=True,
            weights_path=dummy_weights,
            device="cpu"
        )
        assert "newick" in res
        assert res["newick"].endswith(";")
        assert res["root_split"]["left_count"] + res["root_split"]["right_count"] == 3

    def test_bat_oas1_spectral_splits(self, examples_dir, real_weights_path):
        """Test spectral bisection on bat OAS1 with reference tree."""
        if real_weights_path is None:
            pytest.skip("Pretrained model weights not found")
        fa = os.path.join(examples_dir, "bat_oas1.fasta")
        nwk = os.path.join(examples_dir, "bat_oas1.nwk")
        if not os.path.exists(fa) or not os.path.exists(nwk):
            pytest.skip("bat_oas1 example files not found")

        res = run_spectral_splits(fa, tree_path=nwk, weights_path=real_weights_path)
        assert "newick" in res
        assert res["eigengap"] > 0.0

        yin = {"M_lyra", "H_arm", "R_sin", "R_ferr", "P_gig", "P_vamp", "P_alec", "R_aeg"}
        yang = {"S_hond", "A_jam", "D_rot", "P_kuhl", "E_fusc", "M_myot", "M_bran", "M_luc", "M_nat", "M_mol"}

        left = set(res["root_split"]["left_clade"])
        right = set(res["root_split"]["right_clade"])

        # Must match biological subordinal split 100%
        assert (left == yin and right == yang) or (left == yang and right == yin)

    @pytest.mark.skipif(not _HAS_TN93, reason="tn93 package not installed")
    def test_bat_oas1_tree_free_spectral_splits(self, examples_dir, real_weights_path):
        """Test spectral bisection in TREE-FREE mode (TN93 only, no tree input)."""
        if real_weights_path is None:
            pytest.skip("Pretrained model weights not found")
        fa = os.path.join(examples_dir, "bat_oas1.fasta")
        if not os.path.exists(fa):
            pytest.skip("bat_oas1.fasta not found")

        res = run_spectral_splits(fa, use_tn93=True, weights_path=real_weights_path)
        assert "newick" in res

        yin = {"M_lyra", "H_arm", "R_sin", "R_ferr", "P_gig", "P_vamp", "P_alec", "R_aeg"}
        yang = {"S_hond", "A_jam", "D_rot", "P_kuhl", "E_fusc", "M_myot", "M_bran", "M_luc", "M_nat", "M_mol"}

        left = set(res["root_split"]["left_clade"])
        right = set(res["root_split"]["right_clade"])

        assert (left == yin and right == yang) or (left == yang and right == yin)


class TestSplitsCLI:
    """Integration tests for the 'hyphaeon splits' CLI subcommand."""

    def test_cli_splits_subcommand_with_dummy_weights(self, examples_dir, dummy_weights, tmp_path):
        fa = os.path.join(examples_dir, "bat_oas1.fasta")
        nwk = os.path.join(examples_dir, "bat_oas1.nwk")
        if not os.path.exists(fa) or not os.path.exists(nwk):
            pytest.skip("bat_oas1 example files not found")

        out_nwk = str(tmp_path / "out_tree.nwk")
        out_csv = str(tmp_path / "out_clades.csv")

        cmd = [
            sys.executable, "-m", "hyphaeon.cli",
            "splits",
            "-a", fa,
            "-t", nwk,
            "-w", dummy_weights,
            "-o", out_nwk,
            "-c", out_csv,
            "--cpu",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"CLI stderr: {res.stderr}"
        assert "Spectral Graph Bisection Complete" in res.stdout

        # Verify Newick output
        assert os.path.exists(out_nwk)
        assert os.path.getsize(out_nwk) > 0
        with open(out_nwk) as f:
            nwk_content = f.read().strip()
        parsed = Phylo.read(io.StringIO(nwk_content), "newick")
        assert len(parsed.get_terminals()) == 18

        # Verify CSV output
        assert os.path.exists(out_csv)
        assert os.path.getsize(out_csv) > 0
        import pandas as pd
        df = pd.read_csv(out_csv)
        assert list(df.columns) == ["taxon", "primary_clade", "split_eigengap"]
        assert len(df) == 18
        assert set(df["primary_clade"].unique()).issubset({"Left", "Right"})

    @pytest.mark.skipif(not _HAS_TN93, reason="tn93 package not installed")
    def test_cli_splits_tree_free(self, examples_dir, dummy_weights, tmp_path):
        fa = os.path.join(examples_dir, "bat_oas1.fasta")
        if not os.path.exists(fa):
            pytest.skip("bat_oas1.fasta not found")

        out_tf_nwk = str(tmp_path / "out_tf_tree.nwk")
        out_tf_csv = str(tmp_path / "out_tf_clades.csv")

        cmd = [
            sys.executable, "-m", "hyphaeon.cli",
            "splits",
            "-a", fa,
            "--no-tree",
            "-w", dummy_weights,
            "-o", out_tf_nwk,
            "-c", out_tf_csv,
            "--cpu",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"CLI stderr: {res.stderr}"
        assert os.path.exists(out_tf_nwk)
        assert os.path.exists(out_tf_csv)

