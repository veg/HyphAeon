"""Unit tests for hyphaeon.phenotype — resolve_phenotype_vector and run_phenotype_association.

Tests the pure-logic phenotype vector resolution (no model weights needed) and
the full run_phenotype_association pipeline (uses dummy_weights fixture).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from hyphaeon.phenotype import resolve_phenotype_vector, run_phenotype_association, PRESETS


# ---------------------------------------------------------------------------
# resolve_phenotype_vector — pure logic, no model needed
# ---------------------------------------------------------------------------

class TestResolvePhenotypeVectorPresets:
    def test_preset_matches_known_taxa(self):
        taxa = ["turTru", "delLeu", "orcOrc", "musMus", "homSap"]
        y, meta = resolve_phenotype_vector(taxa, preset="echolocation")
        assert y.shape == (5,)
        assert y[0] == 1.0  # turTru matches turTru*
        assert y[1] == 1.0  # delLeu matches delLeu*
        assert y[2] == 1.0  # orcOrc matches orcOrc*
        assert y[3] == 0.0  # musMus not in echolocation
        assert y[4] == 0.0  # homSap not in echolocation
        assert meta["mode"] == "discrete"
        assert meta["foreground_count"] == 3
        assert meta["background_count"] == 2

    def test_preset_case_insensitive(self):
        taxa = ["TURTRU", "MUSMUS"]
        y, meta = resolve_phenotype_vector(taxa, preset="echolocation")
        assert y[0] == 1.0
        assert y[1] == 0.0

    def test_preset_hyphen_normalised(self):
        taxa = ["bosGru", "musMus"]
        y, _ = resolve_phenotype_vector(taxa, preset="high-altitude")
        assert y[0] == 1.0  # bosGru matches bosGru* in high_altitude preset

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            resolve_phenotype_vector(["a"], preset="nonexistent")

    def test_all_presets_have_required_keys(self):
        for key, info in PRESETS.items():
            assert "title" in info
            assert "description" in info
            assert "foreground" in info
            assert isinstance(info["foreground"], list)
            assert len(info["foreground"]) > 0


class TestResolvePhenotypeVectorInline:
    def test_inline_string_patterns(self):
        taxa = ["bat1", "bat2", "whale1", "fish1"]
        y, meta = resolve_phenotype_vector(taxa, foreground="bat*,whale*")
        assert y[0] == 1.0
        assert y[1] == 1.0
        assert y[2] == 1.0
        assert y[3] == 0.0
        assert meta["foreground_count"] == 3

    def test_inline_list_patterns(self):
        taxa = ["bat1", "bat2", "fish1"]
        y, _ = resolve_phenotype_vector(taxa, foreground=["bat1", "bat2"])
        assert y[0] == 1.0
        assert y[1] == 1.0
        assert y[2] == 0.0

    def test_inline_substring_match(self):
        taxa = ["Microbat_long", "Whale_species", "Mouse"]
        y, _ = resolve_phenotype_vector(taxa, foreground="bat")
        assert y[0] == 1.0  # "bat" is substring of "Microbat_long"
        assert y[1] == 0.0
        assert y[2] == 0.0

    def test_no_match_returns_all_zeros(self):
        taxa = ["cat1", "dog1"]
        y, meta = resolve_phenotype_vector(taxa, foreground="bat*")
        assert np.all(y == 0.0)
        assert meta["foreground_count"] == 0


class TestResolvePhenotypeVectorFile:
    def test_csv_discrete(self, tmp_path):
        df = pd.DataFrame({
            "species": ["seq1", "seq2", "seq3"],
            "trait": [1, 0, 1],
        })
        p = tmp_path / "pheno.csv"
        df.to_csv(p, index=False)
        y, meta = resolve_phenotype_vector(
            ["seq1", "seq2", "seq3"], phenotype_file=str(p)
        )
        assert y[0] == 1.0
        assert y[1] == 0.0
        assert y[2] == 1.0
        assert meta["mode"] == "discrete"
        assert meta["foreground_count"] == 2

    def test_tsv_discrete(self, tmp_path):
        df = pd.DataFrame({
            "taxon": ["seq1", "seq2"],
            "is_foreground": ["yes", "no"],
        })
        p = tmp_path / "pheno.tsv"
        df.to_csv(p, index=False, sep="\t")
        y, meta = resolve_phenotype_vector(
            ["seq1", "seq2"], phenotype_file=str(p)
        )
        assert y[0] == 1.0
        assert y[1] == 0.0

    def test_continuous_zscored(self, tmp_path):
        df = pd.DataFrame({
            "species": ["a", "b", "c", "d"],
            "body_mass": [10.0, 50.0, 100.0, 200.0],
        })
        p = tmp_path / "pheno.csv"
        df.to_csv(p, index=False)
        y, meta = resolve_phenotype_vector(
            ["a", "b", "c", "d"], phenotype_file=str(p), continuous=True
        )
        assert meta["mode"] == "continuous"
        assert abs(np.mean(y)) < 1e-10  # z-scored → mean ~0
        assert abs(np.std(y) - 1.0) < 1e-10  # z-scored → std ~1

    def test_auto_detect_species_col(self, tmp_path):
        df = pd.DataFrame({
            "assembly": ["seq1", "seq2"],
            "trait": [1, 0],
        })
        p = tmp_path / "pheno.csv"
        df.to_csv(p, index=False)
        y, _ = resolve_phenotype_vector(["seq1", "seq2"], phenotype_file=str(p))
        assert y[0] == 1.0
        assert y[1] == 0.0

    def test_explicit_column_names(self, tmp_path):
        df = pd.DataFrame({
            "org": ["seq1", "seq2"],
            "val": [1, 0],
        })
        p = tmp_path / "pheno.csv"
        df.to_csv(p, index=False)
        y, _ = resolve_phenotype_vector(
            ["seq1", "seq2"],
            phenotype_file=str(p),
            species_col="org",
            trait_col="val",
        )
        assert y[0] == 1.0
        assert y[1] == 0.0

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            resolve_phenotype_vector(["a"], phenotype_file="/nonexistent/file.csv")

    def test_case_insensitive_taxa_match(self, tmp_path):
        df = pd.DataFrame({
            "species": ["SEQ1", "SEQ2"],
            "trait": [1, 0],
        })
        p = tmp_path / "pheno.csv"
        df.to_csv(p, index=False)
        y, _ = resolve_phenotype_vector(["seq1", "seq2"], phenotype_file=str(p))
        assert y[0] == 1.0
        assert y[1] == 0.0


class TestResolvePhenotypeVectorErrors:
    def test_no_arguments_raises(self):
        with pytest.raises(ValueError, match="Must provide"):
            resolve_phenotype_vector(["a", "b"])

    def test_empty_taxa_list(self):
        y, meta = resolve_phenotype_vector([], foreground="bat*")
        assert y.shape == (0,)
        assert meta["foreground_count"] == 0


# ---------------------------------------------------------------------------
# run_phenotype_association — end-to-end with dummy weights
# ---------------------------------------------------------------------------

class TestRunPhenotypeAssociation:
    def test_runs_with_inline_foreground(self, examples_dir, dummy_weights, tmp_path):
        fa = os.path.join(examples_dir, "Smc6.fasta")
        nwk = os.path.join(examples_dir, "Smc6.nwk")
        result = run_phenotype_association(
            alignment_path=fa,
            tree_path=nwk,
            weights_path=dummy_weights,
            foreground="homSap*,panTro*,panPan*",
            cpu=True,
        )
        assert result["taxa_count"] > 0
        assert result["codon_count"] > 0
        assert isinstance(result["sites"], list)
        assert len(result["sites"]) > 0
        assert len(result["sites"]) <= result["codon_count"]
        assert "spectral_energy" in result
        assert "norm_spectral_ratio" in result
        assert "compact_pars_signature" in result
        assert "significant_sites_count" in result

    def test_site_results_well_formed(self, examples_dir, dummy_weights):
        fa = os.path.join(examples_dir, "Smc6.fasta")
        nwk = os.path.join(examples_dir, "Smc6.nwk")
        result = run_phenotype_association(
            alignment_path=fa,
            tree_path=nwk,
            weights_path=dummy_weights,
            foreground="homSap*,panTro*",
            cpu=True,
        )
        for site in result["sites"]:
            assert "site" in site
            assert "ref_aa" in site
            assert "derived_aa" in site
            assert "p_value" in site
            assert 0.0 <= site["p_value"] <= 1.0
            assert "q_value" in site
            assert 0.0 <= site["q_value"] <= 1.0
            assert "association_rho" in site
            assert -1.0 <= site["association_rho"] <= 1.0
            assert "score" in site

    def test_sites_sorted_by_score_descending(self, examples_dir, dummy_weights):
        fa = os.path.join(examples_dir, "Smc6.fasta")
        nwk = os.path.join(examples_dir, "Smc6.nwk")
        result = run_phenotype_association(
            alignment_path=fa,
            tree_path=nwk,
            weights_path=dummy_weights,
            foreground="homSap*,panTro*",
            cpu=True,
        )
        scores = [s["score"] for s in result["sites"]]
        assert scores == sorted(scores, reverse=True)

    def test_insufficient_foreground_raises(self, examples_dir, dummy_weights):
        fa = os.path.join(examples_dir, "Smc6.fasta")
        nwk = os.path.join(examples_dir, "Smc6.nwk")
        with pytest.raises(ValueError, match="Insufficient foreground"):
            run_phenotype_association(
                alignment_path=fa,
                tree_path=nwk,
                weights_path=dummy_weights,
                foreground="zzzNonexistent*",
                cpu=True,
            )


# ---------------------------------------------------------------------------
# CLI integration — hyphaeon.cli phenotype subcommand
# ---------------------------------------------------------------------------

class TestPhenotypeCLI:
    def test_phenotype_cli_runs(self, examples_dir, dummy_weights, tmp_path):
        fa = os.path.join(examples_dir, "Smc6.fasta")
        nwk = os.path.join(examples_dir, "Smc6.nwk")
        out_json = str(tmp_path / "pheno.json")
        out_csv = str(tmp_path / "pheno.csv")

        cmd = [
            sys.executable, "-m", "hyphaeon.cli", "phenotype",
            "-a", fa, "-t", nwk, "-w", dummy_weights,
            "-fg", "homSap*,panTro*,panPan*",
            "-o", out_json, "-c", out_csv,
            "--cpu",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"phenotype CLI failed: {result.stderr}"
        assert os.path.exists(out_json)
        assert os.path.exists(out_csv)

        with open(out_json) as f:
            data = json.load(f)
        assert "taxa_count" in data
        assert "codon_count" in data
        assert "sites" in data
        assert len(data["sites"]) > 0
        assert len(data["sites"]) <= data["codon_count"]

        df = pd.read_csv(out_csv)
        assert "site" in df.columns
        assert "p_value" in df.columns
        assert len(df) == len(data["sites"])


# ---------------------------------------------------------------------------
# Permulations — Brownian motion phylogenetic null model
# ---------------------------------------------------------------------------

class TestPermulations:
    def test_compute_phylogenetic_covariance_structure(self):
        import io
        from Bio import Phylo
        from hyphaeon.phenotype import compute_phylogenetic_covariance, generate_permulations

        nwk = "(((A:0.1,B:0.1):0.1,C:0.2):0.2,(D:0.3,E:0.3):0.1);"
        tree = Phylo.read(io.StringIO(nwk), "newick")
        taxa = ["A", "B", "C", "D", "E"]

        V = compute_phylogenetic_covariance(tree, taxa)
        assert V.shape == (5, 5)
        # Symmetry
        np.testing.assert_allclose(V, V.T)
        # Shared distance between A and B should be greater than between A and D
        assert V[0, 1] > V[0, 3]

    def test_generate_permulations_binary(self):
        import io
        from Bio import Phylo
        from hyphaeon.phenotype import generate_permulations

        nwk = "((A:0.1,B:0.1):0.2,(C:0.2,D:0.2):0.1);"
        tree = Phylo.read(io.StringIO(nwk), "newick")
        taxa = ["A", "B", "C", "D"]
        y_binary = np.array([1.0, 1.0, 0.0, 0.0])

        perms = generate_permulations(y_binary, tree, taxa, n_perm=50, seed=123)
        assert perms.shape == (50, 4)
        # Each permulation must preserve exact foreground count
        for p in range(50):
            assert np.sum(perms[p] == 1.0) == 2
            assert np.sum(perms[p] == 0.0) == 2

    def test_generate_permulations_continuous(self):
        import io
        from Bio import Phylo
        from hyphaeon.phenotype import generate_permulations

        nwk = "((A:0.1,B:0.1):0.2,(C:0.2,D:0.2):0.1);"
        tree = Phylo.read(io.StringIO(nwk), "newick")
        taxa = ["A", "B", "C", "D"]
        y_cont = np.array([10.5, 5.2, -1.3, 0.0])

        perms = generate_permulations(y_cont, tree, taxa, n_perm=20, seed=456)
        assert perms.shape == (20, 4)
        # Each permulation must preserve exact set of values
        sorted_orig = np.sort(y_cont)
        for p in range(20):
            np.testing.assert_allclose(np.sort(perms[p]), sorted_orig)

    def test_run_phenotype_association_with_permulations(self, examples_dir, dummy_weights):
        fa = os.path.join(examples_dir, "Smc6.fasta")
        nwk = os.path.join(examples_dir, "Smc6.nwk")

        res = run_phenotype_association(
            alignment_path=fa,
            tree_path=nwk,
            weights_path=dummy_weights,
            foreground="homSap*,panTro*,panPan*",
            permulations=50,
            cpu=True,
        )
        assert res["permulations_count"] == 50
        assert "gene_p_value_perm" in res
        assert len(res["sites"]) > 0
        first_site = res["sites"][0]
        assert "p_assoc_perm" in first_site
        assert first_site["p_assoc_perm"] is not None
        assert 0.0 <= first_site["p_assoc_perm"] <= 1.0

