"""Unit tests for hyphaeon.epistasis — co-selection network, sector mining, and DMS.

Tests the pure-logic network/sector functions with synthetic arrays (no model
needed) and the full pipeline + CLI with the dummy_weights fixture.
"""
import json
import os
import subprocess
import sys

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from hyphaeon.epistasis import (
    compute_branch_coselection_network,
    extract_epistatic_sectors_tse,
    run_epistatic_analysis,
    run_insilico_selection_dms,
    compute_transformer_attributions,
    CANONICAL_AA_TO_CODON,
)


# ---------------------------------------------------------------------------
# compute_branch_coselection_network — pure logic on synthetic arrays
# ---------------------------------------------------------------------------

class TestComputeBranchCoselectionNetwork:
    def _make_inputs(self, L=5, N=6, seed=42):
        rng = np.random.default_rng(seed)
        attributions = rng.random((L, N)).astype(np.float32)
        lrts = rng.uniform(0.5, 10.0, size=L).astype(np.float32)
        consensus_aas = list("ACDEFGHIKL"[:L])
        return attributions, lrts, consensus_aas

    def test_returns_graph_and_edges(self):
        attr, lrts, aas = self._make_inputs()
        edges, G = compute_branch_coselection_network(
            attr, lrts, [f"t{i}" for i in range(attr.shape[1])], aas
        )
        assert isinstance(G, nx.Graph)
        assert isinstance(edges, list)
        assert G.number_of_nodes() == attr.shape[0]

    def test_edge_structure(self):
        attr, lrts, aas = self._make_inputs(L=6, N=8)
        edges, G = compute_branch_coselection_network(
            attr, lrts, [f"t{i}" for i in range(attr.shape[1])], aas,
            min_sim=0.0, min_shared=1, min_lrt=0.0, max_fdr=1.0,
        )
        for e in edges:
            assert "site_u" in e
            assert "site_v" in e
            assert "similarity" in e
            assert "shared_taxa" in e
            assert "cesi" in e
            assert "fdr_q" in e
            assert 0.0 <= e["similarity"] <= 1.0
            assert 0.0 <= e["fdr_q"] <= 1.0

    def test_min_sim_filter(self):
        attr, lrts, aas = self._make_inputs()
        edges_loose, _ = compute_branch_coselection_network(
            attr, lrts, [f"t{i}" for i in range(attr.shape[1])], aas,
            min_sim=0.0, min_shared=1, min_lrt=0.0, max_fdr=1.0,
        )
        edges_strict, _ = compute_branch_coselection_network(
            attr, lrts, [f"t{i}" for i in range(attr.shape[1])], aas,
            min_sim=0.99, min_shared=1, min_lrt=0.0, max_fdr=1.0,
        )
        assert len(edges_strict) <= len(edges_loose)

    def test_min_lrt_filter(self):
        attr, lrts, aas = self._make_inputs()
        edges_loose, _ = compute_branch_coselection_network(
            attr, lrts, [f"t{i}" for i in range(attr.shape[1])], aas,
            min_sim=0.0, min_shared=1, min_lrt=0.0, max_fdr=1.0,
        )
        edges_strict, _ = compute_branch_coselection_network(
            attr, lrts, [f"t{i}" for i in range(attr.shape[1])], aas,
            min_sim=0.0, min_shared=1, min_lrt=100.0, max_fdr=1.0,
        )
        assert len(edges_strict) <= len(edges_loose)

    def test_empty_attributions(self):
        attr = np.zeros((3, 4), dtype=np.float32)
        lrts = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        aas = ["A", "C", "D"]
        edges, G = compute_branch_coselection_network(
            attr, lrts, ["t0", "t1", "t2", "t3"], aas
        )
        assert edges == []
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 0

    def test_single_site(self):
        attr = np.ones((1, 4), dtype=np.float32)
        lrts = np.array([5.0], dtype=np.float32)
        aas = ["A"]
        edges, G = compute_branch_coselection_network(
            attr, lrts, ["t0", "t1", "t2", "t3"], aas
        )
        assert edges == []
        assert G.number_of_nodes() == 1

    def test_edges_sorted_by_cesi(self):
        attr, lrts, aas = self._make_inputs(L=8, N=10)
        edges, _ = compute_branch_coselection_network(
            attr, lrts, [f"t{i}" for i in range(attr.shape[1])], aas,
            min_sim=0.0, min_shared=1, min_lrt=0.0, max_fdr=1.0,
        )
        if len(edges) > 1:
            cesi_vals = [e["cesi"] for e in edges]
            assert cesi_vals == sorted(cesi_vals, reverse=True)


# ---------------------------------------------------------------------------
# extract_epistatic_sectors_tse — pure logic on synthetic graphs
# ---------------------------------------------------------------------------

class TestExtractEpistaticSectors:
    def test_empty_graph_returns_empty(self):
        G = nx.Graph()
        G.add_node(1, ref="A", lrt=1.0)
        G.add_node(2, ref="C", lrt=1.0)
        attr = np.ones((2, 4), dtype=np.float32)
        lrts = np.array([1.0, 2.0], dtype=np.float32)
        sectors = extract_epistatic_sectors_tse(G, attr, lrts, ["A", "C"])
        assert sectors == []

    def test_simple_graph_produces_sectors(self):
        G = nx.Graph()
        for s in range(1, 6):
            G.add_node(s, ref="A", lrt=2.0)
        G.add_edge(1, 2, weight=0.8, shared=2, cesi=1.0, fdr_q=0.01)
        G.add_edge(2, 3, weight=0.7, shared=2, cesi=0.9, fdr_q=0.01)
        G.add_edge(3, 4, weight=0.6, shared=2, cesi=0.8, fdr_q=0.01)
        G.add_edge(4, 5, weight=0.5, shared=2, cesi=0.7, fdr_q=0.01)

        attr = np.random.default_rng(0).random((5, 6)).astype(np.float32)
        lrts = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        aas = ["A", "C", "D", "E", "F"]

        sectors = extract_epistatic_sectors_tse(G, attr, lrts, aas)
        assert len(sectors) >= 1
        for sec in sectors:
            assert "sector_id" in sec
            assert "size" in sec
            assert "sites" in sec
            assert "spectral_coherence" in sec
            assert 0.0 <= sec["spectral_coherence"] <= 1.0
            assert "pars_signature" in sec

    def test_focal_taxon_signature(self):
        G = nx.Graph()
        for s in range(1, 4):
            G.add_node(s, ref="A", lrt=2.0)
        G.add_edge(1, 2, weight=0.8, shared=2, cesi=1.0, fdr_q=0.01)
        G.add_edge(2, 3, weight=0.7, shared=2, cesi=0.9, fdr_q=0.01)

        attr = np.random.default_rng(1).random((3, 4)).astype(np.float32)
        lrts = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        aas = ["A", "C", "D"]
        taxa = ["bat1", "bat2", "whale1", "fish1"]
        a_np = np.array([[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0]])

        sectors = extract_epistatic_sectors_tse(
            G, attr, lrts, aas,
            focal_taxon="bat1", a_np=a_np, taxa=taxa,
        )
        assert len(sectors) >= 1
        sec = sectors[0]
        assert "focal_taxon" in sec
        assert sec["focal_taxon"] == "bat1"
        assert "focal_signature" in sec
        assert "focal_mutations" in sec

    def test_sectors_sorted_by_coherence(self):
        G = nx.Graph()
        for s in range(1, 7):
            G.add_node(s, ref="A", lrt=2.0)
        G.add_edge(1, 2, weight=0.9, shared=3, cesi=1.5, fdr_q=0.01)
        G.add_edge(3, 4, weight=0.5, shared=2, cesi=0.5, fdr_q=0.01)

        attr = np.random.default_rng(2).random((6, 8)).astype(np.float32)
        lrts = np.full(6, 2.0, dtype=np.float32)
        aas = list("ACDEFG")

        sectors = extract_epistatic_sectors_tse(G, attr, lrts, aas)
        if len(sectors) > 1:
            coherences = [s["spectral_coherence"] for s in sectors]
            assert coherences == sorted(coherences, reverse=True)


# ---------------------------------------------------------------------------
# run_epistatic_analysis — end-to-end with dummy weights
# ---------------------------------------------------------------------------

class TestRunEpistaticAnalysis:
    def test_runs_end_to_end(self, examples_dir, dummy_weights):
        fa = os.path.join(examples_dir, "Smc6.fasta")
        nwk = os.path.join(examples_dir, "Smc6.nwk")
        result = run_epistatic_analysis(
            alignment_path=fa,
            tree_path=nwk,
            weights_path=dummy_weights,
            cpu=True,
            skip_dms=False,
        )
        assert result["taxa_count"] > 0
        assert result["codon_count"] > 0
        assert "edges" in result
        assert "sectors" in result
        assert "plasticity" in result
        assert isinstance(result["edges"], list)
        assert isinstance(result["sectors"], list)
        assert isinstance(result["plasticity"], list)

    def test_skip_dms(self, examples_dir, dummy_weights):
        fa = os.path.join(examples_dir, "Smc6.fasta")
        nwk = os.path.join(examples_dir, "Smc6.nwk")
        result = run_epistatic_analysis(
            alignment_path=fa,
            tree_path=nwk,
            weights_path=dummy_weights,
            cpu=True,
            skip_dms=True,
        )
        assert result["plasticity"] == []

    def test_plasticity_results_well_formed(self, examples_dir, dummy_weights):
        fa = os.path.join(examples_dir, "Smc6.fasta")
        nwk = os.path.join(examples_dir, "Smc6.nwk")
        result = run_epistatic_analysis(
            alignment_path=fa,
            tree_path=nwk,
            weights_path=dummy_weights,
            cpu=True,
        )
        for p in result["plasticity"]:
            assert "site" in p
            assert "wt_aa" in p
            assert "baseline_lrt" in p
            assert p["baseline_lrt"] >= 0.0
            assert "intrinsic_plasticity" in p
            assert p["intrinsic_plasticity"] >= 0.0
            assert "p_value" in p
            assert 0.0 <= p["p_value"] <= 1.0

    def test_backward_compat_aliases(self):
        from hyphaeon.epistasis import (
            run_epistasis_analysis,
            run_epistatic_sector_mining,
            compute_selection_dms_essm,
            extract_epistatic_sectors,
        )
        assert run_epistasis_analysis is run_epistatic_analysis
        assert run_epistatic_sector_mining is run_epistatic_analysis
        assert compute_selection_dms_essm is run_insilico_selection_dms
        assert extract_epistatic_sectors is extract_epistatic_sectors_tse


# ---------------------------------------------------------------------------
# CANONICAL_AA_TO_CODON sanity
# ---------------------------------------------------------------------------

class TestCanonicalCodons:
    def test_all_20_standard_aas_present(self):
        standard = set("ACDEFGHIKLMNPQRSTVWY")
        assert set(CANONICAL_AA_TO_CODON.keys()) == standard

    def test_all_codons_are_valid(self):
        from hyphaeon.dataset import CODON_TO_AA
        for aa, codon in CANONICAL_AA_TO_CODON.items():
            assert codon in CODON_TO_AA, f"{codon} not in CODON_TO_AA"
            assert CODON_TO_AA[codon] == aa, f"{codon} maps to {CODON_TO_AA[codon]}, expected {aa}"


# ---------------------------------------------------------------------------
# CLI integration — hyphaeon.cli epistasis subcommand
# ---------------------------------------------------------------------------

class TestEpistasisCLI:
    def test_epistasis_cli_runs(self, examples_dir, dummy_weights, tmp_path):
        fa = os.path.join(examples_dir, "Smc6.fasta")
        nwk = os.path.join(examples_dir, "Smc6.nwk")
        out_json = str(tmp_path / "epi.json")
        out_csv = str(tmp_path / "epi.csv")
        out_graphml = str(tmp_path / "network.graphml")

        cmd = [
            sys.executable, "-m", "hyphaeon.cli", "epistasis",
            "-a", fa, "-t", nwk, "-w", dummy_weights,
            "-o", out_json, "-c", out_csv,
            "--graphml", out_graphml,
            "--cpu",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"epistasis CLI failed: {result.stderr}"
        assert os.path.exists(out_json)

        with open(out_json) as f:
            data = json.load(f)
        assert "taxa_count" in data
        assert "codon_count" in data
        assert "edges" in data
        assert "sectors" in data
