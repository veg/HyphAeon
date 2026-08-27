"""
Output contract tests for hyphaeon/epistasis.py (Mode II).

Tests the co-selection network construction, CESI scoring, sector mining,
and output format of the neural transformer attribution pipeline.

Uses the Smc6 example dataset (20 taxa, shallow tree).
"""
import numpy as np
import pytest

from _harness import essm_edge_pvals


class TestCoSelectionNetwork:
    """Co-selection edges should have valid CESI, similarity, and p-values."""

    def test_edges_have_required_fields(self, essm_smc6_result):
        edges = essm_smc6_result["edges"]
        for e in edges:
            assert "site_u" in e and "site_v" in e
            assert "similarity" in e
            assert "cesi" in e
            assert "p_val" in e
            assert "fdr_q" in e
            assert "shared_taxa" in e

    def test_similarity_in_unit_interval(self, essm_smc6_result):
        for e in essm_smc6_result["edges"]:
            assert -1.0 - 1e-6 <= e["similarity"] <= 1.0 + 1e-6

    def test_cesi_non_negative(self, essm_smc6_result):
        """CESI = sim * sqrt(lrt_u * lrt_v) — should be non-negative for significant edges."""
        for e in essm_smc6_result["edges"]:
            assert e["cesi"] >= 0.0, f"Negative CESI {e['cesi']}"

    def test_p_values_in_unit_interval(self, essm_smc6_result):
        pvals = essm_edge_pvals(essm_smc6_result)
        for p in pvals:
            assert 0.0 <= p <= 1.0

    def test_fdr_q_in_unit_interval(self, essm_smc6_result):
        for e in essm_smc6_result["edges"]:
            assert 0.0 <= e["fdr_q"] <= 1.0

    def test_edges_sorted_by_cesi_descending(self, essm_smc6_result):
        cesis = [e["cesi"] for e in essm_smc6_result["edges"]]
        for i in range(1, len(cesis)):
            assert cesis[i] <= cesis[i-1] + 1e-10


class TestSectors:
    """Epistatic sector mining via greedy modularity (TSE)."""

    def test_sectors_have_required_fields(self, essm_smc6_result):
        for sector in essm_smc6_result["sectors"]:
            assert "sites" in sector
            assert "size" in sector
            assert sector["size"] == len(sector["sites"])

    def test_sector_sites_are_valid(self, essm_smc6_result):
        total_sites = essm_smc6_result["codon_count"]
        for sector in essm_smc6_result["sectors"]:
            sites = sector["sites"]
            for s in sites:
                assert 1 <= s <= total_sites, f"Site {s} out of range [1, {total_sites}]"
        # Sites within a sector should be unique
        for sector in essm_smc6_result["sectors"]:
            sites = sector["sites"]
            assert len(sites) == len(set(sites)), \
                f"Sector {sector.get('sector_id', '?')} has duplicate sites"

    def test_sectors_have_spectral_coherence(self, essm_smc6_result):
        """Each sector should have a spectral_coherence value in [0, 1]."""
        for sector in essm_smc6_result["sectors"]:
            assert "spectral_coherence" in sector
            assert 0.0 <= sector["spectral_coherence"] <= 1.0


class TestEdgeFilters:
    """min_sim and min_lrt filters should actually filter.

    NOTE: Edge count is NOT monotonic w.r.t. these thresholds because
    changing the candidate set changes the BH FDR denominator. With fewer
    candidates (strict filter), FDR is less stringent, which can rescue
    edges that would fail with more candidates. So we verify that all
    surviving edges meet the threshold, not that strict produces fewer.
    """

    def test_min_sim_filters_edges(self, essm_runner, smc6_paths):
        fa, nwk = smc6_paths
        result = essm_runner(fa, tree_path=nwk, min_sim=0.8)
        for e in result["edges"]:
            assert e["similarity"] >= 0.8 - 1e-10, \
                f"Edge sim={e['similarity']:.4f} below min_sim=0.8"

    def test_min_lrt_filters_edges(self, essm_runner, smc6_paths):
        fa, nwk = smc6_paths
        result = essm_runner(fa, tree_path=nwk, min_lrt=10.0)
        for e in result["edges"]:
            assert e.get("lrt_u", 0) >= 10.0 - 1e-10 or e.get("lrt_v", 0) >= 10.0 - 1e-10, \
                f"Edge lrts ({e.get('lrt_u')}, {e.get('lrt_v')}) both below min_lrt=10.0"


class TestTreePathUsage:
    """tree_path is now consumed by Mode II (was dead in Mode I)."""

    def test_tree_path_works(self, essm_smc6_result):
        """Verify the runner works with explicit tree path."""
        assert essm_smc6_result["taxa_count"] > 0
        assert essm_smc6_result["codon_count"] > 0


class TestDMS:
    """Digital DMS (deep mutational scanning) sweep."""

    def test_dms_can_be_disabled(self, essm_runner, smc6_paths):
        fa, nwk = smc6_paths
        result = essm_runner(fa, tree_path=nwk, skip_dms=True)
        assert len(result["plasticity"]) == 0
        assert len(result["selection_dms_plasticity"]) == 0

    def test_dms_produces_results(self, essm_smc6_result):
        assert "plasticity" in essm_smc6_result
        assert "selection_dms_plasticity" in essm_smc6_result
        assert len(essm_smc6_result["plasticity"]) > 0, "DMS plasticity is empty — no sites produced"
        assert len(essm_smc6_result["selection_dms_plasticity"]) > 0, "DMS selection plasticity is empty"


class TestEpistasisCrossDataset:
    """Epistasis output contract should hold across different datasets, not
    just Smc6 (20 taxa, shallow tree). bat_oas1 (18 taxa, ultra-deep) and
    camelid (212 taxa, topology-only) stress different code paths.
    """

    def _check_edge_contract(self, result):
        assert "edges" in result
        assert result["codon_count"] > 0
        for e in result["edges"]:
            assert "site_u" in e and "site_v" in e
            assert "similarity" in e
            assert "cesi" in e
            assert "p_val" in e
            assert "fdr_q" in e
            assert -1.0 - 1e-6 <= e["similarity"] <= 1.0 + 1e-6
            assert e["cesi"] >= 0.0
            assert 0.0 <= e["p_val"] <= 1.0
            assert 0.0 <= e["fdr_q"] <= 1.0
            assert e["site_u"] != e["site_v"]

    def test_bat_oas1_produces_valid_edges(self, essm_runner, bat_oas1_paths):
        fa, nwk = bat_oas1_paths
        result = essm_runner(fa, tree_path=nwk)
        self._check_edge_contract(result)

    def test_camelid_produces_valid_edges(self, essm_runner, camelid_paths):
        fa, nwk = camelid_paths
        result = essm_runner(fa, tree_path=nwk)
        self._check_edge_contract(result)
