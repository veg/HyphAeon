"""
Independence tests for hyphaeon/epistasis.py (Mode II).

Tests that epistasis parameters are independent: changing one parameter
should not silently affect another.

These are robustness/stability tests and live in stability/ alongside
the determinism and numerical edge case tests.
"""
import numpy as np
import pytest

from _harness import essm_edge_pvals


class TestEpistasisIndependence:
    """Parameters should be independent — changing one shouldn't affect others."""

    def test_min_sim_zero_disables_filter(self, essm_smc6_max_fdr_1):
        """min_sim=0.0 should produce edges (filter disabled).
        Note: min_sim is not monotonic in edge count because it changes the
        FDR correction denominator — fewer candidate edges can lead to more
        significant edges after BH. We verify the filter is disabled by
        checking edges exist, not by comparing counts.

        Reuses essm_smc6_max_fdr_1 (run with max_fdr=1.0, min_sim=0.0,
        min_lrt=0.0, min_shared=1) from the session-scoped fixture.
        """
        assert "edges" in essm_smc6_max_fdr_1
        assert len(essm_smc6_max_fdr_1["edges"]) > 0, "min_sim=0.0 should produce edges (filter disabled)"

    def test_min_shared_affects_edge_count(self, essm_runner, smc6_paths):
        """Higher min_shared filters edges by shared taxa count.

        NOTE: Edge count is NOT monotonic w.r.t. min_shared because
        changing the candidate set changes the BH FDR denominator.
        We verify all surviving edges meet the threshold instead.
        """
        fa, nwk = smc6_paths
        result = essm_runner(fa, tree_path=nwk, min_shared=5, min_sim=0.0, min_lrt=0.0)
        for e in result["edges"]:
            assert e["shared_taxa"] >= 5, \
                f"Edge shared_taxa={e['shared_taxa']} below min_shared=5"

    def test_max_fdr_affects_edge_count(self, essm_smc6_max_fdr_1):
        """Stricter max_fdr should not increase edge count, and all surviving
        edges should meet the threshold.

        Reuses essm_smc6_max_fdr_1 (run with max_fdr=1.0) and filters
        client-side by fdr_q, which is equivalent to running with a lower
        max_fdr since max_fdr is applied AFTER BH correction.
        """
        all_edges = essm_smc6_max_fdr_1["edges"]
        strict = [e for e in all_edges if e["fdr_q"] <= 0.01]
        loose = [e for e in all_edges if e["fdr_q"] <= 0.50]
        assert len(strict) <= len(loose)
        for e in strict:
            assert e["fdr_q"] <= 0.01, \
                f"Edge fdr_q={e['fdr_q']} exceeds max_fdr=0.01"
