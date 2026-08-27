"""
Determinism tests for BUSTED omnibus inference.

Verifies that running BUSTED twice on the same alignment produces
identical results (deterministic inference in eval mode).

KNOWN ISSUE: The BUSTED head (BustedMultiTaskHead) is non-deterministic
across runs because the current weights checkpoint does not include
head_busted.* parameters. The head is randomly initialized each time
via torch.randn in __init__, producing different outputs on each call.
These tests are xfailed until the checkpoint includes BUSTED head weights.
"""
import numpy as np
import pytest


@pytest.mark.xfail(reason="BUSTED head weights not in checkpoint — random init makes output non-deterministic")
class TestBustedDeterminism:
    """BUSTED inference should be deterministic in eval mode (no dropout, no sampling)."""

    @pytest.fixture(scope="class")
    def busted_pair(self, busted_runner, smc6_paths):
        """Run BUSTED twice on the same alignment, shared across both tests."""
        fa, nwk = smc6_paths
        result1 = busted_runner(fa, tree_path=nwk)
        result2 = busted_runner(fa, tree_path=nwk)
        return result1, result2

    def test_same_alignment_same_result(self, busted_pair):
        """Running BUSTED twice on the same alignment should produce identical output."""
        result1, result2 = busted_pair

        r1, r2 = result1[0], result2[0]
        assert r1["p_value_acat"] == pytest.approx(r2["p_value_acat"], rel=1e-10)
        assert r1["p_value_simes"] == pytest.approx(r2["p_value_simes"], rel=1e-10)
        assert r1["selection_probability"] == pytest.approx(r2["selection_probability"], rel=1e-6)
        assert r1["predicted_gene_lrt"] == pytest.approx(r2["predicted_gene_lrt"], rel=1e-6)
        assert r1["omnibus_lrt"] == pytest.approx(r2["omnibus_lrt"], rel=1e-6)
        assert r1["sig_sites_p05"] == r2["sig_sites_p05"]
        assert r1["positive_selection_detected"] == r2["positive_selection_detected"]

    def test_rate_distributions_deterministic(self, busted_pair):
        """Rate distribution parameters should be identical across runs."""
        result1, result2 = busted_pair

        rd1 = result1[0]["rate_distributions"]
        rd2 = result2[0]["rate_distributions"]
        for key in rd1:
            assert rd1[key] == pytest.approx(rd2[key], rel=1e-6), \
                f"Rate distribution '{key}' differs between runs"
