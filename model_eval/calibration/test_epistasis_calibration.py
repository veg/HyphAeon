"""
Calibration test for epistasis (ESSM) p-values.

Tests that edge-level p-values are honest under the null: on neutral
simulated alignments, the FDR-controlled edge set should not be
excessively large.

Requires seq-gen for neutral simulation. Skips if unavailable.
"""
import numpy as np
import pytest

from _harness import essm_edge_pvals


class TestEpistasisNullCalibration:
    """ESSM edge p-values on neutral simulations should not be
    systematically significant.

    Both tests are xfailed: the t-test on cosine similarity of neural
    attribution vectors has ~87% FPR on neutral data (REVIEW.md item #2).
    The H2 fix (max_fdr=1.0) revealed the true FPR that was previously
    hidden by FDR filtering.
    """

    @pytest.mark.xfail(reason="ESSM edge FPR ~87% on neutral data — t-test on "
                         "cosine similarity of neural attribution vectors is "
                         "massively anti-conservative")
    def test_neutral_edge_fpr(self, essm_runner, sim_alignments):
        """On neutral alignments, the fraction of all candidate edges with
        p_val < 0.05 should not be extremely high (would indicate the t-test
        on cosine similarity is not calibrated under the null).

        Uses max_fdr=1.0 to disable FDR filtering — we need to see ALL
        candidate edges, not just the FDR-significant subset, to measure
        the raw p-value calibration.
        """
        if not sim_alignments:
            pytest.skip("No simulated alignments available")

        all_pvals = []
        for ds in sim_alignments[:2]:
            fa, nwk = ds["fa"], ds["nwk"]
            result = essm_runner(fa, tree_path=nwk, max_fdr=1.0,
                                 min_sim=0.0, min_lrt=0.0, min_shared=1)
            pvals = essm_edge_pvals(result)
            all_pvals.extend(pvals.tolist())

        all_pvals = np.array(all_pvals)
        if len(all_pvals) < 5:
            pytest.skip("Not enough edges for FPR test")

        fpr = float(np.mean(all_pvals < 0.05))
        # Edge p-values from t-test on cosine similarity should be roughly
        # calibrated. Allow up to 30% (the test is approximate due to
        # limited sample size per edge).
        assert fpr < 0.30, (
            f"ESSM edge FPR {fpr:.2%} on neutral data — "
            f"t-test p-values are not honest under the null"
        )

    @pytest.mark.xfail(reason="ESSM discovers 66% of possible edges on neutral "
                         "data with max_fdr=0.05 — BH FDR control is broken "
                         "when the underlying t-test is anti-conservative")
    def test_neutral_fdr_edges_not_excessive(self, essm_runner, sim_alignments):
        """With max_fdr=0.05, the number of significant edges on neutral
        data should not be a large fraction of all possible edges.

        With BH FDR at alpha=0.05, the expected false discovery rate is 5%.
        We assert < 5% of possible edges pass FDR, which is conservative
        since most candidate pairs are filtered by the t-test before FDR.
        """
        if not sim_alignments:
            pytest.skip("No simulated alignments available")

        for ds in sim_alignments[:2]:
            fa, nwk = ds["fa"], ds["nwk"]
            result = essm_runner(fa, tree_path=nwk, max_fdr=0.05,
                                 min_sim=0.0, min_lrt=0.0, min_shared=1)
            n_edges = len(result["edges"])
            n_sites = result["codon_count"]
            max_possible = n_sites * (n_sites - 1) / 2
            if max_possible == 0:
                continue
            edge_fraction = n_edges / max_possible
            assert edge_fraction < 0.05, (
                f"ESSM discovers {edge_fraction:.1%} of possible edges on neutral data "
                f"({n_edges}/{int(max_possible)}) — FDR control may be broken"
            )
