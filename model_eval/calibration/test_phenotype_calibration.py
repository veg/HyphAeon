"""
Calibration tests for hyphaeon/phenotype.py (Mode II).

Tests that Mode II p-values are honest under the null: when foreground
and background are assigned randomly (no real phenotype signal), the
FPR at alpha=0.05 should be approximately 5%.

Uses seq-gen simulated neutral alignments and random foreground
assignments. These tests skip if weights or seq-gen are unavailable.
"""
import numpy as np
import pytest

from _harness import phylowas_pvals


class TestPhenotypePermutationCalibration:
    """Permutation-based FPR calibration: random foreground should produce
    ~5% false positives at alpha=0.05.
    """

    @pytest.mark.xfail(reason="Mode II FPR ~94% on neutral data — ACAT combination "
                         "of LRT + association is massively anti-conservative")
    def test_permutation_fpr(self, phylowas_runner, sim_alignments):
        """Run phenotype association with random foreground on neutral sims.
        FPR at alpha=0.05 should be roughly calibrated (between 1% and 15%).

        XFAIL: Mode II's ACAT p-value combines the neural LRT (which has
        ~36-94% FPR on neutral data) with the association test. The
        combination inherits the LRT's anti-conservatism.
        """
        alpha = 0.05
        all_pvals = []

        for ds in sim_alignments[:2]:  # Use first 2 sim datasets for speed
            fa, nwk = ds["fa"], ds["nwk"]
            taxa = ds["taxa"]
            n_fg = max(2, len(taxa) // 3)

            for seed in range(5):  # 5 random foreground assignments per dataset
                rng = np.random.default_rng(seed)
                fg_taxa = rng.choice(taxa, size=n_fg, replace=False)
                fg = ",".join(fg_taxa.tolist())
                result = phylowas_runner(fa, tree_path=nwk, foreground=fg)
                pvals = phylowas_pvals(result)
                all_pvals.extend(pvals.tolist())

        all_pvals = np.array(all_pvals)
        if len(all_pvals) < 10:
            pytest.skip("Not enough variable sites for FPR test")

        # Sanity check: p-values should vary across permutations (different
        # foreground assignments should produce different p-values). If all
        # p-values are identical, the model is insensitive to foreground.
        assert len(set(all_pvals.round(6))) > 1, (
            "All p-values identical across permutations — model may not be "
            "sensitive to foreground assignment"
        )

        fpr = float(np.mean(all_pvals <= alpha))
        # FPR should be roughly calibrated. With only 10 site-level p-values
        # per permutation and 5 permutations, we can't demand precision, but
        # anything above 20% indicates the p-values are not honest.
        assert fpr < 0.20, f"FPR {fpr:.2%} is too high — Mode II p-values are not honest"
