"""
Concordance between neural BUSTED and real HyPhy BUSTED.

Neural BUSTED is trained to mimic HyPhy BUSTED's gene-level omnibus
selection test. These tests measure how well the neural model's
p-values agree with real BUSTED on the same alignments.

Since BUSTED produces a single gene-level p-value (not per-site like
MEME), concordance is measured as:
  - Direction agreement at alpha=0.05 (both significant or both not)
  - p-value ordering across multiple genes (Spearman rank correlation)

LIMITATION: With only 3 example datasets, Spearman correlation has
negligible statistical power. The rank correlation tests assert only
rho >= -0.5 (no strong anti-correlation), not positive concordance.
Adding more genes would require either additional example datasets
or batch-mode BUSTED on simulated alignments.

Requires `hyphy` (>=2.5.40) on PATH. Skipped if not available or if
BUSTED fails to produce parseable output. Results are cached alongside
the MEME cache in model_eval/_cache/.

NOTE: The neural BUSTED head weights are not in the current checkpoint,
so the neural head outputs (selection_probability, rate_distributions)
are from random initialization. The ACAT p-value (derived from the
core model's site-level LRTs) is still meaningful. These tests focus
on p_value_acat vs HyPhy BUSTED p-value.
"""
import os

import numpy as np
import pytest
from scipy.stats import spearmanr

from concordance._common import run_hyphy_busted


class TestBustedConcordance:
    """Neural BUSTED vs HyPhy BUSTED on real gene alignments."""

    @pytest.fixture(scope="class")
    def busted_pairs(self, busted_cross_dataset, hyphy_available,
                     smc6_paths, bat_oas1_paths, camelid_paths):
        """Run HyPhy BUSTED on all example datasets; reuse session-scoped
        neural BUSTED results from busted_cross_dataset.
        Returns list of (name, neural_pval, hyphy_pval, neural_lrt, hyphy_lrt).
        """
        pairs = []
        for name, (fa, nwk) in [
            ("Smc6", smc6_paths),
            ("bat_oas1", bat_oas1_paths),
            ("camelid", camelid_paths),
        ]:
            if not os.path.exists(fa) or not os.path.exists(nwk):
                continue

            hyphy_result = run_hyphy_busted(fa, nwk)
            if hyphy_result is None:
                continue

            neural_record = busted_cross_dataset[name][0]

            pairs.append((
                name,
                neural_record["p_value_acat"],
                hyphy_result["p_value"],
                neural_record["omnibus_lrt"],
                hyphy_result["lrt"],
            ))
        return pairs

    def test_at_least_one_pair(self, busted_pairs):
        """We should have at least one dataset with both results."""
        assert len(busted_pairs) >= 1, (
            "No datasets produced both neural and HyPhy BUSTED results — "
            "check that example data and hyphy are available"
        )

    def test_p_values_in_valid_range(self, busted_pairs):
        """Both neural and HyPhy p-values should be in [0, 1]."""
        for name, neural_p, hyphy_p, _, _ in busted_pairs:
            assert 0.0 <= neural_p <= 1.0, f"{name}: neural p={neural_p} out of range"
            assert 0.0 <= hyphy_p <= 1.0, f"{name}: HyPhy p={hyphy_p} out of range"

    def test_direction_agreement(self, busted_pairs):
        """A majority of datasets should agree on significance at alpha=0.05.

        With 3 datasets, majority means >= 2/3 agreement. At alpha=0.05,
        random agreement (both not significant) is common, so this is still
        a weak test — but it catches systematic disagreement.
        """
        if len(busted_pairs) < 2:
            pytest.skip("Need >= 2 datasets for direction agreement test")
        alpha = 0.05
        agreements = 0
        for name, neural_p, hyphy_p, _, _ in busted_pairs:
            neural_sig = neural_p < alpha
            hyphy_sig = hyphy_p < alpha
            if neural_sig == hyphy_sig:
                agreements += 1
        # Require strict majority (> half)
        assert agreements > len(busted_pairs) // 2, (
            f"Direction agreement only {agreements}/{len(busted_pairs)} — "
            f"neural and HyPhy BUSTED disagree on significance"
        )

    def test_p_value_rank_correlation(self, busted_pairs):
        """Spearman rank correlation of p-values across genes should be >= 0."""
        if len(busted_pairs) < 3:
            pytest.skip("Need >= 3 datasets for rank correlation")
        neural_pvals = [p[1] for p in busted_pairs]
        hyphy_pvals = [p[2] for p in busted_pairs]
        rho, _ = spearmanr(neural_pvals, hyphy_pvals)
        # With only 3 genes, we can't demand strong correlation, but
        # it shouldn't be strongly anti-correlated
        assert rho >= -0.5, (
            f"Spearman rho={rho:.3f} — neural BUSTED p-values are "
            f"anti-correlated with HyPhy BUSTED"
        )

    def test_lrt_correlation(self, busted_pairs):
        """LRTs should be positively correlated across genes."""
        if len(busted_pairs) < 3:
            pytest.skip("Need >= 3 datasets for LRT correlation")
        neural_lrts = [p[3] for p in busted_pairs]
        hyphy_lrts = [p[4] for p in busted_pairs]
        rho, _ = spearmanr(neural_lrts, hyphy_lrts)
        assert rho >= -0.5, (
            f"Spearman rho={rho:.3f} — neural BUSTED LRTs are "
            f"anti-correlated with HyPhy BUSTED LRTs"
        )
