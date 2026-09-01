"""
Null calibration for HyphAeon: are p-values honest under neutrality?

Feeds the model alignments simulated under neutral evolution (no positive
selection) using seq-gen with an HKY nucleotide model. Under the null, the
p-value distribution across sites should be approximately uniform on [0,1],
and the false positive rate at alpha=0.05 should be ~5%.

A model that calls 20% of neutral sites significant at alpha=0.05 is
anti-conservative — every downstream claim is inflated.

MULTI-CONFIG COVERAGE: Tests run across a grid of (n_taxa, tree_depth)
combinations to prevent "boundary case" dismissals:

  - 20 taxa, shallow (0.1)  — small N, low divergence
  - 50 taxa, moderate (0.2) — moderate N, moderate divergence
  - 100 taxa, deep (0.5)    — large N, high divergence

If the model is anti-conservative on all three, the result generalizes.

Requires seq-gen on PATH. Skipped if not available.
"""
import json
import os

import numpy as np
import pytest
from scipy.stats import chisquare

from _harness import evaluate_alignment, fpr_at, load_tensors, predict, pvals_from_lrt
from _sim import simulate_neutral_alignment

# Grid of (n_taxa, tree_depth, label) for calibration tests
_CALIB_GRID = [
    (20, 0.1, "small_shallow"),
    (50, 0.2, "moderate"),
    (100, 0.5, "large_deep"),
]

# Seeds used by the FPR test (3) and the uniformity test (5).
# The uniformity test extends to seeds 3-4 for more power.
_FPR_SEEDS = [0, 1, 2]
_UNIFORMITY_SEEDS = [0, 1, 2, 3, 4]


@pytest.fixture(scope="module")
def _neutral_pvals(model, seqgen_available):
    """Session-scoped cache of neutral p-values keyed by (n_taxa, depth, seed).

    Avoids re-running seq-gen + model forward passes when both the FPR
    test and the uniformity test need the same simulations.
    """
    cache = {}
    for n_taxa, depth, label in _CALIB_GRID:
        for seed in _UNIFORMITY_SEEDS:
            fa, nwk = simulate_neutral_alignment(
                n_taxa=n_taxa, n_codons=100, tree_depth=depth,
                seed=seed, scale=1.0)
            res = evaluate_alignment(model, fa, nwk)
            tested = res["tested"]
            if tested.sum() > 0:
                cache[(n_taxa, depth, seed)] = res["pval"][tested]
    return cache


@pytest.mark.parametrize("n_taxa,depth,label", _CALIB_GRID)
class TestHyphAeonNullCalibration:
    """Under neutral evolution, HyphAeon's p-values should be calibrated.

    Pools p-values across multiple seeds per config before asserting, so the
    FPR estimate is over ~150+ sites rather than ~50. Asserting per-seed at
    ~50 sites has an ~11% false-alarm rate under a correctly calibrated model
    (P(FPR≥10% | 50 sites, true 5%) ≈ 11%), which makes the suite flake.

    Threshold: pooled FPR at alpha=0.05 should be <= 10% (lenient — allows
    some inflation but catches severe miscalibration). The ideal is 5%.
    """

    def test_neutral_fpr(self, _neutral_pvals, n_taxa, depth, label):
        all_pvals = []
        per_seed = []
        for seed in _FPR_SEEDS:
            key = (n_taxa, depth, seed)
            if key not in _neutral_pvals:
                continue
            p_tested = _neutral_pvals[key]
            fpr_05 = float(np.mean(p_tested <= 0.05))
            fpr_10 = float(np.mean(p_tested <= 0.10))
            per_seed.append((seed, len(p_tested), fpr_05, fpr_10,
                             float(p_tested.mean())))
            all_pvals.append(p_tested)

        if not all_pvals:
            pytest.skip(f"No variable sites for {label}")
        pooled = np.concatenate(all_pvals)
        fpr_05 = float(np.mean(pooled <= 0.05))
        fpr_10 = float(np.mean(pooled <= 0.10))

        print(f"\n[{label}] pooled variable sites: {len(pooled)}")
        for seed, n_var, s_fpr05, s_fpr10, s_mean in per_seed:
            print(f"  seed={seed}: {n_var} sites, FPR@0.05={s_fpr05:.1%}, "
                  f"FPR@0.10={s_fpr10:.1%}, mean_p={s_mean:.3f}")
        print(f"  pooled FPR@0.05: {fpr_05:.1%} (ideal: 5%)")
        print(f"  pooled FPR@0.10: {fpr_10:.1%} (ideal: 10%)")
        print(f"  pooled p-value mean: {pooled.mean():.3f} (ideal: ~0.5)")

        assert fpr_05 <= 0.10, (
            f"HyphAeon FPR at alpha=0.05 is {fpr_05:.1%} on neutral data "
            f"({label}, pooled over {len(_FPR_SEEDS)} seeds, "
            f"{len(pooled)} sites). Threshold: <=10%. Ideal: 5%. "
            f"p-values are anti-conservative."
        )


class TestHyphAeonPValueDistribution:
    """Under neutrality, p-values should be approximately uniform on [0,1].

    Pools p-values across the full calibration grid (3 configs x 5 seeds = 15
    simulations). A chi-squared test on the p-value histogram checks this.
    This is more sensitive to subtle miscalibration than the FPR test alone.

    Shares simulations with TestHyphAeonNullCalibration via the
    _neutral_pvals fixture — no redundant seq-gen runs.
    """

    def test_pvalue_uniformity(self, _neutral_pvals, artifacts_dir):
        all_pvals = list(_neutral_pvals.values())

        if len(all_pvals) == 0:
            pytest.skip("No variable sites across all neutral simulations")

        pooled = np.concatenate(all_pvals)
        if len(pooled) < 50:
            pytest.skip(f"Too few variable sites ({len(pooled)}) for uniformity test")

        n_bins = 10
        observed, _ = np.histogram(pooled, bins=n_bins, range=(0, 1))
        expected = np.full(n_bins, len(pooled) / n_bins)
        chi2, p_chi = chisquare(observed, expected)

        report = {
            "n_pooled_pvalues": len(pooled),
            "n_simulations": len(all_pvals),
            "configs": [label for _, _, label in _CALIB_GRID],
            "chi_squared": float(chi2),
            "chi_squared_p": float(p_chi),
            "fpr_005": float(np.mean(pooled <= 0.05)),
            "fpr_010": float(np.mean(pooled <= 0.10)),
            "mean_p": float(np.mean(pooled)),
            "histogram": observed.tolist(),
        }
        out = os.path.join(artifacts_dir, "hyphaeon_null_calibration.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[report] {out}")
        print(json.dumps(report, indent=2))

        # Lenient chi-squared gate: catch catastrophic non-uniformity
        # (p < 0.001 means the distribution is very far from uniform).
        # The FPR test above is the primary calibration gate; this is a
        # secondary check for gross distributional distortion that FPR
        # might miss (e.g., a spike at p=0.5 with uniform tails).
        assert p_chi > 0.001, (
            f"P-value distribution is far from uniform (chi-squared p="
            f"{p_chi:.2e}). The histogram is {observed.tolist()}. "
            f"This indicates severe distributional distortion beyond "
            f"what the FPR test catches."
        )
