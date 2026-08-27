"""
Null calibration for AxoMEME: are p-values honest under neutrality?

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

# Grid of (n_taxa, tree_depth, label) for calibration tests.
# Moderate and deep configs are xfailed in the FPR test because the model's
# FPR is 16% (moderate) and 72.7% (deep) — well above the 10% threshold.
# See _artifacts/tree_depth_fpr_spread.json for the committed numbers.
_CALIB_GRID = [
    (20, 0.1, "small_shallow"),
    (50, 0.2, "moderate"),
    (100, 0.5, "large_deep"),
]

# Parametrize with xfail marks for configs where FPR is known to exceed 10%.
_CALIB_GRID_FPR = [
    pytest.param(20, 0.1, "small_shallow", id="small_shallow"),
    pytest.param(50, 0.2, "moderate",
                 marks=pytest.mark.xfail(reason="FPR 16% on moderate trees — model calibration is tree-structure-dependent"),
                 id="moderate"),
    pytest.param(100, 0.5, "large_deep",
                 marks=pytest.mark.xfail(reason="FPR 72.7% on deep trees — model calibration is tree-structure-dependent"),
                 id="large_deep"),
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


@pytest.mark.parametrize("n_taxa,depth,label", _CALIB_GRID_FPR)
class TestAxoMEMENullCalibration:
    """Under neutral evolution, AxoMEME's p-values should be calibrated.

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
            f"AxoMEME FPR at alpha=0.05 is {fpr_05:.1%} on neutral data "
            f"({label}, pooled over {len(_FPR_SEEDS)} seeds, "
            f"{len(pooled)} sites). Threshold: <=10%. Ideal: 5%. "
            f"p-values are anti-conservative."
        )


class TestAxoMEMEPValueDistribution:
    """Under neutrality, p-values should be approximately uniform on [0,1].

    Pools p-values across the full calibration grid (3 configs x 5 seeds = 15
    simulations). A chi-squared test on the p-value histogram checks this.
    This is more sensitive to subtle miscalibration than the FPR test alone.

    Shares simulations with TestAxoMEMENullCalibration via the
    _neutral_pvals fixture — no redundant seq-gen runs.
    """

    @pytest.mark.xfail(reason="Pooled p-values across all configs (including "
                         "moderate/deep with 16-72.7% FPR) are far from uniform. "
                         "The histogram is heavily left-skewed.")
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
        out = os.path.join(artifacts_dir, "axomeme_null_calibration.json")
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


class TestTreeDepthSpread:
    """Is FPR systematically different across tree-depth configs?

    The per-config tests above check each (n_taxa, tree_depth) independently.
    This test collects all three FPRs and checks the spread. If FPR is
    calibrated on shallow/moderate trees but inflated on deep trees (or
    vice versa), that supports the distributional mismatch hypothesis:
    the model's calibration is tied to the tree shapes it was trained on.

    The AGENTS.md notes 36% FPR on 100-taxon deep neutral data (should be
    5%). If shallow and moderate are calibrated but deep is not, tree
    structure is a major driver.

    Shares simulations with TestAxoMEMENullCalibration via _neutral_pvals.
    """

    @pytest.mark.xfail(reason="FPR varies 12.9x across tree-depth configs "
                         "(5.6% shallow → 16% moderate → 72.7% deep). "
                         "Tree structure is a major driver of miscalibration — "
                         "the model is calibrated on shallow trees but breaks "
                         "down on deep ones. Supports distributional mismatch "
                         "hypothesis (b), specifically tree-structure mismatch.")
    def test_fpr_spread_across_tree_depths(self, _neutral_pvals, artifacts_dir):
        fprs = {}
        for n_taxa, depth, label in _CALIB_GRID:
            all_pvals = []
            for seed in _FPR_SEEDS:
                key = (n_taxa, depth, seed)
                if key not in _neutral_pvals:
                    continue
                all_pvals.append(_neutral_pvals[key])
            if not all_pvals:
                continue
            pooled = np.concatenate(all_pvals)
            fprs[label] = float(np.mean(pooled <= 0.05))

        if len(fprs) < 2:
            pytest.skip("Not enough configs produced results")

        values = list(fprs.values())
        spread = max(values) - min(values)
        max_ratio = max(values) / max(min(values), 1e-6)

        report = {
            "per_config_fpr": fprs,
            "spread": spread,
            "max_to_min_ratio": max_ratio,
            "interpretation": (
                "If FPR is low on shallow/moderate but high on deep, tree "
                "structure is a driver of miscalibration (supports hypothesis "
                "b). If FPR is uniformly high, the LRT itself is miscalibrated "
                "(supports hypothesis a)."
            ),
        }
        out = os.path.join(artifacts_dir, "tree_depth_fpr_spread.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[report] {out}")
        print(json.dumps(report, indent=2))

        # If the ratio between worst and best config FPR exceeds 5x,
        # tree depth is a major driver of miscalibration.
        assert max_ratio < 5.0, (
            f"FPR varies {max_ratio:.1f}x across tree-depth configs "
            f"({fprs}). The model's calibration is tree-structure-dependent — "
            f"supports the distributional mismatch hypothesis (b)."
        )
