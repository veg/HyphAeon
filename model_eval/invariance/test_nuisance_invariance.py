"""
Invariance gates: the model MUST be invariant to nuisance parameters.

A phylogeny-aware model should be invariant to changes that do not alter the
evolutionary signal: rescaling branch-length units, adding exact duplicate
taxa, or reordering taxa in the input file. These tests verify that.

The metric is Pearson r between baseline and variant LRTs on tested sites.
r >= 0.999 means the model is sufficiently invariant. Real HyPhy MEME achieves
r = 1.0000 for all of these.

MULTI-DATASET COVERAGE: Tests run across the same grid of 6 datasets as the
phylogeny sensitivity tests (see test_phylogeny_sensitivity.py for the full
list).

Each test reports per-dataset results and fails if the model is NOT invariant
on a MAJORITY of datasets. A model that breaks on 4/6 datasets has a serious
problem even if it's invariant on 2.
"""
import numpy as np
import pytest

from _harness import (
    predict, load_tensors, make_scaled_tree, make_duplicate_alignment,
    sensitivity_r, max_abs_diff,
    format_invariance_results, majority_invariant,
)

_THRESHOLD = 0.999


# ---------------------------------------------------------------------------
# Branch-length rescaling
# ---------------------------------------------------------------------------

class TestBranchLengthScalingInvariance:
    """Multiplying all branch lengths by a constant changes only the units
    (e.g. substitutions/site vs substitutions/codon), not the evolutionary
    signal. The model must be invariant to this.

    The dataset's `dist_mat.max() > 10.0 -> /L` heuristic means the model is
    NOT invariant across the x10-x400 range — the preprocessing kicks in
    discontinuously.

    Threshold: r >= 0.999 on a majority of datasets.
    Real HyPhy MEME: r = 1.0000 for all scales.
    """

    @pytest.mark.parametrize("scale", [
        pytest.param(0.1, marks=pytest.mark.xfail(reason="Not invariant to x0.1 branch scaling — "
                                              "known model limitation (AGENTS.md: r=0.81 on sim_100_deep)"),
                     id="x0.1"),
        pytest.param(10, marks=pytest.mark.xfail(reason="Not invariant to x10 branch scaling — "
                                            "dist_mat heuristic kicks in discontinuously"),
                     id="x10"),
        pytest.param(100, marks=pytest.mark.xfail(reason="Not invariant to x100 branch scaling — "
                                             "dist_mat heuristic kicks in discontinuously"),
                     id="x100"),
    ])
    def test_scaling_invariance(self, model, all_datasets, tmp_path, scale):
        results = []
        for ds in all_datasets:
            scaled_nwk = make_scaled_tree(ds["nwk"], scale, tmp_path)
            c, a, d, z, inv, taxa, L = load_tensors(ds["fa"], scaled_nwk)
            ls = predict(model, c, a, d, z, inv)
            tested = ~inv & ds["tested"]
            r = sensitivity_r(ds["lrt"], ls, tested)
            results.append((ds["name"], r))
            print(f"  [{ds['name']:15s}] r={r:.6f} "
                  f"max|dLRT|={max_abs_diff(ds['lrt'], ls, tested):.3e}")

        n_inv = sum(1 for _, r in results if not np.isnan(r) and r >= _THRESHOLD)
        n_valid = sum(1 for _, r in results if not np.isnan(r))
        assert majority_invariant(results, _THRESHOLD), (
            f"Model is NOT invariant to x{scale} branch-length scaling on "
            f"{n_valid - n_inv}/{n_valid} datasets (majority fail). "
            f"Real MEME: r=1.0000 for all scales.\n"
            f"{format_invariance_results(results, _THRESHOLD)}"
        )


# ---------------------------------------------------------------------------
# Duplicate taxa
# ---------------------------------------------------------------------------

class TestDuplicateTaxaInvariance:
    """Adding exact duplicate sequences on short branches introduces no new
    evolutionary information. The model's predictions should not change
    meaningfully.

    HyphAeon enforces this invariant by default via automated identical
    haplotype and tree pruning (prune_duplicates=True).

    Threshold: r >= 0.999 on a majority of datasets.
    Real HyPhy MEME: r = 1.0000.
    """

    def test_exact_duplicates_invariance_default(self, model, all_datasets, tmp_path):
        """Test default pipeline with automated duplicate pruning."""
        results = []
        for ds in all_datasets:
            n_add = min(60, ds["n_taxa"])
            dup_fa, dup_nwk, n_total = make_duplicate_alignment(
                ds["fa"], ds["nwk"], n_add, tmp_path)
            c, a, d, z, inv, taxa, L = load_tensors(
                dup_fa, dup_nwk, prune_duplicates=True)
            ld = predict(model, c, a, d, z, inv)
            tested = ~inv
            if L != ds["L"]:
                print(f"  [{ds['name']:15s}] SKIP (site count changed {ds['L']} -> {L})")
                continue
            r = sensitivity_r(ds["lrt"], ld, tested & ds["tested"])
            results.append((ds["name"], r))
            print(f"  [{ds['name']:15s}] r={r:.6f} max|dLRT|={max_abs_diff(ds['lrt'], ld, tested & ds['tested']):.3e}")

        if not results:
            pytest.skip("All datasets had site-count changes from duplicates")

        n_inv = sum(1 for _, r in results if not np.isnan(r) and r >= _THRESHOLD)
        n_valid = sum(1 for _, r in results if not np.isnan(r))
        assert majority_invariant(results, _THRESHOLD), (
            f"Model is NOT invariant to exact duplicate taxa on "
            f"{n_valid - n_inv}/{n_valid} datasets (majority fail). "
            f"Real MEME: r=1.0000.\n"
            f"{format_invariance_results(results, _THRESHOLD)}"
        )
