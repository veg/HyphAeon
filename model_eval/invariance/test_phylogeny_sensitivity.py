"""
Invariance gates: the model MUST be sensitive to phylogeny.

A phylogeny-aware model of episodic selection must change its predictions when
the relationship between sequences and tree topology changes. These tests
verify that permuting which taxon carries which codon, destroying topology
(star tree), or zeroing out all distances produces a measurable change in LRT
output on variable sites.

The metric is Pearson r between baseline and variant LRTs on tested (variable)
sites. r=1.0 means perfectly invariant (the model ignored the change). The
thresholds are deliberately lenient — they catch "the model completely
ignored phylogeny," not "the model is slightly under-responsive."

MULTI-DATASET COVERAGE: Tests run across a grid of 6 datasets (3 real + 3
simulated) spanning shallow to deep trees, 18 to 212 taxa, and 26% to 100%
variable sites. This prevents "boundary case" dismissals.

Each test reports per-dataset results and fails if the model is invariant on
a MAJORITY of datasets. A model that is tree-blind on 4/6 datasets has a
serious problem even if it's sensitive on 2.
"""
import numpy as np
import pytest

from _harness import (
    predict, permute_columns, make_star_tree, make_zero_distance_tree,
    sensitivity_r, max_abs_diff, load_tensors,
    format_invariance_results, majority_invariant,
)

_THRESHOLD = 0.99


# ---------------------------------------------------------------------------
# Within-column taxon permutation
# ---------------------------------------------------------------------------

class TestPermutationSensitivity:
    """Permuting which taxon carries which codon MUST change predictions.

    MEME's LRT depends on which branches carry substitutions. Permuting taxa
    across columns reassigns substitutions to different branches, so a
    phylogeny-aware model must produce different LRTs. A model that is
    invariant to this is, by construction, ignoring the tree.

    Threshold: r < 0.99 on a majority of datasets.
    Real HyPhy MEME on Smc6: r ~ 0.61 under the same permutation.
    """

    @pytest.mark.parametrize("seed", [
        pytest.param(0, marks=pytest.mark.xfail(reason="Model is largely phylogeny-blind — "
                                              "invariant to taxon permutation on majority of datasets (AGENTS.md)"),
                     id="seed0"),
        pytest.param(1, marks=pytest.mark.xfail(reason="Model is largely phylogeny-blind — "
                                              "invariant to taxon permutation on majority of datasets (AGENTS.md)"),
                     id="seed1"),
        pytest.param(2, marks=pytest.mark.xfail(reason="Model is largely phylogeny-blind — "
                                              "invariant to taxon permutation on majority of datasets (AGENTS.md)"),
                     id="seed2"),
    ])
    def test_permutation_changes_lrt(self, model, all_datasets, seed):
        results = []
        for ds in all_datasets:
            cp, ap = permute_columns(ds["c"], ds["a"], ds["tested"], seed)
            lp = predict(model, cp, ap, ds["d"], ds["z"], ds["inv"])
            r = sensitivity_r(ds["lrt"], lp, ds["tested"])
            results.append((ds["name"], r))
            print(f"  [{ds['name']:15s}] r={r:.6f} "
                  f"max|dLRT|={max_abs_diff(ds['lrt'], lp, ds['tested']):.3e}")

        n_inv = sum(1 for _, r in results if not np.isnan(r) and r >= _THRESHOLD)
        n_valid = sum(1 for _, r in results if not np.isnan(r))
        assert not majority_invariant(results, _THRESHOLD), (
            f"Model is invariant to within-column taxon permutation (seed={seed}) "
            f"on {n_inv}/{n_valid} datasets (majority). "
            f"Real MEME: r~0.61 on Smc6.\n{format_invariance_results(results, _THRESHOLD)}"
        )


# ---------------------------------------------------------------------------
# Star tree (topology destroyed, scale preserved)
# ---------------------------------------------------------------------------

class TestStarTreeSensitivity:
    """Replacing the tree with a star (same mean branch length) MUST change
    predictions, because topology — not just scale — carries the signal for
    which branches have substitutions.

    Threshold: r < 0.99 on a majority of datasets.
    """

    @pytest.mark.xfail(reason="Model is invariant to star-tree substitution on majority "
                         "of datasets — largely phylogeny-blind (AGENTS.md)")
    def test_star_tree_changes_lrt(self, model, all_datasets, tmp_path):
        results = []
        for ds in all_datasets:
            star_nwk = make_star_tree(ds["nwk"], ds["taxa"], tmp_path)
            c, a, d, z, inv, taxa, L = load_tensors(ds["fa"], star_nwk)
            ls = predict(model, c, a, d, z, inv)
            tested = ~inv & ds["tested"]
            r = sensitivity_r(ds["lrt"], ls, tested)
            results.append((ds["name"], r))
            print(f"  [{ds['name']:15s}] r={r:.6f} "
                  f"max|dLRT|={max_abs_diff(ds['lrt'], ls, tested):.3e}")

        n_inv = sum(1 for _, r in results if not np.isnan(r) and r >= _THRESHOLD)
        n_valid = sum(1 for _, r in results if not np.isnan(r))
        assert not majority_invariant(results, _THRESHOLD), (
            f"Model is invariant to star-tree substitution on "
            f"{n_inv}/{n_valid} datasets (majority). "
            f"Destroying topology should change predictions.\n"
            f"{format_invariance_results(results, _THRESHOLD)}"
        )


# ---------------------------------------------------------------------------
# Zero-distance tree
# ---------------------------------------------------------------------------

class TestZeroDistanceSensitivity:
    """Setting all pairwise distances to ~0 collapses all taxa to the same
    point. A phylogeny-aware model must respond to this extreme change.

    Threshold: r < 0.99 on a majority of datasets.
    """

    @pytest.mark.xfail(reason="Model is invariant to zero-distance tree on majority "
                         "of datasets — largely phylogeny-blind (AGENTS.md)")
    def test_zero_distance_changes_lrt(self, model, all_datasets, tmp_path):
        results = []
        for ds in all_datasets:
            zero_nwk = make_zero_distance_tree(ds["taxa"], tmp_path)
            c, a, d, z, inv, taxa, L = load_tensors(ds["fa"], zero_nwk)
            lz = predict(model, c, a, d, z, inv)
            tested = ~inv & ds["tested"]
            r = sensitivity_r(ds["lrt"], lz, tested)
            results.append((ds["name"], r))
            print(f"  [{ds['name']:15s}] r={r:.6f} "
                  f"max|dLRT|={max_abs_diff(ds['lrt'], lz, tested):.3e}")

        n_inv = sum(1 for _, r in results if not np.isnan(r) and r >= _THRESHOLD)
        n_valid = sum(1 for _, r in results if not np.isnan(r))
        assert not majority_invariant(results, _THRESHOLD), (
            f"Model is invariant to zero-distance tree on "
            f"{n_inv}/{n_valid} datasets (majority). "
            f"Collapsing all distances should change predictions.\n"
            f"{format_invariance_results(results, _THRESHOLD)}"
        )
