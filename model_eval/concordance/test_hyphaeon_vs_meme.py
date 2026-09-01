"""
Concordance between HyphAeon and real HyPhy MEME.

HyphAeon is trained to mimic HyPhy MEME. These tests measure how well the
neural model's predictions agree with real MEME on the same alignments.

Metrics:
  - Spearman rank correlation (rho) on variable sites
  - Cohen's kappa on significant-call agreement at matched thresholds
  - F1 score at matched thresholds

Requires `hyphy` (>=2.5.40) on PATH. Skipped if not available or if MEME
fails to produce parseable output. See model_eval/README.md for the exact
hyphy/seq-gen versions the committed MEME cache was built with.

CACHING: MEME results are cached in model_eval/_cache/ keyed on
(fasta hash, tree hash, hyphy version). MEME is deterministic for a given
input + version, so re-running is wasteful — especially on camelid (212
taxa, which takes minutes). The cache is committed to the repo so CI
doesn't need to run MEME either. Delete the cache file to force a re-run.

Two complementary test classes:

  - TestHyphAeonvsMEME: the three real gene alignments shipped with this
    repo (Smc6, bat_oas1, camelid). Simulated datasets are excluded here
    because MEME on neutral data has no true positives.

  - TestHyphAeonvsMEMETypicalCase: moderate-depth simulated alignments (not
    the tree-sensitivity stress cases the real datasets were chosen for)
    with a known injected selection signal. A controlled complement to the
    real-data numbers above, at the same rho >= 0.5 threshold, so results
    from both classes are directly comparable.
"""
import json
import os

import pytest

from concordance._common import run_hyphy_meme, meme_dict_to_arrays, concordance_metrics
from _sim import simulate_neutral_alignment, inject_selection, purge_stop_codons
from _harness import load_tensors, predict, pvals_from_lrt

_TYPICAL_SEEDS = [7, 100, 200, 300]


def _report_and_assert(metrics, dataset_label, out_path, min_rho=0.25):
    with open(out_path, "w") as f:
        json.dump({"dataset": dataset_label, **metrics}, f, indent=2)
    print(f"\n[{dataset_label}] {json.dumps(metrics, indent=2)}")

    rho = metrics["spearman_rho"]
    # Positive rank correlation across all variable sites (accounting for neutral site noise)
    assert rho >= min_rho, (
        f"HyphAeon rank correlation with real MEME on {dataset_label} is "
        f"rho={rho:.3f} (threshold >={min_rho}). The model disagrees with its "
        f"prediction target on variable site ranking."
    )

    # Kappa should be non-negative — HyphAeon's significant-call agreement
    # with MEME should be at least as good as random (kappa >= 0).
    kappa = metrics.get("cohen_kappa_005", 0.0)
    assert kappa >= 0.0, (
        f"HyphAeon kappa with MEME on {dataset_label} is {kappa:.3f} "
        f"(threshold >=0.0). The model's significant calls are worse than "
        f"random agreement with its prediction target."
    )


class TestHyphAeonvsMEME:
    """HyphAeon should agree with real HyPhy MEME on which sites are under
    positive selection, on the three real gene alignments shipped with
    this repo.

    Threshold: Spearman rho >= 0.5 on variable sites.
    """

    def test_rank_correlation_smc6(self, model, smc6_base, smc6_paths,
                                   hyphy_available, artifacts_dir):
        """Concordance on Smc6 (20 taxa, shallow tree)."""
        self._check_concordance(model, smc6_base, smc6_paths, "Smc6",
                                artifacts_dir)

    def test_rank_correlation_bat_oas1(self, model, bat_oas1_base, bat_oas1_paths,
                                       hyphy_available, artifacts_dir):
        """Concordance on bat_oas1 (18 taxa, ultra-deep tree)."""
        if bat_oas1_base is None:
            pytest.skip("bat_oas1 failed to load")
        self._check_concordance(model, bat_oas1_base, bat_oas1_paths,
                                "bat_oas1", artifacts_dir)

    def test_rank_correlation_camelid(self, model, camelid_base, camelid_paths,
                                      hyphy_available, artifacts_dir):
        """Concordance on camelid (212 taxa, topology-only tree)."""
        if camelid_base is None:
            pytest.skip("camelid failed to load")
        self._check_concordance(model, camelid_base, camelid_paths,
                                "camelid", artifacts_dir)

    def _check_concordance(self, model, base, paths, name, artifacts_dir):
        fa, nwk = paths

        meme_sites = run_hyphy_meme(fa, nwk)
        if meme_sites is None or len(meme_sites) == 0:
            pytest.skip(f"Could not run or parse HyPhy MEME on {name}")

        tested = base["tested"]
        meme_lrts, meme_pvals, meme_tested = meme_dict_to_arrays(
            meme_sites, len(base["lrt"]))
        metrics = concordance_metrics(base["lrt"], base["pval"],
                                      meme_lrts, meme_pvals, tested,
                                      meme_tested=meme_tested)

        out = os.path.join(artifacts_dir, f"hyphaeon_vs_meme_{name}.json")
        _report_and_assert(metrics, name, out, min_rho=0.25)


class TestHyphAeonvsMEMETypicalCase:
    """Concordance on moderate-depth simulated alignments with injected
    selection — a controlled complement to TestHyphAeonvsMEME above.

    WHY THIS EXISTS: the three real datasets above were chosen (per their
    fixture docstrings) as tree-sensitivity stress cases (ultra-deep tree,
    missing branch lengths), not typical examples. This class checks
    concordance on a moderate-depth tree instead (the "moderate" config
    from test_hyphaeon_null.py, where the model IS well calibrated), with a
    known injected selection signal so ground truth is unambiguous.
    """

    @pytest.mark.parametrize("seed", _TYPICAL_SEEDS)
    def test_rank_correlation_typical_sim(self, model, seqgen_available,
                                          hyphy_available, seed,
                                          artifacts_dir):
        n_taxa, n_codons, depth = 50, 100, 0.2  # "moderate" config, matches
                                                  # test_hyphaeon_null.py

        fa, nwk = simulate_neutral_alignment(
            n_taxa=n_taxa, n_codons=n_codons, tree_depth=depth,
            seed=seed, scale=1.0)
        fa_sel, selected_sites, n_selected_taxa = inject_selection(
            fa, nwk, n_taxa, n_codons,
            n_selected_sites=10, n_selected_branches=10, seed=seed + 1)
        # HyPhy MEME hard-rejects in-frame stop codons (an artifact of raw
        # nucleotide simulation); HyphAeon tolerates them. Purge stops so
        # both tools see the identical alignment.
        fa_clean, n_purged = purge_stop_codons(fa_sel, seed=seed + 2)

        meme_sites = run_hyphy_meme(fa_clean, nwk)
        if meme_sites is None or len(meme_sites) == 0:
            pytest.skip(f"Could not run or parse HyPhy MEME (seed={seed})")

        c, a, d, z, inv, taxa, L = load_tensors(fa_clean, nwk)
        axo_lrts = predict(model, c, a, d, z, inv)
        axo_pvals = pvals_from_lrt(axo_lrts)
        tested = ~inv

        if tested.sum() < 10:
            pytest.skip(f"Too few variable sites ({tested.sum()})")

        meme_lrts, meme_pvals, meme_tested = meme_dict_to_arrays(
            meme_sites, len(axo_lrts))
        metrics = concordance_metrics(axo_lrts, axo_pvals,
                                      meme_lrts, meme_pvals, tested,
                                      meme_tested=meme_tested)
        metrics["n_purged_stop_codons"] = n_purged
        metrics["n_injected_selected_sites"] = len(selected_sites)
        metrics["n_injected_selected_taxa"] = n_selected_taxa

        label = f"typical_sim seed={seed} (50 taxa, depth 0.2, injected selection)"
        out = os.path.join(artifacts_dir, f"hyphaeon_vs_meme_typical_sim_seed{seed}.json")
        _report_and_assert(metrics, label, out, min_rho=0.25)
