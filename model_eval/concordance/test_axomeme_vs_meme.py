"""
Concordance between AxoMEME and real HyPhy MEME.

AxoMEME is trained to mimic HyPhy MEME. These tests measure how well the
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

  - TestAxoMEMEvsMEME: the three real gene alignments shipped with this
    repo (Smc6, bat_oas1, camelid). Simulated datasets are excluded here
    because MEME on neutral data has no true positives.

  - TestAxoMEMEvsMEMETypicalCase: moderate-depth simulated alignments (not
    the tree-sensitivity stress cases the real datasets were chosen for)
    with a known injected selection signal. A controlled complement to the
    real-data numbers above, at the same rho >= 0.25 threshold, so results
    from both classes are directly comparable.
"""
import json
import os

import numpy as np
import pytest

from concordance._common import run_hyphy_meme, meme_dict_to_arrays, concordance_metrics
from _sim import simulate_neutral_alignment, inject_selection, purge_stop_codons
from _harness import load_tensors, predict, pvals_from_lrt

_TYPICAL_SEEDS = [7, 100, 200, 300]


class TestMEMENegativeFPR:
    """AxoMEME FPR on real-data sites where MEME says neutral.

    This test distinguishes two hypotheses for AxoMEME's high FPR on seq-gen
    neutral data:

    (a) The neural LRT is genuinely miscalibrated (anti-conservative).
    (b) The model detects distributional mismatch — seq-gen HKY data has
        different nucleotide composition, codon usage, and tree structure
        than the real data the model was trained on.

    If (a), AxoMEME should be anti-conservative on BOTH seq-gen neutral and
    MEME-negative real sites. If (b), AxoMEME should be calibrated (or much
    better) on MEME-negative real sites while still anti-conservative on
    seq-gen neutral.

    MEME-negative sites are real-data variable sites where HyPhy MEME's
    p-value > 0.05 — "empirical neutral" with real codon usage, real tree
    structure, real composition. These are not truly neutral (some may have
    weak selection MEME lacks power to detect), but they are much closer to
    the model's training distribution than seq-gen HKY simulations.

    Uses the same cached MEME results as TestAxoMEMEvsMEME above — no
    additional HyPhy runs needed.
    """

    def test_fpr_on_meme_negative_sites(self, model, smc6_base, smc6_paths,
                                         bat_oas1_base, bat_oas1_paths,
                                         camelid_base, camelid_paths,
                                         hyphy_available, artifacts_dir):
        """AxoMEME FPR at alpha=0.05 on sites where MEME p > 0.05.

        Pools across all available real datasets. Reports per-dataset FPR
        alongside the MEME-negative site count.

        Threshold: FPR <= 20% (lenient — MEME-negative sites are not truly
        neutral, so some inflation is expected from weak selection MEME
        missed). The seq-gen neutral FPR is ~36-94%; if real-data FPR is
        much lower, distributional mismatch is the primary driver.
        """
        datasets = [
            ("Smc6", smc6_base, smc6_paths),
            ("bat_oas1", bat_oas1_base, bat_oas1_paths),
            ("camelid", camelid_base, camelid_paths),
        ]

        all_neutral_pvals = []
        per_dataset = []

        for name, base, paths in datasets:
            if base is None:
                continue
            fa, nwk = paths
            meme_sites = run_hyphy_meme(fa, nwk)
            if meme_sites is None or len(meme_sites) == 0:
                continue

            tested = base["tested"]
            meme_lrts, meme_pvals, meme_tested = meme_dict_to_arrays(
                meme_sites, len(base["lrt"]))

            # Sites where MEME tested and found NOT significant (p > 0.05)
            meme_neutral = meme_tested & (meme_pvals > 0.05) & tested
            if meme_neutral.sum() == 0:
                continue

            axo_pvals_neutral = base["pval"][meme_neutral]
            fpr = float(np.mean(axo_pvals_neutral <= 0.05))
            per_dataset.append((name, int(meme_neutral.sum()), fpr))
            all_neutral_pvals.append(axo_pvals_neutral)

        if not all_neutral_pvals:
            pytest.skip("No MEME-negative sites available across datasets")

        pooled = np.concatenate(all_neutral_pvals)
        pooled_fpr = float(np.mean(pooled <= 0.05))

        report = {
            "per_dataset": [
                {"dataset": n, "n_meme_negative_sites": cnt, "fpr_005": fpr}
                for n, cnt, fpr in per_dataset
            ],
            "pooled_n_meme_negative_sites": len(pooled),
            "pooled_fpr_005": pooled_fpr,
            "interpretation": (
                "If pooled FPR here is much lower than seq-gen neutral FPR "
                "(~36-94%), distributional mismatch (hypothesis b) is the "
                "primary driver. If similar, the LRT itself is miscalibrated "
                "(hypothesis a)."
            ),
        }
        out = os.path.join(artifacts_dir, "meme_negative_fpr.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[report] {out}")
        print(json.dumps(report, indent=2))

        assert pooled_fpr <= 0.20, (
            f"AxoMEME FPR on MEME-negative real-data sites is {pooled_fpr:.1%} "
            f"(pooled over {len(pooled)} sites). Threshold: <=20%. If this is "
            f"much lower than the seq-gen neutral FPR (~36-94%), the high "
            f"seq-gen FPR is driven by distributional mismatch, not LRT "
            f"miscalibration."
        )


def _report_and_assert(metrics, dataset_label, out_path, min_rho=0.25):
    with open(out_path, "w") as f:
        json.dump({"dataset": dataset_label, **metrics}, f, indent=2)
    print(f"\n[{dataset_label}] {json.dumps(metrics, indent=2)}")

    rho = metrics["spearman_rho"]
    # Positive rank correlation across all variable sites (accounting for neutral site noise)
    assert rho >= min_rho, (
        f"AxoMEME rank correlation with real MEME on {dataset_label} is "
        f"rho={rho:.3f} (threshold >={min_rho}). The model disagrees with its "
        f"prediction target on variable site ranking."
    )

    # Kappa should be non-negative — AxoMEME's significant-call agreement
    # with MEME should be at least as good as random (kappa >= 0).
    kappa = metrics.get("cohen_kappa_005", 0.0)
    assert kappa >= 0.0, (
        f"AxoMEME kappa with MEME on {dataset_label} is {kappa:.3f} "
        f"(threshold >=0.0). The model's significant calls are worse than "
        f"random agreement with its prediction target."
    )


class TestAxoMEMEvsMEME:
    """AxoMEME should agree with real HyPhy MEME on which sites are under
    positive selection, on the three real gene alignments shipped with
    this repo.

    Threshold: Spearman rho >= 0.25 on variable sites. 
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

        out = os.path.join(artifacts_dir, f"axomeme_vs_meme_{name}.json")
        _report_and_assert(metrics, name, out, min_rho=0.25)


class TestAxoMEMEvsMEMETypicalCase:
    """Concordance on moderate-depth simulated alignments with injected
    selection — a controlled complement to TestAxoMEMEvsMEME above.

    WHY THIS EXISTS: the three real datasets above were chosen (per their
    fixture docstrings) as tree-sensitivity stress cases (ultra-deep tree,
    missing branch lengths), not typical examples. This class checks
    concordance on a moderate-depth tree instead (the "moderate" config
    from test_axomeme_null.py), with a known injected selection signal so
    ground truth is unambiguous.

    NOTE: The moderate config (50 taxa, depth 0.2) has ~16% FPR on neutral
    data — not fully calibrated, but much better than the deep config (72.7%).
    Concordance with MEME on injected-selection data is still meaningful
    because the signal is strong enough to overcome moderate FPR inflation.
    """

    @pytest.mark.parametrize("seed", _TYPICAL_SEEDS)
    def test_rank_correlation_typical_sim(self, model, seqgen_available,
                                          hyphy_available, seed,
                                          artifacts_dir):
        n_taxa, n_codons, depth = 50, 100, 0.2  # "moderate" config, matches
                                                  # test_axomeme_null.py

        fa, nwk = simulate_neutral_alignment(
            n_taxa=n_taxa, n_codons=n_codons, tree_depth=depth,
            seed=seed, scale=1.0)
        fa_sel, selected_sites, n_selected_taxa, _ = inject_selection(
            fa, nwk, n_taxa, n_codons,
            n_selected_sites=10, n_selected_branches=10, seed=seed + 1)
        # HyPhy MEME hard-rejects in-frame stop codons (an artifact of raw
        # nucleotide simulation); AxoMEME tolerates them. Purge stops so
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
        out = os.path.join(artifacts_dir, f"axomeme_vs_meme_typical_sim_seed{seed}.json")
        _report_and_assert(metrics, label, out, min_rho=0.25)
