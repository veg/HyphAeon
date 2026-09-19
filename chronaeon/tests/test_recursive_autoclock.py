"""
test_recursive_autoclock.py
---------------------------
Unit and integration tests for recursive spectral autoclock deconvolution:
1. Matrix-based recursive spectral bisection along Cheeger cuts.
2. Analytical OLS clock fitting with Fieller ratio inversion confidence intervals.
3. Cohesive leaf stopping criterion via Welch rate z-score separability (z < 1.96).
4. Epidemiological community classification (Active Outbreak, Emergent Cluster,
   Chronic Reservoir, Wedged Circulating Lineage, Micro-Chain).
5. Reference panel scaffolding and hybrid wedging deconvolution.
"""

import numpy as np
import pytest

from chronaeon.autoclock import (
    fit_clock,
    recursive_spectral_autoclock,
    classify_community,
    classify_leaf_community,
)


def _generate_synthetic_epidemic_transmission_matrix():
    """
    Constructs a synthetic multi-community transmission system with known ground truth:
    - Community 0 (Active Outbreak): 10 taxa, dates 2020.0 to 2024.0, tight distances,
      high rate mu = 0.005, high R2 > 0.8, recent t_MRCA = 2019.0.
    - Community 1 (Chronic Reservoir): 10 taxa, dates 2012.0 to 2018.0, diffuse distances,
      flat rate, low R2 < 0.15.
    - Community 2 (Wedged Lineage with Reference): 8 taxa (4 local + 4 reference anchors),
      dates 1995.0 to 2015.0, deep divergence.
    Total N = 28 taxa.
    """
    np.random.seed(42)
    n0, n1, n2 = 10, 10, 8
    n_total = n0 + n1 + n2

    # 1. Dates
    dates_0 = np.linspace(2020.0, 2024.0, n0)
    dates_1 = np.linspace(2012.0, 2018.0, n1)
    dates_2 = np.linspace(1995.0, 2015.0, n2)
    dates = np.concatenate([dates_0, dates_1, dates_2])

    # 2. Distance Matrix
    D = np.zeros((n_total, n_total))

    # Intra-community 0: tight linear evolution from root at 2019.0 with rate 0.005
    t0_root = 2019.0
    mu_0 = 0.005
    dists_root_0 = mu_0 * (dates_0 - t0_root) + np.random.normal(0, 0.0003, n0)
    dists_root_0 = np.maximum(dists_root_0, 0.0001)
    for i in range(n0):
        for j in range(i + 1, n0):
            d = abs(dists_root_0[i] - dists_root_0[j]) + 0.0008 * np.random.uniform(0.5, 1.5)
            D[i, j] = D[j, i] = d

    # Intra-community 1: chronic reservoir, flat distances (mean ~0.025, uncorrelated with time)
    for i in range(n1):
        for j in range(i + 1, n1):
            d = np.random.uniform(0.022, 0.028)
            idx_i, idx_j = n0 + i, n0 + j
            D[idx_i, idx_j] = D[idx_j, idx_i] = d

    # Intra-community 2: reference wedged lineage (mean ~0.040)
    for i in range(n2):
        for j in range(i + 1, n2):
            d = np.random.uniform(0.035, 0.045)
            idx_i, idx_j = n0 + n1 + i, n0 + n1 + j
            D[idx_i, idx_j] = D[idx_j, idx_i] = d

    # Inter-community distances: well-separated Cheeger bottlenecks
    # Between 0 and 1: distance ~0.08
    for i in range(n0):
        for j in range(n1):
            D[i, n0 + j] = D[n0 + j, i] = np.random.uniform(0.075, 0.085)

    # Between 0/1 and 2: distance ~0.12
    for i in range(n0 + n1):
        for j in range(n2):
            D[i, n0 + n1 + j] = D[n0 + n1 + j, i] = np.random.uniform(0.115, 0.125)

    np.fill_diagonal(D, 0.0)

    # Reference flag: last 4 taxa of community 2 are reference anchors
    is_ref = np.zeros(n_total, dtype=bool)
    is_ref[n0 + n1 + 4:] = True

    return dates, D, is_ref, (n0, n1, n2)


class TestFitClock:
    """Unit tests for analytical OLS clock fitting with Fieller confidence intervals."""

    def test_fit_clock_linear_signal(self):
        """Validates rate recovery and bounded Fieller confidence interval."""
        dates = np.linspace(2015.0, 2025.0, 15)
        true_mu = 0.003
        true_t0 = 2012.0
        dists = true_mu * (dates - true_t0)

        # Build pairwise distance matrix matching this star tree
        D = np.zeros((15, 15))
        for i in range(15):
            for j in range(15):
                if i != j:
                    D[i, j] = dists[i] + dists[j] - 2 * min(dists[i], dists[j])

        fit = fit_clock(np.arange(15), dates, D)
        assert fit["n"] == 15
        assert fit["mu"] > 0
        assert fit["r2"] > 0.80
        assert fit["ci_mrca"][0] <= fit["tmrca"] <= fit["ci_mrca"][1]
        assert fit["fieller_status"] in ["BOUNDED", "OK", "VALID"]

    def test_fit_clock_small_sample_floor(self):
        """Verifies that sample size < 3 returns safe fallbacks without throwing."""
        dates = np.array([2020.0, 2021.0])
        D = np.array([[0.0, 0.01], [0.01, 0.0]])
        fit = fit_clock(np.arange(2), dates, D)
        assert fit["n"] == 2
        assert fit["fieller_status"] == "TOO_SMALL"
        assert fit["mu"] == 0.0

    def test_fit_clock_zero_temporal_variance(self):
        """Verifies that contemporaneous samples (identical timestamps) are safely handled."""
        dates = np.array([2020.0, 2020.0, 2020.0, 2020.0])
        D = np.ones((4, 4)) * 0.005
        np.fill_diagonal(D, 0.0)
        fit = fit_clock(np.arange(4), dates, D)
        assert fit["n"] == 4
        assert fit["fieller_status"] == "ZERO_TIME_VAR"
        assert fit["mu"] == 0.0


class TestRecursiveSpectralAutoClock:
    """Tests recursive spectral bisection along Cheeger cuts and stopping rules."""

    def test_recursive_deconvolution_recovers_communities(self):
        """
        Validates that recursive spectral bisection separates the three distinct
        synthetic transmission communities without over-fragmenting.
        """
        dates, D, is_ref, (n0, n1, n2) = _generate_synthetic_epidemic_transmission_matrix()
        n_total = len(dates)

        leaves = recursive_spectral_autoclock(
            sub_idx=np.arange(n_total),
            dates=dates,
            D=D,
            min_size=3,
            max_depth=5,
        )

        assert len(leaves) >= 3

        # Verify that community 0 taxa (0..9) are segregated from community 1 taxa (10..19)
        c0_found = False
        c1_found = False
        for sub_idx, fit, stop_reason, path in leaves:
            s = set(sub_idx)
            if s.issubset(set(range(0, n0))):
                c0_found = True
                assert fit["mu"] > 0.002
                assert fit["r2"] > 0.60
            if s.issubset(set(range(n0, n0 + n1))):
                c1_found = True
                assert fit["r2"] < 0.40

        assert c0_found, "Active outbreak cluster (0..9) was not isolated into a cohesive leaf."
        assert c1_found, "Chronic reservoir cluster (10..19) was not isolated into a cohesive leaf."

    def test_cohesive_clock_cluster_stopping_rule(self):
        """
        Tests that when a subcommunity is tight (mean dist < 1.5%), linear (R2 >= 0.20),
        and rate separability Welch z < 1.96, recursion halts immediately.
        """
        np.random.seed(123)
        n = 8
        dates = np.linspace(2021.0, 2024.0, n)
        mu = 0.004
        t0 = 2020.0
        dists = mu * (dates - t0)

        # Monolithic clock with minimal noise
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = abs(dists[i] - dists[j]) + 0.0005

        leaves = recursive_spectral_autoclock(
            sub_idx=np.arange(n),
            dates=dates,
            D=D,
            min_size=2,
            max_depth=4,
        )

        # Must not split: cohesive cluster stopping rule should trigger
        assert len(leaves) == 1
        sub_idx, fit, stop_reason, path = leaves[0]
        assert "cohesive_clock_cluster" in stop_reason
        assert len(sub_idx) == n


class TestCommunityClassification:
    """Tests the five-tier epidemiological transmission classification."""

    def test_active_outbreak_classification(self):
        fit = {
            "n": 6,
            "mu": 0.012,
            "r2": 0.85,
            "tmrca": 2021.5,
            "ci_mrca": [2020.2, 2022.0],
        }
        category = classify_community(fit, is_pure_local=True, is_hybrid=False)
        assert category == "Active Transmission Outbreak"

    def test_emergent_cluster_classification(self):
        fit = {
            "n": 5,
            "mu": 0.0015,
            "r2": 0.35,
            "tmrca": 2006.0,
            "ci_mrca": [2003.0, 2007.5],
        }
        category = classify_community(fit, is_pure_local=True, is_hybrid=False)
        assert category == "Emergent Transmission Cluster"

    def test_chronic_reservoir_classification(self):
        fit = {
            "n": 20,
            "mu": 0.0001,
            "r2": 0.01,
            "tmrca": 1985.0,
            "ci_mrca": [1970.0, 1995.0],
        }
        category = classify_community(fit, is_pure_local=True, is_hybrid=False)
        assert category == "Chronic Reservoir Network"

    def test_wedged_circulating_lineage_classification(self):
        fit = {
            "n": 10,
            "mu": 0.005,
            "r2": 0.75,
            "tmrca": 2015.0,
            "ci_mrca": [2012.0, 2017.0],
        }
        category = classify_community(fit, is_pure_local=False, is_hybrid=True)
        assert category == "Wedged Circulating Lineage"

    def test_micro_chain_classification(self):
        fit = {
            "n": 2,
            "mu": 0.0,
            "r2": 0.0,
            "tmrca": 2022.0,
            "ci_mrca": [2022.0, 2022.0],
        }
        category = classify_community(fit, is_pure_local=True, is_hybrid=False)
        assert category == "Micro-Chain / Pair"


def _write_single_clock_fasta(tmp_path, n=40, seq_len=600, seed=3):
    """Write a tiny single-clock FASTA + dates CSV and return (fa, csv) paths."""
    rng = np.random.default_rng(seed)
    bases = np.array(list("ACGT"))
    ancestor = rng.choice(bases, size=seq_len)
    fa = tmp_path / "aln.fasta"
    csv = tmp_path / "dates.csv"
    with open(fa, "w") as fh, open(csv, "w") as ch:
        ch.write("id,date\n")
        for i in range(n):
            date = 2000.0 + (i % 20)
            n_mut = max(0, int(3.0 * (date - 2000.0)) + int(rng.integers(-1, 2)))
            seq = ancestor.copy()
            if n_mut:
                sites = rng.choice(seq_len, size=min(n_mut, seq_len), replace=False)
                for s in sites:
                    seq[s] = rng.choice([b for b in "ACGT" if b != seq[s]])
            tid = f"t{i:03d}"
            fh.write(f">{tid}\n{''.join(seq)}\n")
            ch.write(f"{tid},{date}\n")
    return str(fa), str(csv)


def test_hierarchical_summary_carries_flat_engine_alias_keys(tmp_path):
    """Issue #57: the hierarchical run() summary's leaf_communities entries must expose the
    same key names as the flat AutoClockDeconvolution community schema / AUTOCLOCK_GUIDE
    (community_id / calibrated_rate / calibrated_tmrca), in addition to the legacy
    node_id / rate / tmrca keys, so both engines' summaries are consumable uniformly."""
    from chronaeon.autoclock import run_hierarchical_autoclock

    fa, csv = _write_single_clock_fasta(tmp_path)
    summary = run_hierarchical_autoclock(
        alignment_path=fa, dates_source=csv, date_col="date", strain_col="id",
        manifold="distance", min_leaf_size=5, max_depth=2,
        output_dir=str(tmp_path / "out"), quiet=True,
    )
    assert "leaf_communities" in summary
    leaves = summary["leaf_communities"]
    assert isinstance(leaves, list) and len(leaves) >= 1
    for lc in leaves:
        # new flat-aligned aliases (the #57 fix)
        for k in ("community_id", "calibrated_rate", "calibrated_tmrca"):
            assert k in lc, f"leaf_communities entry missing alias key {k} (#57)"
        # legacy keys retained (backward compat) and aliases mirror them
        assert lc["community_id"] == lc["node_id"]
        assert lc["calibrated_rate"] == lc["rate"]
        assert lc["calibrated_tmrca"] == lc["tmrca"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
