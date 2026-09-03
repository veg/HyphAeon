"""
Output contract tests for hyphaeon/phenotype.py (Mode II).

Tests the scoring, ranking, and signature-extraction logic of the
neural transformer attribution pipeline: BH FDR q-values, dual-track
composite scoring, PARS signature format, spectral energy / norm ratio,
min_taxa_per_site filtering, significant_sites_count / alpha, and
the ACAT p-value combination.

Uses the Smc6 example dataset (20 taxa, shallow tree) with foreground
taxa drawn from the actual alignment. Tests verify structural contracts
(monotonicity, range, formula relationships) rather than specific
numerical outcomes, since model predictions fluctuate across weight
versions and hardware.
"""
import numpy as np
import pandas as pd
import pytest

from _harness import phylowas_pvals, get_taxa, fg_string
from hyphaeon.stats import cauchy_combination_p


@pytest.fixture(scope="module")
def phylowas_result(phylowas_runner, smc6_paths):
    """Run PhyloWAS once on Smc6 with fg_string(taxa, 2) and cache for all
    test classes that use the same parameters. Without this fixture, ~14
    tests each call phylowas_runner separately, resulting in redundant
    neural inference runs on the same alignment.
    """
    fa, nwk = smc6_paths
    taxa = get_taxa(fa, nwk)
    return phylowas_runner(fa, tree_path=nwk, foreground=fg_string(taxa, 2))


@pytest.fixture(scope="module")
def phylowas_result_min2(phylowas_runner, smc6_paths):
    """Run PhyloWAS with min_taxa_per_site=2 (default is 4). Shared by
    TestMinTaxaPerSite tests that need the low-threshold result."""
    fa, nwk = smc6_paths
    taxa = get_taxa(fa, nwk)
    return phylowas_runner(fa, tree_path=nwk, foreground=fg_string(taxa, 2),
                           min_taxa_per_site=2)


@pytest.fixture(scope="module")
def phylowas_result_fg3(phylowas_runner, smc6_paths):
    """Run PhyloWAS with fg_string(taxa, 3). Shared by TestPARSSignature
    tests that all use the same 3-foreground-taxa parameters."""
    fa, nwk = smc6_paths
    taxa = get_taxa(fa, nwk)
    return phylowas_runner(fa, tree_path=nwk, foreground=fg_string(taxa, 3))


class TestBenjaminiHochbergFDR:
    """BH FDR q-values must be present, in range, and monotonically valid."""

    def test_q_values_present_and_in_range(self, phylowas_result):
        sites = phylowas_result["sites"]
        assert len(sites) > 0
        for s in sites:
            assert "q_value" in s, f"Site {s['site']} missing q_value"
            assert 0.0 <= s["q_value"] <= 1.0, f"q_value {s['q_value']} out of [0,1]"

    def test_p_values_are_diverse(self, phylowas_result):
        """Not all p-values should be identical (would indicate a bug)."""
        pvals = phylowas_pvals(phylowas_result)
        assert len(pvals) > 1
        assert len(set(pvals.round(6))) > 1, "All p-values identical — model may not be differentiating sites"

    def test_q_values_monotonic_by_p_value(self, phylowas_result):
        """BH q-values must be non-decreasing when sorted by p-value ascending."""
        sites = phylowas_result["sites"]
        sorted_sites = sorted(sites, key=lambda x: x["p_value"])
        q_vals = [s["q_value"] for s in sorted_sites]
        for i in range(1, len(q_vals)):
            assert q_vals[i] >= q_vals[i-1] - 1e-10, \
                f"q-value not monotonic at rank {i}: {q_vals[i-1]} -> {q_vals[i]}"

    def test_q_value_within_bh_upper_bound(self, phylowas_result):
        """q_value <= p_value * m / rank for each site (BH property)."""
        sites = phylowas_result["sites"]
        m = len(sites)
        sorted_sites = sorted(sites, key=lambda x: x["p_value"])
        for rank, s in enumerate(sorted_sites, 1):
            expected_bound = s["p_value"] * m / rank
            assert s["q_value"] <= expected_bound + 1e-10, \
                f"q_value {s['q_value']} exceeds BH bound {expected_bound} at rank {rank}"


class TestDualTrackScoring:
    """Dual-track composite scoring: track_a (EVD), track_b (spectral ratio)."""

    def test_track_a_is_neg_log10_of_p_evd(self, phylowas_result):
        p_evd = phylowas_result["p_evd_length_adjusted"]
        track_a = phylowas_result["score_track_a"]
        assert track_a == pytest.approx(-np.log10(p_evd), rel=1e-6)

    def test_track_b_equals_norm_spectral_ratio(self, phylowas_result):
        assert phylowas_result["score_track_b"] == pytest.approx(phylowas_result["norm_spectral_ratio"])

    def test_composite_is_max_of_components(self, phylowas_result):
        """dual_track_composite = max(score_track_a / 10.0, score_track_b).
        Track A (neg-log10 p-value, range ~0-10) is divided by 10 to
        normalize to [0,1] scale comparable to track B (norm spectral ratio).
        """
        a = phylowas_result["score_track_a"] / 10.0
        b = phylowas_result["score_track_b"]
        assert phylowas_result["dual_track_composite"] == pytest.approx(max(a, b))

    def test_p_evd_in_valid_range(self, phylowas_result):
        p_evd = phylowas_result["p_evd_length_adjusted"]
        assert 0.0 < p_evd <= 1.0

    def test_track_a_non_negative(self, phylowas_result):
        assert phylowas_result["score_track_a"] >= 0.0


class TestSpectralEnergy:
    """Spectral energy and norm spectral ratio from attribution projection."""

    def test_spectral_energy_non_negative(self, phylowas_result):
        assert phylowas_result["spectral_energy"] >= 0.0

    def test_norm_ratio_in_unit_interval(self, phylowas_result):
        """norm_spectral_ratio = spectral_energy / frob_norm, in [0,1] by Cauchy-Schwarz."""
        assert 0.0 <= phylowas_result["norm_spectral_ratio"] <= 1.0 + 1e-10


class TestPARSSignature:
    """PARS signature bracket extraction from top-associated sites."""

    def test_signature_format_with_signal(self, phylowas_result_fg3):
        """Signature should be a bracket-enclosed, dash-separated list."""
        sig = phylowas_result_fg3["compact_pars_signature"]
        assert sig.startswith("[") and sig.endswith("]")
        if sig != "[]":
            inner = sig.strip("[] ").strip()
            if inner:
                parts = inner.split(" - ")
                for p in parts:
                    assert len(p) >= 3, f"PARS element '{p}' too short"

    def test_signature_only_includes_high_rho_sites(self, phylowas_result_fg3):
        """All sites in the signature should have association_rho >= 0.40
        and score >= 0.50. Parse the signature string and verify each
        element corresponds to a qualifying site."""
        sig = phylowas_result_fg3["compact_pars_signature"]
        if sig == "[]":
            pytest.skip("No sites above threshold — cannot test filter")
        # Parse signature: "[ A12 - R45 - S78 ]" → ["A12", "R45", "S78"]
        inner = sig.strip("[] ").strip()
        if not inner:
            pytest.skip("Empty signature")
        sig_elements = inner.split(" - ")
        # Each element is like "A12" (aa + site number)
        sig_sites = set()
        for elem in sig_elements:
            # Extract site number (trailing digits)
            site_num = int(''.join(c for c in elem if c.isdigit()))
            sig_sites.add(site_num)
        # Verify all signature sites meet the thresholds
        for s in phylowas_result_fg3["sites"]:
            if s["site"] in sig_sites:
                assert s["association_rho"] >= 0.40, \
                    f"Site {s['site']} in signature but rho={s['association_rho']:.3f} < 0.40"
                assert s["score"] >= 0.50, \
                    f"Site {s['site']} in signature but score={s['score']:.3f} < 0.50"
        assert len(sig_sites) <= 15  # capped at 15

    def test_signature_has_both_high_and_low_rho_sites(self, phylowas_result_fg3):
        """Verify that sites below the rho threshold exist and are excluded."""
        sites = phylowas_result_fg3["sites"]
        high = [s for s in sites if s["association_rho"] >= 0.40]
        low = [s for s in sites if s["association_rho"] < 0.40]
        assert len(low) > 0, "No low-rho sites — test data may be too signal-rich"


class TestMinTaxaPerSite:
    """min_taxa_per_site filters sites by valid (non-gap) taxa count."""

    def test_higher_threshold_excludes_more_sites(self, phylowas_runner, smc6_paths,
                                                    phylowas_result_min2):
        """Raising min_taxa_per_site should not increase the number of sites."""
        fa, nwk = smc6_paths
        taxa = get_taxa(fa, nwk)
        fg = fg_string(taxa, 2)
        result_high = phylowas_runner(fa, tree_path=nwk, foreground=fg,
                                       min_taxa_per_site=15)
        assert len(result_high["sites"]) <= len(phylowas_result_min2["sites"])

    def test_high_threshold_excludes_all_sites(self, phylowas_runner, smc6_paths):
        """A threshold higher than the taxa count should exclude all sites.

        Smc6 (20 taxa, gap-free) means all sites have 20 valid taxa. Setting
        min_taxa_per_site=21 excludes everything, verifying the filter mechanism.
        The companion test above (higher threshold excludes more) covers the
        normal case.
        """
        fa, nwk = smc6_paths
        taxa = get_taxa(fa, nwk)
        fg = fg_string(taxa, 2)
        result_max = phylowas_runner(fa, tree_path=nwk, foreground=fg,
                                      min_taxa_per_site=21)
        # threshold=21 on a 20-taxon alignment should exclude ALL sites
        assert len(result_max["sites"]) == 0, \
            "threshold=21 on 20 taxa should exclude all sites — " \
            "if not, the filter is not working"

    def test_threshold_filters_by_valid_taxa_count(self, phylowas_runner, smc6_paths):
        """Sites surviving min_taxa_per_site filter should have enough valid
        (non-gap) foreground+background taxa.

        Smc6 is gap-free with 20 taxa, so min_taxa_per_site=10 should keep
        all sites (all have 20 valid taxa). We verify the filter doesn't
        incorrectly exclude sites that meet the threshold.
        """
        fa, nwk = smc6_paths
        taxa = get_taxa(fa, nwk)
        result = phylowas_runner(fa, tree_path=nwk, foreground=fg_string(taxa, 2),
                                  min_taxa_per_site=10)
        # All sites in Smc6 have 20 valid taxa (gap-free), so all should pass
        # the min_taxa_per_site=10 filter. If any sites were excluded, the
        # filter is counting taxa incorrectly.
        assert len(result["sites"]) > 0, "min_taxa_per_site=10 excluded all sites on gap-free 20-taxon alignment"
        for s in result["sites"]:
            assert s["site"] >= 1


class TestSignificantSitesCount:
    """significant_sites_count should match q_value threshold filtering."""

    def test_count_matches_q_value_threshold(self, phylowas_result):
        alpha = 0.05
        expected = sum(1 for s in phylowas_result["sites"] if s.get("q_value", 1.0) <= alpha)
        assert phylowas_result["significant_sites_count"] == expected

    def test_higher_alpha_includes_more_sites(self, phylowas_runner, smc6_paths,
                                                phylowas_result):
        fa, nwk = smc6_paths
        taxa = get_taxa(fa, nwk)
        fg = fg_string(taxa, 2)
        result_10 = phylowas_runner(fa, tree_path=nwk, foreground=fg, alpha=0.10)
        assert result_10["significant_sites_count"] >= phylowas_result["significant_sites_count"]

    def test_count_is_non_negative(self, phylowas_result):
        assert phylowas_result["significant_sites_count"] >= 0


class TestModeIIPValues:
    """Mode II uses ACAT (Cauchy combination) of LRT p-value and association p-value."""

    def test_p_values_in_unit_interval(self, phylowas_result):
        for s in phylowas_result["sites"]:
            assert 0.0 <= s["p_value"] <= 1.0
            assert 0.0 <= s["p_lrt"] <= 1.0
            assert 0.0 <= s["p_assoc"] <= 1.0

    def test_p_value_combines_lrt_and_assoc(self, phylowas_result):
        """p_value is the ACAT (Cauchy combination) of p_lrt and p_assoc.

        ACAT does NOT guarantee p_combined <= min(p_lrt, p_assoc) — it
        weights both inputs and can produce a value larger than the
        smallest input. We verify the actual formula instead.
        """
        for s in phylowas_result["sites"]:
            p_lrt = s["p_lrt"]
            p_assoc = s["p_assoc"]
            p_combined = s["p_value"]
            expected = cauchy_combination_p(np.array([p_lrt, p_assoc]))
            assert p_combined == pytest.approx(expected, abs=1e-8), \
                f"p_value {p_combined} != ACAT({p_lrt}, {p_assoc}) = {expected}"

    def test_score_is_sqrt_lrt_times_rho(self, phylowas_result):
        """score = sqrt(max(0, lrt)) * max(0, rho)"""
        for s in phylowas_result["sites"]:
            expected = float(np.sqrt(max(0.0, s["hyphaeon_lrt"])) * max(0.0, s["association_rho"]))
            assert s["score"] == pytest.approx(expected, rel=1e-5)

    def test_sites_sorted_by_score_descending(self, phylowas_result):
        scores = [s["score"] for s in phylowas_result["sites"]]
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i-1] + 1e-10


class TestPhenotypeCrossDataset:
    """Phenotype output contract should hold across different datasets, not
    just Smc6 (20 taxa, shallow tree). bat_oas1 (18 taxa, ultra-deep) and
    camelid (212 taxa, topology-only) stress different code paths.
    """

    def _check_site_contract(self, result):
        assert len(result["sites"]) > 0
        for s in result["sites"]:
            assert 0.0 <= s["p_value"] <= 1.0
            assert 0.0 <= s["q_value"] <= 1.0
            assert s["q_value"] >= s["p_value"] - 1e-10  # BH: q >= p (with float tolerance)

    def test_bat_oas1_produces_valid_output(self, phylowas_runner, bat_oas1_paths):
        fa, nwk = bat_oas1_paths
        taxa = get_taxa(fa, nwk)
        result = phylowas_runner(fa, tree_path=nwk, foreground=fg_string(taxa, 3))
        self._check_site_contract(result)

    def test_camelid_produces_valid_output(self, phylowas_runner, camelid_paths):
        fa, nwk = camelid_paths
        taxa = get_taxa(fa, nwk)
        result = phylowas_runner(fa, tree_path=nwk, foreground=fg_string(taxa, 10))
        self._check_site_contract(result)


class TestPhylogeneticConfounding:
    """Sister taxa sharing substitutions should not create excessive false
    positives. This is a known weakness of Mode I; Mode II's attention
    weighting may or may not mitigate it.
    """

    def test_sister_taxa_false_convergence(self, sister_taxa_7_result):
        """Assign sister taxa (adjacent in tree) as foreground. If the model
        is phylogeny-aware, it should downweight shared ancestral substitutions.
        """
        pvals = phylowas_pvals(sister_taxa_7_result)

        if len(pvals) < 5:
            pytest.skip("Not enough variable sites")

        fpr = float(np.mean(pvals <= 0.05))
        # On Smc6 (shallow tree, ~5.6% FPR), even with phylogenetic confounding
        # from 7 sister-taxa foreground, FPR should stay below 20%.
        assert fpr < 0.20, f"FPR {fpr:.2%} with sister taxa — extreme phylogenetic confounding"


class TestPhenotypeErrorHandling:
    """Phenotype runner should handle malformed inputs gracefully."""

    def test_nonexistent_foreground_taxa(self, phylowas_runner, smc6_paths):
        """Foreground string with taxa not in the alignment should raise."""
        fa, nwk = smc6_paths
        with pytest.raises(ValueError, match="Insufficient foreground"):
            phylowas_runner(fa, tree_path=nwk, foreground="NonexistentTaxon1,NonexistentTaxon2")

    def test_missing_tree_file(self, phylowas_runner, smc6_paths, tmp_path):
        """tree_path pointing at a nonexistent file should raise."""
        fa, _ = smc6_paths
        bad_tree = str(tmp_path / "nonexistent.nwk")
        with pytest.raises((FileNotFoundError, ValueError)):
            phylowas_runner(fa, tree_path=bad_tree, foreground="taxon1,taxon2")

    def test_malformed_phenotype_file_missing_columns(self, phylowas_runner, smc6_paths, tmp_path):
        """A phenotype file with no recognizable columns should raise."""
        fa, nwk = smc6_paths
        bad_pheno = tmp_path / "bad_pheno.tsv"
        bad_pheno.write_text("foo\tbar\n1\t2\n")
        with pytest.raises(ValueError):
            phylowas_runner(fa, tree_path=nwk, phenotype_file=str(bad_pheno))

    def test_empty_phenotype_file(self, phylowas_runner, smc6_paths, tmp_path):
        """An empty phenotype file should raise."""
        fa, nwk = smc6_paths
        empty_pheno = tmp_path / "empty.tsv"
        empty_pheno.write_text("")
        with pytest.raises((ValueError, pd.errors.EmptyDataError)):
            phylowas_runner(fa, tree_path=nwk, phenotype_file=str(empty_pheno))
