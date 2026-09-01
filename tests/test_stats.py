"""Unit tests for hyphaeon.stats shared statistical functions."""
import numpy as np
import pytest
from scipy.stats import chi2

from hyphaeon.stats import (
    pvals_from_lrt_meme,
    pvals_from_lrt_self_liang,
    benjamini_hochberg,
    cauchy_combination_p,
)


class TestPvalsFromLrtMeme:
    """MEME asymptotic mixture: 1/3·δ(0) + 2/3·(0.45·χ²₁ + 0.55·χ²₂)."""

    def test_zero_lrt_returns_two_thirds(self):
        """LRT=0 falls in the point mass at 0, so p = 2/3."""
        lrts = np.array([0.0, 0.0, 0.0])
        pvals = pvals_from_lrt_meme(lrts)
        assert np.allclose(pvals, 2.0 / 3.0)

    def test_positive_lrt_uses_mixture(self):
        """LRT>0 uses the 2/3·(0.45·χ²₁ + 0.55·χ²₂) tail."""
        lrt = 3.84  # ~p=0.05 for χ²₁
        pvals = pvals_from_lrt_meme(np.array([lrt]))
        expected = (2.0 / 3.0) * (0.45 * chi2.sf(lrt, df=1) + 0.55 * chi2.sf(lrt, df=2))
        assert np.isclose(pvals[0], expected)

    def test_mixed_zero_and_positive(self):
        lrts = np.array([0.0, 3.84, 0.0, 10.0])
        pvals = pvals_from_lrt_meme(lrts)
        assert pvals[0] == 2.0 / 3.0
        assert pvals[2] == 2.0 / 3.0
        assert pvals[1] < 2.0 / 3.0
        assert pvals[3] < pvals[1]  # higher LRT → lower p

    def test_monotonic_decreasing(self):
        """P-values should decrease as LRT increases."""
        lrts = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
        pvals = pvals_from_lrt_meme(lrts)
        assert np.all(np.diff(pvals) < 0)

    def test_empty_input(self):
        pvals = pvals_from_lrt_meme(np.array([]))
        assert len(pvals) == 0

    def test_pvalues_in_valid_range(self):
        lrts = np.array([0.0, 0.1, 1.0, 10.0, 100.0])
        pvals = pvals_from_lrt_meme(lrts)
        assert np.all(pvals >= 0.0) and np.all(pvals <= 1.0)


class TestPvalsFromLrtSelfLiang:
    """Self & Liang (1987): 0.5·δ(0) + 0.5·χ²₁."""

    def test_zero_lrt_returns_one(self):
        """LRT=0 falls in the point mass at 0, so p = 0.5·1 = 0.5... 
        Actually: p = 1.0 for LRT=0 (the full point mass)."""
        lrts = np.array([0.0, 0.0])
        pvals = pvals_from_lrt_self_liang(lrts)
        assert np.allclose(pvals, 1.0)

    def test_positive_lrt_uses_half_chi2(self):
        lrt = 3.84
        pvals = pvals_from_lrt_self_liang(np.array([lrt]))
        expected = 0.5 * chi2.sf(lrt, df=1)
        assert np.isclose(pvals[0], expected)

    def test_meme_and_self_liang_differ(self):
        """The two formulas should give different p-values for the same LRT.
        This is the bug we fixed: _harness.py claimed to match the CLI but
        used Self & Liang instead of the MEME mixture."""
        lrt = 5.0
        p_meme = pvals_from_lrt_meme(np.array([lrt]))[0]
        p_sl = pvals_from_lrt_self_liang(np.array([lrt]))[0]
        assert not np.isclose(p_meme, p_sl)

    def test_empty_input(self):
        pvals = pvals_from_lrt_self_liang(np.array([]))
        assert len(pvals) == 0


class TestBenjaminiHochberg:
    """BH FDR q-value computation."""

    def test_empty_input(self):
        qvals = benjamini_hochberg(np.array([]))
        assert len(qvals) == 0

    def test_single_pvalue(self):
        """Single test: q = p (no adjustment needed)."""
        qvals = benjamini_hochberg(np.array([0.05]))
        assert np.isclose(qvals[0], 0.05)

    def test_qvalues_in_valid_range(self):
        pvals = np.array([0.01, 0.05, 0.1, 0.5, 0.9])
        qvals = benjamini_hochberg(pvals)
        assert np.all(qvals >= 0.0) and np.all(qvals <= 1.0)

    def test_qvalues_monotone_with_pvalues(self):
        """Lower p-values should have lower (or equal) q-values."""
        pvals = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        qvals = benjamini_hochberg(pvals)
        # The smallest p should have the smallest q
        assert qvals[0] <= qvals[-1]

    def test_known_result(self):
        """Verify against a hand-computed example.
        p = [0.01, 0.02, 0.03, 0.04, 0.05], n=5
        Sorted: same order, ranks 1-5
        raw_q = [0.05, 0.05, 0.05, 0.05, 0.05]
        After cumulative min from right: [0.05, 0.05, 0.05, 0.05, 0.05]
        """
        pvals = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        qvals = benjamini_hochberg(pvals)
        # All raw q-values are p*n/rank = 0.01*5/1=0.05, 0.02*5/2=0.05, etc.
        assert np.allclose(qvals, 0.05)

    def test_unsorted_input_preserves_order(self):
        """q-values should be returned in the same order as input p-values."""
        pvals = np.array([0.5, 0.01, 0.3, 0.001, 0.2])
        qvals = benjamini_hochberg(pvals)
        # The smallest p (index 3) should have the smallest q
        assert qvals[3] <= qvals[0]
        assert qvals[3] <= qvals[1]

    def test_all_significant(self):
        """All p=0 → all q=0."""
        pvals = np.zeros(10)
        qvals = benjamini_hochberg(pvals)
        assert np.allclose(qvals, 0.0)

    def test_all_nonsignificant(self):
        """All p=1 → all q=1."""
        pvals = np.ones(10)
        qvals = benjamini_hochberg(pvals)
        assert np.allclose(qvals, 1.0)


class TestCauchyCombinationP:
    """Cauchy Combination Test (ACAT/CCT) omnibus p-value."""

    def test_empty_input(self):
        assert cauchy_combination_p(np.array([])) == 1.0

    def test_single_pvalue(self):
        """CCT of a single p-value should return that p-value (approximately)."""
        p = 0.05
        result = cauchy_combination_p(np.array([p]))
        assert np.isclose(result, p, atol=1e-10)

    def test_uniform_pvalues(self):
        """CCT of uniform p-values should be ~0.5."""
        pvals = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = cauchy_combination_p(pvals)
        assert 0.3 < result < 0.7

    def test_all_significant(self):
        """All small p-values → small combined p."""
        pvals = np.array([0.001, 0.001, 0.001, 0.001])
        result = cauchy_combination_p(pvals)
        assert result < 0.01

    def test_all_nonsignificant(self):
        """All large p-values → large combined p."""
        pvals = np.array([0.5, 0.5, 0.5, 0.5])
        result = cauchy_combination_p(pvals)
        assert result > 0.3

    def test_one_significant_among_many(self):
        """A single small p among large ones should pull the combined p down."""
        pvals_with_signal = np.array([0.001, 0.5, 0.5, 0.5, 0.5])
        pvals_no_signal = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        p_with = cauchy_combination_p(pvals_with_signal)
        p_without = cauchy_combination_p(pvals_no_signal)
        assert p_with < p_without

    def test_result_in_valid_range(self):
        pvals = np.array([0.001, 0.01, 0.1, 0.5, 0.99])
        result = cauchy_combination_p(pvals)
        assert 0.0 <= result <= 1.0

    def test_extreme_pvalues_clipped(self):
        """p=0 and p=1 should be handled without numerical issues."""
        pvals = np.array([0.0, 0.5, 1.0])
        result = cauchy_combination_p(pvals)
        assert 0.0 <= result <= 1.0
