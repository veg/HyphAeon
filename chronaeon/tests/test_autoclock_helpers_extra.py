"""Net-new pure-function edge cases for chronaeon.autoclock helpers.

These complement (do NOT duplicate) the helper coverage already in
test_autoclock.py (select_adaptive_n_landmarks user override / string none /
sublinear / memory bound; fit_fast_ols_clock linear / zero distance).

Focus here:
  - fit_fast_ols_clock degenerate branches (n<3, empty, single point),
    zero-date-variance, all-identical-distances r2=1.0 quirk (issue #56),
    negative slope, and the r2/p_val boundary at 0.3.
  - select_adaptive_n_landmarks flooring, invalid-string fallback, and the
    max_landmarks cap.
"""

import math

import numpy as np
import pytest

from chronaeon.autoclock import select_adaptive_n_landmarks, fit_fast_ols_clock


# --------------------------------------------------------------------------- #
# select_adaptive_n_landmarks (net-new edge cases)
# --------------------------------------------------------------------------- #

def test_landmarks_user_override_floored_at_min_landmarks():
    # user=50 below min_landmarks (100) -> raised to 100
    assert select_adaptive_n_landmarks(5000, user_landmarks=50, min_landmarks=100) == 100


def test_landmarks_invalid_user_string_falls_back_to_adaptive():
    # Non-numeric, non-keyword string -> int() raises -> falls through to adaptive.
    adaptive = select_adaptive_n_landmarks(5000, max_k=6)
    got = select_adaptive_n_landmarks(5000, max_k=6, user_landmarks="not_a_number")
    assert got == adaptive


def test_landmarks_max_landmarks_cap():
    # Huge n with generous memory should be capped by max_landmarks.
    r = select_adaptive_n_landmarks(10_000_000, max_landmarks=2500,
                                    max_memory_mb=1e9, min_landmarks=100)
    assert r <= 2500


# --------------------------------------------------------------------------- #
# fit_fast_ols_clock
# --------------------------------------------------------------------------- #

_CLOCK_KEYS = {"mu", "t_mrca", "ci_mrca", "rate_ci", "se_mu", "r2", "rss", "p_val"}


def _assert_finite_result(res):
    """Every scalar/list output must be finite (no NaN / inf)."""
    assert _CLOCK_KEYS.issubset(res.keys())
    for k in ("mu", "t_mrca", "se_mu", "r2", "rss", "p_val"):
        assert math.isfinite(res[k]), f"{k} not finite: {res[k]}"
    for k in ("ci_mrca", "rate_ci"):
        assert len(res[k]) == 2
        for v in res[k]:
            assert math.isfinite(v), f"{k} contains non-finite: {res[k]}"


def test_clock_degenerate_n_less_than_3():
    # n<3 -> degenerate: mu=0, r2=0, p_val=1, t_mrca=min(dates), ci_mrca=[t,t]
    dates = np.array([2005.0, 2003.0])
    dists = np.array([0.02, 0.05])
    res = fit_fast_ols_clock(dates, dists)
    _assert_finite_result(res)
    assert res["mu"] == 0.0
    assert res["r2"] == 0.0
    assert res["p_val"] == 1.0
    assert res["t_mrca"] == pytest.approx(2003.0)
    assert res["ci_mrca"] == [pytest.approx(2003.0), pytest.approx(2003.0)]
    assert res["rate_ci"] == [0.0, 0.0]


def test_clock_empty_input():
    # n==0 -> degenerate with t_mrca=0.0
    res = fit_fast_ols_clock(np.array([]), np.array([]))
    _assert_finite_result(res)
    assert res["mu"] == 0.0
    assert res["t_mrca"] == 0.0
    assert res["p_val"] == 1.0


def test_clock_single_point():
    # n==1 -> degenerate, t_mrca = the single date
    res = fit_fast_ols_clock(np.array([2011.0]), np.array([0.03]))
    _assert_finite_result(res)
    assert res["mu"] == 0.0
    assert res["t_mrca"] == pytest.approx(2011.0)
    assert res["r2"] == 0.0


def test_clock_zero_date_variance():
    # All identical dates -> var_x < 1e-12 -> mu=0, p_val=1, rss=tot_var
    dates = np.array([2010.0, 2010.0, 2010.0, 2010.0])
    dists = np.array([0.01, 0.02, 0.03, 0.04])
    res = fit_fast_ols_clock(dates, dists)
    _assert_finite_result(res)
    assert res["mu"] == 0.0
    assert res["p_val"] == 1.0
    assert res["t_mrca"] == pytest.approx(2010.0)
    # rss equals total variance of dists about its mean
    tot_var = float(np.sum((dists - dists.mean()) ** 2))
    assert res["rss"] == pytest.approx(tot_var)
    assert res["ci_mrca"] == [pytest.approx(2010.0), pytest.approx(2010.0)]


def test_clock_all_identical_dists():
    # Distances constant -> slope 0. Documents CURRENT behavior:
    # tot_var == 0 and rss == 0, so r2 = max(0, 1 - 0/max(1e-12,0)) = 1.0.
    # NOTE: quirk (issue #56) -- with zero distance variance the clock reports a
    # PERFECT fit (r2 == 1.0) and therefore p_val == 0.0, even though there is
    # no molecular-clock signal at all. A degenerate flat clock arguably should
    # yield r2 == 0.0 / p_val == 1.0 like the zero-date-variance branch does.
    dates = np.array([2000.0, 2001.0, 2002.0, 2003.0])
    dists = np.array([0.05, 0.05, 0.05, 0.05])
    res = fit_fast_ols_clock(dates, dists)
    _assert_finite_result(res)
    assert res["mu"] == pytest.approx(0.0, abs=1e-12)
    assert res["r2"] == pytest.approx(1.0, abs=1e-9)  # current (surprising) behavior
    assert res["p_val"] == 0.0
    # mu <= 1e-12 -> t_mrca falls back to dates.min()
    assert res["t_mrca"] == pytest.approx(2000.0)


def test_clock_negative_slope_finite_and_r2_nonneg():
    # Anti-correlated: distance shrinks with later dates -> negative mu.
    dates = np.array([2000.0, 2001.0, 2002.0, 2003.0, 2004.0])
    dists = np.array([0.05, 0.04, 0.03, 0.02, 0.01])
    res = fit_fast_ols_clock(dates, dists)
    _assert_finite_result(res)
    assert res["mu"] < 0.0
    assert res["r2"] >= 0.0
    # mu <= 1e-12 (negative) -> t_mrca falls back to dates.min()
    assert res["t_mrca"] == pytest.approx(2000.0)
    # rate CI lower bound is clamped at 0.0
    assert res["rate_ci"][0] >= 0.0


def test_clock_r2_pval_boundary_high():
    # A strong (but imperfect) clock with r2 > 0.3 -> p_val == 0.0
    dates = np.array([2000.0, 2001.0, 2002.0, 2003.0, 2004.0, 2005.0])
    dists = 0.02 * (dates - 2000.0) + np.array([0.0, 0.001, -0.001, 0.001, -0.001, 0.0])
    res = fit_fast_ols_clock(dates, dists)
    _assert_finite_result(res)
    assert res["r2"] > 0.3
    assert res["p_val"] == 0.0


def test_clock_r2_pval_boundary_low():
    # Essentially no temporal signal (noise dominates) -> r2 <= 0.3 -> p_val == 0.05
    dates = np.array([2000.0, 2001.0, 2002.0, 2003.0, 2004.0, 2005.0])
    dists = np.array([0.03, 0.01, 0.05, 0.00, 0.04, 0.02])  # scrambled, no trend
    res = fit_fast_ols_clock(dates, dists)
    _assert_finite_result(res)
    assert res["r2"] <= 0.3
    assert res["p_val"] == 0.05


def test_clock_se_mu_and_ci_consistency():
    # se_mu >= 0 and rate CI width == 2 * 1.96 * se_mu when lower bound not clamped.
    dates = np.array([2000.0, 2002.0, 2004.0, 2006.0, 2008.0])
    dists = 0.01 * (dates - 2000.0) + np.array([0.0, 0.002, -0.001, 0.001, 0.0])
    res = fit_fast_ols_clock(dates, dists)
    _assert_finite_result(res)
    assert res["se_mu"] >= 0.0
    lo, hi = res["rate_ci"]
    if lo > 0.0:  # not clamped
        assert (hi - lo) == pytest.approx(2 * 1.96 * res["se_mu"], rel=1e-6)
    # t_MRCA delta-method CI must be symmetric about t_mrca with non-negative half-width.
    m_lo, m_hi = res["ci_mrca"]
    t = res["t_mrca"]
    assert m_lo <= t <= m_hi
    assert (t - m_lo) == pytest.approx(m_hi - t, rel=1e-9, abs=1e-12)
    assert (m_hi - m_lo) > 0.0
