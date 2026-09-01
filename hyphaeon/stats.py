"""
hyphaeon/stats.py
-----------------
Shared statistical functions: LRT → p-value conversion, Benjamini-Hochberg
FDR q-values, and Cauchy Combination Test (CCT).

These are used by the CLI (cmd_meme, cmd_busted), model_eval/_harness.py,
hyphaeon/filter.py, hyphaeon/epistasis.py, and hyphaeon/phenotype.py.
Centralizing them here ensures all call sites use the same formula.
"""

import numpy as np
from scipy.stats import chi2


def pvals_from_lrt_meme(lrts: np.ndarray) -> np.ndarray:
    """Convert LRTs to p-values using the MEME asymptotic mixture null.

    MEME.bf null: 1/3 * delta(0) + 2/3 * (0.45 * chi2(1) + 0.55 * chi2(2))

    Sites with LRT = 0 get p = 2/3 (the point mass at 0). Sites with LRT > 0
    use the continuous mixture tail.

    This is the formula HyPhy MEME uses and the one HyphAeon's neural model
    is trained to mimic (training targets are MEME LRTs).
    """
    pvals = np.full(len(lrts), 2.0 / 3.0, dtype=np.float64)
    pos = lrts > 0.0
    if np.any(pos):
        pvals[pos] = (2.0 / 3.0) * (0.45 * chi2.sf(lrts[pos], df=1) + 0.55 * chi2.sf(lrts[pos], df=2))
    return pvals


def pvals_from_lrt_self_liang(lrts: np.ndarray) -> np.ndarray:
    """Convert LRTs to p-values using the Self & Liang (1987) mixture null.

    Null: 0.5 * delta(0) + 0.5 * chi2(1)

    Used by BUSTED (whole-gene omnibus test) and by attribution p-values
    (single-parameter counterfactual tests). Not appropriate for per-site
    MEME-style p-values — use pvals_from_lrt_meme for those.
    """
    pvals = np.ones(len(lrts), dtype=np.float64)
    pos = lrts > 0.0
    if np.any(pos):
        pvals[pos] = 0.5 * chi2.sf(lrts[pos], df=1)
    return pvals


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Compute Benjamini-Hochberg FDR q-values from p-values.

    Returns q-values clipped to [0, 1], same length as input.
    """
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=np.float64)
    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    qvals = np.zeros(n, dtype=np.float64)
    min_q = 1.0
    for i in range(n - 1, -1, -1):
        q = sorted_p[i] * n / (i + 1)
        if q < min_q:
            min_q = q
        qvals[i] = min_q
    res = np.zeros(n, dtype=np.float64)
    res[sorted_idx] = qvals
    return np.clip(res, 0.0, 1.0)


def cauchy_combination_p(pvals: np.ndarray) -> float:
    """Compute the Cauchy Combination Test (CCT) omnibus p-value.

    Combines per-site p-values into a single gene-level p-value using the
    Cauchy combination: tan((0.5 - p) * pi) averaged, then back-transformed.
    Robust to dependency among the input p-values.
    """
    if len(pvals) == 0:
        return 1.0
    p_clipped = np.clip(pvals, 1e-15, 1.0 - 1e-15)
    t = np.mean(np.tan((0.5 - p_clipped) * np.pi))
    p_cct = 0.5 - (np.arctan(t) / np.pi)
    return float(np.clip(p_cct, 1e-15, 1.0))
