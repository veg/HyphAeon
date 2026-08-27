"""
hyphaeon/utils.py
------------------
Shared utilities: device selection, statistical helpers, amino-acid maps.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import scipy.stats as stats
import torch

from .dataset import AA_MAP

REV_AA_MAP = {v: k for k, v in AA_MAP.items()}


def select_device(cpu: bool = False) -> torch.device:
    """Select the best available hardware device (cuda > mps > cpu)."""
    if cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_taxon_index(taxa: List[str], pattern: str) -> Optional[int]:
    """Find the index of the first taxon whose name contains *pattern* (case-insensitive)."""
    pat_lower = pattern.lower()
    for idx, t in enumerate(taxa):
        if pat_lower in t.lower():
            return idx
    return None


def lrt_to_pvals(lrts: np.ndarray) -> np.ndarray:
    """
    Asymptotic p-values for LRTs under the Self & Liang (1987) mixture null:
    0.5 * delta(0) + 0.5 * chi2(1).

    Returns an array of p-values the same shape and dtype as *lrts*.
    """
    pvals = np.ones_like(lrts)
    pos_mask = lrts > 0.0
    pvals[pos_mask] = 0.5 * stats.chi2.sf(lrts[pos_mask], df=1)
    return pvals


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR q-values.

    Accepts a 1-D array of p-values and returns q-values of the same shape.
    """
    n = len(pvals)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    raw_q = pvals * (n / ranks)
    sorted_q = raw_q[order]
    for i in range(n - 2, -1, -1):
        sorted_q[i] = min(sorted_q[i], sorted_q[i + 1])
    raw_q[order] = sorted_q
    return np.clip(raw_q, 0.0, 1.0).astype(np.float32)
