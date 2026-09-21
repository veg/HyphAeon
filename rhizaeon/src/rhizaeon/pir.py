"""
rhizaeon.pir
============
Latent Parental Incongruence Ratio (L-PIR), Geometric Clade Bounding,
and Parental Attribution Engine.
"""

from typing import Optional, Dict, Tuple, List, Any, Union
import numpy as np


def compute_pir(
    D1: np.ndarray,
    D2: np.ndarray,
    r_idx: int,
    p1_idx: int,
    p2_idx: int,
    bound_factor: float = 1.25,
    min_parent_dist: float = 0.025
) -> Tuple[float, float, float, bool]:
    """
    Computes bounded Parental Incongruence Ratio (PIR):
      term1 = (D1(R, P2) - D1(R, P1)) / D1(P1, P2)
      term2 = (D2(R, P1) - D2(R, P2)) / D2(P1, P2)
      PIR = term1 * term2

    Enforces geometric bounding:
      D1(R, P1) <= bound_factor * D1(P1, P2)
      D2(R, P2) <= bound_factor * D2(P1, P2)
    This strictly excludes outgroups that fluctuate spuriously between two clades.
    """
    d1_p = D1[p1_idx, p2_idx]
    d2_p = D2[p1_idx, p2_idx]

    if d1_p < min_parent_dist or d2_p < min_parent_dist:
        return 0.0, 0.0, 0.0, False

    # Outgroup bounding
    if D1[r_idx, p1_idx] > bound_factor * d1_p or D2[r_idx, p2_idx] > bound_factor * d2_p:
        return 0.0, 0.0, 0.0, False

    term1 = (D1[r_idx, p2_idx] - D1[r_idx, p1_idx]) / d1_p
    term2 = (D2[r_idx, p1_idx] - D2[r_idx, p2_idx]) / d2_p

    if term1 <= 0.0 or term2 <= 0.0:
        return 0.0, term1, term2, False

    pir = term1 * term2
    return float(pir), float(term1), float(term2), True


def evaluate_triplets_for_taxon(
    D1: np.ndarray,
    D2: np.ndarray,
    r_idx: int,
    bound_factor: float = 1.25,
    min_parent_dist: float = 0.025,
    weight_by_divergence: bool = True,
    top_k: int = 1
) -> Optional[Any]:
    """
    Vectorized evaluation of all parental pairs (P1, P2) for a focal taxon R.
    Finds the optimal parental pair maximizing bounded PIR in closed matrix form:
      term1 = (D1(R, P2) - D1(R, P1)) / D1(P1, P2)
      term2 = (D2(R, P1) - D2(R, P2)) / D2(P1, P2)
      PIR = term1 * term2

    When weight_by_divergence is True, evidentiary score = PIR * (D1 + D2).
    This strictly prevents intra-clade clonal siblings with 1-2 private mutations
    from shadowing genuine divergent parental lineages.
    """
    N = D1.shape[0]
    d1_r = D1[r_idx]
    d2_r = D2[r_idx]

    # Broadcast differences: [p1, p2]
    # diff1[p1, p2] = D1(R, p2) - D1(R, p1)
    diff1 = d1_r[None, :] - d1_r[:, None]
    # diff2[p1, p2] = D2(R, p1) - D2(R, p2)
    diff2 = d2_r[:, None] - d2_r[None, :]

    # Inter-parent distance requirement
    valid = (D1 >= min_parent_dist) & (D2 >= min_parent_dist)
    valid[r_idx, :] = False
    valid[:, r_idx] = False
    np.fill_diagonal(valid, False)

    # Outgroup bounding:
    # D1(R, p1) <= bound_factor * D1(p1, p2)
    # D2(R, p2) <= bound_factor * D2(p1, p2)
    bounded = (d1_r[:, None] <= bound_factor * D1) & (d2_r[None, :] <= bound_factor * D2)
    valid &= bounded

    term1 = np.where(valid, diff1 / np.maximum(D1, 1e-9), 0.0)
    term2 = np.where(valid, diff2 / np.maximum(D2, 1e-9), 0.0)

    pos = (term1 > 0.0) & (term2 > 0.0)
    pir_mat = np.where(pos, term1 * term2, 0.0)

    score_mat = pir_mat * (D1 + D2) if weight_by_divergence else pir_mat

    max_score = float(np.max(score_mat))
    if max_score <= 0.0:
        return None if top_k == 1 else []

    if top_k == 1:
        p1_best, p2_best = np.unravel_index(np.argmax(score_mat), score_mat.shape)
        return {
            "recombinant_idx": r_idx,
            "parent_left_idx": int(p1_best),
            "parent_right_idx": int(p2_best),
            "pir": float(pir_mat[p1_best, p2_best]),
            "score": float(score_mat[p1_best, p2_best]),
            "term1": float(term1[p1_best, p2_best]),
            "term2": float(term2[p1_best, p2_best]),
            "inter_parent_d1": float(D1[p1_best, p2_best]),
            "inter_parent_d2": float(D2[p1_best, p2_best])
        }

    top_flat = np.argsort(score_mat.ravel())[::-1][:top_k]
    candidates = []
    for idx in top_flat:
        p1, p2 = np.unravel_index(idx, score_mat.shape)
        if score_mat[p1, p2] <= 0:
            break
        candidates.append({
            "recombinant_idx": r_idx,
            "parent_left_idx": int(p1),
            "parent_right_idx": int(p2),
            "pir": float(pir_mat[p1, p2]),
            "score": float(score_mat[p1, p2]),
            "term1": float(term1[p1, p2]),
            "term2": float(term2[p1, p2]),
            "inter_parent_d1": float(D1[p1, p2]),
            "inter_parent_d2": float(D2[p1, p2])
        })
    return candidates


def refine_breakpoint_codon(
    engine,
    bp: int,
    r_idx: int,
    p1_idx: int,
    p2_idx: int,
    flank_len: int = 60,
    search_radius: int = 15
) -> Tuple[int, float]:
    """
    Performs single-codon local refinement to pinpoint the exact boundary.
    Searches bp +/- search_radius to maximize PIR.
    """
    U = engine.num_units
    best_bp = bp
    best_pir = -1.0

    min_s = max(1, bp - search_radius)
    max_s = min(U - 1, bp + search_radius)

    for cand_s in range(min_s, max_s + 1):
        l_start = max(0, cand_s - flank_len)
        r_end = min(U, cand_s + flank_len)

        d1 = engine.query_distance_matrix(l_start, cand_s)
        d2 = engine.query_distance_matrix(cand_s, r_end)

        d1_p = d1[p1_idx, p2_idx]
        d2_p = d2[p1_idx, p2_idx]

        if d1_p > 1e-4 and d2_p > 1e-4:
            t1 = (d1[r_idx, p2_idx] - d1[r_idx, p1_idx]) / d1_p
            t2 = (d2[r_idx, p1_idx] - d2[r_idx, p2_idx]) / d2_p
            if t1 > 0.0 and t2 > 0.0:
                pir = t1 * t2
                if pir > best_pir:
                    best_pir = pir
                    best_bp = cand_s

    return best_bp, float(max(0.0, best_pir))
