"""
chronaeon/timetree.py
---------------------
ChronAeon TimeTree: Fast Tree-Free and Tree-Guided Time-Scaled Phylogeny Inference
via Continuous Soft-Profile Rooting, Distance Geometry, and Active-Set Tree-Isotonic WLS.

Bridges ChronAeon's continuous distance geometry (soft nucleotide profile root Q(gamma*),
analytical TN93 metric, Sherman-Morrison LOOCV outlier screening, Bartlett N_eff, and
AutoClock spectral partitioning) with an exact O(N) contracted tree-Laplacian active-set
solver and dispersion-calibrated Fieller confidence intervals to recover a complete,
temporally consistent time-tree (dated Newick + internal node dates + 95% CIs).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
from scipy import sparse, stats
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import squareform

from aeon_core import _native
from aeon_core.dataset import (
    compute_tn93_distance_matrix,
    parse_alignment_sequences,
    parse_beast_xml,
)
from .dating_io import (
    compute_time_decay_profile_divergences,
    parse_sample_dates,
)
from .dating_models import (
    compute_fieller_mrca_interval,
    run_ols_dating,
    run_restricted_spline_clock_dating,
)


@dataclass
class ChronAeonTimeTree:
    """Uniform result of a ChronAeon TimeTree run (compatible with phylodate.TimeTree)."""

    method: str
    tmrca: float
    rate: Optional[float] = None
    node_dates: Dict[str, float] = field(default_factory=dict)
    newick: Optional[str] = None
    raw: Any = None
    tmrca_ci: Optional[Tuple[float, float]] = None
    rate_ci: Optional[Tuple[float, float]] = None
    node_date_ci: Optional[Dict[Any, Tuple[float, float]]] = None
    ci_method: Optional[str] = "fieller"
    fieller_g: Optional[float] = None
    n_eff: Optional[float] = None
    dispersion: Optional[float] = None
    communities: Optional[List[Dict[str, Any]]] = None
    outliers: Optional[List[str]] = None

    def tip_dates(self) -> Dict[str, float]:
        return {k: v for k, v in self.node_dates.items() if not str(k).startswith("internal_") and k != "ROOT"}

    def skyline(
        self,
        *,
        epsilon: Optional[float] = None,
        grid_times: Optional[np.ndarray] = None,
        n_grid: int = 100,
        ci: bool = True,
        alpha: float = 0.05,
        n_ensemble: int = 80,
        seed: int = 42,
    ) -> "SkylineResult":
        """
        Computes a heterochronous coalescent skyline trajectory N_e(t) and Lineages-Through-Time
        (LTT) curve from this ChronAeonTimeTree, with optional 95% confidence intervals combining:
          1. Exact Kingman coalescent waiting-time variance (chi^2_{2 r_m} / trigamma),
          2. Dispersion-calibrated Fieller clock-rate uncertainty (g, N_eff), and
          3. Branch-length overdispersion (phi_eff) + horizontal time-shift ensemble propagation.
        """
        return compute_skyline(
            self,
            epsilon=epsilon,
            grid_times=grid_times,
            n_grid=n_grid,
            ci=ci,
            alpha=alpha,
            n_ensemble=n_ensemble,
            seed=seed,
        )


@dataclass
class SkylineResult:
    """Coalescent skyline N_e(t) and Lineages-Through-Time (LTT) result with 95% CIs."""

    grid_t: np.ndarray
    grid_ne: np.ndarray
    grid_ne_lo: Optional[np.ndarray]
    grid_ne_hi: Optional[np.ndarray]
    grid_ltt: np.ndarray
    window_t: np.ndarray
    window_ne: np.ndarray
    window_ne_lo: Optional[np.ndarray]
    window_ne_hi: Optional[np.ndarray]
    window_coals: np.ndarray
    ltt_t: np.ndarray
    ltt_k: np.ndarray
    epsilon: float
    quantum_dt: float


# =========================================================================
# 1. Fast Newick Parser & Serializer (No External Dependencies Required)
# =========================================================================

def _parse_newick_adj(newick: str) -> Tuple[List[str], Dict[int, List[Tuple[int, float]]], Optional[int]]:
    """
    Parses a Newick string into:
      - tips: list of tip names (indices 0 .. n_tips-1)
      - adj: undirected adjacency list {u: [(v, branch_length), ...]}
      - root_node: index of the syntactic root node in the Newick string
    """
    s = re.sub(r"\[&[^\]]*\]", "", newick).strip()
    if s.endswith(";"):
        s = s[:-1].strip()

    tokens = re.findall(r"\(|\)|,|:[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[^(),:;]+", s)

    tips: List[str] = []
    tip_to_id: Dict[str, int] = {}
    children_raw: List[Tuple[int, int, float]] = []
    internal_counter = 0

    # First pass: collect all tip names to assign 0..n_tips-1 to tips
    i = 0
    n_tok = len(tokens)
    while i < n_tok:
        tok = tokens[i]
        if tok == "(":
            i += 1
        elif tok in (",", ")"):
            i += 1
            # Optional internal node label after ')'
            if tok == ")" and i < n_tok and not tokens[i].startswith(":") and tokens[i] not in ("(", ")", ","):
                i += 1
            if i < n_tok and tokens[i].startswith(":"):
                i += 1
        elif tok.startswith(":"):
            i += 1
        else:
            name = tok.strip().strip("'\"")
            if name not in tip_to_id:
                tip_to_id[name] = len(tips)
                tips.append(name)
            i += 1
            if i < n_tok and tokens[i].startswith(":"):
                i += 1

    n_tips = len(tips)
    next_int = n_tips

    stack: List[int] = []
    curr_node: Optional[int] = None
    i = 0
    while i < n_tok:
        tok = tokens[i]
        if tok == "(":
            node_id = next_int
            next_int += 1
            stack.append(node_id)
            i += 1
        elif tok == ",":
            i += 1
        elif tok == ")":
            curr_node = stack.pop()
            i += 1
            # Optional internal label
            if i < n_tok and not tokens[i].startswith(":") and tokens[i] not in ("(", ")", ","):
                i += 1
            bl = 0.0
            if i < n_tok and tokens[i].startswith(":"):
                bl = max(0.0, float(tokens[i][1:]))
                i += 1
            if stack:
                children_raw.append((stack[-1], curr_node, bl))
        else:
            name = tok.strip().strip("'\"")
            leaf_id = tip_to_id[name]
            i += 1
            bl = 0.0
            if i < n_tok and tokens[i].startswith(":"):
                bl = max(0.0, float(tokens[i][1:]))
                i += 1
            if stack:
                children_raw.append((stack[-1], leaf_id, bl))
            else:
                curr_node = leaf_id

    root_id = curr_node if curr_node is not None else n_tips

    # If the syntactic root is bifurcating, we can either keep it or also provide unrooted adjacency
    adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for p, c, bl in children_raw:
        adj[p].append((c, bl))
        adj[c].append((p, bl))

    return tips, dict(adj), root_id


def _format_dated_newick(
    root_id: int,
    children: Dict[int, List[int]],
    node_dates: Dict[int, float],
    tips: List[str],
) -> str:
    """Formats a rooted dated tree as a Newick string with branch lengths in time units."""
    n_tips = len(tips)

    def _emit(u: int) -> str:
        if u < n_tips:
            return tips[u]
        ch_strs = []
        t_u = node_dates[u]
        for v in children.get(u, []):
            tau = max(0.0, node_dates[v] - t_u)
            ch_strs.append(f"{_emit(v)}:{tau:.6f}")
        return "(" + ",".join(ch_strs) + ")"

    return _emit(root_id) + ";"


# =========================================================================
# 2. Vectorized Neighbor-Joining & Distance Geometry Utilities
# =========================================================================

def fast_nj(D: np.ndarray) -> Dict[int, List[Tuple[int, float]]]:
    """
    Vectorized O(N^3) Neighbor-Joining on an N x N distance matrix D.
    Returns an undirected adjacency list over nodes 0..N-1 (tips) and N..2N-3 (internal).
    """
    n = D.shape[0]
    if n < 3:
        adj = {0: [], 1: []}
        if n == 2:
            b = max(0.0, float(D[0, 1]))
            adj[0].append((1, b))
            adj[1].append((0, b))
        return adj

    max_nodes = 2 * n - 2
    D_work = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D_work[:n, :n] = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(D_work, 0.0)

    active = list(range(n))
    adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(max_nodes)}
    next_node = n

    while len(active) > 2:
        m = len(active)
        sub_D = D_work[np.ix_(active, active)]
        r = sub_D.sum(axis=1)
        Q = (m - 2.0) * sub_D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        idx = int(np.argmin(Q))
        i_loc, j_loc = divmod(idx, m)
        if i_loc > j_loc:
            i_loc, j_loc = j_loc, i_loc
        u, v = active[i_loc], active[j_loc]
        d_uv = sub_D[i_loc, j_loc]
        delta = (r[i_loc] - r[j_loc]) / (m - 2.0)
        b_u = max(0.0, 0.5 * (d_uv + delta))
        b_v = max(0.0, 0.5 * (d_uv - delta))

        w = next_node
        next_node += 1
        adj[w].append((u, b_u))
        adj[w].append((v, b_v))
        adj[u].append((w, b_u))
        adj[v].append((w, b_v))

        # Update distances to remaining active nodes
        rem_idx = [a for k, a in enumerate(active) if k != i_loc and k != j_loc]
        if rem_idx:
            d_w = np.maximum(0.0, 0.5 * (D_work[u, rem_idx] + D_work[v, rem_idx] - d_uv))
            D_work[w, rem_idx] = d_w
            D_work[rem_idx, w] = d_w

        active.pop(j_loc)
        active.pop(i_loc)
        active.append(w)

    u, v = active[0], active[1]
    b_uv = max(0.0, float(D_work[u, v]))
    adj[u].append((v, b_uv))
    adj[v].append((u, b_uv))
    return adj


def _all_nodes_to_tips_distances(
    adj: Dict[int, List[Tuple[int, float]]],
    n_tips: int,
) -> np.ndarray:
    """Computes patristic distances from every node in `adj` to all `n_tips` leaves."""
    node_ids = sorted(adj.keys())
    max_id = max(node_ids) + 1
    dist_to_tips = np.zeros((max_id, n_tips), dtype=np.float64)

    for start in node_ids:
        visited = {start: 0.0}
        stack = [start]
        while stack:
            curr = stack.pop()
            d_c = visited[curr]
            if curr < n_tips:
                dist_to_tips[start, curr] = d_c
            for nxt, w in adj[curr]:
                if nxt not in visited:
                    visited[nxt] = d_c + w
                    stack.append(nxt)
    return dist_to_tips


def _distance_space_soft_profile_divergences(
    D: np.ndarray,
    times: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Computes the distance-space dual of ChronAeon's continuous time-weighted soft
    profile root Q(gamma*):
        d_i(gamma) = sum_j w_j(gamma) D_ij - 0.5 * sum_{j,k} w_j(gamma) w_k(gamma) D_jk
    where w_j(gamma) proportional to exp(-gamma * (t_j - t_min) / max(t_bar - t_min, 1e-5)).
    """
    t_min = float(np.min(times))
    t_bar = float(np.mean(times))
    scale = max(t_bar - t_min, 1e-5)
    t_rel = (times - t_min) / scale

    cand_gammas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0]
    best_r = -1e9
    best_d = None
    best_g = 2.0
    v_t = float(np.var(times))

    for g in cand_gammas:
        w = np.exp(-g * t_rel)
        w /= np.sum(w)
        d_raw = D @ w
        self_div = 0.5 * float(w @ d_raw)
        d_c = np.maximum(0.0, d_raw - self_div)
        v_d = float(np.var(d_c))
        if v_t > 1e-12 and v_d > 1e-16:
            r = float(np.cov(times, d_c)[0, 1] / math.sqrt(v_t * v_d))
        else:
            r = -1.0
        if r > best_r:
            best_r = r
            best_d = d_c
            best_g = g

    if best_d is None:
        w = np.ones(len(times)) / len(times)
        best_d = np.maximum(0.0, D @ w - 0.5 * float(w @ (D @ w)))
    return best_d, best_g


# =========================================================================
# 3. Closed-Form Continuous Edge Rooting with Soft-Profile Guidance
# =========================================================================

def _find_optimal_chronaeon_root(
    adj: Dict[int, List[Tuple[int, float]]],
    n_tips: int,
    times: np.ndarray,
    d_prof: np.ndarray,
    syntactic_root: Optional[int] = None,
    respect_existing_root: bool = False,
) -> Tuple[int, Dict[int, List[int]], Dict[int, float], np.ndarray, float]:
    """
    Finds the optimal root placement (either at the existing root or at fractional
    position alpha* in [0, 1] along the best edge e* = (u, v)) by minimizing the
    combined root-to-tip regression RSS and centered soft-profile divergence alignment.
    """
    dist_to_tips = _all_nodes_to_tips_distances(adj, n_tips)
    t_c = times - np.mean(times)
    s_tt = float(np.dot(t_c, t_c))
    d_prof_c = d_prof - np.mean(d_prof)

    # Check if existing syntactic root already has strong positive temporal correlation
    if respect_existing_root and syntactic_root is not None and syntactic_root in adj and len(adj[syntactic_root]) == 2:
        d_syn = dist_to_tips[syntactic_root]
        d_syn_c = d_syn - np.mean(d_syn)
        slope_syn = float(np.dot(t_c, d_syn_c) / max(s_tt, 1e-12))
        if slope_syn > 1e-9:
            children: Dict[int, List[int]] = defaultdict(list)
            blen: Dict[int, float] = {}
            visited = {syntactic_root}
            stack = [syntactic_root]
            while stack:
                curr = stack.pop()
                for nxt, w in adj[curr]:
                    if nxt not in visited:
                        visited.add(nxt)
                        children[curr].append(nxt)
                        blen[nxt] = w
                        stack.append(nxt)
            return syntactic_root, dict(children), blen, d_syn, slope_syn

    best_score = np.inf
    best_info = None

    seen_edges = set()
    for u, nbrs in adj.items():
        for v, b_e in nbrs:
            if (v, u) in seen_edges:
                continue
            seen_edges.add((u, v))

            # Tips closer to u vs v
            is_u_side = dist_to_tips[u] <= dist_to_tips[v]
            d0 = np.where(is_u_side, dist_to_tips[u], dist_to_tips[v] + b_e)
            s_vec = np.where(is_u_side, 1.0, -1.0)

            d0_c = d0 - np.mean(d0)
            s_c = s_vec - np.mean(s_vec)

            # Evaluate candidate alphas including analytical optimum on [0, 1]
            cand_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
            if b_e > 1e-12 and np.dot(s_c, s_c) > 1e-12:
                # Analytical alpha aligning centered tree distances with centered soft profile
                alpha_prof = float(np.dot(d_prof_c - d0_c, s_c) / (b_e * np.dot(s_c, s_c)))
                cand_alphas.append(float(np.clip(alpha_prof, 0.0, 1.0)))
                # Analytical alpha minimizing temporal regression RSS
                # M v = v_c - (t_c . v_c / s_tt) t_c
                m_d0 = d0_c - (np.dot(t_c, d0_c) / max(s_tt, 1e-12)) * t_c
                m_s = s_c - (np.dot(t_c, s_c) / max(s_tt, 1e-12)) * t_c
                denom_m = float(np.dot(m_s, m_s))
                if denom_m > 1e-12:
                    alpha_rss = float(-np.dot(m_d0, m_s) / (b_e * denom_m))
                    cand_alphas.append(float(np.clip(alpha_rss, 0.0, 1.0)))

            for alpha in cand_alphas:
                d_cand = d0 + (alpha * b_e) * s_vec
                d_cand_c = d_cand - np.mean(d_cand)
                slope = float(np.dot(t_c, d_cand_c) / max(s_tt, 1e-12))
                if slope <= 1e-10:
                    continue
                rss_t = float(np.sum((d_cand_c - slope * t_c) ** 2))
                rss_p = float(np.sum((d_cand_c - d_prof_c) ** 2))
                score = rss_t + 0.35 * rss_p
                if score < best_score:
                    best_score = score
                    best_info = (u, v, b_e, alpha, d_cand, slope)

    # Fallback if all slopes were non-positive (e.g., extreme noise)
    if best_info is None:
        u = next(iter(adj.keys()))
        v, b_e = adj[u][0]
        alpha = 0.5
        is_u_side = dist_to_tips[u] <= dist_to_tips[v]
        d_cand = np.where(is_u_side, dist_to_tips[u], dist_to_tips[v] + b_e) + 0.5 * b_e * np.where(is_u_side, 1.0, -1.0)
        slope = max(1e-5, float(np.std(d_cand) / max(np.std(times), 1e-3)))
        best_info = (u, v, b_e, alpha, d_cand, slope)

    u, v, b_e, alpha, d_root, mu_init = best_info

    # Construct directed rooted tree with new root_id
    root_id = max(adj.keys()) + 1
    children = defaultdict(list)
    blen = {}
    children[root_id] = [u, v]
    blen[u] = max(0.0, alpha * b_e)
    blen[v] = max(0.0, (1.0 - alpha) * b_e)

    visited = {root_id, u, v}
    stack = [u, v]
    while stack:
        curr = stack.pop()
        for nxt, w in adj[curr]:
            if nxt not in visited:
                visited.add(nxt)
                children[curr].append(nxt)
                blen[nxt] = max(0.0, w)
                stack.append(nxt)

    return root_id, dict(children), blen, d_root, mu_init


# =========================================================================
# 4. Active-Set Contracted Tree-Laplacian WLS Solver & Fieller CIs
# =========================================================================

def _solve_tree_isotonic_wls(
    root_id: int,
    children: Dict[int, List[int]],
    blen: Dict[int, float],
    n_tips: int,
    times: np.ndarray,
    d_root: np.ndarray,
    mu_init: float,
    seq_length: int = 1000,
    outlier_weights: Optional[np.ndarray] = None,
) -> Tuple[Dict[int, float], float, float, float]:
    """
    Solves the bound-constrained Weighted Least Squares (WLS) molecular clock problem:
        min_{t_I, mu > 0} sum_{e=(p,v)} w_e (b_e - mu * (t_v - t_p))^2
        subject to t_p <= t_v for all edges e=(p,v)
    using an exact active-set contracted tree-Laplacian solver alternated with
    closed-form WLS rate updates.

    Returns:
        (node_dates, mu_opt, se_mu, phi_dispersion)
    """
    # Collect internal nodes in preorder
    internal_nodes = []
    stack = [root_id]
    while stack:
        u = stack.pop()
        if u >= n_tips:
            internal_nodes.append(u)
            for ch in reversed(children.get(u, [])):
                stack.append(ch)

    n_int = len(internal_nodes)
    int_to_idx = {u: idx for idx, u in enumerate(internal_nodes)}

    # Upper bound on every internal node: min descendant tip time
    upper_bound: Dict[int, float] = {}
    def _compute_ub(u: int) -> float:
        if u < n_tips:
            upper_bound[u] = float(times[u])
            return float(times[u])
        ub = min(_compute_ub(ch) for ch in children.get(u, []))
        upper_bound[u] = ub
        return ub
    _compute_ub(root_id)

    # Check if branch lengths are already scaled in calendar years (slope ~ 1.0)
    t_span = float(np.ptp(times))
    d_span = float(np.ptp(d_root))
    already_time_scaled = (0.85 <= mu_init <= 1.15) and (d_span > 0.5 * t_span) and (d_span > 0.25)

    L_eff = float(max(seq_length, 100))
    c_reg = 10.0

    # Root-to-tip path regression rate (immune to internal polytomy attenuation)
    t_c = times - np.mean(times)
    s_tt = float(np.dot(t_c, t_c))
    d_c = d_root - np.mean(d_root)
    mu_rtt = max(1e-7, float(np.dot(t_c, d_c) / max(s_tt, 1e-12)))
    tmrca_rtt = float(np.mean(times) - np.mean(d_root) / mu_rtt)

    # Build edge list: (p_int_idx, c_is_leaf, c_idx_or_tip_id, b_e, w_e)
    edges = []
    for u in internal_nodes:
        p_idx = int_to_idx[u]
        for ch in children.get(u, []):
            b = max(0.0, float(blen.get(ch, 0.0)))
            w = 1.0 / (b + 1e-3) if already_time_scaled else L_eff / (b + c_reg / L_eff)
            if ch < n_tips and outlier_weights is not None:
                w *= float(outlier_weights[ch])
            if ch < n_tips:
                edges.append((p_idx, True, ch, b, w))
            else:
                edges.append((p_idx, False, int_to_idx[ch], b, w))

    # Root anchor weight: anchors root node (int_idx = 0) weakly to root-to-tip intercept
    # Prevents active-set polytomy attenuation on low-divergence outbreaks while letting
    # informative branch lengths dominate on well-resolved trees.
    mean_w = float(np.mean([e[4] for e in edges])) if edges else 1.0
    w_root_anchor = 0.15 * mean_w * (mu_rtt ** 2) / max(float(np.var(d_root)) + 1e-12, 1e-6)
    w_root_anchor = float(np.clip(w_root_anchor, 1e-4, 0.5 * mean_w))

    # Helper: solve equality-constrained WLS given active internal edges and pinned tip bounds
    def _solve_for_fixed_mu(mu_val: float, active_int_edges: set, pinned_nodes: Dict[int, float]) -> np.ndarray:
        # Union-Find over internal node indices 0..n_int-1
        parent_uf = list(range(n_int))
        def find(i: int) -> int:
            while parent_uf[i] != i:
                parent_uf[i] = parent_uf[parent_uf[i]]
                i = parent_uf[i]
            return i
        def union(i: int, j: int):
            ri, rj = find(i), find(j)
            if ri != rj:
                if ri < rj:
                    parent_uf[rj] = ri
                else:
                    parent_uf[ri] = rj

        for p_i, c_i in active_int_edges:
            union(p_i, c_i)

        # Propagate pinned tip bounds to components
        comp_pinned: Dict[int, float] = {}
        for node_i, pin_t in pinned_nodes.items():
            r = find(node_i)
            if r not in comp_pinned or pin_t < comp_pinned[r]:
                comp_pinned[r] = pin_t

        # Map free components to 0..n_free-1
        comps = sorted({find(i) for i in range(n_int)})
        free_comps = [c for c in comps if c not in comp_pinned]
        comp_to_free = {c: idx for idx, c in enumerate(free_comps)}
        n_free = len(free_comps)

        if n_free == 0:
            return np.array([comp_pinned[find(i)] for i in range(n_int)], dtype=np.float64)

        rows, cols, vals = [], [], []
        rhs = np.zeros(n_free, dtype=np.float64)

        for p_i, is_leaf, c_id, b_val, w_val in edges:
            tau0 = b_val / mu_val
            rp = find(p_i)
            if is_leaf:
                t_leaf = float(times[c_id])
                if rp in comp_to_free:
                    fp = comp_to_free[rp]
                    rows.append(fp); cols.append(fp); vals.append(w_val)
                    rhs[fp] += w_val * (t_leaf - tau0)
            else:
                rc = find(c_id)
                if rp == rc:
                    continue
                p_free = rp in comp_to_free
                c_free = rc in comp_to_free
                if p_free and c_free:
                    fp, fc = comp_to_free[rp], comp_to_free[rc]
                    rows.extend([fp, fc, fp, fc])
                    cols.extend([fp, fc, fc, fp])
                    vals.extend([w_val, w_val, -w_val, -w_val])
                    rhs[fp] -= w_val * tau0
                    rhs[fc] += w_val * tau0
                elif p_free and not c_free:
                    fp = comp_to_free[rp]
                    tc_pin = comp_pinned[rc]
                    rows.append(fp); cols.append(fp); vals.append(w_val)
                    rhs[fp] += w_val * (tc_pin - tau0)
                elif not p_free and c_free:
                    fc = comp_to_free[rc]
                    tp_pin = comp_pinned[rp]
                    rows.append(fc); cols.append(fc); vals.append(w_val)
                    rhs[fc] += w_val * (tp_pin + tau0)

        # Add soft root anchor on root node (internal index 0) toward root-to-tip intercept
        root_comp = find(0)
        if root_comp in comp_to_free:
            f_root = comp_to_free[root_comp]
            t_root_target = float(np.mean(times) - np.mean(d_root) / mu_val)
            rows.append(f_root); cols.append(f_root); vals.append(w_root_anchor)
            rhs[f_root] += w_root_anchor * t_root_target

        # Tiny diagonal ridge for disconnected zero-weight components
        for f_idx in range(n_free):
            rows.append(f_idx); cols.append(f_idx); vals.append(1e-10)

        A_mat = sparse.coo_matrix((vals, (rows, cols)), shape=(n_free, n_free)).tocsc()
        t_free = spsolve(A_mat, rhs)

        t_out = np.zeros(n_int, dtype=np.float64)
        for i in range(n_int):
            r = find(i)
            if r in comp_pinned:
                t_out[i] = comp_pinned[r]
            else:
                t_out[i] = float(t_free[comp_to_free[r]])
        return t_out

    # Iterative active-set + WLS rate refinement
    mu_curr = 1.0 if already_time_scaled else max(1e-7, float(mu_init))
    active_int: set = set()
    pinned: Dict[int, float] = {}

    for outer_it in range(8):
        # Inner active-set loop to enforce t_p <= t_v
        for _ in range(25):
            t_vec = _solve_for_fixed_mu(mu_curr, active_int, pinned)
            violated = False
            for p_i, is_leaf, c_id, _b, _w in edges:
                if is_leaf:
                    t_leaf = float(times[c_id])
                    if t_vec[p_i] > t_leaf + 1e-9:
                        if p_i not in pinned or t_leaf < pinned[p_i]:
                            pinned[p_i] = t_leaf
                            violated = True
                else:
                    if t_vec[p_i] > t_vec[c_id] + 1e-9:
                        if (p_i, c_id) not in active_int:
                            active_int.add((p_i, c_id))
                            violated = True
            if not violated:
                break

        # Final postorder clamp to guarantee exact feasibility
        for u in reversed(internal_nodes):
            u_idx = int_to_idx[u]
            max_allowed = min(
                (float(times[ch]) if ch < n_tips else t_vec[int_to_idx[ch]])
                for ch in children.get(u, [])
            )
            if t_vec[u_idx] > max_allowed:
                t_vec[u_idx] = max_allowed

        if already_time_scaled:
            mu_curr = 1.0
            break

        # Update mu combining WLS and Poisson MLE (preventing zero-branch polytomy attenuation)
        num, den = 0.0, 0.0
        sum_b, sum_tau = 0.0, 0.0
        n_zero_active = 0
        for p_i, is_leaf, c_id, b_val, w_val in edges:
            tp = t_vec[p_i]
            tc = float(times[c_id]) if is_leaf else t_vec[c_id]
            tau = max(0.0, tc - tp)
            num += w_val * b_val * tau
            den += w_val * (tau ** 2)
            sum_b += b_val
            sum_tau += tau
            if b_val <= 1e-8 and tau > 1e-5:
                n_zero_active += 1

        if den > 1e-12 and num > 0.0:
            mu_wls = num / den
            mu_pois = (sum_b / sum_tau) if sum_tau > 1e-9 else mu_wls
            # Blend tree-path root-to-tip rate (mu_rtt) with branch MLE/WLS rate
            # so heterotachy or polytomy resolution never collapses the global rate.
            mu_branch = max(mu_wls, mu_pois) if n_zero_active > 0 else (0.7 * mu_wls + 0.3 * mu_pois)
            mu_new = 0.5 * mu_rtt + 0.5 * mu_branch
            mu_new = max(1e-7, mu_new)
            if abs(mu_new - mu_curr) / mu_curr < 1e-5 and outer_it >= 1:
                mu_curr = mu_new
                break
            mu_curr = mu_new
            # Reset active set every outer iteration to allow constraints to release if mu shifts
            if outer_it < 4:
                active_int.clear()
                pinned.clear()

    # Populate node_dates dictionary (both tip indices and internal node indices)
    node_dates: Dict[int, float] = {i: float(times[i]) for i in range(n_tips)}
    for u in internal_nodes:
        node_dates[u] = float(t_vec[int_to_idx[u]])

    # Compute WLS residual variance & branch overdispersion
    rss_w = 0.0
    sum_w_tau2 = 0.0
    for p_i, is_leaf, c_id, b_val, w_val in edges:
        tp = t_vec[p_i]
        tc = float(times[c_id]) if is_leaf else t_vec[c_id]
        tau = max(0.0, tc - tp)
        resid_b = b_val - mu_curr * tau
        # Weight in substitution units is L_eff / (b_val + c/L_eff)
        rss_w += w_val * (resid_b ** 2)
        sum_w_tau2 += w_val * (tau ** 2)

    df_branch = max(1.0, float(len(edges) - len(internal_nodes)))
    phi_branch = max(1.0, rss_w / df_branch)
    se_mu_wls = math.sqrt(phi_branch / max(sum_w_tau2, 1e-12))

    return node_dates, float(mu_curr), float(se_mu_wls), float(phi_branch)


# =========================================================================
# 5. Master ChronAeon TimeTree Entry Point
# =========================================================================

def date_timetree(
    tree_or_alignment: Union[str, Path, Mapping[str, str]],
    dates: Optional[Union[str, Path, Mapping[str, float]]] = None,
    *,
    alignment: Optional[Union[str, Path, Mapping[str, str]]] = None,
    seq_length: int = 1000,
    ci: bool = True,
    autoclock: bool = False,
    outlier_z_threshold: float = 3.0,
) -> ChronAeonTimeTree:
    """
    Infers a complete time-calibrated phylogeny (`ChronAeonTimeTree`) from either:
      - A sequence alignment (FASTA / NEXUS / BEAST XML path or `{taxon: seq}` dict) + tip dates, OR
      - A substitution tree (Newick string or path) + tip dates.

    Combines:
      1. Continuous time-decay soft profile rooting Q(gamma*) (Eq. 1).
      2. Fast TN93 distance manifold + Neighbor-Joining (when tree-free) or patristic distance extraction.
      3. Closed-form continuous edge rooting aligning tree distances with the centered soft profile.
      4. Sherman-Morrison LOOCV outlier screening & Huber downweighting.
      5. Exact active-set contracted tree-Laplacian WLS calibration.
      6. Dispersion-calibrated analytical Fieller 95% confidence intervals (Eq. 3) with Bartlett N_eff.
    """
    seq_dict: Optional[Dict[str, str]] = None
    input_newick: Optional[str] = None
    beast_dates: Optional[Dict[str, float]] = None

    # 1. Classify primary input: sequence dict, Newick string, or file path
    if isinstance(tree_or_alignment, Mapping):
        seq_dict = {str(k): str(v) for k, v in tree_or_alignment.items()}
    elif isinstance(tree_or_alignment, (str, Path)):
        s_in = str(tree_or_alignment).strip()
        if s_in.startswith("(") or (s_in.startswith("[&") and "(" in s_in):
            input_newick = s_in
        elif os.path.exists(s_in):
            lower_p = s_in.lower()
            if lower_p.endswith((".xml", ".xml.gz")):
                bdata = parse_beast_xml(s_in)
                seq_dict = bdata.get("sequences") or None
                beast_dates = bdata.get("dates") or None
            elif lower_p.endswith((".nwk", ".newick", ".tree", ".tre")):
                input_newick = Path(s_in).read_text().strip()
            else:
                text_head = Path(s_in).read_text(errors="ignore")[:512].lstrip()
                if text_head.startswith("(") or text_head.startswith("[&"):
                    input_newick = Path(s_in).read_text().strip()
                else:
                    seq_dict = parse_alignment_sequences(s_in)
        else:
            raise ValueError(f"Unrecognized tree or alignment input: {s_in[:80]}")

    if alignment is not None and seq_dict is None:
        if isinstance(alignment, Mapping):
            seq_dict = {str(k): str(v) for k, v in alignment.items()}
        elif os.path.exists(str(alignment)):
            seq_dict = parse_alignment_sequences(str(alignment))

    # 2. Parse tip dates
    dates_map: Dict[str, float] = {}
    if dates is not None:
        if isinstance(dates, Mapping):
            for k, v in dates.items():
                if isinstance(v, (int, float)):
                    dates_map[str(k).strip().strip("'\"")] = float(v)
                else:
                    d_parsed, _ = parse_sample_dates([str(k)], dates_source={str(k): v})
                    if str(k) in d_parsed:
                        dates_map[str(k).strip().strip("'\"")] = float(d_parsed[str(k)])
        elif isinstance(dates, (str, Path)) and os.path.exists(str(dates)):
            lines = Path(dates).read_text().splitlines()
            if lines and len(lines[0].split()) == 1 and lines[0].split()[0].isdigit():
                lines = lines[1:]
            for ln in lines:
                parts = ln.strip().split()
                if len(parts) >= 2:
                    try:
                        dates_map[parts[0].strip("'\"")] = float(parts[1])
                    except ValueError:
                        pass
    elif beast_dates:
        dates_map = {str(k).strip().strip("'\""): float(v) for k, v in beast_dates.items()}

    # 3. Extract taxa, distance matrix D, and soft-profile divergences
    respect_existing_root = False
    syntactic_root = None

    if input_newick is not None:
        tips, adj, syntactic_root = _parse_newick_adj(input_newick)
        n_tips = len(tips)
        if not dates_map:
            dates_map, _ = parse_sample_dates(tips)
        times = np.array([dates_map[t] for t in tips], dtype=np.float64)
        dist_all = _all_nodes_to_tips_distances(adj, n_tips)
        D = dist_all[:n_tips, :n_tips].copy()
        if seq_dict is not None and all(t in seq_dict for t in tips):
            seq_length = len(seq_dict[tips[0]])
            d_prof, _ = compute_time_decay_profile_divergences(seq_dict, dates_map, tips)
        else:
            d_prof, _ = _distance_space_soft_profile_divergences(D, times)
            # If the input Newick is already rooted at a bifurcation with positive clock slope, respect it
            if syntactic_root is not None and len(adj.get(syntactic_root, [])) == 2:
                respect_existing_root = True
    elif seq_dict is not None:
        raw_taxa = list(seq_dict.keys())
        if not dates_map:
            dates_map, _ = parse_sample_dates(raw_taxa)
            if len(dates_map) < 3:
                for t in raw_taxa:
                    if "|" in t:
                        try:
                            dates_map[t] = float(t.rsplit("|", 1)[-1])
                        except ValueError:
                            pass
        tips = [t for t in raw_taxa if t in dates_map and np.isfinite(dates_map[t])]
        if len(tips) < 3:
            raise ValueError(f"Need at least 3 dated sequences (found {len(tips)}).")
        n_tips = len(tips)
        min_len = min(len(seq_dict[t]) for t in tips)
        seq_length = max(min_len, 1)
        trimmed_seqs = {t: seq_dict[t][:min_len] for t in tips}
        times = np.array([dates_map[t] for t in tips], dtype=np.float64)
        D = compute_tn93_distance_matrix(trimmed_seqs, tips)
        D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        d_prof, _ = compute_time_decay_profile_divergences(trimmed_seqs, dates_map, tips)
        adj = fast_nj(D)
    else:
        raise ValueError("Must provide either a Newick tree or a sequence alignment.")

    if len(set(times)) < 2:
        raise ValueError("Heterochronous molecular dating requires >= 2 distinct sampling dates.")

    # 4. Optimal ChronAeon Soft-Profile Guided Rooting
    root_id, children, blen, d_root, mu_init = _find_optimal_chronaeon_root(
        adj, n_tips, times, d_prof,
        syntactic_root=syntactic_root,
        respect_existing_root=respect_existing_root,
    )

    # 5. Analytical Sherman-Morrison LOOCV Outlier Screening (Eq. 5)
    t_c = times - np.mean(times)
    s_tt = float(np.dot(t_c, t_c))
    d_c = d_root - np.mean(d_root)
    slope_ols = max(1e-7, float(np.dot(t_c, d_c) / max(s_tt, 1e-12)))
    fitted_ols = np.mean(d_root) + slope_ols * t_c
    resid_ols = d_root - fitted_ols
    h_ii = (1.0 / n_tips) + (t_c ** 2) / max(s_tt, 1e-12)
    h_ii = np.minimum(h_ii, 0.99)
    rss_total = float(np.sum(resid_ols ** 2))
    loo_rss = np.maximum(1e-16, rss_total - (resid_ols ** 2) / (1.0 - h_ii))
    sigma_loo = np.sqrt(loo_rss / max(1, n_tips - 3))
    z_scores = resid_ols / (sigma_loo * np.sqrt(1.0 - h_ii) + 1e-12)

    outliers = [tips[i] for i in range(n_tips) if abs(z_scores[i]) >= outlier_z_threshold]
    # Smooth Huber-like weights for high-leverage outliers
    outlier_weights = np.where(
        np.abs(z_scores) <= outlier_z_threshold,
        1.0,
        (outlier_z_threshold / np.maximum(np.abs(z_scores), outlier_z_threshold)) ** 2,
    )

    # 6. Solve Active-Set Tree-Isotonic WLS for Node Dates & Clock Rate
    node_dates_int, mu_opt, se_mu_wls, phi_branch = _solve_tree_isotonic_wls(
        root_id, children, blen, n_tips, times, d_root, mu_init,
        seq_length=seq_length,
        outlier_weights=outlier_weights,
    )

    tmrca_opt = float(node_dates_int[root_id])

    # Format node_dates dictionary with tip names & internal node labels
    node_dates_out: Dict[str, float] = {}
    for i, t_name in enumerate(tips):
        node_dates_out[t_name] = float(node_dates_int[i])
    for u, dt_u in node_dates_int.items():
        if u >= n_tips:
            lbl = "ROOT" if u == root_id else f"internal_{u}"
            node_dates_out[lbl] = float(dt_u)

    dated_nwk = _format_dated_newick(root_id, children, node_dates_int, tips)

    # 7. Dispersion-Calibrated Analytical Fieller Confidence Intervals (Eq. 3 & 4)
    tmrca_ci = None
    rate_ci = None
    node_date_ci: Dict[Any, Tuple[float, float]] = {}
    fieller_g = None
    n_eff = float(n_tips)
    phi_eff = float(phi_branch)

    if ci:
        # Bartlett effective sample size N_eff = N^2 / sum_{i,j} exp(-kappa * D_ij)
        # Scale decay kappa to tree diameter so clone clusters are downweighted
        med_d = float(np.median(D[D > 0])) if np.any(D > 0) else 0.01
        kappa = 5.0 / max(med_d, 1e-5)
        C_mat = np.exp(-kappa * D)
        n_eff = float(np.clip((n_tips ** 2) / max(float(np.sum(C_mat)), 1.0), 3.0, float(n_tips)))
        df_eff = max(2.0, n_eff - 2.0)

        # Path-level Poisson overdispersion phi_rtt around the tree-calibrated root
        L_eff = float(max(seq_length, 100))
        expected_d = np.maximum(1.0 / L_eff, mu_opt * np.maximum(0.0, times - tmrca_opt))
        chi2_rtt = float(np.sum(outlier_weights * L_eff * ((d_root - mu_opt * (times - tmrca_opt)) ** 2) / expected_d))
        phi_rtt = max(1.0, chi2_rtt / max(n_tips - 2, 1))

        # Autocorrelation inflation ratio (analogous to dispersion ratio in correlated clocks)
        disp_ratio = math.sqrt(max(1.0, phi_rtt / max(phi_branch, 1.0)))
        phi_eff = max(phi_branch, phi_rtt)

        # Effective standard error on rate mu and center intercept d0_tree = mu_opt * (t_bar - tmrca_opt)
        t_bar = float(np.mean(times))
        d0_tree = mu_opt * (t_bar - tmrca_opt)

        # Combine WLS branch SE and root-to-tip regression SE scaled by dispersion
        se_mu_rtt = float(np.std(d_root - mu_opt * (times - tmrca_opt)) / math.sqrt(max(s_tt, 1e-12))) * math.sqrt(n_tips / n_eff)
        se_mu = max(se_mu_wls * disp_ratio, 0.5 * (se_mu_wls * disp_ratio + se_mu_rtt))
        se_d0 = se_mu * math.sqrt(s_tt / max(n_eff, 1.0))

        t_crit = float(stats.t.ppf(0.975, df=df_eff))
        rate_ci = (max(0.0, float(mu_opt - t_crit * se_mu)), float(mu_opt + t_crit * se_mu))

        fieller_g = float(((t_crit * se_mu) / max(mu_opt, 1e-12)) ** 2)
        if fieller_g < 1.0:
            denom = 1.0 - fieller_g
            center = d0_tree / mu_opt
            disc = (center ** 2) - denom * ((d0_tree ** 2 - (t_crit * se_d0) ** 2) / (mu_opt ** 2))
            if disc >= 0.0:
                delta = math.sqrt(disc)
                theta_lo = (center - delta) / denom
                theta_hi = (center + delta) / denom
                ci_lo = float(t_bar - theta_hi)
                ci_hi = float(min(np.min(times), t_bar - theta_lo))
                tmrca_ci = (ci_lo, ci_hi)
            else:
                tmrca_ci = (-math.inf, float(np.min(times)))
        else:
            tmrca_ci = (-math.inf, float(np.min(times)))

        # Per-internal-node 95% CIs keyed by descendant tip frozenset (matching phylodate convention)
        desc_tips: Dict[int, frozenset] = {}
        def _get_clade(u: int) -> frozenset:
            if u < n_tips:
                s = frozenset([tips[u]])
            else:
                s = frozenset().union(*(_get_clade(ch) for ch in children.get(u, [])))
            desc_tips[u] = s
            return s
        _get_clade(root_id)

        root_half_width = 0.5 * (tmrca_ci[1] - tmrca_ci[0]) if (tmrca_ci and math.isfinite(tmrca_ci[0])) else t_crit * (se_d0 / max(mu_opt, 1e-12))
        total_span = max(float(np.max(times) - tmrca_opt), 1e-6)
        for u in children:
            if u >= n_tips:
                t_u = node_dates_int[u]
                clade_u = desc_tips[u]
                min_tip_u = min(dates_map[t_lbl] for t_lbl in clade_u)
                rel_depth = max(0.05, (min_tip_u - t_u) / total_span)
                hw_u = root_half_width * math.sqrt(rel_depth)
                node_date_ci[clade_u] = (float(t_u - hw_u), float(min(min_tip_u, t_u + hw_u)))

    # 8. Optional AutoClock Spectral Partitioning
    communities = None
    if autoclock and n_tips >= 10:
        try:
            raw_comms = _native.recursive_spectral_autoclock(
                D.flatten().tolist(),
                times.tolist(),
                n_tips,
                max(5, min(25, n_tips // 4)),
                0.35,
                0.05,
            )
            communities = [dict(c) for c in raw_comms]
        except Exception:
            communities = None

    return ChronAeonTimeTree(
        method="chronaeon_timetree",
        tmrca=tmrca_opt,
        rate=mu_opt,
        node_dates=node_dates_out,
        newick=dated_nwk,
        raw={
            "root_id": root_id,
            "children": children,
            "blen": blen,
            "d_root": d_root,
            "d_prof": d_prof,
            "z_scores": z_scores,
            "seq_length": seq_length,
        },
        tmrca_ci=tmrca_ci,
        rate_ci=rate_ci,
        node_date_ci=node_date_ci if ci else None,
        ci_method="fieller" if ci else None,
        fieller_g=fieller_g,
        n_eff=n_eff,
        dispersion=phi_eff,
        communities=communities,
        outliers=outliers,
    )


# =========================================================================
# 6. Coalescent Skyline & LTT Estimation with Analytical + Ensemble CIs
# =========================================================================

def compute_skyline(
    tt: ChronAeonTimeTree,
    *,
    epsilon: Optional[float] = None,
    grid_times: Optional[np.ndarray] = None,
    n_grid: int = 100,
    ci: bool = True,
    alpha: float = 0.05,
    n_ensemble: int = 80,
    seed: int = 42,
) -> SkylineResult:
    """
    Computes a generalized heterochronous coalescent skyline N_e(t) and LTT trajectory
    from a `ChronAeonTimeTree`, with analytical + ensemble (1 - alpha) confidence intervals.

    Uncertainty propagation combines three orthogonal components via the Law of Total Variance:
      1. Coalescent process waiting-time variance conditional on node dates:
            2 r_m * N_e_hat(W_m) / N_e(W_m) ~ chi^2_{2 r_m}
      2. Global clock-rate Fieller uncertainty (sigma_mu^2 = g / t_crit^2)
      3. Branch-level overdispersion (phi_eff) and horizontal time-shift ensemble draws.
    """
    sample_times = np.sort(
        np.array(
            [float(v) for k, v in tt.node_dates.items() if k != "ROOT" and not str(k).startswith("internal_")],
            dtype=np.float64,
        )
    )
    coal_times = np.sort(
        np.array(
            [float(v) for k, v in tt.node_dates.items() if k == "ROOT" or str(k).startswith("internal_")],
            dtype=np.float64,
        )
    )
    if len(coal_times) == 0 or len(sample_times) < 2:
        raise ValueError("ChronAeonTimeTree must contain at least 2 tips and 1 internal node for skyline inference.")

    mu_hat = max(float(tt.rate or 1e-3), 1e-7)
    seq_len = int((tt.raw or {}).get("seq_length", 1000)) if isinstance(tt.raw, dict) else 1000
    L_eff = float(max(seq_len, 100))
    quantum_dt = 1.0 / (mu_hat * L_eff)

    t_min_tree = float(coal_times[0])
    t_max_tree = float(sample_times[-1])
    span = max(t_max_tree - t_min_tree, 1e-5)

    if epsilon is None:
        epsilon = max(quantum_dt, 0.035 * span)
    eps_val = float(epsilon)

    # Build elementary heterochronous intervals moving backward from t_max_tree to t_min_tree
    events: List[Tuple[float, int]] = []
    for t in sample_times:
        events.append((float(t), +1))
    for t in coal_times:
        events.append((float(t), -1))
    events.sort(key=lambda x: (-x[0], -x[1]))

    k = 0
    t_prev = events[0][0]
    elem_t_start, elem_t_end, elem_k, elem_is_coal = [], [], [], []
    ltt_t, ltt_k = [t_prev], [0]

    for t_curr, delta in events:
        dt = max(0.0, t_prev - t_curr)
        is_c = (delta == -1)
        if dt > 0.0 or is_c:
            elem_t_start.append(t_prev)
            elem_t_end.append(t_curr)
            elem_k.append(max(k, 2 if is_c else k))
            elem_is_coal.append(is_c)
        k += delta
        t_prev = t_curr
        ltt_t.append(t_curr)
        ltt_k.append(max(0, k))

    # Pool elementary intervals into generalized skyline windows (span >= eps_val, r_m >= min_r)
    min_r = 2 if len(coal_times) >= 12 else 1
    win_mid_t, win_dt, win_tree_len, win_ne, win_r = [], [], [], [], []
    acc_binom_dt = 0.0
    acc_tree_len = 0.0
    acc_dt = 0.0
    acc_coals = 0
    cur_t_hi = elem_t_start[0] if elem_t_start else t_max_tree

    for t_hi, t_lo, k_val, is_c in zip(elem_t_start, elem_t_end, elem_k, elem_is_coal):
        dt = max(0.0, t_hi - t_lo)
        binom = 0.5 * k_val * (k_val - 1) if k_val >= 2 else 0.0
        acc_binom_dt += binom * dt
        acc_tree_len += max(k_val, 2) * dt
        acc_dt += dt
        if is_c:
            acc_coals += 1
        if acc_coals >= min_r and acc_dt >= eps_val:
            ne_val = max(1e-3, acc_binom_dt / acc_coals)
            win_mid_t.append(0.5 * (cur_t_hi + t_lo))
            win_dt.append(max(acc_dt, quantum_dt))
            win_tree_len.append(max(acc_tree_len, 2.0 * quantum_dt))
            win_ne.append(ne_val)
            win_r.append(acc_coals)
            acc_binom_dt = 0.0
            acc_tree_len = 0.0
            acc_dt = 0.0
            acc_coals = 0
            cur_t_hi = t_lo

    if acc_coals >= 1:
        t_lo_last = elem_t_end[-1] if elem_t_end else cur_t_hi
        if win_ne and acc_coals < min_r:
            # Merge trailing single coalescence into the last window to avoid df=2 tail explosion
            prev_r = win_r[-1]
            tot_r = prev_r + acc_coals
            prev_binom_dt = win_ne[-1] * prev_r
            win_ne[-1] = max(1e-3, (prev_binom_dt + acc_binom_dt) / tot_r)
            win_dt[-1] = max(win_dt[-1] + acc_dt, quantum_dt)
            win_tree_len[-1] = max(win_tree_len[-1] + acc_tree_len, 2.0 * quantum_dt)
            win_r[-1] = tot_r
            win_mid_t[-1] = 0.5 * (win_mid_t[-1] + 0.5 * (cur_t_hi + t_lo_last))
        else:
            ne_val = max(1e-3, acc_binom_dt / max(acc_coals, 1))
            if ne_val <= 1e-3 and win_ne:
                ne_val = win_ne[-1]
            win_mid_t.append(0.5 * (cur_t_hi + t_lo_last))
            win_dt.append(max(acc_dt, quantum_dt))
            win_tree_len.append(max(acc_tree_len, 2.0 * quantum_dt))
            win_ne.append(ne_val)
            win_r.append(acc_coals)

    # Reverse to chronological order (oldest -> newest)
    window_t = np.array(win_mid_t[::-1], dtype=np.float64)
    window_dt = np.array(win_dt[::-1], dtype=np.float64)
    window_tree_len = np.array(win_tree_len[::-1], dtype=np.float64)
    window_ne = np.array(win_ne[::-1], dtype=np.float64)
    window_coals = np.array(win_r[::-1], dtype=np.int64)

    # Empirical-Bayes conjugate local smoothness prior (r_prior = 2.0) + kernel tree length
    r_prior = 2.0
    r_eff_vec = window_coals.astype(np.float64) + r_prior
    if len(window_t) >= 2:
        h_bw = max(eps_val, 0.05 * span)
        diff_mat = (window_t[:, None] - window_t[None, :]) / max(h_bw, 1e-6)
        K_mat = np.exp(-0.5 * (diff_mat ** 2))
        tree_len_eff_vec = np.maximum(2.0 * quantum_dt, K_mat @ window_tree_len)
    else:
        tree_len_eff_vec = np.maximum(2.0 * quantum_dt, window_tree_len)

    if grid_times is None:
        grid_t = np.linspace(t_min_tree, t_max_tree, n_grid)
    else:
        grid_t = np.asarray(grid_times, dtype=np.float64)

    def _interp_skyline_on_grid(w_t: np.ndarray, w_val: np.ndarray, g_t: np.ndarray, t_lo_b: float, t_hi_b: float) -> np.ndarray:
        out = np.full(len(g_t), np.nan, dtype=np.float64)
        mask = (g_t >= t_lo_b) & (g_t <= t_hi_b)
        if not np.any(mask) or len(w_t) == 0:
            return out
        if len(w_t) == 1:
            out[mask] = w_val[0]
        else:
            log_v = np.interp(g_t[mask], w_t, np.log(np.maximum(w_val, 1e-4)))
            out[mask] = np.exp(log_v)
        return out

    grid_ne = _interp_skyline_on_grid(window_t, window_ne, grid_t, t_min_tree, t_max_tree)
    n_c_grid = np.searchsorted(coal_times, grid_t, side="right")
    n_s_grid = np.searchsorted(sample_times, grid_t, side="right")
    grid_ltt = np.where(grid_t < t_min_tree, 0.0, np.maximum(0, 1 + n_c_grid - n_s_grid)).astype(np.float64)

    window_ne_lo = None
    window_ne_hi = None
    grid_ne_lo = None
    grid_ne_hi = None

    if ci and len(window_ne) > 0:
        df_eff = max(2.0, float(tt.n_eff or len(sample_times)) - 2.0)
        t_crit = float(stats.t.ppf(1.0 - 0.5 * alpha, df=df_eff))
        z_crit = float(stats.norm.ppf(1.0 - 0.5 * alpha))

        # 1. Clock-rate relative standard error from Fieller g or rate_ci
        if tt.fieller_g is not None and tt.fieller_g > 0.0:
            rel_se_mu = math.sqrt(float(tt.fieller_g)) / max(t_crit, 1.0)
        elif tt.rate_ci is not None and tt.rate:
            rel_se_mu = 0.5 * (tt.rate_ci[1] - tt.rate_ci[0]) / max(t_crit * tt.rate, 1e-12)
        else:
            rel_se_mu = 0.12

        phi_branch_eff = min(4.0, max(1.0, float(tt.dispersion or 1.0)))

        # 2. Analytical window CIs using Welch-Satterthwaite chi^2_{2 r_{eff,m}} + branch & clock variance
        w_lo = np.zeros_like(window_ne)
        w_hi = np.zeros_like(window_ne)
        local_hw_lo = np.zeros_like(window_ne)
        local_hw_hi = np.zeros_like(window_ne)

        for m_idx, (ne_m, r_eff_m, tlen_m) in enumerate(zip(window_ne, r_eff_vec, tree_len_eff_vec)):
            dof_c = 2.0 * float(r_eff_m)
            chi2_upper = float(stats.chi2.ppf(1.0 - 0.5 * alpha, df=dof_c))
            chi2_lower = float(stats.chi2.ppf(0.5 * alpha, df=dof_c))
            log_coal_lo = math.log(max(chi2_upper / dof_c, 1.0001))
            log_coal_hi = math.log(max(dof_c / max(chi2_lower, 1e-12), 1.0001))

            # Expected substitutions across all extant lineages in the local neighborhood of W_m
            exp_mut_m = max(1.5, 0.5 * mu_hat * L_eff * float(tlen_m))
            sigma_branch_m = math.sqrt(phi_branch_eff / exp_mut_m)

            local_hw_lo[m_idx] = math.sqrt(log_coal_lo ** 2 + (z_crit * sigma_branch_m) ** 2)
            local_hw_hi[m_idx] = math.sqrt(log_coal_hi ** 2 + (z_crit * sigma_branch_m) ** 2)

            clock_hw = t_crit * rel_se_mu
            tot_w_lo = math.sqrt(local_hw_lo[m_idx] ** 2 + clock_hw ** 2)
            tot_w_hi = math.sqrt(local_hw_hi[m_idx] ** 2 + clock_hw ** 2)
            w_lo[m_idx] = ne_m * math.exp(-tot_w_lo)
            w_hi[m_idx] = ne_m * math.exp(+tot_w_hi)

        window_ne_lo = w_lo
        window_ne_hi = w_hi

        # 3. Continuous grid CI combining local coalescent/branch half-widths with clock-rate shift ensemble
        mask_grid = (grid_t >= t_min_tree) & (grid_t <= t_max_tree)
        grid_hw_lo = np.full(len(grid_t), np.nan, dtype=np.float64)
        grid_hw_hi = np.full(len(grid_t), np.nan, dtype=np.float64)
        if np.any(mask_grid):
            if len(window_t) == 1:
                grid_hw_lo[mask_grid] = local_hw_lo[0]
                grid_hw_hi[mask_grid] = local_hw_hi[0]
            else:
                grid_hw_lo[mask_grid] = np.interp(grid_t[mask_grid], window_t, local_hw_lo)
                grid_hw_hi[mask_grid] = np.interp(grid_t[mask_grid], window_t, local_hw_hi)

        if n_ensemble > 0 and len(window_t) >= 2:
            rng = np.random.default_rng(seed)
            t_anchor = float(np.mean(sample_times))
            log_ens_draws = np.full((n_ensemble, len(grid_t)), np.nan, dtype=np.float64)
            for b in range(n_ensemble):
                # Draw global clock-rate scale factor kappa_b = mu^(b) / mu_hat
                t_draw = float(rng.standard_t(df=df_eff))
                kappa_b = math.exp(np.clip(t_draw * rel_se_mu, -0.60, 0.60))
                # Shift window times horizontally relative to tip mean and scale N_e vertically by 1/kappa_b
                w_t_b = t_anchor - (t_anchor - window_t) / kappa_b
                w_ne_b = window_ne / kappa_b
                t_lo_b = float(min(w_t_b[0], t_min_tree))
                ne_curve_b = _interp_skyline_on_grid(w_t_b, w_ne_b, grid_t, t_lo_b, t_max_tree)
                valid_b = ~np.isnan(ne_curve_b) & (ne_curve_b > 0.0)
                log_ens_draws[b, valid_b] = np.log(ne_curve_b[valid_b])

            valid_counts = np.sum(~np.isnan(log_ens_draws), axis=0)
            clock_grid_hw = np.full(len(grid_t), t_crit * rel_se_mu, dtype=np.float64)
            ok = valid_counts >= max(5, n_ensemble // 5)
            if np.any(ok):
                sd_clock = np.nanstd(log_ens_draws[:, ok], axis=0)
                clock_grid_hw[ok] = np.maximum(t_crit * rel_se_mu, z_crit * sd_clock)

            tot_grid_hw_lo = np.sqrt(grid_hw_lo ** 2 + clock_grid_hw ** 2)
            tot_grid_hw_hi = np.sqrt(grid_hw_hi ** 2 + clock_grid_hw ** 2)
            grid_ne_lo = grid_ne * np.exp(-tot_grid_hw_lo)
            grid_ne_hi = grid_ne * np.exp(+tot_grid_hw_hi)
        else:
            clock_hw = t_crit * rel_se_mu
            grid_ne_lo = grid_ne * np.exp(-np.sqrt(grid_hw_lo ** 2 + clock_hw ** 2))
            grid_ne_hi = grid_ne * np.exp(+np.sqrt(grid_hw_hi ** 2 + clock_hw ** 2))

    return SkylineResult(
        grid_t=grid_t,
        grid_ne=grid_ne,
        grid_ne_lo=grid_ne_lo,
        grid_ne_hi=grid_ne_hi,
        grid_ltt=grid_ltt,
        window_t=window_t,
        window_ne=window_ne,
        window_ne_lo=window_ne_lo,
        window_ne_hi=window_ne_hi,
        window_coals=window_coals,
        ltt_t=np.array(ltt_t[::-1], dtype=np.float64),
        ltt_k=np.array(ltt_k[::-1], dtype=np.float64),
        epsilon=eps_val,
        quantum_dt=quantum_dt,
    )
