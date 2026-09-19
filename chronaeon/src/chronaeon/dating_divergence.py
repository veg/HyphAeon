"""
chronaeon/dating_divergence.py
------------------------------
Root-to-tip divergence computation for molecular clock dating.

Supports three divergence modes:
- Tree-based: patristic distance traversal with heuristic root search
  (TempEst R^2 maximization).
- Tree-free: direct pairwise distance estimation (TN93) and ancestral
  consensus anchoring.
- Latent: convex-hull root optimization in neural embedding space
  (requires torch; imported lazily).
"""

from __future__ import annotations

import os
import copy
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

try:
    from Bio import Phylo
    HAS_BIOPHYLO = True
except ImportError:
    HAS_BIOPHYLO = False

from aeon_core.dataset import (
    compute_tn93_distance_matrix,
    compute_tn93_cross_distance_matrix,
)
from scipy.spatial.distance import pdist



from .dating_io import (
    generate_consensus_sequence,
    generate_time_decay_consensus_sequence,
    compute_time_decay_profile_divergences,
)

# =========================================================================
# 3. Root-to-Tip Divergence Calculation & Heuristic Rooting
# =========================================================================

def extract_tree_root_to_tip(
    tree_path: str,
    taxa: List[str],
    dates_map: Dict[str, float],
    root_taxon: Optional[str] = None,
    optimize_root: bool = True
) -> Tuple[Dict[str, float], str]:
    """
    Computes patristic root-to-tip distances from a Newick phylogenetic tree.
    If optimize_root=True, searches candidate rooting nodes to maximize
    the correlation R^2 with tip dates (TempEst / Path-O-Gen emulation).
    """
    if not HAS_BIOPHYLO:
        raise ImportError("Bio.Phylo is required to extract distances from phylogenetic trees.")

    if isinstance(tree_path, str) and (tree_path.strip().startswith('(') or not os.path.exists(tree_path)):
        tree = Phylo.read(StringIO(tree_path), 'newick')
    else:
        tree = Phylo.read(tree_path, 'newick')
    taxa_set = set(taxa)

    # Explicit user rooting
    if root_taxon:
        matching_terminals = [t for t in tree.get_terminals() if t.name == root_taxon]
        if matching_terminals:
            tree.root_with_outgroup(matching_terminals[0])
            dists = {tip.name: tree.distance(tip) for tip in tree.get_terminals() if tip.name in taxa_set}
            return dists, f"user_root_{root_taxon}"

    # Default baseline: evaluate original root
    dists_orig = {tip.name: tree.distance(tip) for tip in tree.get_terminals() if tip.name in taxa_set}
    xs_orig, ys_orig = [], []
    for name, d in dists_orig.items():
        if name in dates_map and not np.isnan(dates_map[name]):
            xs_orig.append(dates_map[name])
            ys_orig.append(d)

    orig_r2 = float(np.corrcoef(xs_orig, ys_orig)[0, 1] ** 2) if len(xs_orig) >= 5 and np.std(ys_orig) > 1e-7 and np.std(xs_orig) > 1e-7 else 0.0
    orig_slope = float(np.polyfit(xs_orig, ys_orig, 1)[0]) if len(xs_orig) >= 5 and np.std(xs_orig) > 1e-7 else 0.0

    # Calculate original root RSS and verify causal consistency (t_MRCA < min(sampling times))
    min_time_orig = float(np.min(xs_orig)) if xs_orig else 0.0
    orig_causal = False
    if len(xs_orig) >= 5 and orig_slope > 0:
        orig_inter = float(np.polyfit(xs_orig, ys_orig, 1)[1])
        orig_t_mrca = -orig_inter / orig_slope
        orig_causal = bool(orig_t_mrca < min_time_orig)
        orig_rss = float(np.sum((np.array(ys_orig) - (orig_inter + orig_slope * np.array(xs_orig))) ** 2))
    else:
        orig_rss = 1e12

    # Heuristic Root Optimization (TempEst / Path-O-Gen emulation using minimum RSS subject to causality)
    if optimize_root:
        non_terminals = tree.get_nonterminals()
        best_rss = orig_rss if (0 < orig_slope < 0.10 and orig_causal) else 1e12
        best_r2 = orig_r2 if (0 < orig_slope < 0.10 and orig_causal) else -1.0
        best_idx = None
        best_dists: Dict[str, float] = dists_orig

        # Subsample candidate nodes for large trees to avoid N^2 tree traversals
        n_non_terms = len(non_terminals)
        step = max(1, n_non_terms // 60) if n_non_terms > 60 else 1
        cand_indices = list(range(0, n_non_terms, step))

        for idx_node in cand_indices:
            t_cand = copy.deepcopy(tree)
            cand_nodes_in_t = t_cand.get_nonterminals()
            if idx_node >= len(cand_nodes_in_t):
                continue
            cand_node = cand_nodes_in_t[idx_node]
            if cand_node == t_cand.root:
                continue
            try:
                t_cand.root_with_outgroup(cand_node)
            except Exception:
                continue

            dists = {tip.name: t_cand.distance(tip) for tip in t_cand.get_terminals() if tip.name in taxa_set}
            d_vals = np.array(list(dists.values()), dtype=np.float64)
            if len(d_vals) < 5:
                continue

            sorted_d = np.sort(d_vals)
            span = float(sorted_d[-1] - sorted_d[0])
            max_adjacent_gap = float(np.max(np.diff(sorted_d)))
            med_d = float(np.median(d_vals))

            # Guard against extreme outlier / bimodal outgroup artifacts (e.g. artificial tip re-rooting)
            # Restrict outgroup rejection guards to deep alignments with span >= 0.01 substitutions per site.
            # In low-divergence outbreak regimes (span < 0.01), genuine root nodes frequently have the earliest tip
            # sampled at distance 0, and a single mutation step naturally constitutes >45% of total tree span.
            if span >= 0.01:
                if span > 1e-6 and (max_adjacent_gap / span > 0.45):
                    continue
                if med_d > 1e-6 and (sorted_d[0] < 0.10 * med_d) and (sorted_d[0] < 0.05 * sorted_d[-1]):
                    continue

            xs, ys = [], []
            for name, d in dists.items():
                if name in dates_map and not np.isnan(dates_map[name]):
                    xs.append(dates_map[name])
                    ys.append(d)

            if len(xs) >= 5 and np.std(ys) > 1e-7 and np.std(xs) > 1e-7:
                slope, inter = np.polyfit(xs, ys, 1)
                slope, inter = float(slope), float(inter)
                cand_rss = float(np.sum((np.array(ys) - (inter + slope * np.array(xs))) ** 2))
                r_val = float(np.corrcoef(xs, ys)[0, 1])

                # TempEst residual criterion: candidate must have positive plausible slope (< 0.10),
                # satisfy temporal causality (t_MRCA < min(sampling times)), and reduce residual variance (RSS)
                min_time_cand = float(np.min(xs))
                cand_t_mrca = -inter / slope if slope > 0 else 9999.0
                cand_causal = bool(cand_t_mrca < min_time_cand)
                if 0 < slope < 0.10 and cand_causal:
                    if cand_rss < best_rss * 0.95 or (best_rss >= 1e11 and r_val > 0):
                        best_rss = cand_rss
                        best_r2 = r_val ** 2
                        best_idx = idx_node
                        best_dists = dists

        if best_idx is not None:
            non_terms = tree.get_nonterminals()
            if best_idx < len(non_terms):
                try:
                    tree.root_with_outgroup(non_terms[best_idx])
                    return best_dists, f"optimized_internal_root_node_{best_idx}"
                except Exception:
                    pass

    # Default: Use original root
    return dists_orig, "original_tree_root"


def compute_tree_free_divergences(
    seq_dict: Dict[str, str],
    dated_taxa: List[str],
    dates_map: Dict[str, float],
    root_taxon: Optional[str] = None,
    decay_gamma: Optional[float] = None,
    decay_half_life: Optional[float] = None,
) -> Tuple[np.ndarray, str]:
    """
    Computes tree-free root-to-tip divergence directly from pairwise TN93 distances.
    Anchors root at:
    1. Specified root_taxon (if found in alignment).
    2. Explicit unweighted consensus if requested ('unweighted_consensus' / 'modal_consensus').
    3. Earliest sampled taxon / cohort if requested ('earliest' / 'earliest_cohort').
    4. Time-Decay Weighted Consensus (default & recommended for tree-free dating).
    """
    all_taxa = list(seq_dict.keys())

    # Case 1: User explicitly specified an existing taxon as root (e.g. outgroup or specific strain)
    if root_taxon and root_taxon in seq_dict:
        eval_taxa = [t for t in dated_taxa if t != root_taxon]
        cross_mat = compute_tn93_cross_distance_matrix(seq_dict, eval_taxa, [root_taxon])
        divergences = cross_mat[:, 0].astype(np.float64)
        return divergences, f"explicit_root_{root_taxon}"

    # Case 2: User requested unweighted modal consensus
    if root_taxon and root_taxon.lower() in ['unweighted_consensus', 'flat_consensus', 'modal_consensus']:
        con_seq = generate_consensus_sequence(seq_dict, dated_taxa)
        aug_dict = dict(seq_dict)
        aug_dict['__SYNTHETIC_CONSENSUS__'] = con_seq
        cross_mat = compute_tn93_cross_distance_matrix(aug_dict, dated_taxa, ['__SYNTHETIC_CONSENSUS__'])
        divergences = cross_mat[:, 0].astype(np.float64)
        return divergences, "unweighted_modal_consensus_root"

    # Case 2b: User requested time-decay weighted modal consensus sequence
    if root_taxon and root_taxon.lower() in ['time_decay_consensus', 'weighted_consensus', 'consensus']:
        decay_seq, eff_gamma = generate_time_decay_consensus_sequence(
            seq_dict, dates_map, dated_taxa, gamma=decay_gamma, half_life=decay_half_life
        )
        aug_dict = dict(seq_dict)
        aug_dict['__TIME_DECAY_ROOT__'] = decay_seq
        cross_mat = compute_tn93_cross_distance_matrix(aug_dict, dated_taxa, ['__TIME_DECAY_ROOT__'])
        divergences = cross_mat[:, 0].astype(np.float64)
        root_desc = f"time_decay_consensus_root (γ={eff_gamma:.4f})"
        return divergences, root_desc

    # Case 3: Anchor on earliest sampled cohort
    if root_taxon and root_taxon.lower() in ['earliest', 'earliest_taxon', 'earliest_cohort']:
        valid_dates = [(t, dates_map[t]) for t in dated_taxa if t in dates_map and not np.isnan(dates_map[t])]
        valid_dates.sort(key=lambda x: x[1])
        min_date = valid_dates[0][1]
        earliest_taxa = [t for t, d in valid_dates if abs(d - min_date) < 1e-4]

        if len(dated_taxa) > 2500 or len(earliest_taxa) <= 10:
            cross_mat = compute_tn93_cross_distance_matrix(seq_dict, dated_taxa, earliest_taxa)
            if len(earliest_taxa) == 1:
                divergences = cross_mat[:, 0].astype(np.float64)
                root_desc = f"earliest_taxon_{earliest_taxa[0]}"
            else:
                divergences = np.mean(cross_mat, axis=1).astype(np.float64)
                root_desc = f"earliest_cohort_n{len(earliest_taxa)}"
            return divergences, root_desc
        else:
            dist_mat = compute_tn93_distance_matrix(seq_dict, dated_taxa)
            taxa_idx = {t: i for i, t in enumerate(dated_taxa)}
            earliest_indices = [taxa_idx[t] for t in earliest_taxa]

            if len(earliest_indices) == 1:
                root_idx = earliest_indices[0]
                divergences = dist_mat[root_idx, :].copy()
                root_desc = f"earliest_taxon_{dated_taxa[root_idx]}"
            else:
                divergences = np.mean(dist_mat[earliest_indices, :], axis=0)
                root_desc = f"earliest_cohort_n{len(earliest_indices)}"

            return divergences, root_desc

    # Case 4 (Default & Recommended for Tree-Free): Continuous Time-Decay Soft Profile Root
    divergences, eff_gamma = compute_time_decay_profile_divergences(
        seq_dict, dates_map, dated_taxa, gamma=decay_gamma, half_life=decay_half_life
    )
    root_desc = f"time_decay_profile_root (γ={eff_gamma:.4f})"
    return divergences, root_desc


def optimize_latent_convex_hull_root(
    taxon_repr: np.ndarray,
    times: np.ndarray,
    taxa_names: Optional[List[str]] = None,
    pairwise_phys_dists: Optional[np.ndarray] = None,
    anchor_mask: Optional[np.ndarray] = None,
    learning_rate: float = 0.05,
    max_iter: int = 250,
    device: Optional[Union[str, torch.device]] = None
) -> Dict[str, Any]:
    """
    Optimizes a continuous ancestral root representation within the convex hull
    of observed sequence embeddings in latent representation space:

        z_root(v) = sum_{i=1}^N softmax(v)_i * z_i

    where v is optimized to maximize the temporal correlation with tip sampling dates.
    Pairwise physical distances (Hamming / TN93) are used to compute an isometric
    scaling factor alpha [subs/site per latent unit], yielding calibrated root-to-tip
    distances and standard evolutionary rates in substitutions / site / year.

    Returns:
        Dictionary containing:
        - 'z_root': (D,) optimal root representation
        - 'weights': (N,) convex hull weights
        - 'dists': (N,) calibrated root-to-tip distances in substitutions/site
        - 'dists_latent': (N,) Euclidean distances in latent space
        - 'alpha': isometric calibration scale factor
        - 'anchor_taxa': list of top contributing anchor taxa
        - 'anchor_mask': boolean mask of eligible anchor sequences
        - 'temporal_r': Pearson correlation R
        - 'temporal_r2': R^2
        - 'mu_ols': OLS rate in subs/site/yr
        - 't_mrca_ols': OLS t_MRCA
    """
    n_taxa = len(times)
    d_dim = taxon_repr.shape[1]

    # Eligible anchor taxa (e.g. non-holdout / sufficient coverage)
    if anchor_mask is None:
        eligible = np.ones(n_taxa, dtype=bool)
    else:
        eligible = np.asarray(anchor_mask, dtype=bool)
    if np.sum(eligible) < 3:
        eligible = np.ones(n_taxa, dtype=bool)

    # 1. Compute physical distances for isometric calibration
    if n_taxa > 2000:
        # Safety chunking: Subsample representative pairs to avoid O(N^2 * D) host RAM explosion
        rng_pairs = np.random.default_rng(42)
        m_sample = min(50000, n_taxa * (n_taxa - 1) // 2)
        idx_i = rng_pairs.integers(0, n_taxa - 1, size=m_sample)
        idx_j = rng_pairs.integers(idx_i + 1, n_taxa, size=m_sample)
        d_latent_sample = np.linalg.norm(taxon_repr[idx_i] - taxon_repr[idx_j], axis=1)
        if pairwise_phys_dists is not None:
            phys_sample = pairwise_phys_dists[idx_i, idx_j]
            denom = float(np.sum(d_latent_sample ** 2))
            alpha = float(np.sum(phys_sample * d_latent_sample) / denom) if denom > 1e-12 else 1.0
        else:
            mean_lat = float(np.mean(d_latent_sample)) if len(d_latent_sample) > 0 else 1.0
            alpha = 0.05 / max(1e-6, mean_lat)
    else:
        d_latent_pairs = pdist(taxon_repr, metric='euclidean')
        if pairwise_phys_dists is not None and n_taxa > 1:
            triu_i, triu_j = np.triu_indices(n_taxa, k=1)
            phys_upper = pairwise_phys_dists[triu_i, triu_j]
            denom = float(np.sum(d_latent_pairs ** 2))
            alpha = float(np.sum(phys_upper * d_latent_pairs) / denom) if denom > 1e-12 else 1.0
        else:
            mean_lat = float(np.mean(d_latent_pairs)) if len(d_latent_pairs) > 0 else 1.0
            alpha = 0.05 / max(1e-6, mean_lat)

    # 2. Continuous convex hull optimization
    import torch  # lazy: only needed for latent-space optimization
    dev = device if device is not None else ("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    Z_t = torch.tensor(taxon_repr, dtype=torch.float32, device=dev)
    times_t = torch.tensor(times, dtype=torch.float32, device=dev)

    min_t = float(torch.min(times_t[eligible]).item())
    span_t = max(1e-6, float(torch.max(times_t[eligible]).item()) - min_t)

    # Initialize logits: early dates get higher initial prior weight, ineligible get -1e4
    v_param = torch.full((n_taxa,), -1e4, dtype=torch.float32, device=dev)
    for i in range(n_taxa):
        if eligible[i]:
            v_param[i] = -0.5 * (float(times[i]) - min_t) / span_t
    v_param.requires_grad = True

    optimizer = torch.optim.Adam([v_param], lr=learning_rate)

    t_el = times_t[eligible]
    t_centered = t_el - torch.mean(t_el)
    std_t = torch.std(t_el) + 1e-8

    converged = False
    final_grad_norm = 0.0
    tol = 1e-5
    patience = 15
    no_improve = 0
    best_loss = 1e9

    for step in range(max_iter):
        optimizer.zero_grad()
        # Softmax over all taxa (ineligible have massive negative logit, so weight ~ 0)
        w = torch.softmax(v_param, dim=0)
        z_r = torch.sum(w[:, None] * Z_t, dim=0)
        d_lat = torch.norm(Z_t - z_r, dim=1)
        d_el = d_lat[eligible]
        d_centered = d_el - torch.mean(d_el)
        cov = torch.mean(t_centered * d_centered)
        corr = cov / (std_t * torch.std(d_el) + 1e-8)
        loss = -corr
        loss.backward()

        if v_param.grad is not None:
            final_grad_norm = float(torch.norm(v_param.grad).item())

        loss_val = float(loss.item())
        if best_loss - loss_val > tol:
            best_loss = loss_val
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                converged = True
                optimizer.step()
                break

        optimizer.step()

    w_opt = torch.softmax(v_param, dim=0).detach().cpu().numpy()
    z_root = np.sum(w_opt[:, None] * taxon_repr, axis=0)
    dists_latent = np.linalg.norm(taxon_repr - z_root, axis=1)
    dists_phys = alpha * dists_latent

    # OLS fit on eligible taxa
    times_el = times[eligible]
    dists_el = dists_phys[eligible]
    r_val = float(np.corrcoef(times_el, dists_el)[0, 1])
    slope_ols, inter_ols = np.polyfit(times_el, dists_el, 1)
    slope_ols, inter_ols = float(slope_ols), float(inter_ols)
    t_mrca_ols = float(-inter_ols / slope_ols) if slope_ols > 1e-6 else np.nan

    # Identify top anchor taxa
    anchor_indices = np.argsort(-w_opt)
    anchor_taxa = []
    for idx in anchor_indices:
        if w_opt[idx] < 0.01 and len(anchor_taxa) >= 3:
            break
        name = taxa_names[idx] if taxa_names and idx < len(taxa_names) else f"taxon_{idx}"
        anchor_taxa.append({
            "taxon": name,
            "weight": float(w_opt[idx]),
            "date": float(times[idx])
        })

    return {
        "z_root": z_root,
        "weights": w_opt,
        "dists": dists_phys,
        "dists_latent": dists_latent,
        "alpha": alpha,
        "anchor_taxa": anchor_taxa,
        "anchor_mask": eligible,
        "temporal_r": r_val,
        "temporal_r2": float(r_val ** 2),
        "mu_ols": slope_ols,
        "t_mrca_ols": t_mrca_ols,
        "converged": bool(converged or (step == max_iter - 1)),
        "final_grad_norm": float(final_grad_norm),
        "steps_taken": int(step + 1)
    }

