"""
HyphAeon Geo: Ultra-Fast Discrete Phylogeography, Directed Spatial Migration Flux,
Permutation-Based BSSVS Bayes Factor Selection, and Spatial PGLS Epicenter Inference.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from Bio import Phylo, SeqIO

from .dataset import compute_tn93_distance_matrix, parse_alignment_sequences


# =========================================================================
# 1. Geographic Metadata Parsing & Alignment Reconciliation
# =========================================================================

def parse_geo_metadata(
    metadata_source: Union[str, Path, pd.DataFrame],
    taxa_list: List[str],
    location_col: Optional[str] = None,
    strain_col: Optional[str] = None,
    date_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Tuple[float, float]]]:
    """
    Parses metadata CSV, TSV, or Auspice JSON and aligns it with alignment sequence taxa.

    Returns:
        df_clean: DataFrame indexed by taxon with columns: location, date, lat, lon
        unique_locs: Sorted list of unique discrete location names (K)
        centroids: Dict mapping location name to (lat, lon) centroid (if coords provided)
    """
    if isinstance(metadata_source, pd.DataFrame):
        df_raw = metadata_source.copy()
    else:
        src_p = Path(metadata_source)
        if not src_p.exists():
            raise FileNotFoundError(f"Metadata file not found: {src_p}")

        # Handle Nextstrain Auspice JSON
        if src_p.suffix.lower() == ".json":
            with open(src_p, "r") as f:
                data = json.load(f)
            records = []

            def _traverse(node):
                name = node.get("name")
                attrs = node.get("node_attrs", {})
                loc_val = None
                date_val = None
                lat_val = None
                lon_val = None
                for k in ["location", "region", "country", "division"]:
                    if k in attrs and "value" in attrs[k]:
                        loc_val = attrs[k]["value"]
                        break
                if "num_date" in attrs and "value" in attrs["num_date"]:
                    date_val = float(attrs["num_date"]["value"])
                if name and loc_val:
                    records.append({
                        "taxon": name,
                        "location": loc_val,
                        "date": date_val,
                        "lat": lat_val,
                        "lon": lon_val,
                    })
                for child in node.get("children", []):
                    _traverse(child)

            if "tree" in data:
                _traverse(data["tree"])
            df_raw = pd.DataFrame(records)
        else:
            sep = "\t" if src_p.suffix.lower() in [".tsv", ".tab"] else ","
            try:
                df_raw = pd.read_csv(src_p, sep=sep)
            except Exception:
                df_raw = pd.read_csv(src_p, sep=None, engine="python")

    # Resolve strain / taxon column
    if strain_col and strain_col in df_raw.columns:
        s_col = strain_col
    else:
        candidates = ["taxon", "strain", "id", "name", "accession", "isolate", "Sequence_ID"]
        s_col = next((c for c in candidates if c in df_raw.columns), None)
        if s_col is None:
            s_col = df_raw.columns[0]

    # Resolve location column
    if location_col and location_col in df_raw.columns:
        l_col = location_col
    else:
        candidates = ["location", "region", "province", "state", "country", "division", "discrete", "geo"]
        l_col = next((c for c in candidates if c in df_raw.columns), None)
        if l_col is None:
            raise ValueError(
                f"Could not auto-detect location column in metadata. "
                f"Available columns: {list(df_raw.columns)}. Please specify --location-col."
            )

    # Resolve date column (optional)
    if date_col and date_col in df_raw.columns:
        d_col = date_col
    else:
        candidates = ["date", "time", "year", "collection_date", "num_date", "sampling_date"]
        d_col = next((c for c in candidates if c in df_raw.columns), None)

    # Resolve lat / lon coordinates (optional)
    if lat_col and lat_col in df_raw.columns:
        lt_col = lat_col
    else:
        candidates = ["lat", "latitude", "y", "coord_lat"]
        lt_col = next((c for c in candidates if c in df_raw.columns), None)

    if lon_col and lon_col in df_raw.columns:
        ln_col = lon_col
    else:
        candidates = ["lon", "long", "longitude", "x", "coord_lon"]
        ln_col = next((c for c in candidates if c in df_raw.columns), None)

    # Standardize column mapping
    df_raw[s_col] = df_raw[s_col].astype(str)
    df_raw = df_raw.drop_duplicates(subset=[s_col])
    meta_dict = df_raw.set_index(s_col).to_dict(orient="index")

    clean_records = []
    for t in taxa_list:
        if t in meta_dict:
            row = meta_dict[t]
            loc = str(row[l_col]).strip() if pd.notna(row.get(l_col)) else None
            date_val = float(row[d_col]) if d_col and pd.notna(row.get(d_col)) else np.nan
            lat_val = float(row[lt_col]) if lt_col and pd.notna(row.get(lt_col)) else np.nan
            lon_val = float(row[ln_col]) if ln_col and pd.notna(row.get(ln_col)) else np.nan
            if loc and loc.lower() not in ["none", "nan", "unknown", ""]:
                clean_records.append({
                    "taxon": t,
                    "location": loc,
                    "date": date_val,
                    "lat": lat_val,
                    "lon": lon_val,
                })

    df_clean = pd.DataFrame(clean_records)
    if len(df_clean) < 3:
        raise ValueError(
            f"Fewer than 3 taxa could be matched between alignment and metadata. "
            f"Alignment taxa: {len(taxa_list)}, Metadata matched: {len(df_clean)}."
        )

    unique_locs = sorted(df_clean["location"].unique())
    centroids: Dict[str, Tuple[float, float]] = {}
    if lt_col and ln_col and df_clean["lat"].notna().any():
        for loc in unique_locs:
            sub = df_clean[df_clean["location"] == loc]
            valid_lat = sub["lat"].dropna()
            valid_lon = sub["lon"].dropna()
            if len(valid_lat) > 0 and len(valid_lon) > 0:
                centroids[loc] = (float(valid_lat.mean()), float(valid_lon.mean()))

    return df_clean, unique_locs, centroids


# =========================================================================
# 2. Heuristic Tree Construction & Temporal Rooting
# =========================================================================

def build_or_load_tree(
    alignment_path: Union[str, Path],
    tree_path: Optional[Union[str, Path]] = None,
    root_taxon: Optional[str] = None,
    dates_map: Optional[Dict[str, float]] = None,
) -> Phylo.BaseTree.Tree:
    """
    Loads an existing tree or automatically constructs an ML tree via FastTree.
    Roots the tree using:
    1. Explicit root_taxon (if specified).
    2. Temporal rooting (earliest sampled isolate).
    3. Midpoint rooting.
    """
    if tree_path and Path(tree_path).exists():
        tree = Phylo.read(str(tree_path), "newick")
    else:
        # Check if FastTree is available
        fasttree_bin = None
        for cand in ["/usr/local/bin/FastTree", "/usr/bin/FastTree", "FastTree", "fasttree"]:
            try:
                subprocess.run([cand, "-help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                fasttree_bin = cand
                break
            except Exception:
                continue

        if fasttree_bin:
            tmp_tree = Path(alignment_path).with_suffix(".fasttree.nwk")
            cmd = f"{fasttree_bin} -nt -gtr {alignment_path} > {tmp_tree}"
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            tree = Phylo.read(str(tmp_tree), "newick")
            if tmp_tree.exists():
                tmp_tree.unlink()
        else:
            raise RuntimeError(
                "No phylogenetic tree provided (--tree) and FastTree is not installed on PATH. "
                "Please provide a Newick tree file via -t/--tree."
            )

    # Rooting strategy
    terminals = tree.get_terminals()
    term_names = {t.name for t in terminals}

    if root_taxon and root_taxon in term_names:
        target = next(t for t in terminals if t.name == root_taxon)
        tree.root_with_outgroup(target)
    elif dates_map:
        # Find earliest sampled tip
        valid_tips = [(t, dates_map[t.name]) for t in terminals if t.name in dates_map and not np.isnan(dates_map[t.name])]
        if valid_tips:
            valid_tips.sort(key=lambda x: x[1])
            earliest = valid_tips[0][0]
            tree.root_with_outgroup(earliest)
        else:
            tree.root_at_midpoint()
    else:
        tree.root_at_midpoint()

    return tree


# =========================================================================
# 3. Discrete Ancestral State Reconstruction & Historical Migration Events
# =========================================================================

def reconstruct_ancestral_states_parsimony(
    tree: Phylo.BaseTree.Tree,
    loc_map: Dict[str, str],
    unique_locs: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Fitch-Sankoff maximum parsimony ancestral state reconstruction
    of discrete geographic locations across the phylogeny.

    Returns:
        root_probs: Array [K] of posterior root state probabilities.
        transitions: Matrix [K, K] counting observed historical branch transitions (source -> target).
    """
    K = len(unique_locs)
    loc_to_idx = {l: i for i, l in enumerate(unique_locs)}

    # Bottom-up postorder traversal to assign state sets
    for clade in tree.find_clades(order="postorder"):
        if clade.is_terminal():
            l = loc_map.get(clade.name)
            clade.state_set = {loc_to_idx[l]} if l in loc_to_idx else set(range(K))
        else:
            child_sets = [c.state_set for c in clade.clades]
            common = set.intersection(*child_sets)
            clade.state_set = common if common else set.union(*child_sets)

    # Top-down preorder traversal to assign definitive states
    root_states = list(tree.root.state_set)
    tree.root.state = root_states[0]
    for clade in tree.find_clades(order="preorder"):
        for c in clade.clades:
            if clade.state in c.state_set:
                c.state = clade.state
            else:
                c.state = list(c.state_set)[0]

    # Root state distribution
    root_probs = np.zeros(K, dtype=np.float64)
    for s in root_states:
        root_probs[s] = 1.0 / len(root_states)

    # Count directed branch transitions
    transitions = np.zeros((K, K), dtype=np.float64)
    for clade in tree.find_clades():
        for c in clade.clades:
            if clade.state != c.state:
                transitions[clade.state, c.state] += 1.0

    return root_probs, transitions


# =========================================================================
# 4. Neural Cross-Taxa Attention Migration Flux
# =========================================================================

def compute_attention_migration_flux(
    attn_matrix: np.ndarray,
    loc_indices: np.ndarray,
    K: int,
) -> np.ndarray:
    """
    Aggregates directed inter-taxa attention weights across discrete geographic partitions:
        F_{jk} = sum_{i in Loc_j} sum_{l in Loc_k} A_{il}
        M_{jk} = F_{jk} / (N_j * N_k)

    Returns:
        M: Normalized directed migration flux matrix [K, K].
    """
    N = len(loc_indices)
    H = np.zeros((N, K), dtype=np.float64)
    for i in range(N):
        H[i, loc_indices[i]] = 1.0

    counts = H.sum(axis=0)
    denom = np.outer(counts, counts)

    # Inter-region flux
    F = H.T @ attn_matrix @ H
    M = np.zeros((K, K), dtype=np.float64)
    for j in range(K):
        for k in range(K):
            if j != k and denom[j, k] > 0:
                M[j, k] = F[j, k] / denom[j, k]
            elif j == k and counts[j] > 0:
                M[j, k] = F[j, k] / (counts[j] ** 2)

    return M


# =========================================================================
# 5. Fast Vectorized Permutation Bayes Factors (BSSVS Alternative)
# =========================================================================

def run_permutation_bssvs(
    tree: Optional[Phylo.BaseTree.Tree],
    attn_matrix: Optional[np.ndarray],
    taxa: List[str],
    loc_indices: np.ndarray,
    unique_locs: List[str],
    n_perms: int = 1000,
    min_bf: float = 3.0,
    fdr_thresh: float = 0.10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Vectorized non-parametric permutation test for discrete transmission routes.
    Computes empirical Bayes Factors, Z-scores, and Benjamini-Hochberg FDR q-values.

    Returns:
        M_obs: Observed migration rate matrix [K, K].
        Z_scores: Route standardized Z-scores [K, K].
        p_vals: Empirical p-values [K, K].
        bf_mat: Empirical Bayes Factors [K, K].
        significant_routes: List of dicts for supported transmission highways.
    """
    np.random.seed(seed)
    K = len(unique_locs)
    N = len(taxa)
    loc_to_idx = {l: i for i, l in enumerate(unique_locs)}

    # Prior inclusion probability under truncated Poisson / minimal graph prior:
    # q_prior = (K - 1) / (K * (K - 1) / 2) = 2 / K
    q_prior = 2.0 / K
    prior_odds = q_prior / (1.0 - q_prior)

    # Mode 1: Tree-Based Branch Transition Counting
    if tree is not None and attn_matrix is None:
        loc_map = {taxa[i]: unique_locs[loc_indices[i]] for i in range(N)}
        _, trans_obs = reconstruct_ancestral_states_parsimony(tree, loc_map, unique_locs)
        counts = np.bincount(loc_indices, minlength=K).astype(np.float64)
        denom = np.outer(counts, counts)

        M_obs = np.zeros((K, K), dtype=np.float64)
        for j in range(K):
            for k in range(K):
                if j != k and denom[j, k] > 0:
                    M_obs[j, k] = trans_obs[j, k] / np.sqrt(counts[j] * counts[k])

        # Permutations
        null_M = np.zeros((n_perms, K, K), dtype=np.float64)
        labels = np.array([unique_locs[loc_indices[i]] for i in range(N)])
        for b in range(n_perms):
            p_labels = np.random.permutation(labels)
            p_map = dict(zip(taxa, p_labels))
            _, t_null = reconstruct_ancestral_states_parsimony(tree, p_map, unique_locs)
            for j in range(K):
                for k in range(K):
                    if j != k and denom[j, k] > 0:
                        null_M[b, j, k] = t_null[j, k] / np.sqrt(counts[j] * counts[k])

    # Mode 2: Neural Cross-Taxa Attention Flux
    else:
        if attn_matrix is None:
            raise ValueError("Must provide either a tree or an attention matrix.")
        M_obs = compute_attention_migration_flux(attn_matrix, loc_indices, K)
        counts = np.bincount(loc_indices, minlength=K).astype(np.float64)
        denom = np.outer(counts, counts)

        H = np.zeros((N, K), dtype=np.float64)
        for i in range(N):
            H[i, loc_indices[i]] = 1.0

        null_M = np.zeros((n_perms, K, K), dtype=np.float64)
        for b in range(n_perms):
            p_idx = np.random.permutation(N)
            H_perm = H[p_idx, :]
            F_perm = H_perm.T @ attn_matrix @ H_perm
            for j in range(K):
                for k in range(K):
                    if j != k and denom[j, k] > 0:
                        null_M[b, j, k] = F_perm[j, k] / denom[j, k]

    # Compute Z-scores, p-values, and Bayes Factors
    mu_null = np.mean(null_M, axis=0)
    std_null = np.std(null_M, axis=0) + 1e-8
    Z_scores = (M_obs - mu_null) / std_null

    p_vals = np.ones((K, K), dtype=np.float64)
    bf_mat = np.zeros((K, K), dtype=np.float64)

    all_routes = []
    for j in range(K):
        for k in range(K):
            if j != k:
                # Empirical one-tailed p-value
                p = (np.sum(null_M[:, j, k] >= M_obs[j, k]) + 1.0) / (n_perms + 1.0)
                p_vals[j, k] = p

                # Posterior inclusion odds & Bayes Factor
                tail_p = np.sum(null_M[:, j, k] < M_obs[j, k]) / n_perms
                post_prob = min(max(tail_p, 0.01), 0.99)
                post_odds = post_prob / (1.0 - post_prob)
                bf = post_odds / prior_odds
                bf_mat[j, k] = bf

                all_routes.append({
                    "source": unique_locs[j],
                    "target": unique_locs[k],
                    "source_idx": j,
                    "target_idx": k,
                    "rate": float(M_obs[j, k]),
                    "z_score": float(Z_scores[j, k]),
                    "p_value": float(p),
                    "bayes_factor": float(bf),
                })

    # Benjamini-Hochberg FDR correction
    all_routes.sort(key=lambda x: x["p_value"])
    m_tests = len(all_routes)
    for rank, r in enumerate(all_routes, 1):
        q = min(r["p_value"] * (m_tests / rank), 1.0)
        r["fdr_q"] = float(q)

    # Monotonic adjustment of FDR q-values (from back to front)
    for i in range(m_tests - 2, -1, -1):
        all_routes[i]["fdr_q"] = min(all_routes[i]["fdr_q"], all_routes[i + 1]["fdr_q"])

    # Categorize support
    for r in all_routes:
        bf = r["bayes_factor"]
        if bf >= 100.0:
            r["support"] = "Decisive (BF >= 100)"
        elif bf >= 10.0:
            r["support"] = "Strong (10 <= BF < 100)"
        elif bf >= 3.0:
            r["support"] = "Substantial (3 <= BF < 10)"
        else:
            r["support"] = "Unsupported (BF < 3)"

    all_routes.sort(key=lambda x: -x["z_score"])
    return M_obs, Z_scores, p_vals, bf_mat, all_routes


# =========================================================================
# 6. Spatial PGLS Epicenter Estimation (Continuous Coordinates)
# =========================================================================

def estimate_spatial_pgls_epicenter(
    coords: np.ndarray,
    cov_matrix: np.ndarray,
    ridge: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes analytical Generalized Least Squares (PGLS) spatial epicenter
    for continuous geographic coordinates (lat, lon) on the phylogenetic manifold:
        y_root = (1^T C^{-1} Y) / (1^T C^{-1} 1)

    Returns:
        epicenter_coords: Array [2] of [lat, lon]
        cov_ellipse: Covariance matrix [2, 2]
        confidence_radius_km: 95% confidence radius in kilometers
    """
    N = len(coords)
    C = cov_matrix + ridge * np.eye(N)
    C_inv = np.linalg.inv(C)
    ones = np.ones(N)

    # PGLS weights
    w = (C_inv @ ones) / (ones @ C_inv @ ones)

    # Inferred root coordinates
    root_xy = w @ coords

    # Residual variance & covariance
    residuals = coords - root_xy
    sigma2 = np.trace(residuals.T @ C_inv @ residuals) / (2 * N - 2)
    cov_xy = (sigma2 / (ones @ C_inv @ ones)) * np.eye(2)

    # 95% radius in km (Chi-square 2 DF critical value = 5.991)
    std_deg = math.sqrt(sigma2 / (ones @ C_inv @ ones))
    radius_km = std_deg * 111.32 * math.sqrt(5.991)

    return root_xy, cov_xy, float(radius_km)


# =========================================================================
# 7. GeoJSON & GIS Serialization
# =========================================================================

def generate_geojson(
    unique_locs: List[str],
    centroids: Dict[str, Tuple[float, float]],
    root_probs: np.ndarray,
    counts: np.ndarray,
    significant_routes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Constructs a GeoJSON FeatureCollection containing:
    1. Point features for discrete location centroids (with root prob and sample count).
    2. LineString features for significant transmission arcs.
    """
    features = []

    # 1. Centroid points
    for idx, loc in enumerate(unique_locs):
        if loc in centroids:
            lat, lon = centroids[loc]
            p_root = float(root_probs[idx])
            n_samples = int(counts[idx])
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],  # GeoJSON standard: [lon, lat]
                },
                "properties": {
                    "name": loc,
                    "sample_count": n_samples,
                    "root_probability": round(p_root, 4),
                    "is_epicenter": bool(p_root == np.max(root_probs)),
                },
            })

    # 2. Transmission arcs
    for r in significant_routes:
        src = r["source"]
        dst = r["target"]
        if src in centroids and dst in centroids:
            lat1, lon1 = centroids[src]
            lat2, lon2 = centroids[dst]
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon1, lat1], [lon2, lat2]],
                },
                "properties": {
                    "source": src,
                    "target": dst,
                    "migration_rate": round(r["rate"], 5),
                    "z_score": round(r["z_score"], 2),
                    "p_value": round(r["p_value"], 4),
                    "fdr_q": round(r["fdr_q"], 4),
                    "bayes_factor": round(r["bayes_factor"], 1),
                    "support": r["support"],
                },
            })

    return {
        "type": "FeatureCollection",
        "generator": "HyphAeon Geo 1.0",
        "features": features,
    }


# =========================================================================
# 8. Publication Diagnostic Plotting
# =========================================================================

def plot_geo_diagnostics(
    res: Dict[str, Any],
    output_path: Union[str, Path],
    title: Optional[str] = None,
) -> None:
    """
    Generates a 3-panel publication figure:
      (A) Epicenter Root Probability Distribution
      (B) Directed Migration Flux / Rate Matrix (Naturally Asymmetric)
      (C) Route Significance Volcano Plot (Z-score vs log10 BF)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    ax1, ax2, ax3 = axes

    unique_locs = res["unique_locations"]
    root_probs = np.array(res["root_probabilities"])
    M = np.array(res["migration_matrix"])
    routes = res["routes"]
    K = len(unique_locs)

    # Panel A: Root Probabilities
    sort_idx = np.argsort(root_probs)
    sorted_locs = [unique_locs[i] for i in sort_idx]
    sorted_p = root_probs[sort_idx]
    colors = ["#1f77b4" if p < np.max(sorted_p) else "#d62728" for p in sorted_p]

    ax1.barh(range(K), sorted_p, color=colors, edgecolor="black", linewidth=0.8)
    ax1.set_yticks(range(K))
    ax1.set_yticklabels(sorted_locs, fontsize=9.5)
    ax1.set_xlabel("Posterior Probability P(Root)", fontsize=10.5, fontweight="bold")
    ax1.set_title("(A) Ancestral Epicenter Origin", fontsize=11, fontweight="bold", loc="left")
    ax1.set_xlim(0, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="x")
    for i, p in enumerate(sorted_p):
        if p > 0.01:
            ax1.text(p + 0.02, i, f"{p:.3f}", va="center", fontsize=9, fontweight="bold")

    # Panel B: Asymmetric Migration Flux Matrix
    im = ax2.imshow(M, cmap="YlOrRd", interpolation="nearest")
    ax2.set_xticks(range(K))
    ax2.set_yticks(range(K))
    ax2.set_xticklabels(unique_locs, rotation=45, ha="right", fontsize=9)
    ax2.set_yticklabels(unique_locs, fontsize=9)
    ax2.set_xlabel("Target Region (Sink)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Source Region", fontsize=10.5, fontweight="bold")
    ax2.set_title("(B) Directed Migration Flux Matrix", fontsize=11, fontweight="bold", loc="left")
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Transmission Flux", fontsize=9)

    # Panel C: Volcano Plot (Z-score vs log10 BF)
    z_scores = [r["z_score"] for r in routes]
    bfs = [r["bayes_factor"] for r in routes]
    log_bfs = [math.log10(max(b, 0.01)) for b in bfs]
    sig_colors = ["#d62728" if b >= 10 else ("#ff7f0e" if b >= 3 else "#7f7f7f") for b in bfs]

    ax3.scatter(z_scores, log_bfs, c=sig_colors, s=50, edgecolors="k", linewidth=0.6, alpha=0.85)
    ax3.axhline(math.log10(3.0), color="black", linestyle="--", linewidth=0.9, label="BF = 3 (Substantial)")
    ax3.axhline(math.log10(10.0), color="#d62728", linestyle=":", linewidth=1.1, label="BF = 10 (Strong)")
    ax3.axvline(2.0, color="gray", linestyle=":", linewidth=0.8)

    # Annotate top routes
    for r in routes[:5]:
        if r["bayes_factor"] >= 3.0:
            ax3.annotate(
                f"{r['source']}→{r['target']}",
                (r["z_score"], math.log10(r["bayes_factor"])),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
            )

    ax3.set_xlabel("Permutation Z-score", fontsize=10.5, fontweight="bold")
    ax3.set_ylabel("log10(Bayes Factor)", fontsize=10.5, fontweight="bold")
    ax3.set_title("(C) Transmission Route Significance", fontsize=11, fontweight="bold", loc="left")
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="lower right", fontsize=8.5, frameon=True)

    plt.suptitle(title or "HyphAeon Phylogeography & Spatial Transmission Architecture", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=300)
    if out_p.suffix.lower() != ".png":
        plt.savefig(out_p.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"[✓] Phylogeography diagnostic plot generated: {out_p}")


# =========================================================================
# 9. Master Phylogeography Pipeline
# =========================================================================

def run_phylogeography_analysis(
    alignment_path: Union[str, Path],
    metadata_path: Union[str, Path],
    tree_path: Optional[Union[str, Path]] = None,
    location_col: Optional[str] = None,
    strain_col: Optional[str] = None,
    date_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    root_taxon: Optional[str] = None,
    n_perms: int = 1000,
    min_bf: float = 3.0,
    fdr_thresh: float = 0.10,
    use_neural: bool = False,
    weights: Optional[str] = None,
    variant: Optional[str] = None,
    device: Optional[Any] = None,
    output_prefix: Optional[str] = None,
    geojson_path: Optional[str] = None,
    plot: bool = False,
    plot_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes end-to-end discrete phylogeographic inference, directed migration flux estimation,
    permutation-based BSSVS Bayes Factors, and spatial PGLS epicenter calculation.
    """
    t0 = time.time()
    align_p = Path(alignment_path)
    if not align_p.exists():
        raise FileNotFoundError(f"Alignment file not found: {align_p}")

    # 1. Parse alignment
    seq_dict = parse_alignment_sequences(str(align_p))
    taxa_all = list(seq_dict.keys())
    print(f"[*] Alignment parsed: {len(taxa_all)} taxa, sequence length {len(seq_dict[taxa_all[0]])} nt.")

    # 2. Parse metadata and reconcile
    df_meta, unique_locs, centroids = parse_geo_metadata(
        metadata_source=metadata_path,
        taxa_list=taxa_all,
        location_col=location_col,
        strain_col=strain_col,
        date_col=date_col,
        lat_col=lat_col,
        lon_col=lon_col,
    )
    taxa_matched = df_meta["taxon"].tolist()
    loc_map = dict(zip(df_meta["taxon"], df_meta["location"]))
    dates_map = dict(zip(df_meta["taxon"], df_meta["date"]))
    loc_to_idx = {l: i for i, l in enumerate(unique_locs)}
    loc_indices = np.array([loc_to_idx[df_meta.loc[df_meta["taxon"] == t, "location"].values[0]] for t in taxa_matched])
    K = len(unique_locs)
    N = len(taxa_matched)
    counts = np.bincount(loc_indices, minlength=K)

    print(f"[*] Geographic metadata mapped: {N} taxa across {K} discrete regions.")
    for loc, cnt in zip(unique_locs, counts):
        print(f"    - {loc:<14}: {cnt:>3} isolates")

    # 3. Load or build tree
    tree = None
    attn_mat = None

    if not use_neural:
        print(f"[*] Constructing / loading phylogenetic backbone...")
        tree = build_or_load_tree(
            alignment_path=alignment_path,
            tree_path=tree_path,
            root_taxon=root_taxon,
            dates_map=dates_map,
        )
        # Filter terminals to matched taxa
        term_names = {t.name for t in tree.get_terminals()}
        taxa_clean = [t for t in taxa_matched if t in term_names]
        loc_indices = np.array([loc_to_idx[loc_map[t]] for t in taxa_clean])
        taxa_matched = taxa_clean
        N = len(taxa_matched)
        counts = np.bincount(loc_indices, minlength=K)

        # Ancestral state reconstruction & branch transition events
        root_probs, trans_obs = reconstruct_ancestral_states_parsimony(tree, loc_map, unique_locs)
    else:
        # Neural attention mode
        from .inference import get_device, load_model, prepare_alignment
        from .splits import extract_cross_taxa_attentions_and_embeddings

        if device is None:
            device = get_device()
        print(f"[*] Running HyphAeon transformer backbone on {device} to extract cross-taxa attention...")
        model = load_model(weights=weights, variant=variant, device=device)
        c, a, d, z, inv, aln_taxa, L, tree_cache = prepare_alignment(
            str(align_p),
            str(tree_path) if tree_path else None,
            model=model,
            device=device,
            prune_duplicates=False,
            use_tn93=(tree_path is None),
        )
        import torch
        try:
            msa_codons = c.to(device)
            msa_aas = a.to(device)
            cross_attn, _ = extract_cross_taxa_attentions_and_embeddings(model, msa_codons, msa_aas, tree_cache, device=device)
        except Exception:
            cpu_device = torch.device("cpu")
            model = model.to(cpu_device)
            msa_codons = c.to(cpu_device)
            msa_aas = a.to(cpu_device)
            cross_attn, _ = extract_cross_taxa_attentions_and_embeddings(model, msa_codons, msa_aas, tree_cache, device=cpu_device)

        attn_mat = cross_attn.detach().cpu().numpy() if hasattr(cross_attn, "numpy") else np.asarray(cross_attn)
        np.fill_diagonal(attn_mat, 0.0)

        # PGLS root state probabilities from attention
        dist_mat = compute_tn93_distance_matrix(seq_dict, taxa_matched)
        scale = 2.0 * np.median(dist_mat[dist_mat > 0]) if np.any(dist_mat > 0) else 1.0
        C = np.exp(-dist_mat / scale) + 0.05 * np.eye(N)
        C_inv = np.linalg.inv(C)
        ones = np.ones(N)
        w = (C_inv @ ones) / (ones @ C_inv @ ones)

        H = np.zeros((N, K), dtype=np.float64)
        for i in range(N):
            H[i, loc_indices[i]] = 1.0
        p_raw = H.T @ w
        p_clean = np.maximum(p_raw, 0)
        root_probs = p_clean / p_clean.sum()

    # 4. Permutation-based BSSVS & Bayes Factor estimation
    print(f"[*] Running vectorized permutation testing ({n_perms} iterations) for transmission routes...")
    M_obs, Z_scores, p_vals, bf_mat, all_routes = run_permutation_bssvs(
        tree=tree,
        attn_matrix=attn_mat,
        taxa=taxa_matched,
        loc_indices=loc_indices,
        unique_locs=unique_locs,
        n_perms=n_perms,
        min_bf=min_bf,
        fdr_thresh=fdr_thresh,
    )

    sig_routes = [r for r in all_routes if r["bayes_factor"] >= min_bf or r["fdr_q"] <= fdr_thresh]

    # 5. Spatial PGLS Continuous Coordinates (if lat/lon available)
    spatial_coords = None
    has_coords = df_meta["lat"].notna().any() and df_meta["lon"].notna().any()
    if has_coords:
        coords_arr = df_meta[["lat", "lon"]].values
        dist_mat = compute_tn93_distance_matrix(seq_dict, taxa_matched)
        scale = 2.0 * np.median(dist_mat[dist_mat > 0]) if np.any(dist_mat > 0) else 1.0
        C = np.exp(-dist_mat / scale) + 0.05 * np.eye(N)
        epicenter_xy, cov_xy, radius_km = estimate_spatial_pgls_epicenter(coords_arr, C)
        spatial_coords = {
            "latitude": float(epicenter_xy[0]),
            "longitude": float(epicenter_xy[1]),
            "covariance_matrix": cov_xy.tolist(),
            "confidence_radius_km": float(radius_km),
        }
        print(f"[✓] Spatial PGLS Epicenter: {epicenter_xy[0]:.4f}°N, {epicenter_xy[1]:.4f}°E (±{radius_km:.1f} km)")

    # 6. Source-Sink Hub Metrics
    hub_metrics = {}
    for j, loc in enumerate(unique_locs):
        out_flux = float(np.sum(M_obs[j, :]) - M_obs[j, j])
        in_flux = float(np.sum(M_obs[:, j]) - M_obs[j, j])
        net_flux = out_flux - in_flux
        n_export = sum(1 for r in sig_routes if r["source"] == loc)
        n_import = sum(1 for r in sig_routes if r["target"] == loc)
        hub_metrics[loc] = {
            "out_flux": out_flux,
            "in_flux": in_flux,
            "net_flux": net_flux,
            "significant_exports": n_export,
            "significant_imports": n_import,
            "role": "Source / Exporter" if net_flux > 0 else "Sink / Importer",
        }

    top_epicenter_idx = int(np.argmax(root_probs))
    epicenter_region = unique_locs[top_epicenter_idx]
    epicenter_prob = float(root_probs[top_epicenter_idx])

    elapsed = time.time() - t0
    print(f"[✓] Phylogeography completed in {elapsed:.2f} s: Inferred Epicenter = {epicenter_region} (P = {epicenter_prob:.3f}).")

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "alignment": str(align_p),
        "metadata": str(metadata_path),
        "n_taxa": N,
        "n_regions": K,
        "unique_locations": unique_locs,
        "sample_counts": counts.tolist(),
        "root_epicenter": {
            "region": epicenter_region,
            "probability": epicenter_prob,
            "distribution": {loc: float(root_probs[i]) for i, loc in enumerate(unique_locs)},
        },
        "continuous_epicenter": spatial_coords,
        "root_probabilities": root_probs.tolist(),
        "migration_matrix": M_obs.tolist(),
        "z_scores": Z_scores.tolist(),
        "bayes_factors": bf_mat.tolist(),
        "routes": all_routes,
        "significant_routes": sig_routes,
        "hub_metrics": hub_metrics,
        "elapsed_seconds": float(elapsed),
    }

    # Export JSON
    if output_prefix:
        out_json = Path(output_prefix)
        if out_json.suffix.lower() != ".json":
            out_json = out_json.with_suffix(".json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[✓] Results written to: {out_json}")

    # Export GeoJSON
    if geojson_path:
        geo_json_p = Path(geojson_path)
        geo_dict = generate_geojson(unique_locs, centroids, root_probs, counts, sig_routes)
        geo_json_p.parent.mkdir(parents=True, exist_ok=True)
        with open(geo_json_p, "w") as f:
            json.dump(geo_dict, f, indent=2)
        print(f"[✓] GeoJSON exported to: {geo_json_p}")

    # Generate Diagnostic Plot
    if plot or plot_path:
        fig_path = plot_path or (Path(output_prefix).with_suffix(".pdf") if output_prefix else "phylogeography_diagnostics.pdf")
        plot_geo_diagnostics(results, fig_path)

    return results
