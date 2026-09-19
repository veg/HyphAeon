"""
chronaeon/dating.py
-------------------
Heterochronous Molecular Clock Calibration and Ancestor Dating (t_MRCA)
for Pathogen Genomics.

Methods:
1. Strict in-frame coding alignment validation (L_nt % 3 == 0, triplet-gap check, stop codon audit).
2. Flexible timestamp ingestion (FASTA headers, CSV/TSV metadata, Nextstrain Auspice JSON v2).
3. Root-to-tip divergence computation:
   - Tree-based: Patristic distance traversal with heuristic root search (TempEst R^2 maximization).
   - Tree-free: Direct pairwise distance estimation (TN93) and ancestral consensus anchoring.
4. Estimators:
   - Centered Root-to-Tip Ordinary Least Squares (OLS / TempEst emulation with delta-method & bootstrap CIs).
   - HyphAeon Attention-Derived Phylogenetic Generalized Least Squares (PGLS) via A_fused covariance.
   - Non-Linear Restricted Cubic Spline Clock (2 DF) and Power-Law Clock with hypothesis testing.
5. Historical outlier scoring & blind tip dating (e.g. dating the 1959 ZR59 archival isolate).
6. Publication-grade diagnostic visualization (PDF and PNG).
"""

import os
import sys
import json
import time
import math
import re
import datetime
import copy
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.stats as stats
import scipy.optimize as optimize
import torch

try:
    from Bio import Phylo
    HAS_BIOPHYLO = True
except ImportError:
    HAS_BIOPHYLO = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from aeon_core.dataset import (
    parse_alignment_sequences,
    compute_tn93_distance_matrix,
    compute_tn93_cross_distance_matrix,
    load_alignment_and_tree,
    parse_beast_xml,
    GENETIC_CODE,
    CODON_TO_AA,
)
from aeon_core.inference import (
    load_model,
    get_device,
    prepare_alignment,
)
from scipy.spatial.distance import pdist, squareform
from aeon_core.splits import (
    extract_cross_taxa_attentions_and_embeddings,
    compute_fused_affinity_matrix,
)
from aeon_core.temporal import parse_date_to_decimal, parse_dates_from_auspice_json, extract_date_from_string
from aeon_core.io import ensure_parent_directory, write_json, write_csv



def compute_neural_covariance_kernel(
    cross_attn: np.ndarray,
    taxa_repr: Optional[np.ndarray] = None,
    mds_coords: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Computes a strictly positive semi-definite (PSD) phylogenetic correlation matrix
    fusing transformer cross-taxa attention profile similarity and 128D continuous sequence representations.
    Features are centered across taxa to eliminate representation anisotropy (the cone effect),
    ensuring proper phylogenetic decoupling between distinct clades and realistic confidence intervals.
    Guarantees unit diagonal C(i, i) = 1.0 and zero negative eigenvalues.
    """
    n = cross_attn.shape[0]

    # 1. Cross-Taxa Attention Profile Correlation Matrix (Centered)
    # Each row a_i is taxon i's attention distribution across all taxa.
    # Centering removes the background entropy baseline across taxa:
    a_centered = cross_attn - np.mean(cross_attn, axis=0, keepdims=True)
    a_cov = a_centered @ a_centered.T
    a_var = np.diag(a_cov)
    a_std = np.sqrt(np.maximum(a_var, 1e-12))[:, None]
    a_denom = a_std @ a_std.T
    K_attn = np.divide(a_cov, a_denom, where=(a_denom > 1e-12), out=np.eye(n))
    np.fill_diagonal(K_attn, 1.0)

    # 2. Continuous Latent Sequence Embedding Correlation Matrix (Centered)
    if taxa_repr is not None and len(taxa_repr) == n:
        # Centering removes the dominant common activation vector across taxa:
        z_centered = taxa_repr - np.mean(taxa_repr, axis=0, keepdims=True)
        z_cov = z_centered @ z_centered.T
        z_var = np.diag(z_cov)
        z_std = np.sqrt(np.maximum(z_var, 1e-12))[:, None]
        z_denom = z_std @ z_std.T
        K_emb = np.divide(z_cov, z_denom, where=(z_denom > 1e-12), out=np.eye(n))
        np.fill_diagonal(K_emb, 1.0)
        K_neural = 0.50 * K_attn + 0.50 * K_emb
    else:
        K_neural = K_attn

    np.fill_diagonal(K_neural, 1.0)
    return K_neural


def compute_attention_covariance_kernel(
    cross_attn: np.ndarray,
    mds_coords: Optional[np.ndarray] = None,
    taxa_repr: Optional[np.ndarray] = None
) -> np.ndarray:
    """Backward-compatible alias for compute_neural_covariance_kernel."""
    return compute_neural_covariance_kernel(cross_attn, taxa_repr=taxa_repr, mds_coords=mds_coords)


def compute_transformer_metricity_diagnostics(
    d_matrix: np.ndarray,
    taxa_repr: Optional[np.ndarray] = None,
    cross_attn: Optional[np.ndarray] = None,
    coverage: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Evaluates foundation model metricity diagnostics to adjudicate between:
      Regime 1: Physical Sequence Space (TN93 distance geometry with time-decay consensus root)
      Regime 2: Continuous Latent Representation Space (Transformer embeddings with convex hull root)

    Parameters
    ----------
    d_matrix : np.ndarray
        N x N physical pairwise distance matrix (TN93 or patristic).
    taxa_repr : Optional[np.ndarray]
        N x D continuous sequence representations from axial transformer forward pass.
    cross_attn : Optional[np.ndarray]
        N x N pairwise cross-taxa attention matrix.
    coverage : Optional[np.ndarray]
        Fraction of non-gap coding positions per taxon (to filter fragmentary records).

    Returns
    -------
    dict with:
        d_mean, d_90, d_max, zero_fraction, rho_neg,
        rho_iso_pearson, rho_iso_spearman, kappa_sat,
        recommended_regime ("tn93" or "latent"),
        regime_label, rationale
    """
    N = d_matrix.shape[0]
    if coverage is not None:
        cov_arr = np.asarray(coverage)
        sub_idx = np.where(cov_arr >= 0.50)[0]
        if len(sub_idx) < 3:
            sub_idx = np.arange(N)
    else:
        sub_idx = np.arange(N)

    N_sub = len(sub_idx)
    D = d_matrix[np.ix_(sub_idx, sub_idx)]
    triu = np.triu_indices(N_sub, k=1)
    d_pairs = D[triu]

    # Check for NaN, Inf, or invalid negative distances (mathematical saturation in TN93)
    nan_mask = np.isnan(d_pairs) | np.isinf(d_pairs) | (d_pairs < 0)
    nan_frac = float(np.mean(nan_mask)) if len(d_pairs) > 0 else 0.0

    valid_pairs = d_pairs[~nan_mask]
    d_mean = float(np.mean(valid_pairs)) if len(valid_pairs) > 0 else 0.0
    d_90 = float(np.percentile(valid_pairs, 90)) if len(valid_pairs) > 0 else 0.0
    d_max = float(np.max(valid_pairs)) if len(valid_pairs) > 0 else 0.0
    zero_frac = float(np.mean(valid_pairs == 0.0)) if len(valid_pairs) > 0 else 0.0

    # Spectral Metricity: Classical MDS negative eigenvalue energy
    if N_sub > 1200:
        # Safety chunking: Subsample representative taxa for MDS eigenvalue calculation
        rng_diag = np.random.default_rng(42)
        diag_idx = np.sort(rng_diag.choice(N_sub, size=1000, replace=False))
        D_diag = D[np.ix_(diag_idx, diag_idx)]
        D_clean = D_diag.copy()
        nan_diag = np.isnan(D_clean) | np.isinf(D_clean) | (D_clean < 0)
        if np.any(nan_diag):
            D_clean[nan_diag] = max(0.50, d_max * 1.5)
        np.fill_diagonal(D_clean, 0.0)
    else:
        D_clean = D.copy()
        if nan_frac > 0:
            D_clean[np.isnan(D_clean) | np.isinf(D_clean) | (D_clean < 0)] = max(0.50, d_max * 1.5)
        np.fill_diagonal(D_clean, 0.0)

    # In-place double-centering via vector broadcasting (eliminates dense N x N centering matrices)
    D_sq = D_clean ** 2
    row_m = np.mean(D_sq, axis=1, keepdims=True)
    col_m = np.mean(D_sq, axis=0, keepdims=True)
    grand_m = np.mean(D_sq)
    B = -0.5 * (D_sq - row_m - col_m + grand_m)
    eigvals = np.linalg.eigvalsh(B)
    sum_abs = np.sum(np.abs(eigvals))
    sum_neg = np.sum(np.abs(eigvals[eigvals < 0]))
    rho_neg = float(sum_neg / (sum_abs + 1e-12))

    r_pearson, r_spearman, kappa_sat = None, None, None
    if taxa_repr is not None and len(taxa_repr) == N:
        Z = np.asarray(taxa_repr)[sub_idx]
        d_lat_pairs = pdist(Z, metric='euclidean')
        valid_both = (~nan_mask) & (d_pairs > 0)
        if np.sum(valid_both) > 10:
            r_pearson = float(stats.pearsonr(d_pairs[valid_both], d_lat_pairs[valid_both])[0])
            r_spearman = float(stats.spearmanr(d_pairs[valid_both], d_lat_pairs[valid_both])[0])
        elif len(valid_pairs) > 2:
            r_pearson = float(stats.pearsonr(d_pairs[~nan_mask], d_lat_pairs[~nan_mask])[0])
            r_spearman = float(stats.spearmanr(d_pairs[~nan_mask], d_lat_pairs[~nan_mask])[0])

        if np.sum(valid_both) > 3:
            X_poly = np.column_stack([d_pairs[valid_both], d_pairs[valid_both]**2])
            beta, _, _, _ = np.linalg.lstsq(X_poly, d_lat_pairs[valid_both], rcond=None)
            kappa_sat = float(beta[1] / (abs(beta[0]) + 1e-12))

    # 1. Physical Failure Condition: Mutational saturation breakdown or severe non-metric distortion
    is_phys_broken = (nan_frac > 0.05) or (rho_neg > 0.35)

    if is_phys_broken:
        regime = "latent"
        label = "Regime 2: Foundation Transformer Latent Space (Saturation Breakdown Recovery)"
        rationale = f"Physical distance geometry broken (nan_frac = {nan_frac:.1%}, spectral distortion rho_neg = {rho_neg:.3f}); continuous latent manifold recommended."
    # 2. Outbreak / Low Divergence: Exact physical distance geometry
    elif d_90 < 0.05:
        regime = "tn93"
        label = "Regime 1: Physical Sequence Space (Outbreak / Low Divergence)"
        rationale = f"Outbreak regime (d_90 = {d_90:.4f} < 0.05 subs/site); multiple substitutions negligible, TN93 distance geometry exact."
    # 3. High Isometric Concordance: Foundation model agrees with physical substitutions
    elif (r_spearman is not None and r_spearman >= 0.70 and r_pearson is not None and r_pearson >= 0.70):
        if d_90 >= 0.25 and (kappa_sat is not None and abs(kappa_sat) > 0.10 and rho_neg > 0.20):
            regime = "latent"
            label = "Regime 2: Foundation Transformer Latent Space (Deep Concordant Manifold)"
            rationale = f"Deep divergence (d_90 = {d_90:.4f}) with high isometric concordance (Spearman rho = {r_spearman:.3f}) and saturation curvature; continuous latent manifold recommended."
        else:
            regime = "tn93"
            label = "Regime 1: Physical Sequence Space (Isometric Concordance Confirmed)"
            rationale = f"Physical divergence d_90 = {d_90:.4f}; high isometric concordance (Spearman rho = {r_spearman:.3f}, Pearson r = {r_pearson:.3f}) confirms TN93 metricity."
    # 4. Non-Isometric Latent Space (or unobserved embeddings): Physical substitutions must be preserved
    else:
        regime = "tn93"
        label = "Regime 1: Physical Sequence Space (Linear Evolutionary Drift)"
        sp_str = f"Spearman rho = {r_spearman:.3f}" if r_spearman is not None else "no latent embeddings"
        rationale = f"Physical divergence d_90 = {d_90:.4f} with well-conditioned pairwise distances (nan_frac = {nan_frac:.1%}, rho_neg = {rho_neg:.3f}); latent space non-isometric ({sp_str} < 0.70), preserving physical substitutions."

    return {
        "d_mean": d_mean,
        "d_90": d_90,
        "d_max": d_max,
        "zero_fraction": zero_frac,
        "nan_fraction": nan_frac,
        "rho_neg": rho_neg,
        "rho_iso_pearson": r_pearson,
        "rho_iso_spearman": r_spearman,
        "kappa_sat": kappa_sat,
        "recommended_regime": regime,
        "regime_label": label,
        "rationale": rationale,
    }


