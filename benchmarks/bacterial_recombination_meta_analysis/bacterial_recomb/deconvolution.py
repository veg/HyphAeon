"""
deconvolution.py
================
Deconvolves the observed recombination landscape into:
1. The Biophysical Mechanistic Delivery Prior: lambda_mech(s)
2. The Evolutionary Selective Sieve: S_sel(s)

Decouples where foreign DNA physically enters and integrates from
which recombinant lineages survive and fix in bacterial populations.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class DeconvolutionResults:
    """Stores chromosome-wide deconvolution profiles."""
    coordinates: np.ndarray
    lambda_observed: np.ndarray
    lambda_mechanistic: np.ndarray
    selective_sieve: np.ndarray
    log_selective_sieve: np.ndarray
    hotspots: List[Dict[str, float]]
    deserts: List[Dict[str, float]]


def fit_mechanistic_prior(
    coordinates: np.ndarray,
    event_counts: np.ndarray,
    chi_density: np.ndarray,
    dist_to_ori: np.ndarray,
    is_tRNA_att: np.ndarray,
    abs_gc_skew: np.ndarray
) -> np.ndarray:
    """
    Fits a Poisson GLM / Log-linear model to estimate the biophysical
    integration rate lambda_mech(s) from physical features alone.
    
    ln lambda_mech(s) = beta_0 + beta_1 * Chi(s) + beta_2 * (1 - d_ori / d_max) + beta_3 * tRNA + beta_4 * Skew
    """
    M = len(coordinates)
    d_max = np.max(dist_to_ori) if np.max(dist_to_ori) > 0 else 1.0
    ori_proximity = 1.0 - (dist_to_ori / d_max)

    # Feature matrix (M, 5)
    X = np.column_stack([
        np.ones(M),
        chi_density,
        ori_proximity,
        is_tRNA_att.astype(float),
        abs_gc_skew
    ])

    # Regularized Poisson / Ridge pseudo-inverse fit for robust baseline
    y = np.maximum(0.01, event_counts.astype(float))
    log_y = np.log(y)

    # Ridge regression on log-rate: (X^T X + alpha I)^(-1) X^T log_y
    alpha = 1.0
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    beta = np.linalg.solve(XtX, X.T @ log_y)

    pred_log_lambda = X @ beta
    lambda_mech = np.exp(pred_log_lambda)

    # Scale lambda_mech to match total mean observed events
    scale = np.mean(y) / (np.mean(lambda_mech) + 1e-8)
    lambda_mech *= scale

    return lambda_mech


def compute_selective_sieve(
    coordinates: np.ndarray,
    lambda_observed: np.ndarray,
    lambda_mechanistic: np.ndarray,
    hotspot_threshold: float = 2.0,
    desert_threshold: float = -1.5,
    min_span_bp: int = 5000
) -> DeconvolutionResults:
    """
    Computes S_sel(s) = lambda_obs(s) / lambda_mech(s) and identifies
    statistically significant adaptive hotspots and purifying deserts.
    """
    eps = 1e-6
    s_sel = (lambda_observed + eps) / (lambda_mechanistic + eps)
    log_s_sel = np.log2(s_sel)

    # Identify Hotspots (log2 S_sel > threshold)
    hotspots: List[Dict[str, float]] = []
    is_hot = log_s_sel >= hotspot_threshold
    in_block = False
    start_c = 0

    for idx, (c, h) in enumerate(zip(coordinates, is_hot)):
        if h and not in_block:
            in_block = True
            start_c = c
            start_idx = idx
        elif not h and in_block:
            in_block = False
            if c - start_c >= min_span_bp:
                peak_val = float(np.max(log_s_sel[start_idx:idx]))
                mean_val = float(np.mean(log_s_sel[start_idx:idx]))
                hotspots.append({
                    "start": int(start_c),
                    "end": int(c),
                    "span_bp": int(c - start_c),
                    "peak_log2_sieve": peak_val,
                    "mean_log2_sieve": mean_val
                })

    # Identify Deserts (log2 S_sel < desert_threshold)
    deserts: List[Dict[str, float]] = []
    is_cold = log_s_sel <= desert_threshold
    in_cold_block = False
    cold_start_c = 0

    for idx, (c, cold) in enumerate(zip(coordinates, is_cold)):
        if cold and not in_cold_block:
            in_cold_block = True
            cold_start_c = c
            cold_start_idx = idx
        elif not cold and in_cold_block:
            in_cold_block = False
            if c - cold_start_c >= min_span_bp:
                min_val = float(np.min(log_s_sel[cold_start_idx:idx]))
                mean_val = float(np.mean(log_s_sel[cold_start_idx:idx]))
                deserts.append({
                    "start": int(cold_start_c),
                    "end": int(c),
                    "span_bp": int(c - cold_start_c),
                    "min_log2_sieve": min_val,
                    "mean_log2_sieve": mean_val
                })

    return DeconvolutionResults(
        coordinates=coordinates,
        lambda_observed=lambda_observed,
        lambda_mechanistic=lambda_mechanistic,
        selective_sieve=s_sel,
        log_selective_sieve=log_s_sel,
        hotspots=hotspots,
        deserts=deserts
    )
