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



from .dating_kernels import (
    compute_neural_covariance_kernel,
    compute_transformer_metricity_diagnostics,
)
from .dating_io import (
    verify_coding_alignment,
    parse_sample_dates,
    generate_consensus_sequence,
    generate_time_decay_consensus_sequence,
    compute_time_decay_profile_divergences,
)
from .dating_divergence import (
    extract_tree_root_to_tip,
    compute_tree_free_divergences,
    optimize_latent_convex_hull_root,
)
from .dating_models import (
    _spectral_eigh,
    _gls_fit,
    _generalized_r2,
    compute_fieller_mrca_interval,
    compute_poisson_mrca_interval,
    compute_residual_bootstrap_mrca_interval,
    run_dating_loocv,
    run_ols_dating,
    run_pgls_dating,
    estimate_reml_pagel_lambda,
    tune_ridge_for_pgls,
    run_powerlaw_clock_dating,
    compute_rcs_basis,
    run_restricted_spline_clock_dating,
)
from .dating_plots import plot_mrca_dating, plot_alluvial_phylogeny

def _compute_precision_weighted_ensemble(
    ols_res: Optional[Dict[str, Any]],
    pgls_res: Optional[Dict[str, Any]],
    spline_res: Optional[Dict[str, Any]],
    times: np.ndarray
) -> Dict[str, Any]:
    """Precision-weighted multi-model ensembling (Hartung-Knapp / Burnham & Anderson).

    Computes precision weights from CI widths, ensembles t_MRCA estimates,
    and returns ensemble results with between-model variance.
    """
    ols_valid = ols_res is not None and not np.isnan(ols_res.get('t_mrca', np.nan)) and ols_res.get('mu', 0) > 0
    pgls_valid = pgls_res is not None and not np.isnan(pgls_res.get('t_mrca', np.nan)) and pgls_res.get('mu', 0) > 0
    spline_valid = spline_res is not None and not np.isnan(spline_res.get('t_mrca', np.nan)) and spline_res.get('rate_ancestral', 0) > 0

    ols_g = float(ols_res.get('fieller_g', np.nan)) if (ols_res and ols_res.get('fieller_g') is not None) else np.nan
    pgls_g = float(pgls_res.get('fieller_g', np.nan)) if (pgls_res and pgls_res.get('fieller_g') is not None) else np.nan

    ols_bounded = (not np.isnan(ols_g)) and (ols_g < 1.0)
    pgls_bounded = (not np.isnan(pgls_g)) and (pgls_g < 1.0)

    mu_ols = float(ols_res.get('mu', 1e-12)) if ols_res else 1e-12
    mu_pgls = float(pgls_res.get('mu', 1e-12)) if pgls_res else 1e-12
    attr = float(mu_pgls / max(1e-12, mu_ols)) if (ols_valid and pgls_valid) else 1.0

    r2_ols = float(ols_res.get('r2', 0.0)) if ols_res else 0.0
    r2_pgls = float(pgls_res.get('r2', 0.0)) if pgls_res else 0.0
    is_clade_attenuated = ols_bounded and (attr < 0.50) and (r2_pgls < 0.65 * r2_ols)

    precisions = {}
    candidate_models = {}
    if ols_valid and ols_bounded:
        ci_w = ols_res['ci_mrca'][1] - ols_res['ci_mrca'][0] if ols_res.get('ci_mrca') and not np.isneginf(ols_res['ci_mrca'][0]) and not np.isinf(ols_res['ci_mrca'][1]) else np.nan
        if not np.isnan(ci_w) and ci_w > 0:
            precisions['ols'] = 1.0 / (ci_w ** 2)
            candidate_models['ols'] = ols_res

    if pgls_valid and pgls_bounded and not is_clade_attenuated:
        ci_w = pgls_res['ci_mrca'][1] - pgls_res['ci_mrca'][0] if pgls_res.get('ci_mrca') and not np.isneginf(pgls_res['ci_mrca'][0]) and not np.isinf(pgls_res['ci_mrca'][1]) else np.nan
        if not np.isnan(ci_w) and ci_w > 0:
            precisions['pgls'] = 1.0 / (ci_w ** 2)
            candidate_models['pgls'] = pgls_res

    if spline_valid and spline_res.get('is_nonlinear_preferred'):
        ci_w = spline_res['ci_mrca'][1] - spline_res['ci_mrca'][0] if spline_res.get('ci_mrca') and not np.isneginf(spline_res['ci_mrca'][0]) and not np.isinf(spline_res['ci_mrca'][1]) else np.nan
        if not np.isnan(ci_w) and ci_w > 0:
            precisions['spline'] = 1.0 / (ci_w ** 2)
            candidate_models['spline'] = spline_res

    ensemble_t_mrca = None
    ensemble_ci = None
    model_weights = {}
    if precisions:
        tot_prec = sum(precisions.values())
        model_weights = {m: float(precisions[m] / max(1e-12, tot_prec)) for m in precisions}
        ensemble_t_mrca = float(sum(model_weights[m] * candidate_models[m]['t_mrca'] for m in model_weights))

        t_crit = float(stats.t.ppf(0.975, df=max(1, len(times) - 2)))
        tot_var = 0.0
        for m, w_m in model_weights.items():
            ci_m = candidate_models[m]['ci_mrca']
            se_m = (ci_m[1] - ci_m[0]) / (2.0 * t_crit)
            tot_var += w_m * (se_m ** 2 + (candidate_models[m]['t_mrca'] - ensemble_t_mrca) ** 2)
        se_ens = float(np.sqrt(max(1e-12, tot_var)))
        min_sample_time = float(np.min(times))
        ensemble_ci = [float(ensemble_t_mrca - t_crit * se_ens), min(min_sample_time, float(ensemble_t_mrca + t_crit * se_ens))]
    elif ols_valid:
        model_weights = {'ols': 1.0}
        ensemble_t_mrca = float(ols_res['t_mrca'])
        ensemble_ci = ols_res.get('ci_mrca')

    return {
        't_mrca': ensemble_t_mrca,
        'ci_mrca': ensemble_ci,
        'weights': model_weights,
        'is_clade_attenuated': is_clade_attenuated,
        'ols_valid': ols_valid,
        'pgls_valid': pgls_valid,
        'spline_valid': spline_valid,
        'ols_bounded': ols_bounded,
        'pgls_bounded': pgls_bounded,
        'ols_g': ols_g,
        'pgls_g': pgls_g,
        'attr': attr,
        'mu_ols': mu_ols,
        'mu_pgls': mu_pgls,
    }


def _select_clock_model(
    clock_model: str,
    ols_res: Optional[Dict[str, Any]],
    pgls_res: Optional[Dict[str, Any]],
    spline_res: Optional[Dict[str, Any]],
    power_res: Optional[Dict[str, Any]],
    ensemble_info: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Select the best clock model based on fit quality and diagnostic flags.

    Returns (active_model, selected_clock_description).
    """
    ols_valid = ensemble_info['ols_valid']
    pgls_valid = ensemble_info['pgls_valid']
    spline_valid = ensemble_info['spline_valid']
    ols_bounded = ensemble_info['ols_bounded']
    pgls_bounded = ensemble_info['pgls_bounded']
    is_clade_attenuated = ensemble_info['is_clade_attenuated']
    ols_g = ensemble_info['ols_g']
    pgls_g = ensemble_info['pgls_g']
    attr = ensemble_info['attr']
    mu_ols = ensemble_info['mu_ols']
    mu_pgls = ensemble_info['mu_pgls']

    selected_clock = "Linear"
    if clock_model == "spline" and spline_res is not None:
        active_model = spline_res
        selected_clock = "Restricted Spline (forced)"
    elif clock_model == "power" and power_res is not None:
        active_model = power_res
        selected_clock = "Power-Law (forced)"
    elif clock_model == "linear":
        if pgls_valid and not is_clade_attenuated and (pgls_bounded or not ols_bounded):
            active_model = pgls_res
            selected_clock = "Linear (HyphAeon PGLS)"
        elif ols_valid:
            active_model = ols_res
            selected_clock = "Linear (Standard OLS)"
        else:
            active_model = pgls_res if pgls_res is not None else ols_res
            selected_clock = "Linear (forced)"
    else:  # auto
        if spline_valid and spline_res.get('is_nonlinear_preferred'):
            active_model = spline_res
            ratio_str = f"acceleration ({spline_res['rate_ratio']:.2f}x)" if spline_res['rate_ratio'] > 1.0 else f"deceleration ({spline_res['rate_ratio']:.2f}x)"
            selected_clock = f"Restricted Spline (rate {ratio_str} detected: F={spline_res['f_stat']:.2f}, p={spline_res['p_f_test']:.4f}, ΔAIC={spline_res['delta_aic']:+.1f})"
        elif pgls_valid and not pgls_bounded and ols_bounded:
            active_model = ols_res
            selected_clock = f"Linear (OLS preferred: PGLS temporal slope non-significant, g={pgls_g:.2f} vs OLS g={ols_g:.3f})"
        elif pgls_valid and is_clade_attenuated:
            active_model = ols_res
            selected_clock = f"Linear (OLS preferred: PGLS clade attenuation detected, rate deflated {1/attr:.1f}x from OLS {mu_ols:.2e} to {mu_pgls:.2e})"
        elif pgls_valid and pgls_bounded:
            active_model = pgls_res
            sp_p = f"p={spline_res['p_f_test']:.4f}" if spline_res else "p=n/a"
            lam_val = pgls_res.get('pagel_lambda')
            lam_str = f", λ*={lam_val:.4f}" if isinstance(lam_val, (float, int)) else ""
            selected_clock = f"Linear PGLS (parsimonious linear clock preferred{lam_str}; {sp_p})"
        elif ols_valid and ols_bounded:
            active_model = ols_res
            selected_clock = "Linear (Standard OLS)"
        elif pgls_valid:
            active_model = pgls_res
            g_p_str = f"{pgls_g:.2f}" if not np.isnan(pgls_g) else "inf"
            g_o_str = f"{ols_g:.2f}" if not np.isnan(ols_g) else "inf"
            selected_clock = f"Linear PGLS (unbounded temporal signal: PGLS g={g_p_str}, OLS g={g_o_str}; slope p >= 0.05)"
        elif ols_valid:
            active_model = ols_res
            selected_clock = "Linear (OLS fallback: PGLS non-positive rate)"
        else:
            active_model = pgls_res if pgls_res is not None else ols_res
            selected_clock = "Linear (parsimonious linear clock; non-positive rate)"

    return active_model, selected_clock


def _compute_taxon_predictions(
    active_model: Dict[str, Any],
    times: np.ndarray,
    dists: np.ndarray,
    taxa: List[str],
    is_train: np.ndarray,
    train_idx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Compute per-taxon fitted values, predicted dates, residuals, and taxon records.

    Returns (fitted_all, pred_dates, residuals, taxon_records).
    """
    if active_model.get('method') == 'RESTRICTED_SPLINE':
        b0 = float(active_model.get('beta_0', active_model.get('beta', [0, 0, 0])[0]))
        b1 = float(active_model.get('beta_1', active_model.get('beta', [0, 0, 0])[1]))
        b2 = float(active_model.get('beta_2', active_model.get('beta', [0, 0, 0])[2]))
        knots_arr = np.array(active_model['knots'])
        B_all, _ = compute_rcs_basis(times, knots_arr)
        fitted_all = b0 + b1 * times + b2 * B_all[:, 0]

        t_kn0 = knots_arr[0]
        B_kn0, _ = compute_rcs_basis(np.array([t_kn0]), knots_arr)
        d_kn0 = float(b0 + b1 * t_kn0 + b2 * B_kn0[0, 0])

        pred_dates = np.zeros(len(dists))
        for idx_d, d_val in enumerate(dists):
            if b1 <= 1e-6:
                pred_dates[idx_d] = np.nan
            elif d_val <= d_kn0 or abs(b2) < 1e-12:
                pred_dates[idx_d] = (d_val - b0) / b1
            else:
                def f_diff(t_cand):
                    B_c, _ = compute_rcs_basis(np.array([t_cand]), knots_arr)
                    return float(b0 + b1 * t_cand + b2 * B_c[0, 0] - d_val)
                try:
                    t_root_sol = optimize.brentq(f_diff, t_kn0, max(times) + 100.0)
                    pred_dates[idx_d] = t_root_sol
                except Exception:
                    pred_dates[idx_d] = (d_val - b0) / b1
    elif active_model.get('method') == 'POWER_LAW':
        k_val = max(1e-12, active_model['k'])
        th_val = max(1e-4, active_model['theta'])
        t0_val = active_model['t_mrca']
        fitted_all = k_val * (np.maximum(1e-6, times - t0_val) ** th_val)
        pred_dates = t0_val + (np.maximum(0.0, dists) / k_val) ** (1.0 / th_val)
    else:
        fitted_all = active_model['d0'] + active_model['mu'] * (times - active_model['t_ref'])
        if active_model['mu'] > 1e-6:
            pred_dates = active_model['t_ref'] + (dists - active_model['d0']) / active_model['mu']
        else:
            pred_dates = np.full(len(dists), np.nan)

    residuals = dists - fitted_all
    train_resids = residuals[train_idx]
    std_res = np.std(train_resids) if (len(train_resids) >= 3 and np.std(train_resids) > 1e-12) else (np.std(residuals) if np.std(residuals) > 1e-12 else 1.0)

    taxon_records = []
    for i, t in enumerate(taxa):
        is_holdout = bool(not is_train[i])
        z_score = float(residuals[i] / std_res)
        is_outlier = bool(abs(z_score) >= 2.5) if not is_holdout else False
        temporal_res = float(pred_dates[i] - times[i]) if not np.isnan(pred_dates[i]) else np.nan
        taxon_records.append({
            'taxon': t,
            'sampling_date': float(times[i]),
            'root_divergence': float(dists[i]),
            'fitted_divergence': float(fitted_all[i]),
            'predicted_date': float(pred_dates[i]),
            'divergence_residual': float(residuals[i]),
            'temporal_residual': temporal_res,
            'z_score': z_score,
            'is_outlier': is_outlier,
            'is_holdout': is_holdout
        })

    return fitted_all, pred_dates, residuals, taxon_records


def _export_dating_results(
    results: Dict[str, Any],
    output_prefix: Optional[str],
    taxon_records: List[Dict[str, Any]],
    ols_res: Optional[Dict[str, Any]],
    pgls_res: Optional[Dict[str, Any]],
    spline_res: Optional[Dict[str, Any]],
    power_res: Optional[Dict[str, Any]],
    loocv_res: Optional[Dict[str, Any]],
    latent_root_res: Optional[Dict[str, Any]],
    metricity_diag: Optional[Dict[str, Any]],
    effective_dist_mode: str,
    active_model_name: str,
    active_tmrca_val: Optional[float],
    active_ci_val: Optional[List[float]],
    active_mu_val: Optional[float],
    clock_model: str,
    ci_method: str,
    selected_clock: str,
    ensemble_res: Dict[str, Any],
    plot: bool
) -> None:
    """Export dating results to JSON, CSV, and optional diagnostic plot."""
    if output_prefix:
        out_p = Path(output_prefix)
        ensure_parent_directory(out_p)

        json_data = {
            'alignment': results['alignment'],
            'tree': results['tree'],
            'root_description': results['root_description'],
            'distance_mode': effective_dist_mode,
            'metricity_diagnostics': metricity_diag,
            'latent_root': {
                'alpha': float(latent_root_res['alpha']),
                'temporal_r': float(latent_root_res['temporal_r']),
                'temporal_r2': float(latent_root_res['temporal_r2']),
                'anchor_taxa': latent_root_res['anchor_taxa']
            } if latent_root_res else None,
            'taxa_count': results['taxa_count'],
            'timespan': results['timespan'],
            'elapsed_seconds': results['elapsed_seconds'],
            'active_model': active_model_name,
            't_mrca': active_tmrca_val,
            'ci_mrca': active_ci_val,
            'mu': active_mu_val,
            'ols': {k: v for k, v in ols_res.items() if k not in ['residuals', 'fitted', 'times']},
            'pgls': {k: v for k, v in pgls_res.items() if k not in ['residuals', 'fitted', 'times']} if pgls_res else None,
            'spline': {k: v for k, v in spline_res.items() if k not in ['residuals', 'fitted']} if spline_res else None,
            'power': {k: v for k, v in power_res.items() if k not in ['residuals', 'fitted']} if power_res else None,
            'clock_model': clock_model,
            'ci_method': ci_method,
            'selected_clock': selected_clock,
            'ensemble': ensemble_res,
            'loocv': {k: v for k, v in loocv_res.items() if k != 'records'} if loocv_res else None,
            'taxa_summary': taxon_records
        }
        json_file = out_p.with_suffix('.json') if not str(out_p).endswith('.json') else out_p
        write_json(json_file, json_data)
        print(f"[✓] Saved JSON summary: {json_file}")

        csv_file = out_p.with_suffix('.csv') if not str(out_p).endswith('.csv') else out_p
        pd.DataFrame(taxon_records).to_csv(csv_file, index=False)
        print(f"[✓] Saved per-taxon CSV: {csv_file}")

    if plot:
        fig_path = f"{output_prefix}_diagnostic.pdf" if output_prefix else "mrca_dating_diagnostic.pdf"
        plot_mrca_dating(results, fig_path)


# =========================================================================
# 6. Master MRCA Dating Pipeline
# =========================================================================

def run_mrca_dating(
    alignment_path: Optional[Union[str, Path]] = None,
    tree_path: Optional[Union[str, Path]] = None,
    dates_source: Optional[Union[str, Path, Dict[str, float]]] = None,
    date_col: Optional[str] = None,
    strain_col: Optional[str] = None,
    date_regex: Optional[str] = None,
    root_taxon: Optional[str] = None,
    decay_gamma: Optional[float] = None,
    decay_half_life: Optional[float] = None,
    optimize_root: bool = True,
    use_tn93: bool = False,
    distance_mode: str = "auto",
    method: str = "all",
    clock_model: str = "auto",
    ci_method: str = "fieller",
    seq_len: Optional[int] = None,
    ridge: Union[float, str] = "auto",
    tune_ridge: bool = False,
    n_bootstrap: int = 1000,
    model: Optional[torch.nn.Module] = None,
    weights: Optional[str] = None,
    variant: Optional[str] = None,
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
    max_species: Optional[int] = None,
    allow_stop_codons: bool = True,
    loocv: bool = False,
    output_prefix: Optional[str] = None,
    plot: bool = False,
    beast_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Executes end-to-end molecular clock calibration and MRCA dating on time-stamped sequences.
    Directly ingests FASTA, NEXUS, or BEAST 1.x / 2.x XML configuration files.

    Returns:
        Structured dictionary with model parameters, confidence intervals,
        per-taxon predictions, and model diagnostics.
    """
    t0 = time.time()

    # Ingest BEAST XML if specified via beast_path or alignment_path
    target_xml = beast_path or (alignment_path if (alignment_path and str(alignment_path).lower().endswith(('.xml', '.xml.gz'))) else None)
    if target_xml is not None:
        target_xml_p = Path(target_xml)
        if not target_xml_p.exists():
            raise FileNotFoundError(f"BEAST XML file not found: {target_xml}")
        beast_data = parse_beast_xml(target_xml)
        n_xml_seqs = len(beast_data.get('sequences', {}))
        n_xml_dates = len(beast_data.get('dates', {}))
        has_xml_tree = bool(beast_data.get('tree_newick'))
        print(f"[*] Ingested BEAST XML ({beast_data.get('version', 'BEAST')}): {n_xml_seqs} sequences, {n_xml_dates} dates, starting tree={'present' if has_xml_tree else 'none'}.")

        if alignment_path is None:
            alignment_path = target_xml
        if dates_source is None and n_xml_dates > 0:
            dates_source = beast_data['dates']
        if tree_path is None and has_xml_tree and not use_tn93:
            tree_path = beast_data['tree_newick']
            print(f"[*] Auto-detected embedded starting tree from BEAST XML.")

    if alignment_path is None:
        raise ValueError("No sequence alignment provided. Please specify an alignment file (-a/--alignment) or a BEAST XML (--beast).")

    align_p = Path(alignment_path)
    if not align_p.exists():
        raise FileNotFoundError(f"Alignment file not found: {align_p}")

    # 1. Parse and Validate In-Frame Coding Alignment
    seq_dict = parse_alignment_sequences(str(align_p))
    n_taxa_raw, n_codons = verify_coding_alignment(seq_dict, allow_stop_codons=allow_stop_codons)
    print(f"[*] Alignment verified: {n_taxa_raw} taxa, {n_codons} codons ({n_codons * 3} nt in-frame).")

    # 2. Extract and Parse Sampling Timestamps
    taxa_all = list(seq_dict.keys())
    dates_map, missing = parse_sample_dates(
        taxa_all,
        dates_source=dates_source,
        date_col=date_col,
        strain_col=strain_col,
        date_regex=date_regex
    )
    
    # Filter to dated taxa (excluding root_taxon if root_taxon is an ancestral reference without timestamp)
    dated_taxa = [t for t in taxa_all if t in dates_map and not np.isnan(dates_map[t]) and (root_taxon is None or t != root_taxon)]
    n_dated = len(dated_taxa)
    print(f"[*] Timestamps mapped: {n_dated}/{len(taxa_all)} taxa successfully dated.")
    if missing and len(missing) <= 10:
        print(f"    Notice: {len(missing)} taxa omitted due to missing timestamps: {missing}")
    elif missing:
        print(f"    Notice: {len(missing)} taxa omitted due to missing timestamps.")

    if n_dated < 3:
        raise ValueError(
            f"Fewer than 3 taxa could be mapped to valid timestamps ({n_dated} dated). "
            f"Please check your metadata file (--dates) or FASTA header format."
        )

    # 3. Compute Patristic, Latent Convex Hull, or TN93 Tree-Free Divergences
    has_tree = (tree_path is not None and not use_tn93 and (
        (isinstance(tree_path, (str, Path)) and os.path.exists(str(tree_path))) or
        (isinstance(tree_path, str) and tree_path.strip().startswith('('))
    ))
    run_neural = method in ["all", "pgls"]
    mode = str(distance_mode).lower().strip()

    if use_tn93 or mode in ["tn93", "consensus"]:
        effective_dist_mode = "tn93"
    elif mode in ["latent", "continuous", "hull", "manifold"]:
        effective_dist_mode = "latent"
    elif mode in ["tree", "patristic"]:
        effective_dist_mode = "tree"
    elif mode == "auto":
        effective_dist_mode = "tree" if has_tree else "tn93"
    else:
        effective_dist_mode = "tree" if has_tree else "tn93"

    latent_root_res = None
    cov_matrix = None
    metricity_diag = None
    msa_codons = None
    msa_aas = None
    tree_cache = None
    aln_taxa = None
    L = n_codons

    if effective_dist_mode == "latent":
        print(f"[*] Latent Distance Mode: Extracting continuous representations and optimizing convex hull root...")
        if device is None:
            device = get_device()
        if model is None:
            print(f"[*] Loading HyphAeon transformer backbone on {device}...")
            model = load_model(weights=weights, variant=variant, device=device)

        # Prepare alignment tensors
        c, a, d, z, inv, aln_taxa, L, tree_cache = prepare_alignment(
            str(align_p),
            str(tree_path) if has_tree else None,
            model=model,
            device=device,
            max_species=max_species,
            prune_duplicates=False,
            use_tn93=(not has_tree)
        )
        msa_codons = c.to(device)
        msa_aas = a.to(device)

        t_fwd = time.time()
        cross_attn, taxa_repr = extract_cross_taxa_attentions_and_embeddings(
            model, msa_codons, msa_aas, tree_cache, device=device
        )
        K_neural = compute_neural_covariance_kernel(cross_attn, taxa_repr)
        print(f"[✓] Extracted {taxa_repr.shape[0]} continuous sequence embeddings in {time.time() - t_fwd:.2f}s.")

        aln_taxa_map = {t: i for i, t in enumerate(aln_taxa)}
        sub_indices = [aln_taxa_map[t] for t in dated_taxa if t in aln_taxa_map]
        taxa = [dated_taxa[i] for i, t in enumerate(dated_taxa) if t in aln_taxa_map]
        times = np.array([dates_map[t] for t in taxa], dtype=np.float64)
        z_sub = taxa_repr[sub_indices]
        cov_matrix = K_neural[sub_indices, :][:, sub_indices]

        # Filter out heavily degraded or partial sequences (<50% coverage) from defining the ancestral root
        char_mat = np.array([list(seq_dict[t]) for t in taxa])
        valid_counts = np.array([np.sum(np.isin(char_mat[i], list('ACGT'))) for i in range(len(taxa))])
        coverage = valid_counts / max(1, char_mat.shape[1])
        anchor_mask = (coverage >= 0.50)
        n_masked = int(np.sum(~anchor_mask))
        if n_masked > 0:
            print(f"[*] Latent Convex Hull: Masked {n_masked} partial/degraded sequence(s) (<50% coverage) from root anchor set.")

        # Compute exact pairwise nucleotide Hamming/TN93 distance for isometric calibration
        if d is not None:
            d_phys_all = d.squeeze(0).cpu().numpy()
            pairwise_phys = d_phys_all[np.ix_(sub_indices, sub_indices)]
        else:
            N_t = len(taxa)
            pairwise_phys = np.zeros((N_t, N_t), dtype=np.float64)
            for i in range(N_t):
                for j in range(i + 1, N_t):
                    v = np.isin(char_mat[i], list('ACGT')) & np.isin(char_mat[j], list('ACGT'))
                    diffs = np.sum((char_mat[i] != char_mat[j]) & v)
                    tot = np.sum(v)
                    pairwise_phys[i, j] = diffs / max(1, tot)
                    pairwise_phys[j, i] = pairwise_phys[i, j]

        latent_root_res = optimize_latent_convex_hull_root(
            z_sub, times, taxa_names=taxa, pairwise_phys_dists=pairwise_phys, anchor_mask=anchor_mask, device=device
        )
        dists = latent_root_res['dists']
        root_desc = f"latent_convex_hull (α={latent_root_res['alpha']:.5f} subs/site/unit, R={latent_root_res['temporal_r']:+.3f})"

    elif effective_dist_mode == "tree":
        tree_desc = "embedded starting tree" if (isinstance(tree_path, str) and tree_path.strip().startswith('(')) else str(tree_path)
        print(f"[*] Computing patristic tree distances from: {tree_desc}...")
        tree_dists, root_desc = extract_tree_root_to_tip(
            str(tree_path), dated_taxa, dates_map,
            root_taxon=root_taxon, optimize_root=optimize_root
        )
        taxa = [t for t in dated_taxa if t in tree_dists]
        times = np.array([dates_map[t] for t in taxa], dtype=np.float64)
        dists = np.array([tree_dists[t] for t in taxa], dtype=np.float64)

        # In auto mode, check if tree patristic distances suffer from non-positive slope or negligible temporal correlation
        if mode == "auto" and run_neural and len(times) >= 5:
            std_t = np.std(times)
            std_d = np.std(dists)
            tree_slope = float(np.polyfit(times, dists, 1)[0]) if (std_t > 1e-7 and std_d > 1e-7) else 0.0
            tree_r2 = float(np.corrcoef(times, dists)[0, 1] ** 2) if (std_t > 1e-7 and std_d > 1e-7) else 0.0

            if tree_slope <= 1e-6 or tree_r2 < 0.02:
                print(f"[!] Notice: Tree root-to-tip patristic regression has negligible temporal signal (slope={tree_slope:.2e}, R^2={tree_r2:.3f}).")
                print(f"[*] Auto-evaluating continuous sequence representation space (latent convex hull)...")
                try:
                    if device is None:
                        device = get_device()
                    if model is None:
                        print(f"[*] Loading HyphAeon transformer backbone on {device}...")
                        model = load_model(weights=weights, variant=variant, device=device)

                    c_lat, a_lat, d_lat, _, _, aln_taxa_lat, _, tree_cache_lat = prepare_alignment(
                        str(align_p), str(tree_path) if has_tree else None,
                        model=model, device=device, max_species=max_species,
                        prune_duplicates=False, use_tn93=False
                    )
                    cross_attn_lat, taxa_repr_lat = extract_cross_taxa_attentions_and_embeddings(
                        model, c_lat.to(device), a_lat.to(device), tree_cache_lat, device=device
                    )
                    K_neural_lat = compute_neural_covariance_kernel(cross_attn_lat, taxa_repr_lat)

                    aln_taxa_map_l = {t: i for i, t in enumerate(aln_taxa_lat)}
                    sub_indices_l = [aln_taxa_map_l[t] for t in dated_taxa if t in aln_taxa_map_l]
                    taxa_l = [dated_taxa[i] for i, t in enumerate(dated_taxa) if t in aln_taxa_map_l]
                    times_l = np.array([dates_map[t] for t in taxa_l], dtype=np.float64)
                    z_sub_l = taxa_repr_lat[sub_indices_l]

                    char_mat_l = np.array([list(seq_dict[t]) for t in taxa_l])
                    valid_counts_l = np.array([np.sum(np.isin(char_mat_l[i], list('ACGT'))) for i in range(len(taxa_l))])
                    coverage_l = valid_counts_l / max(1, char_mat_l.shape[1])
                    anchor_mask_l = (coverage_l >= 0.50)

                    if d_lat is not None:
                        d_phys_l_all = d_lat.squeeze(0).cpu().numpy()
                        pairwise_phys_l = d_phys_l_all[np.ix_(sub_indices_l, sub_indices_l)]
                    else:
                        N_l = len(taxa_l)
                        pairwise_phys_l = np.zeros((N_l, N_l), dtype=np.float64)
                        for i in range(N_l):
                            for j in range(i + 1, N_l):
                                v = np.isin(char_mat_l[i], list('ACGT')) & np.isin(char_mat_l[j], list('ACGT'))
                                diffs = np.sum((char_mat_l[i] != char_mat_l[j]) & v)
                                tot = np.sum(v)
                                pairwise_phys_l[i, j] = diffs / max(1, tot)
                                pairwise_phys_l[j, i] = pairwise_phys_l[i, j]

                    cand_lat_res = optimize_latent_convex_hull_root(
                        z_sub_l, times_l, taxa_names=taxa_l, pairwise_phys_dists=pairwise_phys_l,
                        anchor_mask=anchor_mask_l, device=device
                    )
                    lat_r2 = cand_lat_res.get('temporal_r2', 0.0)
                    lat_r = cand_lat_res.get('temporal_r', 0.0)

                    if lat_r > 0 and lat_r2 > tree_r2 + 0.05:
                        print(f"[✓] Auto-Promoted Continuous Latent Distance Mode: Temporal signal improved from Tree R^2={tree_r2:.3f} to Latent R^2={lat_r2:.3f} (R={lat_r:+.3f}).")
                        taxa = taxa_l
                        times = times_l
                        dists = cand_lat_res['dists']
                        latent_root_res = cand_lat_res
                        cov_matrix = K_neural_lat[sub_indices_l, :][:, sub_indices_l]
                        z_sub = z_sub_l
                        effective_dist_mode = "latent"
                        root_desc = f"latent_convex_hull (promoted over tree: R^2={lat_r2:.3f} vs tree R^2={tree_r2:.3f})"
                except Exception as e_lat:
                    print(f"[*] Latent space auto-evaluation notice: {e_lat}")

    else:  # "tn93"
        print(f"[*] Estimating tree-free pairwise distances via TN93...")
        dists, root_desc = compute_tree_free_divergences(
            seq_dict, dated_taxa, dates_map,
            root_taxon=root_taxon, decay_gamma=decay_gamma, decay_half_life=decay_half_life
        )
        if root_taxon and root_taxon in seq_dict:
            taxa = [t for t in dated_taxa if t != root_taxon]
        else:
            taxa = list(dated_taxa)
        times = np.array([dates_map[t] for t in taxa], dtype=np.float64)

    print(f"[*] Root configuration: {root_desc} (Timespan: {np.min(times):.1f} - {np.max(times):.1f})")

    # Evaluate sequence coverage to detect partial / degraded holdout isolates (<50% coverage)
    char_mat = np.array([list(seq_dict[t]) for t in taxa])
    valid_counts = np.array([np.sum(np.isin(char_mat[i], list('ACGT'))) for i in range(len(taxa))])
    coverage = valid_counts / max(1, char_mat.shape[1])
    is_train = (coverage >= 0.50)
    n_holdouts = int(np.sum(~is_train))

    if n_holdouts > 0 and np.sum(is_train) >= 3:
        holdout_names = [taxa[i] for i in range(len(taxa)) if not is_train[i]]
        print(f"[*] Clock Calibration Discipline: Reserved {n_holdouts} partial/holdout sequence(s) (<50% coverage) as out-of-sample test taxa: {holdout_names}")
        train_idx = np.where(is_train)[0]
    else:
        is_train = np.ones(len(taxa), dtype=bool)
        train_idx = np.arange(len(taxa))

    eff_seq_len = seq_len if seq_len is not None else (n_codons * 3 if 'n_codons' in locals() else 1000)

    # 4. Fit Standard OLS on clean training set
    ols_res = run_ols_dating(times[train_idx], dists[train_idx], ci_method=ci_method, seq_len=eff_seq_len, n_boot=n_bootstrap)
    if np.isnan(ols_res['t_mrca']) or ols_res['mu'] <= 0:
        t0_ols_str = "n/a (rate <= 0)"
        ci_ols_str = "[non-pos rate]"
    elif np.isneginf(ols_res['ci_mrca'][0]):
        t0_ols_str = f"{ols_res['t_mrca']:.2f}"
        ci_ols_str = f"[-inf, {ols_res['ci_mrca'][1]:.1f}]"
    else:
        t0_ols_str = f"{ols_res['t_mrca']:.2f}"
        ci_ols_str = f"[{ols_res['ci_mrca'][0]:.1f}, {ols_res['ci_mrca'][1]:.1f}]"
    print(f"[✓] OLS Molecular Clock: t_MRCA = {t0_ols_str} {ci_ols_str}, μ = {ols_res['mu']:.6f} subs/site/yr (R^2 = {ols_res['r2']:.3f})")

    # 5. ChronAeon Neural Attention PGLS (if requested)
    pgls_res = None
    effective_ridge = 0.05
    opt_lambda = 0.95

    if run_neural:
        if cov_matrix is None:
            if device is None:
                device = get_device()
            if model is None:
                print(f"[*] Loading HyphAeon transformer backbone on {device}...")
                model = load_model(weights=weights, variant=variant, device=device)

            # Load alignment into model tensors
            c, a, d, z, inv, aln_taxa, L, tree_cache = prepare_alignment(
                str(align_p),
                str(tree_path) if has_tree else None,
                model=model,
                device=device,
                max_species=max_species,
                prune_duplicates=False,
                use_tn93=(not has_tree)
            )

            t_fwd = time.time()
            print(f"[*] Extracting cross-taxa attention and 128D continuous representations...")
            msa_codons = c.to(device)
            msa_aas = a.to(device)

            cross_attn, taxa_repr = extract_cross_taxa_attentions_and_embeddings(
                model, msa_codons, msa_aas, tree_cache, device=device
            )
            K_neural = compute_neural_covariance_kernel(cross_attn, taxa_repr)
            print(f"[✓] Forward pass complete in {time.time() - t_fwd:.2f}s! Extracted {taxa_repr.shape[0]} taxa neural representations.")

            # Zero-cost foundation model metricity diagnostics
            D_phys_mat = d.squeeze(0).cpu().numpy()
            cov_for_diag = coverage if 'coverage' in locals() else None
            metricity_diag = compute_transformer_metricity_diagnostics(
                d_matrix=D_phys_mat,
                taxa_repr=taxa_repr,
                cross_attn=cross_attn,
                coverage=cov_for_diag
            )
            print(f"[*] Transformer Metricity Diagnostic: {metricity_diag['regime_label']}")
            print(f"    d_bar = {metricity_diag['d_mean']:.4f}, d_90 = {metricity_diag['d_90']:.4f} subs/site | Spectral distortion rho = {metricity_diag['rho_neg']:.4f}")
            if metricity_diag['rho_iso_spearman'] is not None:
                print(f"    Isometric concordance: Spearman rho = {metricity_diag['rho_iso_spearman']:.4f}, Pearson r = {metricity_diag['rho_iso_pearson']:.4f}")
            print(f"    Decision: {metricity_diag['rationale']}")

            # Align taxa order with dated taxa
            aln_taxa_map = {t: i for i, t in enumerate(aln_taxa)}
            sub_indices = [aln_taxa_map[t] for t in taxa if t in aln_taxa_map]
            sub_taxa = [taxa[i] for i, t in enumerate(taxa) if t in aln_taxa_map]
            sub_times = times[[i for i, t in enumerate(taxa) if t in aln_taxa_map]]
            sub_dists = dists[[i for i, t in enumerate(taxa) if t in aln_taxa_map]]

            cov_matrix = K_neural[sub_indices, :][:, sub_indices]
            z_sub = taxa_repr[sub_indices]

            # Autonomous promotion from TN93 to Latent Space when mutational saturation is detected
            if mode == "auto" and metricity_diag['recommended_regime'] == 'latent' and not has_tree:
                try:
                    anchor_mask_l = (coverage[sub_indices] >= 0.50) if 'coverage' in locals() and coverage is not None else None
                    cand_lat_res = optimize_latent_convex_hull_root(
                        z_sub, sub_times, taxa_names=sub_taxa,
                        pairwise_phys_dists=D_phys_mat[np.ix_(sub_indices, sub_indices)],
                        anchor_mask=anchor_mask_l, device=device
                    )
                    lat_r2 = cand_lat_res.get('temporal_r2', 0.0)
                    lat_r = cand_lat_res.get('temporal_r', 0.0)
                    if lat_r > 0 and (lat_r2 > ols_res['r2'] - 0.05):
                        print(f"[✓] Auto-Promoted Continuous Latent Distance Mode: {metricity_diag['rationale']}. Temporal signal in Latent Space: R^2={lat_r2:.3f} (R={lat_r:+.3f}).")
                        sub_dists = cand_lat_res['dists']
                        dists = sub_dists
                        latent_root_res = cand_lat_res
                        effective_dist_mode = "latent"
                        root_desc = f"latent_convex_hull (promoted over tn93: {metricity_diag['regime_label']})"
                except Exception as e_lat:
                    print(f"[*] Latent auto-promotion notice: {e_lat}")
        else:
            sub_taxa = taxa
            sub_times = times
            sub_dists = dists
            sub_indices = list(range(len(taxa)))

        train_sub = [i for i in range(len(sub_taxa)) if is_train[i]]
        if len(train_sub) < 3:
            train_sub = list(range(len(sub_taxa)))
        cov_train = cov_matrix[train_sub, :][:, train_sub]
        train_times = sub_times[train_sub]
        train_dists = sub_dists[train_sub]

        effective_ridge = 0.05
        opt_lambda = 0.95
        if isinstance(ridge, (int, float)):
            effective_ridge = float(ridge)
            opt_lambda = 1.0 - effective_ridge
        elif str(ridge).lower() in ["auto", "reml"]:
            opt_reml = estimate_reml_pagel_lambda(train_times, train_dists, cov_train)
            opt_lambda = float(opt_reml['best_lambda'])
            # Map Pagel's lambda to complementary nugget ridge: ridge = (1 - lambda)
            effective_ridge = float(np.clip(1.0 - opt_lambda, 0.01, 0.20))
            print(f"[*] REML Estimated Phylogenetic Signal: Pagel's λ* = {opt_lambda:.4f} (nugget ridge = {effective_ridge:.4f})")

        # Fit PGLS on training set
        if method in ["all", "pgls"]:
            eff_seq_len = seq_len if seq_len is not None else (3 * L if 'L' in locals() else eff_seq_len)
            pgls_res = run_pgls_dating(
                train_times, train_dists, cov_train,
                ridge=effective_ridge, pagel_lambda=opt_lambda,
                ci_method=ci_method, seq_len=eff_seq_len, n_boot=n_bootstrap
            )

            # Optional: full neural cross-attention site bootstrap if explicitly requested
            if str(ci_method).lower() in ["site-boot", "site_boot"]:
                b_count = min(50, n_bootstrap)
                print(f"[*] Running {b_count} neural cross-attention site bootstraps...")
                site_t0s = []
                for _ in range(b_count):
                    idx_b = torch.randint(0, L, (L,), device=device)
                    c_b = msa_codons[idx_b, :, :]
                    a_b = msa_aas[idx_b, :, :]
                    c_attn, t_repr = extract_cross_taxa_attentions_and_embeddings(
                        model, c_b, a_b, tree_cache, device=device
                    )
                    del c_b, a_b
                    K_b = compute_neural_covariance_kernel(c_attn, t_repr)
                    del c_attn, t_repr
                    cov_b = K_b[sub_indices, :][:, sub_indices]
                    del K_b
                    res_b = run_pgls_dating(sub_times, sub_dists, cov_b, ridge=effective_ridge, pagel_lambda=opt_lambda, ci_method="delta")
                    del cov_b
                    if not np.isnan(res_b['t_mrca']) and res_b['mu'] > 0:
                        site_t0s.append(res_b['t_mrca'])
                    if "mps" in str(device).lower():
                        torch.mps.empty_cache()
                    elif "cuda" in str(device).lower():
                        torch.cuda.empty_cache()
                if len(site_t0s) >= 10:
                    pgls_res['ci_mrca'] = [float(np.percentile(site_t0s, 2.5)), float(np.percentile(site_t0s, 97.5))]
                    pgls_res['ci_method'] = 'site-boot'

            if np.isnan(pgls_res['t_mrca']) or pgls_res['mu'] <= 0:
                t0_str = "n/a (rate <= 0)"
                ci_str = "[non-pos rate]"
            elif np.isneginf(pgls_res['ci_mrca'][0]):
                t0_str = f"{pgls_res['t_mrca']:.2f}"
                ci_str = f"[-inf, {pgls_res['ci_mrca'][1]:.1f}]"
            else:
                t0_str = f"{pgls_res['t_mrca']:.2f}"
                ci_str = f"[{pgls_res['ci_mrca'][0]:.1f}, {pgls_res['ci_mrca'][1]:.1f}]"
            print(f"[✓] HyphAeon PGLS Clock: t_MRCA = {t0_str} {ci_str}, μ = {pgls_res['mu']:.6f} subs/site/yr (R^2_gls = {pgls_res['r2']:.3f})")

    # 5b. Non-Linear Clock Models: Restricted Natural Spline & Power-Law
    spline_res = None
    power_res = None

    if clock_model in ["auto", "spline"]:
        fit_times = train_times if (run_neural and cov_matrix is not None) else times[train_idx]
        fit_dists = train_dists if (run_neural and cov_matrix is not None) else dists[train_idx]
        spline_cov = cov_train if (run_neural and cov_matrix is not None) else None
        try:
            spline_res = run_restricted_spline_clock_dating(
                fit_times, fit_dists, cov_matrix=spline_cov, ridge=effective_ridge, n_boot=min(500, n_bootstrap)
            )
            ratio_sym = "acceleration" if spline_res['rate_ratio'] > 1.0 else "deceleration"
            if np.isnan(spline_res['t_mrca']) or spline_res['rate_ancestral'] <= 0:
                t0_str = "n/a (ancestral rate <= 0)"
            else:
                t0_str = f"{spline_res['t_mrca']:.2f} [{spline_res['ci_mrca'][0]:.1f}, {spline_res['ci_mrca'][1]:.1f}]"
            print(f"[✓] Restricted Spline Clock: t_MRCA = {t0_str}, μ_anc = {spline_res['rate_ancestral']:.6f}, μ_rec = {spline_res['rate_recent']:.6f} ({ratio_sym} {spline_res['rate_ratio']:.2f}x) (R^2 = {spline_res['r2']:.3f}, ΔAIC = {spline_res['delta_aic']:+.2f}, AIC drop = {spline_res.get('aic_reduction', -spline_res['delta_aic']):.2f}, p = {spline_res['p_f_test']:.4f})")
        except Exception as e:
            print(f"[!] Notice: Restricted spline fitting fell back to linear ({e})")

    elif clock_model == "power":
        fit_times = train_times if (run_neural and cov_matrix is not None) else times[train_idx]
        fit_dists = train_dists if (run_neural and cov_matrix is not None) else dists[train_idx]
        power_cov = opt_lambda * cov_train + (1.0 - opt_lambda) * np.eye(len(fit_times)) if (run_neural and cov_matrix is not None) else None
        try:
            power_res = run_powerlaw_clock_dating(
                fit_times, fit_dists, cov_matrix=power_cov, ridge=0.01, n_boot=min(500, n_bootstrap)
            )
            ci_th_str = f"[{power_res['ci_theta'][0]:.3f}, {power_res['ci_theta'][1]:.3f}]" if (power_res['ci_theta'] and not np.isnan(power_res['ci_theta'][0])) else "[n/a]"
            print(f"[✓] Power-Law Clock: t_MRCA = {power_res['t_mrca']:.2f} [{power_res['ci_mrca'][0]:.1f}, {power_res['ci_mrca'][1]:.1f}], θ = {power_res['theta']:.3f} {ci_th_str}, mean rate = {power_res['rate_mean']:.6f} subs/site/yr (R^2 = {power_res['r2']:.3f}, ΔAIC = {power_res['delta_aic']:+.2f}, AIC drop = {power_res.get('aic_reduction', -power_res['delta_aic']):.2f}, p = {power_res['p_f_test']:.4f})")
        except Exception as e:
            print(f"[!] Notice: Power-law clock fitting fell back to linear ({e})")

    # -------------------------------------------------------------
    # Principled Automated Model Selection & Ensembling Framework
    # -------------------------------------------------------------
    ensemble_info = _compute_precision_weighted_ensemble(ols_res, pgls_res, spline_res, times)
    ensemble_res = {
        't_mrca': ensemble_info['t_mrca'],
        'ci_mrca': ensemble_info['ci_mrca'],
        'weights': ensemble_info['weights']
    }

    active_model, selected_clock = _select_clock_model(
        clock_model, ols_res, pgls_res, spline_res, power_res, ensemble_info
    )
    print(f"[✓] Clock Model Selection: {selected_clock}")

    # 6. Per-Taxon Residuals and Predictions
    fitted_all, pred_dates, residuals, taxon_records = _compute_taxon_predictions(
        active_model, times, dists, taxa, is_train, train_idx
    )

    elapsed_time = time.time() - t0

    active_model_name = 'spline' if (spline_res is not None and active_model == spline_res) else ('pgls' if (pgls_res is not None and active_model == pgls_res) else ('power' if (power_res is not None and active_model == power_res) else 'ols'))
    active_tmrca_val = float(active_model['t_mrca']) if (active_model and not np.isnan(active_model.get('t_mrca', np.nan))) else None
    active_ci_val = active_model.get('ci_mrca') if active_model else None
    active_mu_val = float(active_model.get('mu', active_model.get('rate_ancestral', 0.0))) if active_model else None

    # 6b. Leave-One-Out Cross-Validation (LOOCV) if requested
    loocv_res = None
    if loocv and len(train_idx) >= 4:
        train_taxa_list = [taxa[i] for i in train_idx]
        train_t_arr = times[train_idx]
        train_d_arr = dists[train_idx]
        loocv_method = "pgls" if (active_model_name == 'pgls' and cov_matrix is not None) else "ols"
        cov_for_loocv = cov_train if (loocv_method == "pgls" and 'cov_train' in locals()) else None

        print(f"[*] Running Leave-One-Out Cross-Validation (LOOCV) ({loocv_method.upper()}, N={len(train_idx)} taxa)...")
        loocv_res = run_dating_loocv(
            times=train_t_arr,
            dists=train_d_arr,
            taxa=train_taxa_list,
            cov_matrix=cov_for_loocv,
            ridge=effective_ridge,
            pagel_lambda=opt_lambda if 'opt_lambda' in locals() else None,
            method=loocv_method,
            min_time=float(np.min(times))
        )

        # Compare with Fieller analytical interval
        if active_ci_val and not np.isnan(active_ci_val[0]) and not np.isneginf(active_ci_val[0]) and not np.isnan(active_ci_val[1]):
            fieller_w = float(active_ci_val[1] - active_ci_val[0])
            loocv_res['fieller_ci_width_years'] = fieller_w
            loocv_res['fieller_ci_width_days'] = fieller_w * 365.25
            if loocv_res.get('jackknife_ci_width_years') and not np.isnan(loocv_res['jackknife_ci_width_years']) and loocv_res['jackknife_ci_width_years'] > 0:
                loocv_res['fieller_to_jackknife_ratio'] = float(fieller_w / loocv_res['jackknife_ci_width_years'])

        loocv_map = {r['taxon']: r for r in loocv_res['records']}
        for tr in taxon_records:
            t_name = tr['taxon']
            if t_name in loocv_map:
                rec = loocv_map[t_name]
                tr['loocv_predicted_date'] = rec['predicted_date']
                tr['loocv_error_days'] = rec['loocv_error_days']
                tr['jackknife_tmrca'] = rec['jackknife_tmrca']
                tr['leverage'] = rec['leverage']
            else:
                tr['loocv_predicted_date'] = None
                tr['loocv_error_days'] = None
                tr['jackknife_tmrca'] = None
                tr['leverage'] = None

        print(f"[✓] LOOCV Tip Date MAE = {loocv_res['tip_mae_days']:.1f} days (RMSE = {loocv_res['tip_rmse_days']:.1f} days, 95% Pred Interval = {loocv_res['tip_pred_ci_95_days']:.1f} days)")
        if not np.isnan(loocv_res.get('jackknife_se_years', np.nan)):
            ci_j = loocv_res['jackknife_ci']
            print(f"[✓] Jackknife t_MRCA = {loocv_res['jackknife_mean']:.2f} [{ci_j[0]:.1f}, {ci_j[1]:.1f}] (SE = {loocv_res['jackknife_se_years']:.4f} yr / {loocv_res['jackknife_se_days']:.1f} days)")

    results = {
        'alignment': str(align_p),
        'tree': str(tree_path) if has_tree else None,
        'root_description': root_desc,
        'distance_mode': effective_dist_mode,
        'metricity_diagnostics': metricity_diag,
        'latent_root': latent_root_res,
        'taxa_count': len(taxa),
        'timespan': [float(np.min(times)), float(np.max(times))],
        'elapsed_seconds': elapsed_time,
        'times': times,
        'dists': dists,
        'taxa': taxa,
        'active_model': active_model_name,
        't_mrca': active_tmrca_val,
        'ci_mrca': active_ci_val,
        'mu': active_mu_val,
        'ols': ols_res,
        'pgls': pgls_res,
        'spline': spline_res,
        'power': power_res,
        'clock_model': clock_model,
        'ci_method': ci_method,
        'selected_clock': selected_clock,
        'ensemble': ensemble_res,
        'loocv': loocv_res,
        'taxa_records': taxon_records
    }

    # 7. Export Results
    _export_dating_results(
        results, output_prefix, taxon_records,
        ols_res, pgls_res, spline_res, power_res, loocv_res,
        latent_root_res, metricity_diag, effective_dist_mode,
        active_model_name, active_tmrca_val, active_ci_val, active_mu_val,
        clock_model, ci_method, selected_clock, ensemble_res, plot
    )

    return results
