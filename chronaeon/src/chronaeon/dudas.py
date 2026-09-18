#!/usr/bin/env python3
"""
chronaeon.dudas
---------------
Implementation of the Suchard / Dudas suite of time-dependent molecular clock models:
1. Exact Quadratic Polynomial Clock (integrated linear rate acceleration/deceleration)
2. Exact Profile Log-Linear Rate Clock (exponential rate decay or expansion)
3. Exact Bilinear Surge-and-Crash Clock (piecewise quadratic distance spline)
4. Polyepoch Piecewise Constant Clock (multi-epoch non-negative least squares)
"""

import math
import time
from typing import Dict, Any, Optional
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar, nnls


def evaluate_dudas_clock_models(
    times: np.ndarray,
    dists: np.ndarray,
    rss_ols: Optional[float] = None,
    aic_ols: Optional[float] = None,
    n_epochs: int = 3,
    n_eff: Optional[float] = None
) -> Dict[str, Any]:
    """
    Fits and evaluates the Suchard / Dudas suite of time-varying regression clock models.

    Args:
        times: Array of sampling dates (in calendar years).
        dists: Array of root-to-tip genetic distances.
        rss_ols: Optional residual sum of squares of the baseline linear OLS model.
        aic_ols: Optional AIC of the baseline linear OLS model.
        n_epochs: Number of discrete temporal epochs for the polyepoch clock (default: 3).
        n_eff: Optional effective sample size accounting for lineage covariance.

    Returns:
        Dictionary of model fits, including AIC, Delta AIC, rates, and t_MRCA.
    """
    times = np.asarray(times, dtype=np.float64)
    dists = np.asarray(dists, dtype=np.float64)
    n = len(times)
    if n < 5:
        return {'status': 'INSUFFICIENT_DATA', 'error': f'At least 5 observations required (got {n})'}

    t_min = float(np.min(times))
    t_max = float(np.max(times))
    span = max(1e-6, t_max - t_min)
    t_ref = float(np.mean(times))
    ss_tot = float(np.sum((dists - np.mean(dists))**2))

    # Compute baseline OLS if not provided
    if rss_ols is None or aic_ols is None:
        X_ols = np.column_stack([np.ones(n), times])
        b_ols, _, _, _ = la.lstsq(X_ols, dists)
        rss_ols = float(np.sum((dists - X_ols @ b_ols)**2))
        aic_ols = float(n * np.log(max(1e-12, rss_ols / n)) + 4)
    else:
        rss_ols = float(rss_ols)
        aic_ols = float(aic_ols)

    eff_n = float(min(float(n), max(3.0, n_eff))) if (n_eff is not None and n_eff < n) else None
    aic_eff_ols = float(eff_n * np.log(max(1e-12, rss_ols / eff_n)) + 4) if eff_n is not None else None

    models: Dict[str, Any] = {}

    # -------------------------------------------------------------
    # 1. Exact Quadratic (Integrated Linear Rate)
    # d(t) = b0 + b1*(t - t_ref) + b2*(t - t_ref)^2
    # mu(t) = b1 + 2*b2*(t - t_ref)
    # -------------------------------------------------------------
    t0 = time.time()
    t_scaled = times - t_ref
    X_quad = np.column_stack([np.ones(n), t_scaled, t_scaled**2])
    b_quad, _, _, _ = la.lstsq(X_quad, dists)
    pred_quad = X_quad @ b_quad
    rss_quad = float(np.sum((dists - pred_quad)**2))
    r2_quad = float(1.0 - rss_quad / max(1e-12, ss_tot))
    aic_quad = float(n * np.log(max(1e-12, rss_quad / n)) + 2 * 3)
    daic_quad = float(aic_quad - aic_ols)  # Formal definition: negative indicates improvement
    aic_red_quad = float(aic_ols - aic_quad)

    daic_eff_quad = None
    aic_eff_quad = None
    if eff_n is not None:
        aic_eff_quad = float(eff_n * np.log(max(1e-12, rss_quad / eff_n)) + 2 * 3)
        daic_eff_quad = float(aic_eff_quad - aic_eff_ols)

    mu_start_quad = float(b_quad[1] + 2 * b_quad[2] * (t_min - t_ref))
    mu_end_quad = float(b_quad[1] + 2 * b_quad[2] * (t_max - t_ref))

    # Solve d(t) = 0 for t_MRCA
    c, b, a = b_quad[0], b_quad[1], b_quad[2]
    disc = b**2 - 4 * a * c
    tmrca_quad = float('nan')
    if disc >= 0 and abs(a) > 1e-12:
        r1 = (-b - math.sqrt(disc)) / (2 * a) + t_ref
        r2 = (-b + math.sqrt(disc)) / (2 * a) + t_ref
        valid_roots = [r for r in [r1, r2] if r <= t_min + 5.0]
        if valid_roots:
            tmrca_quad = float(max(valid_roots))
    elif abs(a) <= 1e-12 and abs(b) > 1e-12:
        tmrca_quad = float(-c / b + t_ref)

    models['quadratic'] = {
        'name': 'Exact Quadratic (Linear Rate)',
        'rss': rss_quad,
        'r2': r2_quad,
        'aic': aic_quad,
        'delta_aic': daic_quad,
        'aic_reduction': aic_red_quad,
        'aic_eff': aic_eff_quad,
        'delta_aic_eff': daic_eff_quad,
        'n_eff': eff_n,
        'rate_ancestral': mu_start_quad,
        'rate_recent': mu_end_quad,
        'rate_slope': float(2 * b_quad[2]),
        't_mrca': tmrca_quad,
        'beta': [float(x) for x in b_quad],
        'elapsed_ms': round((time.time() - t0) * 1000, 2)
    }

    # -------------------------------------------------------------
    # 2. Exact Profile Log-Linear (Exponential Rate Clock)
    # mu(t) = mu_0 * exp(beta * (t - t_min))
    # d(t) = theta_0 + theta_1 * [exp(beta * (t - t_min)) - 1] / beta
    # -------------------------------------------------------------
    t0 = time.time()
    t_span = times - t_min

    def eval_beta(beta: float) -> float:
        if abs(beta) < 1e-6:
            Z = t_span
        else:
            b_clamped = np.clip(beta * t_span, -50.0, 50.0)
            Z = (np.exp(b_clamped) - 1.0) / beta
        X = np.column_stack([np.ones(n), Z])
        c_est, _, _, _ = la.lstsq(X, dists)
        return float(np.sum((dists - X @ c_est)**2))

    opt_beta = minimize_scalar(eval_beta, bounds=(-1.0, 1.0), method='bounded')
    best_beta = float(opt_beta.x)
    if abs(best_beta) < 1e-6:
        Z_opt = t_span
    else:
        Z_opt = (np.exp(np.clip(best_beta * t_span, -50.0, 50.0)) - 1.0) / best_beta
    X_exp = np.column_stack([np.ones(n), Z_opt])
    b_exp, _, _, _ = la.lstsq(X_exp, dists)
    pred_exp = X_exp @ b_exp
    rss_exp = float(np.sum((dists - pred_exp)**2))
    r2_exp = float(1.0 - rss_exp / max(1e-12, ss_tot))
    aic_exp = float(n * np.log(max(1e-12, rss_exp / n)) + 2 * 3)
    daic_exp = float(aic_exp - aic_ols)
    aic_red_exp = float(aic_ols - aic_exp)

    daic_eff_exp = None
    aic_eff_exp = None
    if eff_n is not None:
        aic_eff_exp = float(eff_n * np.log(max(1e-12, rss_exp / eff_n)) + 2 * 3)
        daic_eff_exp = float(aic_eff_exp - aic_eff_ols)

    mu_start_exp = float(b_exp[1])
    mu_end_exp = float(b_exp[1] * np.exp(np.clip(best_beta * span, -50.0, 50.0)))

    tmrca_exp = float('nan')
    if b_exp[1] > 1e-9 and best_beta != 0:
        arg = 1.0 - (best_beta * b_exp[0]) / b_exp[1]
        if arg > 0:
            tmrca_exp = float(t_min + math.log(arg) / best_beta)

    models['exponential'] = {
        'name': 'Profile Exponential (Log-Linear Rate)',
        'rss': rss_exp,
        'r2': r2_exp,
        'aic': aic_exp,
        'delta_aic': daic_exp,
        'aic_reduction': aic_red_exp,
        'aic_eff': aic_eff_exp,
        'delta_aic_eff': daic_eff_exp,
        'n_eff': eff_n,
        'rate_ancestral': mu_start_exp,
        'rate_recent': mu_end_exp,
        'beta_decay': best_beta,
        't_mrca': tmrca_exp,
        'beta': [float(x) for x in b_exp],
        'elapsed_ms': round((time.time() - t0) * 1000, 2)
    }

    # -------------------------------------------------------------
    # 3. Exact Bilinear Surge-and-Crash
    # d(t) = b0 + b1*(t - t_min) + b2*(t - t_min)^2 + gamma * max(0, t - knot)^2
    # -------------------------------------------------------------
    t0 = time.time()
    def eval_crash(k: float) -> float:
        X = np.column_stack([
            np.ones(n),
            times - t_min,
            (times - t_min)**2,
            np.maximum(0.0, times - k)**2
        ])
        c_est, _, _, _ = la.lstsq(X, dists)
        return float(np.sum((dists - X @ c_est)**2))

    opt_crash = minimize_scalar(eval_crash, bounds=(t_min + 0.1 * span, t_max - 0.1 * span), method='bounded')
    best_k = float(opt_crash.x)
    X_bi = np.column_stack([
        np.ones(n),
        times - t_min,
        (times - t_min)**2,
        np.maximum(0.0, times - best_k)**2
    ])
    b_bi, _, _, _ = la.lstsq(X_bi, dists)
    pred_bi = X_bi @ b_bi
    rss_bi = float(np.sum((dists - pred_bi)**2))
    r2_bi = float(1.0 - rss_bi / max(1e-12, ss_tot))
    aic_bi = float(n * np.log(max(1e-12, rss_bi / n)) + 2 * 4)
    daic_bi = float(aic_bi - aic_ols)
    aic_red_bi = float(aic_ols - aic_bi)

    daic_eff_bi = None
    aic_eff_bi = None
    if eff_n is not None:
        aic_eff_bi = float(eff_n * np.log(max(1e-12, rss_bi / eff_n)) + 2 * 4)
        daic_eff_bi = float(aic_eff_bi - aic_eff_ols)

    mu_start_bi = float(b_bi[1])
    mu_knot_bi = float(b_bi[1] + 2 * b_bi[2] * (best_k - t_min))
    mu_end_bi = float(b_bi[1] + 2 * b_bi[2] * span + 2 * b_bi[3] * (t_max - best_k))

    models['bilinear_crash'] = {
        'name': 'Bilinear Surge-and-Crash',
        'rss': rss_bi,
        'r2': r2_bi,
        'aic': aic_bi,
        'delta_aic': daic_bi,
        'aic_reduction': aic_red_bi,
        'aic_eff': aic_eff_bi,
        'delta_aic_eff': daic_eff_bi,
        'n_eff': eff_n,
        'rate_ancestral': mu_start_bi,
        'rate_at_knot': mu_knot_bi,
        'rate_recent': mu_end_bi,
        'knot_date': best_k,
        'beta': [float(x) for x in b_bi],
        'elapsed_ms': round((time.time() - t0) * 1000, 2)
    }

    # -------------------------------------------------------------
    # 4. Polyepoch Piecewise Constant Clock (M-Epoch NNLS)
    # -------------------------------------------------------------
    t0 = time.time()
    epoch_bounds = np.linspace(t_min, t_max, n_epochs + 1)
    Delta = np.zeros((n, n_epochs))
    for i in range(n):
        for k in range(n_epochs):
            Delta[i, k] = max(0.0, min(times[i], epoch_bounds[k+1]) - epoch_bounds[k])
    X_pe = np.column_stack([np.ones(n), Delta])
    b_pe, _ = nnls(X_pe, dists)
    pred_pe = X_pe @ b_pe
    rss_pe = float(np.sum((dists - pred_pe)**2))
    r2_pe = float(1.0 - rss_pe / max(1e-12, ss_tot))
    aic_pe = float(n * np.log(max(1e-12, rss_pe / n)) + 2 * (n_epochs + 1))
    daic_pe = float(aic_pe - aic_ols)
    aic_red_pe = float(aic_ols - aic_pe)

    daic_eff_pe = None
    aic_eff_pe = None
    if eff_n is not None:
        aic_eff_pe = float(eff_n * np.log(max(1e-12, rss_pe / eff_n)) + 2 * (n_epochs + 1))
        daic_eff_pe = float(aic_eff_pe - aic_eff_ols)

    models['polyepoch'] = {
        'name': f'Polyepoch ({n_epochs}-Epoch NNLS)',
        'rss': rss_pe,
        'r2': r2_pe,
        'aic': aic_pe,
        'delta_aic': daic_pe,
        'aic_reduction': aic_red_pe,
        'aic_eff': aic_eff_pe,
        'delta_aic_eff': daic_eff_pe,
        'n_eff': eff_n,
        'epoch_boundaries': [round(float(b), 4) for b in epoch_bounds],
        'epoch_rates': [float(r) for r in b_pe[1:]],
        'elapsed_ms': round((time.time() - t0) * 1000, 2)
    }

    return models


def print_dudas_models_table(dudas_res: Dict[str, Any]) -> None:
    """Prints a clean CLI summary table for non-linear and time-varying extended clock models."""
    sample_model = next((m for m in dudas_res.values() if isinstance(m, dict) and 'delta_aic' in m), None)
    has_neff = sample_model is not None and sample_model.get('delta_aic_eff') is not None
    n_eff_val = sample_model.get('n_eff') if has_neff else None

    if has_neff and n_eff_val is not None:
        col_neff_header = f"ΔAIC (N_eff={int(round(n_eff_val))})"
        print("\n" + "=" * 115)
        print(f"{'Non-Linear / Time-Varying Model':<32} {'t_MRCA':<8} {'Rate Dynamic':<22} {'Raw AIC':<11} {'ΔAIC vs OLS':<13} {col_neff_header:<18} {'R^2':<6}")
        print("-" * 115)
    else:
        print("\n" + "=" * 110)
        print(f"{'Non-Linear / Time-Varying Model':<32} {'t_MRCA':<8} {'Rate Dynamic':<22} {'Raw AIC':<11} {'ΔAIC vs OLS':<13} {'AIC Drop':<10} {'R^2':<6}")
        print("-" * 110)

    for key in ['quadratic', 'exponential', 'bilinear_crash', 'polyepoch']:
        if key not in dudas_res or not isinstance(dudas_res[key], dict):
            continue
        m = dudas_res[key]
        name = m.get('name', key)
        tmrca_str = f"{m['t_mrca']:>7.2f}" if ('t_mrca' in m and not math.isnan(m['t_mrca'])) else f"{'n/a':>7}"
        if key == 'quadratic':
            dyn_str = f"linear ({m['rate_ancestral']:.2e}->{m['rate_recent']:.2e})"
        elif key == 'exponential':
            dyn_str = f"exp (β={m['beta_decay']:.3f})"
        elif key == 'bilinear_crash':
            dyn_str = f"knot={m['knot_date']:.2f}"
        elif key == 'polyepoch':
            rates_sub = ",".join([f"{r:.1e}" for r in m['epoch_rates']])
            dyn_str = f"epochs [{rates_sub}]"
        else:
            dyn_str = "-"

        daic = m['delta_aic']
        if has_neff and m.get('delta_aic_eff') is not None:
            daic_eff = m['delta_aic_eff']
            print(f"{name:<32} {tmrca_str}  {dyn_str:<22} {m['aic']:>10.2f} {daic:>+11.2f}  {daic_eff:>+16.2f}    {m['r2']:>5.3f}")
        else:
            aic_drop = m.get('aic_reduction', -daic)
            print(f"{name:<32} {tmrca_str}  {dyn_str:<22} {m['aic']:>10.2f} {daic:>+11.2f}  {aic_drop:>+8.2f}    {m['r2']:>5.3f}")

    if has_neff:
        print("=" * 115)
        print("  Note: Formal ΔAIC = AIC_model - AIC_OLS (negative values indicate lower / superior information criteria).")
        print("        Lineage-adjusted ΔAIC (N_eff) rescales degrees of freedom to protect against phylogenetic pseudoreplication.")
    else:
        print("=" * 110)
        print("  Note: Formal ΔAIC = AIC_model - AIC_OLS (negative values indicate lower / superior information criteria).")
        print("        AIC Drop = points dropped below linear baseline (positive values indicate improvement).")


# Modern aliases
evaluate_nonlinear_clock_models = evaluate_dudas_clock_models
print_nonlinear_models_table = print_dudas_models_table
