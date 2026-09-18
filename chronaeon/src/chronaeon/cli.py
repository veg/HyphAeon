"""
chronaeon/cli.py
----------------
Command-line interface for ChronAeon: molecular clock dating,
phylodynamics, and genomic surveillance.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aeon_core.inference import get_device
from aeon_core.weights import DEFAULT_VARIANT, print_available_variants

DEFAULT_VARIANT_ENV = os.environ.get("CHRONAEON_VARIANT", os.environ.get("HYPHAEON_VARIANT", DEFAULT_VARIANT))  # HYPHAEON_* fallback for pre-refactor users

_local_repo_weights = Path(__file__).resolve().parent.parent.parent.parent / "model.safetensors"
DEFAULT_WEIGHTS_ENV = os.environ.get("CHRONAEON_WEIGHTS", os.environ.get("HYPHAEON_WEIGHTS", str(_local_repo_weights) if _local_repo_weights.exists() else None))  # HYPHAEON_* fallback for pre-refactor users

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXAMPLES_DIR = _REPO_ROOT / "chronaeon" / "examples"


def cmd_dating(args):
    """Executes Heterochronous Molecular Clock Calibration and Ancestor Dating (t_MRCA)."""
    from .dating import run_mrca_dating, plot_mrca_dating

    alignment = getattr(args, "alignment", None)
    beast_file = getattr(args, "beast", None)
    if not alignment and not beast_file:
        print("[!] Error: You must provide either an alignment file (-a/--alignment) or a BEAST XML configuration file (--beast).")
        sys.exit(1)

    use_tn93 = getattr(args, "no_tree", False) or getattr(args, "use_tn93", False) or (getattr(args, "tree", None) == "tn93")
    device = get_device(cpu=getattr(args, "cpu", False))
    print(f"[*] Hardware device selected: {device.type.upper()}")

    plot_requested = getattr(args, "plot", False) or (getattr(args, "plot_path", None) is not None)
    ci_method = getattr(args, "ci_method", "fieller")

    res = run_mrca_dating(
        alignment_path=alignment,
        tree_path=getattr(args, "tree", None),
        dates_source=getattr(args, "dates", None),
        date_col=getattr(args, "date_col", None),
        strain_col=getattr(args, "strain_col", None),
        date_regex=getattr(args, "date_regex", None),
        root_taxon=getattr(args, "root_taxon", None),
        decay_gamma=getattr(args, "decay_gamma", None),
        optimize_root=not getattr(args, "no_optimize_root", False),
        use_tn93=use_tn93,
        distance_mode=getattr(args, "distance_mode", "auto"),
        method=getattr(args, "method", "all"),
        clock_model=getattr(args, "clock_model", "auto"),
        ci_method=ci_method,
        ridge=getattr(args, "ridge", "auto"),
        n_bootstrap=getattr(args, "bootstrap", 1000),
        weights=getattr(args, "weights", DEFAULT_WEIGHTS_ENV),
        variant=getattr(args, "variant", DEFAULT_VARIANT_ENV),
        device=device,
        batch_size=getattr(args, "batch_size", None),
        max_species=getattr(args, "max_species", None),
        loocv=getattr(args, "loocv", False),
        output_prefix=getattr(args, "output", None),
        plot=plot_requested,
        beast_path=beast_file,
    )

    def format_ci_str(ci, mu):
        if ci is None or np.isnan(ci[0]) or np.isnan(ci[1]) or mu <= 0:
            return "[rate <= 0]" if mu <= 0 else "[n/a]"
        if np.isneginf(ci[0]):
            return f"[-inf, {ci[1]:.1f}]"
        return f"[{ci[0]:.1f}, {ci[1]:.1f}]"

    ci_label = f"95% CI ({res.get('ci_method', 'Fieller').capitalize()})"
    print("\n" + "=" * 105)
    print(f"{'Method / Estimator':<36} {'Estimated t_MRCA':<20} {ci_label:<26} {'Rate (μ / year)':<18} {'R^2':<6}")
    print("-" * 105)

    ols = res['ols']
    ols_t0_str = f"{'n/a':>7}" if (np.isnan(ols['t_mrca']) or ols['mu'] <= 0) else f"{ols['t_mrca']:>7.2f}"
    ci_ols_str = format_ci_str(ols['ci_mrca'], ols['mu'])
    print(f"{'1. Standard OLS (TempEst RTT)':<36} {ols_t0_str}            {ci_ols_str:<26} {ols['mu']:>11.6f}      {ols['r2']:>5.3f}")

    curr_idx = 2
    if res.get('pgls'):
        pgls = res['pgls']
        pgls_t0_str = f"{'n/a':>7}" if (np.isnan(pgls['t_mrca']) or pgls['mu'] <= 0) else f"{pgls['t_mrca']:>7.2f}"
        ci_pgls_str = format_ci_str(pgls['ci_mrca'], pgls['mu'])
        print(f"{f'{curr_idx}. HyphAeon Attention PGLS':<36} {pgls_t0_str}            {ci_pgls_str:<26} {pgls['mu']:>11.6f}      {pgls['r2']:>5.3f}")
        curr_idx += 1

    if res.get('spline'):
        sp = res['spline']
        if np.isnan(sp['t_mrca']) or sp['rate_ancestral'] <= 0:
            sp_t0_str = f"{'n/a':>7}"
            ci_sp_str = f"{'[non-pos rate]':<26}"
        else:
            sp_t0_str = f"{sp['t_mrca']:>7.2f}"
            ci_sp_str = f"[{sp['ci_mrca'][0]:.1f}, {sp['ci_mrca'][1]:.1f}]"
        sp_label = f"{curr_idx}. Restricted Spline Clock"
        print(f"{sp_label:<36} {sp_t0_str}            {ci_sp_str:<26} {sp['rate_ancestral']:>11.6f}      {sp['r2']:>5.3f}")
        curr_idx += 1

    if res.get('power'):
        pwr = res['power']
        ci_pwr_str = f"[{pwr['ci_mrca'][0]:.1f}, {pwr['ci_mrca'][1]:.1f}]"
        pwr_label = f"{curr_idx}. Power-Law Clock (θ={pwr['theta']:.3f})"
        print(f"{pwr_label:<36} {pwr['t_mrca']:>7.2f}            {ci_pwr_str:<26} {pwr['rate_mean']:>11.6f}      {pwr['r2']:>5.3f}")
        curr_idx += 1

    print("-" * 105)
    print(f"[*] Selected Clock Model: {res.get('selected_clock', 'Linear')}")
    if res.get('ensemble') and res['ensemble'].get('t_mrca') is not None:
        ens = res['ensemble']
        weights_str = ", ".join([f"{k.upper()}={v*100:.1f}%" for k, v in ens.get('weights', {}).items()])
        ci_ens = ens.get('ci_mrca')
        ci_ens_str = f" [{ci_ens[0]:.1f}, {ci_ens[1]:.1f}]" if (ci_ens and not np.isnan(ci_ens[0]) and not np.isnan(ci_ens[1])) else ""
        print(f"[*] Model-Averaged Ensemble: t_MRCA = {ens['t_mrca']:.2f}{ci_ens_str} (Weights: {weights_str})")
    if res['ols'].get('fieller_g') is not None:
        g_ols = res['ols']['fieller_g']
        g_status = "significant temporal signal" if g_ols < 1.0 else "unbounded (slope p >= 0.05)"
        print(f"    OLS Fieller g-statistic: g = {g_ols:.4f} ({g_status})")
    if res.get('pgls') and res['pgls'].get('fieller_g') is not None:
        g_pgls = res['pgls']['fieller_g']
        g_status = "significant temporal signal" if g_pgls < 1.0 else "unbounded (slope p >= 0.05)"
        print(f"    PGLS Fieller g-statistic: g = {g_pgls:.4f} ({g_status})")
    if res.get('spline'):
        sp = res['spline']
        p_str = f"p={sp['p_f_test']:.4f}" if sp['p_f_test'] >= 0.0001 else "p<0.0001"
        ratio_sym = "acceleration" if sp['rate_ratio'] > 1.0 else "deceleration"
        sp_drop = sp.get('aic_reduction', -sp['delta_aic'])
        print(f"    Restricted Spline (2 DF): μ_anc = {sp['rate_ancestral']:.6f}, μ_rec = {sp['rate_recent']:.6f} ({ratio_sym} {sp['rate_ratio']:.2f}x) | F = {sp['f_stat']:.3f} ({p_str}) | ΔAIC = {sp['delta_aic']:+.2f} (AIC drop: {sp_drop:.2f})")
    if res.get('power'):
        pwr = res['power']
        p_str = f"p={pwr['p_f_test']:.4f}" if pwr['p_f_test'] >= 0.0001 else "p<0.0001"
        pwr_drop = pwr.get('aic_reduction', -pwr['delta_aic'])
        print(f"    Power-Law Curvature: θ = {pwr['theta']:.3f} [{pwr['ci_theta'][0]:.3f}, {pwr['ci_theta'][1]:.3f}] | F = {pwr['f_stat']:.3f} ({p_str}) | ΔAIC = {pwr['delta_aic']:+.2f} (AIC drop: {pwr_drop:.2f})")

    if getattr(args, "nonlinear_clocks", False) or getattr(args, "dudas_models", False):
        from .dudas import evaluate_dudas_clock_models, print_dudas_models_table
        d_times = res.get('times')
        d_dists = res.get('dists')
        if d_times is not None and d_dists is not None:
            rss_o = res.get('ols', {}).get('rss') if isinstance(res.get('ols'), dict) else None
            aic_o = res.get('ols', {}).get('aic') if isinstance(res.get('ols'), dict) else None
            n_eff_cand = None
            if res.get('pgls') and 'n_eff' in res['pgls']:
                n_eff_cand = res['pgls']['n_eff']
            elif 'n_eff' in res:
                n_eff_cand = res['n_eff']

            nonlinear_res = evaluate_dudas_clock_models(
                times=d_times,
                dists=d_dists,
                rss_ols=rss_o,
                aic_ols=aic_o,
                n_eff=n_eff_cand
            )
            res['nonlinear_clocks'] = nonlinear_res
            res['dudas_models'] = nonlinear_res
            print_dudas_models_table(nonlinear_res)

            output_prefix = getattr(args, "output", None)
            if output_prefix:
                out_p = Path(output_prefix)
                json_file = out_p.with_suffix('.json') if not str(out_p).endswith('.json') else out_p
                if json_file.exists():
                    try:
                        import json
                        with open(json_file, 'r') as jf:
                            jdata = json.load(jf)
                        jdata['nonlinear_clocks'] = nonlinear_res
                        jdata['dudas_models'] = nonlinear_res
                        with open(json_file, 'w') as jf:
                            json.dump(jdata, jf, indent=2)
                    except Exception:
                        pass

    if res.get('loocv'):
        lv = res['loocv']
        print("\n" + "=" * 105)
        print(f"Leave-One-Out Cross-Validation (LOOCV) & Jackknife Uncertainty ({lv['method']}, N = {lv['n_taxa']} taxa)")
        print("-" * 105)
        print("Out-of-Sample Tip Date Recovery:")
        if not np.isnan(lv.get('tip_mae_days', np.nan)):
            print(f"  • Mean Absolute Error (MAE):          {lv['tip_mae_days']:>7.1f} days ({lv['tip_mae_years']:.4f} yr)")
            print(f"  • Root Mean Squared Error (RMSE):     {lv['tip_rmse_days']:>7.1f} days ({lv['tip_rmse_years']:.4f} yr)")
            print(f"  • Median Absolute Error:              {lv['tip_median_abs_error_days']:>7.1f} days")
            print(f"  • 95% Tip Predictive Interval Width:  {lv['tip_pred_ci_95_days']:>7.1f} days ({lv['tip_pred_ci_95_years']:.4f} yr, ±{lv['tip_pred_ci_95_days']/2:.1f} d)")
            print(f"  • Empirical 95% Error Bound:          {lv['tip_empirical_95_days']:>7.1f} days")
            if not np.isnan(lv.get('tip_r2_pred', np.nan)):
                print(f"  • Out-of-Sample Predictive R^2:       {lv['tip_r2_pred']:>7.3f}")
        else:
            print("  • Notice: Tip date inversion non-computable (rate <= 0).")

        print("\nJackknife Root (t_MRCA) Uncertainty vs. Analytical Intervals:")
        if not np.isnan(lv.get('jackknife_mean', np.nan)):
            act_t0 = res.get('t_mrca')
            act_t0_str = f"{act_t0:.2f}" if (act_t0 is not None and not np.isnan(act_t0)) else "n/a"
            act_model_name = str(res.get('active_model', 'active')).lower()
            ols_t0 = res.get('ols', {}).get('t_mrca', np.nan) if isinstance(res.get('ols'), dict) else np.nan

            if act_model_name in ['spline', 'restricted spline', 'power', 'power-law']:
                print(f"  • Selected Clock Model ({res.get('active_model', 'Spline').capitalize()}):  {act_t0_str}")
                if not np.isnan(ols_t0):
                    print(f"  • Linear {lv['method']} Full-Sample t_MRCA:        {ols_t0:.2f}")
                print(f"  • Linear {lv['method']} Jackknife Mean t_MRCA:   {lv['jackknife_mean']:.2f}")
                print(f"  • Linear {lv['method']} Jackknife SE(t_MRCA):     {lv['jackknife_se_years']:.4f} yr ({lv['jackknife_se_days']:.1f} days)")
                ci_j = lv['jackknife_ci']
                print(f"  • Linear {lv['method']} Jackknife 95% CI:         [{ci_j[0]:.2f}, {ci_j[1]:.2f}] (Width: {lv['jackknife_ci_width_years']:.3f} yr / {lv['jackknife_ci_width_days']:.1f} days)")
            else:
                print(f"  • Full-Sample Estimated t_MRCA ({lv['method']}):      {act_t0_str}")
                print(f"  • Jackknife Mean t_MRCA ({lv['method']}):             {lv['jackknife_mean']:.2f}")
                print(f"  • Jackknife SE(t_MRCA):                    {lv['jackknife_se_years']:.4f} yr ({lv['jackknife_se_days']:.1f} days)")
                ci_j = lv['jackknife_ci']
                print(f"  • Jackknife 95% CI:                        [{ci_j[0]:.2f}, {ci_j[1]:.2f}] (Width: {lv['jackknife_ci_width_years']:.3f} yr / {lv['jackknife_ci_width_days']:.1f} days)")

            if 'fieller_ci_width_years' in lv and not np.isnan(lv['fieller_ci_width_years']):
                ratio_val = lv.get('fieller_to_jackknife_ratio', np.nan)
                ratio_str = f"{ratio_val:.2f}x" if not np.isnan(ratio_val) else "n/a"
                act_ci = res.get('ci_mrca')
                act_ci_str = f"[{act_ci[0]:.2f}, {act_ci[1]:.2f}]" if act_ci else "n/a"
                print(f"  • Fieller Analytical 95% CI:               {act_ci_str} (Width: {lv['fieller_ci_width_years']:.3f} yr / {lv['fieller_ci_width_days']:.1f} days)")
                print(f"  • Fieller-to-Jackknife Width Ratio:        {ratio_str} (Non-parametric empirical calibration)")
            print(f"  • Total Jackknife Root Spread:             {lv['jackknife_spread_days']:.1f} days across leave-one-out iterations")
        else:
            print("  • Notice: Jackknife root estimates non-computable (rate <= 0).")

    if res.get('latent_root'):
        lr = res['latent_root']
        print(f"\n[*] Continuous Latent Convex Hull Root Optimization:")
        print(f"    Isometric Calibration Scale: α = {lr['alpha']:.6f} subs/site per latent unit")
        print(f"    Latent Temporal Alignment:  R = {lr['temporal_r']:+.4f} (R^2 = {lr['temporal_r2']:.3f})")
        print(f"    Top Ancestral Anchor Taxa in Convex Hull:")
        for anc in lr['anchor_taxa'][:5]:
            print(f"      • {anc['taxon']:<25} (Weight: {anc['weight']*100:>5.1f}%, Sampling Date: {anc['date']:.1f})")

    holdouts = [r for r in res['taxa_records'] if r.get('is_holdout', False)]
    if holdouts:
        print(f"\n[*] Out-of-Sample Holdout / Validation Predictions:")
        for h in holdouts:
            if not np.isnan(h['predicted_date']):
                print(f"    • {h['taxon']}: Sampling Date={h['sampling_date']:.1f}, Predicted Date={h['predicted_date']:.1f} (Discrepancy: {h['temporal_residual']:+.2f} yr)")
            else:
                print(f"    • {h['taxon']}: Sampling Date={h['sampling_date']:.1f}, Root Div={h['root_divergence']:.4f}")

    outliers = [r for r in res['taxa_records'] if r['is_outlier']]
    if outliers:
        print(f"\n[*] Flagged Temporal Outliers (|Z| >= 2.5, possible latent proviruses, archival isolates, or lab artifacts):")
        for o in outliers[:10]:
            if np.isnan(o['predicted_date']):
                print(f"    • {o['taxon']}: Sampling Date={o['sampling_date']:.1f}, Root Div={o['root_divergence']:.4f} (Residual Z={o['z_score']:+.2f}, Rate <= 0)")
            else:
                print(f"    • {o['taxon']}: Sampling Date={o['sampling_date']:.1f}, Predicted Date={o['predicted_date']:.1f} (Discrepancy: {o['temporal_residual']:+.2f} yr, Z={o['z_score']:+.2f})")
        if len(outliers) > 10:
            print(f"    ... and {len(outliers) - 10} more (see CSV for full table).")

    if getattr(args, "csv", None):
        csv_path = args.csv
        pd.DataFrame(res['taxa_records']).to_csv(csv_path, index=False)
        print(f"[✓] Per-taxon dating diagnostics saved to: {csv_path}")

    plot_path = getattr(args, "plot_path", None)
    if plot_path:
        plot_mrca_dating(res, plot_path)

    alluvial_requested = getattr(args, "alluvial", False) or (getattr(args, "alluvial_path", None) is not None)
    if alluvial_requested:
        alluvial_path = getattr(args, "alluvial_path", None)
        if not alluvial_path:
            if getattr(args, "output", None):
                base = os.path.splitext(args.output)[0]
                alluvial_path = f"{base}_alluvial.png"
            elif getattr(args, "alignment", None):
                base = os.path.splitext(args.alignment)[0]
                alluvial_path = f"{base}_alluvial.png"
            else:
                alluvial_path = "chronaeon_alluvial_phylogeny.png"
        from .dating import plot_alluvial_phylogeny
        plot_alluvial_phylogeny(
            dating_results=res,
            output_path=alluvial_path,
            alignment_path=alignment,
            tree_path=getattr(args, "tree", None),
            dates_source=getattr(args, "dates", None),
            color_by=getattr(args, "color_by", None),
        )


def cmd_phylogeo(args):
    """Executes Fast Discrete Phylogeography & Spatial Transmission Network Inference."""
    from .geo import run_phylogeography_analysis, plot_geo_diagnostics
    device = get_device(cpu=getattr(args, "cpu", False))
    print(f"[*] Hardware device selected: {device.type.upper()}")

    # Handle built-in worked example
    if getattr(args, "example", False) or (args.alignment is None and args.metadata is None):
        example_aln = EXAMPLES_DIR / "H5N1_HA_geo.fasta"
        example_meta = EXAMPLES_DIR / "H5N1_HA_metadata.csv"
        example_tree = EXAMPLES_DIR / "H5N1_HA.nwk"
        if not example_aln.exists() or not example_meta.exists():
            raise FileNotFoundError(
                f"Built-in example files not found at {EXAMPLES_DIR}. "
                f"Please provide -a/--alignment and -g/--metadata explicitly."
            )
        print(f"[*] Executing built-in worked example: Avian Influenza A (H5N1) Dispersal (Lemey et al. 2009)...")
        print(f"    • Alignment: {example_aln.name} (98 taxa, 1698 nt)")
        print(f"    • Metadata:  {example_meta.name} (7 Chinese provinces)")
        print(f"    • Phylogeny: {example_tree.name}")
        args.alignment = str(example_aln)
        args.metadata = str(example_meta)
        if getattr(args, "tree", None) is None:
            args.tree = str(example_tree)
        if getattr(args, "plot_path", None) is None and not getattr(args, "plot", False):
            args.plot_path = "h5n1_geo_example.png"
            args.plot = True
        if getattr(args, "geojson", None) is None:
            args.geojson = "h5n1_geo_example.geojson"
    elif not args.alignment or not args.metadata:
        raise ValueError(
            "Both -a/--alignment and -g/--metadata are required to run phylogeography "
            "(or use --example to execute the built-in worked benchmark example)."
        )

    plot_requested = getattr(args, "plot", False) or (getattr(args, "plot_path", None) is not None)

    res = run_phylogeography_analysis(
        alignment_path=args.alignment,
        metadata_path=args.metadata,
        tree_path=getattr(args, "tree", None),
        location_col=getattr(args, "location_col", None),
        strain_col=getattr(args, "strain_col", None),
        date_col=getattr(args, "date_col", None),
        lat_col=getattr(args, "lat_col", None),
        lon_col=getattr(args, "lon_col", None),
        root_taxon=getattr(args, "root_taxon", None),
        n_perms=getattr(args, "n_perms", 1000),
        min_bf=getattr(args, "min_bf", 3.0),
        fdr_thresh=getattr(args, "fdr", 0.10),
        use_neural=not getattr(args, "no_neural", False),
        weights=getattr(args, "weights", DEFAULT_WEIGHTS_ENV),
        variant=getattr(args, "variant", DEFAULT_VARIANT_ENV),
        device=device,
        output_prefix=getattr(args, "output", None),
        geojson_path=getattr(args, "geojson", None),
        plot=plot_requested,
        plot_path=getattr(args, "plot_path", None),
    )

    print("\n" + "=" * 105)
    print(f"{'Rank':<6} {'Geographic Region':<24} {'Posterior P(Root)':<22} {'Isolates':<12} {'Role / Dynamics':<24}")
    print("-" * 105)

    root_dist = res["root_epicenter"]["distribution"]
    sorted_locs = sorted(root_dist.items(), key=lambda x: -x[1])
    sample_counts = dict(zip(res["unique_locations"], res["sample_counts"]))

    for rank, (loc, p) in enumerate(sorted_locs, 1):
        cnt = sample_counts.get(loc, 0)
        role = res["hub_metrics"][loc]["role"]
        star = " ★ EPICENTER" if rank == 1 and p >= 0.5 else ""
        print(f"{rank:<6} {loc:<24} {p:>8.4f}              {cnt:>4}         {role:<20}{star}")
    print("-" * 105)

    sig_routes = res["significant_routes"]
    print(f"\n[*] Statistically Supported Transmission Routes (BF >= {getattr(args, 'min_bf', 3.0):.1f} or FDR <= {getattr(args, 'fdr', 0.10):.2f}):")
    if sig_routes:
        print(f"{'Source':<16} {'Target (Sink)':<16} {'Flux':<12} {'Z-Score':<10} {'p-value':<10} {'FDR q':<10} {'Bayes Factor':<14} {'Support':<16}")
        print("-" * 105)
        for r in sig_routes[:20]:
            print(f"{r['source']:<16} {r['target']:<16} {r['rate']:>8.5f}    {r['z_score']:>7.2f}   {r['p_value']:>8.4f}  {r['fdr_q']:>8.4f}  {r['bayes_factor']:>9.1f}     {r['support']}")
        if len(sig_routes) > 20:
            print(f"    ... and {len(sig_routes) - 20} more routes (see JSON / CSV for full list).")
    else:
        print("    No routes passed the specified Bayes Factor / FDR thresholds.")

    if getattr(args, "csv", None):
        csv_path = args.csv
        pd.DataFrame(res["routes"]).to_csv(csv_path, index=False)
        print(f"[✓] Transmission routes saved to: {csv_path}")


def cmd_dynamics(args):
    """Executes Fast Phylodynamic Estimation of Growth Rate (r) and Reproduction Numbers (R0, Rt)."""
    from .r0 import run_r0_analysis, plot_r0_diagnostics, PATHOGEN_PRESETS

    # Handle built-in worked example
    if getattr(args, "example", False) or (args.alignment is None and getattr(args, "tree", None) is None):
        example_aln = EXAMPLES_DIR / "H1N1_2009_pandemic.fasta"
        example_tree = EXAMPLES_DIR / "H1N1_2009_pandemic.nwk"
        if not example_aln.exists() or not example_tree.exists():
            raise FileNotFoundError(
                f"Built-in example files not found at {EXAMPLES_DIR}. "
                f"Please provide -a/--alignment or -t/--tree explicitly."
            )
        print(f"[*] Executing built-in worked example: 2009 Pandemic Influenza A (H1N1) Origin (Fraser et al. 2009)...")
        print(f"    • Alignment: {example_aln.name} (100 taxa, whole genome coding segments)")
        print(f"    • Phylogeny: {example_tree.name} (time-scaled tree)")
        args.alignment = str(example_aln)
        args.tree = str(example_tree)
        if getattr(args, "pathogen", None) is None:
            args.pathogen = "h1n1"
        if getattr(args, "plot_path", None) is None and not getattr(args, "plot", False):
            args.plot_path = "h1n1_r0_example.png"
            args.plot = True

    plot_requested = getattr(args, "plot", False) or (getattr(args, "plot_path", None) is not None)

    res = run_r0_analysis(
        alignment_path=args.alignment,
        tree_path=getattr(args, "tree", None),
        metadata_path=getattr(args, "metadata", None),
        pathogen=getattr(args, "pathogen", None),
        generation_time=getattr(args, "generation_time", None),
        generation_sd=getattr(args, "generation_sd", None),
        latent_time=getattr(args, "latent_time", None),
        units=getattr(args, "units", "days"),
        date_col=getattr(args, "date_col", None),
        strain_col=getattr(args, "strain_col", None),
        mu_prior=getattr(args, "mu_prior", None),
        window_size=getattr(args, "window_size", None),
        step_size=getattr(args, "step_size", None),
    )

    if plot_requested:
        plot_out = getattr(args, "plot_path", None) or "r0_diagnostics.png"
        plot_r0_diagnostics(res, output_path=plot_out)

    if getattr(args, "output", None):
        out_path = args.output
        clean_res = {
            "status": res["status"],
            "pathogen": res["pathogen"],
            "n_taxa": res["n_taxa"],
            "n_coalescent_events": res["n_coalescent_events"],
            "t_mrca": res["t_mrca"],
            "clock_rate_mu": res["clock_rate_mu"],
            "growth_rate": res["growth_rate"],
            "reproduction_number": res["reproduction_number"],
            "rt_skyline": res["rt_skyline"],
        }
        with open(out_path, "w") as f:
            json.dump(clean_res, f, indent=2)
        print(f"[✓] Phylodynamic R0 results saved to: {out_path}")

    if getattr(args, "csv", None) and len(res.get("rt_skyline", [])) > 0:
        csv_path = args.csv
        pd.DataFrame(res["rt_skyline"]).to_csv(csv_path, index=False)
        print(f"[✓] Dynamic R(t) skyline saved to: {csv_path}")


def cmd_triage(args):
    from .triage import ChronAeonSieve
    from aeon_core.dataset import parse_alignment_sequences

    print("=" * 80)
    print("CHRONAEON SIEVE: HIGH-THROUGHPUT QUALITY CONTROL & CLOCK TRIAGE")
    print("=" * 80)

    # 1. Build Anchor Harness
    sieve = ChronAeonSieve.build_from_alignment(
        alignment_path=args.alignment,
        dates_path=args.dates,
        date_col=getattr(args, "date_col", None),
        strain_col=getattr(args, "strain_col", None),
        n_anchor=args.n_anchor,
        n_temporal_bins=args.n_bins,
        root_taxon=args.root_taxon,
        tolerance_days=args.tolerance_days,
        z_threshold=args.z_threshold,
        max_ambig_ratio=getattr(args, "max_ambig", 0.05)
    )

    # 2. Screen Streaming Sequences
    if not args.stream:
        print("\n[!] No streaming sequences provided (--stream). Anchor harness built and verified.")
        return

    print(f"\n[*] Streaming sequences from: {args.stream}")
    df_results = sieve.screen_stream(
        stream_fasta=args.stream,
        stream_dates=getattr(args, "stream_dates", None),
        date_col=getattr(args, "date_col", None),
        strain_col=getattr(args, "strain_col", None),
        align_minimap2=getattr(args, "align_minimap2", False)
    )

    # 3. Print Summary
    n_tot = len(df_results)
    n_pass = int((df_results['status'] == 'PASS').sum())
    n_sus = int((df_results['status'] == 'SUS').sum())
    n_flag = int((df_results['status'] == 'FLAG_MISSING_DATE').sum())
    mean_ms = float(df_results['elapsed_ms'].mean())

    print("\n" + "=" * 80)
    print(f"SIEVE TRIAGE SUMMARY (N={n_tot})")
    print("-" * 80)
    print(f"  • PASS (Analysis-Ready)  : {n_pass:>6} ({n_pass/max(1,n_tot)*100:>5.1f}%)")
    print(f"  • SUS (Quarantined)      : {n_sus:>6} ({n_sus/max(1,n_tot)*100:>5.1f}%)")
    if n_flag > 0:
        print(f"  • MISSING DATE           : {n_flag:>6} ({n_flag/max(1,n_tot)*100:>5.1f}%)")
    print(f"  • Throughput             : {1000.0/max(0.01, mean_ms):.1f} seq/s ({mean_ms:.2f} ms/seq)")

    if n_sus > 0:
        print("\nBreakdown of Quarantined Anomalies:")
        sus_df = df_results[df_results['status'] == 'SUS']
        categories = sus_df['sus_reason'].str.split().str[0].value_counts()
        for cat, cnt in categories.items():
            print(f"    - {cat:<28}: {cnt:>5}")

    # 4. Save Outputs
    out_csv = args.output or "sieve_triage_report.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n[✓] Diagnostic triage report written to: {out_csv}")

    # Optionally write segregated FASTAs
    if getattr(args, "clean_out", None) or getattr(args, "sus_out", None):
        stream_dict = parse_alignment_sequences(str(args.stream))
        if getattr(args, "clean_out", None):
            clean_ids = set(df_results[df_results['status'] == 'PASS']['query_id'])
            with open(args.clean_out, "w") as f:
                for qid in clean_ids:
                    if qid in stream_dict:
                        f.write(f">{qid}\n{stream_dict[qid]}\n")
            print(f"[✓] Clean analysis-ready FASTA written to: {args.clean_out}")

        if getattr(args, "sus_out", None):
            sus_ids = set(df_results[df_results['status'] == 'SUS']['query_id'])
            with open(args.sus_out, "w") as f:
                for qid in sus_ids:
                    if qid in stream_dict:
                        f.write(f">{qid}\n{stream_dict[qid]}\n")
            print(f"[✓] Quarantined anomalies FASTA written to: {args.sus_out}")


def cmd_autoclock(args):
    """Subcommand handler for ChronAeon AutoClock (unsupervised multi-clock community deconvolution)."""
    import shutil
    from .autoclock import AutoClockDeconvolution

    print("=" * 80)
    print("CHRONAEON AUTOCLOCK: UNSUPERVISED MULTI-CLOCK COMMUNITY DECONVOLUTION")
    print("=" * 80)

    alignment = getattr(args, "alignment", None)
    beast_file = getattr(args, "beast", None)
    dates_source = getattr(args, "dates", None)
    if beast_file:
        if not alignment:
            alignment = beast_file
        if not dates_source:
            dates_source = beast_file

    if not alignment:
        print("[!] Error: You must provide either an alignment file (-a/--alignment) or a BEAST XML configuration file (--beast).")
        sys.exit(1)

    is_hierarchical = getattr(args, "hierarchical", False)
    if is_hierarchical:
        print("[*] Hierarchical recursive mode enabled (--hierarchical).")
        from .autoclock import HierarchicalAutoClock
        engine = HierarchicalAutoClock(
            alignment_path=alignment,
            dates_source=dates_source,
            date_col=getattr(args, "date_col", None),
            strain_col=getattr(args, "strain_col", None),
            date_regex=getattr(args, "date_regex", None),
            max_depth=getattr(args, "max_depth", 3),
            min_leaf_size=getattr(args, "min_leaf_size", 25),
            min_delta_aicc=getattr(args, "min_delta_aicc", 15.0),
            max_k_per_node=getattr(args, "max_k", 6),
            manifold=getattr(args, "manifold", "transformer"),
            weights=getattr(args, "weights", None),
            device=getattr(args, "device", None),
            kernel_bandwidth=getattr(args, "kernel_bandwidth", None),
            output_dir=getattr(args, "output_dir", None) or (Path(args.output).parent if getattr(args, "output", None) else None),
            allow_stop_codons=not getattr(args, "disallow_stop_codons", False),
            quiet=getattr(args, "quiet", False),
            n_landmarks=getattr(args, "n_landmarks", "auto"),
            max_memory_mb=getattr(args, "max_memory_mb", 1024.0),
            rooting_mode=getattr(args, "rooting_mode", "convex_decay"),
            contemporaneous_dyads=getattr(args, "contemporaneous_dyads", False),
            dyad_max_days=getattr(args, "dyad_max_days", 90.0),
            dyad_max_dist=getattr(args, "dyad_max_dist", 0.010),
        )
    else:
        engine = AutoClockDeconvolution(
            alignment_path=alignment,
            dates_source=dates_source,
            date_col=getattr(args, "date_col", None),
            strain_col=getattr(args, "strain_col", None),
            date_regex=getattr(args, "date_regex", None),
            max_k=getattr(args, "max_k", 6),
            manifold=getattr(args, "manifold", "transformer"),
            weights=getattr(args, "weights", None),
            device=getattr(args, "device", None),
            kernel_bandwidth=getattr(args, "kernel_bandwidth", None),
            min_cluster_size=getattr(args, "min_cluster_size", 5),
            output_dir=getattr(args, "output_dir", None) or (Path(args.output).parent if getattr(args, "output", None) else None),
            allow_stop_codons=not getattr(args, "disallow_stop_codons", False),
            quiet=getattr(args, "quiet", False),
            n_landmarks=getattr(args, "n_landmarks", "auto"),
            max_memory_mb=getattr(args, "max_memory_mb", 1024.0),
            rooting_mode=getattr(args, "rooting_mode", "convex_decay"),
            contemporaneous_dyads=getattr(args, "contemporaneous_dyads", False),
            dyad_max_days=getattr(args, "dyad_max_days", 90.0),
            dyad_max_dist=getattr(args, "dyad_max_dist", 0.010),
        )

    results = engine.run(
        plot=getattr(args, "plot", False) or (getattr(args, "plot_path", None) is not None),
        plot_path=getattr(args, "plot_path", None)
    )

    if getattr(args, "contemporaneous_dyads", False) and results.get("n_contemporaneous_clusters", 0) > 0:
        print(f"\n[★] Contemporaneous Direct Transmission Screening:")
        print(f"    Discovered {results['n_contemporaneous_clusters']} point-source transmission clusters ({results.get('n_contemporaneous_taxa', 0)} taxa) sampled <= {getattr(args, 'dyad_max_days', 90.0):.0f} days apart.")
        if "transmission_mode_counts" in results:
            print(f"    Dual-Track Transmission Breakdown:")
            for mode, cnt in results["transmission_mode_counts"].items():
                print(f"      - {mode}: {cnt}")

    if getattr(args, "output", None) and not str(args.output).endswith("/"):
        out_p = Path(args.output)
        if out_p.suffix.lower() == ".json":
            with open(out_p, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"[✓] AutoClock JSON summary copied to: {out_p}")

    if getattr(args, "csv", None):
        shutil.copyfile(results["classified_metadata_path"], args.csv)
        print(f"[✓] AutoClock classified metadata copied to: {args.csv}")

    print("\n[✓] ChronAeon AutoClock execution completed successfully.")


def cmd_sketch(args):
    """Alignment-free MinHash sketching and sequence binning."""
    from .sketch import AlignmentFreeBinner
    from aeon_core.dataset import parse_alignment_sequences

    seqs = parse_alignment_sequences(args.alignment)
    binner = AlignmentFreeBinner(
        k=args.kmer,
        sketch_size=args.sketch_size,
        alignable_threshold=args.alignable_threshold,
        quarantine_threshold=args.quarantine_threshold,
        min_bin_size=args.min_bin_size,
        quiet=args.quiet,
    )
    results = binner.bin_dataset(seqs)

    import json
    output = {
        "bins": results["bins"],
        "quarantined": results["quarantined"],
        "n_bins": len(results["bins"]),
        "n_quarantined": len(results["quarantined"]),
        "n_total": len(seqs),
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"[✓] Sketch binning results written to: {args.output}")
    else:
        print(json.dumps(output, indent=2, default=str))


def cmd_align(args):
    """Reference-guided codon-aware alignment threading."""
    from .alignment import ReferenceCodonAligner
    from aeon_core.dataset import parse_alignment_sequences

    ref_seqs = parse_alignment_sequences(args.reference)
    if len(ref_seqs) != 1:
        print(f"[!] Error: reference file must contain exactly one sequence (got {len(ref_seqs)})")
        return
    ref_name = next(iter(ref_seqs))
    ref_seq = ref_seqs[ref_name]

    query_seqs = parse_alignment_sequences(args.query)
    aligner = ReferenceCodonAligner(
        ref_seq=ref_seq,
        ref_name=ref_name,
        open_gap_score=args.open_gap,
        extend_gap_score=args.extend_gap,
    )
    results = aligner.align_batch(
        query_seqs,
        max_workers=args.threads,
        quiet=args.quiet,
    )

    if args.output:
        from pathlib import Path
        out = Path(args.output)
        with open(out, "w") as f:
            for taxon, seq in results["aligned_seqs"].items():
                f.write(f">{taxon}\n{seq}\n")
        n_failed = len(results["failed"])
        print(f"[✓] Aligned {len(results['aligned_seqs'])}/{len(query_seqs)} sequences to '{ref_name}' ({aligner.l_ref_nt} bp)")
        if n_failed:
            print(f"[!] {n_failed} sequences failed alignment:")
            for taxon, reason in results["failed"].items():
                print(f"    {taxon}: {reason}")
        print(f"[✓] Aligned sequences written to: {out}")
    else:
        print(f"[✓] Aligned {len(results['aligned_seqs'])}/{len(query_seqs)} sequences to '{ref_name}' ({aligner.l_ref_nt} bp)")
        if results["failed"]:
            print(f"[!] {len(results['failed'])} sequences failed alignment")


def list_models():
    """List available model variants from Hugging Face."""
    print_available_variants(cli_name="chronaeon")


def main():
    import argparse

    from . import __version__
    parser = argparse.ArgumentParser(
        prog="chronaeon",
        description="ChronAeon: Ultra-Fast Molecular Clock Dating, Phylodynamics, and Genomic Surveillance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"chronaeon {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Heterochronous Molecular Clock & MRCA Dating Subcommand
    date_parser = subparsers.add_parser(
        "date",
        aliases=["dating", "mrca", "clock", "chronaeon"],
        help="Calibrate heterochronous molecular clocks and date MRCA ancestor"
    )
    date_parser.add_argument("-a", "--alignment", default=None, help="Path to in-frame codon FASTA or NEXUS alignment (or BEAST XML)")
    date_parser.add_argument("--beast", default=None, help="Path to BEAST 1.x or BEAST 2.x XML configuration file containing sequences and tip sampling dates")
    date_parser.add_argument("-t", "--tree", default=None, help="Optional Newick/NEXUS phylogenetic tree (optional if embedded, or if --no-tree/--use-tn93 is set)")
    date_parser.add_argument("--no-tree", action="store_true", help="Skip phylogenetic tree and estimate pairwise evolutionary distances directly from alignment using TN93")
    date_parser.add_argument("--use-tn93", action="store_true", help="Estimate pairwise distances directly from alignment using TN93 (skips tree)")
    date_parser.add_argument("--distance-mode", choices=["auto", "tree", "latent", "tn93"], default="auto", help="Root-to-tip distance calculation mode: 'auto' (patristic tree distances if tree provided, else tree-free TN93 time-decay consensus distances; matches 55 published benchmarks), 'tree' (patristic tree distances), 'latent' (continuous convex hull root optimization in sequence representation space), or 'tn93' (empirical TN93 consensus distances)")
    date_parser.add_argument("-d", "--dates", default=None, help="Path to Nextstrain Auspice JSON, metadata CSV/TSV, or omitted to auto-extract timestamps from FASTA headers")
    date_parser.add_argument("--date-col", default=None, help="Column name for sample collection date in metadata CSV/TSV")
    date_parser.add_argument("--strain-col", default=None, help="Column name for taxon / strain identifier in metadata CSV/TSV")
    date_parser.add_argument("--date-regex", default=None, help="Optional regex with capture group to extract dates from sequence headers")
    date_parser.add_argument("--root-taxon", default=None, help="Reference taxon to use as ancestral root founder (e.g. 'CONSENSUS', earliest taxon, or outgroup; default for tree-free: time-decay weighted consensus)")
    date_parser.add_argument("--decay-gamma", type=float, default=None, help="Exponential decay rate gamma for Time-Decay Weighted Consensus (default: 0.05 or auto-scaled)")
    date_parser.add_argument("--no-optimize-root", action="store_true", help="Disable heuristic root search when a tree is provided")
    date_parser.add_argument("--method", choices=["all", "ols", "pgls"], default="all", help="Dating estimator(s) to run: 'all', 'pgls', or 'ols' (default: all)")
    date_parser.add_argument("--clock-model", choices=["auto", "linear", "spline", "power"], default="auto", help="Clock curvature model: 'auto' (F-test/AIC adjudication against restricted spline), 'linear', 'spline' (2-DF restricted natural cubic spline), or 'power' (power-law)")
    date_parser.add_argument(
        "--nonlinear-clocks",
        "--non-linear-clocks",
        "--dudas-models",
        "--dudas",
        dest="nonlinear_clocks",
        action="store_true",
        help="Fit and report non-linear and time-varying clock models (quadratic, exponential decay, bilinear crash, polyepoch; date mode only, non-default)",
    )
    date_parser.add_argument("--ridge", default="auto", help="Regularization parameter for PGLS neural covariance: 'auto' (exact REML profile likelihood estimation of Pagel's lambda) or float (default: auto)")
    date_parser.add_argument("--tune-ridge", action="store_true", help="(Compatibility flag) Automated REML regularization is active by default")
    date_parser.add_argument("--bootstrap", type=int, default=1000, help="Number of bootstrap resamples for empirical confidence intervals (default: 1000)")
    date_parser.add_argument("--ci-method", choices=["fieller", "delta", "poisson", "residual-boot", "site-boot", "jackknife"], default="fieller", help="Confidence interval calculation method for t_MRCA: 'fieller' (default; exact analytical ratio-test inversion), 'delta' (asymptotic linear tangent), 'poisson' (finite sequence length Poisson substitution sampling), 'residual-boot' (Wild Rademacher residual bootstrap), 'site-boot' (full transformer site bootstrap), or 'jackknife' (leave-one-out leverage)")
    date_parser.add_argument("--loocv", action="store_true", help="Perform leave-one-out cross-validation (LOOCV) for out-of-sample tip recovery accuracy and Jackknife root stability")
    date_parser.add_argument("--plot", action="store_true", help="Generate publication-grade diagnostic PDF and PNG figures")
    date_parser.add_argument("--plot-path", default=None, help="Custom output path for diagnostic plot (e.g. mrca_clock.pdf)")
    date_parser.add_argument("--alluvial", action="store_true", help="Generate continuous manifold alluvial / river-flow phylogeny plot")
    date_parser.add_argument("--alluvial-path", default=None, help="Custom output path for alluvial / river-flow plot (PNG or PDF)")
    date_parser.add_argument("--color-by", default=None, help="Column name in metadata CSV to color streamlines by (e.g. 'city', 'country', 'subtype')")
    date_parser.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS_ENV, help="Path to local model weights file (overrides HF download)")
    date_parser.add_argument("--model-variant", dest="variant", default=DEFAULT_VARIANT_ENV, help=f"Model variant to download from HF (default: {DEFAULT_VARIANT})")
    date_parser.add_argument("-s", "--max-species", type=int, default=None, help="Maximum number of taxa to include")
    date_parser.add_argument("-b", "--batch-size", type=int, default=None, help="Site batch size (default: adaptive hardware budget)")
    date_parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    date_parser.add_argument("-o", "--output", help="Optional path to output JSON results")
    date_parser.add_argument("-c", "--csv", help="Optional path to output per-taxon CSV results")

    # 2. Phylogeography Subcommand
    geo_parser = subparsers.add_parser("phylogeo", aliases=["geo", "phylogeography", "spatial", "migration", "dispersal"], help="Run discrete phylogeography & spatial transmission network inference")
    geo_parser.add_argument("--example", "--run-example", action="store_true", help="Execute the built-in worked benchmark example (Avian Influenza A/H5N1 across 7 Chinese provinces from Lemey et al. 2009)")
    geo_parser.add_argument("-a", "--alignment", required=False, default=None, help="Path to FASTA alignment (required unless --example is set)")
    geo_parser.add_argument("-g", "--metadata", required=False, default=None, help="Path to metadata CSV/TSV or Auspice JSON containing location labels (required unless --example is set)")
    geo_parser.add_argument("-t", "--tree", default=None, help="Optional path to Newick tree (auto-built via FastTree if omitted)")
    geo_parser.add_argument("--location-col", default=None, help="Metadata column name for discrete location / region")
    geo_parser.add_argument("--strain-col", default=None, help="Metadata column name for taxon / strain identifier")
    geo_parser.add_argument("--date-col", default=None, help="Metadata column name for sample collection date")
    geo_parser.add_argument("--lat-col", default=None, help="Metadata column name for latitude")
    geo_parser.add_argument("--lon-col", default=None, help="Metadata column name for longitude")
    geo_parser.add_argument("--root-taxon", default=None, help="Reference taxon to use as tree root / outgroup (default: earliest isolate or temporal root)")
    geo_parser.add_argument("--n-perms", type=int, default=1000, help="Number of permutations for null flux distribution and Bayes Factor calculation (default: 1000)")
    geo_parser.add_argument("--min-bf", type=float, default=3.0, help="Bayes Factor threshold for significant transmission routes (default: 3.0)")
    geo_parser.add_argument("--fdr", type=float, default=0.10, help="FDR threshold for significant transmission routes (default: 0.10)")
    geo_parser.add_argument("--no-neural", action="store_true", help="Disable neural attention backbone; use phylogenetic tree branch transitions")
    geo_parser.add_argument("--plot", action="store_true", help="Generate publication-grade diagnostic PDF and PNG figures")
    geo_parser.add_argument("--plot-path", default=None, help="Custom output path for diagnostic plot (e.g. phylogeography.pdf)")
    geo_parser.add_argument("--geojson", default=None, help="Optional path to output GeoJSON feature collection")
    geo_parser.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS_ENV, help="Path to local model weights file (overrides HF download)")
    geo_parser.add_argument("--model-variant", dest="variant", default=DEFAULT_VARIANT_ENV, help=f"Model variant to download from HF (default: {DEFAULT_VARIANT})")
    geo_parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    geo_parser.add_argument("-o", "--output", help="Optional path to output JSON results")
    geo_parser.add_argument("-c", "--csv", help="Optional path to output transmission routes CSV")

    # 3. Phylodynamics R0 & Rt Subcommand
    r0_parser = subparsers.add_parser("dynamics", aliases=["r0", "rt", "phylodynamics", "growth"], help="Estimate epidemic growth rate (r) and reproduction numbers (R0, Rt)")
    r0_parser.add_argument("--example", "--run-example", action="store_true", help="Execute the built-in worked benchmark example (2009 Pandemic H1N1 Origin from Fraser et al. 2009)")
    r0_parser.add_argument("-a", "--alignment", required=False, default=None, help="Path to FASTA alignment with dates in headers (optional if --tree is provided)")
    r0_parser.add_argument("-t", "--tree", required=False, default=None, help="Path to Newick tree with calibrated or substitution branch lengths")
    r0_parser.add_argument("-g", "--metadata", required=False, default=None, help="Path to metadata CSV/TSV containing collection dates")
    r0_parser.add_argument("--pathogen", default=None, help="Pathogen preset (e.g. 'h1n1', 'ebola', 'sars-cov-2', 'measles', 'hiv_early')")
    r0_parser.add_argument("--generation-time", type=float, default=None, help="Mean clinical generation interval / serial interval (default from pathogen preset or 5.0 days)")
    r0_parser.add_argument("--generation-sd", type=float, default=None, help="Standard deviation of clinical generation interval (for gamma distribution)")
    r0_parser.add_argument("--latent-time", type=float, default=None, help="Optional latent period in days for SEIR renewal model")
    r0_parser.add_argument("--units", choices=["days", "years"], default="days", help="Time units for generation interval parameters (default: days)")
    r0_parser.add_argument("--date-col", default=None, help="Metadata CSV column name for sample collection date")
    r0_parser.add_argument("--strain-col", default=None, help="Metadata CSV column name for taxon / strain identifier")
    r0_parser.add_argument("--mu-prior", type=float, default=None, help="Clock rate prior (substitutions/site/year) to convert substitution trees to calendar time")
    r0_parser.add_argument("--window-size", type=float, default=0.25, help="Window span in years for dynamic R(t) skyline (default: 0.25 years / 3 months)")
    r0_parser.add_argument("--step-size", type=float, default=0.05, help="Step size in years for dynamic R(t) skyline sliding window (default: 0.05 years)")
    r0_parser.add_argument("--plot", action="store_true", help="Generate publication-grade 3-panel diagnostic figures")
    r0_parser.add_argument("--plot-path", default=None, help="Custom output path for diagnostic plot (e.g. r0_diagnostics.png)")
    r0_parser.add_argument("-o", "--output", help="Optional path to output JSON results")
    r0_parser.add_argument("-c", "--csv", help="Optional path to output dynamic R(t) skyline CSV")

    # 4. ChronAeon Sieve: High-Throughput Streaming QC & Clock Manifold Triage
    triage_parser = subparsers.add_parser(
        "triage",
        aliases=["sieve", "qc-stream", "stream-qc", "chronaeon-sieve", "radar", "qc"],
        help="Streaming high-throughput quality control and molecular clock outlier triage"
    )
    triage_parser.add_argument("-a", "--alignment", required=True, help="Path to reference FASTA or anchor alignment")
    triage_parser.add_argument("-d", "--dates", default=None, help="Path to reference metadata CSV/TSV containing collection dates")
    triage_parser.add_argument("-s", "--stream", default=None, help="Path to incoming FASTA file of streaming sequences to screen")
    triage_parser.add_argument("--stream-dates", default=None, help="Optional metadata CSV/TSV for streaming sequences (auto-extracted from FASTA if omitted)")
    triage_parser.add_argument("--date-col", default=None, help="Column name for sample collection date")
    triage_parser.add_argument("--strain-col", default=None, help="Column name for taxon / strain identifier")
    triage_parser.add_argument("--n-anchor", type=int, default=512, help="Number of anchor taxa for reference skeleton (default: 512)")
    triage_parser.add_argument("--n-bins", type=int, default=32, help="Number of temporal bins for stratified sampling (default: 32)")
    triage_parser.add_argument("--root-taxon", default=None, help="Reference taxon to use as ancestral root (default: earliest sampled)")
    triage_parser.add_argument("--tolerance-days", type=float, default=90.0, help="Maximum tolerated temporal error before flagging (default: 90.0 days)")
    triage_parser.add_argument("--z-threshold", type=float, default=2.5, help="Studentized residual divergence Z threshold (default: 2.5)")
    triage_parser.add_argument("--max-ambig", type=float, default=0.05, help="Maximum tolerated fraction of ambiguous/missing bases (Ns) before flagging as SUS_LOW_QUALITY (default: 0.05 = 5%%)")
    triage_parser.add_argument("-o", "--output", default=None, help="Output CSV path for diagnostic triage report (default: sieve_triage_report.csv)")
    triage_parser.add_argument("--clean-out", default=None, help="Optional path to output filtered clean FASTA (PASS sequences only)")
    triage_parser.add_argument("--sus-out", default=None, help="Optional path to output quarantined FASTA (SUS sequences)")
    triage_parser.add_argument("--align", "--minimap2", dest="align_minimap2", action="store_true", help="Dynamically align raw, unaligned streaming sequences to root reference using minimap2")

    # 5. ChronAeon AutoClock: Unsupervised Multi-Clock Community Deconvolution
    autoclock_parser = subparsers.add_parser(
        "autoclock",
        aliases=["auto-clock", "multi-clock", "deconvolve", "communities"],
        help="Unsupervised multi-clock community deconvolution and outlier triage"
    )
    autoclock_parser.add_argument("-a", "--alignment", default=None, help="Path to in-frame codon FASTA, NEXUS, or BEAST XML alignment")
    autoclock_parser.add_argument("--beast", default=None, help="Path to BEAST 1.x or BEAST 2.x XML configuration file containing sequences and tip sampling dates")
    autoclock_parser.add_argument("-d", "--dates", default=None, help="Path to sample collection dates (CSV, TSV, Auspice JSON, or auto-extracted from FASTA headers)")
    autoclock_parser.add_argument("--date-col", default=None, help="Column name for sample collection date in metadata CSV/TSV")
    autoclock_parser.add_argument("--strain-col", default=None, help="Column name for taxon / strain identifier in metadata CSV/TSV")
    autoclock_parser.add_argument("--date-regex", default=None, help="Optional regex with capture group to extract dates from headers")
    autoclock_parser.add_argument("-k", "--max-k", type=int, default=6, help="Maximum candidate number of clock communities to evaluate (default: 6)")
    autoclock_parser.add_argument("--manifold", choices=["transformer", "distance", "tn93", "auto"], default="transformer", help="Embedding manifold for spectral partitioning: 'transformer' (neural latent representations) or 'distance'/'tn93' (analytic pairwise continuous distance) (default: transformer)")
    autoclock_parser.add_argument("--weights", default=None, help="Path to custom model weights file or checkpoint")
    autoclock_parser.add_argument("--device", default=None, help="Execution hardware device ('cuda', 'mps', 'cpu')")
    autoclock_parser.add_argument("--kernel-bandwidth", type=float, default=None, help="Spectral affinity kernel bandwidth sigma for distance manifold (default: adaptive 10th percentile)")
    autoclock_parser.add_argument("--min-cluster-size", type=int, default=5, help="Minimum community size to consider a valid clock branch (default: 5)")
    autoclock_parser.add_argument("--output-dir", default=None, help="Directory to write community models, triage, and plots")
    autoclock_parser.add_argument("-o", "--output", default=None, help="Optional path to output JSON summary")
    autoclock_parser.add_argument("-c", "--csv", default=None, help="Optional path to output classified metadata CSV")
    autoclock_parser.add_argument("-H", "--hierarchical", action="store_true", help="Enable recursive hierarchical multi-clock community deconvolution")
    autoclock_parser.add_argument("--max-depth", type=int, default=3, help="Maximum recursion depth for hierarchical deconvolution (default: 3)")
    autoclock_parser.add_argument("--min-leaf-size", type=int, default=25, help="Minimum leaf community size for hierarchical deconvolution (default: 25)")
    autoclock_parser.add_argument("--min-delta-aicc", type=float, default=15.0, help="Minimum AICc improvement (AICc(K=1) - min AICc(K>=2)) required to justify splitting a subcommunity (default: 15.0)")
    autoclock_parser.add_argument("--n-landmarks", default="auto", help="Number of landmark sequences for Nyström low-rank approximation ('auto' or integer, default: auto)")
    autoclock_parser.add_argument("--max-memory-mb", type=float, default=1024.0, help="Maximum RAM budget (MB) for adaptive landmark matrix allocation (default: 1024.0)")
    autoclock_parser.add_argument("--rooting-mode", choices=["convex_decay", "consensus", "earliest"], default="convex_decay", help="Rooting mode for tree-free root-to-tip divergence anchoring: 'convex_decay' (time-decay weighted consensus across cohort, default), 'consensus' (unweighted modal consensus), or 'earliest' (earliest sampled sequence)")
    autoclock_parser.add_argument("--contemporaneous-dyads", dest="contemporaneous_dyads", action="store_true", default=False, help="Enable screening for contemporaneous direct transmission dyads and point-source clusters (default: disabled)")
    autoclock_parser.add_argument("--no-contemporaneous-dyads", dest="contemporaneous_dyads", action="store_false", help="Disable contemporaneous transmission dyad screening")
    autoclock_parser.add_argument("--dyad-max-days", type=float, default=90.0, help="Maximum sampling interval in days for contemporaneous transmission dyads (default: 90.0 days)")
    autoclock_parser.add_argument("--dyad-max-dist", type=float, default=0.010, help="Maximum Tamura-Nei 93 distance for contemporaneous transmission dyads (default: 0.010 subs/site)")
    autoclock_parser.add_argument("--plot", action="store_true", help="Generate publication diagnostic figures (.png and .pdf)")
    autoclock_parser.add_argument("--plot-path", default=None, help="Custom output path for diagnostic plot")
    autoclock_parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")

    # 6. Alignment-Free MinHash Sketching & Binning
    sketch_parser = subparsers.add_parser(
        "sketch",
        aliases=["cluster", "bin", "centrifuge"],
        help="Alignment-free MinHash sketching and sequence binning",
    )
    sketch_parser.add_argument("-a", "--alignment", required=True, help="Path to FASTA file of unaligned sequences to bin")
    sketch_parser.add_argument("-k", "--kmer", type=int, default=15, help="K-mer size for MinHash sketching (default: 15)")
    sketch_parser.add_argument("--sketch-size", type=int, default=1024, help="MinHash sketch size (default: 1024)")
    sketch_parser.add_argument("--alignable-threshold", type=float, default=0.12, help="Jaccard similarity threshold for alignable binning (default: 0.12)")
    sketch_parser.add_argument("--quarantine-threshold", type=float, default=0.02, help="Jaccard similarity threshold below which sequences are quarantined (default: 0.02)")
    sketch_parser.add_argument("--min-bin-size", type=int, default=5, help="Minimum bin size to retain (default: 5)")
    sketch_parser.add_argument("-o", "--output", default=None, help="Optional path to output JSON binning results")
    sketch_parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")

    # 7. Reference-Guided Codon Alignment
    align_parser = subparsers.add_parser(
        "align",
        aliases=["thread", "codon-align"],
        help="Reference-guided codon-aware alignment threading",
    )
    align_parser.add_argument("-r", "--reference", required=True, help="Path to reference CDS FASTA (single sequence)")
    align_parser.add_argument("-q", "--query", required=True, help="Path to query CDS FASTA (unaligned sequences)")
    align_parser.add_argument("--open-gap", type=float, default=-10.0, help="Gap opening penalty (default: -10.0)")
    align_parser.add_argument("--extend-gap", type=float, default=-1.0, help="Gap extension penalty (default: -1.0)")
    align_parser.add_argument("--threads", type=int, default=4, help="Number of parallel worker threads (default: 4)")
    align_parser.add_argument("-o", "--output", default=None, help="Optional path to output aligned FASTA")
    align_parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")

    # 8. List-models Subcommand
    list_parser = subparsers.add_parser("list-models", help="List available model variants from Hugging Face")

    args = parser.parse_args()
    if args.command == "list-models":
        list_models()
    elif args.command in ["date", "dating", "mrca", "clock", "chronaeon"]:
        cmd_dating(args)
    elif args.command in ["phylogeo", "geo", "phylogeography", "spatial", "migration", "dispersal"]:
        cmd_phylogeo(args)
    elif args.command in ["dynamics", "r0", "rt", "phylodynamics", "growth"]:
        cmd_dynamics(args)
    elif args.command in ["triage", "sieve", "qc-stream", "stream-qc", "chronaeon-sieve", "radar", "qc"]:
        cmd_triage(args)
    elif args.command in ["autoclock", "auto-clock", "multi-clock", "deconvolve", "communities"]:
        cmd_autoclock(args)
    elif args.command in ["sketch", "cluster", "bin", "centrifuge"]:
        cmd_sketch(args)
    elif args.command in ["align", "thread", "codon-align"]:
        cmd_align(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
