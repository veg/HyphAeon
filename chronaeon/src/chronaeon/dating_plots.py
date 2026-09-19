"""
chronaeon/dating_plots.py
-------------------------
Publication-grade diagnostic visualization for molecular clock dating.

Generates root-to-tip regression plots, residual diagnostics, and
continuous-manifold alluvial phylogeny figures (PDF and PNG).
"""

import re
from pathlib import Path
from typing import Dict, Optional, Union, Any

import numpy as np
import pandas as pd

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
)
from aeon_core.io import ensure_parent_directory



from .dating_models import compute_rcs_basis
from .dating_kernels import compute_neural_covariance_kernel

# =========================================================================
# 5. Diagnostic Visualization & Figure Generation
# =========================================================================

def plot_mrca_dating(
    dating_results: Dict[str, Any],
    output_path: Union[str, Path],
    title: Optional[str] = None
):
    """Generates a publication-grade diagnostic PDF and PNG figure."""
    if not HAS_MATPLOTLIB:
        print("[!] Matplotlib not available; skipping diagnostic plot.")
        return

    fig = plt.figure(figsize=(13, 5.5), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1.0])

    times = dating_results['times']
    dists = dating_results['dists']
    ols = dating_results['ols']
    pgls = dating_results.get('pgls')
    spline = dating_results.get('spline')
    power = dating_results.get('power')

    # Panel A: Root-to-Tip Molecular Clock Regression
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(times, dists, color='#1f77b4', s=42, alpha=0.75, edgecolors='black', linewidth=0.5, label='Dated Strains', zorder=3)

    t_min = float(np.min(times))
    t_max = float(np.max(times))
    mrca_candidates = [ols['t_mrca']]
    if pgls:
        mrca_candidates.append(pgls['t_mrca'])
    if spline:
        mrca_candidates.append(spline['t_mrca'])
    if power:
        mrca_candidates.append(power['t_mrca'])

    plot_left = min(t_min - (t_max - t_min) * 0.35, min(mrca_candidates) - (t_max - t_min) * 0.1)
    plot_right = t_max + (t_max - t_min) * 0.05
    x_grid = np.linspace(plot_left, plot_right, 200)

    # OLS fitted line
    y_ols = ols['mu'] * (x_grid - ols['t_ref']) + ols['d0']
    ax1.plot(x_grid, y_ols, color='#e63946', linestyle='--', linewidth=2.0,
             label=f"OLS (t_MRCA={ols['t_mrca']:.1f}, μ={ols['mu']:.5f})", zorder=4)

    # PGLS fitted line
    if pgls:
        y_pgls = pgls['mu'] * (x_grid - pgls['t_ref']) + pgls['d0']
        ax1.plot(x_grid, y_pgls, color='#7b2cbf', linestyle='-', linewidth=2.5,
                 label=f"HyphAeon PGLS (t_MRCA={pgls['t_mrca']:.1f}, μ={pgls['mu']:.5f})", zorder=5)

    # Restricted Spline fitted curve
    if spline:
        knots_arr = np.array(spline['knots'])
        b0 = spline['beta_0']
        b1 = spline['beta_1']
        b2 = spline['beta_2']
        x_spline = np.linspace(max(plot_left, spline['t_mrca']), plot_right, 250)
        B_grid, _ = compute_rcs_basis(x_spline, knots_arr)
        y_spline = b0 + b1 * x_spline + (b2 * B_grid[:, 0] if B_grid.shape[1] > 0 else 0.0)
        lbl_spline = f"Restricted Spline (t_MRCA={spline['t_mrca']:.1f}, μ_anc={spline['rate_ancestral']:.5f})"
        color_s = '#2a9d8f' if spline.get('is_nonlinear_preferred') else '#f4a261'
        style_s = '-' if spline.get('is_nonlinear_preferred') else ':'
        ax1.plot(x_spline, y_spline, color=color_s, linestyle=style_s, linewidth=2.4, label=lbl_spline, zorder=6)

    # Power-law fitted curve
    if power:
        x_power = np.linspace(max(plot_left, power['t_mrca'] + 1e-4), plot_right, 200)
        y_power = power['k'] * (np.maximum(0.0, x_power - power['t_mrca']) ** power['theta'])
        lbl_power = f"Power-Law (t_MRCA={power['t_mrca']:.1f}, θ={power['theta']:.3f})"
        color_p = '#2a9d8f' if power.get('is_nonlinear_preferred') else '#f4a261'
        style_p = '-' if power.get('is_nonlinear_preferred') else ':'
        ax1.plot(x_power, y_power, color=color_p, linestyle=style_p, linewidth=2.2, label=lbl_power, zorder=6)

    # MRCA markers and CI error bars at distance = 0
    ax1.axhline(0, color='gray', linestyle=':', linewidth=0.8, zorder=1)
    if not np.isnan(ols['t_mrca']):
        ci_l = ols['ci_mrca'][0]
        ci_r = ols['ci_mrca'][1]
        if not np.isnan(ci_l) and not np.isnan(ci_r) and not np.isneginf(ci_l):
            e_ols_l = max(0.0, ols['t_mrca'] - min(ci_l, ci_r))
            e_ols_r = max(0.0, max(ci_l, ci_r) - ols['t_mrca'])
            ax1.errorbar([ols['t_mrca']], [0], xerr=[[e_ols_l], [e_ols_r]],
                         fmt='s', color='#e63946', markersize=6, capsize=4, capthick=1.5, zorder=6)
        else:
            ax1.plot([ols['t_mrca']], [0], marker='s', color='#e63946', markersize=6, zorder=6)
    if pgls and not np.isnan(pgls['t_mrca']):
        ci_l = pgls['ci_mrca'][0]
        ci_r = pgls['ci_mrca'][1]
        if not np.isnan(ci_l) and not np.isnan(ci_r) and not np.isneginf(ci_l):
            e_pgls_l = max(0.0, pgls['t_mrca'] - min(ci_l, ci_r))
            e_pgls_r = max(0.0, max(ci_l, ci_r) - pgls['t_mrca'])
            ax1.errorbar([pgls['t_mrca']], [-0.002], xerr=[[e_pgls_l], [e_pgls_r]],
                         fmt='D', color='#7b2cbf', markersize=6, capsize=4, capthick=1.5, zorder=6)
        else:
            ax1.plot([pgls['t_mrca']], [-0.002], marker='D', color='#7b2cbf', markersize=6, zorder=6)
    if spline and spline.get('is_nonlinear_preferred'):
        e_spl_l = max(0.0, spline['t_mrca'] - min(spline['ci_mrca'][0], spline['ci_mrca'][1]))
        e_spl_r = max(0.0, max(spline['ci_mrca'][0], spline['ci_mrca'][1]) - spline['t_mrca'])
        ax1.errorbar([spline['t_mrca']], [-0.003], xerr=[[e_spl_l], [e_spl_r]],
                     fmt='^', color='#2a9d8f', markersize=6, capsize=4, capthick=1.5, zorder=6)

    ax1.set_xlim(plot_left, plot_right)
    ax1.set_xlabel("Sampling Date / Time", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Root-to-Tip Divergence (subs/site)", fontsize=11, fontweight='bold')
    ax1.set_title("(A) Heterochronous Molecular Clock Regression", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, fontsize=8.5)
    ax1.grid(True, linestyle=':', alpha=0.4)

    # Panel B: Residual Error Diagnostics
    ax2 = fig.add_subplot(gs[1])
    res_ols = ols['residuals']
    t_ols = ols.get('times', times)
    if len(t_ols) != len(res_ols):
        t_ols = times[:len(res_ols)]
    ax2.scatter(t_ols, res_ols, color='#e63946', alpha=0.7, s=40, edgecolors='black', linewidth=0.5, label='OLS Residuals', zorder=3)
    if pgls:
        res_pgls = pgls['residuals']
        t_pgls = pgls.get('times', times)
        if len(t_pgls) != len(res_pgls):
            t_pgls = times[:len(res_pgls)]
        ax2.scatter(t_pgls, res_pgls, color='#7b2cbf', alpha=0.7, s=40, marker='^', edgecolors='black', linewidth=0.5, label='HyphAeon PGLS Residuals', zorder=4)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.2)
    ax2.set_xlabel("Sampling Date / Time", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Residual Divergence (d - d_pred)", fontsize=11, fontweight='bold')
    ax2.set_title("(B) Residual Error Diagnostics", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, fontsize=8.5)
    ax2.grid(True, linestyle=':', alpha=0.4)

    plt.suptitle(title or "ChronAeon Molecular Clock Calibration & Ancestor Dating", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()

    out_p = Path(output_path)
    ensure_parent_directory(out_p)
    plt.savefig(out_p, dpi=300)
    if out_p.suffix.lower() != '.png':
        plt.savefig(out_p.with_suffix('.png'), dpi=300)
    plt.close(fig)
    print(f"[✓] Diagnostic plot generated: {out_p}")


def plot_alluvial_phylogeny(
    dating_results: Dict[str, Any],
    output_path: Union[str, Path],
    alignment_path: Optional[Union[str, Path]] = None,
    tree_path: Optional[Union[str, Path]] = None,
    dates_source: Optional[Union[str, Path, Dict[str, float]]] = None,
    color_by: Optional[str] = None,
    title: Optional[str] = None,
    tau_factor: float = 0.04,
    show_uncertainty_capsules: bool = True,
    max_tip_labels: int = 25,
) -> Optional[str]:
    """
    Generates a publication-grade Continuous Manifold Alluvial / River-Flow Phylogeny.

    Replaces forced binary bifurcations with smooth continuous coalescent streamlines
    emerging from the ancestral founder origin (t_MRCA) and fanning out into the
    empirical sequence manifold.
    """
    if not HAS_MATPLOTLIB:
        print("[!] Matplotlib not available; skipping alluvial plot.")
        return None

    import matplotlib.patheffects as pe

    records = dating_results.get('taxa_records', [])
    if not records:
        print("[!] No taxa records found in dating results; skipping alluvial plot.")
        return None

    taxa = [r['taxon'] for r in records]
    N = len(taxa)
    sample_dates = np.array([float(r['sampling_date']) for r in records])
    dists = np.array([float(r['root_divergence']) for r in records])

    # Inferred parameters
    t_mrca = dating_results.get('t_mrca')
    if t_mrca is None or np.isnan(t_mrca):
        t_mrca = dating_results.get('ols', {}).get('t_mrca', np.min(sample_dates) - 10.0)

    ci_mrca = dating_results.get('ci_mrca')
    if ci_mrca is None or np.isnan(ci_mrca[0]) or np.isnan(ci_mrca[1]):
        ci_mrca = [t_mrca - 5.0, t_mrca + 5.0]

    mu = dating_results.get('mu')
    if mu is None or mu <= 0 or np.isnan(mu):
        mu = max(1e-5, (np.mean(dists) / max(1.0, np.mean(sample_dates) - t_mrca)))

    selected_clock = dating_results.get('selected_clock', 'Linear Clock')

    # Pairwise Distance Matrix D
    D = None
    if tree_path and Path(tree_path).exists():
        try:
            import ete3
            tree = ete3.Tree(str(tree_path))
            leaf_map = {l.name: l for l in tree.get_leaves()}
            shared = [t for t in taxa if t in leaf_map]
            if len(shared) == N:
                D = np.zeros((N, N))
                for i in range(N):
                    for j in range(i+1, N):
                        d_val = leaf_map[taxa[i]].get_distance(leaf_map[taxa[j]])
                        D[i, j] = d_val
                        D[j, i] = d_val
        except Exception:
            D = None

    if D is None:
        aln_file = alignment_path or dating_results.get('alignment')
        if aln_file and Path(aln_file).exists():
            try:
                seq_dict = parse_alignment_sequences(aln_file)
                D = compute_tn93_distance_matrix(seq_dict, taxa)
            except Exception as e:
                print(f"[!] Warning: Could not compute TN93 matrix from {aln_file}: {e}")
                D = None

    if D is None:
        # Fallback to rank-1 metric from root distances
        D = np.abs(dists[:, None] - dists[None, :])

    # 1D Classical MDS Embedding
    H = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * H @ (D ** 2) @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]

    y1 = eigvecs[:, idx[0]] * np.sqrt(np.maximum(0, eigvals[idx[0]]))
    y1 = y1 - np.mean(y1)
    # Orient consistently
    earliest_idx = np.argmin(sample_dates)
    if y1[earliest_idx] < 0:
        y1 = -y1

    # Pairwise shared divergence times
    t_div = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                t_div[i, j] = sample_dates[i]
            else:
                d_shared = max(0.0, 0.5 * (dists[i] + dists[j] - D[i, j]))
                t_div[i, j] = min(min(sample_dates[i], sample_dates[j]), t_mrca + d_shared / mu)

    # Time grid and trajectory calculation
    t_min = t_mrca
    t_max = max(sample_dates)
    time_grid = np.linspace(t_min, t_max, 250)
    timespan = max(1.0, t_max - t_min)
    tau = max(1.2, tau_factor * timespan)

    trajectories = []
    for i in range(N):
        t_i = sample_dates[i]
        valid_t = time_grid[time_grid <= t_i]
        if len(valid_t) == 0 or valid_t[-1] < t_i:
            valid_t = np.append(valid_t, t_i)

        y_path = np.zeros(len(valid_t))
        for k, t in enumerate(valid_t):
            if t <= t_mrca:
                y_path[k] = 0.0
                continue
            delta = (t_div[i, :] - t) / tau
            K_t = 1.0 / (1.0 + np.exp(-np.clip(delta, -18, 18)))
            active_mask = (sample_dates >= (t - 0.5 * tau))
            w = K_t * active_mask
            w[i] = max(w[i], 1.0)
            w_sum = np.sum(w)
            root_blend = min(1.0, (t - t_mrca) / (2.0 * tau))
            y_path[k] = root_blend * (np.sum(w * y1) / w_sum)

        trajectories.append((valid_t, y_path))

    # Metadata extraction for coloring
    meta_df = None
    if dates_source and (isinstance(dates_source, (str, Path)) and Path(dates_source).exists()):
        try:
            p_ext = Path(dates_source).suffix.lower()
            if p_ext in ['.csv', '.tsv', '.txt']:
                sep = '\t' if p_ext in ['.tsv', '.txt'] else ','
                meta_df = pd.read_csv(dates_source, sep=sep)
        except Exception:
            meta_df = None

    # Determine grouping column
    group_labels = None
    color_col_name = None
    if meta_df is not None:
        for cand_col in ['taxon', 'strain', 'id', 'Taxon', 'accession', 'name']:
            if cand_col in meta_df.columns:
                meta_df = meta_df.set_index(cand_col)
                break

        target_cols = [color_by] if color_by else ['city', 'location', 'country', 'subtype', 'clade', 'group']
        for col in target_cols:
            if col and col in meta_df.columns:
                matched_series = meta_df[col].reindex(taxa)
                if matched_series.notna().sum() > 0.4 * N:
                    color_col_name = col
                    group_labels = matched_series.fillna("Unknown").astype(str).tolist()
                    break

    if group_labels is None:
        import re
        extracted = []
        for t in taxa:
            m = re.search(r't([A-Za-z0-9]+)s', t)
            if m:
                st = m.group(1)
                extracted.append(f"Subtype {st}" if len(st) <= 2 else st)
            elif 'CG' in t:
                extracted.append("Republic of Congo")
            else:
                extracted.append(None)
        if sum(1 for e in extracted if e is not None) > 0.4 * N:
            group_labels = [e or "Other" for e in extracted]
            color_col_name = "Clade / Subtype"

    # Set up Figure
    fig, ax = plt.subplots(figsize=(16, 9.5), dpi=300)
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#ffffff")

    # Plot Shaded Fieller Root CI
    ci_low, ci_high = float(ci_mrca[0]), float(ci_mrca[1])
    ax.axvspan(ci_low, ci_high, color="#38bdf8", alpha=0.18, zorder=1,
               label=rf"Root $t_{{\mathrm{{MRCA}}}}$ 95% CI [{ci_low:.1f}, {ci_high:.1f}]")
    ax.axvline(t_mrca, color="#0284c7", linestyle="--", lw=1.8, zorder=2,
               label=rf"Inferred Founder $t_{{\mathrm{{MRCA}}}}$: {t_mrca:.1f} CE")

    # Color mapping
    unique_groups = sorted(list(set(group_labels))) if group_labels else None
    if unique_groups and len(unique_groups) <= 15:
        palette = [
            '#2563eb', '#059669', '#dc2626', '#9333ea', '#d97706',
            '#0891b2', '#ea580c', '#4f46e5', '#db2777', '#65a30d',
            '#0284c7', '#ca8a04', '#475569', '#b91c1c', '#047857'
        ]
        group_to_color = {g: palette[i % len(palette)] for i, g in enumerate(unique_groups)}
        colors = [group_to_color[g] for g in group_labels]
    else:
        cmap = plt.get_cmap("viridis")
        norm = (sample_dates - min(sample_dates)) / max(1e-6, max(sample_dates) - min(sample_dates))
        colors = [cmap(val) for val in norm]

    # Draw Streamlines (Alluvial Flow)
    plotted_groups = set()
    for i in range(N):
        valid_t, y_path = trajectories[i]
        c = colors[i]
        lbl = None
        if unique_groups and group_labels[i] not in plotted_groups:
            lbl = group_labels[i]
            plotted_groups.add(group_labels[i])

        # Ribbon Halo (Soft Glow)
        ax.plot(valid_t, y_path, color=c, lw=2.8, alpha=0.22, zorder=3)
        # Core Streamline
        ax.plot(valid_t, y_path, color=c, lw=1.3, alpha=0.85, zorder=4, label=lbl)

        # Tip Marker
        tip_t = sample_dates[i]
        tip_y = y_path[-1]
        is_outlier = records[i].get('is_outlier', False)
        if is_outlier:
            ax.scatter(tip_t, tip_y, color='#ef4444', s=65, marker='X', edgecolors='#7f1d1d', lw=1.2, zorder=7)
        else:
            ax.scatter(tip_t, tip_y, color=c, s=42, edgecolors='#0f172a', lw=0.7, zorder=6)

    # Ancestral Founder Marker
    ax.scatter([t_mrca], [0.0], color="#0284c7", s=160, marker="D",
               edgecolors="#082f49", lw=1.8, zorder=8, label="Ancestral Founder Origin")

    # Selected Tip Labels
    labeled_count = 0
    for i in range(N):
        is_outlier = records[i].get('is_outlier', False)
        if is_outlier or (labeled_count < max_tip_labels and (i % max(1, N // max_tip_labels) == 0)):
            valid_t, y_path = trajectories[i]
            tip_t = sample_dates[i]
            tip_y = y_path[-1]
            txt_label = f"{taxa[i]} ({tip_t:.0f})"
            if is_outlier:
                z_sc = records[i].get('z_score', 0.0)
                txt_label += f" [Z={z_sc:+.1f}]"
            ax.text(tip_t + 0.35, tip_y, txt_label, fontsize=7.2, va="center",
                    color="#b91c1c" if is_outlier else "#334155",
                    fontweight="bold" if is_outlier else "normal",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")], zorder=9)
            labeled_count += 1

    # Formatting and Labels
    plot_title = title or f"HyphAeon Continuous Manifold Alluvial Phylogeny (N = {N} taxa)"
    ax.set_title(plot_title, fontsize=14, weight="bold", pad=15, color="#0f172a")
    ax.set_xlabel("Calendar Time (Years CE)", fontsize=11, weight="bold", color="#1e293b")
    ax.set_ylabel("Continuous Manifold Divergence (Leading MDS Coordinate, subs/site)", fontsize=11, weight="bold", color="#1e293b")

    ax.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1", zorder=0)
    ax.set_xlim(min(ci_low - 4.0, t_mrca - 6.0), t_max + max(2.5, 0.08 * timespan))

    # Summary Information Box
    outlier_count = sum(1 for r in records if r.get('is_outlier', False))
    info_lines = [
        r"$\mathbf{HyphAeon\ Continuous\ Timetree:}$",
        rf"• Ancestor $t_{{\mathrm{{MRCA}}}}$: {t_mrca:.1f} CE [{ci_low:.1f}, {ci_high:.1f}]",
        rf"• Evolutionary Rate $\mu$: {mu:.5f} subs/site/yr",
        f"• Clock Model: {selected_clock}",
        f"• Flagged Outliers: {outlier_count} / {N} taxa",
        f"• Honest Polytomy: No arbitrary binary bifurcations"
    ]
    ax.text(0.02, 0.04, "\n".join(info_lines), transform=ax.transAxes, fontsize=9.0,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffffff", edgecolor="#94a3b8", alpha=0.94, lw=1.2))

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        leg = ax.legend(handles[:14], labels[:14], loc="upper left", frameon=True,
                        facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8.5, title=color_col_name or "Lineages")
        leg.get_title().set_fontweight('bold')

    plt.tight_layout()
    out_p = Path(output_path)
    ensure_parent_directory(out_p)
    plt.savefig(out_p, dpi=300)
    if out_p.suffix.lower() != '.png':
        plt.savefig(out_p.with_suffix('.png'), dpi=300)
    plt.close(fig)
    print(f"[✓] Saved publication alluvial phylogeny to: {out_p}")
    return str(out_p)

