#!/usr/bin/env python3
"""
plot_meme_vs_axomeme_results.py

Visualizes head-to-head comparison between AxoMEME and HyPhy MEME:
  1. LRT Scatter plot and correlation across all sites
  2. Paired ROC Curves
  3. Q-Q Null Calibration plot
  4. Runtime & Speedup Comparison
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

def main():
    db_path = 'scratch/meme_vs_axomeme_benchmarks.db'
    conn = sqlite3.connect(db_path)
    
    df_reps = pd.read_sql_query("SELECT * FROM benchmark_replicates", conn)
    df_sites = pd.read_sql_query("SELECT * FROM site_level_predictions", conn)
    conn.close()
    
    print(f"[*] Loaded {len(df_reps)} replicates and {len(df_sites)} site predictions.")
    if len(df_reps) == 0:
        print("[!] No benchmark data yet. Waiting for runs to finish...")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    
    # -------------------------------------------------------------
    # Panel 1: Site-Level LRT Scatter Plot (AxoMEME vs HyPhy MEME)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    regime_colors = {
        '1_SweetSpot_Pervasive': '#2e7d32',      # Green
        '2_PhaseTransition_Boundary': '#f57f17', # Amber
        '3_Episodic_Clade_25pct': '#1565c0',    # Blue
        '4_Purifying_Null_Omega015': '#757575',  # Grey
        '5_Neutral_Null_Omega100': '#c2185b'     # Pink
    }
    regime_labels = {
        '1_SweetSpot_Pervasive': 'Pervasive (100% Clade)',
        '2_PhaseTransition_Boundary': 'Boundary (80% Clade)',
        '3_Episodic_Clade_25pct': 'Episodic (25% Clade)',
        '4_Purifying_Null_Omega015': 'Purifying Null (w=0.15)',
        '5_Neutral_Null_Omega100': 'Neutral Null (w=1.00)'
    }
    
    for reg, grp in df_sites.groupby('regime_name'):
        c = regime_colors.get(reg, 'black')
        lbl = regime_labels.get(reg, reg)
        ax1.scatter(grp['meme_lrt'], grp['axomeme_lrt'], c=c, label=lbl, alpha=0.55, s=22, edgecolors='none')
        
    max_val = max(df_sites['meme_lrt'].max(), df_sites['axomeme_lrt'].max(), 10.0) * 1.05
    ax1.plot([0, max_val], [0, max_val], 'k--', lw=1.5, alpha=0.7, label='Identity (y=x)')
    
    sp_r, _ = stats.spearmanr(df_sites['meme_lrt'], df_sites['axomeme_lrt'])
    pe_r, _ = stats.pearsonr(df_sites['meme_lrt'], df_sites['axomeme_lrt'])
    
    ax1.set_xlim(-0.5, max_val)
    ax1.set_ylim(-0.5, max_val)
    ax1.set_xlabel('HyPhy MEME Numerical LRT', fontsize=11, fontweight='bold')
    ax1.set_ylabel('AxoMEME Predicted LRT', fontsize=11, fontweight='bold')
    ax1.set_title(f'A. Site-Level Concordance (Spearman ρ = {sp_r:.3f}, Pearson r = {pe_r:.3f})', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8.5, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # -------------------------------------------------------------
    # Panel 2: Ground-Truth ROC-AUC Comparison Across Regimes
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    sel_reps = df_reps[df_reps['n_sel'] > 0]
    
    bar_width = 0.35
    regimes_sel = ['1_SweetSpot_Pervasive', '2_PhaseTransition_Boundary', '3_Episodic_Clade_25pct']
    clean_names = ['Pervasive (100%)', 'Boundary (80%)', 'Episodic (25%)']
    
    x = np.arange(len(regimes_sel))
    ax_aucs = [sel_reps[sel_reps['regime_name'] == r]['auc_axomeme'].mean() for r in regimes_sel]
    meme_aucs = [sel_reps[sel_reps['regime_name'] == r]['auc_meme'].mean() for r in regimes_sel]
    
    rects1 = ax2.bar(x - bar_width/2, ax_aucs, bar_width, label='AxoMEME (Neural Forward)', color='#1976d2', edgecolor='black')
    rects2 = ax2.bar(x + bar_width/2, meme_aucs, bar_width, label='HyPhy MEME (Numerical MLE)', color='#388e3c', edgecolor='black')
    
    ax2.set_ylabel('Ground-Truth ROC-AUC', fontsize=11, fontweight='bold')
    ax2.set_title('B. Discrimination Accuracy vs. Selection Breadth', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(clean_names, fontsize=10)
    ax2.set_ylim(0.5, 1.05)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    for r in rects1:
        h = r.get_height()
        ax2.annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for r in rects2:
        h = r.get_height()
        ax2.annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # -------------------------------------------------------------
    # Panel 3: Q-Q Plot on Pure Null Regimes
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    purifying_sites = df_sites[df_sites['regime_name'] == '4_Purifying_Null_Omega015']
    neutral_sites = df_sites[df_sites['regime_name'] == '5_Neutral_Null_Omega100']
    
    # Generate theoretical 50:50 mixture quantiles (1/2 delta_0 + 1/2 chi_1^2)
    def mixture_quantiles(n_points):
        u = np.linspace(0.5 / n_points, 1.0 - 0.5 / n_points, n_points)
        theo = np.zeros(n_points)
        chi_mask = (u > 0.5)
        # For u > 0.5, p_chi = 2 * (u - 0.5)
        theo[chi_mask] = stats.chi2.ppf(2.0 * (u[chi_mask] - 0.5), df=1)
        return theo

    if len(purifying_sites) > 0:
        n_pts = len(purifying_sites)
        theo_q = mixture_quantiles(n_pts)
        axo_pur_q = np.sort(purifying_sites['axomeme_lrt'].values)
        meme_pur_q = np.sort(purifying_sites['meme_lrt'].values)
        ax3.plot(theo_q, axo_pur_q, 'o-', color='#1976d2', label='AxoMEME (Purifying $\\omega=0.15$)', markersize=3.5, alpha=0.8)
        ax3.plot(theo_q, meme_pur_q, 's-', color='#388e3c', label='HyPhy MEME (Purifying $\\omega=0.15$)', markersize=3.5, alpha=0.8)

    if len(neutral_sites) > 0:
        n_pts = len(neutral_sites)
        theo_q = mixture_quantiles(n_pts)
        axo_neu_q = np.sort(neutral_sites['axomeme_lrt'].values)
        meme_neu_q = np.sort(neutral_sites['meme_lrt'].values)
        ax3.plot(theo_q, axo_neu_q, '^--', color='#d32f2f', label='AxoMEME (Neutral $\\omega=1.00$)', markersize=3.5, alpha=0.8)
        ax3.plot(theo_q, meme_neu_q, 'd--', color='#7b1fa2', label='HyPhy MEME (Neutral $\\omega=1.00$)', markersize=3.5, alpha=0.8)

    max_q = 8.0
    ax3.plot([0, max_q], [0, max_q], 'k--', lw=1.5, alpha=0.7, label='Theoretical $\\frac{1}{2}\\delta_0 + \\frac{1}{2}\\chi_1^2$')
    ax3.set_xlim(-0.2, max_q)
    ax3.set_ylim(-0.2, max_q)
    ax3.set_xlabel('Theoretical Null Quantile (LRT)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Empirical Null Quantile (LRT)', fontsize=11, fontweight='bold')
    ax3.set_title('C. Q-Q Null Calibration (Type-I Error Control)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax3.grid(True, linestyle='--', alpha=0.3)

    # -------------------------------------------------------------
    # Panel 4: Runtime & Speedup Comparison
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    mean_meme_time = df_reps['meme_time_sec'].mean()
    mean_axo_time = df_reps['axomeme_time_sec'].mean()
    overall_speedup = mean_meme_time / mean_axo_time if mean_axo_time > 0 else 1000.0
    
    methods = ['HyPhy MEME\n(Numerical MLE)', 'AxoMEME\n(Neural Transformer)']
    times = [mean_meme_time, mean_axo_time]
    colors = ['#e53935', '#1e88e5']
    
    bars = ax4.bar(methods, times, color=colors, edgecolor='black', width=0.45)
    ax4.set_yscale('log')
    ax4.set_ylabel('Mean Execution Time per Alignment (seconds, log scale)', fontsize=10, fontweight='bold')
    ax4.set_title(f'D. Computational Throughput ({overall_speedup:,.0f}× Acceleration)', fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    for b in bars:
        h = b.get_height()
        ax4.annotate(f'{h:.3f} s' if h < 1.0 else f'{h:.1f} s',
                     xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
                     
    ax4.text(0.5, 0.45, f"⚡ {overall_speedup:,.0f}× Faster\nSingle Forward Pass", 
             transform=ax4.transAxes, ha='center', va='center',
             fontsize=12, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f5e9', edgecolor='#43a047', lw=1.5))
             
    plt.suptitle('AxoMEME vs. HyPhy MEME: Ground-Truth Calibration & Benchmark Suite', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    out_png = 'scratch/meme_vs_axomeme_benchmark_results.png'
    plt.savefig(out_png, bbox_inches='tight')
    print(f"[✓] Saved benchmark visualization to: {out_png}")

if __name__ == '__main__':
    main()
