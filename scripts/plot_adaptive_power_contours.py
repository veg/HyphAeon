#!/usr/bin/env python3
"""
plot_adaptive_power_contours.py

Plots Gaussian Process 2D/3D iso-power contour surfaces (50% and 80% power frontiers)
from adaptive simulation database and analyzes sweet spots and blind spots.
"""

import sys
sys.path.insert(0, '/Users/sergei/Projects/TOGA_MEME')
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

def main(db_path='scratch/scaled_adaptive_simulations.db', out_png='scratch/adaptive_power_contours.png'):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT n_taxa, t_depth, beta_fg, fg_frac, mean_power_10, mean_power_20, mean_roc_auc, mean_sel_lrt
        FROM adaptive_evaluations
    """, conn)
    conn.close()
    
    if len(df) < 4:
        print(f"[*] Insufficient data points in {db_path} (found {len(df)}). Waiting for more iterations...")
        return
        
    print(f"[*] Loaded {len(df)} adaptive evaluation points from {db_path}.")
    
    # Fit GP
    param_bounds = {
        'n_taxa': (32, 384),
        't_depth': (2.5, 26.0),
        'log_beta_fg': (np.log10(2.0), np.log10(800.0)),
        'fg_frac': (0.02, 1.00)
    }
    
    def normalize(X):
        n_min, n_max = param_bounds['n_taxa']
        t_min, t_max = param_bounds['t_depth']
        b_min, b_max = param_bounds['log_beta_fg']
        f_min, f_max = param_bounds['fg_frac']
        
        X_n = np.zeros_like(X, dtype=float)
        X_n[:, 0] = (X[:, 0] - n_min) / (n_max - n_min)
        X_n[:, 1] = (X[:, 1] - t_min) / (t_max - t_min)
        X_n[:, 2] = (X[:, 2] - b_min) / (b_max - b_min)
        X_n[:, 3] = (X[:, 3] - f_min) / (f_max - f_min)
        return np.clip(X_n, 0.0, 1.0)

    X = np.column_stack([
        df['n_taxa'].values,
        df['t_depth'].values,
        np.log10(df['beta_fg'].values),
        df['fg_frac'].values
    ])
    X_norm = normalize(X)
    y = df['mean_power_10'].values
    
    kernel = ConstantKernel(1.0) * Matern(length_scale=[0.5, 0.5, 0.5, 0.5], length_scale_bounds=(1e-2, 10.0), nu=2.5) + WhiteKernel(0.01)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=False)
    gp.fit(X_norm, y)
    
    # Generate 3 Slices: (N=64, T=10), (N=128, T=16), (N=256, T=22)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    grid_f = np.linspace(0.02, 1.00, 100)
    grid_log_b = np.linspace(np.log10(2.0), np.log10(800.0), 100)
    FF, BB = np.meshgrid(grid_f, grid_log_b)
    
    slices = [
        ("Shallow / Moderate (N=64, T=10.0)", 64, 10.0),
        ("Dense / Divergent   (N=128, T=16.0)", 128, 16.0),
        ("Large TOGA Scale    (N=256, T=22.0)", 256, 22.0)
    ]
    
    for ax_idx, (title, n_val, t_val) in enumerate(slices):
        ax = axes[ax_idx]
        
        flat_f = FF.ravel()
        flat_b = BB.ravel()
        flat_n = np.full_like(flat_f, n_val)
        flat_t = np.full_like(flat_f, t_val)
        
        query_raw = np.column_stack([flat_n, flat_t, flat_b, flat_f])
        query_norm = normalize(query_raw)
        
        pred_mu, pred_std = gp.predict(query_norm, return_std=True)
        pred_mu = np.clip(pred_mu, 0.0, 1.0).reshape(FF.shape)
        
        # Contour fill
        cf = ax.contourf(FF * 100, 10 ** BB, pred_mu * 100, levels=np.linspace(0, 100, 21), cmap='viridis', alpha=0.9)
        
        # Iso-Power lines: 20%, 50%, 80%
        cs = ax.contour(FF * 100, 10 ** BB, pred_mu * 100, levels=[20, 50, 80], colors=['#ffffff', '#ffeb3b', '#ff1744'], linewidths=2.0)
        ax.clabel(cs, fmt='%d%% Power', inline=True, fontsize=10, colors='black')
        
        # Overlay evaluated points in this slice range
        mask = (np.abs(df['n_taxa'] - n_val) <= 64)
        sub_pts = df[mask]
        if len(sub_pts) > 0:
            ax.scatter(sub_pts['fg_frac'] * 100, sub_pts['beta_fg'], c=sub_pts['mean_power_10'] * 100,
                       cmap='viridis', edgecolor='black', s=50, zorder=5, label='Sampled Points')
            
        ax.set_yscale('log')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Foreground Branch Fraction (%)', fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel('Selection Intensity (beta+)', fontsize=10)
            
        ax.grid(True, linestyle='--', alpha=0.3, which='both')
        
    cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.06, pad=0.18)
    cbar.set_label('AxoMEME Predicted Statistical Power (at nominal p <= 0.10, %)', fontsize=11)
    
    plt.suptitle('AxoMEME 4D Iso-Power Response Surface & Active Learning Boundary Map', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight')
    print(f"[✓] Saved contour surface visualization to: {out_png}")

if __name__ == '__main__':
    db = sys.argv[1] if len(sys.argv) > 1 else 'scratch/scaled_adaptive_simulations.db'
    png = sys.argv[2] if len(sys.argv) > 2 else 'scratch/adaptive_power_contours.png'
    main(db, png)
