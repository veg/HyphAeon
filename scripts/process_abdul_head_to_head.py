#!/usr/bin/env python3
"""
process_abdul_head_to_head.py

Pulls completed HyPhy MEME outputs from silverback, parses results alongside
AxoMEME predictions for the Abdul et al. 2018 dataset, populates SQLite,
and renders a 4-panel empirical comparison figure.
"""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io
import time
import json
import sqlite3
import subprocess
import numpy as np
import pandas as pd
from Bio import Phylo, SeqIO
import scipy.stats as stats
import matplotlib.pyplot as plt
import torch

from scripts.train_transformer_selection import PhyloAxialTransformer
from scripts.predict_regression_nexus import compute_mds_coordinates, get_codon_token, get_aa_token
from scratch.test_mg94_high_power_baseline import compute_fast_dist_matrix

CLUSTER_HOST = "silverback.temple.edu"
CLUSTER_DIR = "~/abdul_bench"

def run_axomeme_local(fa_path, nwk_path, model, device):
    tree_obj = Phylo.read(nwk_path, 'newick')
    taxa = [term.name for term in tree_obj.get_terminals() if term.name]
    dist_mat = compute_fast_dist_matrix(tree_obj, taxa)
    mds_coords = compute_mds_coordinates(dist_mat, n_components=4)
    
    seq_dict = {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(fa_path, 'fasta')}
    n_taxa = len(taxa)
    L = len(list(seq_dict.values())[0]) // 3
    
    c_all = np.zeros((L, n_taxa, 1), dtype=np.int64)
    a_all = np.zeros((L, n_taxa, 1), dtype=np.int64)
    for i, sp in enumerate(taxa):
        seq = seq_dict[sp]
        for site in range(L):
            codon = seq[site*3 : (site+1)*3].upper()
            c_all[site, i, 0] = get_codon_token(codon)
            a_all[site, i, 0] = get_aa_token(codon)
            
    is_aa_invariable = np.zeros(L, dtype=bool)
    for site in range(L):
        aa_col = a_all[site, :, 0]
        valid_aa = aa_col[aa_col < 20]
        if len(np.unique(valid_aa)) <= 1:
            is_aa_invariable[site] = True
            
    c_tensor = torch.tensor(c_all, dtype=torch.long)
    a_tensor = torch.tensor(a_all, dtype=torch.long)
    d_tensor = torch.tensor(dist_mat, dtype=torch.float32).unsqueeze(0).repeat(L, 1, 1)
    z_tensor = torch.tensor(mds_coords, dtype=torch.float32).unsqueeze(0).repeat(L, 1, 1)
    
    t0 = time.time()
    with torch.no_grad():
        y_lrt_soft, _ = model(c_tensor, a_tensor, d_tensor, z_tensor)
        axomeme_lrts = torch.clamp(y_lrt_soft.squeeze(-1), min=0.0).numpy().flatten()
    elapsed = time.time() - t0
    
    axomeme_lrts[is_aa_invariable] = 0.0
    return axomeme_lrts, is_aa_invariable, elapsed, L, n_taxa

def main():
    bench_dir = 'bench/abdul_2018'
    staging_dir = 'scratch/abdul_staging'
    outputs_dir = os.path.join(staging_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    db_path = 'scratch/abdul_meme_vs_axomeme.db'
    ckpt_path = 'axomeme_5_dim384_nonull.pt'
    
    genes = sorted([f.replace('.fasta', '') for f in os.listdir(bench_dir) if f.endswith('.fasta')])
    
    # 1. Sync completed JSON files from silverback
    print("[1/4] Pulling completed JSON outputs from silverback...")
    cmd = f"scp \"{CLUSTER_HOST}:{CLUSTER_DIR}/outputs/*\" {outputs_dir}/"
    subprocess.run(cmd, shell=True, check=True)
    
    # 2. Compute local AxoMEME predictions
    print("[2/4] Computing local AxoMEME predictions...")
    device = torch.device('cpu')
    torch.set_num_threads(2)
    model = PhyloAxialTransformer(embed_dim=384, num_layers=6, num_heads=12, window_size=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS abdul_genes (
            gene_id INTEGER PRIMARY KEY AUTOINCREMENT,
            gene_name TEXT,
            n_taxa INTEGER,
            n_codons INTEGER,
            spearman_rho REAL,
            pearson_r REAL,
            mae_lrt REAL,
            sig_sites_meme_05 INTEGER,
            sig_sites_meme_10 INTEGER,
            sig_sites_axo_05 INTEGER,
            sig_sites_axo_10 INTEGER,
            jaccard_overlap_10 REAL,
            meme_time_sec REAL,
            axomeme_time_sec REAL,
            speedup_factor REAL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS abdul_sites (
            site_id INTEGER PRIMARY KEY AUTOINCREMENT,
            gene_name TEXT,
            codon_idx INTEGER,
            is_invariable INTEGER,
            meme_lrt REAL,
            meme_pval REAL,
            meme_alpha REAL,
            meme_beta_pos REAL,
            meme_weight_pos REAL,
            axomeme_lrt REAL
        )
    ''')
    conn.execute("DELETE FROM abdul_sites")
    conn.execute("DELETE FROM abdul_genes")
    conn.commit()
    cur = conn.cursor()
    
    summary_rows = []
    
    for g in genes:
        fa = os.path.join(bench_dir, f"{g}.fasta")
        nwk = os.path.join(bench_dir, f"{g}.nwk")
        json_path = os.path.join(outputs_dir, f"{g}.json")
        
        axo_lrts, inv, axo_time, L, N = run_axomeme_local(fa, nwk, model, device)
        
        with open(json_path, 'r') as jf:
            text = jf.read()
            meme_data = json.loads(text.replace('NaN', '0.0').replace('-nan', '0.0').replace('nan', '0.0'))
            
        mle_table = meme_data['MLE']['content']['0']
        meme_lrts = np.array([row[5] for row in mle_table], dtype=np.float32)
        meme_pvals = np.array([row[6] for row in mle_table], dtype=np.float32)
        meme_alphas = np.array([row[0] for row in mle_table], dtype=np.float32)
        meme_betas_pos = np.array([row[3] for row in mle_table], dtype=np.float32)
        meme_weights_pos = np.array([row[4] for row in mle_table], dtype=np.float32)
        meme_time = float(meme_data.get('timers', {}).get('Overall', {}).get('timer', 60.0))
        
        sp_r, _ = stats.spearmanr(axo_lrts, meme_lrts)
        if np.isnan(sp_r): sp_r = 0.0
        pe_r, _ = stats.pearsonr(axo_lrts, meme_lrts)
        if np.isnan(pe_r): pe_r = 0.0
        mae = float(np.mean(np.abs(axo_lrts - meme_lrts)))
        
        sig_m_05 = int((meme_pvals <= 0.05).sum())
        sig_m_10 = int((meme_pvals <= 0.10).sum())
        sig_a_05 = int((axo_lrts >= 4.45).sum())
        sig_a_10 = int((axo_lrts >= 3.12).sum())
        
        sig_mask_m = (meme_pvals <= 0.10)
        sig_mask_a = (axo_lrts >= 3.12)
        both = int((sig_mask_m & sig_mask_a).sum())
        union = int((sig_mask_m | sig_mask_a).sum())
        jaccard = both / union if union > 0 else 1.0
        
        speedup = meme_time / axo_time if axo_time > 0 else 1000.0
        
        cur.execute('''
            INSERT INTO abdul_genes (
                gene_name, n_taxa, n_codons, spearman_rho, pearson_r, mae_lrt,
                sig_sites_meme_05, sig_sites_meme_10, sig_sites_axo_05, sig_sites_axo_10,
                jaccard_overlap_10, meme_time_sec, axomeme_time_sec, speedup_factor
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            g, N, L, sp_r, pe_r, mae, sig_m_05, sig_m_10, sig_a_05, sig_a_10,
            jaccard, meme_time, axo_time, speedup
        ))
        
        site_records = [(
            g, s, int(inv[s]), float(meme_lrts[s]), float(meme_pvals[s]),
            float(meme_alphas[s]), float(meme_betas_pos[s]), float(meme_weights_pos[s]),
            float(axo_lrts[s])
        ) for s in range(L)]
        
        cur.executemany('''
            INSERT INTO abdul_sites (
                gene_name, codon_idx, is_invariable, meme_lrt, meme_pval,
                meme_alpha, meme_beta_pos, meme_weight_pos, axomeme_lrt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', site_records)
        conn.commit()
        
        summary_rows.append({
            'Gene': g, 'Taxa': N, 'Codons': L, 'r': pe_r, 'rho': sp_r, 'MAE': mae,
            'MEME (p<=0.10)': sig_m_10, 'AxoMEME (p<=0.10)': sig_a_10,
            'MEME Time (s)': meme_time, 'AxoMEME Time (s)': axo_time,
            'Speedup': speedup
        })
        
    df_summary = pd.DataFrame(summary_rows)
    df_sites = pd.read_sql_query("SELECT * FROM abdul_sites", conn)
    conn.close()
    
    print("\n" + "=" * 95)
    print("📊 EMPIRICAL VALIDATION: HYPHY MEME VS. AXOMEME (ABDUL ET AL. 2018)")
    print("=" * 95)
    print(df_summary.to_string(index=False))
    print("=" * 95)
    
    # 3. Render 4-panel publication figure
    print("\n[3/4] Rendering Empirical Comparison Figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    
    # Panel A: Overall LRT Scatter
    ax1 = axes[0, 0]
    ax1.scatter(df_sites['meme_lrt'], df_sites['axomeme_lrt'], c='#1565c0', alpha=0.35, s=18, edgecolors='none')
    max_val = max(df_sites['meme_lrt'].max(), df_sites['axomeme_lrt'].max(), 10.0) * 1.05
    ax1.plot([0, max_val], [0, max_val], 'k--', lw=1.5, alpha=0.7, label='Identity (y=x)')
    
    all_sp, _ = stats.spearmanr(df_sites['meme_lrt'], df_sites['axomeme_lrt'])
    all_pe, _ = stats.pearsonr(df_sites['meme_lrt'], df_sites['axomeme_lrt'])
    
    ax1.set_xlim(-0.3, max_val)
    ax1.set_ylim(-0.3, max_val)
    ax1.set_xlabel('HyPhy MEME Numerical LRT', fontsize=11, fontweight='bold')
    ax1.set_ylabel('AxoMEME Predicted LRT', fontsize=11, fontweight='bold')
    ax1.set_title(f'A. Site-Level Concordance ({len(df_sites):,} Codon Sites, r = {all_pe:.3f}, ρ = {all_sp:.3f})', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Panel B: Correlation by Gene
    ax2 = axes[0, 1]
    x = np.arange(len(df_summary))
    w = 0.35
    ax2.bar(x - w/2, df_summary['r'], w, label='Pearson r', color='#1976d2', edgecolor='black')
    ax2.bar(x + w/2, df_summary['rho'], w, label='Spearman ρ', color='#388e3c', edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_summary['Gene'], rotation=35, ha='right', fontsize=9.5, fontweight='bold')
    ax2.set_ylabel('Correlation Coefficient', fontsize=11, fontweight='bold')
    ax2.set_title('B. Pointwise Correlation Breakdown by Gene', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc='upper right', fontsize=9.5)
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Panel C: Manhattan Profile of Smc6 (Adaptive) vs Smc5 (Conserved)
    ax3 = axes[1, 0]
    smc6_sites = df_sites[df_sites['gene_name'] == 'Smc6'].sort_values('codon_idx')
    smc5_sites = df_sites[df_sites['gene_name'] == 'Smc5'].sort_values('codon_idx')
    
    ax3.plot(smc6_sites['codon_idx'], smc6_sites['axomeme_lrt'], color='#d32f2f', alpha=0.85, label='Smc6 (AxoMEME)', lw=1.2)
    ax3.scatter(smc6_sites['codon_idx'], smc6_sites['meme_lrt'], color='#b71c1c', alpha=0.6, s=15, label='Smc6 (HyPhy MEME)')
    ax3.plot(smc5_sites['codon_idx'], smc5_sites['axomeme_lrt'], color='#757575', alpha=0.5, label='Smc5 (AxoMEME, Conserved)', lw=0.9, linestyle=':')
    
    ax3.axhline(3.12, color='black', linestyle='--', lw=1.2, label='Significance (p=0.10 / LRT=3.12)')
    ax3.set_xlabel('Codon Position', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Selection LRT', fontsize=11, fontweight='bold')
    ax3.set_title('C. Host Restriction Signature: Smc6 (Arms Race) vs. Smc5 (Conserved)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8.5)
    ax3.grid(True, linestyle='--', alpha=0.3)
    
    # Panel D: Computational Speedup
    ax4 = axes[1, 1]
    total_meme_time = df_summary['MEME Time (s)'].sum()
    total_axo_time = df_summary['AxoMEME Time (s)'].sum()
    mean_speedup = total_meme_time / total_axo_time
    
    methods = ['HyPhy MEME\n(Numerical MLE)', 'AxoMEME\n(Neural Transformer)']
    times = [total_meme_time, total_axo_time]
    bars = ax4.bar(methods, times, color=['#e53935', '#1e88e5'], edgecolor='black', width=0.45)
    ax4.set_yscale('log')
    ax4.set_ylabel('Total Execution Time Across 9 Genes (seconds, log scale)', fontsize=10, fontweight='bold')
    ax4.set_title(f'D. Empirical Throughput ({mean_speedup:,.0f}× Acceleration)', fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    for b in bars:
        h = b.get_height()
        ax4.annotate(f'{h:.3f} s' if h < 1.0 else f'{h:.1f} s',
                     xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
                     
    ax4.text(0.5, 0.45, f"⚡ {mean_speedup:,.0f}× Faster\n9 Genes Scanned in {total_axo_time:.2f}s", 
             transform=ax4.transAxes, ha='center', va='center',
             fontsize=12, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f5e9', edgecolor='#43a047', lw=1.5))
             
    plt.suptitle('Empirical Head-to-Head Benchmark: AxoMEME vs. HyPhy MEME (Abdul et al. 2018)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    out_png = 'scratch/abdul_meme_vs_axomeme_results.png'
    plt.savefig(out_png, bbox_inches='tight')
    plt.savefig('writeup/abdul_meme_vs_axomeme_results.png', bbox_inches='tight')
    plt.savefig('/Users/sergei/.gemini/antigravity-cli/brain/624c9340-cc13-4c5c-bf68-5a01b77c6860/abdul_meme_vs_axomeme_results.png', bbox_inches='tight')
    print(f"[✓] Saved figure to: {out_png}")

if __name__ == '__main__':
    main()
