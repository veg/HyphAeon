#!/usr/bin/env python3
"""
run_abdul_head_to_head.py

Head-to-head empirical comparison of AxoMEME vs. HyPhy MEME on the
Abdul et al. 2018 host restriction factor dataset (SMC1-6 and NSMCE1-4 complexes).
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
    inputs_dir = os.path.join(staging_dir, 'inputs')
    outputs_dir = os.path.join(staging_dir, 'outputs')
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    db_path = 'scratch/abdul_meme_vs_axomeme.db'
    ckpt_path = 'axomeme_5_dim384_nonull.pt'
    
    genes = sorted([f.replace('.fasta', '') for f in os.listdir(bench_dir) if f.endswith('.fasta')])
    print("=" * 95)
    print("🔬 EMPIRICAL BENCHMARK: HYPHY MEME VS. AXOMEME (ABDUL ET AL. 2018 DATASET)")
    print(f"   Genes ({len(genes)}): {', '.join(genes)}")
    print("=" * 95)
    
    # 1. Run local AxoMEME forward passes
    print("[1/5] Executing AxoMEME Forward Passes on Local System...")
    device = torch.device('cpu')
    torch.set_num_threads(2)
    model = PhyloAxialTransformer(embed_dim=384, num_layers=6, num_heads=12, window_size=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    gene_data = {}
    for g in genes:
        fa = os.path.join(bench_dir, f"{g}.fasta")
        nwk = os.path.join(bench_dir, f"{g}.nwk")
        
        lrts, inv, t_sec, L, N = run_axomeme_local(fa, nwk, model, device)
        gene_data[g] = {
            'axomeme_lrts': lrts,
            'is_invariable': inv,
            'axo_time': t_sec,
            'codons': L,
            'taxa': N
        }
        # Copy to staging inputs
        shutil_fa = os.path.join(inputs_dir, f"{g}.fasta")
        shutil_nwk = os.path.join(inputs_dir, f"{g}.nwk")
        with open(fa, 'r') as sf, open(shutil_fa, 'w') as df: df.write(sf.read())
        with open(nwk, 'r') as sf, open(shutil_nwk, 'w') as df: df.write(sf.read())
        print(f"    [AxoMEME] {g:<10} | N={N:2d}, L={L:4d} codons | Inference Time: {t_sec:.3f} s")
        
    # 2. Deploy HyPhy MEME to Silverback Cluster
    print("\n[2/5] Syncing Files to Silverback HPC Cluster...")
    subprocess.run(f"ssh {CLUSTER_HOST} 'mkdir -p {CLUSTER_DIR}/inputs {CLUSTER_DIR}/outputs'", shell=True, check=True)
    subprocess.run(f"rsync -avz -e 'ssh -o BatchMode=yes' {inputs_dir}/ {CLUSTER_HOST}:{CLUSTER_DIR}/inputs/", shell=True, check=True)
    
    # Slurm script
    gene_list_str = " ".join(genes)
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=abdul_meme
#SBATCH --partition=general
#SBATCH --array=0-{len(genes)-1}%9
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null

GENES=({gene_list_str})
GENE=${{GENES[${{SLURM_ARRAY_TASK_ID}}]}}

INPUT_FA="$HOME/abdul_bench/inputs/${{GENE}}.fasta"
INPUT_NWK="$HOME/abdul_bench/inputs/${{GENE}}.nwk"
OUTPUT_JSON="$HOME/abdul_bench/outputs/${{GENE}}.json"

hyphy meme --alignment "${{INPUT_FA}}" --tree "${{INPUT_NWK}}" --output "${{OUTPUT_JSON}}"
"""
    with open(os.path.join(staging_dir, 'submit_abdul.sh'), 'w') as f:
        f.write(slurm_script)
    subprocess.run(f"scp -o BatchMode=yes {staging_dir}/submit_abdul.sh {CLUSTER_HOST}:{CLUSTER_DIR}/submit_abdul.sh", shell=True, check=True)
    
    print("\n[3/5] Submitting Slurm Array on Silverback...")
    out_submit = subprocess.check_output(f"ssh {CLUSTER_HOST} 'cd {CLUSTER_DIR} && sbatch submit_abdul.sh'", shell=True).decode().strip()
    print(f"[*] Slurm Output: {out_submit}")
    job_id = out_submit.split()[-1]
    
    # 3. Monitor Slurm Execution
    print(f"\n[4/5] Monitoring Slurm Job Array #{job_id} on Silverback...")
    t0_clus = time.time()
    while True:
        try:
            line_count = int(subprocess.check_output(f"ssh {CLUSTER_HOST} 'squeue -j {job_id} | wc -l'", shell=True).decode().strip())
        except:
            line_count = 1
        elapsed = time.time() - t0_clus
        if line_count <= 1:
            print(f"\n[✓] All {len(genes)} HyPhy MEME runs completed in {elapsed:.1f}s ({elapsed/60:.2f} mins)!")
            break
        active = line_count - 1
        print(f"    [{elapsed:.0f}s elapsed] Active Tasks on Cluster: {active} / {len(genes)}...", end='\r', flush=True)
        time.sleep(6)
        
    # 4. Sync Outputs Back
    print("\n[5/5] Retrieving JSON Outputs & Ingesting into Database...")
    subprocess.run(f"rsync -avz -e 'ssh -o BatchMode=yes' {CLUSTER_HOST}:{CLUSTER_DIR}/outputs/ {outputs_dir}/", shell=True, check=True)
    
    # 5. Database Setup
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
    
    table_rows = []
    
    for g in genes:
        json_path = os.path.join(outputs_dir, f"{g}.json")
        L = gene_data[g]['codons']
        N = gene_data[g]['taxa']
        
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
        
        axo_lrts = gene_data[g]['axomeme_lrts']
        inv = gene_data[g]['is_invariable']
        axo_time = gene_data[g]['axo_time']
        
        # Statistics
        sp_r, _ = stats.spearmanr(axo_lrts, meme_lrts)
        if np.isnan(sp_r): sp_r = 0.0
        pe_r, _ = stats.pearsonr(axo_lrts, meme_lrts)
        if np.isnan(pe_r): pe_r = 0.0
        mae = float(np.mean(np.abs(axo_lrts - meme_lrts)))
        
        sig_meme_05 = int((meme_pvals <= 0.05).sum())
        sig_meme_10 = int((meme_pvals <= 0.10).sum())
        sig_axo_05 = int((axo_lrts >= 4.45).sum())
        sig_axo_10 = int((axo_lrts >= 3.12).sum())
        
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
            g, N, L, sp_r, pe_r, mae, sig_meme_05, sig_meme_10, sig_axo_05, sig_axo_10,
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
        
        table_rows.append({
            'Gene': g, 'Taxa': N, 'Codons': L, 'r': pe_r, 'rho': sp_r,
            'MEME Sig (p<=0.10)': sig_meme_10, 'AxoMEME Sig (p<=0.10)': sig_axo_10,
            'MEME Time': f"{meme_time:.1f}s", 'AxoMEME Time': f"{axo_time:.3f}s",
            'Speedup': f"{speedup:.0f}×"
        })
        
    conn.close()
    
    df_summary = pd.DataFrame(table_rows)
    print("\n" + "=" * 95)
    print("📊 EMPIRICAL RESULTS SUMMARY: ABDUL ET AL. 2018 (SMC5/6 HOST COMPLEX)")
    print("=" * 95)
    print(df_summary.to_string(index=False))
    print("=" * 95)

if __name__ == '__main__':
    main()
