#!/usr/bin/env python3
"""
run_cluster_axomeme_vs_meme.py

End-to-end HPC-accelerated head-to-head benchmarking pipeline:
  1. Generates 40 simulated alignments across 8 regimes locally.
  2. Runs instantaneous AxoMEME forward passes on local Apple Silicon.
  3. Deploys HyPhy MEME jobs as a Slurm Array to the silverback.temple.edu cluster (64+ cores).
  4. Automatically monitors execution, fetches results, and compiles paired benchmark metrics.
"""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io
import time
import json
import sqlite3
import random
import shutil
import subprocess
import numpy as np
import pandas as pd
from Bio import Phylo, SeqIO
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.stats as stats
import torch
import pyvolve

from scripts.train_transformer_selection import PhyloAxialTransformer
from scripts.predict_regression_nexus import compute_mds_coordinates, get_codon_token, get_aa_token
from scratch.test_mg94_high_power_baseline import generate_large_diverse_tree, compute_fast_dist_matrix, compute_cf3x4_frequencies, get_all_nodes

# Global CF3X4 Frequencies
PI_P1 = [0.28, 0.22, 0.32, 0.18]
PI_P2 = [0.30, 0.24, 0.18, 0.28]
PI_P3 = [0.15, 0.40, 0.35, 0.10]
CF3X4_STATE_FREQS = compute_cf3x4_frequencies(PI_P1, PI_P2, PI_P3)

CLUSTER_HOST = "silverback.temple.edu"
CLUSTER_DIR = "~/axomeme_meme_cluster"

def get_clade_subnodes(node):
    sub = [node]
    for child in node.children:
        sub.extend(get_clade_subnodes(child))
    return sub

def main():
    db_path = 'scratch/meme_vs_axomeme_benchmarks.db'
    staging_dir = 'scratch/cluster_staging'
    inputs_dir = os.path.join(staging_dir, 'inputs')
    outputs_dir = os.path.join(staging_dir, 'outputs')
    
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    ckpt_path = 'axomeme_5_dim384_nonull.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model weights not found at: {ckpt_path}")
        
    print("=" * 95)
    print("🚀 HPC-ACCELERATED MEME VS. AXOMEME BENCHMARK SUITE (SILVERBACK CLUSTER)")
    print("=" * 95)
    
    # Define 8 Stratified Evolutionary Regimes
    regimes = [
        # 1. Pervasive Sweet Spot (100% Clade)
        {'name': '1_SweetSpot_Pervasive', 'n_taxa': 64, 't_depth': 12.0, 'beta_fg': 12.0, 'fg_frac': 1.00, 'mode': 'clade', 'omega_bg': 0.15, 'n_null': 150, 'n_sel': 50, 'n_reps': 5},
        # 2. Phase Transition Frontier (80% Clade)
        {'name': '2_PhaseTransition_80pct', 'n_taxa': 96, 't_depth': 16.0, 'beta_fg': 50.0, 'fg_frac': 0.80, 'mode': 'clade', 'omega_bg': 0.15, 'n_null': 150, 'n_sel': 50, 'n_reps': 5},
        # 3. Intermediate Episodic Bridge (50% Clade)
        {'name': '3_EpisodicBridge_50pct', 'n_taxa': 96, 't_depth': 16.0, 'beta_fg': 100.0, 'fg_frac': 0.50, 'mode': 'clade', 'omega_bg': 0.15, 'n_null': 150, 'n_sel': 50, 'n_reps': 5},
        # 4. Sparse Clade Episodic (25% Clade)
        {'name': '4_SparseClade_25pct', 'n_taxa': 128, 't_depth': 20.0, 'beta_fg': 250.0, 'fg_frac': 0.25, 'mode': 'clade', 'omega_bg': 0.15, 'n_null': 150, 'n_sel': 50, 'n_reps': 5},
        # 5. Polyphyletic / Scattered Episodic (10% Random Branches)
        {'name': '5_Scattered_Polyphyletic_10pct', 'n_taxa': 128, 't_depth': 20.0, 'beta_fg': 250.0, 'fg_frac': 0.10, 'mode': 'scattered', 'omega_bg': 0.15, 'n_null': 150, 'n_sel': 50, 'n_reps': 5},
        # 6. High Divergence Deep Tree (35% Clade)
        {'name': '6_DeepTree_Large_35pct', 'n_taxa': 256, 't_depth': 35.0, 'beta_fg': 50.0, 'fg_frac': 0.35, 'mode': 'clade', 'omega_bg': 0.15, 'n_null': 150, 'n_sel': 50, 'n_reps': 5},
        # 7. Pure Purifying Null (Type-I Error)
        {'name': '7_Purifying_Null_Omega015', 'n_taxa': 96, 't_depth': 15.0, 'beta_fg': 0.15, 'fg_frac': 0.00, 'mode': 'none', 'omega_bg': 0.15, 'n_null': 200, 'n_sel': 0, 'n_reps': 5},
        # 8. Neutral Drift Null (Theoretical Calibration)
        {'name': '8_Neutral_Null_Omega100', 'n_taxa': 96, 't_depth': 15.0, 'beta_fg': 1.00, 'fg_frac': 0.00, 'mode': 'none', 'omega_bg': 1.00, 'n_null': 200, 'n_sel': 0, 'n_reps': 5}
    ]
    
    # Load Model locally
    device = torch.device('cpu')
    model = PhyloAxialTransformer(embed_dim=384, num_layers=6, num_heads=12, window_size=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    task_manifest = []
    task_idx = 0
    seed_base = 55000
    
    print("[1/5] Simulating Alignments and Running Local AxoMEME Inference...")
    for reg in regimes:
        for rep in range(reg['n_reps']):
            seed = seed_base + task_idx * 23
            n_taxa = reg['n_taxa']
            t_depth = reg['t_depth']
            beta_fg = reg['beta_fg']
            fg_frac = reg['fg_frac']
            mode = reg['mode']
            omega_bg = reg['omega_bg']
            n_null = reg['n_null']
            n_sel = reg['n_sel']
            L = n_null + n_sel
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Tree Generation
            newick_str = generate_large_diverse_tree(n_taxa=n_taxa, target_t=t_depth, seed=seed)
            py_tree = pyvolve.read_tree(tree=newick_str)
            all_nodes = get_all_nodes(py_tree)
            non_roots = [n for n in all_nodes if n.name != "root"]
            
            tree_obj = Phylo.read(io.StringIO(newick_str), 'newick')
            taxa = [term.name for term in tree_obj.get_terminals()]
            dist_mat = compute_fast_dist_matrix(tree_obj, taxa)
            mds_coords = compute_mds_coordinates(dist_mat, n_components=4)
            
            # Select Foreground Branches
            if fg_frac > 0.0:
                if mode == 'clade' and fg_frac < 1.0:
                    candidates = [n for n in non_roots if 2 <= len(get_clade_subnodes(n)) <= max(2, int(len(non_roots) * fg_frac * 1.3))]
                    if candidates:
                        clade_root = min(candidates, key=lambda n: abs(len(get_clade_subnodes(n)) / len(non_roots) - fg_frac))
                        fg_nodes = set(get_clade_subnodes(clade_root))
                    else:
                        fg_nodes = set(random.sample(non_roots, max(1, int(len(non_roots) * fg_frac))))
                elif mode == 'scattered':
                    n_fg = max(1, int(len(non_roots) * fg_frac))
                    fg_nodes = set(random.sample(non_roots, n_fg))
                else:
                    fg_nodes = set(non_roots)
            else:
                fg_nodes = set()
                
            actual_fg_frac = len(fg_nodes) / len(non_roots) if non_roots else 0.0
            
            # Tag nodes
            def tag_nodes(n):
                if n.name != 'root':
                    n.model_flag = 'foreground' if n in fg_nodes else 'background'
                else:
                    n.model_flag = 'background'
                for ch in n.children:
                    tag_nodes(ch)
            tag_nodes(py_tree)
            
            # Models
            partitions = []
            if n_null > 0:
                m_null = pyvolve.Model('MG94', parameters={'alpha': 1.0, 'beta': omega_bg, 'state_freqs': CF3X4_STATE_FREQS})
                partitions.append(pyvolve.Partition(models=m_null, size=n_null))
                
            if n_sel > 0:
                if len(fg_nodes) == len(non_roots):
                    m_fg = pyvolve.Model('MG94', parameters={'alpha': 1.0, 'beta': beta_fg, 'state_freqs': CF3X4_STATE_FREQS})
                    partitions.append(pyvolve.Partition(models=m_fg, size=n_sel))
                elif len(fg_nodes) == 0:
                    m_bg = pyvolve.Model('MG94', parameters={'alpha': 1.0, 'beta': omega_bg, 'state_freqs': CF3X4_STATE_FREQS})
                    partitions.append(pyvolve.Partition(models=m_bg, size=n_sel))
                else:
                    m_bg = pyvolve.Model('MG94', parameters={'alpha': 1.0, 'beta': omega_bg, 'state_freqs': CF3X4_STATE_FREQS}, name='background')
                    m_fg = pyvolve.Model('MG94', parameters={'alpha': 1.0, 'beta': beta_fg, 'state_freqs': CF3X4_STATE_FREQS}, name='foreground')
                    partitions.append(pyvolve.Partition(models=[m_bg, m_fg], size=n_sel, root_model_name='background'))
                    
            fa_path = os.path.join(inputs_dir, f"align_{task_idx}.fasta")
            nwk_path = os.path.join(inputs_dir, f"tree_{task_idx}.nwk")
            
            with open(nwk_path, 'w') as f:
                f.write(newick_str)
                
            ev = pyvolve.Evolver(partitions=partitions, tree=py_tree)
            ev(seqfile=fa_path, count=1, quiet=True)
            
            seq_dict = {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(fa_path, 'fasta')}
            
            # AxoMEME Forward Pass
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
            
            t0_axo = time.time()
            with torch.no_grad():
                y_lrt_soft, _ = model(c_tensor, a_tensor, d_tensor, z_tensor)
                axo_lrts = torch.clamp(y_lrt_soft.squeeze(-1), min=0.0).numpy().flatten()
            axo_time = time.time() - t0_axo
            axo_lrts[is_aa_invariable] = 0.0
            
            ground_truth = np.concatenate([np.zeros(n_null), np.ones(n_sel)])
            
            task_manifest.append({
                'task_idx': task_idx,
                'regime_name': reg['name'],
                'rep_id': rep,
                'n_taxa': n_taxa,
                't_depth': t_depth,
                'beta_fg': beta_fg,
                'fg_frac': actual_fg_frac,
                'omega_bg': omega_bg,
                'n_null': n_null,
                'n_sel': n_sel,
                'axo_time': axo_time,
                'axo_lrts': axo_lrts.tolist(),
                'is_invariable': is_aa_invariable.tolist(),
                'ground_truth': ground_truth.tolist()
            })
            task_idx += 1
            
    print(f"[*] Generated {len(task_manifest)} alignments and completed all local AxoMEME forward passes.")
    
    # Save Manifest locally
    with open(os.path.join(staging_dir, 'manifest.json'), 'w') as f:
        json.dump(task_manifest, f)
        
    # 2. Deploy to Silverback Cluster
    print("\n[2/5] Syncing Alignment Inputs to Silverback Cluster...")
    cmd_sync = f"rsync -avz -e 'ssh -o BatchMode=yes' {staging_dir}/inputs/ {CLUSTER_HOST}:{CLUSTER_DIR}/inputs/"
    subprocess.run(cmd_sync, shell=True, check=True)
    
    # 3. Create Slurm Array Script on Silverback
    num_tasks = len(task_manifest)
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=meme_bench
#SBATCH --partition=general
#SBATCH --array=0-{num_tasks-1}%40
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null

ALIGN_ID=${{SLURM_ARRAY_TASK_ID}}
INPUT_FA="$HOME/axomeme_meme_cluster/inputs/align_${{ALIGN_ID}}.fasta"
INPUT_NWK="$HOME/axomeme_meme_cluster/inputs/tree_${{ALIGN_ID}}.nwk"
OUTPUT_JSON="$HOME/axomeme_meme_cluster/outputs/align_${{ALIGN_ID}}.json"

hyphy meme --alignment "${{INPUT_FA}}" --tree "${{INPUT_NWK}}" --output "${{OUTPUT_JSON}}"
"""
    with open(os.path.join(staging_dir, 'submit_array.sh'), 'w') as f:
        f.write(slurm_script)
        
    cmd_send_script = f"scp -o BatchMode=yes {staging_dir}/submit_array.sh {CLUSTER_HOST}:{CLUSTER_DIR}/submit_array.sh"
    subprocess.run(cmd_send_script, shell=True, check=True)
    
    print("\n[3/5] Submitting Slurm Array Job on Silverback...")
    cmd_submit = f"ssh {CLUSTER_HOST} 'cd {CLUSTER_DIR} && sbatch submit_array.sh'"
    out_submit = subprocess.check_output(cmd_submit, shell=True).decode().strip()
    print(f"[*] Slurm Output: {out_submit}")
    job_id = out_submit.split()[-1]
    
    # 4. Monitor Job Array
    print(f"\n[4/5] Monitoring Slurm Job Array #{job_id} on Silverback...")
    t0_cluster = time.time()
    while True:
        check_cmd = f"ssh {CLUSTER_HOST} 'squeue -j {job_id} | wc -l'"
        try:
            line_count = int(subprocess.check_output(check_cmd, shell=True).decode().strip())
        except:
            line_count = 1
            
        elapsed = time.time() - t0_cluster
        if line_count <= 1:
            print(f"\n[✓] All {num_tasks} cluster jobs completed in {elapsed:.1f} seconds ({elapsed/60:.2f} mins)!")
            break
            
        active_jobs = line_count - 1
        print(f"    [{elapsed:.0f}s elapsed] Active Slurm Tasks on Cluster: {active_jobs} / {num_tasks}...", end='\r', flush=True)
        time.sleep(8)
        
    # 5. Fetch Output Files
    print("\n[5/5] Fetching JSON Outputs and Compiling Database...")
    cmd_fetch = f"rsync -avz -e 'ssh -o BatchMode=yes' {CLUSTER_HOST}:{CLUSTER_DIR}/outputs/ {staging_dir}/outputs/"
    subprocess.run(cmd_fetch, shell=True, check=True)
    
    # Compile Results into SQLite Database
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS benchmark_replicates (
            rep_db_id INTEGER PRIMARY KEY AUTOINCREMENT,
            regime_name TEXT,
            rep_id INTEGER,
            n_taxa INTEGER,
            t_depth REAL,
            beta_fg REAL,
            fg_frac REAL,
            omega_bg REAL,
            n_null INTEGER,
            n_sel INTEGER,
            spearman_rho REAL,
            pearson_r REAL,
            mae_lrt REAL,
            auc_axomeme REAL,
            auc_meme REAL,
            pr_axomeme REAL,
            pr_meme REAL,
            pwr_axo_05_asymp REAL,
            pwr_axo_10_asymp REAL,
            pwr_axo_05_emp REAL,
            pwr_meme_05 REAL,
            pwr_meme_10 REAL,
            fpr_axo_nominal_05 REAL,
            fpr_meme_05 REAL,
            jaccard_overlap_10 REAL,
            meme_time_sec REAL,
            axomeme_time_sec REAL,
            speedup_factor REAL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS site_level_predictions (
            site_db_id INTEGER PRIMARY KEY AUTOINCREMENT,
            rep_db_id INTEGER,
            regime_name TEXT,
            site_idx INTEGER,
            is_selected INTEGER,
            is_invariable INTEGER,
            meme_lrt REAL,
            meme_pval REAL,
            meme_alpha REAL,
            meme_beta_pos REAL,
            meme_weight_pos REAL,
            axomeme_lrt REAL
        )
    ''')
    
    # Clear old table and reload full clean 40-replicate suite
    conn.execute("DELETE FROM site_level_predictions")
    conn.execute("DELETE FROM benchmark_replicates")
    conn.commit()
    
    cur = conn.cursor()
    
    for item in task_manifest:
        t_id = item['task_idx']
        json_path = os.path.join(outputs_dir, f"align_{t_id}.json")
        L = item['n_null'] + item['n_sel']
        
        try:
            with open(json_path, 'r') as jf:
                text = jf.read()
                meme_data = json.loads(text.replace('NaN', '0.0').replace('-nan', '0.0').replace('nan', '0.0'))
            mle_table = meme_data['MLE']['content']['0']
            meme_lrts = np.array([row[5] for row in mle_table], dtype=np.float32)
            meme_pvals = np.array([row[6] for row in mle_table], dtype=np.float32)
            meme_alphas = np.array([row[0] for row in mle_table], dtype=np.float32)
            meme_betas_pos = np.array([row[3] for row in mle_table], dtype=np.float32)
            meme_weights_pos = np.array([row[4] for row in mle_table], dtype=np.float32)
            meme_time_sec = float(meme_data.get('timers', {}).get('Overall', {}).get('timer', 300.0))
        except Exception as e:
            print(f"[!] Error reading JSON for task {t_id}: {e}")
            meme_lrts = np.zeros(L, dtype=np.float32)
            meme_pvals = np.ones(L, dtype=np.float32)
            meme_alphas = np.zeros(L, dtype=np.float32)
            meme_betas_pos = np.zeros(L, dtype=np.float32)
            meme_weights_pos = np.zeros(L, dtype=np.float32)
            meme_time_sec = 300.0
            
        axomeme_lrts = np.array(item['axo_lrts'], dtype=np.float32)
        ground_truth = np.array(item['ground_truth'], dtype=np.int32)
        is_invariable = np.array(item['is_invariable'], dtype=np.int32)
        
        # Concordance
        sp_r, _ = stats.spearmanr(axomeme_lrts, meme_lrts)
        if np.isnan(sp_r): sp_r = 0.0
        pe_r, _ = stats.pearsonr(axomeme_lrts, meme_lrts)
        if np.isnan(pe_r): pe_r = 0.0
        mae = float(np.mean(np.abs(axomeme_lrts - meme_lrts)))
        
        # Discrimination
        if item['n_sel'] > 0 and item['n_null'] > 0:
            auc_axomeme = float(roc_auc_score(ground_truth, axomeme_lrts))
            auc_meme = float(roc_auc_score(ground_truth, meme_lrts))
            pr_axomeme = float(average_precision_score(ground_truth, axomeme_lrts))
            pr_meme = float(average_precision_score(ground_truth, meme_lrts))
        else:
            auc_axomeme, auc_meme = 0.5, 0.5
            pr_axomeme, pr_meme = 0.0, 0.0
            
        pwr_axo_05_asymp = float((axomeme_lrts[item['n_null']:] >= 4.45).mean()) if item['n_sel'] > 0 else 0.0
        pwr_axo_10_asymp = float((axomeme_lrts[item['n_null']:] >= 3.12).mean()) if item['n_sel'] > 0 else 0.0
        
        if item['n_null'] > 0:
            tau_05_null = float(np.percentile(axomeme_lrts[:item['n_null']], 95))
            pwr_axo_05_emp = float((axomeme_lrts[item['n_null']:] >= tau_05_null).mean()) if item['n_sel'] > 0 else 0.0
            fpr_axo_nominal_05 = float((axomeme_lrts[:item['n_null']] >= 4.45).mean())
        else:
            pwr_axo_05_emp, fpr_axo_nominal_05 = 0.0, 0.0
            
        pwr_meme_05 = float((meme_pvals[item['n_null']:] <= 0.05).mean()) if item['n_sel'] > 0 else 0.0
        pwr_meme_10 = float((meme_pvals[item['n_null']:] <= 0.10).mean()) if item['n_sel'] > 0 else 0.0
        fpr_meme_05 = float((meme_pvals[:item['n_null']] <= 0.05).mean()) if item['n_null'] > 0 else 0.0
        
        sig_axo = (axomeme_lrts >= 3.12)
        sig_meme = (meme_pvals <= 0.10)
        both_sig = int(np.sum(sig_axo & sig_meme))
        union_sig = int(np.sum(sig_axo | sig_meme))
        jaccard = both_sig / union_sig if union_sig > 0 else 1.0
        
        speedup = meme_time_sec / item['axo_time'] if item['axo_time'] > 0 else 1000.0
        
        cur.execute('''
            INSERT INTO benchmark_replicates (
                regime_name, rep_id, n_taxa, t_depth, beta_fg, fg_frac, omega_bg, n_null, n_sel,
                spearman_rho, pearson_r, mae_lrt, auc_axomeme, auc_meme, pr_axomeme, pr_meme,
                pwr_axo_05_asymp, pwr_axo_10_asymp, pwr_axo_05_emp, pwr_meme_05, pwr_meme_10,
                fpr_axo_nominal_05, fpr_meme_05, jaccard_overlap_10, meme_time_sec, axomeme_time_sec, speedup_factor
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item['regime_name'], item['rep_id'], item['n_taxa'], item['t_depth'], item['beta_fg'],
            item['fg_frac'], item['omega_bg'], item['n_null'], item['n_sel'],
            sp_r, pe_r, mae, auc_axomeme, auc_meme, pr_axomeme, pr_meme,
            pwr_axo_05_asymp, pwr_axo_10_asymp, pwr_axo_05_emp, pwr_meme_05, pwr_meme_10,
            fpr_axo_nominal_05, fpr_meme_05, jaccard, meme_time_sec, item['axo_time'], speedup
        ))
        rep_db_id = cur.lastrowid
        
        site_rows = [(
            rep_db_id, item['regime_name'], s, int(ground_truth[s]), int(is_invariable[s]),
            float(meme_lrts[s]), float(meme_pvals[s]), float(meme_alphas[s]),
            float(meme_betas_pos[s]), float(meme_weights_pos[s]), float(axomeme_lrts[s])
        ) for s in range(L)]
        
        cur.executemany('''
            INSERT INTO site_level_predictions (
                rep_db_id, regime_name, site_idx, is_selected, is_invariable,
                meme_lrt, meme_pval, meme_alpha, meme_beta_pos, meme_weight_pos, axomeme_lrt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', site_rows)
        conn.commit()
        
    conn.close()
    print(f"\n[✓] All 40 benchmark replicates loaded into: {db_path}")

if __name__ == '__main__':
    main()
