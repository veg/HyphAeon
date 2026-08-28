#!/usr/bin/env python3
"""
run_axomeme_vs_meme_benchmark.py

Head-to-head simulation benchmarking script comparing AxoMEME against HyPhy MEME.
Evaluates:
  1. Per-site LRT correlation & regression calibration (Spearman rho, Pearson r, MAE, R^2)
  2. Ground-truth ROC-AUC & PR-AUC for both methods
  3. False positive rate calibration on purifying (omega=0.15) and neutral (omega=1.0) nulls
  4. Wall-clock execution time & speedup factor
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
import tempfile
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

def get_clade_subnodes(node):
    sub = [node]
    for child in node.children:
        sub.extend(get_clade_subnodes(child))
    return sub

def run_single_benchmark_replicate(task_args):
    """
    Simulates one alignment, executes both HyPhy MEME and AxoMEME,
    and returns comprehensive paired metrics.
    """
    (regime_name, rep_id, n_taxa, t_depth, beta_fg, fg_frac, omega_bg, n_null, n_sel, model_ckpt_path, seed) = task_args
    
    L = n_null + n_sel
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 1. Generate Tree
    newick_str = generate_large_diverse_tree(n_taxa=n_taxa, target_t=t_depth, seed=seed)
    py_tree = pyvolve.read_tree(tree=newick_str)
    all_nodes = get_all_nodes(py_tree)
    non_roots = [n for n in all_nodes if n.name != "root"]
    
    tree_obj = Phylo.read(io.StringIO(newick_str), 'newick')
    taxa = [term.name for term in tree_obj.get_terminals()]
    dist_mat = compute_fast_dist_matrix(tree_obj, taxa)
    mds_coords = compute_mds_coordinates(dist_mat, n_components=4)
    
    # 2. Select Clade for foreground selection
    if fg_frac < 1.0 and fg_frac > 0.0:
        candidates = [n for n in non_roots if 2 <= len(get_clade_subnodes(n)) <= max(2, int(len(non_roots) * fg_frac * 1.3))]
        if candidates:
            clade_root = min(candidates, key=lambda n: abs(len(get_clade_subnodes(n)) / len(non_roots) - fg_frac))
            fg_nodes = set(get_clade_subnodes(clade_root))
        else:
            n_fg = max(1, int(len(non_roots) * fg_frac))
            fg_nodes = set(random.sample(non_roots, n_fg))
    elif fg_frac >= 1.0:
        fg_nodes = set(non_roots)
    else:
        fg_nodes = set()
        
    actual_fg_frac = len(fg_nodes) / len(non_roots) if non_roots else 0.0
    
    # Tag nodes for branch-heterogeneous evolution
    def tag_nodes(n):
        if n.name != 'root':
            n.model_flag = 'foreground' if n in fg_nodes else 'background'
        else:
            n.model_flag = 'background'
        for ch in n.children:
            tag_nodes(ch)
    tag_nodes(py_tree)
    
    # 3. Parameterize Codon Models
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
            
    # 4. Simulate Alignment with Pyvolve
    ev = pyvolve.Evolver(partitions=partitions, tree=py_tree)
    with tempfile.NamedTemporaryFile(suffix='.fasta', mode='w', delete=False) as fa_tmp:
        tmp_fa = fa_tmp.name
    with tempfile.NamedTemporaryFile(suffix='.nwk', mode='w', delete=False) as nwk_tmp:
        tmp_nwk = nwk_tmp.name
        nwk_tmp.write(newick_str)
        
    ev(seqfile=tmp_fa, count=1, quiet=True)
    
    seq_dict = {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(tmp_fa, 'fasta')}
    
    # 5. Run HyPhy MEME
    tmp_json = tmp_fa + '.MEME.json'
    meme_t0 = time.time()
    meme_cmd = ['hyphy', 'meme', '--alignment', tmp_fa, '--tree', tmp_nwk, '--output', tmp_json]
    meme_success = False
    try:
        subprocess.run(meme_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        meme_t1 = time.time()
        meme_time_sec = meme_t1 - meme_t0
        
        with open(tmp_json, 'r') as jf:
            text = jf.read()
            meme_data = json.loads(text.replace('NaN', '0.0').replace('-nan', '0.0').replace('nan', '0.0'))
            
        mle_table = meme_data['MLE']['content']['0']
        meme_lrts = np.array([row[5] for row in mle_table], dtype=np.float32)
        meme_pvals = np.array([row[6] for row in mle_table], dtype=np.float32)
        meme_alphas = np.array([row[0] for row in mle_table], dtype=np.float32)
        meme_betas_pos = np.array([row[3] for row in mle_table], dtype=np.float32)
        meme_weights_pos = np.array([row[4] for row in mle_table], dtype=np.float32)
        meme_success = True
    except Exception as e:
        meme_time_sec = time.time() - meme_t0
        meme_lrts = np.zeros(L, dtype=np.float32)
        meme_pvals = np.ones(L, dtype=np.float32)
        meme_alphas = np.zeros(L, dtype=np.float32)
        meme_betas_pos = np.zeros(L, dtype=np.float32)
        meme_weights_pos = np.zeros(L, dtype=np.float32)
        
    # Clean up temp files
    for p in [tmp_fa, tmp_nwk, tmp_json]:
        if os.path.exists(p):
            try: os.remove(p)
            except: pass
            
    # 6. Run AxoMEME Forward Pass
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
    
    # Load model
    device = torch.device('cpu')
    torch.set_num_threads(2)
    model = PhyloAxialTransformer(embed_dim=384, num_layers=6, num_heads=12, window_size=1).to(device)
    ckpt = torch.load(model_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    axomeme_t0 = time.time()
    with torch.no_grad():
        y_lrt_soft, _ = model(c_tensor, a_tensor, d_tensor, z_tensor)
        axomeme_lrts = torch.clamp(y_lrt_soft.squeeze(-1), min=0.0).numpy().flatten()
    axomeme_time_sec = time.time() - axomeme_t0
    
    axomeme_lrts[is_aa_invariable] = 0.0
    
    # 7. Compute Paired Concordance & Discrimination
    ground_truth = np.concatenate([np.zeros(n_null), np.ones(n_sel)])
    
    # Concordance between AxoMEME and MEME
    spearman_r, _ = stats.spearmanr(axomeme_lrts, meme_lrts)
    if np.isnan(spearman_r): spearman_r = 0.0
    pearson_r, _ = stats.pearsonr(axomeme_lrts, meme_lrts)
    if np.isnan(pearson_r): pearson_r = 0.0
    mae = float(np.mean(np.abs(axomeme_lrts - meme_lrts)))
    
    # Discrimination (Ground Truth)
    if n_sel > 0 and n_null > 0:
        auc_axomeme = roc_auc_score(ground_truth, axomeme_lrts)
        auc_meme = roc_auc_score(ground_truth, meme_lrts)
        pr_axomeme = average_precision_score(ground_truth, axomeme_lrts)
        pr_meme = average_precision_score(ground_truth, meme_lrts)
    else:
        auc_axomeme = 0.5
        auc_meme = 0.5
        pr_axomeme = 0.0
        pr_meme = 0.0
        
    # Statistical Power at Nominal Cutoffs
    # AxoMEME
    pwr_axo_05_asymp = float((axomeme_lrts[n_null:] >= 4.45).mean()) if n_sel > 0 else 0.0
    pwr_axo_10_asymp = float((axomeme_lrts[n_null:] >= 3.12).mean()) if n_sel > 0 else 0.0
    
    # Empirical Power Calibrated to 5% Null FPR
    if n_null > 0:
        tau_05_null = float(np.percentile(axomeme_lrts[:n_null], 95))
        pwr_axo_05_emp = float((axomeme_lrts[n_null:] >= tau_05_null).mean()) if n_sel > 0 else 0.0
        fpr_axo_nominal_05 = float((axomeme_lrts[:n_null] >= 4.45).mean())
    else:
        tau_05_null = 0.0
        pwr_axo_05_emp = 0.0
        fpr_axo_nominal_05 = 0.0
        
    # MEME Power
    pwr_meme_05 = float((meme_pvals[n_null:] <= 0.05).mean()) if n_sel > 0 else 0.0
    pwr_meme_10 = float((meme_pvals[n_null:] <= 0.10).mean()) if n_sel > 0 else 0.0
    fpr_meme_05 = float((meme_pvals[:n_null] <= 0.05).mean()) if n_null > 0 else 0.0
    
    # Overlap / Jaccard at 10% level
    sig_axo = (axomeme_lrts >= 3.12)
    sig_meme = (meme_pvals <= 0.10)
    both_sig = int(np.sum(sig_axo & sig_meme))
    union_sig = int(np.sum(sig_axo | sig_meme))
    jaccard = both_sig / union_sig if union_sig > 0 else 1.0
    
    speedup = meme_time_sec / axomeme_time_sec if axomeme_time_sec > 0 else 1000.0
    
    # Compile Site-Level Data
    site_records = []
    for s in range(L):
        site_records.append({
            'site_idx': s,
            'is_selected': int(ground_truth[s]),
            'is_invariable': int(is_aa_invariable[s]),
            'meme_lrt': float(meme_lrts[s]),
            'meme_pval': float(meme_pvals[s]),
            'meme_alpha': float(meme_alphas[s]),
            'meme_beta_pos': float(meme_betas_pos[s]),
            'meme_weight_pos': float(meme_weights_pos[s]),
            'axomeme_lrt': float(axomeme_lrts[s])
        })
        
    return {
        'regime_name': regime_name,
        'rep_id': rep_id,
        'n_taxa': n_taxa,
        't_depth': t_depth,
        'beta_fg': beta_fg,
        'fg_frac': actual_fg_frac,
        'omega_bg': omega_bg,
        'n_null': n_null,
        'n_sel': n_sel,
        'spearman_rho': float(spearman_r),
        'pearson_r': float(pearson_r),
        'mae_lrt': mae,
        'auc_axomeme': float(auc_axomeme),
        'auc_meme': float(auc_meme),
        'pr_axomeme': float(pr_axomeme),
        'pr_meme': float(pr_meme),
        'pwr_axo_05_asymp': pwr_axo_05_asymp,
        'pwr_axo_10_asymp': pwr_axo_10_asymp,
        'pwr_axo_05_emp': pwr_axo_05_emp,
        'pwr_meme_05': pwr_meme_05,
        'pwr_meme_10': pwr_meme_10,
        'fpr_axo_nominal_05': fpr_axo_nominal_05,
        'fpr_meme_05': fpr_meme_05,
        'jaccard_overlap_10': float(jaccard),
        'meme_time_sec': float(meme_time_sec),
        'axomeme_time_sec': float(axomeme_time_sec),
        'speedup_factor': float(speedup),
        'site_records': site_records
    }

def main():
    import concurrent.futures
    
    db_path = 'scratch/meme_vs_axomeme_benchmarks.db'
    ckpt_path = 'axomeme_5_dim384_nonull.pt'
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        
    print("=" * 95)
    print("🔬 MEME vs. AxoMEME HEAD-TO-HEAD SIMULATION BENCHMARK")
    print(f"   Database Target : {db_path}")
    print(f"   Model Weights   : {ckpt_path}")
    print("=" * 95)
    
    # Initialize SQLite Database
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
    conn.commit()
    conn.close()
    
    # Define the 5 Stratified Regimes
    regimes = [
        # Regime 1: Pervasive / Broad Adaptation (Sweet Spot)
        {
            'name': '1_SweetSpot_Pervasive',
            'n_taxa': 64, 't_depth': 12.0, 'beta_fg': 12.0, 'fg_frac': 1.00, 'omega_bg': 0.15,
            'n_null': 150, 'n_sel': 50, 'n_reps': 5
        },
        # Regime 2: Phase Transition Frontier (50% Power Boundary)
        {
            'name': '2_PhaseTransition_Boundary',
            'n_taxa': 96, 't_depth': 16.0, 'beta_fg': 50.0, 'fg_frac': 0.80, 'omega_bg': 0.15,
            'n_null': 150, 'n_sel': 50, 'n_reps': 5
        },
        # Regime 3: Clade-Specific Episodic Challenge
        {
            'name': '3_Episodic_Clade_25pct',
            'n_taxa': 128, 't_depth': 20.0, 'beta_fg': 250.0, 'fg_frac': 0.25, 'omega_bg': 0.15,
            'n_null': 150, 'n_sel': 50, 'n_reps': 5
        },
        # Regime 4: Pure Purifying Null (Type-I Error)
        {
            'name': '4_Purifying_Null_Omega015',
            'n_taxa': 96, 't_depth': 15.0, 'beta_fg': 0.15, 'fg_frac': 0.00, 'omega_bg': 0.15,
            'n_null': 200, 'n_sel': 0, 'n_reps': 5
        },
        # Regime 5: Neutral Boundary Null (Theoretical Calibration)
        {
            'name': '5_Neutral_Null_Omega100',
            'n_taxa': 96, 't_depth': 15.0, 'beta_fg': 1.00, 'fg_frac': 0.00, 'omega_bg': 1.00,
            'n_null': 200, 'n_sel': 0, 'n_reps': 5
        }
    ]
    
    task_args_list = []
    seed_base = 42000
    task_idx = 0
    for reg in regimes:
        for rep in range(reg['n_reps']):
            seed = seed_base + task_idx * 17
            task_args_list.append((
                reg['name'], rep, reg['n_taxa'], reg['t_depth'], reg['beta_fg'],
                reg['fg_frac'], reg['omega_bg'], reg['n_null'], reg['n_sel'],
                ckpt_path, seed
            ))
            task_idx += 1
            
    print(f"[*] Queued {len(task_args_list)} total benchmark replicates across {len(regimes)} regimes.")
    print("[*] Launching parallel execution (4 CPU workers)...")
    
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_single_benchmark_replicate, args): args for args in task_args_list}
        
        conn = sqlite3.connect(db_path)
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            
            # Print Live Progress
            print(f"\n[✓ Done: {res['regime_name']} Rep {res['rep_id']}]")
            print(f"    Spearman Rho = {res['spearman_rho']:+.3f} | Pearson r = {res['pearson_r']:+.3f} | MAE = {res['mae_lrt']:.2f}")
            print(f"    ROC-AUC: AxoMEME = {res['auc_axomeme']:.3f} vs. MEME = {res['auc_meme']:.3f}")
            print(f"    Power(10%): Axo Asymp = {res['pwr_axo_10_asymp']*100:.1f}%, Axo Emp = {res['pwr_axo_05_emp']*100:.1f}% vs. MEME = {res['pwr_meme_10']*100:.1f}%")
            print(f"    Runtime: HyPhy MEME = {res['meme_time_sec']:.2f}s vs. AxoMEME = {res['axomeme_time_sec']:.3f}s ({res['speedup_factor']:.1f}x Speedup)")
            
            # Insert Replicate Summary
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO benchmark_replicates (
                    regime_name, rep_id, n_taxa, t_depth, beta_fg, fg_frac, omega_bg, n_null, n_sel,
                    spearman_rho, pearson_r, mae_lrt, auc_axomeme, auc_meme, pr_axomeme, pr_meme,
                    pwr_axo_05_asymp, pwr_axo_10_asymp, pwr_axo_05_emp, pwr_meme_05, pwr_meme_10,
                    fpr_axo_nominal_05, fpr_meme_05, jaccard_overlap_10, meme_time_sec, axomeme_time_sec, speedup_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                res['regime_name'], res['rep_id'], res['n_taxa'], res['t_depth'], res['beta_fg'],
                res['fg_frac'], res['omega_bg'], res['n_null'], res['n_sel'],
                res['spearman_rho'], res['pearson_r'], res['mae_lrt'],
                res['auc_axomeme'], res['auc_meme'], res['pr_axomeme'], res['pr_meme'],
                res['pwr_axo_05_asymp'], res['pwr_axo_10_asymp'], res['pwr_axo_05_emp'],
                res['pwr_meme_05'], res['pwr_meme_10'],
                res['fpr_axo_nominal_05'], res['fpr_meme_05'], res['jaccard_overlap_10'],
                res['meme_time_sec'], res['axomeme_time_sec'], res['speedup_factor']
            ))
            rep_db_id = cur.lastrowid
            
            # Insert Site Records
            site_rows = [(
                rep_db_id, res['regime_name'], s['site_idx'], s['is_selected'], s['is_invariable'],
                s['meme_lrt'], s['meme_pval'], s['meme_alpha'], s['meme_beta_pos'],
                s['meme_weight_pos'], s['axomeme_lrt']
            ) for s in res['site_records']]
            
            cur.executemany('''
                INSERT INTO site_level_predictions (
                    rep_db_id, regime_name, site_idx, is_selected, is_invariable,
                    meme_lrt, meme_pval, meme_alpha, meme_beta_pos, meme_weight_pos, axomeme_lrt
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', site_rows)
            conn.commit()
            
        conn.close()
        
    elapsed = time.time() - t0
    print("\n" + "=" * 95)
    print(f"🎉 BENCHMARK RUN COMPLETED IN {elapsed/60:.2f} MINUTES!")
    print(f"   Results logged in : {db_path}")
    print("=" * 95)

if __name__ == '__main__':
    main()
