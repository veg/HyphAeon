#!/usr/bin/env python3
import sys
import os
import glob
import json
import subprocess
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_dataset(base_name, nex_path, meme_path, model_path="./axomeme_3.0.pt"):
    # 1. Run inference
    csv_out = f"scratch/{base_name}_predictions.csv"
    cmd = [
        "python3", "scripts/predict_regression_nexus.py",
        "--model", model_path,
        "--alignment", nex_path,
        "--output", csv_out,
        "--device", "cpu"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(csv_out):
        return {
            'dataset': base_name,
            'status': 'FAIL',
            'error': res.stderr[:100]
        }
        
    # 2. Parse Predictions CSV
    df_pred = pd.read_csv(csv_out)
    
    # 3. Parse MEME JSON
    try:
        with open(meme_path) as f:
            meme_data = json.load(f)
        content = meme_data['MLE']['content']['0']
        df_meme = pd.DataFrame(content, columns=['alpha', 'beta1', 'p1', 'beta_pos', 'p_pos', 'lrt', 'p_value', 'branches_sel', 'tot_len', 'meme_logl', 'fel_logl', 'lrt_meme_fel', 'fel_alpha', 'fel_beta'])
        df_meme['codon_site'] = list(range(1, len(df_meme) + 1))
        df_meme['lrt'] = df_meme['lrt'].astype(float)
        df_meme['p_value'] = df_meme['p_value'].astype(float)
    except Exception as ex:
        return {
            'dataset': base_name,
            'status': 'FAIL_JSON',
            'error': str(ex)
        }
        
    # Merge and evaluate 100% of codon sites (never exclude gap sites)
    df = pd.merge(df_pred, df_meme, on='codon_site')
    df_valid = df.copy().sort_values(by='local_z_score', ascending=False).reset_index(drop=True)
    n_sites = len(df_valid)
    df_valid['is_sig_005'] = (df_valid['p_value'] <= 0.05).astype(int)
    df_valid['is_sig_010'] = (df_valid['p_value'] <= 0.10).astype(int)
    
    n_hits_005 = df_valid['is_sig_005'].sum()
    n_hits_010 = df_valid['is_sig_010'].sum()
    base_rate_005 = n_hits_005 / max(1, n_sites)
    
    # Correlations
    pr, _ = pearsonr(df_valid['predicted_lrt'], df_valid['lrt'])
    sr, _ = spearmanr(df_valid['predicted_lrt'], df_valid['lrt'])
    
    # ROC AUC
    # ROC AUC
    auc_005 = roc_auc_score(df_valid['is_sig_005'], df_valid['predicted_lrt']) if n_hits_005 > 0 and n_hits_005 < n_sites else float('nan')
    auc_010 = roc_auc_score(df_valid['is_sig_010'], df_valid['predicted_lrt']) if n_hits_010 > 0 and n_hits_010 < n_sites else float('nan')
    
    # PR AUC (Average Precision)
    pr_auc_005 = average_precision_score(df_valid['is_sig_005'], df_valid['predicted_lrt']) if n_hits_005 > 0 and n_hits_005 < n_sites else float('nan')
    pr_auc_010 = average_precision_score(df_valid['is_sig_010'], df_valid['predicted_lrt']) if n_hits_010 > 0 and n_hits_010 < n_sites else float('nan')
    
    # PPV at Top 25
    k_top = min(25, n_sites)
    top_k = df_valid.iloc[:k_top]
    tp_005 = top_k['is_sig_005'].sum()
    tp_010 = top_k['is_sig_010'].sum()
    ppv_005 = tp_005 / k_top * 100.0
    ppv_010 = tp_010 / k_top * 100.0
    
    top1_row = df_valid.iloc[0]
    top1_site = int(top1_row['codon_site'])
    top1_z = top1_row['local_z_score']
    top1_sig = "YES ✅" if top1_row['is_sig_005'] == 1 else ("p<=0.10" if top1_row['is_sig_010'] == 1 else "No")
    
    return {
        'dataset': base_name,
        'status': 'OK',
        'n_sites': n_sites,
        'n_hits_005': n_hits_005,
        'base_rate': base_rate_005 * 100.0,
        'pearson_r': pr,
        'spearman_rho': sr,
        'roc_auc_005': auc_005,
        'roc_auc_010': auc_010,
        'pr_auc_005': pr_auc_005,
        'pr_auc_010': pr_auc_010,
        'top25_ppv_005': ppv_005,
        'top25_ppv_010': ppv_010,
        'top1_site': top1_site,
        'top1_z': top1_z,
        'top1_sig': top1_sig
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model across all 20 MEME benchmarks")
    parser.add_argument("--model", type=str, default="./axomeme_3.1.pt", help="Path to model checkpoint")
    args = parser.parse_args()
    
    data_dir = '/Users/sergei/Development/hyphy/tests/data/'
    meme_files = sorted(glob.glob(os.path.join(data_dir, '*.MEME.json')))
    
    results = []
    print(f"=== 🔬 EVALUATING MODEL ({args.model}) ACROSS {len(meme_files)} BENCHMARK DATASETS ===")
    
    for m_path in meme_files:
        base_name = os.path.basename(m_path).replace('.MEME.json', '')
        nex_path = os.path.join(data_dir, base_name)
        if not os.path.exists(nex_path):
            continue
            
        print(f"[*] Processing {base_name}...", end="", flush=True)
        res = evaluate_dataset(base_name, nex_path, m_path, model_path=args.model)
        results.append(res)
        if res['status'] == 'OK':
            print(f" Done! Spearman rho = {res['spearman_rho']:+.4f}, ROC AUC(0.05) = {res['roc_auc_005']:.4f}, PR AUC(0.05) = {res['pr_auc_005']:.4f}, Top25 PPV = {res['top25_ppv_005']:.1f}%")
        else:
            print(f" FAILED: {res.get('error', '')}")
            
    df_res = pd.DataFrame(results)
    df_res.to_csv("scratch/all_meme_benchmark_evaluations.csv", index=False)
    
    df_ok = df_res[df_res['status'] == 'OK'].copy()
    print("\n" + "=" * 90)
    print(f"🏆 BENCHMARK EVALUATION SUMMARY TABLE ACROSS ALL DATASETS ({args.model})")
    print("=" * 90)
    cols_show = ['dataset', 'n_sites', 'n_hits_005', 'base_rate', 'spearman_rho', 'roc_auc_005', 'top25_ppv_005', 'top1_site', 'top1_z', 'top1_sig']
    print(df_ok[cols_show].to_string(index=False))
    
    print("\n" + "=" * 90)
    print("📊 OVERALL AGGREGATE SUMMARY Across %d Datasets:" % len(df_ok))
    print("  Mean Spearman Rank Correlation (rho) : %+.4f" % df_ok['spearman_rho'].mean())
    print("  Mean Pearson Linear Correlation (r)  : %+.4f" % df_ok['pearson_r'].mean())
    print("  Mean ROC AUC Score (p <= 0.05 Hits)  : %.4f" % df_ok['roc_auc_005'].mean())
    print("  Mean ROC AUC Score (p <= 0.10 Hits)  : %.4f" % df_ok['roc_auc_010'].mean())
    print("  Mean Top-25 PPV / Precision (p<=0.05): %.1f%%" % df_ok['top25_ppv_005'].mean())
    print("=" * 90)

if __name__ == "__main__":
    main()
