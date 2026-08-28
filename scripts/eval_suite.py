import os
import sys
import subprocess
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_dataset(alignment_path, meme_path, model_path, output_csv, name, lrt_idx=5, pval_idx=6):
    print(f"\n[{name}] Running fresh inference from {model_path}...")
    if os.path.exists(output_csv):
        os.remove(output_csv)
        
    cmd = [
        "python3", "scripts/predict_regression_nexus.py",
        "--alignment", alignment_path,
        "--model", model_path,
        "--output", output_csv
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] Error running inference on {name}:\n{res.stderr}")
        return None
        
    df = pd.read_csv(output_csv)
    with open(meme_path) as f:
        meme_data = json.load(f)
    records = meme_data['MLE']['content']['0']
    
    df['meme_lrt'] = [records[i][lrt_idx] for i in range(len(df))]
    df['meme_pval'] = [records[i][pval_idx] for i in range(len(df))]
    df['pred_log_lrt'] = np.log1p(df['predicted_lrt'])
    df['meme_log_lrt'] = np.log1p(np.maximum(df['meme_lrt'], 0.0))
    df['true_05'] = (df['meme_pval'] <= 0.05).astype(int)
    df['true_10'] = (df['meme_pval'] <= 0.10).astype(int)
    
    # Statistical Metrics
    r, p_r = pearsonr(df['pred_log_lrt'], df['meme_log_lrt'])
    rho, p_rho = spearmanr(df['predicted_lrt'], -np.log10(np.maximum(df['meme_pval'], 1e-12)))
    roc_05 = roc_auc_score(df['true_05'], df['predicted_lrt'])
    roc_10 = roc_auc_score(df['true_10'], df['predicted_lrt'])
    pr_05 = average_precision_score(df['true_05'], df['predicted_lrt'])
    base_05 = df['true_05'].mean()
    pr_10 = average_precision_score(df['true_10'], df['predicted_lrt'])
    base_10 = df['true_10'].mean()
    
    # PPV @ K
    df_sorted = df.sort_values('predicted_lrt', ascending=False).reset_index(drop=True)
    df_sorted['rank'] = range(1, len(df_sorted) + 1)
    
    ppv_5 = df_sorted.iloc[:5]['true_10'].mean() * 100
    ppv_10 = df_sorted.iloc[:10]['true_10'].mean() * 100
    ppv_15 = df_sorted.iloc[:15]['true_10'].mean() * 100
    ppv_20 = df_sorted.iloc[:20]['true_10'].mean() * 100
    
    print("=" * 90)
    print(f"📊 {name} BENCHMARK (Codons: {len(df)})")
    print("=" * 90)
    print(f"  • Spearman rho (-log10 p) : {rho:.4f}  (p = {p_rho:.2e})")
    print(f"  • Pearson r (log(1+LRT))  : {r:.4f}  (p = {p_r:.2e})")
    print(f"  • ROC-AUC (p <= 0.05)     : {roc_05:.4f}")
    print(f"  • ROC-AUC (p <= 0.10)     : {roc_10:.4f}")
    print(f"  • PR-AUC (p <= 0.05)      : {pr_05:.4f}  (Base: {base_05:.4f} -> {pr_05/base_05:.2f}x Lift)")
    print(f"  • PR-AUC (p <= 0.10)      : {pr_10:.4f}  (Base: {base_10:.4f} -> {pr_10/base_10:.2f}x Lift)")
    print(f"  • PPV @ Top 5  (p <= 0.10): {ppv_5:.1f}% ({int(df_sorted.iloc[:5]['true_10'].sum())}/5 True Positives)")
    print(f"  • PPV @ Top 10 (p <= 0.10): {ppv_10:.1f}% ({int(df_sorted.iloc[:10]['true_10'].sum())}/10 True Positives)")
    print(f"  • PPV @ Top 15 (p <= 0.10): {ppv_15:.1f}% ({int(df_sorted.iloc[:15]['true_10'].sum())}/15 True Positives)")
    print(f"  • PPV @ Top 20 (p <= 0.10): {ppv_20:.1f}% ({int(df_sorted.iloc[:20]['true_10'].sum())}/20 True Positives)")
    
    print("\n🏆 Top 15 Ranked Sites:")
    for i in range(15):
        row = df_sorted.iloc[i]
        status = '✅ Pos (p<=0.01)' if row['meme_pval'] <= 0.01 else ('✅ Pos (p<=0.05)' if row['meme_pval'] <= 0.05 else ('⚠️ Pos (p<=0.10)' if row['meme_pval'] <= 0.10 else '❌ Neg'))
        print(f"  Rank {i+1:2d} | Site {int(row['codon_site']):3d} ({row['ref_aa']}) | Pred: {row['predicted_lrt']:.4f} | MEME LRT: {row['meme_lrt']:5.2f} (p={row['meme_pval']:.6f}) | {status}")
        
    return {
        'name': name,
        'rho': rho,
        'r': r,
        'roc_05': roc_05,
        'pr_lift_05': pr_05/base_05,
        'ppv_10': ppv_10
    }

if __name__ == '__main__':
    model_ckpt = sys.argv[1] if len(sys.argv) > 1 else './axomeme_3.6_root.pt'
    t0 = time.time()
    res_rt = evaluate_dataset('/Users/sergei/Development/hyphy/tests/data/HIV_RT.nex', 'HIV_RT_meme_results.json', model_ckpt, 'eval_fresh_rt.csv', 'HIV-1 RT (476 Taxa, 335 Codons)')
    res_ia = evaluate_dataset('/Users/sergei/Development/hyphy/tests/data/InfluenzaA.nex', '/Users/sergei/Development/hyphy/tests/data/InfluenzaA.nex.MEME.json', model_ckpt, 'eval_fresh_ia.csv', 'Influenza A HA (349 Taxa, 329 Codons)')
    res_bg = evaluate_dataset('/Users/sergei/Development/hyphy/tests/data/bglobin.nex', 'bglobin.nex.MEME.json', model_ckpt, 'eval_fresh_bg.csv', 'Beta-Globin (17 Taxa, 144 Codons)')
    
    print("\n" + "=" * 90)
    print(f"🏁 COMPREHENSIVE BENCHMARK EVALUATION FINISHED in {time.time() - t0:.1f}s")
    print("=" * 90)
