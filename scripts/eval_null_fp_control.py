#!/usr/bin/env python3
"""
eval_null_fp_control.py
-----------------------
Evaluates AxoMEME checkpoint on pure neutral null (MG94) simulation alignments
to confirm empirical False Positive Rate (FPR) calibration and nominal alpha control.
"""

import os
import sys
import glob
import random
import subprocess
import numpy as np
import pandas as pd

def evaluate_null_directory(model_path="axomeme_5_dim384.pt", null_dir="null/aln", num_samples=25, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    print("=" * 115)
    print(f"🔬 EVALUATING FALSE POSITIVE RATE (FPR) ON PURE NEUTRAL NULL SIMULATION ALIGNMENTS (MG94)")
    print(f"[*] Model Checkpoint: {model_path}")
    print(f"[*] Null Directory:   {null_dir}")
    print("=" * 115)
    
    null_files = sorted(glob.glob(os.path.join(null_dir, "*.nex")))
    if not null_files:
        print(f"[!] No NEXUS files found in '{null_dir}'")
        return
        
    sampled_files = random.sample(null_files, min(num_samples, len(null_files)))
    print(f"[*] Sampled {len(sampled_files)} pure neutral null alignments.\n")
    
    os.makedirs("scratch/null_eval", exist_ok=True)
    
    results = []
    total_codons_all = 0
    total_fp_05_all = 0
    total_fp_10_all = 0
    
    for idx, nex_path in enumerate(sampled_files, 1):
        gene_name = os.path.basename(nex_path).replace('.replicate.1.nex', '').replace('null_', '')
        out_csv = f"scratch/null_eval/{gene_name}_null_preds.csv"
        
        cmd = [
            sys.executable, "scripts/predict_regression_nexus.py",
            "--alignment", nex_path,
            "--model", model_path,
            "--output", out_csv,
            "--max_species", "256"
        ]
        
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not os.path.exists(out_csv):
            print(f"  [{idx:02d}/{len(sampled_files):02d}] {gene_name:<22} | Error running inference: {res.stderr.strip()[:80]}")
            continue
            
        df = pd.read_csv(out_csv)
        total_codons = len(df)
        pred_lrts = df['predicted_lrt'].values
        
        fp_05 = int((pred_lrts >= 5.14).sum())
        fp_10 = int((pred_lrts >= 3.81).sum())
        
        fpr_05 = (fp_05 / total_codons) * 100.0
        fpr_10 = (fp_10 / total_codons) * 100.0
        mean_lrt = float(np.mean(pred_lrts))
        max_lrt = float(np.max(pred_lrts))
        
        total_codons_all += total_codons
        total_fp_05_all += fp_05
        total_fp_10_all += fp_10
        
        results.append({
            'Gene': gene_name,
            'Codons': total_codons,
            'Mean Pred LRT': mean_lrt,
            'Max Pred LRT': max_lrt,
            'FP (p<=0.05)': fp_05,
            'FPR (0.05)': fpr_05,
            'FP (p<=0.10)': fp_10,
            'FPR (0.10)': fpr_10
        })
        
        print(f"  [{idx:02d}/{len(sampled_files):02d}] {gene_name:<22} | Codons: {total_codons:4d} | Mean LRT: {mean_lrt:6.4f} | Max: {max_lrt:5.2f} | FP(0.05): {fp_05:2d} ({fpr_05:4.1f}%) | FP(0.10): {fp_10:2d} ({fpr_10:4.1f}%)")

    df_res = pd.DataFrame(results)
    
    print("\n" + "=" * 115)
    print(f"{'Alignment / Gene':<22} | {'Codons':<6} | {'Mean Pred LRT':<13} | {'Max LRT':<8} | {'FP (p<=0.05)':<11} | {'FPR (0.05)':<10} | {'FP (p<=0.10)':<11} | {'FPR (0.10)'}")
    print("-" * 115)
    for _, r in df_res.iterrows():
        print(f"{r['Gene']:<22} | {int(r['Codons']):<6d} | {r['Mean Pred LRT']:13.4f} | {r['Max Pred LRT']:8.2f} | {int(r['FP (p<=0.05)']):<11d} | {r['FPR (0.05)']:9.2f}% | {int(r['FP (p<=0.10)']):<11d} | {r['FPR (0.10)']:9.2f}%")
        
    overall_fpr_05 = (total_fp_05_all / total_codons_all) * 100.0
    overall_fpr_10 = (total_fp_10_all / total_codons_all) * 100.0
    
    print("=" * 115)
    print(f"📊 OVERALL AGGREGATE NULL SIMULATION METRICS ({len(df_res)} Alignments, {total_codons_all:,} Codon Sites):")
    print(f"   • Overall Empirical FPR at alpha = 0.05 (LRT >= 5.14) : {overall_fpr_05:.2f}%  (Nominal Target: <= 5.00%)")
    print(f"   • Overall Empirical FPR at alpha = 0.10 (LRT >= 3.81) : {overall_fpr_10:.2f}%  (Nominal Target: <= 10.00%)")
    print(f"   • Mean Predicted LRT on Neutral Data                  : {df_res['Mean Pred LRT'].mean():.4f}  (Ideal: ~0.0)")
    print(f"   • Total False Positive Rejections at p <= 0.05        : {total_fp_05_all} / {total_codons_all:,}")
    print(f"   • Total False Positive Rejections at p <= 0.10        : {total_fp_10_all} / {total_codons_all:,}")
    print("=" * 115)

if __name__ == '__main__':
    evaluate_null_directory()
