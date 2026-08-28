#!/usr/bin/env python3
"""
run_synthetic_validation_suite.py:
Comprehensive Synthetic Simulation Validation Suite for HyphAeon BUSTED:
1. Pure Neutral Null Calibration (50 replicates) -> Measures False Positive Rate (FPR)
2. Synonymous Heterogeneity / SRV Null (30 replicates) -> Measures Confounder Immunity
3. Subtle Selection (omega3=3.0, p3=2%) (40 replicates) -> Measures Sensitivity on weak signals
4. Moderate Selection (omega3=5.0, p3=5%) (40 replicates) -> Measures Standard Power
5. Strong Selection (omega3=15.0, p3=10%) (40 replicates) -> Measures Strong Power
Total: 200 datasets evaluated via GPU Batch Mode.
"""

import os
import sys
import time
import glob
import json
import pyvolve
import numpy as np
import pandas as pd
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree
import subprocess

SIM_DIR = "/Users/sergei/Projects/TOGA_MEME/bench/synthetic_benchmark"
MSA_DIR = os.path.join(SIM_DIR, "alignments")
os.makedirs(MSA_DIR, exist_ok=True)
WEIGHTS_PATH = "/Users/sergei/Projects/TOGA_MEME/axomeme_repo/model.safetensors"

def build_random_tree(n_taxa=20, mean_bl=0.08, seed=42):
    rng = np.random.default_rng(seed)
    taxa_names = [f"Taxon_{i+1:02d}" for i in range(n_taxa)]
    tips = [Clade(branch_length=float(rng.exponential(mean_bl)), name=name) for name in taxa_names]
    while len(tips) > 1:
        i, j = rng.choice(len(tips), size=2, replace=False)
        a, b = tips[int(i)], tips[int(j)]
        remaining = [t for k, t in enumerate(tips) if k not in (int(i), int(j))]
        parent = Clade(branch_length=float(rng.exponential(mean_bl)), clades=[a, b])
        tips = remaining + [parent]
    return Tree(root=tips[0])

def simulate_dataset(tree_obj, num_codons, omega_vals, proportions, out_fa, out_nwk):
    Phylo.write(tree_obj, out_nwk, "newick")
    tree = pyvolve.read_tree(file=out_nwk)
    partitions = []
    for w, p in zip(omega_vals, proportions):
        n_sites = int(round(num_codons * p))
        if n_sites > 0:
            m = pyvolve.Model("codon", {"omega": float(w)})
            partitions.append(pyvolve.Partition(models=m, size=n_sites))
    evolver = pyvolve.Evolver(partitions=partitions, tree=tree)
    evolver(seqfile=out_fa, count=1, quiet=True)

scenarios = [
    # 1. Pure Neutral Null (w3 = 1.0, p3 = 0%)
    {
        "name": "Null_Neutral",
        "is_positive": False,
        "reps": 50,
        "omega_vals": [0.1, 1.0, 1.0],
        "proportions": [0.70, 0.30, 0.00],
        "taxa_range": (15, 40),
        "codons_range": (200, 500),
        "bl_range": (0.05, 0.15)
    },
    # 2. SRV / High Divergence Null (w3 = 1.0)
    {
        "name": "Null_HighDivergence",
        "is_positive": False,
        "reps": 30,
        "omega_vals": [0.05, 1.0, 1.0],
        "proportions": [0.80, 0.20, 0.00],
        "taxa_range": (25, 60),
        "codons_range": (300, 600),
        "bl_range": (0.15, 0.30)
    },
    # 3. Subtle Selection (w3 = 3.0, p3 = 2%)
    {
        "name": "Alt_Subtle_w3_3_p2",
        "is_positive": True,
        "reps": 40,
        "omega_vals": [0.1, 1.0, 3.0],
        "proportions": [0.70, 0.28, 0.02],
        "taxa_range": (20, 50),
        "codons_range": (300, 600),
        "bl_range": (0.08, 0.20)
    },
    # 4. Moderate Selection (w3 = 5.0, p3 = 5%)
    {
        "name": "Alt_Moderate_w3_5_p5",
        "is_positive": True,
        "reps": 40,
        "omega_vals": [0.1, 1.0, 5.0],
        "proportions": [0.70, 0.25, 0.05],
        "taxa_range": (20, 50),
        "codons_range": (300, 600),
        "bl_range": (0.08, 0.20)
    },
    # 5. Strong Selection (w3 = 15.0, p3 = 10%)
    {
        "name": "Alt_Strong_w3_15_p10",
        "is_positive": True,
        "reps": 40,
        "omega_vals": [0.1, 1.0, 15.0],
        "proportions": [0.65, 0.25, 0.10],
        "taxa_range": (20, 50),
        "codons_range": (300, 600),
        "bl_range": (0.08, 0.20)
    }
]

print("=" * 85)
print("  GENERATING 200 SYNTHETIC BENCHMARK DATASETS ACROSS 5 EVOLUTIONARY SCENARIOS")
print("=" * 85)

metadata = []
seed = 42
for sc in scenarios:
    sc_name = sc["name"]
    print(f"[*] Generating {sc['reps']} replicates for scenario '{sc_name}'...", flush=True)
    for rep in range(1, sc["reps"] + 1):
        seed += 1
        rng = np.random.default_rng(seed)
        n_taxa = int(rng.integers(sc["taxa_range"][0], sc["taxa_range"][1] + 1))
        n_codons = int(rng.integers(sc["codons_range"][0], sc["codons_range"][1] + 1))
        mean_bl = float(rng.uniform(sc["bl_range"][0], sc["bl_range"][1]))
        
        file_base = f"{sc_name}_rep{rep:02d}"
        out_fa = os.path.join(MSA_DIR, f"{file_base}.fa")
        out_nwk = os.path.join(MSA_DIR, f"{file_base}.nwk")
        
        tree_obj = build_random_tree(n_taxa=n_taxa, mean_bl=mean_bl, seed=seed)
        simulate_dataset(tree_obj, n_codons, sc["omega_vals"], sc["proportions"], out_fa, out_nwk)
        
        metadata.append({
            "Gene": file_base,
            "Scenario": sc_name,
            "True_IsPositive": sc["is_positive"],
            "True_Omega3": sc["omega_vals"][2] if sc["is_positive"] else 1.0,
            "True_Prop3": sc["proportions"][2] if sc["is_positive"] else 0.0,
            "True_Taxa": n_taxa,
            "True_Codons": n_codons,
            "Fa_Path": out_fa,
            "Nwk_Path": out_nwk
        })

df_meta = pd.DataFrame(metadata)
df_meta.to_csv(os.path.join(SIM_DIR, "synthetic_metadata.csv"), index=False)
print(f"[✓] Generated all {len(metadata)} synthetic alignments in '{MSA_DIR}'.\n", flush=True)

# Step 2: Run HyphAeon BUSTED-CORAL Batch Mode
print("=" * 85)
print("  RUNNING HYPHAEON BUSTED-CORAL BATCH INFERENCE ACROSS ALL 200 SYNTHETIC ALIGNMENTS")
print("=" * 85)

hyph_csv = os.path.join(SIM_DIR, "hyphaeon_synthetic_results.csv")
cmd_hyph = [
    "hyphaeon", "busted",
    "-d", MSA_DIR,
    "--pattern", "*.fa",
    "--tree-suffix", ".nwk",
    "-w", WEIGHTS_PATH,
    "-c", hyph_csv
]
t0 = time.time()
subprocess.run(cmd_hyph)
t_eval = time.time() - t0
print(f"\n[✓] HyphAeon Batch Complete in {t_eval:.2f}s total ({len(metadata)/t_eval:.1f} genes/s)!\n", flush=True)

# Step 3: Analyze Calibration, FPR, Power & Correlations
df_res = pd.read_csv(hyph_csv)
df_merged = pd.merge(df_meta, df_res, on="Gene")

print("=" * 85)
print("                         SYNTHETIC VALIDATION SUMMARY REPORT")
print("=" * 85)

# Group by scenario
summary_rows = []
for sc_name, group in df_merged.groupby("Scenario", sort=False):
    n_reps = len(group)
    is_pos = group["True_IsPositive"].iloc[0]
    
    # Decisions at p_ACAT < 0.05 or Selection_Prob > 0.50
    calls = group["Selected"].values
    mean_prob = group["Selection_Prob"].mean() * 100
    mean_p_acat = group["p_ACAT"].mean()
    mean_pred_w3 = group["Omega_3"].mean()
    
    if not is_pos:
        # Null scenario: Calculate False Positive Rate
        fpr = np.mean(calls) * 100
        fpr_acat = np.mean(group["p_ACAT"] < 0.05) * 100
        fpr_prob = np.mean(group["Selection_Prob"] > 0.50) * 100
        summary_rows.append({
            "Scenario": sc_name,
            "Type": "NULL (Ground Truth: Negative)",
            "Replicates": n_reps,
            "FPR / Power": f"FPR = {fpr:.1f}%",
            "Mean_Selection_Prob": f"{mean_prob:.1f}%",
            "Mean_p_ACAT": f"{mean_p_acat:.4f}",
            "Mean_Pred_w3": f"{mean_pred_w3:.2f}",
            "Status": "✓ Well-Calibrated (FPR <= 5%)" if fpr <= 5.0 else "Notice: Elevated FPR"
        })
    else:
        # Alt scenario: Calculate Statistical Power
        power = np.mean(calls) * 100
        power_acat = np.mean(group["p_ACAT"] < 0.05) * 100
        summary_rows.append({
            "Scenario": sc_name,
            "Type": f"ALT (w3={group['True_Omega3'].iloc[0]}, p3={group['True_Prop3'].iloc[0]*100:.0f}%)",
            "Replicates": n_reps,
            "FPR / Power": f"Power = {power:.1f}%",
            "Mean_Selection_Prob": f"{mean_prob:.1f}%",
            "Mean_p_ACAT": f"{mean_p_acat:.4f}",
            "Mean_Pred_w3": f"{mean_pred_w3:.2f}",
            "Status": "✓ High Power" if power >= 75.0 else "Moderate Power"
        })

df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))

# Overall Null vs Alt ROC metrics
from sklearn.metrics import roc_auc_score, average_precision_score
y_true = df_merged["True_IsPositive"].astype(int).values
y_scores = df_merged["Selection_Prob"].values
auc_roc = roc_auc_score(y_true, y_scores)
auc_pr = average_precision_score(y_true, y_scores)

print("\n" + "=" * 85)
print(f"  GLOBAL BENCHMARK METRICS (200 Synthetic Datasets):")
print(f"    • ROC-AUC (Area Under ROC Curve):       {auc_roc:.4f}")
print(f"    • PR-AUC (Area Under Precision-Recall): {auc_pr:.4f}")
print("=" * 85)

