#!/usr/bin/env python3
"""
benchmark_hyphy_test_suite.py:
Runs HyphAeon BUSTED and HyPhy BUSTED+S on all test datasets in
/Users/sergei/Development/hyphy-2.5.27/tests/data/ and outputs live comparisons.
"""

import os
import sys
import glob
import time
import json
import subprocess
import pandas as pd
import numpy as np

DATA_DIR = "/Users/sergei/Development/hyphy-2.5.27/tests/data"
OUTPUT_DIR = "/Users/sergei/Projects/TOGA_MEME/bench/hyphy_test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
WEIGHTS_PATH = "/Users/sergei/Projects/TOGA_MEME/axomeme_repo/model.safetensors"

datasets = sorted(glob.glob(os.path.join(DATA_DIR, "*.nex")))

print("=" * 80)
print(f"  RUNNING HYPHAEON BUSTED vs HYPHY BUSTED+S BENCHMARK ON {len(datasets)} DATASETS")
print("=" * 80)

results = []

for i, nex_path in enumerate(datasets, 1):
    gene_name = os.path.splitext(os.path.basename(nex_path))[0]
    print(f"\n[{i}/{len(datasets)}] Processing: {gene_name} ({os.path.basename(nex_path)})")
    
    # -------------------------------------------------------------
    # 1. Run HyphAeon BUSTED
    # -------------------------------------------------------------
    hyph_json = os.path.join(OUTPUT_DIR, f"{gene_name}_hyphaeon.json")
    t0_hyph = time.time()
    cmd_hyph = [
        "hyphaeon", "busted",
        "-a", nex_path,
        "-w", WEIGHTS_PATH,
        "-o", hyph_json
    ]
    res_hyph = subprocess.run(cmd_hyph, capture_output=True, text=True)
    t_hyph = time.time() - t0_hyph
    
    if res_hyph.returncode != 0:
        print(f"  [!] HyphAeon error: {res_hyph.stderr[:300]}")
        continue
        
    with open(hyph_json) as f:
        data_hyph = json.load(f)
        
    # -------------------------------------------------------------
    # 2. Run HyPhy BUSTED+S (Ground Truth)
    # -------------------------------------------------------------
    hyphy_json = os.path.join(OUTPUT_DIR, f"{gene_name}_hyphy_busted_srv.json")
    t0_hyphy = time.time()
    cmd_hyphy = [
        "hyphy", "busted",
        "--alignment", nex_path,
        "--srv", "Yes",
        "--output", hyphy_json
    ]
    res_hyphy = subprocess.run(cmd_hyphy, capture_output=True, text=True)
    t_hyphy = time.time() - t0_hyphy
    
    if res_hyphy.returncode != 0:
        print(f"  [!] HyPhy error: {res_hyphy.stderr[:300]}")
        continue
        
    with open(hyphy_json) as f:
        data_hyphy = json.load(f)

    # Extract HyPhy statistics
    fits = data_hyphy.get("fits", {})
    busted_fit = fits.get("Unconstrained model", {})
    lrt = data_hyphy.get("test results", {}).get("LRT", 0.0)
    p_val_hyphy = data_hyphy.get("test results", {}).get("p-value", 1.0)
    
    # Extract Omega 3 and Proportion from HyPhy
    rate_dist = busted_fit.get("Rate Distributions", {}).get("Test", {})
    w3_hyphy = 1.0
    p3_hyphy = 0.0
    if rate_dist:
        w3_hyphy = rate_dist.get("2", {}).get("omega", 1.0)
        p3_hyphy = rate_dist.get("2", {}).get("proportion", 0.0)
        
    # Extract Synonymous Rate Variation
    srv_dist = fits.get("Unconstrained model", {}).get("Rate Distributions", {}).get("Synonymous site-to-site rates", {})
    srv_var_hyphy = 0.0
    if srv_dist:
        # Compute Var(alpha) from discrete classes
        alphas = [srv_dist.get(str(k), {}).get("rate", 1.0) for k in range(len(srv_dist))]
        props = [srv_dist.get(str(k), {}).get("proportion", 1.0/len(srv_dist)) for k in range(len(srv_dist))]
        mean_a = sum(a * p for a, p in zip(alphas, props))
        srv_var_hyphy = sum(p * ((a - mean_a) ** 2) for a, p in zip(alphas, props))

    # Extract HyphAeon statistics
    taxa = data_hyph.get("taxa", 0)
    sites = data_hyph.get("sites", 0)
    p_acat = data_hyph.get("p_value_acat", 1.0)
    neural_prob = data_hyph.get("selection_probability", 0.0)
    neural_lrt = data_hyph.get("predicted_gene_lrt", 0.0)
    w3_hyph = data_hyph.get("rate_distributions", {}).get("omega_3", 1.0)
    p3_hyph = data_hyph.get("rate_distributions", {}).get("proportion_3", 0.0)
    srv_var_hyph = data_hyph.get("synonymous_rate_variation", 0.0)

    selected_hyphy = p_val_hyphy < 0.05
    selected_hyph = p_acat < 0.05 or neural_prob > 0.50
    concordant = (selected_hyphy == selected_hyph)

    row = {
        "Gene": gene_name,
        "Taxa": taxa,
        "Codons": sites,
        "HyPhy_p": p_val_hyphy,
        "HyPhy_LRT": lrt,
        "HyPhy_w3": w3_hyphy,
        "HyPhy_p3": p3_hyphy,
        "HyPhy_VarAlpha": srv_var_hyphy,
        "HyPhy_Time_s": t_hyphy,
        "HyphAeon_pACAT": p_acat,
        "HyphAeon_NeuralProb": neural_prob,
        "HyphAeon_NeuralLRT": neural_lrt,
        "HyphAeon_w3": w3_hyph,
        "HyphAeon_p3": p3_hyph,
        "HyphAeon_VarAlpha": srv_var_hyph,
        "HyphAeon_Time_s": t_hyph,
        "Speedup": t_hyphy / max(0.001, t_hyph),
        "Selected_HyPhy": selected_hyphy,
        "Selected_HyphAeon": selected_hyph,
        "Concordant": concordant
    }
    results.append(row)

    # Print Live Dataset Summary Card
    print("-" * 80)
    print(f"  DATASET: {gene_name} (Taxa: {taxa}, Codons: {sites})")
    print(f"  HyPhy BUSTED+S:   p = {p_val_hyphy:.4e} | LRT = {lrt:6.2f} | w3 = {w3_hyphy:6.2f} (p3={p3_hyphy*100:4.1f}%) | Var(a) = {srv_var_hyphy:.4f} | Time: {t_hyphy:6.2f}s")
    print(f"  HyphAeon BUSTED:  p = {p_acat:.4e} | Prob = {neural_prob*100:4.1f}% | w3 = {w3_hyph:6.2f} (p3={p3_hyph*100:4.1f}%) | Var(a) = {srv_var_hyph:.4f} | Time: {t_hyph:6.3f}s")
    print(f"  Decision Match:   {'✓ CONCORDANT' if concordant else '✗ DISCORDANT'} ({'POSITIVE SELECTION' if selected_hyph else 'NEUTRAL'}) | Speedup: {t_hyphy/t_hyph:.1f}x")
    print("-" * 80)

# Final Table
df = pd.DataFrame(results)
csv_out = os.path.join(OUTPUT_DIR, "hyphy_test_suite_comparison.csv")
df.to_csv(csv_out, index=False)
print(f"\n[✓] Saved complete benchmark table to: {csv_out}")

