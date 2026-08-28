#!/usr/bin/env python3
"""
run_hyphy_test_suite_parallel.py:
Parallel benchmark of HyphAeon BUSTED vs HyPhy BUSTED+S on all test datasets.
Reports each dataset immediately as it finishes.
"""

import os
import sys
import glob
import time
import json
import subprocess
import concurrent.futures
import pandas as pd
import numpy as np

DATA_DIR = "/Users/sergei/Development/hyphy-2.5.27/tests/data"
OUTPUT_DIR = "/Users/sergei/Projects/TOGA_MEME/bench/hyphy_test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
WEIGHTS_PATH = "/Users/sergei/Projects/TOGA_MEME/axomeme_repo/model.safetensors"

datasets = sorted(glob.glob(os.path.join(DATA_DIR, "*.nex")))

print("=" * 80, flush=True)
print(f"  RUNNING PARALLEL BENCHMARK ON {len(datasets)} HYPHY TEST DATASETS", flush=True)
print("=" * 80, flush=True)

def process_dataset(nex_path):
    gene_name = os.path.splitext(os.path.basename(nex_path))[0]
    
    # 1. Run HyphAeon BUSTED
    hyph_json = os.path.join(OUTPUT_DIR, f"{gene_name}_hyphaeon.json")
    t0_hyph = time.time()
    cmd_hyph = [
        "hyphaeon", "busted",
        "-a", nex_path,
        "-w", WEIGHTS_PATH,
        "-o", hyph_json,
        "--cpu"  # thread-safe CPU execution
    ]
    res_hyph = subprocess.run(cmd_hyph, capture_output=True, text=True)
    t_hyph = time.time() - t0_hyph
    
    if res_hyph.returncode != 0:
        return {"error": f"HyphAeon failed: {res_hyph.stderr[:200]}", "gene": gene_name}
        
    with open(hyph_json) as f:
        data_hyph = json.load(f)
        
    # 2. Run HyPhy BUSTED+S
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
        return {"error": f"HyPhy failed: {res_hyphy.stderr[:200]}", "gene": gene_name}
        
    with open(hyphy_json) as f:
        data_hyphy = json.load(f)

    # Extract HyPhy
    fits = data_hyphy.get("fits", {})
    busted_fit = fits.get("Unconstrained model", {})
    lrt = float(data_hyphy.get("test results", {}).get("LRT", 0.0))
    p_val_hyphy = float(data_hyphy.get("test results", {}).get("p-value", 1.0))
    
    rate_dist = busted_fit.get("Rate Distributions", {}).get("Test", {})
    w3_hyphy = 1.0
    p3_hyphy = 0.0
    if rate_dist:
        w3_hyphy = float(rate_dist.get("2", {}).get("omega", 1.0))
        p3_hyphy = float(rate_dist.get("2", {}).get("proportion", 0.0))
        
    srv_dist = fits.get("Unconstrained model", {}).get("Rate Distributions", {}).get("Synonymous site-to-site rates", {})
    srv_var_hyphy = 0.0
    if srv_dist:
        alphas = [float(srv_dist.get(str(k), {}).get("rate", 1.0)) for k in range(len(srv_dist))]
        props = [float(srv_dist.get(str(k), {}).get("proportion", 1.0/len(srv_dist))) for k in range(len(srv_dist))]
        mean_a = sum(a * p for a, p in zip(alphas, props))
        srv_var_hyphy = sum(p * ((a - mean_a) ** 2) for a, p in zip(alphas, props))

    # Extract HyphAeon
    taxa = data_hyph.get("taxa", 0)
    sites = data_hyph.get("sites", 0)
    p_acat = float(data_hyph.get("p_value_acat", 1.0))
    neural_prob = float(data_hyph.get("selection_probability", 0.0))
    neural_lrt = float(data_hyph.get("predicted_gene_lrt", 0.0))
    w3_hyph = float(data_hyph.get("rate_distributions", {}).get("omega_3", 1.0))
    p3_hyph = float(data_hyph.get("rate_distributions", {}).get("proportion_3", 0.0))
    srv_var_hyph = float(data_hyph.get("synonymous_rate_variation", 0.0))

    selected_hyphy = bool(p_val_hyphy < 0.05)
    selected_hyph = bool(p_acat < 0.05 or neural_prob > 0.50)
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

    # Print Live Report Immediately
    print("-" * 80, flush=True)
    print(f"  [FINISHED] DATASET: {gene_name} (Taxa: {taxa}, Codons: {sites})", flush=True)
    print(f"  HyPhy BUSTED+S:   p = {p_val_hyphy:.4e} | LRT = {lrt:6.2f} | w3 = {w3_hyphy:6.2f} (p3={p3_hyphy*100:4.1f}%) | Var(a) = {srv_var_hyphy:.4f} | Time: {t_hyphy:6.2f}s", flush=True)
    print(f"  HyphAeon BUSTED:  p = {p_acat:.4e} | Prob = {neural_prob*100:4.1f}% | w3 = {w3_hyph:6.2f} (p3={p3_hyph*100:4.1f}%) | Var(a) = {srv_var_hyph:.4f} | Time: {t_hyph:6.3f}s", flush=True)
    decision_str = "POSITIVE SELECTION" if selected_hyph else "NEUTRAL"
    match_str = "✓ CONCORDANT" if concordant else "✗ DISCORDANT"
    print(f"  Decision Match:   {match_str} ({decision_str}) | Speedup: {t_hyphy/t_hyph:.1f}x", flush=True)
    print("-" * 80, flush=True)

    return row

# Run 4 parallel workers
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    future_to_gene = {executor.submit(process_dataset, nex): nex for nex in datasets}
    for future in concurrent.futures.as_completed(future_to_gene):
        res = future.result()
        if "error" not in res:
            results.append(res)

# Save Final Cumulative Results
df = pd.DataFrame(results)
csv_out = os.path.join(OUTPUT_DIR, "hyphy_test_suite_comparison.csv")
df.to_csv(csv_out, index=False)
print("\n" + "=" * 80, flush=True)
print(f"  BENCHMARK COMPLETE! Processed {len(results)} datasets. Results saved to: {csv_out}", flush=True)
print("=" * 80, flush=True)

