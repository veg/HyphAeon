#!/usr/bin/env python3
"""
run_hyphy_busted_synthetic_benchmark.py:
Runs HyPhy BUSTED+S on all 200 synthetic simulation replicates across 6 parallel CPU workers.
Streams real-time progress to stdout and computes comparative FPR/Power vs HyphAeon BUSTED-CORAL.
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

SIM_DIR = "/Users/sergei/Projects/TOGA_MEME/bench/synthetic_benchmark"
MSA_DIR = os.path.join(SIM_DIR, "alignments")
OUTPUT_DIR = os.path.join(SIM_DIR, "hyphy_busted_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
META_PATH = os.path.join(SIM_DIR, "synthetic_metadata.csv")

df_meta = pd.read_csv(META_PATH)
total_datasets = len(df_meta)

print("=" * 88, flush=True)
print(f"  RUNNING HYPHY BUSTED+S SEQUENTIAL REJECTION BENCHMARK ACROSS {total_datasets} SIMULATIONS", flush=True)
print("=" * 88, flush=True)

def process_replicate(row_tuple):
    idx, row = row_tuple
    gene = row["Gene"]
    scenario = row["Scenario"]
    is_positive = bool(row["True_IsPositive"])
    aln_path = row["Fa_Path"]
    tree_path = row["Nwk_Path"]
    
    out_json = os.path.join(OUTPUT_DIR, f"{gene}_hyphy_busted.json")
    
    t0 = time.time()
    cmd = [
        "hyphy", "busted",
        "--alignment", aln_path,
        "--tree", tree_path,
        "--srv", "Yes",
        "--output", out_json
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    
    if res.returncode != 0:
        return {"error": f"HyPhy failed on {gene}: {res.stderr[:200]}", "gene": gene, "idx": idx}
        
    with open(out_json) as f:
        data = json.load(f)
        
    fits = data.get("fits", {})
    lrt = float(data.get("test results", {}).get("LRT", 0.0))
    p_val = float(data.get("test results", {}).get("p-value", 1.0))
    
    rate_dist = fits.get("Unconstrained model", {}).get("Rate Distributions", {}).get("Test", {})
    w3 = 1.0
    p3 = 0.0
    if rate_dist:
        w3 = float(rate_dist.get("2", {}).get("omega", 1.0))
        p3 = float(rate_dist.get("2", {}).get("proportion", 0.0))
        
    srv_dist = fits.get("Unconstrained model", {}).get("Rate Distributions", {}).get("Synonymous site-to-site rates", {})
    srv_var = 0.0
    if srv_dist:
        alphas = [float(srv_dist.get(str(k), {}).get("rate", 1.0)) for k in range(len(srv_dist))]
        props = [float(srv_dist.get(str(k), {}).get("proportion", 1.0/len(srv_dist))) for k in range(len(srv_dist))]
        mean_a = sum(a * p for a, p in zip(alphas, props))
        srv_var = sum(p * ((a - mean_a) ** 2) for a, p in zip(alphas, props))

    selected = bool(p_val < 0.05)
    
    if is_positive:
        call_type = "✓ True Positive" if selected else "✗ False Negative (Missed)"
    else:
        call_type = "✗ False Positive" if selected else "✓ True Negative"

    result_dict = {
        "Gene": gene,
        "Scenario": scenario,
        "True_IsPositive": is_positive,
        "True_Omega3": row["True_Omega3"],
        "True_Prop3": row["True_Prop3"],
        "Taxa": row["True_Taxa"],
        "Codons": row["True_Codons"],
        "HyPhy_p": p_val,
        "HyPhy_LRT": lrt,
        "HyPhy_w3": w3,
        "HyPhy_p3": p3,
        "HyPhy_VarAlpha": srv_var,
        "HyPhy_Selected": selected,
        "HyPhy_Time_s": elapsed,
        "Call_Type": call_type
    }
    
    # Live Real-Time Reporting
    print(f"[{idx+1:03d}/{total_datasets}] {gene:<24s} | {scenario:<20s} | p = {p_val:.4e} | LRT = {lrt:6.2f} | w3 = {w3:5.1f} ({p3*100:4.1f}%) | Time: {elapsed:5.1f}s | {call_type}", flush=True)
    return result_dict

# Run 6 parallel workers
results = []
items = list(df_meta.iterrows())
t_global_start = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(process_replicate, (idx, row)) for idx, row in items]
    for future in concurrent.futures.as_completed(futures):
        res = future.result()
        if "error" not in res:
            results.append(res)
        else:
            print(f"[!] Error: {res['error']}", flush=True)

t_total = time.time() - t_global_start
df_out = pd.DataFrame(results)
csv_path = os.path.join(SIM_DIR, "hyphy_busted_results.csv")
df_out.to_csv(csv_path, index=False)

print("\n" + "=" * 88, flush=True)
print(f"  HYPHY BUSTED+S BENCHMARK COMPLETE! Processed {len(df_out)} datasets in {t_total:.1f}s ({t_total/60:.2f} min)", flush=True)
print("=" * 88, flush=True)

# Comparative Analysis
df_hyph = pd.read_csv(os.path.join(SIM_DIR, "hyphaeon_synthetic_results.csv"))
df_cmp = pd.merge(df_out, df_hyph, on="Gene")
df_cmp.to_csv(os.path.join(SIM_DIR, "synthetic_paired_comparison.csv"), index=False)

print("\n" + "=" * 88, flush=True)
print("             HEAD-TO-HEAD COMPARISON: HYPHY BUSTED+S vs HYPHAEON BUSTED-CORAL", flush=True)
print("=" * 88, flush=True)

for sc, grp in df_cmp.groupby("Scenario", sort=False):
    is_pos = grp["True_IsPositive"].iloc[0]
    n = len(grp)
    
    hyphy_calls = grp["HyPhy_Selected"].mean() * 100
    hyph_calls = grp["Selected"].mean() * 100
    
    hyphy_lrt = grp["HyPhy_LRT"].mean()
    hyph_lrt = grp["Omnibus_LRT"].mean()
    
    hyphy_time = grp["HyPhy_Time_s"].mean()
    hyph_time_ms = grp["Time_ms"].mean()
    
    metric_name = "Power" if is_pos else "False Positive Rate (FPR)"
    print(f"\nScenario: {sc} (N = {n} replicates) [{ 'ALT POSITIVE' if is_pos else 'NULL NEGATIVE' }]:")
    print(f"  • {metric_name}:")
    print(f"      - HyPhy BUSTED+S:   {hyphy_calls:.1f}%")
    print(f"      - HyphAeon BUSTED:  {hyph_calls:.1f}%")
    print(f"  • Mean Omnibus LRT:")
    print(f"      - HyPhy BUSTED+S:   {hyphy_lrt:.2f}")
    print(f"      - HyphAeon BUSTED:  {hyph_lrt:.2f}")
    print(f"  • Average Runtime:")
    print(f"      - HyPhy BUSTED+S:   {hyphy_time:.2f} s")
    print(f"      - HyphAeon BUSTED:  {hyph_time_ms:.1f} ms  ({hyphy_time / (hyph_time_ms/1000.0):.1f}x speedup)")

print("=" * 88, flush=True)

