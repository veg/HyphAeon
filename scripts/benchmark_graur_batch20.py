#!/usr/bin/env python3
import os
import sys
import glob
import time
import json
import random
import subprocess
import concurrent.futures
import pandas as pd
import numpy as np
import torch
from safetensors.torch import load_file

MSA_DIR = "/Users/sergei/Dropbox/Work/BUSTED-E/data/graur-7-species/MSAs"
OUTPUT_DIR = "/Users/sergei/Projects/TOGA_MEME/bench/graur_batch20"
os.makedirs(OUTPUT_DIR, exist_ok=True)
WEIGHTS_PATH = "/Users/sergei/Projects/TOGA_MEME/axomeme_repo/model.safetensors"

all_alns = sorted(glob.glob(os.path.join(MSA_DIR, "OMA*.aln")))
valid_pairs = []
for aln in all_alns:
    tree = aln + ".raxml.bestTree"
    if os.path.exists(tree) and os.path.getsize(aln) > 1000 and os.path.getsize(tree) > 20:
        valid_pairs.append((aln, tree))

# Random sample of 20 NEW datasets (seed 2026)
random.seed(2026)
sample_pairs = random.sample(valid_pairs, 20)

print("=" * 80, flush=True)
print(f"  RUNNING BENCHMARK ON 20 NEW RANDOM GRAUR DATASETS (BUSTED+S vs BUSTED-CORAL)", flush=True)
print("=" * 80, flush=True)

# Step 1: Run HyPhy BUSTED+S in parallel (6 workers)
def run_hyphy(pair):
    aln_path, tree_path = pair
    gene_name = os.path.basename(aln_path).replace(".aln", "")
    hyphy_json = os.path.join(OUTPUT_DIR, f"{gene_name}_hyphy.json")
    
    t0 = time.time()
    cmd = [
        "hyphy", "busted",
        "--alignment", aln_path,
        "--tree", tree_path,
        "--srv", "Yes",
        "--output", hyphy_json
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    t_hyphy = time.time() - t0
    
    if res.returncode != 0:
        return {"gene": gene_name, "error": res.stderr[:200]}
        
    with open(hyphy_json) as f:
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
        
    return {
        "gene": gene_name,
        "aln": aln_path,
        "tree": tree_path,
        "hyphy_p": p_val,
        "hyphy_lrt": lrt,
        "hyphy_w3": w3,
        "hyphy_p3": p3,
        "hyphy_time": t_hyphy
    }

print("[*] Running HyPhy BUSTED+S across 6 CPU workers...", flush=True)
hyphy_results = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(run_hyphy, pair) for pair in sample_pairs]
    for fut in concurrent.futures.as_completed(futures):
        res = fut.result()
        if "error" not in res:
            hyphy_results[res["gene"]] = res
            print(f"  [✓ HyPhy] {res['gene']}: p = {res['hyphy_p']:.4e}, LRT = {res['hyphy_lrt']:5.2f} in {res['hyphy_time']:.1f}s", flush=True)
        else:
            print(f"  [!] HyPhy error on {res['gene']}: {res['error']}", flush=True)

# Step 2: Run HyphAeon BUSTED-CORAL batch mode on the same 20 datasets
sample_dir = os.path.join(OUTPUT_DIR, "batch_msas")
os.makedirs(sample_dir, exist_ok=True)
for aln, tree in sample_pairs:
    subprocess.run(["cp", aln, sample_dir])
    subprocess.run(["cp", tree, sample_dir])

print("\n[*] Running HyphAeon BUSTED-CORAL (In-Memory Batch)...", flush=True)
hyph_csv = os.path.join(OUTPUT_DIR, "hyphaeon_batch_results.csv")
cmd_hyph = [
    "hyphaeon", "busted",
    "-d", sample_dir,
    "-w", WEIGHTS_PATH,
    "-c", hyph_csv
]
t0_batch = time.time()
subprocess.run(cmd_hyph)
t_batch = time.time() - t0_batch
print(f"[*] HyphAeon Batch Complete in {t_batch:.2f}s total!", flush=True)

# Step 3: Merge & Build Clean Comparison Tables
df_hyph = pd.read_csv(hyph_csv)
rows = []
for _, r_hyph in df_hyph.iterrows():
    gene = r_hyph["Gene"]
    if gene in hyphy_results:
        r_hyphy = hyphy_results[gene]
        sel_hyphy = bool(r_hyphy["hyphy_p"] < 0.05)
        sel_hyph = bool(r_hyph["Selected"])
        match = (sel_hyphy == sel_hyph)
        rows.append({
            "Gene": gene,
            "Sites": int(r_hyph["Sites"]),
            "HyPhy_p": r_hyphy["hyphy_p"],
            "HyPhy_LRT": r_hyphy["hyphy_lrt"],
            "HyPhy_w3": r_hyphy["hyphy_w3"],
            "HyPhy_Time": r_hyphy["hyphy_time"],
            "Hyph_pACAT": r_hyph["p_ACAT"],
            "Hyph_Prob": r_hyph["Selection_Prob"],
            "Hyph_w3": r_hyph["Omega_3"],
            "Hyph_Time_ms": r_hyph["Time_ms"],
            "Speedup": r_hyphy["hyphy_time"] / (r_hyph["Time_ms"] / 1000.0),
            "Selected_HyPhy": sel_hyphy,
            "Selected_Hyph": sel_hyph,
            "Concordant": match
        })

df_all = pd.DataFrame(rows)
df_all.to_csv(os.path.join(OUTPUT_DIR, "graur_20_comparison.csv"), index=False)

