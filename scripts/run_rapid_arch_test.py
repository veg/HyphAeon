#!/usr/bin/env python3
import sys
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from Bio import AlignIO, Phylo

sys.path.insert(0, './scripts')
from train_transformer_selection import PhyloAxialTransformer, CODON_LOOKUP, AA_LOOKUP, compute_mds_coordinates
from predict_regression_nexus import parse_nexus_alignment_and_embedded_tree, calculate_patristic_distances

def run_rapid_arch_test():
    print("=" * 70)
    print("🚀 AXOMEME 2-MINUTE RAPID ARCHITECTURE OVERFIT & DISCRIMINATION TEST")
    print("=" * 70)
    
    device = torch.device("cpu")
    print(f"[*] Running local optimization on device: {device}")
    
    # 1. Load Ground-Truth MEME Target LRTs for HIV_RT.nex
    meme_json_path = '/Users/sergei/Development/hyphy/tests/data/HIV_RT.nex.MEME.json'
    with open(meme_json_path) as f:
        meme_data = json.load(f)

    content = meme_data['MLE']['content']['0']
    df_meme = pd.DataFrame(content, columns=['alpha', 'beta1', 'p1', 'beta_pos', 'p_pos', 'lrt', 'p_value', 'branches_sel', 'tot_len', 'meme_logl', 'fel_logl', 'lrt_meme_fel', 'fel_alpha', 'fel_beta'])
    df_meme['lrt'] = df_meme['lrt'].astype(float)
    df_meme['p_value'] = df_meme['p_value'].astype(float)
    target_lrts = df_meme['lrt'].values # [335]
    
    # 2. Parse Alignment and Tree
    alignment_path = '/Users/sergei/Development/hyphy/tests/data/HIV_RT.nex'
    aln = AlignIO.read(alignment_path, 'nexus')
    seq_dict = {str(rec.id): str(rec.seq).upper() for rec in aln}
    selected_species = list(seq_dict.keys())[:512]
    ref_seq = seq_dict[selected_species[0]]
    total_codons = len(ref_seq) // 3
    
    tree_nwk_path = "HIV_RT_regression_predictions_estimated_tree.nwk"
    tree_real = Phylo.read(tree_nwk_path, 'newick')
    _, dist_real_dict, _ = calculate_patristic_distances(tree_real)
    
    N_sp = len(selected_species)
    dist_real_np = np.zeros((N_sp, N_sp))
    for i, s1 in enumerate(selected_species):
        for j, s2 in enumerate(selected_species):
            if i != j:
                dist_real_np[i, j] = dist_real_dict.get(s1, {}).get(s2, 0.0)
                
    dist_real_np = dist_real_np.astype(np.float32)
    dist_tensor_real = torch.zeros(512, 512, dtype=torch.float32, device=device)
    dist_tensor_real[:N_sp, :N_sp] = torch.from_numpy(dist_real_np).to(device)
    mds_coords_real = torch.zeros(512, 4, dtype=torch.float32, device=device)
    mds_coords_real[:N_sp] = torch.from_numpy(compute_mds_coordinates(dist_real_np, 4)).to(device)
    
    b_c = torch.ones(total_codons, 512, 1, dtype=torch.long, device=device) * 65
    b_a = torch.ones(total_codons, 512, 1, dtype=torch.long, device=device) * 22
    b_p = torch.ones(total_codons, 512, dtype=torch.bool, device=device)
    
    for site in range(total_codons):
        for s_idx, spec in enumerate(selected_species):
            seq = seq_dict[spec]
            codon = seq[site*3:site*3+3].upper()
            if '-' in codon or 'N' in codon or '?' in codon or len(codon) != 3:
                b_p[site, s_idx] = True
            else:
                b_p[site, s_idx] = False
                
            if len(codon) == 3:
                b_encoded = codon.encode('ascii')
                idx = b_encoded[0] * 65536 + b_encoded[1] * 256 + b_encoded[2]
                if idx < len(CODON_LOOKUP):
                    b_c[site, s_idx, 0] = CODON_LOOKUP[idx]
                    b_a[site, s_idx, 0] = AA_LOOKUP[idx]
                    
    # Build CORAL ordinal targets
    thresholds = [0.1, 0.5, 1.0, 2.0, 3.81, 5.14, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0]
    num_thresholds = len(thresholds)
    coral_targets = torch.zeros(total_codons, num_thresholds, device=device)
    for site_idx, lrt_val in enumerate(target_lrts):
        for k, t_val in enumerate(thresholds):
            if lrt_val >= t_val:
                coral_targets[site_idx, k] = 1.0
                
    # 3. Instantiate Clean Model
    model = PhyloAxialTransformer(
        num_tokens=66,
        embed_dim=256,
        num_heads=8,
        num_layers=8,
        window_size=1,
        max_species=512,
        pure_coral=True,
        num_thresholds=num_thresholds
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("\n[*] Starting 60 Local Optimization Steps to Evaluate Architecture Potential...")
    start_time = time.time()
    
    b_d_all = dist_tensor_real.unsqueeze(0).expand(total_codons, -1, -1)
    b_m_all = mds_coords_real.unsqueeze(0).expand(total_codons, -1, -1)
    
    for step in range(1, 16):
        model.train()
        optimizer.zero_grad()
        
        logits = model(b_c, b_a, b_d_all, b_m_all, b_p)
        loss = criterion(logits, coral_targets)
        loss.backward()
        optimizer.step()
        
        if step % 5 == 0 or step == 1 or step == 15:
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pred_lrt_sum = probs.sum(dim=-1).cpu().numpy()
                
            pr, _ = pearsonr(pred_lrt_sum, target_lrts)
            sr, _ = spearmanr(pred_lrt_sum, target_lrts)
            
            s189_pred = pred_lrt_sum[188]
            s245_pred = pred_lrt_sum[244]
            s188_pred = pred_lrt_sum[187]
            s184_pred = pred_lrt_sum[183]
            s200_pred = pred_lrt_sum[199]
            
            print(f"  Step {step:>2}/15 | Loss = {loss.item():.4f} | Pearson r = {pr:+.4f} | Spearman rho = {sr:+.4f}", flush=True)
            print(f"    -> Site 189 (False Pos Target 1.21) : Pred = {s189_pred:.2f}", flush=True)
            print(f"    -> Site 245 (Target 49.93)          : Pred = {s245_pred:.2f}", flush=True)
            print(f"    -> Site 188 (Target 36.92)          : Pred = {s188_pred:.2f}", flush=True)
            print(f"    -> Site 184 (Target 16.40)          : Pred = {s184_pred:.2f}", flush=True)
            print(f"    -> Site 200 (Target 21.39)          : Pred = {s200_pred:.2f}", flush=True)

    elapsed = time.time() - start_time
    print(f"\n🎉 Test Completed in {elapsed:.2f} seconds.")
    
    # Final Rank Analysis
    df_eval = pd.DataFrame({'site': range(1, 336), 'lrt': target_lrts, 'pred': pred_lrt_sum})
    df_eval['pred_rank'] = df_eval['pred'].rank(ascending=False, method='min').astype(int)
    
    print("\n" + "=" * 70)
    print("🔬 FINAL CLEAN ARCHITECTURE OVERFIT EVALUATION")
    print("=" * 70)
    for key_site in [245, 188, 200, 184, 189, 10]:
        row = df_eval[df_eval['site'] == key_site].iloc[0]
        print(f"  Site {row['site']:>3} | Ground Truth LRT = {row['lrt']:>6.2f} | Pred = {row['pred']:>5.2f} | Rank = #{row['pred_rank']:>2}")
    print("=" * 70)

if __name__ == "__main__":
    run_rapid_arch_test()
