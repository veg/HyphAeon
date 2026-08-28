#!/usr/bin/env python3
import sys
import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
from Bio import AlignIO, Phylo

sys.path.insert(0, './scripts')
from train_transformer_selection import PhyloAxialTransformer, CODON_LOOKUP, AA_LOOKUP, compute_mds_coordinates
from predict_regression_nexus import parse_nexus_alignment_and_embedded_tree, calculate_patristic_distances

def run_preflight_check(model_path, alignment_path, device_str='cpu'):
    print("=" * 70)
    print("🚀 AXOMEME 60-SECOND PRE-FLIGHT MODEL DIAGNOSTIC SUITE")
    print("=" * 70)
    
    device = torch.device(device_str)
    
    if not os.path.exists(model_path):
        print(f"[❌ PRE-FLIGHT CRITICAL FAIL] Checkpoint file not found: {model_path}")
        return False
        
    if not os.path.exists(alignment_path):
        print(f"[❌ PRE-FLIGHT CRITICAL FAIL] Alignment file not found: {alignment_path}")
        return False
        
    print(f"[*] Loading Model Checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    model = PhyloAxialTransformer(
        num_tokens=66,
        embed_dim=256,
        num_heads=8,
        num_layers=8,
        window_size=1,
        max_species=512,
        pure_coral=True,
        num_thresholds=16
    ).to(device)
    
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model.load_state_dict(filtered_dict, strict=False)
    model.eval()
    
    # 1. Parse Alignment
    aln = AlignIO.read(alignment_path, 'nexus')
    seq_dict = {str(rec.id): str(rec.seq).upper() for rec in aln}
    selected_species = list(seq_dict.keys())[:512]
    ref_seq = seq_dict[selected_species[0]]
    total_codons = len(ref_seq) // 3
    
    # 2. Parse Tree / Estimate Tree
    tree_nwk_path = alignment_path.rsplit('.', 1)[0] + "_estimated_tree.nwk"
    if not os.path.exists(tree_nwk_path):
        tree_nwk_path = "HIV_RT_regression_predictions_estimated_tree.nwk"
        
    if os.path.exists(tree_nwk_path):
        tree_real = Phylo.read(tree_nwk_path, 'newick')
        _, dist_real_dict, _ = calculate_patristic_distances(tree_real)
        
        N_sp = len(selected_species)
        dist_real_np = np.zeros((N_sp, N_sp))
        for i, s1 in enumerate(selected_species):
            for j, s2 in enumerate(selected_species):
                if i != j:
                    dist_real_np[i, j] = dist_real_dict.get(s1, {}).get(s2, 0.0)
                    
        dist_tensor_real = torch.zeros(512, 512, device=device)
        dist_tensor_real[:N_sp, :N_sp] = torch.from_numpy(dist_real_np).to(device)
        mds_coords_real = torch.zeros(512, 4, device=device)
        mds_coords_real[:N_sp] = torch.from_numpy(compute_mds_coordinates(dist_real_np, 4)).to(device)
    else:
        print("[!] Warning: Tree nwk file not found. Running with flat distance matrix.")
        dist_tensor_real = torch.zeros(512, 512, device=device)
        mds_coords_real = torch.zeros(512, 4, device=device)
        
    # 3. Build Tensors
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

    # Register hook to capture stream fusion inputs
    saved_inputs = {}
    def hook_fusion(module, input, output):
        saved_inputs['fusion_input'] = input[0]
        
    model.stream_fusion.register_forward_hook(hook_fusion)

    all_preds = []
    all_expert_norms = []
    
    for start in range(0, total_codons, 64):
        end = min(start + 64, total_codons)
        b_c_sub = b_c[start:end]
        b_a_sub = b_a[start:end]
        b_p_sub = b_p[start:end]
        b_d_sub = dist_tensor_real.unsqueeze(0).expand(end - start, -1, -1)
        b_m_sub = mds_coords_real.unsqueeze(0).expand(end - start, -1, -1)
        
        with torch.no_grad():
            preds = model(b_c_sub, b_a_sub, b_d_sub, b_m_sub, b_p_sub)
            all_preds.extend(preds.cpu().numpy())
            
            fin = saved_inputs['fusion_input'] # [batch_size, num_experts * 256]
            num_exp = fin.shape[1] // 256
            for b in range(fin.shape[0]):
                norms = [torch.norm(fin[b, i*256:(i+1)*256]).item() for i in range(num_exp)]
                all_expert_norms.append(norms)
                
    all_preds = np.array(all_preds)
    all_expert_norms = np.array(all_expert_norms) # [total_codons, num_experts]
    
    df_res = pd.DataFrame({
        'codon_site': list(range(1, total_codons + 1)),
        'pred_lrt': all_preds
    })
    df_res['pred_rank'] = df_res['pred_lrt'].rank(ascending=False, method='min').astype(int)
    df_res['pred_z'] = (df_res['pred_lrt'] - df_res['pred_lrt'].mean()) / df_res['pred_lrt'].std()
    
    # Run 4 Pre-Flight Diagnostic Gates
    gate1_pass, gate2_pass, gate3_pass, gate4_pass = True, True, True, True
    
    print("\n" + "=" * 70)
    print("🔬 PRE-FLIGHT GATE EVALUATION RESULTS")
    print("=" * 70)
    
    # Gate 1: Neutral Background FPR Check (Site 10)
    s10_lrt = df_res.loc[df_res['codon_site'] == 10, 'pred_lrt'].values[0]
    s10_z = df_res.loc[df_res['codon_site'] == 10, 'pred_z'].values[0]
    if s10_lrt > 0.50:
        gate1_pass = False
        print(f"❌ GATE 1 FAILED (Neutral Background): Site 10 Pred LRT = {s10_lrt:.4f} (> 0.50 threshold)")
    else:
        print(f"✅ GATE 1 PASSED (Neutral Background): Site 10 Pred LRT = {s10_lrt:.4f} (Z = {s10_z:+.2f})")
        
    # Gate 2: Low-Frequency Variant Discrimination (Site 189, 6/476 Mutants, True LRT = 1.21)
    s189_lrt = df_res.loc[df_res['codon_site'] == 189, 'pred_lrt'].values[0]
    s189_rank = df_res.loc[df_res['codon_site'] == 189, 'pred_rank'].values[0]
    if s189_lrt >= 3.81:
        gate2_pass = False
        print(f"❌ GATE 2 FAILED (False Positive Burst): Site 189 Pred LRT = {s189_lrt:.4f} (Rank #{s189_rank}, Exceeds Tier 2 gate 3.81)")
    else:
        print(f"✅ GATE 2 PASSED (False Positive Control): Site 189 Pred LRT = {s189_lrt:.4f} (Rank #{s189_rank}, Controlled)")
        
    # Gate 3: True Positive Selection Sensitivity (Sites 245, 188, 200)
    s245_lrt = df_res.loc[df_res['codon_site'] == 245, 'pred_lrt'].values[0]
    s245_rank = df_res.loc[df_res['codon_site'] == 245, 'pred_rank'].values[0]
    s188_rank = df_res.loc[df_res['codon_site'] == 188, 'pred_rank'].values[0]
    s200_rank = df_res.loc[df_res['codon_site'] == 200, 'pred_rank'].values[0]
    
    if s245_rank > 10 or s188_rank > 50:
        gate3_pass = False
        print(f"❌ GATE 3 FAILED (Sensitivity): Site 245 Rank #{s245_rank}, Site 188 Rank #{s188_rank}")
    else:
        print(f"✅ GATE 3 PASSED (Sensitivity): Site 245 Rank #{s245_rank} (LRT = {s245_lrt:.2f}), Site 188 Rank #{s188_rank}, Site 200 Rank #{s200_rank}")
        
    # Gate 4: Feature Activation Saturation Audit
    mean_expert_norms = all_expert_norms.mean(axis=0)
    s189_expert_norms = all_expert_norms[188]
    max_norm_ratio = np.max(s189_expert_norms / (mean_expert_norms + 1e-5))
    
    if max_norm_ratio > 1.50:
        gate4_pass = False
        print(f"❌ GATE 4 FAILED (Expert Saturation): Max Expert Norm Ratio at Site 189 = {max_norm_ratio:.2f}x (> 1.50x limit)")
    else:
        print(f"✅ GATE 4 PASSED (Activation Stability): Max Expert Norm Ratio = {max_norm_ratio:.2f}x (No Saturated Spikes)")
        
    print("=" * 70)
    all_passed = gate1_pass and gate2_pass and gate3_pass and gate4_pass
    if all_passed:
        print("🎉 ALL 4 PRE-FLIGHT DIAGNOSTIC GATES PASSED CLEANLY!")
        print("   Model is verified safe and effective for downstream analysis.")
    else:
        print("⚠️ MODEL FAILED PRE-FLIGHT DIAGNOSTIC SUITE.")
        print("   Do NOT deploy or train on this checkpoint without resolving failed gates.")
    print("=" * 70)
    
    return all_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AxoMeme 60-Second Pre-Flight Diagnostic Suite")
    parser.add_argument("--model", default="./axomeme_2.7.pt", help="Path to checkpoint file")
    parser.add_argument("--alignment", default="/Users/sergei/Development/hyphy/tests/data/HIV_RT.nex", help="Path to alignment file")
    parser.add_argument("--device", default="cpu", help="Device to run inference on")
    args = parser.parse_args()
    
    run_preflight_check(args.model, args.alignment, args.device)
