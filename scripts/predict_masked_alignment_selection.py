#!/usr/bin/env python3
"""
predict_masked_alignment_selection.py

Positive Selection Inference on Masked Phylogenetic Alignment Transformer (PhyloMLM)
-----------------------------------------------------------------------------------
Performs positive selection detection from Multiple Sequence Alignments and Trees
using a self-supervised PhyloMLM checkpoint.

Provides 3 Complementary Statistical Inference Modes:
  Mode 1: Cumulative Evolutionary Surprisal (Residual Negative Log-Likelihood)
  Mode 2: Non-Synonymous vs Synonymous Log-Odds Ratio (Local dN/dS Selection Tilt)
  Mode 3: Leave-One-Out (Masked-Column) Likelihood Ratio Test (LRT)
"""

import os
import sys
import math
import json
import re
import gzip
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from io import StringIO
from Bio import Phylo
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

from train_masked_alignment_transformer import (
    PhyloMaskedAxialTransformer,
    CODON_TABLE,
    CODONS_ORDERED,
    CODON_TO_ID,
    ID_TO_CODON,
    AA_TO_ID,
    PAD_CODON_ID,
    PAD_AA_ID,
    MASK_CODON_ID,
    MASK_AA_ID
)

from train_transformer_selection import compute_mds_coordinates

# ---------------------------------------------------------------------------
# Pre-compute Synonymous & Non-Synonymous Mask Matrices (64 x 64)
# ---------------------------------------------------------------------------
def build_codon_exchangeability_matrices():
    syn_mask = torch.zeros(64, 64, dtype=torch.bool)
    nonsyn_mask = torch.zeros(64, 64, dtype=torch.bool)
    
    for i, c1 in enumerate(CODONS_ORDERED):
        aa1 = CODON_TABLE[c1]
        for j, c2 in enumerate(CODONS_ORDERED):
            if i == j:
                continue
            aa2 = CODON_TABLE[c2]
            if aa1 == aa2:
                syn_mask[i, j] = True
            else:
                nonsyn_mask[i, j] = True
                
    return syn_mask, nonsyn_mask


# ---------------------------------------------------------------------------
# Alignment & Tree Parsers
# ---------------------------------------------------------------------------
def parse_nexus_file(filepath):
    filepath = os.path.expanduser(filepath)
    open_func = gzip.open if filepath.endswith('.gz') else open
    seq_dict = {}
    taxlabels = []
    tree_lines = []
    tree_str = None
    
    with open_func(filepath, 'rt') as f:
        lines = f.readlines()
        
    in_matrix = False
    in_taxa = False
    
    for line in lines:
        stripped = line.strip()
        if '[' in stripped and ']' in stripped:
            stripped = re.sub(r'\[.*?\]', '', stripped).strip()
            
        if not stripped:
            continue
            
        upper_line = stripped.upper()
        
        if 'BEGIN TAXA;' in upper_line or 'BEGIN TAXA ;' in upper_line:
            in_taxa = True
            continue
        if in_taxa:
            if 'TAXLABELS' in upper_line:
                content = stripped[stripped.upper().find('TAXLABELS') + 9:].strip().rstrip(';')
                taxlabels.extend(content.split())
            if ';' in stripped:
                in_taxa = False
                
        if 'MATRIX' in upper_line and not in_matrix:
            in_matrix = True
            continue
            
        if in_matrix:
            if stripped == ';' or upper_line.startswith('END;') or upper_line.startswith('END ;'):
                in_matrix = False
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                sp_name = parts[0].replace("'", "").replace('"', '')
                seq = "".join(parts[1:]).upper()
                seq_dict[sp_name] = seq
            elif len(parts) == 1 and taxlabels and len(seq_dict) < len(taxlabels):
                sp_name = taxlabels[len(seq_dict)]
                seq = parts[0].upper()
                seq_dict[sp_name] = seq
                
        if ('TREE ' in upper_line or 'UTREE ' in upper_line or stripped.startswith('(')) and not in_matrix:
            tree_lines.append(stripped)
            
    if tree_lines:
        combined_tree = " ".join(tree_lines)
        first_p = combined_tree.find('(')
        if first_p != -1:
            last_s = combined_tree.rfind(';')
            if last_s != -1 and last_s > first_p:
                tree_str = combined_tree[first_p:last_s+1]
            else:
                tree_str = combined_tree[first_p:] + ';'
                
    return seq_dict, taxlabels, tree_str


def compute_fast_dist_matrix(tree, taxa):
    leaves = tree.get_terminals()
    node_to_root = {}
    node_to_parent = {}
    def traverse(node, current_dist, parent):
        node_to_root[node] = current_dist
        node_to_parent[node] = parent
        for child in node.clades:
            traverse(child, current_dist + (child.branch_length or 0.0), node)
    traverse(tree.root, 0.0, None)
    leaf_by_name = {leaf.name: leaf for leaf in leaves if leaf.name}
    leaf_paths = {name: [] for name in taxa}
    for name in taxa:
        curr = leaf_by_name.get(name, None)
        while curr is not None:
            leaf_paths[name].append(curr)
            curr = node_to_parent[curr]
    
    N = len(taxa)
    dist_mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        name1 = taxa[i]
        path1 = leaf_paths[name1]
        set1 = set(path1)
        for j in range(i + 1, N):
            name2 = taxa[j]
            path2 = leaf_paths[name2]
            lca = None
            for node in path2:
                if node in set1:
                    lca = node
                    break
            if lca is not None:
                d = node_to_root[leaf_by_name[name1]] + node_to_root[leaf_by_name[name2]] - 2 * node_to_root[lca]
            else:
                d = node_to_root[leaf_by_name[name1]] + node_to_root[leaf_by_name[name2]]
            dist_mat[i, j] = d
            dist_mat[j, i] = d
    return dist_mat


# ---------------------------------------------------------------------------
# Selection Mode Implementations
# ---------------------------------------------------------------------------
def compute_evolutionary_surprisal(logits, codon_ids, bg_freqs=None):
    log_probs = F.log_softmax(logits, dim=-1) # [B, S, L, 64]
    B, S, L, _ = logits.shape
    
    true_codons = codon_ids.clamp(0, 63).unsqueeze(-1) # [B, S, L, 1]
    obs_log_p = torch.gather(log_probs, dim=-1, index=true_codons).squeeze(-1) # [B, S, L]
    
    if bg_freqs is None:
        bg_log_p = -math.log(64.0)
    else:
        bg_log_p = torch.log(bg_freqs.clamp(min=1e-6)).view(1, 1, 1, 64)
        bg_log_p = torch.gather(bg_log_p.expand(B, S, L, -1), dim=-1, index=true_codons).squeeze(-1)
        
    raw_surprisal = -obs_log_p + bg_log_p # [B, S, L]
    valid_mask = (codon_ids < 64)
    raw_surprisal = raw_surprisal * valid_mask.float()
    
    taxa_counts = valid_mask.sum(dim=1).clamp(min=1)
    site_surprisal = (raw_surprisal.sum(dim=1) / taxa_counts).squeeze(0).cpu().numpy()
    branch_surprisals = raw_surprisal.squeeze(0).cpu().numpy()
    
    return site_surprisal, branch_surprisals


def compute_local_dnds_log_odds(logits, codon_ids, syn_mask, nonsyn_mask):
    probs = F.softmax(logits, dim=-1) # [1, S, L, 64]
    _, S, L, _ = logits.shape
    device = logits.device
    
    syn_mask = syn_mask.to(device)
    nonsyn_mask = nonsyn_mask.to(device)
    
    syn_counts = syn_mask.sum(dim=1, keepdim=True).float().view(1, 1, 1, 64) # [1, 1, 1, 64]
    nonsyn_counts = nonsyn_mask.sum(dim=1, keepdim=True).float().view(1, 1, 1, 64)
    
    true_codons = codon_ids.clamp(0, 63)
    
    cell_syn_masks = syn_mask[true_codons] # [1, S, L, 64]
    cell_nonsyn_masks = nonsyn_mask[true_codons]
    
    # Identify which positions have zero synonymous codons (e.g. ATG, TGG)
    has_syn = (cell_syn_masks.sum(dim=-1) > 0) # [1, S, L]
    
    p_syn = (probs * cell_syn_masks.float()).sum(dim=-1).clamp(min=1e-8)
    p_nonsyn = (probs * cell_nonsyn_masks.float()).sum(dim=-1).clamp(min=1e-8)
    
    syn_counts_4d = syn_counts.expand(1, S, L, 64)
    nonsyn_counts_4d = nonsyn_counts.expand(1, S, L, 64)
    
    norm_syn = torch.gather(syn_counts_4d, dim=-1, index=true_codons.unsqueeze(-1)).squeeze(-1).clamp(min=1.0)
    norm_nonsyn = torch.gather(nonsyn_counts_4d, dim=-1, index=true_codons.unsqueeze(-1)).squeeze(-1).clamp(min=1.0)
    
    local_omega = (p_nonsyn / norm_nonsyn) / (p_syn / norm_syn)
    log_omega_tilt = torch.log(local_omega)
    
    # For single-codon amino acids (Met, Trp), neutral tilt is 0.0
    log_omega_tilt = torch.where(has_syn, log_omega_tilt, torch.zeros_like(log_omega_tilt))
    
    pos_selection_tilt = F.relu(log_omega_tilt)
    valid_mask = (codon_ids < 64)
    pos_selection_tilt = pos_selection_tilt * valid_mask.float()
    
    taxa_counts = (valid_mask & has_syn).sum(dim=1).clamp(min=1)
    site_dnds_score = (pos_selection_tilt.sum(dim=1) / taxa_counts).squeeze(0).cpu().numpy()
    branch_dnds_tilt = log_omega_tilt.squeeze(0).cpu().numpy()
    
    return site_dnds_score, branch_dnds_tilt


def compute_leave_one_out_lrt(model, codon_sub, aa_sub, dist_sub, mds_sub, window_size=9):
    device = codon_sub.device
    num_taxa, num_sites = codon_sub.shape
    lrt_scores = []
    num_windows = math.ceil(num_sites / window_size)
    win_count = 0
    
    with torch.no_grad():
        for start in range(0, num_sites, window_size):
            win_count += 1
            end = min(start + window_size, num_sites)
            cur_L = end - start
            
            if win_count % 5 == 0 or win_count == num_windows:
                print(f"  • Evaluated LOO-LRT Window {win_count}/{num_windows} (Codons {start+1}-{end})...", flush=True)
            
            c_win = codon_sub[:, start:end].unsqueeze(0).clone()
            a_win = aa_sub[:, start:end].unsqueeze(0).clone()
            
            if cur_L < window_size:
                pad_len = window_size - cur_L
                c_win = F.pad(c_win, (0, pad_len), value=PAD_CODON_ID)
                a_win = F.pad(a_win, (0, pad_len), value=PAD_AA_ID)
                
            logits_unmasked = model(c_win, a_win, dist_sub, mds_sub)
            log_p_unmasked = F.log_softmax(logits_unmasked[:, :, :cur_L, :], dim=-1)
            
            # Batch all cur_L masked columns together in 1 single forward pass
            c_batch = c_win.repeat(cur_L, 1, 1) # [cur_L, S, window_size]
            a_batch = a_win.repeat(cur_L, 1, 1)
            d_batch = dist_sub.repeat(cur_L, 1, 1)
            m_batch = mds_sub.repeat(cur_L, 1, 1)
            
            for col_idx in range(cur_L):
                c_batch[col_idx, :, col_idx] = MASK_CODON_ID
                a_batch[col_idx, :, col_idx] = MASK_AA_ID
                
            logits_masked_batch = model(c_batch, a_batch, d_batch, m_batch)
            log_p_masked_batch = F.log_softmax(logits_masked_batch, dim=-1)
            
            win_lrts = []
            for col_idx in range(cur_L):
                valid_mask = (c_win[0, :, col_idx] < 64)
                true_codons = codon_sub[:, start + col_idx].unsqueeze(0).clamp(0, 63).unsqueeze(-1)
                
                ll_unmasked = torch.gather(log_p_unmasked[:, :, col_idx, :], dim=-1, index=true_codons).squeeze(-1)
                ll_masked = torch.gather(log_p_masked_batch[col_idx : col_idx+1, :, col_idx, :], dim=-1, index=true_codons).squeeze(-1)
                
                delta_ll = 2.0 * F.relu(ll_unmasked - ll_masked) * valid_mask.float().unsqueeze(0)
                mean_lrt = (delta_ll.sum() / valid_mask.float().sum().clamp(min=1.0)).item()
                win_lrts.append(mean_lrt)
                
            lrt_scores.extend(win_lrts)
            
    return np.array(lrt_scores)


# ---------------------------------------------------------------------------
# Main Inference Execution
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run Positive Selection Inference on Masked Alignment Transformer (PhyloMLM)")
    parser.add_argument('--model', type=str, required=True, help="Path to trained PhyloMLM checkpoint (.pt)")
    parser.add_argument('--alignment', type=str, required=True, help="Input alignment (.nex, .fasta, or .npz)")
    parser.add_argument('--tree', type=str, default=None, help="Input Newick tree (.nwk) if separate from alignment")
    parser.add_argument('--output', type=str, default="phylomlm_selection_results.csv", help="Output CSV results file")
    parser.add_argument('--mode', type=str, default="all", choices=["all", "surprisal", "dnds", "lrt"], help="Selection inference mode")
    parser.add_argument('--meme_benchmark', type=str, default=None, help="Path to ground-truth HyPhy MEME JSON for benchmarking")
    parser.add_argument('--max_species', type=int, default=128, help="Max species cap for forward windowing")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        torch.set_num_threads(8)
        
    print(f"[*] Running PhyloMLM Inference on device: {device}", flush=True)

    alignment_path = os.path.expanduser(args.alignment)
    model_path = os.path.expanduser(args.model)
    tree_path = os.path.expanduser(args.tree) if args.tree else None
    benchmark_path = os.path.expanduser(args.meme_benchmark) if args.meme_benchmark else None

    # 1. Load Alignment & Tree
    if alignment_path.endswith('.npz'):
        npz = np.load(alignment_path)
        codon_mat = npz['codon_ids_matrix']
        aa_mat = npz['aa_ids_matrix']
        dist_arr = npz['dist_arr']
        mds_coords = npz['mds_coords']
        num_taxa, num_sites = codon_mat.shape
        ref_aas = [CODON_TABLE[ID_TO_CODON[c]] if c < 64 else 'X' for c in codon_mat[0]]
    else:
        seq_dict, taxlabels, embedded_tree_str = parse_nexus_file(alignment_path)
        if not seq_dict:
            from Bio import SeqIO
            records = list(SeqIO.parse(alignment_path, 'fasta'))
            seq_dict = {r.id: str(r.seq) for r in records}
            taxlabels = list(seq_dict.keys())
            
        tree_str = tree_path if tree_path else embedded_tree_str
        clean_tree_str = re.sub(r'\{[^}]*\}', '', tree_str)
        clean_tree_str = re.sub(r'\[.*?\]', '', clean_tree_str).strip().rstrip('; \t\n\r') + ';'
        tree_obj = Phylo.read(StringIO(clean_tree_str), 'newick')
        
        matching_species = [term.name for term in tree_obj.get_terminals() if term.name in seq_dict]
        if not matching_species:
            matching_species = list(seq_dict.keys())
            
        N = min(len(matching_species), args.max_species)
        selected_species = matching_species[:N]
        
        ref_seq = seq_dict[selected_species[0]]
        num_sites = len(ref_seq) // 3
        num_taxa = len(selected_species)
        
        codon_mat = np.zeros((num_taxa, num_sites), dtype=np.int64)
        aa_mat = np.zeros((num_taxa, num_sites), dtype=np.int64)
        
        for i, sp in enumerate(selected_species):
            seq = seq_dict[sp]
            for s in range(num_sites):
                codon = seq[s*3:s*3+3].upper()
                c_id = CODON_TO_ID.get(codon, PAD_CODON_ID)
                aa = CODON_TABLE.get(codon, '*')
                a_id = AA_TO_ID.get(aa, PAD_AA_ID)
                codon_mat[i, s] = c_id
                aa_mat[i, s] = a_id
                
        dist_arr = compute_fast_dist_matrix(tree_obj, selected_species)
        mds_coords = compute_mds_coordinates(dist_arr, n_components=4)
        ref_aas = []
        for s in range(num_sites):
            col = codon_mat[:, s]
            valid_col = col[col < 64]
            if len(valid_col) > 0:
                ref_aas.append(CODON_TABLE[ID_TO_CODON[valid_col[0]]])
            else:
                ref_aas.append('-')

    print(f"[*] Alignment loaded: {num_taxa} species, {num_sites} codons.", flush=True)

    # 2. Load Checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    embed_dim = ckpt.get('embed_dim', 192)
    num_heads = ckpt.get('num_heads', 6)
    num_layers = ckpt.get('num_layers', 4)
    window_size = ckpt.get('window_size', 9)

    model = PhyloMaskedAxialTransformer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        window_size=window_size
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[*] Model Checkpoint loaded: Dim={embed_dim}, Layers={num_layers}, Heads={num_heads}, Window={window_size}", flush=True)

    # Subsample species if necessary
    max_sp = min(num_taxa, args.max_species)
    c_sub = torch.tensor(codon_mat[:max_sp, :], dtype=torch.long, device=device)
    a_sub = torch.tensor(aa_mat[:max_sp, :], dtype=torch.long, device=device)
    d_sub = torch.tensor(dist_arr[:max_sp, :max_sp], dtype=torch.float32, device=device).unsqueeze(0)
    m_sub = torch.tensor(mds_coords[:max_sp, :], dtype=torch.float32, device=device).unsqueeze(0)

    # 3. Forward Pass in Windows
    print(f"[*] Running Forward Windows ({math.ceil(num_sites / window_size)} windows)...", flush=True)
    all_logits = []
    with torch.no_grad():
        for start in range(0, num_sites, window_size):
            end = min(start + window_size, num_sites)
            cur_L = end - start
            c_win = c_sub[:, start:end].unsqueeze(0)
            a_win = a_sub[:, start:end].unsqueeze(0)
            
            if cur_L < window_size:
                pad_len = window_size - cur_L
                c_win = F.pad(c_win, (0, pad_len), value=PAD_CODON_ID)
                a_win = F.pad(a_win, (0, pad_len), value=PAD_AA_ID)
                
            logits = model(c_win, a_win, d_sub, m_sub)
            all_logits.append(logits[:, :, :cur_L, :].cpu())
            
    full_logits = torch.cat(all_logits, dim=2).to(device)
    full_c_sub = c_sub.unsqueeze(0)

    # 4. Compute Inferences Across Modes
    syn_mask, nonsyn_mask = build_codon_exchangeability_matrices()
    
    site_surprisal, _ = compute_evolutionary_surprisal(full_logits, full_c_sub)
    site_dnds_tilt, _ = compute_local_dnds_log_odds(full_logits, full_c_sub, syn_mask, nonsyn_mask)
    
    if args.mode in ["all", "lrt"]:
        print("[*] Running Mode 3 (Leave-One-Out LRT)...", flush=True)
        site_loo_lrt = compute_leave_one_out_lrt(model, c_sub, a_sub, d_sub, m_sub, window_size=window_size)
    else:
        site_loo_lrt = np.zeros(num_sites)

    # 5. Composite Selection Score
    df_out = pd.DataFrame({
        'site_index': np.arange(1, num_sites + 1),
        'ref_aa': ref_aas,
        'mode1_surprisal': site_surprisal,
        'mode2_dnds_tilt': site_dnds_tilt,
        'mode3_loo_lrt': site_loo_lrt
    })

    composite = (
        (site_surprisal - np.min(site_surprisal)) / max(1e-6, np.ptp(site_surprisal)) * 0.4 +
        (site_dnds_tilt - np.min(site_dnds_tilt)) / max(1e-6, np.ptp(site_dnds_tilt)) * 0.3 +
        (site_loo_lrt - np.min(site_loo_lrt)) / max(1e-6, np.ptp(site_loo_lrt)) * 0.3
    ) * 100.0
    df_out['composite_selection_score'] = composite

    # 6. Benchmark Evaluation against Ground Truth (if provided)
    if benchmark_path and os.path.exists(benchmark_path):
        with open(benchmark_path) as f:
            meme = json.load(f)
        recs = meme['MLE']['content']['0']
        meme_lrt = [r[5] for r in recs[:num_sites]]
        meme_pval = [r[6] for r in recs[:num_sites]]
        
        df_out['meme_lrt'] = meme_lrt
        df_out['meme_pval'] = meme_pval
        df_out['true_05'] = (df_out['meme_pval'] <= 0.05).astype(int)
        df_out['true_10'] = (df_out['meme_pval'] <= 0.10).astype(int)
        
        # Mode-specific score selector
        if args.mode == "surprisal":
            target_score_col = 'mode1_surprisal'
            score_label = "Mode 1 Surprisal"
        elif args.mode == "dnds":
            target_score_col = 'mode2_dnds_tilt'
            score_label = "Mode 2 dN/dS Tilt"
        elif args.mode == "lrt":
            target_score_col = 'mode3_loo_lrt'
            score_label = "Mode 3 LOO LRT"
        else:
            target_score_col = 'composite_selection_score'
            score_label = "Composite Selection Score"
            
        roc_target_05 = roc_auc_score(df_out['true_05'], df_out[target_score_col]) if df_out['true_05'].sum() > 0 else float('nan')
        roc_target_10 = roc_auc_score(df_out['true_10'], df_out[target_score_col]) if df_out['true_10'].sum() > 0 else float('nan')
        r_target, _ = pearsonr(np.log1p(np.maximum(df_out[target_score_col], 0.0)), np.log1p(np.maximum(df_out['meme_lrt'], 0.0)))
        
        print("\n" + "=" * 90, flush=True)
        print(f"📊 GROUND-TRUTH HYPHY MEME BENCHMARK ({num_sites} Codons | Mode: {args.mode.upper()})", flush=True)
        print("=" * 90, flush=True)
        print(f"  • {score_label} ROC-AUC (p <= 0.05) : {roc_target_05:.4f}", flush=True)
        print(f"  • {score_label} ROC-AUC (p <= 0.10) : {roc_target_10:.4f}", flush=True)
        print(f"  • Pearson r ({score_label} vs MEME LRT): {r_target:.4f}", flush=True)
        
        print(f"\n🏆 Top 15 Ranked Positive Selection Sites (Ranked by {score_label}):", flush=True)
        top15 = df_out.sort_values(target_score_col, ascending=False).head(15)
        for rank, (_, r) in enumerate(top15.iterrows(), 1):
            status = '✅ Pos (p<=0.05)' if r['meme_pval'] <= 0.05 else ('⚠️ Pos (p<=0.10)' if r['meme_pval'] <= 0.10 else '❌ Neg')
            val = r[target_score_col]
            print(f"  Rank {rank:2d} | Site {int(r['site_index']):3d} ({r['ref_aa']}) | {score_label}: {val:6.3f} | MEME LRT: {r['meme_lrt']:5.2f} (p={r['meme_pval']:.4f}) | {status}", flush=True)
        print("=" * 90, flush=True)

    df_out.to_csv(args.output, index=False)
    print(f"\n[✓] Saved selection predictions to '{args.output}'", flush=True)


if __name__ == '__main__':
    main()
