#!/usr/bin/env python3
"""
extract_and_train_busted.py: 
Stage 1: Incremental 1-Pass Feature Extraction via Frozen PhyloAxialTransformer Backbone (Batch size 1, MPS/GPU).
Stage 2: Full 5,000-Gene Multi-Task Cross-Attention Head Optimization & Evaluation.
"""

import os
import sys
import glob
import json
import time
import math
import argparse
import random
import re
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. GENETIC CODE & VOCABULARY SETUP
# ==============================================================================

GENETIC_CODE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*', 'TGA':'*',
    'TGC':'C', 'TGT':'C', 'TGG':'W',
}

CODONS_LIST = [a+b+c for a in "TCAG" for b in "TCAG" for c in "TCAG"]
CODON_TO_IDX = {c: i for i, c in enumerate(CODONS_LIST)}
CODON_TO_IDX['-'] = 64
CODON_TO_IDX['?'] = 65

AA_LIST = "ACDEFGHIKLMNPQRSTVWY*-?"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

def codon_to_token(codon_str):
    codon_str = codon_str.upper()
    if '-' in codon_str: return 64, 21
    if len(codon_str) != 3 or any(b not in "ACGTU" for b in codon_str): return 65, 22
    codon_str = codon_str.replace('U', 'T')
    c_idx = CODON_TO_IDX.get(codon_str, 65)
    aa = GENETIC_CODE.get(codon_str, '?')
    a_idx = AA_TO_IDX.get(aa, 22)
    return c_idx, a_idx

class TreeNode:
    def __init__(self, name=None, length=0.0):
        self.name = name
        self.length = length
        self.children = []
        self.parent = None

def parse_newick(newick_str):
    newick_str = re.sub(r'\{[^}]*\}', '', newick_str.strip())
    newick_str = re.sub(r'\[.*?\]', '', newick_str)
    tokens = []
    i = 0
    while i < len(newick_str):
        c = newick_str[i]
        if c in '(),;':
            tokens.append(c)
            i += 1
        elif c == ':':
            i += 1
            start = i
            while i < len(newick_str) and newick_str[i] not in '(),;':
                i += 1
            tokens.append(('length', float(newick_str[start:i])))
        else:
            start = i
            while i < len(newick_str) and newick_str[i] not in '(),;:':
                i += 1
            tokens.append(('name', newick_str[start:i].strip().strip("'\"")))
    root = TreeNode()
    current = root
    for t in tokens:
        if t == '(':
            child = TreeNode()
            child.parent = current
            current.children.append(child)
            current = child
        elif t == ',':
            current = current.parent
            child = TreeNode()
            child.parent = current
            current.children.append(child)
            current = child
        elif t == ')':
            current = current.parent
        elif isinstance(t, tuple) and t[0] == 'name':
            current.name = t[1]
        elif isinstance(t, tuple) and t[0] == 'length':
            current.length = t[1]
    return root

def get_leaves(node):
    if not node.children: return [node]
    leaves = []
    for c in node.children: leaves.extend(get_leaves(c))
    return leaves

def get_path_to_root(node):
    path = []
    curr = node
    while curr is not None:
        path.append(curr)
        curr = curr.parent
    return path

def compute_tree_matrices(root, ordered_species):
    leaves = {leaf.name: leaf for leaf in get_leaves(root) if leaf.name}
    leaf_paths = {sp: get_path_to_root(leaves[sp]) for sp in ordered_species if sp in leaves}
    leaf_len_to_root = {sp: sum(n.length for n in path[:-1]) for sp, path in leaf_paths.items()}
    M = len(ordered_species)
    D = np.zeros((M, M), dtype=np.float32)
    
    for i in range(M):
        sp1 = ordered_species[i]
        if sp1 not in leaf_paths: continue
        p1 = leaf_paths[sp1]
        set1 = set(p1)
        for j in range(i + 1, M):
            sp2 = ordered_species[j]
            if sp2 not in leaf_paths: continue
            p2 = leaf_paths[sp2]
            lca = next((n for n in p2 if n in set1), None)
            lca_dist = sum(n.length for n in get_path_to_root(lca)[:-1]) if lca else 0.0
            dist = leaf_len_to_root[sp1] + leaf_len_to_root[sp2] - 2 * lca_dist
            D[i, j] = dist
            D[j, i] = dist
            
    try:
        H = np.eye(M) - np.ones((M, M)) / M
        B = -0.5 * H.dot(D ** 2).dot(H)
        eigvals, eigvecs = np.linalg.eigh(B)
        idx = np.argsort(eigvals)[::-1]
        top_eigvals = np.maximum(0, eigvals[idx[:4]])
        top_eigvecs = eigvecs[:, idx[:4]]
        mds_coords = top_eigvecs * np.sqrt(top_eigvals)
        if mds_coords.shape[1] < 4:
            pad = np.zeros((M, 4 - mds_coords.shape[1]), dtype=np.float32)
            mds_coords = np.hstack([mds_coords, pad])
    except Exception:
        mds_coords = np.zeros((M, 4), dtype=np.float32)
        
    return D, mds_coords.astype(np.float32)

def parse_fasta(filepath):
    sequences = {}
    current_name = None
    current_seq = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                if current_name: sequences[current_name] = "".join(current_seq)
                current_name = line[1:].split()[0].strip("'\"")
                current_seq = []
            else: current_seq.append(line)
        if current_name: sequences[current_name] = "".join(current_seq)
    return sequences

# ==============================================================================
# 2. ARCHITECTURE
# ==============================================================================

class StableAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        B, Sq, _ = q.shape
        _, Sk, _ = k.shape
        qh = self.q_proj(q).view(B, Sq, self.num_heads, self.head_dim).transpose(1, 2)
        kh = self.k_proj(k).view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)
        vh = self.v_proj(v).view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None: scores = scores.masked_fill(mask, -1e9)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, vh).transpose(1, 2).contiguous().reshape(B, Sq, self.embed_dim)
        return self.out_proj(out)

class PhyloRowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.phylo_scale = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, dist_matrix, padding_mask=None):
        B, M, _ = x.shape
        q = self.q_proj(x).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias = dist_matrix.unsqueeze(1)
        scores = scores - torch.exp(self.phylo_scale.clamp(max=10.0)) * bias
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e9)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().reshape(B, M, self.embed_dim)
        return self.out_proj(out)

class StableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = StableAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, src):
        attn_out = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout1(attn_out))
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff_out))
        return src

class PhyloAxialTransformer(nn.Module):
    def __init__(self, num_tokens=66, embed_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.codon_embedding = nn.Embedding(num_tokens, embed_dim // 2)
        self.aa_embedding = nn.Embedding(23, embed_dim // 2)
        self.mds_proj = nn.Linear(4, embed_dim)
        self.col_layers = nn.ModuleList([
            StableTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=2*embed_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.row_layers = nn.ModuleList([
            PhyloRowAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.row_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
    def forward(self, codons, aa, dist_matrix, mds_coords, species_mask=None):
        B, M, L = codons.shape
        x = torch.cat([self.codon_embedding(codons), self.aa_embedding(aa)], dim=-1)
        x = x + self.mds_proj(mds_coords).unsqueeze(2)
        for col_layer, row_layer, row_norm in zip(self.col_layers, self.row_layers, self.row_norms):
            x_col = col_layer(x.reshape(B * M, L, self.embed_dim))
            x = x_col.reshape(B, M, L, self.embed_dim)
            x_row = x.permute(0, 2, 1, 3).contiguous().reshape(B * L, M, self.embed_dim)
            dist_rep = dist_matrix.repeat_interleave(L, dim=0)
            mask_rep = species_mask.repeat_interleave(L, dim=0) if species_mask is not None else None
            x_row = row_layer(x_row, dist_rep, padding_mask=mask_rep)
            x = row_norm(x_row).reshape(B, L, M, self.embed_dim).permute(0, 2, 1, 3).contiguous()
        if species_mask is not None:
            valid_weights = (~species_mask).float().unsqueeze(-1).unsqueeze(-1)
            site_reps = (x * valid_weights).sum(dim=1) / valid_weights.sum(dim=1).clamp(min=1.0)
        else:
            site_reps = x.mean(dim=1)
        return site_reps

class BustedMultiTaskHead(nn.Module):
    def __init__(self, embed_dim=256, num_queries=4, num_heads=4, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        self.pool_attn = StableAttention(embed_dim, num_heads=num_heads, dropout=dropout)
        self.mlp_shared = nn.Sequential(
            nn.Linear(num_queries * embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.head_sqrt_lrt = nn.Linear(hidden_dim, 1)
        self.head_logp = nn.Linear(hidden_dim, 1)
        self.head_cls = nn.Linear(hidden_dim, 1)
        self.head_glob_logp = nn.Linear(hidden_dim, 1)
        self.head_syn_var = nn.Linear(hidden_dim, 1)
        self.head_omega = nn.Linear(hidden_dim, 3)
        self.head_prop = nn.Linear(hidden_dim, 3)

    def forward(self, site_reps):
        B, L, d = site_reps.shape
        q = self.queries.repeat(B, 1, 1)
        pooled = self.pool_attn(q, site_reps, site_reps)
        feat = self.mlp_shared(pooled.reshape(B, -1))
        return {
            "sqrt_lrt": F.relu(self.head_sqrt_lrt(feat)).squeeze(-1),
            "log10_p": self.head_logp(feat).squeeze(-1),
            "cls_logit": self.head_cls(feat).squeeze(-1),
            "glob_log10_p": self.head_glob_logp(feat).squeeze(-1),
            "syn_var": F.relu(self.head_syn_var(feat)).squeeze(-1),
            "omega_vals": torch.exp(self.head_omega(feat)),
            "omega_weights": torch.softmax(self.head_prop(feat), dim=-1)
        }

# ==============================================================================
# 3. EXTRACTION & TRAINING
# ==============================================================================

def extract_features(msa_dir, json_dir, output_cache_path, max_samples=5000, device="mps"):
    existing_data = []
    processed_genes = set()
    if os.path.exists(output_cache_path):
        print(f"[*] Found existing cache at {output_cache_path}, loading...")
        existing_data = torch.load(output_cache_path)
        processed_genes = {item["gene"] for item in existing_data}
        print(f"[*] Already cached: {len(existing_data)} genes.")
        
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))[:max_samples]
    to_process = [j for j in json_files if os.path.basename(j).split('.')[0] not in processed_genes]
    print(f"[*] Extracting remaining {len(to_process)} genes (out of {max_samples} total) on {device}...")
    
    if len(to_process) == 0:
        return existing_data
        
    backbone = PhyloAxialTransformer().to(device)
    backbone.eval()
    start_t = time.time()
    
    with torch.no_grad():
        for idx, j_path in enumerate(to_process):
            base_name = os.path.basename(j_path).split('.')[0]
            msa_path = os.path.join(msa_dir, f"{base_name}.fa")
            if not os.path.exists(msa_path):
                matches = glob.glob(os.path.join(msa_dir, f"{base_name}*"))
                if not matches: continue
                msa_path = matches[0]
            try:
                with open(j_path, 'r') as f: data = json.load(f)
                newick = None
                if "input" in data and "trees" in data["input"]: newick = list(data["input"]["trees"].values())[0]
                elif "trees" in data: newick = list(data["trees"].values())[0] if isinstance(data["trees"], dict) else data["trees"][0]
                if not newick: continue
                
                test_res = data.get("Comparative selection test results", {}) or data.get("test results", {})
                lrt = float(test_res.get("LRT", 0.0))
                p_val = max(1e-15, min(1.0, float(test_res.get("p-value", 1.0))))
                log10_p = float(np.log10(p_val))
                is_sig = 1.0 if p_val < 0.05 else 0.0
                
                glob_test = data.get("test results", {})
                glob_lrt = float(glob_test.get("LRT", 0.0))
                glob_p = max(1e-15, min(1.0, float(glob_test.get("p-value", 1.0))))
                glob_log10_p = float(np.log10(glob_p))
                
                fits = data.get("fits", {})
                same_dist = fits.get("Same distributions model", {}) or fits.get("Unconstrained model", {})
                rate_dists = same_dist.get("Rate Distributions", {})
                
                omega_dist = rate_dists.get("Test", {}) or rate_dists.get("non-synonymous/synonymous rate ratio", {})
                omega_vals = [1.0, 1.0, 1.0]; omega_weights = [0.33, 0.33, 0.34]
                if omega_dist and isinstance(omega_dist, dict):
                    items = sorted([v for k, v in omega_dist.items() if isinstance(v, dict)], key=lambda x: x.get("omega", 1.0))
                    if len(items) >= 3:
                        omega_vals = [float(it.get("omega", 1.0)) for it in items[:3]]
                        omega_weights = [float(it.get("proportion", 0.33)) for it in items[:3]]
                        
                syn_dist = rate_dists.get("Synonymous site-to-site rates", {})
                syn_var = 0.0
                if syn_dist and isinstance(syn_dist, dict):
                    syn_rates = [float(v.get("rate", 1.0)) for v in syn_dist.values() if isinstance(v, dict)]
                    syn_props = [float(v.get("proportion", 0.33)) for v in syn_dist.values() if isinstance(v, dict)]
                    if syn_rates and syn_props:
                        mean_syn = sum(r * p for r, p in zip(syn_rates, syn_props))
                        syn_var = sum(p * ((r - mean_syn)**2) for r, p in zip(syn_rates, syn_props))
                        
                seqs = parse_fasta(msa_path)
                root = parse_newick(newick)
                species = list(seqs.keys())[:120]
                M = len(species)
                raw_len = min(len(seqs[sp]) for sp in species)
                L = min(raw_len // 3, 1000)
                if L < 30 or M < 10: continue
                
                codon_arr = np.zeros((1, M, L), dtype=np.int64)
                aa_arr = np.zeros((1, M, L), dtype=np.int64)
                for i, sp in enumerate(species):
                    seq = seqs[sp]
                    for s in range(L):
                        c, a = codon_to_token(seq[s*3:(s+1)*3])
                        codon_arr[0, i, s] = c
                        aa_arr[0, i, s] = a
                        
                dist_mat, mds_coords = compute_tree_matrices(root, species)
                t_codons = torch.tensor(codon_arr, dtype=torch.long, device=device)
                t_aa = torch.tensor(aa_arr, dtype=torch.long, device=device)
                t_dist = torch.tensor(dist_mat, dtype=torch.float32, device=device).unsqueeze(0)
                t_mds = torch.tensor(mds_coords, dtype=torch.float32, device=device).unsqueeze(0)
                
                site_reps = backbone(t_codons, t_aa, t_dist, t_mds).squeeze(0).cpu()
                
                existing_data.append({
                    "gene": base_name,
                    "site_reps": site_reps,
                    "sqrt_lrt": float(np.sqrt(max(0.0, lrt))),
                    "log10_p": log10_p,
                    "is_sig": is_sig,
                    "glob_log10_p": glob_log10_p,
                    "omega_vals": omega_vals,
                    "omega_weights": omega_weights,
                    "syn_var": syn_var
                })
                
                if (idx + 1) % 250 == 0 or (idx + 1) == len(to_process):
                    elapsed = time.time() - start_t
                    rate = (idx + 1) / elapsed
                    print(f"  Extracted {idx + 1}/{len(to_process)} ({len(existing_data)} total cached, {rate:.1f} genes/s)...")
                    torch.save(existing_data, output_cache_path)
            except Exception:
                continue
                
    torch.save(existing_data, output_cache_path)
    print(f"[+] Total cached: {len(existing_data)} genes in {output_cache_path} ({os.path.getsize(output_cache_path)/1e6:.1f} MB)")
    return existing_data

class CachedFeatureDataset(Dataset):
    def __init__(self, data_list): self.data = data_list
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_cached_batch(batch):
    max_L = max(item["site_reps"].shape[0] for item in batch)
    B = len(batch)
    d = batch[0]["site_reps"].shape[1]
    padded_reps = torch.zeros((B, max_L, d), dtype=torch.float32)
    for b, item in enumerate(batch):
        l = item["site_reps"].shape[0]
        padded_reps[b, :l] = item["site_reps"]
    return {
        "site_reps": padded_reps,
        "sqrt_lrt": torch.tensor([item["sqrt_lrt"] for item in batch], dtype=torch.float32),
        "log10_p": torch.tensor([item["log10_p"] for item in batch], dtype=torch.float32),
        "is_sig": torch.tensor([item["is_sig"] for item in batch], dtype=torch.float32),
        "glob_log10_p": torch.tensor([item["glob_log10_p"] for item in batch], dtype=torch.float32),
        "omega_vals": torch.tensor([item["omega_vals"] for item in batch], dtype=torch.float32),
        "omega_weights": torch.tensor([item["omega_weights"] for item in batch], dtype=torch.float32),
        "syn_var": torch.tensor([item["syn_var"] for item in batch], dtype=torch.float32)
    }

def train_head_stage(cached_data, epochs=40, batch_size=64, lr=1e-3, device="mps", output_dir="./output"):
    print(f"\n[*] STAGE 2: Finetuning BUSTED Multi-Task Pooling Head on {len(cached_data)} Genes ({epochs} epochs, batch size {batch_size})...")
    os.makedirs(output_dir, exist_ok=True)
    
    n_val = max(50, int(0.15 * len(cached_data)))
    n_train = len(cached_data) - n_val
    train_data, val_data = torch.utils.data.random_split(cached_data, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(CachedFeatureDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_cached_batch)
    val_loader = DataLoader(CachedFeatureDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=collate_cached_batch)
    
    head = BustedMultiTaskHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"\n{'Epoch':6s} | {'Train Loss':12s} | {'Val Spearman r':16s} | {'Val Pearson r':14s} | {'Val Acc':10s}")
    print("-" * 65)
    
    best_r = -1.0
    for ep in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            reps = batch["site_reps"].to(device)
            preds = head(reps)
            
            loss_cls = F.binary_cross_entropy_with_logits(preds["cls_logit"], batch["is_sig"].to(device))
            loss_logp = F.smooth_l1_loss(preds["log10_p"], batch["log10_p"].to(device))
            loss_lrt = F.smooth_l1_loss(preds["sqrt_lrt"], batch["sqrt_lrt"].to(device))
            loss_glob = F.smooth_l1_loss(preds["glob_log10_p"], batch["glob_log10_p"].to(device))
            loss_syn = F.smooth_l1_loss(preds["syn_var"], batch["syn_var"].to(device))
            loss_omega = F.smooth_l1_loss(torch.log(preds["omega_vals"] + 1e-4), torch.log(batch["omega_vals"].to(device) + 1e-4))
            loss_weights = F.kl_div(torch.log(preds["omega_weights"] + 1e-6), batch["omega_weights"].to(device), reduction='batchmean')
            
            loss = (2.0 * loss_cls + 1.0 * loss_logp + 0.5 * loss_lrt + 0.5 * loss_glob + 0.2 * loss_syn + 0.2 * loss_omega + 0.5 * loss_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        head.eval()
        all_pred_p, all_true_p = [], []
        all_pred_sig, all_true_sig = [], []
        with torch.no_grad():
            for batch in val_loader:
                preds = head(batch["site_reps"].to(device))
                all_pred_p.extend(preds["log10_p"].cpu().numpy())
                all_true_p.extend(batch["log10_p"].numpy())
                all_pred_sig.extend(torch.sigmoid(preds["cls_logit"]).cpu().numpy())
                all_true_sig.extend(batch["is_sig"].numpy())
                
        r_sp, _ = stats.spearmanr(all_pred_p, all_true_p)
        r_pe, _ = stats.pearsonr(all_pred_p, all_true_p)
        acc = np.mean((np.array(all_pred_sig) >= 0.5) == np.array(all_true_sig))
        
        print(f"{ep:6d} | {avg_loss:12.4f} | {r_sp:16.4f} | {r_pe:14.4f} | {acc*100:9.1f}%")
        
        if r_sp > best_r:
            best_r = r_sp
            torch.save({
                "epoch": ep,
                "head_state_dict": head.state_dict(),
                "metrics": {"spearman_r": r_sp, "pearson_r": r_pe, "acc": acc}
            }, os.path.join(output_dir, "best_busted_head_5000.pt"))
            
    print(f"\n[+] 5,000-Gene BUSTED Head Training Complete! Best Validation Spearman r = {best_r:.4f}")
    print(f"[+] Saved checkpoint to {os.path.join(output_dir, 'best_busted_head_5000.pt')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msa_dir", type=str, default="/Users/sergei/HyphAeon_BUSTED/msas")
    parser.add_argument("--json_dir", type=str, default="/Users/sergei/HyphAeon_BUSTED/echo_msa_results_take3")
    parser.add_argument("--cache_file", type=str, default="/Users/sergei/HyphAeon_BUSTED/cached_features.pt")
    parser.add_argument("--output_dir", type=str, default="/Users/sergei/HyphAeon_BUSTED/output")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Executing HyphAeon 5,000-Gene BUSTED Pipeline on Device: {device}")
    
    cached_data = extract_features(args.msa_dir, args.json_dir, args.cache_file, max_samples=args.max_samples, device=device)
    train_head_stage(cached_data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
