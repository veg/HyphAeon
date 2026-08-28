#!/usr/bin/env python3
"""
train_busted_head.py: Train an Alignment-Wide BUSTED-PH / BUSTED+S Neural Readout Head
on top of the Frozen AxoMEME (PhyloAxialTransformer) Backbone.

Designed for execution on compute clusters (e.g., silverback.temple.edu) or local machines.
Directly ingests:
  1. Multiple Sequence Alignments (FASTA / NEXUS)
  2. Fitted BUSTED-PH / BUSTED+S JSON results (including trees, LRTs, p-values, omega, and synonymous rate distributions)

Author: Deep Time Foundation Genomics
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
    if '-' in codon_str:
        return 64, 21  # gap
    if len(codon_str) != 3 or any(b not in "ACGTU" for b in codon_str):
        return 65, 22  # unknown
    codon_str = codon_str.replace('U', 'T')
    c_idx = CODON_TO_IDX.get(codon_str, 65)
    aa = GENETIC_CODE.get(codon_str, '?')
    a_idx = AA_TO_IDX.get(aa, 22)
    return c_idx, a_idx

# ==============================================================================
# 2. STANDALONE NEWICK PARSER & CLASSICAL MDS PHYLOGENY ENCODER
# ==============================================================================

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
    if not node.children:
        return [node]
    leaves = []
    for c in node.children:
        leaves.extend(get_leaves(c))
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
    
    leaf_len_to_root = {}
    for sp, path in leaf_paths.items():
        leaf_len_to_root[sp] = sum(n.length for n in path[:-1])
        
    M = len(ordered_species)
    D = np.zeros((M, M), dtype=np.float32)
    
    for i in range(M):
        sp1 = ordered_species[i]
        if sp1 not in leaf_paths:
            continue
        p1 = leaf_paths[sp1]
        set1 = set(p1)
        for j in range(i + 1, M):
            sp2 = ordered_species[j]
            if sp2 not in leaf_paths:
                continue
            p2 = leaf_paths[sp2]
            lca = next((n for n in p2 if n in set1), None)
            lca_dist = sum(n.length for n in get_path_to_root(lca)[:-1]) if lca else 0.0
            dist = leaf_len_to_root[sp1] + leaf_len_to_root[sp2] - 2 * lca_dist
            D[i, j] = dist
            D[j, i] = dist
            
    # Classical MDS to 4D
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

# ==============================================================================
# 3. BUSTED-PH / BUSTED+S DATASET & TARGET PARSER
# ==============================================================================

def parse_fasta(filepath):
    sequences = {}
    current_name = None
    current_seq = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_name:
                    sequences[current_name] = "".join(current_seq)
                current_name = line[1:].split()[0].strip("'\"")
                current_seq = []
            else:
                current_seq.append(line)
        if current_name:
            sequences[current_name] = "".join(current_seq)
    return sequences

class BustedDataset(Dataset):
    def __init__(self, msa_dir, json_dir, max_samples=None, max_species=128, max_codons=1000):
        self.samples = []
        self.max_species = max_species
        self.max_codons = max_codons
        
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        if max_samples:
            json_files = json_files[:max_samples]
        print(f"[*] Scanning {len(json_files)} BUSTED-PH/BUSTED+S JSON files...")
        
        for j_path in json_files:
            base_name = os.path.basename(j_path).split('.')[0]
            msa_path = os.path.join(msa_dir, f"{base_name}.fa")
            if not os.path.exists(msa_path):
                msa_matches = glob.glob(os.path.join(msa_dir, f"{base_name}*"))
                if not msa_matches:
                    continue
                msa_path = msa_matches[0]
            
            try:
                with open(j_path, 'r') as f:
                    data = json.load(f)
                    
                # 1. Parse Tree
                newick = None
                if "input" in data and "trees" in data["input"]:
                    newick = list(data["input"]["trees"].values())[0]
                elif "trees" in data:
                    newick = list(data["trees"].values())[0] if isinstance(data["trees"], dict) else data["trees"][0]
                if not newick:
                    continue
                    
                # 2. Parse Comparative & Global Test Results
                test_res = data.get("Comparative selection test results", {}) or data.get("test results", {})
                lrt = float(test_res.get("LRT", 0.0))
                p_val = float(test_res.get("p-value", 1.0))
                p_val = max(1e-15, min(1.0, p_val))
                log10_p = float(np.log10(p_val))
                is_sig = 1.0 if p_val < 0.05 else 0.0
                
                # Global BUSTED+S test results
                glob_test = data.get("test results", {})
                glob_lrt = float(glob_test.get("LRT", 0.0))
                glob_p = float(glob_test.get("p-value", 1.0))
                glob_p = max(1e-15, min(1.0, glob_p))
                glob_log10_p = float(np.log10(glob_p))
                
                # 3. Parse Rate Distributions (omega and synonymous alpha)
                fits = data.get("fits", {})
                same_dist = fits.get("Same distributions model", {}) or fits.get("Unconstrained model", {})
                rate_dists = same_dist.get("Rate Distributions", {})
                
                # Extract omega distribution
                omega_dist = rate_dists.get("Test", {}) or rate_dists.get("non-synonymous/synonymous rate ratio", {})
                omega_vals = [1.0, 1.0, 1.0]
                omega_weights = [0.33, 0.33, 0.34]
                if omega_dist and isinstance(omega_dist, dict):
                    items = sorted([v for k, v in omega_dist.items() if isinstance(v, dict)], key=lambda x: x.get("omega", 1.0))
                    if len(items) >= 3:
                        omega_vals = [float(it.get("omega", 1.0)) for it in items[:3]]
                        omega_weights = [float(it.get("proportion", 0.33)) for it in items[:3]]
                    elif len(items) > 0:
                        omega_vals = [float(items[-1].get("omega", 1.0)), 1.0, 1.0]
                        omega_weights = [float(items[-1].get("proportion", 1.0)), 0.0, 0.0]
                
                # Extract synonymous alpha distribution (SRV)
                syn_dist = rate_dists.get("Synonymous site-to-site rates", {})
                syn_var = 0.0
                if syn_dist and isinstance(syn_dist, dict):
                    syn_rates = [float(v.get("rate", 1.0)) for v in syn_dist.values() if isinstance(v, dict)]
                    syn_props = [float(v.get("proportion", 0.33)) for v in syn_dist.values() if isinstance(v, dict)]
                    if syn_rates and syn_props:
                        mean_syn = sum(r * p for r, p in zip(syn_rates, syn_props))
                        syn_var = sum(p * ((r - mean_syn)**2) for r, p in zip(syn_rates, syn_props))
                
                self.samples.append({
                    "gene": base_name,
                    "msa_path": msa_path,
                    "newick": newick,
                    "lrt": lrt,
                    "sqrt_lrt": np.sqrt(max(0.0, lrt)),
                    "p_val": p_val,
                    "log10_p": log10_p,
                    "is_sig": is_sig,
                    "glob_lrt": glob_lrt,
                    "glob_log10_p": glob_log10_p,
                    "omega_vals": omega_vals,
                    "omega_weights": omega_weights,
                    "syn_var": syn_var
                })
            except Exception as e:
                continue
                
        print(f"[+] Successfully loaded {len(self.samples)} valid MSA / BUSTED pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        seqs = parse_fasta(item["msa_path"])
        root = parse_newick(item["newick"])
        
        species = list(seqs.keys())[:self.max_species]
        M = len(species)
        
        raw_len = min(len(seqs[sp]) for sp in species)
        L = min(raw_len // 3, self.max_codons)
        
        codon_tensor = np.zeros((M, L), dtype=np.int64)
        aa_tensor = np.zeros((M, L), dtype=np.int64)
        
        for i, sp in enumerate(species):
            seq = seqs[sp]
            for s in range(L):
                codon = seq[s*3:(s+1)*3]
                c_idx, a_idx = codon_to_token(codon)
                codon_tensor[i, s] = c_idx
                aa_tensor[i, s] = a_idx
                
        dist_mat, mds_coords = compute_tree_matrices(root, species)
        
        return {
            "codon_tensor": torch.tensor(codon_tensor, dtype=torch.long),
            "aa_tensor": torch.tensor(aa_tensor, dtype=torch.long),
            "dist_matrix": torch.tensor(dist_mat, dtype=torch.float32),
            "mds_coords": torch.tensor(mds_coords, dtype=torch.float32),
            "sqrt_lrt": torch.tensor(item["sqrt_lrt"], dtype=torch.float32),
            "log10_p": torch.tensor(item["log10_p"], dtype=torch.float32),
            "is_sig": torch.tensor(item["is_sig"], dtype=torch.float32),
            "glob_log10_p": torch.tensor(item["glob_log10_p"], dtype=torch.float32),
            "omega_vals": torch.tensor(item["omega_vals"], dtype=torch.float32),
            "omega_weights": torch.tensor(item["omega_weights"], dtype=torch.float32),
            "syn_var": torch.tensor(item["syn_var"], dtype=torch.float32)
        }

def collate_busted_batch(batch):
    max_M = max(item["codon_tensor"].shape[0] for item in batch)
    max_L = max(item["codon_tensor"].shape[1] for item in batch)
    B = len(batch)
    
    padded_codons = torch.full((B, max_M, max_L), 64, dtype=torch.long)
    padded_aa = torch.full((B, max_M, max_L), 21, dtype=torch.long)
    padded_dist = torch.zeros((B, max_M, max_M), dtype=torch.float32)
    padded_mds = torch.zeros((B, max_M, 4), dtype=torch.float32)
    species_mask = torch.ones((B, max_M), dtype=torch.bool)
    
    for b, item in enumerate(batch):
        m, l = item["codon_tensor"].shape
        padded_codons[b, :m, :l] = item["codon_tensor"]
        padded_aa[b, :m, :l] = item["aa_tensor"]
        padded_dist[b, :m, :m] = item["dist_matrix"]
        padded_mds[b, :m, :] = item["mds_coords"]
        species_mask[b, :m] = False
        
    return {
        "codon_tensor": padded_codons,
        "aa_tensor": padded_aa,
        "dist_matrix": padded_dist,
        "mds_coords": padded_mds,
        "species_mask": species_mask,
        "sqrt_lrt": torch.stack([item["sqrt_lrt"] for item in batch]),
        "log10_p": torch.stack([item["log10_p"] for item in batch]),
        "is_sig": torch.stack([item["is_sig"] for item in batch]),
        "glob_log10_p": torch.stack([item["glob_log10_p"] for item in batch]),
        "omega_vals": torch.stack([item["omega_vals"] for item in batch]),
        "omega_weights": torch.stack([item["omega_weights"] for item in batch]),
        "syn_var": torch.stack([item["syn_var"] for item in batch])
    }

# ==============================================================================
# 4. NEURAL ARCHITECTURE: AXOMEME BACKBONE + BUSTED HEAD
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
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
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
        x_codon = self.codon_embedding(codons)
        x_aa = self.aa_embedding(aa)
        x = torch.cat([x_codon, x_aa], dim=-1) # (B, M, L, d)
        
        phylo_emb = self.mds_proj(mds_coords).unsqueeze(2) # (B, M, 1, d)
        x = x + phylo_emb
        
        for col_layer, row_layer, row_norm in zip(self.col_layers, self.row_layers, self.row_norms):
            x_col = x.reshape(B * M, L, self.embed_dim)
            x_col = col_layer(x_col)
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
            
        return site_reps # (B, L, d)

class BustedMultiTaskHead(nn.Module):
    """
    Multi-Query Cross-Attention Pooling & Prediction Head for BUSTED-PH / BUSTED+S.
    """
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
        
        # 1. Comparative Selection Heads
        self.head_sqrt_lrt = nn.Linear(hidden_dim, 1)
        self.head_logp = nn.Linear(hidden_dim, 1)
        self.head_cls = nn.Linear(hidden_dim, 1) # logit for p < 0.05
        
        # 2. Global Selection & SRV Heads
        self.head_glob_logp = nn.Linear(hidden_dim, 1)
        self.head_syn_var = nn.Linear(hidden_dim, 1)
        
        # 3. Rate Mixture Distribution Heads
        self.head_omega = nn.Linear(hidden_dim, 3) # log(omega_1, omega_2, omega_3)
        self.head_prop = nn.Linear(hidden_dim, 3)  # mixture proportions (p1, p2, p3)

    def forward(self, site_reps):
        B, L, d = site_reps.shape
        q = self.queries.repeat(B, 1, 1)
        pooled = self.pool_attn(q, site_reps, site_reps)
        pooled_flat = pooled.reshape(B, -1)
        feat = self.mlp_shared(pooled_flat)
        
        pred_sqrt_lrt = F.relu(self.head_sqrt_lrt(feat)).squeeze(-1)
        pred_logp = self.head_logp(feat).squeeze(-1)
        pred_cls_logit = self.head_cls(feat).squeeze(-1)
        pred_glob_logp = self.head_glob_logp(feat).squeeze(-1)
        pred_syn_var = F.relu(self.head_syn_var(feat)).squeeze(-1)
        
        pred_omega = torch.exp(self.head_omega(feat))
        pred_prop = torch.softmax(self.head_prop(feat), dim=-1)
        
        return {
            "sqrt_lrt": pred_sqrt_lrt,
            "log10_p": pred_logp,
            "cls_logit": pred_cls_logit,
            "glob_log10_p": pred_glob_logp,
            "syn_var": pred_syn_var,
            "omega_vals": pred_omega,
            "omega_weights": pred_prop
        }

class FullAxoBustedModel(nn.Module):
    def __init__(self, backbone, head, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone
        self.head = head
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
    def forward(self, codons, aa, dist_matrix, mds_coords, species_mask=None):
        site_reps = self.backbone(codons, aa, dist_matrix, mds_coords, species_mask)
        return self.head(site_reps)

# ==============================================================================
# 5. TRAINING & EVALUATION LOOP
# ==============================================================================

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        optimizer.zero_grad()
        codons = batch["codon_tensor"].to(device)
        aa = batch["aa_tensor"].to(device)
        dist_mat = batch["dist_matrix"].to(device)
        mds = batch["mds_coords"].to(device)
        mask = batch["species_mask"].to(device)
        
        target_sqrt_lrt = batch["sqrt_lrt"].to(device)
        target_logp = batch["log10_p"].to(device)
        target_is_sig = batch["is_sig"].to(device)
        target_glob_logp = batch["glob_log10_p"].to(device)
        target_omega = batch["omega_vals"].to(device)
        target_weights = batch["omega_weights"].to(device)
        target_syn_var = batch["syn_var"].to(device)
        
        preds = model(codons, aa, dist_mat, mds, mask)
        
        loss_cls = F.binary_cross_entropy_with_logits(preds["cls_logit"], target_is_sig)
        loss_logp = F.smooth_l1_loss(preds["log10_p"], target_logp)
        loss_lrt = F.smooth_l1_loss(preds["sqrt_lrt"], target_sqrt_lrt)
        loss_glob = F.smooth_l1_loss(preds["glob_log10_p"], target_glob_logp)
        loss_syn = F.smooth_l1_loss(preds["syn_var"], target_syn_var)
        loss_omega = F.smooth_l1_loss(torch.log(preds["omega_vals"] + 1e-4), torch.log(target_omega + 1e-4))
        loss_weights = F.kl_div(torch.log(preds["omega_weights"] + 1e-6), target_weights, reduction='batchmean')
        
        loss = (2.0 * loss_cls + 1.0 * loss_logp + 0.5 * loss_lrt + 
                0.5 * loss_glob + 0.2 * loss_syn + 0.2 * loss_omega + 0.5 * loss_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_pred_logp, all_true_logp = [], []
    all_pred_sig, all_true_sig = [], []
    all_pred_lrt, all_true_lrt = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            codons = batch["codon_tensor"].to(device)
            aa = batch["aa_tensor"].to(device)
            dist_mat = batch["dist_matrix"].to(device)
            mds = batch["mds_coords"].to(device)
            mask = batch["species_mask"].to(device)
            
            preds = model(codons, aa, dist_mat, mds, mask)
            
            all_pred_logp.extend(preds["log10_p"].cpu().numpy())
            all_true_logp.extend(batch["log10_p"].numpy())
            all_pred_sig.extend(torch.sigmoid(preds["cls_logit"]).cpu().numpy())
            all_true_sig.extend(batch["is_sig"].numpy())
            all_pred_lrt.extend((preds["sqrt_lrt"]**2).cpu().numpy())
            all_true_lrt.extend((batch["sqrt_lrt"]**2).numpy())
            
    r_spearman, _ = stats.spearmanr(all_pred_logp, all_true_logp)
    r_pearson, _ = stats.pearsonr(all_pred_logp, all_true_logp)
    acc = np.mean((np.array(all_pred_sig) >= 0.5) == np.array(all_true_sig))
    
    return {
        "spearman_r": r_spearman,
        "pearson_r": r_pearson,
        "cls_acc": acc,
        "true_logp": np.array(all_true_logp),
        "pred_logp": np.array(all_pred_logp)
    }

# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train BUSTED-PH / BUSTED+S Readout Head on AxoMEME Backbone")
    parser.add_argument("--msa_dir", type=str, default="/home/sergei/Projects/BUSTED-PH/mammalian120/msas")
    parser.add_argument("--json_dir", type=str, default="/home/sergei/Projects/BUSTED-PH/mammalian120/echo_msa_results_take3")
    parser.add_argument("--checkpoint", type=str, default=None, help="Pretrained AxoMEME checkpoint (.pt)")
    parser.add_argument("--output_dir", type=str, default="./busted_head_output")
    parser.add_argument("--max_samples", type=int, default=1000, help="Number of gene samples for fast training run")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Executing on device: {device}")

    dataset = BustedDataset(args.msa_dir, args.json_dir, max_samples=args.max_samples)
    if len(dataset) == 0:
        print("[!] Error: No valid samples found. Check directory paths.")
        sys.exit(1)
        
    n_val = max(10, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_busted_batch)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_busted_batch)
    
    backbone = PhyloAxialTransformer(embed_dim=args.embed_dim)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[*] Loading pretrained backbone weights from {args.checkpoint}...")
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        backbone.load_state_dict(state_dict, strict=False)
        
    head = BustedMultiTaskHead(embed_dim=args.embed_dim)
    model = FullAxoBustedModel(backbone, head, freeze_backbone=args.freeze_backbone).to(device)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"\n[*] Commencing Training of BUSTED-PH Readout Head ({args.epochs} epochs across {len(dataset)} genes)...")
    print(f"{'Epoch':6s} | {'Train Loss':12s} | {'Val Spearman r':16s} | {'Val Pearson r':14s} | {'Val Acc':10s}")
    print("-" * 65)
    
    best_r = -1.0
    for ep in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        
        r_sp = val_metrics["spearman_r"]
        r_pe = val_metrics["pearson_r"]
        acc = val_metrics["cls_acc"]
        
        print(f"{ep:6d} | {train_loss:12.4f} | {r_sp:16.4f} | {r_pe:14.4f} | {acc*100:9.1f}%")
        
        if r_sp > best_r:
            best_r = r_sp
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "head_state_dict": head.state_dict(),
                "metrics": val_metrics,
                "args": args
            }, os.path.join(args.output_dir, "best_busted_head.pt"))
            
    print(f"\n[+] Training Complete! Best Validation Spearman r = {best_r:.4f}")
    print(f"[+] Saved best checkpoint to {os.path.join(args.output_dir, 'best_busted_head.pt')}")

if __name__ == "__main__":
    main()
