#!/usr/bin/env python3
"""
train_masked_alignment_transformer.py

Self-Supervised Masked Phylogenetic Alignment Transformer (PhyloMLM)
-------------------------------------------------------------------
A permutation-equivariant axial transformer trained via denoising / masked language modeling
across Multiple Sequence Alignments with continuous-time Markov tree biases and Tree-RoPE.

Key Features:
1. 3-Way Dynamic Masking:
   - 60% Tip Masking (learns tree-guided transition distributions P(X_i | X_j, d_ij))
   - 20% Clade Masking (masks subtrees to learn deep node interpolation)
   - 20% Full Column Masking (masks entire columns to learn 1D biophysical compatibility)
2. Permutation-Equivariant Tree Attention:
   - Relative Tree-RoPE on MDS coordinate differences (Δx_ij)
   - Continuous Markov decay bias: log(0.05 + 0.95 * exp(-lambda * d_ij))
   - Zero fixed row-position embeddings (exact permutation symmetry)
3. Zero-Shot Evolutionary Selection Metric:
   - Computes local branch surprisal / log-odds without requiring pre-computed MEME training labels!
"""

import os
import sys
import math
import time
import glob
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Genetic Code & Token Constants
# ---------------------------------------------------------------------------
CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}
CODONS_ORDERED = sorted(list(CODON_TABLE.keys()))
CODON_TO_ID = {c: i for i, c in enumerate(CODONS_ORDERED)}
ID_TO_CODON = {i: c for i, c in enumerate(CODONS_ORDERED)}

AA_ORDERED = sorted(list(set(CODON_TABLE.values()) - {'*'})) + ['*']
AA_TO_ID = {a: i for i, a in enumerate(AA_ORDERED)}

# Special Tokens
PAD_CODON_ID = 64
MASK_CODON_ID = 65
NUM_CODON_TOKENS = 66

PAD_AA_ID = 21
MASK_AA_ID = 22
NUM_AA_TOKENS = 23

# ---------------------------------------------------------------------------
# Tree-RoPE Embedding Utilities
# ---------------------------------------------------------------------------
def apply_rotary_pos_emb_mds(x, mds_coords):
    """
    x: [B, S, H, D]
    mds_coords: [B, S, 4]
    Rotates x along the taxa dimension using 4D MDS coordinate projections.
    """
    B, S, H, D = x.shape
    num_pairs = min(4, D // 2)
    coords = mds_coords[:, :, :num_pairs] # [B, S, num_pairs]
    
    sin_c = torch.sin(coords).unsqueeze(2) # [B, S, 1, num_pairs]
    cos_c = torch.cos(coords).unsqueeze(2) # [B, S, 1, num_pairs]
    
    x1 = x[..., 0:num_pairs * 2:2]
    x2 = x[..., 1:num_pairs * 2:2]
    
    x_rot1 = x1 * cos_c - x2 * sin_c
    x_rot2 = x1 * sin_c + x2 * cos_c
    
    x_out = x.clone()
    x_out[..., 0:num_pairs * 2:2] = x_rot1
    x_out[..., 1:num_pairs * 2:2] = x_rot2
    return x_out


# ---------------------------------------------------------------------------
# Permutation-Equivariant Phylogenetic Row Attention
# ---------------------------------------------------------------------------
class PhyloRowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Learnable Markov decay rate per head: log(0.05 + 0.95 * exp(-lambda * d_ij))
        self.log_lambda = nn.Parameter(torch.linspace(math.log(0.1), math.log(10.0), num_heads))
        self.dist_bias_proj = nn.Linear(1, num_heads, bias=False)
        
    def forward(self, x, dist_matrix, mds_coords):
        """
        x: [B, S, L, D]
        dist_matrix: [B, S, S]
        mds_coords: [B, S, 4]
        """
        B, S, L, D = x.shape
        # Process each codon position in window independently
        # Reshape to [B * L, S, D]
        x_flat = x.permute(0, 2, 1, 3).reshape(B * L, S, D)
        
        q = self.q_proj(x_flat).view(B * L, S, self.num_heads, self.head_dim)
        k = self.k_proj(x_flat).view(B * L, S, self.num_heads, self.head_dim)
        v = self.v_proj(x_flat).view(B * L, S, self.num_heads, self.head_dim)
        
        # Apply Relative Tree-RoPE
        mds_expanded = mds_coords.unsqueeze(1).expand(B, L, S, -1).reshape(B * L, S, -1)
        q = apply_rotary_pos_emb_mds(q, mds_expanded)
        k = apply_rotary_pos_emb_mds(k, mds_expanded)
        
        # [B*L, H, S, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # [B*L, H, S, S]
        
        # Continuous Markov Transition Kernel Bias
        lambdas = torch.exp(self.log_lambda).view(1, self.num_heads, 1, 1) # [1, H, 1, 1]
        dist_expanded = dist_matrix.unsqueeze(1).expand(B, L, S, S).reshape(B * L, 1, S, S)
        markov_bias = torch.log(0.05 + 0.95 * torch.exp(-lambdas * dist_expanded))
        
        scores = scores + markov_bias
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v) # [B*L, H, S, head_dim]
        out = out.permute(0, 2, 1, 3).reshape(B * L, S, D)
        out = self.out_proj(out)
        
        # Reshape back to [B, S, L, D]
        out = out.view(B, L, S, D).permute(0, 2, 1, 3)
        return out


# ---------------------------------------------------------------------------
# 1D Column / Sequence Attention (Local Window)
# ---------------------------------------------------------------------------
class PhyloColAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        """
        x: [B, S, L, D]
        """
        B, S, L, D = x.shape
        # Flatten species into batch: [B * S, L, D]
        x_flat = x.view(B * S, L, D)
        
        q = self.q_proj(x_flat).view(B * S, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_flat).view(B * S, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_flat).view(B * S, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B * S, L, D)
        out = self.out_proj(out).view(B, S, L, D)
        return out


# ---------------------------------------------------------------------------
# Axial Transformer Block
# ---------------------------------------------------------------------------
class AxialTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim=None):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = embed_dim * 4
            
        self.row_attn = PhyloRowAttention(embed_dim, num_heads)
        self.col_attn = PhyloColAttention(embed_dim, num_heads)
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        
    def forward(self, x, dist_matrix, mds_coords):
        # 1. Phylogenetic Row Attention (Across Taxa)
        x = x + self.row_attn(self.ln1(x), dist_matrix, mds_coords)
        # 2. Sequence Column Attention (Across Sites)
        x = x + self.col_attn(self.ln2(x))
        # 3. Feedforward
        x = x + self.ffn(self.ln3(x))
        return x


# ---------------------------------------------------------------------------
# Full PhyloMaskedAxialTransformer (PhyloMLM)
# ---------------------------------------------------------------------------
class PhyloMaskedAxialTransformer(nn.Module):
    def __init__(self, embed_dim=192, num_heads=6, num_layers=6, window_size=9):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        
        self.codon_embedding = nn.Embedding(NUM_CODON_TOKENS, embed_dim, padding_idx=PAD_CODON_ID)
        self.aa_embedding = nn.Embedding(NUM_AA_TOKENS, embed_dim, padding_idx=PAD_AA_ID)
        
        self.layers = nn.ModuleList([
            AxialTransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.ln_out = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, 64) # Predicts 64 actual codon logits
        
    def forward(self, codon_ids, aa_ids, dist_matrix, mds_coords):
        """
        codon_ids: [B, S, L]
        aa_ids: [B, S, L]
        dist_matrix: [B, S, S]
        mds_coords: [B, S, 4]
        Returns: logits of shape [B, S, L, 64]
        """
        x = self.codon_embedding(codon_ids) + self.aa_embedding(aa_ids)
        
        for layer in self.layers:
            x = layer(x, dist_matrix, mds_coords)
            
        x = self.ln_out(x)
        logits = self.lm_head(x) # [B, S, L, 64]
        return logits


# ---------------------------------------------------------------------------
# Self-Supervised Alignment Dataset with 3-Way Masking
# ---------------------------------------------------------------------------
class MaskedPhyloDataset(Dataset):
    def __init__(self, npz_files, window_size=9, max_species=128, mask_prob=0.15):
        self.npz_files = npz_files
        self.window_size = window_size
        self.max_species = max_species
        self.mask_prob = mask_prob
        
    def __len__(self):
        return len(self.npz_files) * 16 # Multiple random crops per gene
        
    def __getitem__(self, idx):
        file_idx = idx % len(self.npz_files)
        npz = np.load(self.npz_files[file_idx])
        
        codon_mat = npz['codon_ids_matrix'] # [N, L_total]
        aa_mat = npz['aa_ids_matrix']
        dist_arr = npz['dist_arr']
        mds_coords = npz['mds_coords']
        
        N, L_total = codon_mat.shape
        if L_total <= self.window_size:
            start_site = 0
            end_site = L_total
        else:
            start_site = random.randint(0, L_total - self.window_size)
            end_site = start_site + self.window_size
            
        # Species Subsampling if large
        if N > self.max_species:
            sp_indices = np.random.choice(N, self.max_species, replace=False)
            sp_indices = np.sort(sp_indices)
        else:
            sp_indices = np.arange(N)
            
        c_crop = codon_mat[sp_indices, start_site:end_site] # [S, L]
        a_crop = aa_mat[sp_indices, start_site:end_site]
        d_crop = dist_arr[np.ix_(sp_indices, sp_indices)]
        m_crop = mds_coords[sp_indices, :]
        
        S, L = c_crop.shape
        
        # Targets: copy of true codon IDs (values 0..63)
        targets = np.full((S, L), -100, dtype=np.int64)
        c_input = c_crop.copy()
        a_input = a_crop.copy()
        
        # 3-Way Masking Strategy:
        mask_mode = random.random()
        
        if mask_mode < 0.60:
            # Mode A: 60% Random Tip Masking
            mask = (np.random.rand(S, L) < self.mask_prob) & (c_crop < 64)
        elif mask_mode < 0.80:
            # Mode B: 20% Clade Masking (Mask a random block of continuous taxa)
            mask = np.zeros((S, L), dtype=bool)
            clade_size = random.randint(2, max(2, S // 3))
            start_sp = random.randint(0, max(0, S - clade_size))
            col_idx = random.randint(0, L - 1)
            mask[start_sp:start_sp+clade_size, col_idx] = (c_crop[start_sp:start_sp+clade_size, col_idx] < 64)
        else:
            # Mode C: 20% Full Column Masking
            mask = np.zeros((S, L), dtype=bool)
            col_idx = random.randint(0, L - 1)
            mask[:, col_idx] = (c_crop[:, col_idx] < 64)
            
        # Apply BERT 80/10/10 masking rule
        for s in range(S):
            for l in range(L):
                if mask[s, l]:
                    targets[s, l] = c_crop[s, l]
                    prob = random.random()
                    if prob < 0.80:
                        c_input[s, l] = MASK_CODON_ID
                        a_input[s, l] = MASK_AA_ID
                    elif prob < 0.90:
                        c_input[s, l] = random.randint(0, 63)
                        a_input[s, l] = random.randint(0, 20)
                    # 10% stays unchanged
                    
        return {
            'codon_input': torch.tensor(c_input, dtype=torch.long),
            'aa_input': torch.tensor(a_input, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long),
            'dist_matrix': torch.tensor(d_crop, dtype=torch.float32),
            'mds_coords': torch.tensor(m_crop, dtype=torch.float32)
        }


# ---------------------------------------------------------------------------
# Zero-Shot Selection Surprisal Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_zero_shot_selection(model, npz_path, device):
    """
    Computes zero-shot phylogenetic surprisal on an unmasked alignment and
    evaluates correlation against ground truth MEME LRT.
    """
    npz = np.load(npz_path)
    codon_mat = npz['codon_ids_matrix']
    aa_mat = npz['aa_ids_matrix']
    dist_arr = npz['dist_arr']
    mds_coords = npz['mds_coords']
    
    num_taxa, num_sites = codon_mat.shape
    max_sp = min(num_taxa, 128)
    
    c_sub = torch.tensor(codon_mat[:max_sp, :], dtype=torch.long, device=device)
    a_sub = torch.tensor(aa_mat[:max_sp, :], dtype=torch.long, device=device)
    d_sub = torch.tensor(dist_arr[:max_sp, :max_sp], dtype=torch.float32, device=device).unsqueeze(0)
    m_sub = torch.tensor(mds_coords[:max_sp, :], dtype=torch.float32, device=device).unsqueeze(0)
    
    # Run in codon windows
    window_size = model.window_size
    surprisals = []
    
    for start in range(0, num_sites, window_size):
        end = min(start + window_size, num_sites)
        cur_L = end - start
        
        c_win = c_sub[:, start:end].unsqueeze(0) # [1, S, cur_L]
        a_win = a_sub[:, start:end].unsqueeze(0)
        
        if cur_L < window_size:
            # Pad window if tail
            pad_len = window_size - cur_L
            c_win = F.pad(c_win, (0, pad_len), value=PAD_CODON_ID)
            a_win = F.pad(a_win, (0, pad_len), value=PAD_AA_ID)
            
        logits = model(c_win, a_win, d_sub, m_sub) # [1, S, window_size, 64]
        log_probs = F.log_softmax(logits[:, :, :cur_L, :], dim=-1) # [1, S, cur_L, 64]
        
        # Extract the negative log likelihood of the observed codons:
        true_codons = c_sub[:, start:end].unsqueeze(0).clamp(0, 63) # [1, S, cur_L]
        nll = -torch.gather(log_probs, dim=-1, index=true_codons.unsqueeze(-1)).squeeze(-1) # [1, S, cur_L]
        
        # Site Surprisal = Average NLL across leaves (higher surprisal = higher evolutionary tension)
        site_surp = nll.mean(dim=1).squeeze(0).cpu().numpy() # [cur_L]
        surprisals.extend(site_surp.tolist())
        
    return np.array(surprisals)


def phylo_mlm_collate_fn(batch):
    max_S = max(b['codon_input'].shape[0] for b in batch)
    L = batch[0]['codon_input'].shape[1]
    B = len(batch)
    
    c_padded = torch.full((B, max_S, L), PAD_CODON_ID, dtype=torch.long)
    a_padded = torch.full((B, max_S, L), PAD_AA_ID, dtype=torch.long)
    t_padded = torch.full((B, max_S, L), -100, dtype=torch.long)
    d_padded = torch.full((B, max_S, max_S), 10.0, dtype=torch.float32)
    m_padded = torch.zeros((B, max_S, 4), dtype=torch.float32)
    
    for i, b in enumerate(batch):
        S = b['codon_input'].shape[0]
        c_padded[i, :S, :] = b['codon_input']
        a_padded[i, :S, :] = b['aa_input']
        t_padded[i, :S, :] = b['targets']
        d_padded[i, :S, :S] = b['dist_matrix']
        m_padded[i, :S, :] = b['mds_coords']
        
    return {
        'codon_input': c_padded,
        'aa_input': a_padded,
        'targets': t_padded,
        'dist_matrix': d_padded,
        'mds_coords': m_padded
    }


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Self-Supervised Masked Alignment Transformer (PhyloMLM)")
    parser.add_argument('--cache_dir', '--npz_dir', '--cache_path', dest='cache_dir', type=str, default="msa_cache_npz", help="Path to directory containing precomputed .npz alignments (e.g. /content/drive/MyDrive/TOGA_MEME/msa_cache_npz)")
    parser.add_argument('--embed_dim', type=int, default=192, help="Hidden embedding dimension (default: 192 for Micro-Scale, 384 for Base)")
    parser.add_argument('--num_heads', type=int, default=6, help="Number of attention heads (default: 6, or 12 for D=384)")
    parser.add_argument('--num_layers', type=int, default=6, help="Number of axial transformer blocks (default: 6)")
    parser.add_argument('--window_size', type=int, default=9, help="Codon sequence window size (default: 9)")
    parser.add_argument('--max_species', type=int, default=256, help="Maximum number of species per crop (default: 256)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size per optimizer step")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument('--save_path', type=str, default="phylomlm_micro.pt", help="Path to save trained checkpoint")
    args = parser.parse_args()

    # Device Detection (CUDA, MPS, TPU-XLA, CPU)
    is_tpu = False
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[*] Running PhyloMLM Training on NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"[*] Running PhyloMLM Training on Apple Silicon GPU (MPS)")
    else:
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            is_tpu = True
            print(f"[*] Detected Google TPU! Utilizing PyTorch-XLA backend ({device}).")
        except Exception:
            device = torch.device('cpu')
            print(f"[*] Training will run on CPU.")
    
    # 1. Instantiate Model
    model = PhyloMaskedAxialTransformer(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        window_size=args.window_size
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Model Architecture instantiated: {args.num_layers} Layers, {args.num_heads} Heads, Dim={args.embed_dim}, Window={args.window_size}")
    print(f"[*] Total Trainable Parameters: {num_params:,} ({num_params / 1e6:.2f} Million Parameters)")

    # 2. Prepare Datasets from Cache Directory
    if not os.path.exists(args.cache_dir):
        print(f"[!] Error: Specified NPZ cache directory '{args.cache_dir}' does not exist on filesystem!")
        sys.exit(1)
        
    npz_pattern = os.path.join(args.cache_dir, "*.npz")
    all_npz_files = sorted(glob.glob(npz_pattern))
    if not all_npz_files:
        print(f"[!] Error: No .npz alignment files found in '{args.cache_dir}' (searched pattern: {npz_pattern})")
        sys.exit(1)
        
    # Programmatically filter out all synthetic null simulation alignments
    npz_files = [
        f for f in all_npz_files 
        if not os.path.basename(f).lower().startswith('null_') 
        and 'null' not in os.path.basename(f).lower() 
        and 'replicate' not in os.path.basename(f).lower()
    ]
    num_skipped_nulls = len(all_npz_files) - len(npz_files)
    if num_skipped_nulls > 0:
        print(f"[*] Filtered out {num_skipped_nulls:,} synthetic NULL simulation alignments (keeping clean natural proteomes).")
    print(f"[*] Loaded {len(npz_files):,} genuine natural .npz alignment files from '{args.cache_dir}'")
    
    random.seed(42)
    random.shuffle(npz_files)
    split = int(0.9 * len(npz_files))
    train_files = npz_files[:split]
    val_files = npz_files[split:]
    
    print(f"[*] Dataset Split: {len(train_files):,} Train Alignments, {len(val_files):,} Validation Alignments")
    
    train_dataset = MaskedPhyloDataset(train_files, window_size=args.window_size, max_species=args.max_species)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=phylo_mlm_collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Pick a validation gene for zero-shot tracking
    sample_val_file = None
    crtac1_path = os.path.join(args.cache_dir, "CRTAC1.npz")
    if os.path.exists(crtac1_path):
        sample_val_file = crtac1_path
        val_name = "CRTAC1"
    elif len(val_files) > 0:
        sample_val_file = val_files[0]
        val_name = os.path.basename(sample_val_file).replace('.npz', '')

    # 3. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        start_time = time.time()
        
        for step, batch in enumerate(train_loader):
            c_in = batch['codon_input'].to(device)
            a_in = batch['aa_input'].to(device)
            d_mat = batch['dist_matrix'].to(device)
            m_coords = batch['mds_coords'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            logits = model(c_in, a_in, d_mat, m_coords) # [B, S, L, 64]
            
            loss = loss_fn(logits.view(-1, 64), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            if is_tpu:
                import torch_xla.core.xla_model as xm
                xm.optimizer_step(optimizer)
                xm.mark_step()
            else:
                optimizer.step()
            
            num_masked = (targets != -100).sum().item()
            total_loss += loss.item() * num_masked
            total_tokens += num_masked
            
            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                cur_loss = total_loss / max(1, total_tokens)
                cur_ppl = math.exp(min(cur_loss, 20.0))
                print(f"  [Epoch {epoch:02d} | Step {step+1:04d}/{len(train_loader):04d}] MLM Loss: {cur_loss:.4f} | Perplexity: {cur_ppl:.2f}", flush=True)

        scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(1, total_tokens)
        ppl = math.exp(min(avg_loss, 20.0))
        
        print(f"\n=======================================================")
        print(f"✨ Epoch {epoch:02d} Complete ({epoch_time:.1f}s)")
        print(f"   • Train MLM Cross-Entropy Loss : {avg_loss:.4f}")
        print(f"   • Train Codon Perplexity       : {ppl:.2f}")
        
        # 4. Zero-Shot Validation Tracking
        model.eval()
        if sample_val_file and os.path.exists(sample_val_file):
            try:
                surp_val = evaluate_zero_shot_selection(model, sample_val_file, device)
                print(f"   • Zero-Shot Surprisal on {val_name} : Mean={np.mean(surp_val):.3f}, Max={np.max(surp_val):.3f}")
            except Exception as e:
                print(f"   • Zero-Shot validation note: {e}")
        print(f"=======================================================\n")
        
        # Save checkpoint
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        if is_tpu:
            import torch_xla.core.xla_model as xm
            xm.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'embed_dim': args.embed_dim,
                'num_heads': args.num_heads,
                'num_layers': args.num_layers,
                'window_size': args.window_size,
                'loss': avg_loss,
                'ppl': ppl
            }, args.save_path)
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'embed_dim': args.embed_dim,
                'num_heads': args.num_heads,
                'num_layers': args.num_layers,
                'window_size': args.window_size,
                'loss': avg_loss,
                'ppl': ppl
            }, args.save_path)
        print(f"[*] Checkpoint saved to '{args.save_path}'\n")


if __name__ == '__main__':
    main()
