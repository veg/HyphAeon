#!/usr/bin/env python3
"""
train_busted_coral.py:
Trains the BustedCoralMultiTaskHead using Rank-Consistent Ordinal Logits (CORAL)
on pre-extracted HyphAeon site representations.
Completely eliminates regression compression and mean-reversion artifacts.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.stats as stats

# ==============================================================================
# 1. CORAL THRESHOLDS DEFINITIONS
# ==============================================================================

# 16 LRT Thresholds spanning 0.1 to 500.0
CORAL_LRT_THRESHOLDS = torch.tensor([
    0.10, 0.50, 1.00, 2.00, 3.841, 5.991, 10.0, 15.0, 
    25.0, 40.0, 65.0, 100.0, 150.0, 220.0, 350.0, 500.0
], dtype=torch.float32)

# 12 -log10(p) Thresholds spanning p=0.5 down to p=1e-15
CORAL_LOGP_THRESHOLDS = torch.tensor([
    0.301, 0.699, 1.000, 1.301, 2.000, 3.000, 
    4.000, 6.000, 8.000, 10.00, 12.00, 15.00
], dtype=torch.float32)

# 12 log(omega_3) Thresholds spanning omega=1.0 to omega=1000.0
CORAL_OMEGA3_THRESHOLDS = torch.tensor([
    1.00, 1.50, 2.00, 3.00, 5.00, 8.00, 
    15.0, 30.0, 60.0, 150.0, 400.0, 1000.0
], dtype=torch.float32)


# ==============================================================================
# 2. RANK-CONSISTENT CORAL MODULE
# ==============================================================================

class CoralHead(nn.Module):
    """
    Rank-Consistent Ordinal Regression Head (Cao et al. 2020).
    Guarantees monotonic threshold cutoffs b_1 > b_2 > ... > b_K via cumulative softplus.
    """
    def __init__(self, in_features, num_thresholds, b0_init=0.0):
        super().__init__()
        self.num_thresholds = num_thresholds
        self.fc = nn.Linear(in_features, 1, bias=False)
        self.b0 = nn.Parameter(torch.tensor(float(b0_init)))
        self.theta_steps = nn.Parameter(torch.full((num_thresholds - 1,), 0.40))
        nn.init.normal_(self.fc.weight, mean=0.0, std=1.0 / (in_features ** 0.5))

    def get_cutoffs(self):
        steps = F.softplus(self.theta_steps)
        cum_steps = torch.cumsum(steps, dim=0)
        return torch.cat([self.b0.unsqueeze(0), self.b0 - cum_steps])

    def forward(self, x):
        # x: [B, in_features]
        proj = self.fc(x)  # [B, 1]
        cutoffs = self.get_cutoffs().to(device=x.device, dtype=x.dtype) # [K]
        return proj + cutoffs.unsqueeze(0)  # [B, K]


def decode_coral_lrt(logits):
    """Smooth continuous integration for LRT."""
    probs = torch.sigmoid(logits)
    # Log-space deltas
    log_thresh = torch.log1p(CORAL_LRT_THRESHOLDS.to(logits.device))
    deltas = torch.cat([log_thresh[0:1], log_thresh[1:] - log_thresh[:-1]])
    expected_log_lrt = (probs * deltas.unsqueeze(0)).sum(dim=-1)
    return torch.expm1(expected_log_lrt)

def decode_coral_logp(logits):
    """Smooth continuous integration for -log10(p)."""
    probs = torch.sigmoid(logits)
    thresh = CORAL_LOGP_THRESHOLDS.to(logits.device)
    deltas = torch.cat([thresh[0:1], thresh[1:] - thresh[:-1]])
    expected_logp = (probs * deltas.unsqueeze(0)).sum(dim=-1)
    return expected_logp

def decode_coral_omega3(logits):
    """Smooth continuous integration for omega_3."""
    probs = torch.sigmoid(logits)
    log_thresh = torch.log(CORAL_OMEGA3_THRESHOLDS.to(logits.device))
    deltas = torch.cat([log_thresh[0:1], log_thresh[1:] - log_thresh[:-1]])
    expected_log_omega = (probs * deltas.unsqueeze(0)).sum(dim=-1)
    return torch.exp(expected_log_omega)


# ==============================================================================
# 3. BUSTED CORAL MULTI-TASK NEURAL HEAD
# ==============================================================================

class BustedCoralMultiTaskHead(nn.Module):
    """
    Pillar 4: BUSTED+S Alignment-Wide Head with CORAL Rank-Consistent Ordinal Logits.
    """
    def __init__(self, embed_dim=256, num_queries=4, num_heads=4, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
        self.mlp_shared = nn.Sequential(
            nn.Linear(num_queries * embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 1. Binary Selection Classifier (alpha = 0.05 gate)
        self.head_cls = nn.Linear(hidden_dim, 1)
        
        # 2. CORAL LRT Head (16 thresholds)
        self.coral_lrt = CoralHead(hidden_dim, num_thresholds=len(CORAL_LRT_THRESHOLDS), b0_init=1.0)
        
        # 3. CORAL -log10(p) Head (12 thresholds)
        self.coral_logp = CoralHead(hidden_dim, num_thresholds=len(CORAL_LOGP_THRESHOLDS), b0_init=0.5)
        
        # 4. CORAL omega_3 Head (12 thresholds)
        self.coral_omega3 = CoralHead(hidden_dim, num_thresholds=len(CORAL_OMEGA3_THRESHOLDS), b0_init=0.0)
        
        # 5. Synonymous Rate Variation Var(alpha)
        self.head_syn_var = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # 6. Omega Mixture Proportions (p1, p2, p3)
        self.head_prop = nn.Linear(hidden_dim, 3)

    def forward(self, x, mask=None):
        # x: [B, L, D] site embeddings
        B = x.shape[0]
        q = self.queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(q, x, x, key_padding_mask=mask)
        h = self.norm(attn_out).reshape(B, -1)
        feat = self.mlp_shared(h)
        
        return {
            "cls_logit": self.head_cls(feat).squeeze(-1),
            "logits_lrt": self.coral_lrt(feat),
            "logits_logp": self.coral_logp(feat),
            "logits_omega3": self.coral_omega3(feat),
            "syn_var": F.softplus(self.head_syn_var(feat)).squeeze(-1),
            "omega_prop": torch.softmax(self.head_prop(feat), dim=-1)
        }


# ==============================================================================
# 4. DATASET & CORAL LOSS COMPUTATION
# ==============================================================================

class BustedFeatureDataset(Dataset):
    def __init__(self, data_list, max_len=1024):
        self.items = data_list
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        feat = item["site_reps"] if "site_reps" in item else item.get("features")
        L, D = feat.shape
        if L > self.max_len:
            feat = feat[:self.max_len]
            L = self.max_len
        
        lrt = float(item["sqrt_lrt"] ** 2) if "sqrt_lrt" in item else float(item.get("lrt", 0.0))
        p_val = float(10.0 ** item["log10_p"]) if "log10_p" in item else float(item.get("p_val", 1.0))
        selected = float(item["is_sig"]) if "is_sig" in item else float(item.get("selected", 0.0))
        w3 = float(item["omega_vals"][2]) if "omega_vals" in item else float(item.get("w3", 1.0))
        p3 = float(item["omega_weights"][2]) if "omega_weights" in item else float(item.get("p3", 0.05))
        syn_var = float(item.get("syn_var", 0.0))

        return {
            "features": feat,
            "L": L,
            "lrt": lrt,
            "p_val": p_val,
            "selected": selected,
            "w3": w3,
            "p3": p3,
            "syn_var": syn_var
        }

def collate_fn(batch):
    B = len(batch)
    D = batch[0]["features"].shape[-1]
    max_L = max(b["L"] for b in batch)
    
    padded_feat = torch.zeros(B, max_L, D, dtype=torch.float32)
    mask = torch.ones(B, max_L, dtype=torch.bool)
    
    lrts = torch.zeros(B, dtype=torch.float32)
    p_vals = torch.zeros(B, dtype=torch.float32)
    selected = torch.zeros(B, dtype=torch.float32)
    w3 = torch.zeros(B, dtype=torch.float32)
    p3 = torch.zeros(B, dtype=torch.float32)
    syn_var = torch.zeros(B, dtype=torch.float32)
    
    for i, b in enumerate(batch):
        L = b["L"]
        padded_feat[i, :L] = b["features"]
        mask[i, :L] = False
        lrts[i] = b["lrt"]
        p_vals[i] = b["p_val"]
        selected[i] = b["selected"]
        w3[i] = max(1.0, b["w3"])
        p3[i] = b["p3"]
        syn_var[i] = b["syn_var"]
        
    return {
        "features": padded_feat,
        "mask": mask,
        "lrt": lrts,
        "p_val": p_vals,
        "selected": selected,
        "w3": w3,
        "p3": p3,
        "syn_var": syn_var
    }


def compute_coral_loss(logits, targets, thresholds):
    """
    Computes Cumulative Binary Cross-Entropy across ordinal thresholds.
    targets: [B], thresholds: [K]
    """
    B = targets.shape[0]
    K = thresholds.shape[0]
    thresh = thresholds.to(targets.device)
    # Binary indicator matrix: [B, K]
    binary_targets = (targets.unsqueeze(1) > thresh.unsqueeze(0)).float()
    return F.binary_cross_entropy_with_logits(logits, binary_targets)


# ==============================================================================
# 5. TRAINING LOOP
# ==============================================================================

def train_coral_model(cache_file, output_dir, epochs=40, batch_size=64, lr=1e-3):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Loading cached features from {cache_file}...")
    data_list = torch.load(cache_file, map_location="cpu")
    print(f"[*] Loaded {len(data_list)} alignment feature embeddings. Initializing CORAL Head...")
    
    np.random.seed(42)
    indices = np.random.permutation(len(data_list))
    split = int(0.85 * len(data_list))
    train_items = [data_list[i] for i in indices[:split]]
    val_items = [data_list[i] for i in indices[split:]]
    
    train_loader = DataLoader(BustedFeatureDataset(train_items), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(BustedFeatureDataset(val_items), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    first_feat = train_items[0]["site_reps"] if "site_reps" in train_items[0] else train_items[0].get("features")
    embed_dim = first_feat.shape[-1]
    model = BustedCoralMultiTaskHead(embed_dim=embed_dim, num_queries=4, num_heads=4, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float("inf")
    best_acc = 0.0
    best_spearman = -1.0
    
    print("\n" + "=" * 90)
    print(f"  STARTING BUSTED CORAL MULTI-TASK TRAINING ({epochs} EPOCHS, {len(train_items)} TRAIN / {len(val_items)} VAL)")
    print("=" * 90)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for b in train_loader:
            x = b["features"].to(device)
            mask = b["mask"].to(device)
            lrt_t = b["lrt"].to(device)
            p_val_t = b["p_val"].to(device)
            sel_t = b["selected"].to(device)
            w3_t = b["w3"].to(device)
            syn_var_t = b["syn_var"].to(device)
            
            optimizer.zero_grad()
            out = model(x, mask=mask)
            
            # 1. Binary Classification Loss
            loss_cls = F.binary_cross_entropy_with_logits(out["cls_logit"], sel_t)
            
            # 2. CORAL LRT Loss
            loss_coral_lrt = compute_coral_loss(out["logits_lrt"], lrt_t, CORAL_LRT_THRESHOLDS)
            
            # 3. CORAL -log10(p) Loss
            neg_log_p = -torch.log10(torch.clamp(p_val_t, min=1e-15))
            loss_coral_logp = compute_coral_loss(out["logits_logp"], neg_log_p, CORAL_LOGP_THRESHOLDS)
            
            # 4. CORAL omega_3 Loss
            loss_coral_w3 = compute_coral_loss(out["logits_omega3"], w3_t, CORAL_OMEGA3_THRESHOLDS)
            
            # 5. Synonymous Variation Smooth L1 Loss
            loss_syn = F.smooth_l1_loss(out["syn_var"], syn_var_t)
            
            loss = loss_cls + 1.0 * loss_coral_lrt + 0.8 * loss_coral_logp + 0.8 * loss_coral_w3 + 0.5 * loss_syn
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        scheduler.step()
        train_loss = total_loss / max(1, n_batches)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_n = 0
        all_pred_cls = []
        all_true_cls = []
        all_pred_lrt = []
        all_true_lrt = []
        all_pred_w3 = []
        all_true_w3 = []
        
        with torch.no_grad():
            for b in val_loader:
                x = b["features"].to(device)
                mask = b["mask"].to(device)
                lrt_t = b["lrt"].to(device)
                p_val_t = b["p_val"].to(device)
                sel_t = b["selected"].to(device)
                w3_t = b["w3"].to(device)
                syn_var_t = b["syn_var"].to(device)
                
                out = model(x, mask=mask)
                loss_cls = F.binary_cross_entropy_with_logits(out["cls_logit"], sel_t)
                loss_coral_lrt = compute_coral_loss(out["logits_lrt"], lrt_t, CORAL_LRT_THRESHOLDS)
                neg_log_p = -torch.log10(torch.clamp(p_val_t, min=1e-15))
                loss_coral_logp = compute_coral_loss(out["logits_logp"], neg_log_p, CORAL_LOGP_THRESHOLDS)
                loss_coral_w3 = compute_coral_loss(out["logits_omega3"], w3_t, CORAL_OMEGA3_THRESHOLDS)
                loss_syn = F.smooth_l1_loss(out["syn_var"], syn_var_t)
                
                v_loss = loss_cls + 1.0 * loss_coral_lrt + 0.8 * loss_coral_logp + 0.8 * loss_coral_w3 + 0.5 * loss_syn
                val_loss += v_loss.item()
                val_n += 1
                
                # Decode predictions
                p_cls = torch.sigmoid(out["cls_logit"])
                pred_lrt = decode_coral_lrt(out["logits_lrt"])
                pred_w3 = decode_coral_omega3(out["logits_omega3"])
                
                all_pred_cls.extend(p_cls.cpu().numpy())
                all_true_cls.extend(sel_t.cpu().numpy())
                all_pred_lrt.extend(pred_lrt.cpu().numpy())
                all_true_lrt.extend(lrt_t.cpu().numpy())
                all_pred_w3.extend(pred_w3.cpu().numpy())
                all_true_w3.extend(w3_t.cpu().numpy())
                
        val_loss /= max(1, val_n)
        all_pred_cls = np.array(all_pred_cls)
        all_true_cls = np.array(all_true_cls)
        all_pred_lrt = np.array(all_pred_lrt)
        all_true_lrt = np.array(all_true_lrt)
        
        acc = float(np.mean((all_pred_cls > 0.5) == all_true_cls))
        spearman_r, _ = stats.spearmanr(all_pred_lrt, all_true_lrt)
        if np.isnan(spearman_r): spearman_r = 0.0
        
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc*100:.1f}% | Spearman r_s: {spearman_r:+.4f}", flush=True)
        
        if acc > best_acc or (acc == best_acc and spearman_r > best_spearman):
            best_acc = acc
            best_spearman = spearman_r
            best_ckpt = os.path.join(output_dir, "best_busted_coral_head.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": acc,
                "spearman_r": spearman_r,
                "embed_dim": embed_dim
            }, best_ckpt)
            
    print("=" * 90)
    print(f"[✓] Best Model Saved: {best_ckpt} (Accuracy: {best_acc*100:.1f}%, Spearman r_s: {best_spearman:+.4f})")
    print("=" * 90)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_file", default="/Users/sergei/Projects/TOGA_MEME/scratch/cached_busted_features_1000.pt")
    parser.add_argument("--output_dir", default="/Users/sergei/Projects/TOGA_MEME/scratch")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    train_coral_model(args.cache_file, args.output_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

