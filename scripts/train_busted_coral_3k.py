import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cpu")
print(f"[*] Training on {device} across 8 threads...")
torch.set_num_threads(8)

CORAL_LRT_THRESHOLDS = torch.tensor([0.10, 0.50, 1.00, 2.00, 3.84, 5.99, 9.21, 13.82, 20.00, 35.00, 50.00, 75.00, 100.0, 150.0, 250.0, 500.0], dtype=torch.float32)
CORAL_LOGP_THRESHOLDS = torch.tensor([0.30, 0.52, 1.00, 1.30, 2.00, 3.00, 4.00, 5.00, 7.00, 9.00, 12.00, 15.00], dtype=torch.float32)
CORAL_OMEGA3_THRESHOLDS = torch.tensor([1.00, 1.25, 1.50, 2.00, 3.00, 5.00, 10.00, 20.00, 50.00, 100.0, 250.0, 1000.0], dtype=torch.float32)

class CoralHead(nn.Module):
    def __init__(self, hidden_dim, num_thresholds, b0_init=1.0):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1, bias=False)
        self.b0 = nn.Parameter(torch.tensor(float(b0_init)))
        self.raw_steps = nn.Parameter(torch.full((num_thresholds - 1,), 0.2))
    def get_cutoffs(self):
        steps = F.softplus(self.raw_steps)
        cum_steps = torch.cumsum(steps, dim=0)
        return torch.cat([self.b0.unsqueeze(0), self.b0 - cum_steps])
    def forward(self, x):
        proj = self.fc(x)
        cutoffs = self.get_cutoffs().to(device=x.device, dtype=x.dtype)
        return proj + cutoffs.unsqueeze(0)

class BustedCoralMultiTaskHead(nn.Module):
    def __init__(self, embed_dim=256, num_queries=4, num_heads=4, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp_shared = nn.Sequential(
            nn.Linear(num_queries * embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.head_cls = nn.Linear(hidden_dim, 1)
        self.coral_lrt = CoralHead(hidden_dim, len(CORAL_LRT_THRESHOLDS), b0_init=1.0)
        self.coral_logp = CoralHead(hidden_dim, len(CORAL_LOGP_THRESHOLDS), b0_init=0.5)
        self.coral_omega3 = CoralHead(hidden_dim, len(CORAL_OMEGA3_THRESHOLDS), b0_init=0.0)

    def forward(self, x, mask=None):
        if x.dim() == 2: x = x.unsqueeze(0)
        B = x.shape[0]
        q = self.queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(q, x, x, key_padding_mask=mask)
        h = self.norm(attn_out).reshape(B, -1)
        feat = self.mlp_shared(h)
        logits_cls = self.head_cls(feat).squeeze(-1)
        prob_cls = torch.sigmoid(logits_cls)
        return {
            "cls_prob": prob_cls,
            "logits_cls": logits_cls,
            "logits_lrt": self.coral_lrt(feat),
            "logits_logp": self.coral_logp(feat),
            "logits_omega3": self.coral_omega3(feat),
        }

cache_file = os.path.expanduser("~/HyphAeon_BUSTED/cached_features.pt")
existing_data = torch.load(cache_file, map_location="cpu")
print(f"[*] Loaded {len(existing_data)} cached datasets.")

class FastDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        reps = item["site_reps"].float()
        lrt = float(item["sqrt_lrt"] ** 2)
        logp = float(item["log10_p"])
        w3 = float(item["omega_vals"][2]) if len(item.get("omega_vals", [])) > 2 else 1.0
        return reps, float(item["is_sig"]), lrt, logp, w3

def collate_fn(batch):
    reps_list, sigs, lrts, logps, w3s = zip(*batch)
    max_L = max(r.shape[0] for r in reps_list)
    B = len(batch)
    padded = torch.zeros(B, max_L, 256)
    mask = torch.ones(B, max_L, dtype=torch.bool)
    for i, r in enumerate(reps_list):
        padded[i, :r.shape[0]] = r
        mask[i, :r.shape[0]] = False
    return padded, mask, torch.tensor(sigs, dtype=torch.float32), torch.tensor(lrts, dtype=torch.float32), torch.tensor(logps, dtype=torch.float32), torch.tensor(w3s, dtype=torch.float32)

np.random.seed(42)
indices = np.random.permutation(len(existing_data))
split = int(0.9 * len(existing_data))
train_data = [existing_data[i] for i in indices[:split]]
val_data = [existing_data[i] for i in indices[split:]]

train_loader = DataLoader(FastDataset(train_data), batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(FastDataset(val_data), batch_size=64, shuffle=False, collate_fn=collate_fn)

head = BustedCoralMultiTaskHead(embed_dim=256).to(device)
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

lrt_thresh = CORAL_LRT_THRESHOLDS.to(device)
logp_thresh = CORAL_LOGP_THRESHOLDS.to(device)
w3_thresh = CORAL_OMEGA3_THRESHOLDS.to(device)

best_val_loss = float("inf")
best_ckpt_path = os.path.expanduser("~/HyphAeon_BUSTED/output/best_busted_coral_head_3k.pt")

t0 = time.time()
for epoch in range(1, 41):
    head.train()
    total_loss = 0.0
    for x, mask, sig, lrt, logp, w3 in train_loader:
        out = head(x, mask=mask)
        loss_cls = F.binary_cross_entropy_with_logits(out["logits_cls"], sig)
        y_lrt = (lrt.unsqueeze(1) > lrt_thresh.unsqueeze(0)).float()
        loss_lrt = F.binary_cross_entropy_with_logits(out["logits_lrt"], y_lrt)
        y_logp = (logp.unsqueeze(1) > logp_thresh.unsqueeze(0)).float()
        loss_logp = F.binary_cross_entropy_with_logits(out["logits_logp"], y_logp)
        y_w3 = (w3.unsqueeze(1) > w3_thresh.unsqueeze(0)).float()
        loss_w3 = F.binary_cross_entropy_with_logits(out["logits_omega3"], y_w3)
        loss = loss_cls + 1.5 * loss_lrt + 1.0 * loss_logp + 1.0 * loss_w3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    head.eval()
    val_loss = 0.0
    correct = 0
    total_val = 0
    with torch.no_grad():
        for x, mask, sig, lrt, logp, w3 in val_loader:
            out = head(x, mask=mask)
            loss_cls = F.binary_cross_entropy_with_logits(out["logits_cls"], sig)
            y_lrt = (lrt.unsqueeze(1) > lrt_thresh.unsqueeze(0)).float()
            loss_lrt = F.binary_cross_entropy_with_logits(out["logits_lrt"], y_lrt)
            y_logp = (logp.unsqueeze(1) > logp_thresh.unsqueeze(0)).float()
            loss_logp = F.binary_cross_entropy_with_logits(out["logits_logp"], y_logp)
            y_w3 = (w3.unsqueeze(1) > w3_thresh.unsqueeze(0)).float()
            loss_w3 = F.binary_cross_entropy_with_logits(out["logits_omega3"], y_w3)
            val_loss += (loss_cls + 1.5 * loss_lrt + 1.0 * loss_logp + 1.0 * loss_w3).item()
            preds = (out["cls_prob"] > 0.5).float()
            correct += (preds == sig).sum().item()
            total_val += sig.shape[0]

    val_acc = correct / max(1, total_val) * 100
    val_loss /= max(1, len(val_loader))

    if epoch % 5 == 0 or epoch == 40:
        print(f"Epoch {epoch:02d}/40 | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(head.state_dict(), best_ckpt_path)

print(f"[✓] Training Complete in {time.time()-t0:.1f}s! Best model saved to: {best_ckpt_path}")
