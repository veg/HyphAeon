#!/usr/bin/env python3
"""
train.py
--------
Clean training pipeline for HyphAeon (PhyloAxialTransformer).
Supports mixed precision, cosine annealing learning rate scheduling, per-gene NPZ
loading, and within-gene site batching.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hyphaeon.model import PhyloAxialTransformer, decode_soft_ordinal_lrt
from hyphaeon.training_data import GeneTensorsDataset


def load_initial_checkpoint(model, checkpoint_path):
    """Strictly initialize model weights without resuming optimizer state."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a state dictionary: {checkpoint_path}")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"Checkpoint architecture is incompatible with the requested model: "
            f"{checkpoint_path}"
        ) from exc
    return checkpoint


def collate_single_gene(items):
    """Remove DataLoader's outer batch dimension; gene batching is always one."""
    if len(items) != 1:
        raise ValueError(f"Expected exactly one gene from DataLoader, received {len(items)}")
    return items[0]


def iter_site_indices(eligible_mask, batch_size, shuffle=True, generator=None):
    """Yield each eligible site index exactly once in chunks of ``batch_size``."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    indices = torch.nonzero(eligible_mask, as_tuple=False).reshape(-1)
    if shuffle and indices.numel() > 1:
        order = torch.randperm(indices.numel(), generator=generator)
        indices = indices[order]
    for start in range(0, indices.numel(), batch_size):
        yield indices[start : start + batch_size]


def train_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    total_loss = 0.0
    total_sites = 0
    
    for gene in loader:
        d = gene['d'].to(device)
        z = gene['z'].to(device)
        for site_indices in iter_site_indices(
            gene['eligible_mask'], args.batch_size, shuffle=True
        ):
            site_count = site_indices.numel()
            c = gene['c'].index_select(0, site_indices).to(device)
            a = gene['a'].index_select(0, site_indices).to(device)
            y_true = gene['target_lrt'].index_select(0, site_indices).to(device)
            batch_d = d.unsqueeze(0).expand(site_count, -1, -1)
            batch_z = z.unsqueeze(0).expand(site_count, -1, -1)

            optimizer.zero_grad()
            use_fp16 = bool(args.fp16 and device.type == 'cuda')
            with torch.amp.autocast('cuda', enabled=use_fp16):
                out = model(c, a, batch_d, batch_z)
                if isinstance(out, tuple):
                    y_pred, logits = out
                else:
                    logits = out
                    y_pred, _ = decode_soft_ordinal_lrt(logits)
                y_true = y_true.reshape(-1)
                if y_pred.shape != y_true.shape:
                    raise ValueError(
                        f"Target shape {tuple(y_true.shape)} is incompatible with "
                        f"decoded prediction shape {tuple(y_pred.shape)}"
                    )
                loss = nn.functional.smooth_l1_loss(y_pred, y_true, beta=1.0)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * site_count
            total_sites += site_count

    if total_sites == 0:
        raise ValueError("No eligible sites were found in any training gene")
    return total_loss / total_sites


def main():
    parser = argparse.ArgumentParser(description="Train HyphAeon Neural Selection Predictor")
    parser.add_argument(
        "--data_dir", required=True, help="Directory containing per-gene .npz tensors"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of sites per optimizer step")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--embed_dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--layers", type=int, default=6, help="Number of axial transformer layers")
    parser.add_argument("--heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision")
    parser.add_argument(
        "--init_checkpoint",
        help="Optional checkpoint used to initialize model weights for fine-tuning",
    )
    parser.add_argument("--output_dir", default="weights", help="Directory to save checkpoint snapshots")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be greater than zero")
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = GeneTensorsDataset(args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_single_gene,
    )
    
    model = PhyloAxialTransformer(
        embed_dim=args.embed_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        window_size=1
    )
    if args.init_checkpoint:
        load_initial_checkpoint(model, args.init_checkpoint)
        print(f"[*] Initialized model weights from: {args.init_checkpoint}")
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(
        'cuda', enabled=bool(args.fp16 and device.type == 'cuda')
    )
    
    print(f"[*] Starting training on {device} ({args.epochs} epochs)...")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, loader, optimizer, scaler, device, args)
        scheduler.step()
        elapsed = time.time() - t0
        
        print(f"Epoch [{epoch:2d}/{args.epochs:2d}] - Loss: {loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s")
        
        if loss < best_loss:
            best_loss = loss
            ckpt_path = os.path.join(args.output_dir, "hyphaeon_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'args': vars(args)
            }, ckpt_path)
            print(f"    [✓] Saved new best model to: {ckpt_path}")

if __name__ == '__main__':
    main()
