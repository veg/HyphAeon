#!/usr/bin/env python3
"""
package_hyphaeon_safetensors.py:
Unified SafeTensors (Standard 1) Packager for HyphAeon Foundation Model Suite.
Packs the Backbone + MEME Head + BUSTED Head + aBSREL Head into a single
production-ready, zero-copy memory-mapped `model.safetensors` file.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

# Add axomeme_repo to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "axomeme_repo")))
from hyphaeon.model import PhyloAxialTransformer, RankConsistentCoralHead, BustedMultiTaskHead

class aBSRELBranchHead(nn.Module):
    """Pillar 5: Branch-Site Episodic Selection Head (aBSREL Emulation)."""
    def __init__(self, embed_dim=384, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.head_p_branch = nn.Linear(hidden_dim, 1)
        self.head_omega_branch = nn.Linear(hidden_dim, 1)
        self.head_prop_branch = nn.Linear(hidden_dim, 1)

# ==============================================================================
# 2. UNIFIED HYPHAEON CONTAINER
# ==============================================================================

class HyphAeonFoundationSuite(nn.Module):
    """
    Master container uniting the Backbone with all downstream specialized heads.
    """
    def __init__(self, embed_dim=384, num_layers=6, num_heads=12, num_thresholds=16):
        super().__init__()
        self.backbone = PhyloAxialTransformer(embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads)
        self.head_meme = RankConsistentCoralHead(embed_dim=embed_dim, num_thresholds=num_thresholds)
        self.head_busted = BustedMultiTaskHead(embed_dim=embed_dim)
        self.head_absrel = aBSRELBranchHead(embed_dim=embed_dim)

def package_unified_model(backbone_pt=None, busted_pt=None, output_path="model.safetensors"):
    embed_dim = 384
    num_layers = 6
    num_heads = 12
    num_thresholds = 16
    
    if backbone_pt and os.path.exists(backbone_pt):
        ckpt = torch.load(backbone_pt, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        if "pos_embedding" in state_dict:
            embed_dim = state_dict["pos_embedding"].shape[-1]
        
        # Count layers
        layer_keys = [k for k in state_dict.keys() if k.startswith("row_layers.")]
        if layer_keys:
            max_l = max(int(k.split('.')[1]) for k in layer_keys)
            num_layers = max_l + 1
            
        if "row_layers.0.tree_w1" in state_dict:
            num_heads = state_dict["row_layers.0.tree_w1"].shape[0]
            
        if "lrt_ordinal_head.theta_steps" in state_dict:
            num_thresholds = state_dict["lrt_ordinal_head.theta_steps"].shape[0] + 1
            
    print(f"[*] Initializing HyphAeon Foundation Suite ({embed_dim}d, {num_layers} layers, {num_heads} heads, {num_thresholds} thresholds)...")
    suite = HyphAeonFoundationSuite(embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads, num_thresholds=num_thresholds)
    
    # 1. Load Backbone & MEME Weights
    if backbone_pt and os.path.exists(backbone_pt):
        print(f"[*] Loading backbone weights from {backbone_pt}...")
        bb_dict = {k: v for k, v in state_dict.items() if not k.startswith("lrt_ordinal_head")}
        suite.backbone.load_state_dict(bb_dict, strict=False)
        
        meme_dict = {k.replace("lrt_ordinal_head.", ""): v for k, v in state_dict.items() if k.startswith("lrt_ordinal_head.")}
        if meme_dict:
            suite.head_meme.load_state_dict(meme_dict, strict=False)
            
    # 2. Load BUSTED Head Weights
    if busted_pt and os.path.exists(busted_pt):
        print(f"[*] Loading BUSTED head weights from {busted_pt}...")
        ckpt_b = torch.load(busted_pt, map_location="cpu", weights_only=False)
        head_dict = ckpt_b.get("model_state_dict", ckpt_b.get("head_state_dict", ckpt_b))
        try:
            suite.head_busted.load_state_dict(head_dict, strict=True)
            print("[✓] Successfully loaded exact BUSTED CORAL Head weights into suite.")
        except Exception as e:
            print(f"[*] Loading BUSTED adapter with partial match: {e}")
            suite.head_busted.load_state_dict(head_dict, strict=False)
            
    # 3. Export to SafeTensors
    state_dict_all = suite.state_dict()
    print(f"[*] Packing {len(state_dict_all)} tensor parameters into SafeTensors format...")
    save_file(state_dict_all, output_path)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[+] Successfully generated Unified SafeTensors Model: {output_path} ({file_size_mb:.2f} MB)")
    
    # 4. Generate metadata config.json
    config = {
        "model_type": "hyphaeon",
        "architectures": ["HyphAeonFoundationSuite"],
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_thresholds": num_thresholds,
        "pillars": {
            "pillar1_meme": "Site-level episodic diversifying selection",
            "pillar2_epistasis": "3D epistatic co-evolution & contact recovery",
            "pillar3_digital_dms": "Compensated pathogenic deviations & in silico sweeps",
            "pillar4_busted": "Alignment-wide omnibus selection & SRV filtering",
            "pillar5_absrel": "Branch-site lineage-specific episodic selection bursts"
        },
        "version": "1.0.0"
    }
    
    config_path = os.path.splitext(output_path)[0] + "_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[+] Generated accompanying architecture config: {config_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="/Users/sergei/Projects/TOGA_MEME/axomeme_5_dim384_nonull.pt")
    parser.add_argument("--busted", default="/Users/sergei/Projects/TOGA_MEME/scratch/best_busted_head.pt")
    parser.add_argument("--output", default="/Users/sergei/Projects/TOGA_MEME/axomeme_repo/model.safetensors")
    args = parser.parse_args()
    
    package_unified_model(
        backbone_pt=args.backbone,
        busted_pt=args.busted,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
