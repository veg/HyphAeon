"""
aeon_core/inference.py
---------------------
Shared inference helpers: device selection, model loading, alignment
preparation, and adaptive batch sizing.

These are used by both hyphaeon and chronaeon packages.
"""

import os
from typing import Optional, Tuple
import numpy as np
import torch

from .model import PhyloAxialTransformer
from .weights import resolve_weights_path, load_arch_config, load_weights
from .dataset import load_alignment_and_tree


def get_live_available_memory(device: torch.device) -> int:
    """
    Dynamically probes live, unallocated accelerator VRAM, Unified Memory, or host RAM capacity
    across CUDA, MPS, TPU (XLA), and CPU to determine safe operational headroom (in bytes).
    """
    dev_type = device.type if hasattr(device, 'type') else str(device).lower()

    if 'cuda' in dev_type and torch.cuda.is_available():
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
            return int(free_bytes)
        except Exception:
            return 3 * 1024 * 1024 * 1024  # 3 GB fallback

    elif 'mps' in dev_type and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            rec_max = torch.mps.recommended_max_memory() if hasattr(torch.mps, 'recommended_max_memory') else 24 * (1024 ** 3)
            curr_alloc = torch.mps.current_allocated_memory() if hasattr(torch.mps, 'current_allocated_memory') else 0
            mps_free = max(0, rec_max - curr_alloc)
            try:
                import psutil
                host_free = psutil.virtual_memory().available
                return int(min(mps_free, host_free))
            except Exception:
                return int(mps_free)
        except Exception:
            return 4 * 1024 * 1024 * 1024

    else:  # CPU or other
        try:
            import psutil
            return int(psutil.virtual_memory().available)
        except Exception:
            try:
                sys_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
                return int(sys_ram * 0.20)
            except Exception:
                return 2 * 1024 * 1024 * 1024


def get_device_memory_budget(device: torch.device) -> float:
    """
    Dynamically probes accelerator VRAM, Unified Memory, or host RAM capacity
    across CUDA, MPS, TPU (XLA), and CPU to determine an optimal per-layer
    attention allocation budget (in bytes) based on live memory availability.
    """
    live_free = get_live_available_memory(device)
    # Target 15-20% of live free memory, clamped between 1.0 GB and 6.0 GB
    return float(min(6.0e9, max(1.0e9, live_free * 0.20)))


def compute_live_adaptive_chunk_sizes(
    device: torch.device,
    num_nodes: int,
    num_heads: int = 8,
    embed_dim: int = 128,
    window_size: int = 1
) -> Tuple[int, int]:
    """
    Computes memory-safe and latency-optimal (macro_batch_size, inner_chunk_size)
    dynamically calibrated to live available accelerator VRAM or host RAM.

    Guarantees:
      1. Peak attention tensor allocation stays within ~5% of live free memory (clamped 64MB - 768MB).
      2. Macro-batch embedding tensor stays within ~5% of live free memory (clamped 64MB - 512MB).
      3. Chunk sizes dynamically scale up on roomy machines (e.g. 8-16 sites on 16GB+ MPS / A100)
         while gracefully throttling to 1-2 sites under severe memory pressure, eliminating OOM crashes.
    """
    live_free = get_live_available_memory(device)

    # Attention score budget: 5% of live free memory, clamped between 64 MB and 768 MB
    attn_budget = max(64 * 1024 * 1024, min(768 * 1024 * 1024, int(live_free * 0.05)))
    bytes_per_site_attn = num_heads * (num_nodes ** 2) * 4  # float32
    inner_chunk_size = max(1, min(32, int(attn_budget / max(1, bytes_per_site_attn))))

    # Macro-batch embedding budget: 5% of live free memory, clamped between 64 MB and 512 MB
    emb_budget = max(64 * 1024 * 1024, min(512 * 1024 * 1024, int(live_free * 0.05)))
    bytes_per_site_emb = window_size * num_nodes * embed_dim * 4
    macro_batch_size = max(1, min(128, int(emb_budget / max(1, bytes_per_site_emb))))

    return macro_batch_size, inner_chunk_size


def compute_adaptive_safe_batch_size(
    n_taxa: int, 
    user_batch_size: Optional[int] = None,
    device: Optional[torch.device] = None
) -> int:
    """
    Computes an optimal, memory-safe batch size dynamically adapted to:
      1. Sequence depth N (quadratic scaling: 48 * N^2 bytes per sequence)
      2. Hardware accelerator capacity (CUDA, MPS, TPU/XLA, or CPU)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        
    budget_bytes = get_device_memory_budget(device)
    # Memory per sequence = 12 heads * N^2 * 4 bytes = 48 * N^2 bytes
    safe_max_b = max(1, int(budget_bytes / (48.0 * max(1, n_taxa) ** 2)))
    
    # Structure safe batches in clean multiples of 19 (for 19 AA DMS passes)
    if n_taxa >= 600:
        base_cap = 38 if budget_bytes <= 3.0e9 else (76 if budget_bytes <= 6.0e9 else 152)
        safe_b = min(base_cap, safe_max_b)
    elif n_taxa >= 400:
        base_cap = 57 if budget_bytes <= 3.0e9 else (114 if budget_bytes <= 6.0e9 else 228)
        safe_b = min(base_cap, safe_max_b)
    elif n_taxa >= 250:
        base_cap = 76 if budget_bytes <= 3.0e9 else (152 if budget_bytes <= 6.0e9 else 256)
        safe_b = min(base_cap, safe_max_b)
    elif n_taxa >= 100:
        safe_b = min(128 if budget_bytes <= 3.0e9 else 256, safe_max_b)
    else:
        safe_b = min(256 if budget_bytes <= 3.0e9 else 512, safe_max_b)
        
    safe_b = max(1, safe_b)
    
    if user_batch_size is not None and user_batch_size > 0:
        if user_batch_size > safe_b * 2:
            print(f"[!] Warning: Requested batch size {user_batch_size} exceeds hardware safety threshold ({safe_b}) for N={n_taxa} on {device.type.upper()}.", flush=True)
            print(f"    Automatically capping batch size to {safe_b} to guarantee stable execution.", flush=True)
            return safe_b
        return user_batch_size
        
    return safe_b


def get_device(cpu: bool = False) -> torch.device:
    """Select the best available hardware device.

    CUDA > MPS > CPU, unless cpu=True forces CPU.
    """
    if cpu:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(weights=None, variant=None, device=None, strict=False):
    """Load a PhyloAxialTransformer from weights path or HuggingFace variant.

    Returns an eval-mode model on the specified device.
    """
    if device is None:
        device = get_device()

    weights_path = resolve_weights_path(weights=weights, variant=variant)
    config = load_arch_config(weights=weights, variant=variant)
    model = PhyloAxialTransformer(
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        window_size=config['window_size'],
    ).to(device)
    state_dict = load_weights(weights=weights_path, variant=variant, map_location=device)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model


def prepare_alignment(alignment_path, tree_path=None, model=None, device=None,
                      max_species=None, prune_duplicates=True, use_tn93=False):
    """Load an alignment + tree and precompute the tree attention cache.

    If use_tn93=True, skips the phylogenetic tree and computes pairwise
    distances directly from the alignment using TN93.

    Returns (c, a, d, z, inv, taxa, L, tree_cache).
    If model is None, tree_cache will be None.
    """
    c, a, d, z, inv, taxa, L = load_alignment_and_tree(
        alignment_path, tree_path, max_species=max_species,
        prune_duplicates=prune_duplicates, use_tn93=use_tn93
    )
    tree_cache = None
    if model is not None:
        tree_cache = model.precompute_tree_cache(d.to(device), z.to(device))
    return c, a, d, z, inv, taxa, L, tree_cache
