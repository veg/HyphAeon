"""
hyphaeon/inference.py
---------------------
Shared inference helpers: device selection, model loading, alignment
preparation, and batched site-level LRT prediction.

These are used by the CLI subcommands (cmd_meme, cmd_busted, etc.),
hyphaeon/phenotype.py, hyphaeon/epistasis.py, hyphaeon/disease.py,
hyphaeon/filter.py, and model_eval/_harness.py.
"""

import os
from typing import Optional
import numpy as np
import torch

from .model import PhyloAxialTransformer
from .weights import resolve_weights_path, load_arch_config, load_weights
from .dataset import load_alignment_and_tree


def get_device_memory_budget(device: torch.device) -> float:
    """
    Dynamically probes accelerator VRAM, Unified Memory, or host RAM capacity
    across CUDA, MPS, TPU (XLA), and CPU to determine an optimal per-layer
    attention allocation budget (in bytes).
    """
    dev_type = device.type
    
    if dev_type == 'cuda' and torch.cuda.is_available():
        try:
            total_vram = torch.cuda.get_device_properties(device).total_memory
            # Allocate up to 20% of VRAM for peak single-layer attention tensor (bounded between 1.5 GB and 8.0 GB)
            return float(min(8.0e9, max(1.5e9, total_vram * 0.20)))
        except Exception:
            return 3.0e9
            
    elif dev_type == 'mps':
        try:
            # Query macOS unified RAM capacity
            sys_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            # 10% of unified memory (bounded between 1.5 GB and 6.0 GB)
            return float(min(6.0e9, max(1.5e9, sys_ram * 0.10)))
        except Exception:
            return 2.0e9
            
    elif dev_type == 'xla':
        # TPU v2/v3/v4/v5e typically feature 16GB-32GB HBM per core
        return 4.0e9
        
    else: # CPU
        try:
            sys_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            return float(min(4.0e9, max(1.0e9, sys_ram * 0.10)))
        except Exception:
            return 1.5e9


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
                      max_species=None, prune_duplicates=True):
    """Load an alignment + tree and precompute the tree attention cache.

    Returns (c, a, d, z, inv, taxa, L, tree_cache).
    If model is None, tree_cache will be None.
    """
    c, a, d, z, inv, taxa, L = load_alignment_and_tree(
        alignment_path, tree_path, max_species=max_species,
        prune_duplicates=prune_duplicates
    )
    tree_cache = None
    if model is not None:
        tree_cache = model.precompute_tree_cache(d.to(device), z.to(device))
    return c, a, d, z, inv, taxa, L, tree_cache


from ._progress import ChunkProgress


def predict_site_lrts(model, c, a, d, z, inv, tree_cache=None,
                      batch_size=64, device=None, progress=True, desc="Predict"):
    """Run model.forward_cached on variable sites only; return LRT array [L].

    Invariable sites get LRT=0. Uses tree_cache if provided, otherwise
    precomputes it.
    """
    if device is None:
        device = next(model.parameters()).device

    L = c.shape[0]
    lrts = np.zeros(L, dtype=np.float32)
    var_idx = np.where(~inv)[0]
    num_variable = len(var_idx)
    if num_variable == 0:
        return lrts

    if tree_cache is None:
        tree_cache = model.precompute_tree_cache(d.to(device), z.to(device))

    pb = ChunkProgress(num_variable, desc, 'codon', enabled=progress and num_variable > 0)
    with torch.no_grad():
        for s in range(0, num_variable, batch_size):
            end_idx = min(s + batch_size, num_variable)
            idx = var_idx[s:end_idx]
            y, _ = model.forward_cached(c[idx].to(device), a[idx].to(device), tree_cache)
            lrts[idx] = torch.clamp(y.squeeze(-1), min=0.0).cpu().numpy().flatten()
            pb.update(end_idx)
    pb.finish()

    return lrts


def run_busted_inference(
    model,
    busted_head,
    c, a, d, z, inv, taxa, L,
    *,
    device=None,
    batch_size=64,
    progress=True,
    desc="BUSTED",
):
    """Run BUSTED omnibus inference on pre-loaded tensors.

    Returns a record dict with: taxa, sites, p_value_acat, p_value_simes,
    omnibus_lrt, predicted_gene_lrt, selection_probability,
    synonymous_rate_variation, total_selection_energy, sig_sites_p05,
    sig_sites_p10, rate_distributions, positive_selection_detected,
    elapsed_seconds.

    Used by cmd_busted (CLI) and model_eval/_harness.py (tests) so the
    inference logic lives in exactly one place.
    """
    import time
    from .stats import pvals_from_lrt_self_liang, cauchy_combination_p

    if device is None:
        device = next(model.parameters()).device
    t0 = time.time()

    variable_indices = np.where(~inv)[0]
    num_variable = len(variable_indices)
    num_species = len(taxa)
    embed_dim = next(model.parameters()).shape[-1]

    tree_cache = model.precompute_tree_cache(d.to(device), z.to(device))
    lrts = np.zeros(L, dtype=np.float32)
    hidden_all = torch.zeros((1, L, embed_dim), dtype=torch.float32)

    if num_variable > 0:
        pb = ChunkProgress(num_variable, desc, 'codon',
                          enabled=progress and num_variable > 0)
        with torch.no_grad():
            for start_idx in range(0, num_variable, batch_size):
                end_idx = min(start_idx + batch_size, num_variable)
                batch_site_idx = variable_indices[start_idx:end_idx]
                c_chunk = c[batch_site_idx].to(device)
                a_chunk = a[batch_site_idx].to(device)
                y_soft, _, root_repr = model.forward_cached(
                    c_chunk, a_chunk, tree_cache, return_hidden=True)
                chunk_lrts = torch.clamp(
                    y_soft.squeeze(-1), min=0.0).cpu().numpy().flatten()
                lrts[batch_site_idx] = chunk_lrts
                hidden_all[0, batch_site_idx] = root_repr.cpu()
                pb.update(end_idx)
        pb.finish()

    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()

    # Neural BUSTED head evaluation
    with torch.no_grad():
        neural_out = busted_head(hidden_all.to(device))
        pred_prob_pos = float(neural_out["cls_prob"].item())
        pred_neural_lrt = float(neural_out["pred_lrt"].item())
        pred_syn_var = float(neural_out["syn_var"].item())
        pred_w3 = float(neural_out["pred_omega3"].item())
        pred_prop = neural_out["omega_prop"].squeeze().cpu().numpy()
        pred_omega = [0.10, 1.00, pred_w3]

    elapsed = time.time() - t0

    # Asymptotic mixture p-values (Self & Liang) + ACAT/Simes combination
    pvals = pvals_from_lrt_self_liang(lrts)
    var_p = pvals[variable_indices] if num_variable > 0 else pvals
    p_acat = cauchy_combination_p(var_p)

    sorted_p = np.sort(pvals)
    ranks = np.arange(1, L + 1)
    p_simes = float(np.min((L / ranks) * sorted_p))
    p_simes = max(1e-15, min(1.0, p_simes))

    sig_sites_05 = int(np.sum(pvals < 0.05))
    sig_sites_10 = int(np.sum(pvals < 0.10))
    total_selection_energy = float(np.sum(lrts))
    omnibus_lrt = float(np.sum(np.maximum(0.0, lrts - 3.841)))
    is_significant = bool(p_acat < 0.05 or pred_prob_pos > 0.50)

    return {
        "taxa": num_species,
        "sites": L,
        "p_value_acat": p_acat,
        "p_value_simes": p_simes,
        "omnibus_lrt": omnibus_lrt,
        "predicted_gene_lrt": pred_neural_lrt,
        "selection_probability": pred_prob_pos,
        "synonymous_rate_variation": pred_syn_var,
        "total_selection_energy": total_selection_energy,
        "sig_sites_p05": sig_sites_05,
        "sig_sites_p10": sig_sites_10,
        "rate_distributions": {
            "omega_1": float(pred_omega[0]), "proportion_1": float(pred_prop[0]),
            "omega_2": float(pred_omega[1]), "proportion_2": float(pred_prop[1]),
            "omega_3": float(pred_omega[2]), "proportion_3": float(pred_prop[2]),
        },
        "positive_selection_detected": is_significant,
        "elapsed_seconds": elapsed,
    }
