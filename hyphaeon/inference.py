"""
hyphaeon/inference.py
---------------------
Shared inference utilities: model loading and transformer attribution extraction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .dataset import AA_MAP
from .model import PhyloAxialTransformer
from .utils import REV_AA_MAP, lrt_to_pvals, select_device
from .weights import load_arch_config, load_weights


def load_model(
    weights: str | None = None,
    variant: str | None = None,
    device: torch.device | None = None,
    *,
    cpu: bool = False,
) -> Tuple[PhyloAxialTransformer, torch.device]:
    """
    Load a PhyloAxialTransformer with pretrained weights.

    Returns (model, device). The model is in eval mode.
    """
    if device is None:
        device = select_device(cpu=cpu)

    config = load_arch_config(weights=weights, variant=variant)
    model = PhyloAxialTransformer(
        embed_dim=config["embed_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        window_size=config["window_size"],
    ).to(device)

    state_dict = load_weights(weights=weights, variant=variant, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, device


def compute_transformer_attributions(
    model: PhyloAxialTransformer,
    c_tensor: torch.Tensor,
    a_tensor: torch.Tensor,
    tree_cache: Dict[str, Any],
    taxa: List[str],
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Compute continuous Transformer Attribution Vectors directly from axial attention maps:
      a_{s, n} = alpha_{root->n, s} * delta_{s, n}
    where alpha is the multi-head root-to-leaf attention weight and delta is the
    non-consensus mutational indicator.

    Returns: (leaf_attributions [L, N], lrts [L], pvals [L], consensus_aas [L]).
    """
    L, n_taxa, _ = c_tensor.shape
    a_np = a_tensor.squeeze(-1).numpy()  # [L, N]

    # 1. Determine consensus amino acid per site
    consensus_aas: List[str] = []
    for s in range(L):
        valid = a_np[s][a_np[s] < 20]
        if len(valid) > 0:
            major_aa = int(np.argmax(np.bincount(valid)))
            consensus_aas.append(REV_AA_MAP.get(major_aa, "-"))
        else:
            consensus_aas.append("-")

    # 2. Mutational indicator matrix delta [L, N]
    delta = np.zeros((L, n_taxa), dtype=np.float32)
    for s in range(L):
        cons_tok = AA_MAP.get(consensus_aas[s], 20)
        if cons_tok < 20:
            for n in range(n_taxa):
                aa_val = a_np[s, n]
                if aa_val < 20 and aa_val != cons_tok:
                    delta[s, n] = 1.0

    # 3. Batched neural inference with root attention extraction
    lrts = np.zeros(L, dtype=np.float32)
    mean_attns = np.zeros((L, n_taxa), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start_idx in range(0, L, batch_size):
            end_idx = min(start_idx + batch_size, L)
            c_chunk = c_tensor[start_idx:end_idx].to(device)
            a_chunk = a_tensor[start_idx:end_idx].to(device)

            y_soft, _, root_attns = model.forward_cached(
                c_chunk, a_chunk, tree_cache, return_attentions=True
            )
            lrts[start_idx:end_idx] = torch.clamp(y_soft, min=0.0).cpu().numpy().flatten()
            mean_attns[start_idx:end_idx] = root_attns.cpu().numpy()

    # 4. Asymptotic p-values
    pvals = lrt_to_pvals(lrts)

    # 5. Continuous Leaf attributions: A = alpha * delta [L, N]
    leaf_attributions = mean_attns * delta

    return leaf_attributions, lrts, pvals, consensus_aas
