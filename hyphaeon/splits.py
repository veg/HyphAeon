"""
hyphaeon/splits.py
------------------
Spectral Graph Bisection via Cross-Taxa Attention Maps and Geometric MDS Fusion.

Recovers well-supported hierarchical phylogenetic splits (macro-clades and deep bipartitions)
by fusing:
1. Continuous 4D metric tree geometry from Multidimensional Scaling (MDS).
2. Discrete synapomorphic character state dynamics from HyphAeon's cross-taxa attention maps.
3. Latent contextual sequence representations from the PhyloAxialTransformer backbone.
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn.functional as F

from .inference import load_model, get_device
from .dataset import load_alignment_and_tree


def extract_cross_taxa_attentions_and_embeddings(
    model: torch.nn.Module,
    msa_codons: torch.Tensor,
    msa_aas: torch.Tensor,
    tree_cache: Dict[str, Any],
    padding_mask: Optional[torch.Tensor] = None,
    device: Union[str, torch.device] = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Executes a forward pass through the PhyloAxialTransformer, extracting:
    1. The full (N x N) pairwise cross-taxa attention matrix averaged across codon sites,
       attention heads, and transformer layers.
    2. The sequence-level latent embeddings Z in R^(N x D) pooled across all codon sites.

    Returns:
        mean_cross_attn: np.ndarray of shape (N, N)
        mean_taxa_repr: np.ndarray of shape (N, D)
    """
    model.eval()
    batch_size, num_species, window_size = msa_codons.shape
    central_idx = window_size // 2
    num_nodes = num_species + 1

    static_biases = tree_cache["static_phylo_biases"]
    static_coss = tree_cache["static_rope_coss"]
    static_sins = tree_cache["static_rope_sins"]

    # 1. Embeddings
    codon_emb = model.codon_embedding(msa_codons)
    aa_emb = model.aa_embedding(msa_aas)
    x = torch.cat([codon_emb, aa_emb], dim=-1)
    x = x + model.pos_embedding.unsqueeze(1)
    phylo_pos = tree_cache.get("mds_pos_static", tree_cache.get("static_phylo_pos"))
    if phylo_pos.dim() == 3:
        x = x + phylo_pos.unsqueeze(2)
    else:
        x = x + phylo_pos.unsqueeze(0).unsqueeze(2)

    root = model.root_token.expand(batch_size, 1, window_size, -1)
    x_full = torch.cat([root, x], dim=1)

    # 2. Padding Mask
    if padding_mask is not None:
        root_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=msa_codons.device)
        padding_mask_full = torch.cat([root_mask, padding_mask], dim=1)
        padding_mask_dup = padding_mask_full.unsqueeze(1).expand(-1, window_size, -1).contiguous().view(batch_size * window_size, num_nodes)
    else:
        padding_mask_dup = None

    # 3. Column Layers
    for i in range(len(model.col_layers)):
        col_in = x_full.reshape(batch_size * num_nodes, window_size, model.embed_dim) if window_size > 1 else x_full.reshape(batch_size * num_nodes, model.embed_dim)
        col_out = model.col_layers[i](col_in)
        x_full = col_out.reshape(batch_size, num_nodes, window_size, model.embed_dim)

    # 4. Row Layers with Full Cross-Taxa Attention Extraction
    x0_dup = x_full.transpose(1, 2).contiguous().view(batch_size * window_size, num_nodes, model.embed_dim)

    accum_attn = torch.zeros((num_species, num_species), dtype=torch.float32, device=device)
    accum_taxa_repr = torch.zeros((num_species, model.embed_dim), dtype=torch.float32, device=device)

    total_sites = batch_size * window_size
    if num_nodes > 2000:
        chunk_size = 2
    elif num_nodes > 1000:
        chunk_size = 8
    elif num_nodes > 500:
        chunk_size = 16
    elif total_sites > 32:
        chunk_size = 32
    else:
        chunk_size = total_sites

    with torch.no_grad():
        for i, layer in enumerate(model.row_layers):
            row_in = x_full.transpose(1, 2).contiguous().view(total_sites, num_nodes, model.embed_dim)
            row_out_chunks = []

            for c_start in range(0, total_sites, chunk_size):
                c_end = min(total_sites, c_start + chunk_size)
                r_chunk = row_in[c_start:c_end]
                n_c = c_end - c_start

                q = layer.q_proj(r_chunk).view(n_c, num_nodes, layer.num_heads, layer.head_dim).transpose(1, 2)
                k = layer.k_proj(r_chunk).view(n_c, num_nodes, layer.num_heads, layer.head_dim).transpose(1, 2)
                v = layer.v_proj(r_chunk).view(n_c, num_nodes, layer.num_heads, layer.head_dim).transpose(1, 2)

                cos, sin = static_coss[i], static_sins[i]
                half_dim = layer.head_dim // 2
                q1, q2 = q[..., :half_dim], q[..., half_dim:]
                k1, k2 = k[..., :half_dim], k[..., half_dim:]
                q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
                k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

                scores = torch.matmul(q, k.transpose(-2, -1)) / (layer.head_dim ** 0.5)
                scores = scores + static_biases[i]

                if padding_mask_dup is not None:
                    p_chunk = padding_mask_dup[c_start:c_end].unsqueeze(1).unsqueeze(2)
                    scores = scores.masked_fill(p_chunk, -1e4)
                    attn_weights = torch.softmax(scores, dim=-1)
                    attn_weights = torch.where(p_chunk, torch.zeros_like(attn_weights), attn_weights)
                else:
                    attn_weights = torch.softmax(scores, dim=-1)

                # Extract cross-taxa attention weights (taxa 1:N to taxa 1:N, omitting root at index 0)
                cross_taxa = attn_weights[:, :, 1:, 1:]
                accum_attn += cross_taxa.mean(dim=1).sum(dim=0)

                out = torch.matmul(attn_weights, v)
                out = out.transpose(1, 2).contiguous().view(n_c, num_nodes, layer.embed_dim)
                out = layer.out_proj(out) + layer.alpha_skip * x0_dup[c_start:c_end]
                r_out = model.row_norms[i](r_chunk + out)
                row_out_chunks.append(r_out)

                del q, k, v, scores, attn_weights, cross_taxa, out
                if "mps" in str(device).lower() and (c_start // chunk_size) % 10 == 0:
                    torch.mps.empty_cache()

            row_out = torch.cat(row_out_chunks, dim=0)
            x_full = row_out.reshape(batch_size, window_size, num_nodes, model.embed_dim).transpose(1, 2)

        # Taxa embeddings across sites: [batch_size, num_species, embed_dim]
        taxa_emb_site = x_full[:, 1:, central_idx, :]
        accum_taxa_repr += taxa_emb_site.sum(dim=0)

    num_layers = max(1, len(model.row_layers))
    mean_cross_attn = (accum_attn / (batch_size * num_layers)).cpu().numpy()
    mean_taxa_repr = (accum_taxa_repr / batch_size).cpu().numpy()

    return mean_cross_attn, mean_taxa_repr


def compute_fused_affinity_matrix(
    cross_attn_mat: np.ndarray,
    mds_coords: np.ndarray,
    taxa_repr: Optional[np.ndarray] = None,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Fuses cross-taxa attention affinities with MDS geometric distances and latent embedding similarities.

    A_fused = S_attn ⊙ K_mds ⊙ K_emb
    """
    # Symmetrize raw cross-taxa attention
    S_attn = (cross_attn_mat + cross_attn_mat.T) / 2.0
    np.fill_diagonal(S_attn, 0.0)

    # MDS geometric Gaussian kernel
    d_mds = squareform(pdist(mds_coords))
    sigma_mds = np.median(d_mds[d_mds > 0]) if np.any(d_mds > 0) else 1.0
    K_mds = np.exp(- (d_mds ** 2) / (2 * (sigma_mds ** 2)))
    np.fill_diagonal(K_mds, 0.0)

    # Embedding cosine similarity kernel
    if taxa_repr is not None:
        norms = np.linalg.norm(taxa_repr, axis=1, keepdims=True)
        cos_sim = (taxa_repr @ taxa_repr.T) / (norms @ norms.T + 1e-8)
        K_emb = np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0)
        np.fill_diagonal(K_emb, 0.0)
    else:
        K_emb = 1.0

    # Multiplicative tensor fusion
    A_fused = S_attn * K_mds * K_emb
    np.fill_diagonal(A_fused, 0.0)
    return A_fused


def spectral_bisection(
    affinity_matrix: np.ndarray,
    taxa_names: List[str],
    min_clade_size: int = 2,
    max_depth: int = 10,
    current_depth: int = 0
) -> Dict[str, Any]:
    """
    Recursively partitions taxa using the Fiedler vector of the normalized graph Laplacian.

    Args:
        affinity_matrix: (N x N) symmetric non-negative affinity matrix.
        taxa_names: List of N taxon identifiers.
        min_clade_size: Minimum number of taxa in a clade before terminating bisection.
        max_depth: Maximum recursion depth.

    Returns:
        Hierarchical tree dictionary with node attributes:
          - fiedler_val: 2nd smallest eigenvalue (algebraic connectivity).
          - eigengap: lambda_3 - lambda_2 (split stability margin).
          - cut_weight: sum of inter-clade edge weights.
          - left, right: child clade nodes.
    """
    n = len(taxa_names)
    if n <= min_clade_size or current_depth >= max_depth:
        return {"type": "leaf", "taxa": taxa_names}

    A = (affinity_matrix + affinity_matrix.T) / 2.0
    np.fill_diagonal(A, 0.0)

    d = A.sum(axis=1)
    d[d == 0] = 1e-8
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D_inv_sqrt = np.diag(d_inv_sqrt)

    # L_sym = I - D^(-1/2) A D^(-1/2)
    L_sym = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    evals, evecs = np.linalg.eigh(L_sym)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    fiedler_val = float(evals[1]) if n > 1 else 0.0
    fiedler_vec = evecs[:, 1] if n > 1 else np.zeros(n)
    eigengap = float(evals[2] - evals[1]) if n > 2 else float(evals[1]) if n > 1 else 0.0

    # Unnormalized indicator y = D^(-1/2) v_2
    y = d_inv_sqrt * fiedler_vec

    left_mask = (y >= 0)
    right_mask = ~left_mask

    if left_mask.sum() == 0 or right_mask.sum() == 0:
        median_val = np.median(y)
        left_mask = (y >= median_val)
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            left_mask[:n // 2] = True
            right_mask = ~left_mask

    left_taxa = [taxa_names[i] for i in range(n) if left_mask[i]]
    right_taxa = [taxa_names[i] for i in range(n) if right_mask[i]]

    cut_weight = float(A[left_mask][:, right_mask].sum())

    A_left = A[np.ix_(left_mask, left_mask)]
    A_right = A[np.ix_(right_mask, right_mask)]

    left_child = spectral_bisection(A_left, left_taxa, min_clade_size, max_depth, current_depth + 1)
    right_child = spectral_bisection(A_right, right_taxa, min_clade_size, max_depth, current_depth + 1)

    return {
        "type": "node",
        "fiedler_val": fiedler_val,
        "eigengap": eigengap,
        "cut_weight": cut_weight,
        "depth": current_depth,
        "taxa_count": n,
        "left": left_child,
        "right": right_child
    }


def tree_dict_to_newick(tree_dict: Dict[str, Any]) -> str:
    """Converts a hierarchical bisection tree dictionary into a formatted Newick string."""
    if tree_dict["type"] == "leaf":
        if len(tree_dict["taxa"]) == 1:
            return tree_dict["taxa"][0]
        else:
            return "(" + ",".join(tree_dict["taxa"]) + ")"
    left_nwk = tree_dict_to_newick(tree_dict["left"])
    right_nwk = tree_dict_to_newick(tree_dict["right"])
    support = tree_dict.get("eigengap", 0.0)
    return f"({left_nwk},{right_nwk}):{support:.4f}"


def get_all_clade_taxa(node: Dict[str, Any]) -> List[str]:
    """Helper to collect all leaf taxa beneath a tree node."""
    if node["type"] == "leaf":
        return node["taxa"]
    return get_all_clade_taxa(node["left"]) + get_all_clade_taxa(node["right"])


def run_spectral_splits(
    alignment_path: str,
    tree_path: Optional[str] = None,
    use_tn93: bool = False,
    weights_path: Optional[str] = None,
    min_clade_size: int = 2,
    max_depth: int = 10,
    device: Optional[Union[str, torch.device]] = None
) -> Dict[str, Any]:
    """
    End-to-end pipeline to recover well-supported phylogenetic splits via spectral bisection.

    Args:
        alignment_path: Path to FASTA alignment.
        tree_path: Optional path to Newick tree.
        use_tn93: If True, computes pairwise TN93 distance matrix directly (skips tree).
        weights_path: Path to HyphAeon pretrained weights.
        min_clade_size: Clade size floor.
        max_depth: Max tree depth.
        device: PyTorch device.

    Returns:
        Dictionary with derived Newick string, root split clades, Fiedler value, eigengap, and tree dict.
    """
    if device is None:
        device = get_device(cpu=True)

    if weights_path is None:
        import os
        candidates = [
            os.path.join("weights", "hyphaeon_v1.pt"),
            os.path.join("weights", "axomeme_v1.pt"),
            "model.safetensors",
        ]
        for c in candidates:
            if os.path.exists(c):
                weights_path = c
                break
        if weights_path is None:
            try:
                from .weights import resolve_weights_path
                weights_path = resolve_weights_path(None)
            except Exception:
                weights_path = "model.safetensors"

    model = load_model(weights_path, device=device)

    c_tensor, a_tensor, d_tensor, z_tensor, _, taxa, L = load_alignment_and_tree(
        alignment_path, nwk_path=tree_path, use_tn93=use_tn93, prune_duplicates=False
    )

    msa_codons = c_tensor.to(device)
    msa_aas = a_tensor.to(device)
    dist_mat = d_tensor.squeeze(0).cpu().numpy()
    mds_coords = z_tensor.squeeze(0).cpu().numpy()

    tree_cache = model.precompute_tree_cache(dist_mat, mds_coords)

    cross_attn, taxa_repr = extract_cross_taxa_attentions_and_embeddings(
        model, msa_codons, msa_aas, tree_cache, device=device
    )

    A_fused = compute_fused_affinity_matrix(cross_attn, mds_coords, taxa_repr)
    split_tree = spectral_bisection(A_fused, taxa, min_clade_size=min_clade_size, max_depth=max_depth)
    newick_str = tree_dict_to_newick(split_tree) + ";"

    left_taxa = get_all_clade_taxa(split_tree["left"])
    right_taxa = get_all_clade_taxa(split_tree["right"])

    return {
        "newick": newick_str,
        "fiedler_val": split_tree.get("fiedler_val", 0.0),
        "eigengap": split_tree.get("eigengap", 0.0),
        "cut_weight": split_tree.get("cut_weight", 0.0),
        "root_split": {
            "left_clade": left_taxa,
            "right_clade": right_taxa,
            "left_count": len(left_taxa),
            "right_count": len(right_taxa)
        },
        "taxa": taxa,
        "L": L,
        "tree_dict": split_tree
    }
