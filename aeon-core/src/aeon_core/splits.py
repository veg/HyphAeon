"""
aeon_core/splits.py
------------------
Cross-taxa attention extraction and fused affinity matrix computation.

These are model-attention utilities used by both hyphaeon (spectral splits)
and chronaeon (dating).
"""

from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch


from .inference import compute_live_adaptive_chunk_sizes


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

    Memory Safety:
    Uses live-budgeted safety chunking and streaming macro-batching over codon sites
    calibrated dynamically to live available accelerator VRAM and host RAM,
    guaranteeing bounded memory footprint without GPU pipeline stalls.

    Returns:
        mean_cross_attn: np.ndarray of shape (N, N)
        mean_taxa_repr: np.ndarray of shape (N, D)
    """
    model.eval()
    dev_obj = device if isinstance(device, torch.device) else torch.device(device)
    dev_str = dev_obj.type
    is_mps = "mps" in dev_str
    is_cuda = "cuda" in dev_str

    batch_size, num_species, window_size = msa_codons.shape
    central_idx = window_size // 2
    num_nodes = num_species + 1
    num_layers = max(1, len(model.row_layers))

    static_biases = tree_cache["static_phylo_biases"]
    static_coss = tree_cache["static_rope_coss"]
    static_sins = tree_cache["static_rope_sins"]

    num_heads = model.row_layers[0].num_heads if len(model.row_layers) > 0 else 8
    macro_batch_size, inner_chunk_size = compute_live_adaptive_chunk_sizes(
        dev_obj, num_nodes=num_nodes, num_heads=num_heads,
        embed_dim=model.embed_dim, window_size=window_size
    )

    accum_attn = torch.zeros((num_species, num_species), dtype=torch.float32, device=device)
    accum_taxa_repr = torch.zeros((num_species, model.embed_dim), dtype=torch.float32, device=device)

    with torch.no_grad():
        for b_start in range(0, batch_size, macro_batch_size):
            b_end = min(batch_size, b_start + macro_batch_size)
            m_b = b_end - b_start

            c_sub = msa_codons[b_start:b_end].to(device)
            a_sub = msa_aas[b_start:b_end].to(device)

            # 1. Embeddings for current macro-batch
            codon_emb = model.codon_embedding(c_sub)
            aa_emb = model.aa_embedding(a_sub)
            x = torch.cat([codon_emb, aa_emb], dim=-1)
            del codon_emb, aa_emb
            x = x + model.pos_embedding.unsqueeze(1)
            phylo_pos = tree_cache.get("mds_pos_static", tree_cache.get("static_phylo_pos"))
            if phylo_pos.dim() == 3:
                x = x + phylo_pos.unsqueeze(2)
            else:
                x = x + phylo_pos.unsqueeze(0).unsqueeze(2)

            root = model.root_token.expand(m_b, 1, window_size, -1)
            x_full = torch.cat([root, x], dim=1)
            del root, x

            # 2. Padding Mask for current macro-batch
            if padding_mask is not None:
                p_sub = padding_mask[b_start:b_end].to(device)
                root_mask = torch.zeros(m_b, 1, dtype=torch.bool, device=device)
                padding_mask_full = torch.cat([root_mask, p_sub], dim=1)
                padding_mask_dup = padding_mask_full.unsqueeze(1).expand(-1, window_size, -1).contiguous().view(m_b * window_size, num_nodes)
                del p_sub, root_mask, padding_mask_full
            else:
                padding_mask_dup = None

            # 3. Column Layers
            for i in range(len(model.col_layers)):
                col_in = x_full.reshape(m_b * num_nodes, window_size, model.embed_dim) if window_size > 1 else x_full.reshape(m_b * num_nodes, model.embed_dim)
                col_out = model.col_layers[i](col_in)
                x_full = col_out.reshape(m_b, num_nodes, window_size, model.embed_dim)

            # 4. Row Layers with Dynamic Chunked Attention Extraction
            x0_dup = x_full.transpose(1, 2).contiguous().view(m_b * window_size, num_nodes, model.embed_dim)
            sub_total_sites = m_b * window_size

            for i, layer in enumerate(model.row_layers):
                row_in = x_full.transpose(1, 2).contiguous().view(sub_total_sites, num_nodes, model.embed_dim)
                row_out_chunks = []

                for c_start in range(0, sub_total_sites, inner_chunk_size):
                    c_end = min(sub_total_sites, c_start + inner_chunk_size)
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
                    del q, k
                    scores.add_(static_biases[i])

                    if padding_mask_dup is not None:
                        p_chunk = padding_mask_dup[c_start:c_end].unsqueeze(1).unsqueeze(2)
                        scores.masked_fill_(p_chunk, -1e4)
                        attn_weights = torch.softmax(scores, dim=-1)
                        attn_weights = torch.where(p_chunk, torch.zeros_like(attn_weights), attn_weights)
                        del p_chunk
                    else:
                        attn_weights = torch.softmax(scores, dim=-1)
                    del scores

                    # Extract cross-taxa attention weights (taxa 1:N to taxa 1:N, omitting root at index 0)
                    cross_taxa = attn_weights[:, :, 1:, 1:]
                    accum_attn.add_(cross_taxa.mean(dim=1).sum(dim=0))
                    del cross_taxa

                    out = torch.matmul(attn_weights, v)
                    del attn_weights, v
                    out = out.transpose(1, 2).contiguous().view(n_c, num_nodes, layer.embed_dim)
                    out = layer.out_proj(out) + layer.alpha_skip * x0_dup[c_start:c_end]
                    r_out = model.row_norms[i](r_chunk + out)
                    del out
                    row_out_chunks.append(r_out)

                row_out = torch.cat(row_out_chunks, dim=0)
                del row_out_chunks
                x_full = row_out.reshape(m_b, window_size, num_nodes, model.embed_dim).transpose(1, 2)
                del row_out

            # Taxa embeddings across current macro-batch: [m_b, num_species, embed_dim]
            taxa_emb_site = x_full[:, 1:, central_idx, :]
            accum_taxa_repr.add_(taxa_emb_site.sum(dim=0))
            del x_full, x0_dup, c_sub, a_sub

            if is_mps:
                torch.mps.empty_cache()
            elif is_cuda:
                torch.cuda.empty_cache()

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

    # Embedding cosine similarity kernel with safe in-place normalization
    if taxa_repr is not None:
        norms = np.linalg.norm(taxa_repr, axis=1, keepdims=True)
        repr_norm = taxa_repr / np.maximum(norms, 1e-12)
        cos_sim = repr_norm @ repr_norm.T
        K_emb = np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0)
        np.fill_diagonal(K_emb, 0.0)
    else:
        K_emb = 1.0

    # Multiplicative tensor fusion
    A_fused = S_attn * K_mds * K_emb
    np.fill_diagonal(A_fused, 0.0)
    return A_fused

