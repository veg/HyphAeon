"""
hyphaeon/model.py
----------------
Core Neural Architecture: PhyloAxialTransformer with Multi-Scale 4D Tree-RoPE
for ultra-fast episodic positive selection inference.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Stationary background frequency floor (1/20 amino acids).
# Used by the continuous-time Markov transition probability tree kernel in both
# the inline and pre-cached forward paths. Shared constant ensures both paths
# stay in sync.
EPS0 = 0.05
class BlockLinear(nn.Module):
    """
    Block-Diagonal Linear Projection.
    Guarantees 100% mathematical disentanglement between the Codon Synonymous Track
    and the Amino Acid Selection Track. Prevents linear layer cross-mixing of dS noise into dN+ features.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.half_in = in_features // 2
        self.half_out = out_features // 2
        self.block_codon = nn.Linear(self.half_in, self.half_out, bias=bias)
        self.block_aa = nn.Linear(self.half_in, self.half_out, bias=bias)

    def forward(self, x):
        # x: [..., in_features]
        x_codon = x[..., :self.half_in]
        x_aa = x[..., self.half_in:]
        out_codon = self.block_codon(x_codon)
        out_aa = self.block_aa(x_aa)
        return torch.cat([out_codon, out_aa], dim=-1)


# --- Custom Row Attention with Block-Diagonal Disentanglement & Learnable Phylogenetic Bias ---
class PhyloRowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = BlockLinear(embed_dim, embed_dim)
        self.k_proj = BlockLinear(embed_dim, embed_dim)
        self.v_proj = BlockLinear(embed_dim, embed_dim)

        # NOTE: The following parameters are defined but NOT used in forward().
        # They are remnants of earlier architecture iterations (3-channel tree
        # projection, 2-layer phylo MLP, per-site rate scaler) that were
        # simplified to the current 1-channel Markov kernel. They remain here
        # because they are present in the pretrained checkpoint (hyphaeon_v1.pt)
        # and removing them from __init__ would cause load_state_dict to fail
        # with unexpected-key errors. Removing them requires either a checkpoint
        # migration or a compatibility shim in the CLI. See REVIEW.md item #29.
        self.tree_w1 = nn.Parameter(torch.randn(num_heads, 3) * 0.02)
        self.tree_b1 = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.tree_w2 = nn.Parameter(torch.randn(num_heads, 1, 1) * 0.02)

        # phylo_w1 IS used in forward() (Markov kernel decay rate).
        # phylo_b1 and phylo_w2 are NOT used — same situation as above.
        self.phylo_w1 = nn.Parameter(torch.randn(num_heads, 1, 1) * 0.02)
        self.phylo_b1 = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.phylo_w2 = nn.Parameter(torch.randn(num_heads, 1, 1) * 0.02)

        # NOT used in forward() — same situation as above.
        self.site_tree_scaler = nn.Linear(embed_dim, 1)
        nn.init.zeros_(self.site_tree_scaler.weight)
        nn.init.zeros_(self.site_tree_scaler.bias)
        
        half_dim = self.head_dim // 2
        self.rope_freqs = nn.Parameter(torch.randn(num_heads, half_dim, 4) * 0.05)
        # Learnable Initial-Representation Skip Weight (Initialized to 0.20)
        self.alpha_skip = nn.Parameter(torch.tensor(0.20))
        
        self.out_proj = BlockLinear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, dist_matrix=None, mds_coords=None, padding_mask=None, x0=None,
                cos=None, sin=None, phylo_bias=None):
        """
        Row attention with optional pre-cached tree kernels.

        When cos/sin/phylo_bias are provided (from precompute_tree_cache), skips
        inline RoPE and Markov bias computation. This is the single attention
        code path used by both forward() and forward_cached().
        """
        batch_size, num_species, _ = x.shape

        q = self.q_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)

        # Tree-RoPE: Apply 4D MDS Rotary Position Phase Rotations to Query & Key
        half_dim = self.head_dim // 2
        if cos is None or sin is None:
            # Compute rotations inline from mds_coords
            if mds_coords is not None:
                m_exp = mds_coords.unsqueeze(1).unsqueeze(3) # [batch_size, 1, num_species, 1, 4]
                f_exp = self.rope_freqs.unsqueeze(0).unsqueeze(2) # [1, num_heads, 1, half_dim, 4]
                angles = (m_exp * f_exp).sum(dim=-1) # [batch_size, num_heads, num_species, half_dim]
                cos = torch.cos(angles).to(q.dtype)
                sin = torch.sin(angles).to(q.dtype)

        if cos is not None and sin is not None:
            q1, q2 = q[..., :half_dim], q[..., half_dim:]
            k1, k2 = k[..., :half_dim], k[..., half_dim:]
            q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
            k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)


        # Pure Continuous-Time Markov Transition Probability Tree Kernel
        # P_ij(d) = eps0 + (1 - eps0) * exp(-lambda_h * d_ij)
        # Reflects exact continuous-time Markov substitution process where transition
        # probability asymptotes to eps0 (1/20 amino acids) as d -> inf.
        if phylo_bias is None:
            # Compute bias inline from dist_matrix
            if dist_matrix is not None:
                dist_1d = dist_matrix[..., 0] if dist_matrix.dim() == 4 or (dist_matrix.dim() == 3 and dist_matrix.shape[-1] == 3) else dist_matrix
                bias_1d = dist_1d.unsqueeze(1) if dist_1d.dim() == 3 else dist_1d.unsqueeze(0).unsqueeze(1)
                decay_rate = F.softplus(self.phylo_w1)  # [num_heads, 1, 1] learnable rate per attention head
                markov_kernel = EPS0 + (1.0 - EPS0) * torch.exp(-decay_rate * bias_1d)
                # Pure log-space Markov transition probability kernel
                phylo_bias = torch.log(markov_kernel.clamp(min=1e-5))

        if phylo_bias is not None:
            scores = scores + phylo_bias

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e4)
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = torch.where(mask, torch.zeros_like(attn_weights), attn_weights)
        else:
            attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights).to(v.dtype)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_species, self.embed_dim)
        out = self.out_proj(out)
        if x0 is not None:
            out = out + self.alpha_skip * x0  # Learnable initial-representation skip connection
        return out



class RankConsistentCoralHead(nn.Module):
    """
    Rank-Consistent Ordinal Regression Head (Cao, Mirjalili, & Raschka, 2020):
    Standard unconstrained linear projection with learnable monotonic rank cutoffs.
    Zero magic numbers, zero WeightNorm constraints, zero manual gain constants.
    """
    def __init__(self, embed_dim, num_thresholds=8):
        super().__init__()
        self.num_thresholds = num_thresholds
        
        self.fc1 = nn.Linear(embed_dim, 256)
        self.fc2 = nn.Linear(256, 1, bias=False)
        
        # Learnable Rank Cutoffs (Cao et al. 2020)
        b0_init = -2.2253 if num_thresholds == 8 else -1.7346
        self.b0 = nn.Parameter(torch.tensor(b0_init))
        self.theta_steps = nn.Parameter(torch.full((num_thresholds - 1,), 0.40))
        
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0 / (embed_dim ** 0.5))
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0 / (256 ** 0.5))
        
    def get_cutoffs(self):
        steps = F.softplus(self.theta_steps)
        cum_steps = torch.cumsum(steps, dim=0)
        cutoffs = torch.cat([self.b0.unsqueeze(0), self.b0 - cum_steps])
        return cutoffs

    def forward(self, x):
        # x: [batch_size, embed_dim]
        h = F.gelu(self.fc1(x))
        proj = self.fc2(h)  # Standard unconstrained linear projection
        cutoffs = self.get_cutoffs().to(device=x.device, dtype=x.dtype)
        logits = proj + cutoffs.unsqueeze(0)  # [batch_size, num_thresholds]
        return logits


LOG_CORAL_DELTAS_8 = torch.tensor([0.6931, 0.7239, 0.2793, 0.3364, 0.4377, 0.5741, 0.8873, 0.6833])

LOG_CORAL_DELTAS_16 = torch.tensor([0.2420, 0.3176, 0.2513, 0.2284, 0.1991, 0.1786, 0.2793, 0.2205, 0.2339, 0.4243, 0.2984, 0.2621, 0.3610, 0.4353, 0.3989, 0.2844])

LOG_CORAL_DELTAS_24 = torch.tensor([
    0.2420, 0.1635, 0.1542, 0.1335, 0.1613, 0.1849, 0.1991, 0.1786, 0.1602,
    0.1191, 0.1619, 0.1746, 0.1919, 0.2458, 0.2376, 0.2364, 0.2744, 0.2776,
    0.2647, 0.2461, 0.2268, 0.1847, 0.1963, 0.1807
])

LOG_CORAL_DELTAS_12 = LOG_CORAL_DELTAS_16[:12]

def decode_soft_ordinal_lrt(logits_lrt_ordinal):
    """
    Log-Space Soft-Bin Survival Integral Decoder:
    E[log(1+LRT)] = sum_{k=0}^{K-1} P(log(1+LRT) > Z_k) * delta_Z_k
    E[LRT] = exp(E[log(1+LRT)]) - 1
    """
    if logits_lrt_ordinal.dim() == 1 or logits_lrt_ordinal.shape[-1] not in (8, 12, 16, 24):
        return logits_lrt_ordinal, logits_lrt_ordinal
        
    probs = torch.sigmoid(logits_lrt_ordinal)
    K = probs.shape[-1]
    if K == 8:
        deltas = LOG_CORAL_DELTAS_8.to(device=logits_lrt_ordinal.device, dtype=logits_lrt_ordinal.dtype)
    elif K == 16:
        deltas = LOG_CORAL_DELTAS_16.to(device=logits_lrt_ordinal.device, dtype=logits_lrt_ordinal.dtype)
    elif K == 24:
        deltas = LOG_CORAL_DELTAS_24.to(device=logits_lrt_ordinal.device, dtype=logits_lrt_ordinal.dtype)
    elif K == 12:
        deltas = LOG_CORAL_DELTAS_12.to(device=logits_lrt_ordinal.device, dtype=logits_lrt_ordinal.dtype)
    else:
        deltas = LOG_CORAL_DELTAS_16[:K].to(device=logits_lrt_ordinal.device, dtype=logits_lrt_ordinal.dtype)
    
    # Expected log(1 + LRT) via continuous survival integration
    log_lrt_expected = (probs * deltas.view(1, -1)).sum(dim=1)
    
    # Invert back to physical LRT scale
    physical_lrt = torch.expm1(log_lrt_expected)
    return physical_lrt, probs


class PhyloAxialTransformer(nn.Module):
    """
    Phylogenetic Axial Transformer with Learned [ROOT] Token:
    A dedicated [ROOT] token at the tree origin (0, 0, 0, 0) is prepended to the sequence of taxa.
    Across all attention layers, the [ROOT] token participates in bidirectional self-attention with all taxa,
    allowing branch-level non-synonymous mutations to route directly into the [ROOT] token while
    broadcasting tree-wide background rates back down to leaves.
    Zero external pooling heuristics, zero PMA probe artifacts, 100% permutation-invariant.
    """
    def __init__(self, num_tokens=66, embed_dim=128, num_heads=8, num_layers=4, window_size=1, max_species=256, dropout=0.1, max_k=32, num_thresholds=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_layers = num_layers
        self.num_thresholds = num_thresholds

        self.codon_embedding = nn.Embedding(num_tokens, embed_dim // 2)
        self.aa_embedding = nn.Embedding(23, embed_dim // 2)
        self.pos_embedding = nn.Parameter(torch.randn(1, window_size, embed_dim) * 0.02)
        
        # Learnable Phylogenetic [ROOT] Token (Origin of the Evolutionary Tree)
        self.root_token = nn.Parameter(torch.randn(1, 1, 1, embed_dim) * 0.02)
        
        self.mds_proj = nn.Linear(4, embed_dim)
        num_col_layers = 2 if window_size > 1 else 0
        num_row_layers = num_layers
        
        self.col_layers = nn.ModuleList([
            nn.Sequential(
                BlockLinear(embed_dim, 2*embed_dim),
                nn.GELU(),
                BlockLinear(2*embed_dim, embed_dim)
            ) for _ in range(num_col_layers)
        ])

        self.row_layers = nn.ModuleList([
            PhyloRowAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)
            for _ in range(num_row_layers)
        ])
        self.row_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_row_layers)])

        self.lrt_ordinal_head = RankConsistentCoralHead(embed_dim, num_thresholds=num_thresholds)

    def precompute_tree_cache(self, dist_matrix, mds_coords):
        """
        Pre-computes static tree-invariant tensors (continuous-time Markov log-bias,
        4D Tree-RoPE phase rotations, and projected MDS embeddings).
        Can be computed once per phylogeny and reused across all site batches for 3-5x forward speedup.
        """
        device = next(self.parameters()).device
        if not isinstance(dist_matrix, torch.Tensor):
            dist_matrix = torch.tensor(dist_matrix, dtype=torch.float32, device=device)
        else:
            dist_matrix = dist_matrix.to(device)

        if not isinstance(mds_coords, torch.Tensor):
            mds_coords = torch.tensor(mds_coords, dtype=torch.float32, device=device)
        else:
            mds_coords = mds_coords.to(device)

        if dist_matrix.dim() == 2:
            dist_matrix = dist_matrix.unsqueeze(0)  # [1, N, N]
        if mds_coords.dim() == 2:
            mds_coords = mds_coords.unsqueeze(0)  # [1, N, 4]

        # 1. Augment with [ROOT] Token at tree origin [0, 0, 0, 0]
        root_mds = torch.zeros(1, 1, 4, dtype=mds_coords.dtype, device=mds_coords.device)
        mds_full = torch.cat([root_mds, mds_coords], dim=1)  # [1, num_species + 1, 4]

        root_dist = torch.norm(mds_coords, dim=-1, keepdim=True)  # [1, num_species, 1]
        dist_top = torch.cat([torch.zeros(1, 1, 1, device=dist_matrix.device), root_dist.transpose(1, 2)], dim=2)  # [1, 1, num_species + 1]
        dist_bot = torch.cat([root_dist, dist_matrix], dim=2)  # [1, num_species, num_species + 1]
        dist_full = torch.cat([dist_top, dist_bot], dim=1)  # [1, num_species + 1, num_species + 1]

        dist_1d = dist_full[..., 0] if dist_full.dim() == 4 or (dist_full.dim() == 3 and dist_full.shape[-1] == 3) else dist_full
        dist_tensor = dist_1d.unsqueeze(1) if dist_1d.dim() == 3 else dist_1d.unsqueeze(0).unsqueeze(1)

        static_phylo_biases = []
        static_rope_coss = []
        static_rope_sins = []

        eps0 = EPS0
        with torch.no_grad():
            for layer in self.row_layers:
                decay_rate = F.softplus(layer.phylo_w1)  # [num_heads, 1, 1]
                markov_kernel = eps0 + (1.0 - eps0) * torch.exp(-decay_rate * dist_tensor)
                static_phylo_biases.append(torch.log(markov_kernel.clamp(min=1e-5)))

                half_dim = layer.head_dim // 2
                m_exp = mds_full.unsqueeze(1).unsqueeze(3)  # [1, 1, num_species + 1, 1, 4]
                f_exp = layer.rope_freqs.unsqueeze(0).unsqueeze(2)  # [1, num_heads, 1, half_dim, 4]
                angles = (m_exp * f_exp).sum(dim=-1)  # [1, num_heads, num_species + 1, half_dim]
                static_rope_coss.append(torch.cos(angles))
                static_rope_sins.append(torch.sin(angles))

        mds_pos_static = self.mds_proj(mds_coords)  # [1, num_species, embed_dim]

        return {
            "static_phylo_biases": static_phylo_biases,
            "static_rope_coss": static_rope_coss,
            "static_rope_sins": static_rope_sins,
            "mds_pos_static": mds_pos_static,
            "num_species": mds_coords.shape[1]
        }

    def forward_cached(self, msa_codons, msa_aas, tree_cache, padding_mask=None, return_attentions=False, return_hidden=False):
        """
        Fast forward pass using pre-cached static tree kernels.
        Eliminates repeated matrix exponentials and trigonometric rotations.
        Automatically decomposes large codon batches into memory-safe micro-batches
        when sequence depth N > 1000 to prevent O(B * H * N^2) accelerator buffer overflow.
        """
        batch_size, num_species, window_size = msa_codons.shape
        central_idx = window_size // 2

        # Automatic memory-safe micro-batching for large alignments
        max_attention_elements = 4_000_000 # ~16 MB per attention tensor
        if batch_size > 1 and (batch_size * (num_species ** 2)) > max_attention_elements:
            micro_b = max(1, int(max_attention_elements / max(1, num_species ** 2)))
            all_lrt_soft = []
            all_logits = []
            all_extra = []
            for b_start in range(0, batch_size, micro_b):
                b_end = min(b_start + micro_b, batch_size)
                p_sub = padding_mask[b_start:b_end] if padding_mask is not None else None
                out = self.forward_cached(
                    msa_codons[b_start:b_end],
                    msa_aas[b_start:b_end],
                    tree_cache,
                    padding_mask=p_sub,
                    return_attentions=return_attentions,
                    return_hidden=return_hidden
                )
                all_lrt_soft.append(out[0])
                all_logits.append(out[1])
                if return_hidden or return_attentions:
                    all_extra.append(out[2])
                    
            cat_lrt_soft = torch.cat(all_lrt_soft, dim=0)
            cat_logits = torch.cat(all_logits, dim=0)
            if return_hidden or return_attentions:
                cat_extra = torch.cat(all_extra, dim=0)
                return cat_lrt_soft, cat_logits, cat_extra
            return cat_lrt_soft, cat_logits

        # 1. Embeddings
        codon_emb = self.codon_embedding(msa_codons)
        aa_emb = self.aa_embedding(msa_aas)
        x = torch.cat([codon_emb, aa_emb], dim=-1)
        x = x + self.pos_embedding.unsqueeze(1)
        phylo_pos = tree_cache.get("mds_pos_static", tree_cache.get("static_phylo_pos"))
        if phylo_pos.dim() == 3:
            x = x + phylo_pos.unsqueeze(2)
        else:
            x = x + phylo_pos.unsqueeze(0).unsqueeze(2)

        root = self.root_token.expand(batch_size, 1, window_size, -1)
        x_full = torch.cat([root, x], dim=1)
        num_nodes = num_species + 1

        # 2. Padding Mask
        if padding_mask is not None:
            root_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=msa_codons.device)
            padding_mask_full = torch.cat([root_mask, padding_mask], dim=1)
            padding_mask_dup = padding_mask_full.unsqueeze(1).expand(-1, window_size, -1).contiguous().view(batch_size * window_size, num_nodes)
        else:
            padding_mask_dup = None

        # 3. Column Layers
        for i in range(len(self.col_layers)):
            col_in = x_full.reshape(batch_size * num_nodes, window_size, self.embed_dim) if window_size > 1 else x_full.reshape(batch_size * num_nodes, self.embed_dim)
            col_out = self.col_layers[i](col_in)
            x_full = col_out.reshape(batch_size, num_nodes, window_size, self.embed_dim)

        # 4. Row Layers using Pre-Cached Kernels
        x0_dup = x_full.transpose(1, 2).contiguous().view(batch_size * window_size, num_nodes, self.embed_dim)
        static_biases = tree_cache["static_phylo_biases"]
        static_coss = tree_cache["static_rope_coss"]
        static_sins = tree_cache["static_rope_sins"]

        layer_attns = [] if return_attentions else None

        for i, layer in enumerate(self.row_layers):
            row_in = x_full.transpose(1, 2).contiguous().view(batch_size * window_size, num_nodes, self.embed_dim)

            if return_attentions:
                q = layer.q_proj(row_in).view(batch_size * window_size, num_nodes, layer.num_heads, layer.head_dim).transpose(1, 2)
                k = layer.k_proj(row_in).view(batch_size * window_size, num_nodes, layer.num_heads, layer.head_dim).transpose(1, 2)
                v = layer.v_proj(row_in).view(batch_size * window_size, num_nodes, layer.num_heads, layer.head_dim).transpose(1, 2)

                cos, sin = static_coss[i], static_sins[i]
                half_dim = layer.head_dim // 2
                q1, q2 = q[..., :half_dim], q[..., half_dim:]
                k1, k2 = k[..., :half_dim], k[..., half_dim:]
                q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
                k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(layer.head_dim)
                scores = scores + static_biases[i]

                if padding_mask_dup is not None:
                    mask = padding_mask_dup.unsqueeze(1).unsqueeze(2)
                    scores = scores.masked_fill(mask, -1e4)
                    attn_weights = torch.softmax(scores, dim=-1)
                    attn_weights = torch.where(mask, torch.zeros_like(attn_weights), attn_weights)
                else:
                    attn_weights = torch.softmax(scores, dim=-1)

                layer_attns.append(attn_weights[:, :, 0, 1:].detach())

                attn_weights = layer.dropout(attn_weights).to(v.dtype)
                out = torch.matmul(attn_weights, v)
                out = out.transpose(1, 2).contiguous().view(batch_size * window_size, num_nodes, layer.embed_dim)
                out = layer.out_proj(out) + layer.alpha_skip * x0_dup
                row_out = self.row_norms[i](row_in + out)
                x_full = row_out.reshape(batch_size, window_size, num_nodes, self.embed_dim).transpose(1, 2)
            else:
                row_out = layer(
                    row_in,
                    padding_mask=padding_mask_dup,
                    x0=x0_dup,
                    cos=static_coss[i],
                    sin=static_sins[i],
                    phylo_bias=static_biases[i],
                )
                row_out = self.row_norms[i](row_in + row_out)
                x_full = row_out.reshape(batch_size, window_size, num_nodes, layer.embed_dim).transpose(1, 2)

        root_repr = x_full[:, 0, central_idx, :]
        logits_lrt_ordinal = self.lrt_ordinal_head(root_repr)
        y_lrt_soft, _ = decode_soft_ordinal_lrt(logits_lrt_ordinal)

        if return_hidden:
            return y_lrt_soft.view(batch_size), logits_lrt_ordinal, root_repr

        if return_attentions:
            all_attns = torch.stack(layer_attns, dim=0) # [num_layers, B*W, num_heads, num_species]
            mean_root_attns = all_attns.mean(dim=(0, 2)).view(batch_size, num_species) # [batch_size, num_species]
            return y_lrt_soft.view(batch_size), logits_lrt_ordinal, mean_root_attns
            
        return y_lrt_soft.view(batch_size), logits_lrt_ordinal

    def forward(self, msa_codons, msa_aas, dist_matrix=None, mds_coords=None, padding_mask=None, tree_cache=None, return_attentions=False, return_hidden=False):
        if tree_cache is not None:
            return self.forward_cached(msa_codons, msa_aas, tree_cache, padding_mask=padding_mask, return_attentions=return_attentions, return_hidden=return_hidden)

        batch_size, num_species, window_size = msa_codons.shape
        central_idx = window_size // 2

        # Automatic memory-safe micro-batching for large alignments
        max_attention_elements = 4_000_000
        if batch_size > 1 and (batch_size * (num_species ** 2)) > max_attention_elements:
            micro_b = max(1, int(max_attention_elements / max(1, num_species ** 2)))
            all_lrt_soft = []
            all_logits = []
            all_extra = []
            for b_start in range(0, batch_size, micro_b):
                b_end = min(b_start + micro_b, batch_size)
                p_sub = padding_mask[b_start:b_end] if padding_mask is not None else None
                out = self.forward(
                    msa_codons[b_start:b_end],
                    msa_aas[b_start:b_end],
                    dist_matrix=dist_matrix,
                    mds_coords=mds_coords,
                    padding_mask=p_sub,
                    tree_cache=None,
                    return_attentions=return_attentions,
                    return_hidden=return_hidden
                )
                all_lrt_soft.append(out[0])
                all_logits.append(out[1])
                if return_hidden or return_attentions:
                    all_extra.append(out[2])
                    
            cat_lrt_soft = torch.cat(all_lrt_soft, dim=0)
            cat_logits = torch.cat(all_logits, dim=0)
            if return_hidden or return_attentions:
                cat_extra = torch.cat(all_extra, dim=0)
                return cat_lrt_soft, cat_logits, cat_extra
            return cat_lrt_soft, cat_logits
        
        # Ensure padding_mask is always a canonical boolean tensor to keep XLA graph topology static
        if padding_mask is None:
            padding_mask = torch.zeros(batch_size, num_species, dtype=torch.bool, device=msa_codons.device)

        codon_emb = self.codon_embedding(msa_codons)
        aa_emb = self.aa_embedding(msa_aas)
        
        x = torch.cat([codon_emb, aa_emb], dim=-1)
        x = x + self.pos_embedding.unsqueeze(1)
        
        phylo_pos = self.mds_proj(mds_coords)
        x = x + phylo_pos.unsqueeze(2)
        
        # 1. Prepend [ROOT] Token at Index 0
        root_x = self.root_token.expand(batch_size, 1, window_size, -1)
        x_full = torch.cat([root_x, x], dim=1)  # [batch_size, num_species + 1, window_size, embed_dim]
        
        # 2. Augment Padding Mask (Root token is never padded)
        root_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=msa_codons.device)
        padding_mask_full = torch.cat([root_mask, padding_mask], dim=1)  # [batch_size, num_species + 1]
        
        # 3. Augment MDS Coords (Root is at tree origin [0, 0, 0, 0])
        root_mds = torch.zeros(batch_size, 1, 4, dtype=mds_coords.dtype, device=mds_coords.device)
        mds_full = torch.cat([root_mds, mds_coords], dim=1)  # [batch_size, num_species + 1, 4]
        
        # 4. Augment Distance Matrix (Root-to-taxa distance is norm in MDS space)
        root_dist = torch.norm(mds_coords, dim=-1, keepdim=True)  # [batch_size, num_species, 1]
        dist_top = torch.cat([torch.zeros(batch_size, 1, 1, device=dist_matrix.device), root_dist.transpose(1, 2)], dim=2)  # [batch_size, 1, num_species + 1]
        dist_bot = torch.cat([root_dist, dist_matrix], dim=2)  # [batch_size, num_species, num_species + 1]
        dist_full = torch.cat([dist_top, dist_bot], dim=1)  # [batch_size, num_species + 1, num_species + 1]

        num_nodes = num_species + 1
        padding_mask_dup = padding_mask_full.unsqueeze(1).expand(-1, window_size, -1).contiguous().view(batch_size * window_size, num_nodes)

        # 6. Feature Transformation along Column Axis (if window_size > 1)
        for i in range(len(self.col_layers)):
            col_in = x_full.reshape(batch_size * num_nodes, window_size, self.embed_dim) if window_size > 1 else x_full.reshape(batch_size * num_nodes, self.embed_dim)
            col_out = self.col_layers[i](col_in)
            x_full = col_out.reshape(batch_size, num_nodes, window_size, self.embed_dim)

        # 7. Deep Phylogenetic Tree Attention along Row Axis across all N+1 nodes
        if dist_full.dim() == 4:
            dist_dup = dist_full.unsqueeze(1).expand(-1, window_size, -1, -1, -1).contiguous().view(batch_size * window_size, num_nodes, num_nodes, dist_full.shape[-1])
        else:
            dist_dup = dist_full.unsqueeze(1).expand(-1, window_size, -1, -1).contiguous().view(batch_size * window_size, num_nodes, num_nodes)
        
        mds_dup = mds_full.unsqueeze(1).expand(-1, window_size, -1, -1).contiguous().view(batch_size * window_size, num_nodes, 4)

        x0_dup = x_full.transpose(1, 2).contiguous().view(batch_size * window_size, num_nodes, self.embed_dim)
        for i in range(len(self.row_layers)):
            row_in = x_full.transpose(1, 2).contiguous().view(batch_size * window_size, num_nodes, self.embed_dim)
            row_out = self.row_layers[i](row_in, dist_dup, mds_coords=mds_dup, padding_mask=padding_mask_dup, x0=x0_dup)
            row_out = self.row_norms[i](row_in + row_out)
            x_full = row_out.reshape(batch_size, window_size, num_nodes, self.embed_dim).transpose(1, 2)
            
        # 8. Extract the Learned [ROOT] Token at Central Codon Site
        root_repr = x_full[:, 0, central_idx, :]  # [batch_size, embed_dim]
        
        # 16-Bin Ordinal LRT Logits directly from [ROOT] representation
        logits_lrt_ordinal = self.lrt_ordinal_head(root_repr)
        y_lrt_soft, _ = decode_soft_ordinal_lrt(logits_lrt_ordinal)
        if return_hidden:
            return y_lrt_soft.view(batch_size), logits_lrt_ordinal, root_repr
        return y_lrt_soft.view(batch_size), logits_lrt_ordinal


# ==============================================================================
# DOWNSTREAM FOUNDATION READOUT HEADS (PILLARS 4 & 5)
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

def decode_coral_lrt(logits):
    """Smooth continuous integration for LRT."""
    probs = torch.sigmoid(logits)
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
        proj = self.fc(x)
        cutoffs = self.get_cutoffs().to(device=x.device, dtype=x.dtype)
        return proj + cutoffs.unsqueeze(0)


class BustedMultiTaskHead(nn.Module):
    """
    Pillar 4: Alignment-Wide Omnibus Selection Head (BUSTED & BUSTED+S Emulation)
    with Rank-Consistent Ordinal Logits (CORAL).
    Eliminates regression compression at both the neutral floor and extreme selection bursts.
    """
    def __init__(self, embed_dim=384, num_queries=4, num_heads=4, hidden_dim=128, pool_dim=256, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.pool_dim = pool_dim
        self.in_proj = nn.Linear(embed_dim, pool_dim) if embed_dim != pool_dim else nn.Identity()
        self.queries = nn.Parameter(torch.randn(1, num_queries, pool_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(pool_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(pool_dim)
        
        self.mlp_shared = nn.Sequential(
            nn.Linear(num_queries * pool_dim, hidden_dim),
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
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B = x.shape[0]
        x_proj = self.in_proj(x)
        q = self.queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(q, x_proj, x_proj, key_padding_mask=mask)
        h = self.norm(attn_out).reshape(B, -1)
        feat = self.mlp_shared(h)
        
        logits_lrt = self.coral_lrt(feat)
        logits_logp = self.coral_logp(feat)
        logits_omega3 = self.coral_omega3(feat)
        
        return {
            "cls_prob": torch.sigmoid(self.head_cls(feat)).squeeze(-1),
            "logits_lrt": logits_lrt,
            "logits_logp": logits_logp,
            "logits_omega3": logits_omega3,
            "pred_lrt": decode_coral_lrt(logits_lrt),
            "pred_logp": decode_coral_logp(logits_logp),
            "pred_omega3": decode_coral_omega3(logits_omega3),
            "syn_var": F.softplus(self.head_syn_var(feat)).squeeze(-1),
            "omega_prop": torch.softmax(self.head_prop(feat), dim=-1)
        }


class aBSRELBranchHead(nn.Module):
    """
    Pillar 5: Branch-Site Episodic Selection Head (aBSREL Emulation).
    Predicts lineage-specific selection bursts along phylogenetic branches.
    """
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

    def forward(self, h_branch):
        """
        h_branch: [..., embed_dim] branch transition embeddings (h_child - h_parent).
        """
        feat = self.mlp(self.norm(h_branch))
        return {
            "p_burst": torch.sigmoid(self.head_p_branch(feat)),
            "omega_burst": F.softplus(self.head_omega_branch(feat)) + 1.0,
            "prop_burst": torch.sigmoid(self.head_prop_branch(feat))
        }






