"""
hyphaeon/export.py
------------------
ONNX export of the HyphAeon backbone (three outputs) and the BUSTED head, plus
models/manifest.json with the hashes every runtime surface verifies against.

WHY THIS FILE EXISTS

The JavaScript port and every runtime built on it (browser, Node MCP, Node
server) run the neural part of HyphAeon through onnxruntime, not through torch.
The graph published on Hugging Face as model.viral.onnx (sha256 de765904...)
returns only `lrt`, which is enough for `meme` but not for `epistasis`
(needs the mean root->leaf attention per taxon) or `busted` (needs the pooled
[ROOT] hidden state that the BustedMultiTaskHead consumes). This module exports
one graph per weight variant with all three:

    inputs   msa_codons  int64   [batch, num_species, 1]
             msa_aas     int64   [batch, num_species, 1]
             dist_matrix float32 [batch, num_species, num_species]
             mds_coords  float32 [batch, num_species, 4]
    outputs  lrt             float32 [batch]               (ordinal-decoded, eval forward)
             mean_root_attns float32 [batch, num_species]  (forward_cached(return_attentions=True)[2])
             root_repr       float32 [batch, embed_dim]    (forward_cached(return_hidden=True)[2])

and a second graph, busted_head.onnx, that maps a whole alignment's root_repr
[1, L, embed_dim] plus a key-padding mask [1, L] to the BUSTED head outputs.

Ported from (same repository, branch feat/js-port, base commit 3cb9cc6):
    hyphaeon/model.py  PhyloAxialTransformer.precompute_tree_cache  (tree augmentation, Markov bias, RoPE)
    hyphaeon/model.py  PhyloAxialTransformer.forward_cached          (the return_attentions=True branch)
    hyphaeon/model.py  decode_soft_ordinal_lrt, BustedMultiTaskHead
    hyphaeon/cli.py    cmd_busted                                     (how the head is built and loaded)
    hyphaeon/weights.py load_weights                                  (suite key remapping)

The wrapper below re-expresses forward_cached's attention branch as a single
traceable function of the four inputs because neither forward() nor
forward_cached() can return lrt, attentions and hidden state in one call
(return_hidden takes precedence over return_attentions), and because the tree
cache is computed outside the graph in the reference. The arithmetic and its
order are copied line for line; nothing is simplified. Verification of that
claim is scripts/verify_onnx.py, which compares every output against the
reference model's own forward_cached on the bundled examples.

Exporter: torch.onnx.export(dynamo=False), the legacy TorchScript tracer, is
the one that produced v1-viral (producer "pytorch 2.13.0", opset 17) and it is
tried first. With torch 2.10.0 / onnx 1.22.0 it produced all three graphs
(producer "pytorch 2.10.0", IR 8, opset 17) once two things were handled:
  - aten::expm1 has no opset-17 symbolic in this torch; it is registered as
    Exp followed by Sub (_register_legacy_symbolics), which is what v1-viral's
    graph contains.
  - the tracer bakes the example sequence length into nn.MultiheadAttention's
    internal reshapes, so the BUSTED head's cross-attention is traced through
    _cross_attention, a line-for-line copy of the need_weights=True branch of
    F.multi_head_attention_forward.
The dynamo exporter (dynamo=True) is kept as a fallback on exception; it was
exercised once before the expm1 symbolic existed and also produced a working
opset-17 graph, but is not what the committed artifacts came from. One graph
serves every num_species because both batch and num_species are declared
dynamic axes; verify_onnx.py runs the same file on 20 and 212 taxa, and
_smoke_test_dynamic re-runs each graph at export time on sizes different
from the tracing example.

Decisions behind constants:
    OPSET_VERSION = 17     matches v1-viral, and the earliest opset with a native
                           LayerNormalization node (which the tracer emits for
                           nn.LayerNorm), so ORT web/node need no contrib ops.
    TAXON_CAP = 512        the CLI's --max-species default for busted and DM3's
                           MAX_SPECIES_DEFAULT; above it Faith's-PD subsampling
                           runs in preprocessing, so the graph never sees more.
    DEFAULT_TAXON_CAP = 256  the max_species the checkpoints were trained at
                           (HF config.json "max_species": 256) and the density
                           reference in PhyloRowAttention (log(active/256)).
    BUSTED_HEAD_INIT_SEED = 0  see the BUSTED section below.

BUSTED HEAD: THE PYTHON REFERENCE IS NOT DETERMINISTIC. The suite checkpoint's
head_busted.* keys do not cover BustedMultiTaskHead: in_proj (384->256),
head_syn_var.*, head_prop.* are absent, and the three CORAL heads store their
cutoff steps as `raw_steps` while the module parameter is `theta_steps`.
cmd_busted loads with strict=False, so those parameters keep their random
torch.nn.init values and `hyphaeon busted` prints different
selection_probability / predicted_gene_lrt / omega proportions on every run
(measured: two runs on examples/Smc6.fasta gave 0.308 vs 0.549 for
selection_probability). A graph has to be deterministic, so the export seeds
torch with BUSTED_HEAD_INIT_SEED before constructing the head and then loads
the checkpoint exactly as cmd_busted does (strict=False, no key remapping).
That reproduces ONE of the reference's possible outputs and is recorded here
rather than fixed, because fixing it means retraining or a checkpoint
migration, not an export change. The manifest carries the resulting hash; the
numbers are not meaningful until the checkpoint is completed upstream.

Weight-key note replicated, not fixed: the suite stores a MEME head twice
(backbone.lrt_ordinal_head.* and head_meme.*, different values) and
load_weights maps both onto lrt_ordinal_head.*; because safetensors keys are
sorted, head_meme.* is written last and wins. The export goes through
load_weights so it uses the same head the CLI uses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import (
    EPS0,
    PhyloAxialTransformer,
    BustedMultiTaskHead,
    decode_soft_ordinal_lrt,
)
from .weights import load_weights, load_arch_config, HF_REPO_ID, get_variant_filename

MODEL_VERSION = "v1"
REFERENCE_VERSION = "1.0.0"
OPSET_VERSION = 17
TAXON_CAP = 512
DEFAULT_TAXON_CAP = 256
BUSTED_HEAD_INIT_SEED = 0

INPUT_NAMES = ["msa_codons", "msa_aas", "dist_matrix", "mds_coords"]
OUTPUT_NAMES = ["lrt", "mean_root_attns", "root_repr"]
BUSTED_INPUT_NAMES = ["root_repr", "mask"]
BUSTED_OUTPUT_NAMES = ["cls_prob", "pred_gene_lrt", "omega_prop", "syn_var", "pred_omega3", "pred_logp"]

VARIANTS = {
    "general": {
        "trained_on": "TOGA mammalian, 742 species",
        "regime": "deep / cross-species",
    },
    "viral": {
        "trained_on": "base + ~9,300 Datamonkey viral",
        "regime": "viral / shallow",
    },
}

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_GENERAL_WEIGHTS = REPO_ROOT / "model.safetensors"
DEFAULT_HF_DIR = DEFAULT_MODELS_DIR / "_hf"
HF_VIRAL_ONNX_SHA256 = "de765904107ba436c6ad6abbecb8af54962abd8444e1b5044947bb945d8ccda3"


# --------------------------------------------------------------------------- #
# Backbone graph
# --------------------------------------------------------------------------- #

class BackboneGraph(nn.Module):
    """
    forward_cached(return_attentions=True) with the tree cache folded in, and
    root_repr exposed. Every line has its origin in model.py; the comments cite
    the method it came from.
    """

    def __init__(self, model: PhyloAxialTransformer):
        super().__init__()
        self.model = model

    def forward(self, msa_codons, msa_aas, dist_matrix, mds_coords):
        m = self.model
        batch_size, num_species, window_size = msa_codons.shape
        central_idx = window_size // 2

        # forward_cached step 1: embeddings. mds_pos_static = mds_proj(mds_coords)
        # comes from precompute_tree_cache and is added per batch row here.
        codon_emb = m.codon_embedding(msa_codons)
        aa_emb = m.aa_embedding(msa_aas)
        x = torch.cat([codon_emb, aa_emb], dim=-1)
        x = x + m.pos_embedding.unsqueeze(1)
        phylo_pos = m.mds_proj(mds_coords)  # [batch, num_species, embed_dim]
        x = x + phylo_pos.unsqueeze(2)

        root = m.root_token.expand(batch_size, 1, window_size, -1)
        x_full = torch.cat([root, x], dim=1)  # [batch, num_species + 1, window_size, embed_dim]
        num_nodes = num_species + 1

        # precompute_tree_cache: augment MDS and distance matrix with the [ROOT] node.
        root_mds = torch.zeros_like(mds_coords[:, :1, :])
        mds_full = torch.cat([root_mds, mds_coords], dim=1)  # [batch, num_nodes, 4]

        root_dist = torch.norm(mds_coords, dim=-1, keepdim=True)  # [batch, num_species, 1]
        dist_top = torch.cat([torch.zeros_like(root_dist[:, :1, :]), root_dist.transpose(1, 2)], dim=2)
        dist_bot = torch.cat([root_dist, dist_matrix], dim=2)
        dist_full = torch.cat([dist_top, dist_bot], dim=1)  # [batch, num_nodes, num_nodes]
        dist_tensor = dist_full.unsqueeze(1)  # [batch, 1, num_nodes, num_nodes]

        # forward_cached step 3: column layers (none when window_size == 1; kept for fidelity).
        for i in range(len(m.col_layers)):
            col_in = (x_full.reshape(batch_size * num_nodes, window_size, m.embed_dim)
                      if window_size > 1 else x_full.reshape(batch_size * num_nodes, m.embed_dim))
            col_out = m.col_layers[i](col_in)
            x_full = col_out.reshape(batch_size, num_nodes, window_size, m.embed_dim)

        # forward_cached step 4: row layers, return_attentions branch.
        x0_dup = x_full.transpose(1, 2).contiguous().view(batch_size * window_size, num_nodes, m.embed_dim)
        layer_attns = []

        for i, layer in enumerate(m.row_layers):
            row_in = x_full.transpose(1, 2).contiguous().view(batch_size * window_size, num_nodes, m.embed_dim)

            # precompute_tree_cache per layer: Markov log-bias and RoPE phases.
            decay_rate = F.softplus(layer.phylo_w1)  # [num_heads, 1, 1]
            markov_kernel = EPS0 + (1.0 - EPS0) * torch.exp(-decay_rate * dist_tensor)
            phylo_bias = torch.log(markov_kernel.clamp(min=1e-5))  # [batch, num_heads, num_nodes, num_nodes]

            half_dim = layer.head_dim // 2
            m_exp = mds_full.unsqueeze(1).unsqueeze(3)  # [batch, 1, num_nodes, 1, 4]
            f_exp = layer.rope_freqs.unsqueeze(0).unsqueeze(2)  # [1, num_heads, 1, half_dim, 4]
            angles = (m_exp * f_exp).sum(dim=-1)  # [batch, num_heads, num_nodes, half_dim]
            cos = torch.cos(angles)
            sin = torch.sin(angles)

            q = layer.q_proj(row_in).view(batch_size * window_size, num_nodes, layer.num_heads, layer.head_dim).transpose(1, 2)
            k = layer.k_proj(row_in).view(batch_size * window_size, num_nodes, layer.num_heads, layer.head_dim).transpose(1, 2)
            v = layer.v_proj(row_in).view(batch_size * window_size, num_nodes, layer.num_heads, layer.head_dim).transpose(1, 2)

            q1, q2 = q[..., :half_dim], q[..., half_dim:]
            k1, k2 = k[..., :half_dim], k[..., half_dim:]
            q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
            k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(layer.head_dim)
            scores = scores + phylo_bias
            attn_weights = torch.softmax(scores, dim=-1)

            layer_attns.append(attn_weights[:, :, 0, 1:])

            # layer.dropout is identity in eval mode; the graph is exported in eval mode.
            out = torch.matmul(attn_weights, v)
            out = out.transpose(1, 2).contiguous().view(batch_size * window_size, num_nodes, layer.embed_dim)
            out = layer.out_proj(out) + layer.alpha_skip * x0_dup
            row_out = m.row_norms[i](row_in + out)
            x_full = row_out.reshape(batch_size, window_size, num_nodes, m.embed_dim).transpose(1, 2)

        root_repr = x_full[:, 0, central_idx, :]  # [batch, embed_dim]
        logits_lrt_ordinal = m.lrt_ordinal_head(root_repr)
        y_lrt_soft, _ = decode_soft_ordinal_lrt(logits_lrt_ordinal)

        all_attns = torch.stack(layer_attns, dim=0)  # [num_layers, batch*window, num_heads, num_species]
        mean_root_attns = all_attns.mean(dim=(0, 2)).view(batch_size, num_species)

        return y_lrt_soft.view(batch_size), mean_root_attns, root_repr


def _cross_attention(mha: nn.MultiheadAttention, query, key, key_padding_mask):
    """
    torch.nn.functional.multi_head_attention_forward, need_weights=True branch
    (the one nn.MultiheadAttention takes with its default need_weights=True, so
    the one BustedMultiTaskHead.forward runs), written out with batch-first
    tensors. It is spelled out because the TorchScript tracer bakes the example
    sequence length into the reshapes inside the library implementation
    (measured: a graph traced at 11 sites failed at 1097 with "requested shape
    {11,4,64}"), which would make busted_head.onnx unusable for any other L.
    Scaling by sqrt(1/head_dim) is applied to q before the matmul, as torch
    does, so the rounding order is identical.
    """
    B, Q, E = query.shape
    L = key.shape[1]
    H = mha.num_heads
    hd = E // H
    w_q, w_k, w_v = mha.in_proj_weight.chunk(3)
    b_q, b_k, b_v = mha.in_proj_bias.chunk(3)
    q = F.linear(query, w_q, b_q).view(B, Q, H, hd).transpose(1, 2)  # [B, H, Q, hd]
    k = F.linear(key, w_k, b_k).view(B, L, H, hd).transpose(1, 2)    # [B, H, L, hd]
    v = F.linear(key, w_v, b_v).view(B, L, H, hd).transpose(1, 2)    # [B, H, L, hd]
    q_scaled = q * math.sqrt(1.0 / float(hd))
    scores = torch.matmul(q_scaled, k.transpose(-2, -1))  # [B, H, Q, L]
    # key_padding_mask (bool, True = ignore) becomes an additive -inf mask, as _canonical_mask does.
    additive = torch.zeros_like(scores).masked_fill(key_padding_mask.view(B, 1, 1, L), float("-inf"))
    scores = scores + additive
    attn = torch.softmax(scores, dim=-1)
    # mha.dropout is 0.0 for this head and the graph is exported in eval mode.
    out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Q, E)
    return mha.out_proj(out)


class BustedHeadGraph(nn.Module):
    """
    BustedMultiTaskHead.forward as a tuple, in the order of BUSTED_OUTPUT_NAMES,
    with the cross-attention call replaced by _cross_attention (same math).
    """

    def __init__(self, head: BustedMultiTaskHead):
        super().__init__()
        self.head = head

    def forward(self, root_repr, mask):
        h_ = self.head
        x = root_repr
        B = x.shape[0]
        x_proj = h_.in_proj(x)
        q = h_.queries.expand(B, -1, -1)
        attn_out = _cross_attention(h_.cross_attn, q, x_proj, mask)
        h = h_.norm(attn_out).reshape(B, -1)
        feat = h_.mlp_shared(h)

        logits_lrt = h_.coral_lrt(feat)
        logits_logp = h_.coral_logp(feat)
        logits_omega3 = h_.coral_omega3(feat)

        from .model import decode_coral_lrt, decode_coral_logp, decode_coral_omega3
        return (
            torch.sigmoid(h_.head_cls(feat)).squeeze(-1),
            decode_coral_lrt(logits_lrt),
            torch.softmax(h_.head_prop(feat), dim=-1),
            F.softplus(h_.head_syn_var(feat)).squeeze(-1),
            decode_coral_omega3(logits_omega3),
            decode_coral_logp(logits_logp),
        )


# --------------------------------------------------------------------------- #
# Model construction (mirrors inference.load_model and cli.cmd_busted)
# --------------------------------------------------------------------------- #

def build_backbone(weights_path: str) -> PhyloAxialTransformer:
    arch = load_arch_config(weights_path)
    model = PhyloAxialTransformer(
        embed_dim=arch["embed_dim"],
        num_layers=arch["num_layers"],
        num_heads=arch["num_heads"],
        window_size=arch["window_size"],
    )
    state_dict = load_weights(weights=weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def build_busted_head(weights_path: str, seed: int = BUSTED_HEAD_INIT_SEED) -> BustedMultiTaskHead:
    """
    Exactly cmd_busted's construction, with the random-init made reproducible.
    Keys the checkpoint lacks stay at their seeded init (see the module header).
    """
    from safetensors.torch import load_file
    raw = load_file(weights_path, device="cpu")
    arch = load_arch_config(weights_path)
    torch.manual_seed(seed)
    head = BustedMultiTaskHead(embed_dim=arch["embed_dim"])
    busted_dict = {k.replace("head_busted.", ""): v for k, v in raw.items() if k.startswith("head_busted.")}
    if busted_dict:
        head.load_state_dict(busted_dict, strict=False)
    head.eval()
    return head


def busted_checkpoint_coverage(weights_path: str) -> Dict[str, list]:
    """Which BustedMultiTaskHead parameters the checkpoint does and does not provide."""
    from safetensors.torch import load_file
    raw = load_file(weights_path, device="cpu")
    ckpt = {k.replace("head_busted.", "") for k in raw if k.startswith("head_busted.")}
    head = BustedMultiTaskHead(embed_dim=load_arch_config(weights_path)["embed_dim"])
    expected = set(head.state_dict().keys())
    return {
        "loaded": sorted(ckpt & expected),
        "missing_from_checkpoint": sorted(expected - ckpt),
        "unused_in_checkpoint": sorted(ckpt - expected),
    }


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #

def _example_inputs(num_species: int = 7, batch: int = 3):
    g = torch.Generator().manual_seed(1234)
    msa_codons = torch.randint(0, 66, (batch, num_species, 1), generator=g, dtype=torch.int64)
    msa_aas = torch.randint(0, 23, (batch, num_species, 1), generator=g, dtype=torch.int64)
    d = torch.rand(num_species, num_species, generator=g)
    d = (d + d.transpose(0, 1)) * 0.5
    d.fill_diagonal_(0.0)
    dist_matrix = d.unsqueeze(0).expand(batch, -1, -1).contiguous()
    mds_coords = (torch.rand(num_species, 4, generator=g) - 0.5).unsqueeze(0).expand(batch, -1, -1).contiguous()
    return msa_codons, msa_aas, dist_matrix, mds_coords


def _register_legacy_symbolics():
    """
    ONNX has no Expm1 operator and the TorchScript exporter in torch 2.10 has no
    symbolic for aten::expm1 at opset 17 (decode_soft_ordinal_lrt ends with
    torch.expm1). v1-viral's graph lowers it to Exp followed by Sub (its op
    census has one more Exp and Sub than the six Markov kernels account for), so
    the same lowering is registered here. exp(x) - 1 differs from expm1(x) by at
    most one fp32 ulp of exp(x) (about 6e-8 near x = 0), inside the 1e-5 lrt
    tolerance verify_onnx.py enforces.
    """
    from torch.onnx import register_custom_op_symbolic
    from torch.onnx import symbolic_helper

    def expm1_symbolic(g, self):
        one = g.op("Constant", value_t=torch.tensor(1.0, dtype=torch.float32))
        return g.op("Sub", g.op("Exp", self), one)

    for opset in range(9, OPSET_VERSION + 1):
        register_custom_op_symbolic("aten::expm1", expm1_symbolic, opset)


def _onnx_export(module, args, out_path, input_names, output_names, dynamic_axes, dynamic_shapes):
    """Legacy tracer first (how v1-viral was made); dynamo exporter as fallback."""
    out_path = str(out_path)
    try:
        _register_legacy_symbolics()
        with torch.no_grad():
            torch.onnx.export(
                module, args, out_path,
                input_names=input_names, output_names=output_names,
                opset_version=OPSET_VERSION, dynamic_axes=dynamic_axes,
                dynamo=False, do_constant_folding=True, export_params=True,
            )
        return "torchscript"
    except Exception as e:  # pragma: no cover - fallback path
        print(f"[!] Legacy exporter failed ({type(e).__name__}: {e}); retrying with dynamo=True")
        with torch.no_grad():
            prog = torch.onnx.export(
                module, args,
                input_names=input_names, output_names=output_names,
                opset_version=OPSET_VERSION, dynamic_shapes=dynamic_shapes,
                dynamo=True, external_data=False,
            )
        prog.save(out_path, external_data=False)
        return "dynamo"


def export_backbone(weights_path: str, out_path: Path) -> str:
    model = build_backbone(weights_path)
    graph = BackboneGraph(model).eval()
    args = _example_inputs()
    dynamic_axes = {
        "msa_codons": {0: "batch", 1: "num_species"},
        "msa_aas": {0: "batch", 1: "num_species"},
        "dist_matrix": {0: "batch", 1: "num_species", 2: "num_species"},
        "mds_coords": {0: "batch", 1: "num_species"},
        "lrt": {0: "batch"},
        "mean_root_attns": {0: "batch", 1: "num_species"},
        "root_repr": {0: "batch"},
    }
    B, N = torch.export.Dim("batch"), torch.export.Dim("num_species")
    dynamic_shapes = ({0: B, 1: N}, {0: B, 1: N}, {0: B, 1: N, 2: N}, {0: B, 1: N})
    exporter = _onnx_export(graph, args, out_path, INPUT_NAMES, OUTPUT_NAMES, dynamic_axes, dynamic_shapes)
    _pin_output_dim(out_path, "root_repr", 1, model.embed_dim)
    _smoke_test_dynamic(out_path, graph, _example_inputs(num_species=13, batch=2), INPUT_NAMES)
    return exporter


def _pin_output_dim(onnx_path, output_name: str, axis: int, value: int) -> None:
    """
    The tracer names root_repr's last axis after the Gather that produced it
    instead of writing its static size. The value is embed_dim by construction
    (x_full[:, 0, central_idx, :]); write it so readers of the graph metadata
    see [batch, 384] rather than [batch, <symbol>]. Bytes of the weights and
    nodes are untouched; only the output ValueInfo changes.
    """
    import onnx
    m = onnx.load(str(onnx_path))
    for o in m.graph.output:
        if o.name == output_name:
            dim = o.type.tensor_type.shape.dim[axis]
            dim.ClearField("dim_param")
            dim.dim_value = value
    onnx.save(m, str(onnx_path))


def export_busted_head(weights_path: str, out_path: Path) -> str:
    head = build_busted_head(weights_path)
    graph = BustedHeadGraph(head).eval()
    L, E = 11, head.embed_dim
    g = torch.Generator().manual_seed(1234)
    root_repr = torch.randn(1, L, E, generator=g)
    mask = torch.zeros(1, L, dtype=torch.bool)
    dynamic_axes = {
        "root_repr": {0: "batch", 1: "num_sites"},
        "mask": {0: "batch", 1: "num_sites"},
        "cls_prob": {0: "batch"},
        "pred_gene_lrt": {0: "batch"},
        "omega_prop": {0: "batch"},
        "syn_var": {0: "batch"},
        "pred_omega3": {0: "batch"},
        "pred_logp": {0: "batch"},
    }
    Ls = torch.export.Dim("num_sites")
    dynamic_shapes = ({1: Ls}, {1: Ls})
    # nn.MultiheadAttention's fused fast path is not traceable; the math path is the reference math.
    prev = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        exporter = _onnx_export(graph, (root_repr, mask), out_path, BUSTED_INPUT_NAMES, BUSTED_OUTPUT_NAMES,
                                dynamic_axes, dynamic_shapes)
    finally:
        torch.backends.mha.set_fastpath_enabled(prev)
    other = (torch.randn(1, 37, E, generator=g), torch.zeros(1, 37, dtype=torch.bool))
    _smoke_test_dynamic(out_path, graph, other, BUSTED_INPUT_NAMES)
    return exporter


def _smoke_test_dynamic(onnx_path, module, args, input_names, tol=1e-4):
    """
    Run the exported graph under onnxruntime on inputs whose dynamic axes differ
    from the tracing example and compare with the torch module. This catches a
    tracer that baked an example size into the graph (see _cross_attention).
    Silently skipped when onnxruntime is not installed; verify_onnx.py is the
    full check.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    feeds = {n: t.numpy() for n, t in zip(input_names, args)}
    got = sess.run(None, feeds)
    with torch.no_grad():
        want = module(*args)
    for name, g, w in zip([o.name for o in sess.get_outputs()], got, want):
        delta = float(abs(torch.from_numpy(g) - w).max())
        if g.shape != tuple(w.shape) or delta > tol:
            raise RuntimeError(f"{onnx_path}: output {name} shape {g.shape} vs {tuple(w.shape)}, max|delta|={delta:.3e} "
                               f"on inputs with dynamic axes different from the tracing example")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_variant_weights(variant: str, general_weights: Optional[str], hf_dir: Path) -> str:
    """general -> the repository's model.safetensors; others -> models/_hf/model.<variant>.safetensors."""
    if variant == "general":
        p = Path(general_weights) if general_weights else DEFAULT_GENERAL_WEIGHTS
        if not p.exists():
            raise FileNotFoundError(f"general weights not found at {p}")
        return str(p)
    filename = get_variant_filename(variant)
    p = hf_dir / filename
    if not p.exists():
        from huggingface_hub import hf_hub_download
        print(f"[*] Downloading {filename} from {HF_REPO_ID} into {hf_dir}")
        hf_hub_download(HF_REPO_ID, filename, local_dir=str(hf_dir))
    return str(p)


def write_manifest(models_dir: Path, variants: Dict[str, dict], exported_with: dict) -> Path:
    import onnx
    manifest = {
        "model_version": MODEL_VERSION,
        "variants": variants,
        "taxon_cap": TAXON_CAP,
        "default_taxon_cap": DEFAULT_TAXON_CAP,
        "dropped_heads_policy": "omit",
        "onnx": {"opset": OPSET_VERSION, "inputs": list(INPUT_NAMES), "outputs": list(OUTPUT_NAMES)},
        "prng": {"algorithm": "xoshiro256**", "default_seed": 42},
        "reference_version": REFERENCE_VERSION,
        "exported_with": exported_with,
    }
    path = models_dir / "manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return path


def run_export(variants, models_dir: Path, general_weights: Optional[str] = None,
               hf_dir: Optional[Path] = None, skip_busted: bool = False) -> dict:
    import onnx
    models_dir = Path(models_dir)
    hf_dir = Path(hf_dir) if hf_dir else models_dir / "_hf"
    models_dir.mkdir(parents=True, exist_ok=True)

    manifest_variants = {}
    existing = models_dir / "manifest.json"
    if existing.exists():
        try:
            manifest_variants = json.load(open(existing)).get("variants", {})
        except Exception:
            manifest_variants = {}

    exporters = {}
    for variant in variants:
        weights = resolve_variant_weights(variant, general_weights, hf_dir)
        out = models_dir / f"{variant}.onnx"
        print(f"[*] Exporting {variant} backbone from {weights} -> {out}")
        exporters[variant] = export_backbone(weights, out)
        onnx.checker.check_model(str(out))
        entry = {
            "safetensors_sha256": sha256_file(weights),
            "onnx_sha256": sha256_file(out),
        }
        if variant == "general" and not skip_busted:
            bh = models_dir / "busted_head.onnx"
            print(f"[*] Exporting BUSTED head from {weights} -> {bh}")
            exporters["busted_head"] = export_busted_head(weights, bh)
            onnx.checker.check_model(str(bh))
            entry["busted_head_onnx_sha256"] = sha256_file(bh)
        elif variant == "general" and "busted_head_onnx_sha256" in manifest_variants.get("general", {}):
            entry["busted_head_onnx_sha256"] = manifest_variants["general"]["busted_head_onnx_sha256"]
        entry.update(VARIANTS.get(variant, {}))
        manifest_variants[variant] = entry
        print(f"[✓] {variant}: exporter={exporters[variant]} sha256={entry['onnx_sha256']}")

    ordered = {k: manifest_variants[k] for k in ("general", "viral") if k in manifest_variants}
    ordered.update({k: v for k, v in manifest_variants.items() if k not in ordered})
    exported_with = {"torch": torch.__version__, "onnx": onnx.__version__}
    path = write_manifest(models_dir, ordered, exported_with)
    print(f"[✓] Manifest written to {path}")
    return {"manifest": str(path), "variants": ordered, "exporters": exporters}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variant", choices=["general", "viral", "all"], default="all",
                        help="Which weight variant(s) to export")
    parser.add_argument("--out", default=str(DEFAULT_MODELS_DIR),
                        help="Output directory for <variant>.onnx, busted_head.onnx and manifest.json")
    parser.add_argument("--general-weights", default=None,
                        help="Path to the suite safetensors for the general variant (default: repo model.safetensors)")
    parser.add_argument("--hf-dir", default=None,
                        help="Directory holding model.<variant>.safetensors downloaded from HF (default: <out>/_hf)")
    parser.add_argument("--skip-busted-head", action="store_true",
                        help="Do not export busted_head.onnx")


def command(args: argparse.Namespace) -> dict:
    variants = ["general", "viral"] if args.variant == "all" else [args.variant]
    return run_export(
        variants,
        Path(args.out),
        general_weights=args.general_weights,
        hf_dir=Path(args.hf_dir) if args.hf_dir else None,
        skip_busted=args.skip_busted_head,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="hyphaeon export-onnx",
                                     description="Export HyphAeon ONNX graphs and models/manifest.json")
    configure_parser(parser)
    command(parser.parse_args(argv))


if __name__ == "__main__":
    main()
