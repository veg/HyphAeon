"""
_harness.py — shared helpers for model_eval tests.

Not collected by pytest (prefixed with _). Imported by test modules and
conftest.py.

Provides:
  - load_tensors: parse an alignment+tree into the model's input tensors.
  - predict: run the model on in-memory tensors and return LRT array.
  - pvals_from_lrt: convert LRTs to p-values (MEME asymptotic mixture).
  - evaluate_alignment: load + predict + p-values in one call, with the
    skip-if-too-few-variable-sites check shared by calibration tests.
  - fpr_at: false positive rate among tested sites at a given alpha.
  - make_star_tree, make_scaled_tree, make_duplicate_alignment: generate
    phylogenetic variants for invariance testing.
  - permute_columns: within-column taxon permutation of codon tensors.
"""
import os
from io import StringIO

import numpy as np
import torch
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade

from hyphaeon import dataset as ds
from hyphaeon.stats import pvals_from_lrt_meme as pvals_from_lrt
from hyphaeon.attribution import attribute_selection
from hyphaeon.inference import predict_site_lrts


# ---------------------------------------------------------------------------
# Tensor loading & prediction
# ---------------------------------------------------------------------------

def load_tensors(fa_path, nwk_path, **kw):
    """Load alignment + tree into model input tensors.

    Returns (c, a, d, z, inv, taxa, L) where:
      c, a: [L, N, 1] long tensors (codon tokens, aa tokens)
      d:    [1, N, N] float tensor (patristic distance matrix)
      z:    [1, N, 4] float tensor (MDS coordinates)
      inv:  [L] bool array (True = invariable site)
      taxa: list of taxon names
      L:    number of codon sites
    """
    return ds.load_alignment_and_tree(fa_path, nwk_path, **kw)


def predict(model, c, a, d, z, inv, batch=64, attribute=False, taxa=None, focal_sites=None, min_lrt=3.84):
    """Run model.forward_cached on variable sites only; return LRT array [L].

    If attribute=True, returns (lrts, attribution_dict) where attribution_dict
    details which species drive the selection signal and when selection occurred.
    """
    lrts = predict_site_lrts(model, c, a, d, z, inv, batch_size=batch)
    if not attribute:
        return lrts

    cache = model.precompute_tree_cache(d, z)
    attr_dict = attribute_selection(
        model=model, c=c, a=a, d=d, z=z, inv=inv, taxa=taxa,
        focal_sites=focal_sites, min_lrt=min_lrt, base_lrts=lrts, cache=cache
    )
    return lrts, attr_dict


def evaluate_alignment(model, fa_path, nwk_path, min_tested=10, **load_kw):
    """Load an alignment, run the model, and compute p-values.

    Shared entry point for calibration tests (null FPR, power, composition
    bias, alignment length) that all follow the same
    load -> predict -> p-values -> skip-if-too-few-variable-sites pipeline.

    Returns a dict: {lrt, pval, tested, taxa, L}. Calls pytest.skip (via a
    lazy import, since this module has no pytest dependency otherwise) if
    fewer than min_tested sites are variable.
    """
    import pytest
    c, a, d, z, inv, taxa, L = load_tensors(fa_path, nwk_path, **load_kw)
    lrt = predict(model, c, a, d, z, inv)
    pval = pvals_from_lrt(lrt)
    tested = ~inv
    if tested.sum() < min_tested:
        pytest.skip(f"Too few variable sites ({tested.sum()})")
    return {"lrt": lrt, "pval": pval, "tested": tested, "taxa": taxa, "L": L}


def fpr_at(pval, tested, alpha=0.05):
    """False positive rate among tested sites at the given alpha."""
    return float(np.mean(pval[tested] <= alpha))


# ---------------------------------------------------------------------------
# Variant generators — produce (fasta, newick) pairs for invariance tests
# ---------------------------------------------------------------------------

def _parse_base(fa_path, nwk_path):
    """Return (seq_dict, tree_obj, taxa, L) from the base alignment+tree."""
    seqs = ds.parse_alignment_sequences(fa_path)
    tree = ds.extract_tree_from_string_or_file(nwk_path)
    taxa = [t.name.strip("'\"") for t in tree.get_terminals()
            if t.name and t.name.strip("'\"") in seqs]
    L = len(seqs[taxa[0]]) // 3
    return seqs, tree, taxa, L


def _write_fasta(path, seq_dict, order):
    with open(path, "w") as f:
        for t in order:
            f.write(f">{t}\n{seq_dict[t]}\n")


def _write_newick(path, tree):
    out = StringIO()
    Phylo.write(tree, out, "newick")
    with open(path, "w") as f:
        f.write(out.getvalue())


def make_star_tree(base_nwk, taxa, tmp_path):
    """Star tree preserving mean patristic scale. Returns newick path.

    The tree is structured as a shallow binary ((t0,t1),t2,...,tN) so that
    the dataset parser (which requires >=2 opening parens) accepts it.
    The topology is still effectively a star — all taxa are equidistant
    from the root, with no meaningful internal structure.
    """
    tree = ds.extract_tree_from_string_or_file(base_nwk)
    dm = ds.compute_fast_dist_matrix(tree, taxa)
    off = dm[np.triu_indices(len(taxa), 1)]
    bl = 0.5 * off.mean()
    # Group first two taxa in a cherry, rest as direct children of root.
    # This gives count('(') >= 2 while preserving the star topology's
    # key property: all taxa equidistant from the root.
    inner = f"({taxa[0]}:{bl:.6f},{taxa[1]}:{bl:.6f}):0"
    rest = ",".join(f"{t}:{bl:.6f}" for t in taxa[2:])
    star = f"({inner},{rest});"
    p = os.path.join(str(tmp_path), "star.nwk")
    with open(p, "w") as f:
        f.write(star + "\n")
    return p


def make_scaled_tree(base_nwk, scale, tmp_path):
    """Multiply all branch lengths by `scale`. Returns newick path."""
    tree = ds.extract_tree_from_string_or_file(base_nwk)
    for cl in tree.find_clades():
        if cl.branch_length is not None:
            cl.branch_length *= scale
    p = os.path.join(str(tmp_path), f"scale_x{scale}.nwk")
    _write_newick(p, tree)
    return p


def make_zero_distance_tree(taxa, tmp_path):
    """All-zero distance tree (all taxa at the same point).

    Uses 1e-4 branch lengths (the minimum enforced by the dataset pipeline)
    so the tree parses successfully. The resulting distance matrix will be
    near-zero, collapsing all taxa to the same MDS coordinate.

    Structured as a shallow binary ((t0,t1),t2,...,tN) so the dataset parser
    (which requires >=2 opening parens) accepts it.
    """
    bl = 1e-4
    inner = f"({taxa[0]}:{bl},{taxa[1]}:{bl})"
    rest = ",".join(f"{t}:{bl}" for t in taxa[2:])
    star = f"({inner},{rest});"
    p = os.path.join(str(tmp_path), "zero.nwk")
    with open(p, "w") as f:
        f.write(star + "\n")
    return p


def make_duplicate_alignment(base_fa, base_nwk, n_duplicates, tmp_path):
    """Add n_duplicates exact-copy taxa on short terminal branches.

    Returns (fasta_path, newick_path, n_total_taxa).
    """
    seqs, tree, taxa, L = _parse_base(base_fa, base_nwk)
    dup_seqs = dict(seqs)
    for i in range(n_duplicates):
        src = taxa[i % len(taxa)]
        dup_seqs[f"dup{i}"] = seqs[src]
        term = next(t for t in tree.get_terminals()
                    if t.name.strip("'\"") == src)
        orig_bl = term.branch_length or 1e-4
        child_a = Clade(branch_length=1e-4, name=term.name)
        child_b = Clade(branch_length=0.001, name=f"dup{i}")
        term.name = None
        term.branch_length = orig_bl
        term.clades = [child_a, child_b]
    all_taxa = [t.name.strip("'\"") for t in tree.get_terminals() if t.name]
    fa = os.path.join(str(tmp_path), "dup.fasta")
    nwk = os.path.join(str(tmp_path), "dup.nwk")
    _write_fasta(fa, dup_seqs, all_taxa)
    _write_newick(nwk, tree)
    return fa, nwk, len(all_taxa)


def permute_columns(c, a, tested, seed):
    """Permute which taxon carries which codon, independently per variable site.

    Returns (c_perm, a_perm) — new tensors, originals unchanged.
    """
    N = c.shape[1]
    rng = np.random.default_rng(seed)
    cp, ap = c.clone(), a.clone()
    for s in np.where(tested)[0]:
        perm = rng.permutation(N)
        cp[s, :, 0] = c[s, perm, 0]
        ap[s, :, 0] = a[s, perm, 0]
    return cp, ap


def make_frameshift_alignment(base_fa, taxa, tmp_path, shift=1):
    """Shift the reading frame by `shift` nucleotides. Returns fasta path."""
    seqs = ds.parse_alignment_sequences(base_fa)
    shifted = {t: seqs[t][shift:] for t in taxa}
    p = os.path.join(str(tmp_path), f"frameshift_{shift}.fasta")
    _write_fasta(p, shifted, taxa)
    return p


def make_uracil_alignment(base_fa, taxa, tmp_path):
    """Replace T with U (RNA alphabet). Returns fasta path."""
    seqs = ds.parse_alignment_sequences(base_fa)
    ura = {t: seqs[t].replace("T", "U") for t in taxa}
    p = os.path.join(str(tmp_path), "uracil.fasta")
    _write_fasta(p, ura, taxa)
    return p


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def sensitivity_r(base_lrt, variant_lrt, tested):
    """Pearson r of LRTs on tested (variable) sites. 1.0 = invariant."""
    from scipy.stats import pearsonr
    if tested.sum() < 3:
        return float("nan")
    return float(pearsonr(base_lrt[tested], variant_lrt[tested])[0])


def max_abs_diff(base_lrt, variant_lrt, tested):
    """Max |ΔLRT| on tested sites."""
    return float(np.abs(variant_lrt - base_lrt)[tested].max())


# ---------------------------------------------------------------------------
# Invariance/sensitivity majority-vote helpers
# ---------------------------------------------------------------------------

def format_invariance_results(results, threshold):
    """Format per-dataset (name, r) pairs for assertion messages.

    Labels each dataset as INVARIANT, sensitive, or NaN (undefined).
    """
    lines = []
    for name, r in results:
        if np.isnan(r):
            status = "NaN"
        elif r >= threshold:
            status = "INVARIANT"
        else:
            status = "sensitive"
        lines.append(f"  {name:15s} r={r:.6f} [{status}]")
    return "\n".join(lines)


def majority_invariant(results, threshold):
    """Return True if the model is invariant on the majority of datasets.

    Datasets with NaN sensitivity_r (too few tested sites) are excluded
    from the count — they should not count as either invariant or
    sensitive, since the metric is undefined.
    """
    valid = [(name, r) for name, r in results if not np.isnan(r)]
    if not valid:
        return False
    n_inv = sum(1 for _, r in valid if r >= threshold)
    return n_inv > len(valid) / 2


# ---------------------------------------------------------------------------
# Method-level helpers — BUSTED, PhyloWAS, ESSM
# ---------------------------------------------------------------------------

def phylowas_pvals(result):
    """Extract p-values from PhyloWAS result dict."""
    sites = result.get("sites", [])
    return np.array([s.get("p_value", 1.0) for s in sites], dtype=np.float64)


def essm_edge_pvals(result):
    """Extract edge p-values from ESSM result dict."""
    edges = result.get("edges", [])
    return np.array([e.get("p_val", 1.0) for e in edges], dtype=np.float64)


def get_taxa(fa_path, nwk_path):
    """Load taxa names from an alignment+tree. Returns list of taxon names."""
    _, _, _, _, _, taxa, _ = ds.load_alignment_and_tree(fa_path, nwk_path)
    return taxa


def fg_string(taxa, n_fg):
    """Return a comma-separated foreground string with the first n_fg taxa."""
    return ",".join(taxa[:n_fg])


# ---------------------------------------------------------------------------
# BUSTED omnibus inference
# ---------------------------------------------------------------------------

def run_busted(model, fa_path, nwk_path, busted_head=None, batch=64):
    """Run BUSTED omnibus inference on a single alignment.

    Delegates to hyphaeon.inference.run_busted_inference (the shared backend
    also used by cmd_busted) so the inference logic lives in exactly one place.

    If busted_head is None, a random-init BustedMultiTaskHead is used (the
    ACAT p-value is still meaningful; only the neural head outputs are from
    random weights).

    Returns the record dict (same keys as run_busted_inference, plus
    alignment and gene).
    """
    from hyphaeon.model import BustedMultiTaskHead
    from hyphaeon.inference import run_busted_inference

    device = next(model.parameters()).device
    c, a, d, z, inv, taxa, L = load_tensors(fa_path, nwk_path)

    if busted_head is None:
        embed_dim = next(model.parameters()).shape[-1]
        busted_head = BustedMultiTaskHead(embed_dim=embed_dim).to(device)
    busted_head.eval()

    record = run_busted_inference(
        model, busted_head, c, a, d, z, inv, taxa, L,
        device=device, batch_size=batch, progress=False,
    )
    record["alignment"] = fa_path
    record["gene"] = os.path.splitext(os.path.basename(fa_path))[0]
    return record
