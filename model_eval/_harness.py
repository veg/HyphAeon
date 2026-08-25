"""
_harness.py — shared helpers for model_eval tests.

Not collected by pytest (prefixed with _). Imported by test modules and
conftest.py.

Provides:
  - load_tensors: parse an alignment+tree into the model's input tensors.
  - predict: run the model on in-memory tensors and return LRT array.
  - pvals_from_lrt: convert LRTs to p-values (matches the CLI's formula).
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
import scipy.stats as stats
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade

from hyphaeon import dataset as ds


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


def predict(model, c, a, d, z, inv, batch=64):
    """Run model.forward_cached on variable sites only; return LRT array [L].

    Invariable sites get LRT=0 (the model would skip them too).
    """
    L = c.shape[0]
    lrts = np.zeros(L, dtype=np.float32)
    var_idx = np.where(~inv)[0]
    if len(var_idx) == 0:
        return lrts
    cache = model.precompute_tree_cache(d, z)
    with torch.no_grad():
        for s in range(0, len(var_idx), batch):
            idx = var_idx[s:s + batch]
            y, _ = model.forward_cached(c[idx], a[idx], cache)
            lrts[idx] = torch.clamp(y.squeeze(-1), min=0.0).numpy().flatten()
    return lrts


def pvals_from_lrt(lrts):
    """Convert LRT to p-value using the CLI's Self & Liang (1987) mixture
    formula: 0.5 * chi2.sf(LRT, df=1) under 0.5·δ₀ + 0.5·χ²₁."""
    p = np.ones(len(lrts), dtype=np.float64)
    m = lrts > 0
    p[m] = 0.5 * stats.chi2.sf(lrts[m], df=1)
    return p


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
