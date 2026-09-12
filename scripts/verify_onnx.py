#!/usr/bin/env python
"""
scripts/verify_onnx.py
----------------------
WHY THIS FILE EXISTS

The exported graphs are only useful if every output equals what the Python
reference computes for the same tensors, on a small alignment and a large one,
from ONE graph file (dynamic num_species). This script is that proof, and the
Python half of the parity harness in PLAN.md section 3.3:

  1. For examples/Smc6 (20 taxa, 1097 codons) and examples/camelid (212 taxa,
     96 codons) it builds the tensors with hyphaeon.dataset.load_alignment_and_tree,
     runs the reference model (eval, CPU, fp32) through the exact code paths the
     CLI uses -- precompute_tree_cache + forward_cached(return_attentions=True)
     for lrt and mean_root_attns, forward_cached(return_hidden=True) for
     root_repr -- and runs models/<variant>.onnx under onnxruntime on the same
     inputs with dist_matrix / mds_coords tiled along the batch axis the way
     DM3's assemble.js does. Tolerances: |dmean_root_attns| <= 1e-5,
     |droot_repr| <= 1e-4 (root_repr is pre-LayerNorm-scale magnitude ~1-10, so
     one extra decade of slack for accumulated fp32 reordering), and
     |dlrt| <= 1e-5 * max(1, |lrt|). The lrt bound is absolute below LRT 1 and
     relative above it because the absolute figure is not achievable in fp32
     for large LRTs: on camelid/general the ORT-vs-torch maximum is 1.19e-5 at
     LRT 5.02 (relative 2.4e-6), while torch's own two fp32 code paths
     (forward vs forward_cached) differ by 6.7e-6 on the same inputs and each
     sits 3-6e-6 from an fp64 run. expm1 at the end of the decoder multiplies
     any logit-level rounding by (1 + LRT). Both the absolute and the relative
     figures are printed so the noise floor stays visible.
  2. It downloads Hugging Face's existing model.viral.onnx (sha256 de765904...),
     the graph DM3 ships today, and checks our viral export's lrt against its
     lrt within 1e-5 on the same inputs, so the new three-output graph is a
     drop-in replacement for the one-output graph.
  3. It runs models/busted_head.onnx on Smc6's root_repr (invariable sites
     zeroed, as cmd_busted does) and compares every output with the PyTorch
     head built by hyphaeon.export.build_busted_head, and demonstrates why that
     builder seeds torch: the checkpoint does not cover the head, so an unseeded
     head gives different numbers (see hyphaeon/export.py header).
  4. It checks manifest.json's hashes against the files on disk.

Exit status is non-zero on any failed assertion so CI can gate on it.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hyphaeon.dataset import load_alignment_and_tree  # noqa: E402
from hyphaeon.export import (  # noqa: E402
    build_backbone, build_busted_head, busted_checkpoint_coverage, resolve_variant_weights,
    sha256_file, INPUT_NAMES, OUTPUT_NAMES, TAXA_OUTPUT_NAMES, BUSTED_OUTPUT_NAMES,
    HF_VIRAL_ONNX_SHA256, HF_REPO_ID,
)
from hyphaeon.inference import prepare_alignment  # noqa: E402
from hyphaeon.splits import extract_cross_taxa_attentions_and_embeddings  # noqa: E402
from hyphaeon.dating import compute_neural_covariance_kernel  # noqa: E402

TOL_LRT = 1e-5
TOL_ATTN = 1e-5
TOL_REPR = 1e-4
TOL_HF_LRT = 1e-5
TOL_BUSTED = 1e-5

# <variant>_taxa.onnx, checked against splits.py's own extractor.
#
# THE BOUND IS RELATIVE, BECAUSE THE ERROR IS. The graph sums over the whole
# call at once where splits.py sums in chunks of 32 (splits.py:85-96), so the
# difference is float32 reassociation of L terms: absolute error scales with the
# largest entry, which scales as 1/N. Measured, one graph file, general variant:
#
#   example   N    L      max entry   max|delta|   relative
#   Smc6      20   1097   0.0450      2.72e-08     6.04e-07
#   korber    143   981   0.0182      1.17e-08     6.42e-07
#   camelid   212   96    0.0079      1.86e-09     2.36e-07
#
# The absolute figure spans 14x across those three; the relative figure spans
# 2.7x and is flat in both N and L. So a relative bound is the scale-free
# statement of the same fact, and an absolute one calibrated at a single N
# silently tightens or loosens as taxa are added. 1e-6 is ~1.6x the worst
# observation and still an order of magnitude below any structural error -- a
# transposed block, an off-by-one slice or a dropped layer moves entries by
# >= 1e-4, which this catches with four decades to spare.
TOL_CROSS_REL = 1e-6
# taxa_repr_sum is post-LayerNorm, components O(1); measured worst relative
# 4.0e-7 over the same three examples.
TOL_REPR_REL = 1e-6
# K_neural entries are correlations, so |max| is 1 and absolute == relative.
# compute_neural_covariance_kernel (dating.py:95-101) centres across taxa and
# correlation-normalises, which amplifies its input by a measured ~500x
# (Smc6 2.72e-08 -> 1.35e-05, korber 1.17e-08 -> 4.80e-06). Measured worst
# 1.50e-05; 3e-5 is 2x that.
TOL_K_NEURAL = 3e-5

# An EXTRA absolute bound, on the dating example only, and the only tolerance
# here derived from a product quantity rather than from float32 noise. korber is
# the alignment PLAN-TEMPORAL phase 4 is specified against. Downstream of the
# kernel, a unit of K is worth ~100 years of t_MRCA, and the measured chain
# sensitivity on korber is: perturb cross_attn by 1e-8 -> K moves 8.1e-6 ->
# t_MRCA moves 7.8e-4 years. The phase's acceptance class for t_MRCA is 1e-3
# years, so cross_attn has to hold ~2e-8 on THIS alignment for the date to be
# reproducible; the relative bound above would permit 1.8e-8, which is the same
# statement, and this pins it independently so that a future change to the
# relative bound cannot quietly spend the years budget.
DATING_EXAMPLE = "korber"
TOL_CROSS_ABS_DATING = 2e-8

EXAMPLES = [
    ("Smc6", REPO / "examples" / "Smc6.fasta", REPO / "examples" / "Smc6.nwk"),
    ("camelid", REPO / "examples" / "camelid.fasta", REPO / "examples" / "camelid.nwk"),
]

# For <variant>_taxa.onnx. korber is the alignment the dating pillar is actually
# specified against (PLAN-TEMPORAL phase 4) and is the tree-free / TN93 path;
# Smc6 and camelid come along so the check spans 20, 143 and 212 taxa from one
# graph file, as the backbone checks do.
TAXA_EXAMPLES = [
    ("Smc6", REPO / "examples" / "Smc6.fasta", REPO / "examples" / "Smc6.nwk"),
    ("korber", REPO / "examples" / "korber_env_gp160.fasta", None),
    ("camelid", REPO / "examples" / "camelid.fasta", REPO / "examples" / "camelid.nwk"),
]

FAILURES = []


def check(name, value, tol):
    ok = bool(value <= tol)
    print(f"    {'PASS' if ok else 'FAIL'}  {name:<32} max|delta| = {value:.3e}  (tol {tol:.0e})")
    if not ok:
        FAILURES.append(f"{name}: {value:.3e} > {tol:.0e}")
    return ok


def check_lrt(name, got, want, tol=TOL_LRT):
    """Absolute tolerance below |lrt| = 1, relative above; prints both figures."""
    delta = np.abs(got - want)
    scaled = delta / np.maximum(1.0, np.abs(want))
    i = int(scaled.argmax())
    ok = bool(scaled.max() <= tol)
    print(f"    {'PASS' if ok else 'FAIL'}  {name:<32} max|delta| = {delta.max():.3e}, "
          f"max|delta|/max(1,|lrt|) = {scaled.max():.3e} at lrt={want[i]:.4f}  (tol {tol:.0e})")
    if not ok:
        FAILURES.append(f"{name}: {scaled.max():.3e} > {tol:.0e}")
    return ok


def ort_session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def reference_outputs(model, c, a, d, z):
    with torch.no_grad():
        cache = model.precompute_tree_cache(d, z)
        lrt, _, attns = model.forward_cached(c, a, cache, return_attentions=True)
        lrt2, _, repr_ = model.forward_cached(c, a, cache, return_hidden=True)
    assert torch.equal(lrt, lrt2), "forward_cached lrt differs between the attentions and hidden branches"
    return lrt.numpy(), attns.numpy(), repr_.numpy()


def onnx_feeds(c, a, d, z):
    L = c.shape[0]
    return {
        "msa_codons": c.numpy().astype(np.int64),
        "msa_aas": a.numpy().astype(np.int64),
        "dist_matrix": np.ascontiguousarray(np.repeat(d.numpy(), L, axis=0).astype(np.float32)),
        "mds_coords": np.ascontiguousarray(np.repeat(z.numpy(), L, axis=0).astype(np.float32)),
    }


def taxa_reference(model, c, a, d, z):
    """splits.py's own extractor, on CPU, exactly as dating.py:2758 calls it."""
    with torch.no_grad():
        cache = model.precompute_tree_cache(d, z)
        return extract_cross_taxa_attentions_and_embeddings(model, c, a, cache, device="cpu")


def taxa_graph_accumulate(sess, c, a, d, z, batch, num_layers):
    """
    What runtime/ will do: feed every site in batches, accumulate the graph's
    per-call SUMS, divide once. Accumulation is float64 on purpose -- the graph
    emits float32 sums over its own batch, and adding ~L/batch of those in
    float32 would reintroduce exactly the reordering error this check is
    measuring. splits.py:151-153 divides by batch_size * num_layers and by
    batch_size; L is that batch_size when the whole alignment is one call.
    """
    L, N = c.shape[0], c.shape[1]
    cross = np.zeros((N, N), dtype=np.float64)
    repr_ = np.zeros((N, model_embed_dim(sess)), dtype=np.float64)
    dn, zn = d.numpy().astype(np.float32), z.numpy().astype(np.float32)
    for s in range(0, L, batch):
        e = min(L, s + batch)
        n = e - s
        feeds = {
            "msa_codons": c[s:e].numpy().astype(np.int64),
            "msa_aas": a[s:e].numpy().astype(np.int64),
            "dist_matrix": np.ascontiguousarray(np.repeat(dn, n, axis=0)),
            "mds_coords": np.ascontiguousarray(np.repeat(zn, n, axis=0)),
        }
        ca, tr = sess.run(TAXA_OUTPUT_NAMES, feeds)
        cross += ca.astype(np.float64)
        repr_ += tr.astype(np.float64)
    return cross / (L * num_layers), repr_ / L


def model_embed_dim(sess):
    for o in sess.get_outputs():
        if o.name == "taxa_repr_sum":
            return int(o.shape[1])
    raise RuntimeError("taxa_repr_sum not among the graph's outputs")


def check_rel(name, got, want, tol, absolute=False):
    """
    max|delta| against max|want|. Both figures are always printed so the noise
    floor stays visible whichever one is being gated on.
    """
    delta = np.abs(got.astype(np.float64) - want.astype(np.float64))
    scale = max(float(np.abs(want).max()), 1e-30)
    value = delta.max() if absolute else delta.max() / scale
    ok = bool(value <= tol)
    kind = "abs" if absolute else "rel"
    print(f"    {'PASS' if ok else 'FAIL'}  {name:<42} max|delta| = {delta.max():.3e}  "
          f"(rel {delta.max() / scale:.3e}, {kind} tol {tol:.0e})")
    if not ok:
        FAILURES.append(f"{name}: {kind} {value:.3e} > {tol:.0e}")
    return ok


def check_taxa_graph(models_dir, variant, weights, examples):
    """
    <variant>_taxa.onnx against splits.py extract_cross_taxa_attentions_and_embeddings.

    Three things are being proved, and only the third is about ONNX:
      1. The graph's inlined tree augmentation (it recomputes the Markov bias and
         the RoPE phases from dist_matrix / mds_coords) equals what
         precompute_tree_cache put in tree_cache, which splits.py reads.
      2. The reduction is right: mean over heads, sum over sites, accumulated
         over layers, root dropped from BOTH axes.
      3. The reduction is BATCH-INVARIANT -- the app never gets to make
         splits.py's single whole-alignment call, so the same matrix has to come
         back whatever site batching the runtime picks. Nothing else in this
         file tests that, and it is the property runtime/ depends on.
    It also carries both matrices through compute_neural_covariance_kernel,
    because centring and correlation-normalising is where a small error
    amplifies (see TOL_CROSS) and K_neural is what the estimators actually see.
    """
    path = models_dir / f"{variant}_taxa.onnx"
    if not path.exists():
        print(f"  [skip] {path.name} not present")
        return
    sess = ort_session(path)
    assert [i.name for i in sess.get_inputs()] == INPUT_NAMES
    assert [o.name for o in sess.get_outputs()] == TAXA_OUTPUT_NAMES
    model = build_backbone(weights)
    num_layers = max(1, len(model.row_layers))

    for name, (c, a, d, z, taxa, L) in examples.items():
        print(f"  {variant}_taxa on {name}: {len(taxa)} taxa x {L} codons, "
              f"{num_layers} row layers")
        want_cross, want_repr = taxa_reference(model, c, a, d, z)
        print(f"    reference row sums {want_cross.sum(axis=1).min():.4f}"
              f"..{want_cross.sum(axis=1).max():.4f} (not 1: the root column is dropped)")
        first = None
        for batch in (64, 128):
            got_cross, got_repr = taxa_graph_accumulate(sess, c, a, d, z, batch, num_layers)
            check_rel(f"{name} cross_attn (batch {batch})", got_cross, want_cross, TOL_CROSS_REL)
            check_rel(f"{name} taxa_repr (batch {batch})", got_repr, want_repr, TOL_REPR_REL)
            check_rel(f"{name} K_neural (batch {batch})",
                      compute_neural_covariance_kernel(got_cross, got_repr),
                      compute_neural_covariance_kernel(want_cross, want_repr),
                      TOL_K_NEURAL, absolute=True)
            if name == DATING_EXAMPLE:
                check_rel(f"{name} cross_attn dating budget (batch {batch})",
                          got_cross, want_cross, TOL_CROSS_ABS_DATING, absolute=True)
            if first is None:
                first = (got_cross, got_repr)
            else:
                check_rel(f"{name} cross_attn batch-invariance", got_cross, first[0], TOL_CROSS_REL)
                check_rel(f"{name} taxa_repr batch-invariance", got_repr, first[1], TOL_REPR_REL)


def load_example(name, fasta, nwk):
    c, a, d, z, inv, taxa, L = load_alignment_and_tree(str(fasta), str(nwk))
    print(f"  {name}: {len(taxa)} taxa x {L} codons, {int((~inv).sum())} variable sites")
    return c, a, d, z, inv, taxa, L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models-dir", default=str(REPO / "models"))
    ap.add_argument("--variants", default="general,viral")
    args = ap.parse_args()
    models_dir = Path(args.models_dir)
    hf_dir = models_dir / "_hf"
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    manifest = json.load(open(models_dir / "manifest.json"))
    print(f"[*] manifest {manifest['model_version']} reference {manifest['reference_version']} "
          f"exported_with {manifest['exported_with']}")

    print("[*] Loading examples")
    examples = {name: load_example(name, fa, nwk) for name, fa, nwk in EXAMPLES}

    # The dating graph's examples are loaded the way dating.py:2743 loads them,
    # not the way the backbone's are: prune_duplicates=False, max_species=None
    # and, with no tree, use_tn93=True. The dating pass keeps duplicate taxa and
    # caps nothing, so a graph checked on the backbone's tensors would be
    # checked on a different taxon set than the one it will run on.
    taxa_examples = {}
    for name, fasta, nwk in TAXA_EXAMPLES:
        if not Path(fasta).exists():
            print(f"  [skip] {name}: {fasta} not present")
            continue
        c, a, d, z, inv, taxa, L, _ = prepare_alignment(
            str(fasta), str(nwk) if nwk else None, model=None,
            max_species=None, prune_duplicates=False, use_tn93=(nwk is None),
        )
        print(f"  {name}: {len(taxa)} taxa x {L} codons"
              f"{' (tree-free, TN93)' if nwk is None else ''}, duplicates kept, uncapped")
        taxa_examples[name] = (c, a, d, z, taxa, L)

    root_repr_smc6_general = None

    for variant in args.variants.split(","):
        weights = resolve_variant_weights(variant, None, hf_dir)
        onnx_path = models_dir / f"{variant}.onnx"
        print(f"\n[*] Variant {variant}: {onnx_path.name} ({onnx_path.stat().st_size} bytes)")
        entry = manifest["variants"][variant]
        check(f"manifest.{variant}.onnx_sha256", 0.0 if sha256_file(onnx_path) == entry["onnx_sha256"] else 1.0, 0.0)
        check(f"manifest.{variant}.safetensors_sha256",
              0.0 if sha256_file(weights) == entry["safetensors_sha256"] else 1.0, 0.0)
        taxa_path = models_dir / f"{variant}_taxa.onnx"
        if taxa_path.exists():
            check(f"manifest.{variant}.taxa_onnx_sha256",
                  0.0 if sha256_file(taxa_path) == entry.get("taxa_onnx_sha256") else 1.0, 0.0)

        model = build_backbone(weights)
        sess = ort_session(onnx_path)
        in_names = [i.name for i in sess.get_inputs()]
        out_names = [o.name for o in sess.get_outputs()]
        assert in_names == INPUT_NAMES, in_names
        assert out_names == OUTPUT_NAMES, out_names

        for name, (c, a, d, z, inv, taxa, L) in examples.items():
            print(f"  -- {name} ({len(taxa)} taxa), one graph file, batch={L}")
            t0 = time.time()
            ref_lrt, ref_attn, ref_repr = reference_outputs(model, c, a, d, z)
            t_ref = time.time() - t0
            feeds = onnx_feeds(c, a, d, z)
            t0 = time.time()
            ort_lrt, ort_attn, ort_repr = sess.run(None, feeds)
            t_ort = time.time() - t0
            assert ort_lrt.shape == (L,), ort_lrt.shape
            assert ort_attn.shape == (L, len(taxa)), ort_attn.shape
            assert ort_repr.shape == (L, model.embed_dim), ort_repr.shape
            print(f"    torch {t_ref:.2f}s  onnxruntime {t_ort:.2f}s  "
                  f"lrt range [{ref_lrt.min():.3f}, {ref_lrt.max():.3f}]  "
                  f"attn row-sum mean {ort_attn.sum(1).mean():.4f}")
            check_lrt(f"{variant}/{name} lrt", ort_lrt, ref_lrt)
            check(f"{variant}/{name} mean_root_attns", float(np.abs(ort_attn - ref_attn).max()), TOL_ATTN)
            check(f"{variant}/{name} root_repr", float(np.abs(ort_repr - ref_repr).max()), TOL_REPR)
            if variant == "general" and name == "Smc6":
                root_repr_smc6_general = (ref_repr, inv)

            if variant == "viral":
                hf_onnx = hf_dir / "model.viral.onnx"
                if not hf_onnx.exists():
                    from huggingface_hub import hf_hub_download
                    hf_hub_download(HF_REPO_ID, "model.viral.onnx", local_dir=str(hf_dir))
                digest = sha256_file(hf_onnx)
                check("hf model.viral.onnx sha256", 0.0 if digest == HF_VIRAL_ONNX_SHA256 else 1.0, 0.0)
                hf_sess = ort_session(hf_onnx)
                (hf_lrt,) = hf_sess.run(None, feeds)
                check_lrt(f"viral/{name} lrt vs HF model.viral.onnx", ort_lrt, hf_lrt, TOL_HF_LRT)
                check_lrt(f"viral/{name} torch lrt vs HF model.viral.onnx", ref_lrt, hf_lrt, TOL_HF_LRT)

    # ------------------------------------------------------------------ BUSTED head
    if "general" in args.variants.split(","):
        print("\n[*] BUSTED head: models/busted_head.onnx vs BustedMultiTaskHead on Smc6 root_repr")
        weights = resolve_variant_weights("general", None, hf_dir)
        bh_path = models_dir / "busted_head.onnx"
        check("manifest.general.busted_head_onnx_sha256",
              0.0 if sha256_file(bh_path) == manifest["variants"]["general"]["busted_head_onnx_sha256"] else 1.0, 0.0)
        cov = busted_checkpoint_coverage(weights)
        print(f"    checkpoint covers {len(cov['loaded'])} head tensors; missing: {cov['missing_from_checkpoint']}; "
              f"unused: {cov['unused_in_checkpoint']}")

        ref_repr, inv = root_repr_smc6_general
        L = ref_repr.shape[0]
        hidden_all = np.zeros((1, L, ref_repr.shape[1]), dtype=np.float32)
        var_idx = np.where(~inv)[0]
        hidden_all[0, var_idx] = ref_repr[var_idx]
        mask = np.zeros((1, L), dtype=bool)

        head = build_busted_head(weights)
        with torch.no_grad():
            out = head(torch.from_numpy(hidden_all), mask=torch.from_numpy(mask))
        ref = {
            "cls_prob": out["cls_prob"].numpy(), "pred_gene_lrt": out["pred_lrt"].numpy(),
            "omega_prop": out["omega_prop"].numpy(), "syn_var": out["syn_var"].numpy(),
            "pred_omega3": out["pred_omega3"].numpy(), "pred_logp": out["pred_logp"].numpy(),
        }
        sess = ort_session(bh_path)
        assert [o.name for o in sess.get_outputs()] == BUSTED_OUTPUT_NAMES
        ort_out = dict(zip(BUSTED_OUTPUT_NAMES, sess.run(None, {"root_repr": hidden_all, "mask": mask})))
        for k in BUSTED_OUTPUT_NAMES:
            check(f"busted_head {k} (torch {np.array2string(ref[k].ravel(), precision=4)})",
                  float(np.abs(ort_out[k] - ref[k]).max()), TOL_BUSTED)

        head_other = build_busted_head(weights, seed=1)
        with torch.no_grad():
            other = head_other(torch.from_numpy(hidden_all), mask=torch.from_numpy(mask))
        print(f"    NOTE reference nondeterminism: cls_prob with init seed 0 = {ref['cls_prob'][0]:.4f}, "
              f"with seed 1 = {other['cls_prob'].item():.4f}, pred_gene_lrt {ref['pred_gene_lrt'][0]:.4f} vs "
              f"{other['pred_lrt'].item():.4f} (checkpoint lacks in_proj/head_syn_var/head_prop/theta_steps)")

    for variant in args.variants.split(","):
        weights = resolve_variant_weights(variant, None, hf_dir)
        print(f"\n[*] Dating graph {variant}_taxa.onnx vs splits.py extract_cross_taxa_attentions_and_embeddings")
        check_taxa_graph(models_dir, variant, weights, taxa_examples)

    print()
    if FAILURES:
        print(f"[!] {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"    {f}")
        sys.exit(1)
    print("[✓] All checks passed: one graph per variant serves 20 and 212 taxa; "
          "lrt / mean_root_attns / root_repr match the PyTorch reference; viral lrt matches HF model.viral.onnx.")


if __name__ == "__main__":
    main()
