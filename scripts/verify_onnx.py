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
    sha256_file, INPUT_NAMES, OUTPUT_NAMES, BUSTED_OUTPUT_NAMES, HF_VIRAL_ONNX_SHA256, HF_REPO_ID,
)

TOL_LRT = 1e-5
TOL_ATTN = 1e-5
TOL_REPR = 1e-4
TOL_HF_LRT = 1e-5
TOL_BUSTED = 1e-5

EXAMPLES = [
    ("Smc6", REPO / "examples" / "Smc6.fasta", REPO / "examples" / "Smc6.nwk"),
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

    root_repr_smc6_general = None

    for variant in args.variants.split(","):
        weights = resolve_variant_weights(variant, None, hf_dir)
        onnx_path = models_dir / f"{variant}.onnx"
        print(f"\n[*] Variant {variant}: {onnx_path.name} ({onnx_path.stat().st_size} bytes)")
        entry = manifest["variants"][variant]
        check(f"manifest.{variant}.onnx_sha256", 0.0 if sha256_file(onnx_path) == entry["onnx_sha256"] else 1.0, 0.0)
        check(f"manifest.{variant}.safetensors_sha256",
              0.0 if sha256_file(weights) == entry["safetensors_sha256"] else 1.0, 0.0)

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
