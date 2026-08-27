"""
axomeme/cli.py
--------------
Command-line interface for AxoMEME:
1. 'predict': Ultra-Fast Neural Inference of Episodic Positive Selection (AxoMEME Transformer)
2. 'phenotype' (phylowas): Directional Phenotype-Genotype Association & PARS Signature Extraction
3. 'epistasis' (essm): Multi-Scale Epistatic Sector Mining (Two-Stage Seed-and-Extend TSE)
"""

import os
import sys
import time
import json
import glob
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import networkx as nx

from .model import PhyloAxialTransformer, BustedMultiTaskHead
from .dataset import load_alignment_and_tree
from .weights import (
    resolve_weights_path,
    load_arch_config,
    load_weights,
    list_available_variants,
    DEFAULT_VARIANT,
    HF_REPO_ID,
)
from .inference import load_model
from .utils import select_device, lrt_to_pvals, benjamini_hochberg
from .phenotype import run_phenotype_association, PRESETS
from .epistasis import run_epistasis_analysis, run_epistatic_sector_mining

DEFAULT_VARIANT_ENV = os.environ.get("AXOMEME_VARIANT", DEFAULT_VARIANT)
# If set, AXOMEME_WEIGHTS points to a local weights file and bypasses HF download.
_local_repo_weights = Path(__file__).resolve().parent.parent / "weights" / "axomeme_v1.pt"
DEFAULT_WEIGHTS_ENV = os.environ.get("AXOMEME_WEIGHTS", str(_local_repo_weights) if _local_repo_weights.exists() else None)

def ensure_parent_directory(path):
    if path:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

def determine_adaptive_batch_size(num_species: int, total_sites: int, device: torch.device, user_batch_size: int = None) -> int:
    if user_batch_size is not None and user_batch_size > 0:
        return min(user_batch_size, total_sites)
        
    n = num_species + 1
    available_bytes = 4 * (1024 ** 3)
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            free_mem, _ = torch.cuda.mem_get_info(device)
            available_bytes = free_mem
        except Exception:
            available_bytes = 8 * (1024 ** 3)
    else:
        try:
            import psutil
            vm = psutil.virtual_memory()
            available_bytes = vm.available
        except Exception:
            available_bytes = 8 * (1024 ** 3)
            
    target_budget_bytes = max(int(available_bytes * 0.40), 256 * (1024 ** 2))
    bytes_per_site = 320 * (n ** 2) + 10000 * n + 4096
    calculated_batch = max(1, target_budget_bytes // bytes_per_site)
    return min(total_sites, int(calculated_batch))

def cmd_predict(args):
    device = select_device(cpu=args.cpu)
    print(f"[*] Hardware device selected: {device.type.upper()}")

    # Resolve weights: explicit --weights path > --model-variant (download from HF) > default variant
    try:
        weights_path = resolve_weights_path(
            weights=args.weights,
            variant=args.model_variant,
        )
    except RuntimeError as e:
        print(f"[!] {e}")
        sys.exit(1)
    print(f"[*] Loading AxoMEME model from: {weights_path}")

    model, device = load_model(weights=args.weights, variant=args.model_variant, device=device)
    
    print(f"[*] Parsing Alignment: {args.alignment}")
    if args.tree:
        print(f"[*] Parsing Tree:      {args.tree}")
    else:
        print(f"[*] Tree argument not provided; extracting tree from alignment...")
    
    t0 = time.time()
    try:
        prune_dups = not getattr(args, "no_prune_duplicates", False)
        c, a, d, z, inv, taxa, L = load_alignment_and_tree(
            args.alignment, args.tree, max_species=args.max_species, prune_duplicates=prune_dups
        )
    except Exception as e:
        print(f"\n[!] Error loading alignment and tree: {e}")
        sys.exit(1)

    variable_indices = np.where(~inv)[0]
    num_variable = len(variable_indices)

    batch_size = determine_adaptive_batch_size(len(taxa), max(1, num_variable), device, args.batch_size)
    num_chunks = (num_variable + batch_size - 1) // batch_size if num_variable > 0 else 0
    mode_desc = "manual override" if args.batch_size else "hardware adaptive"
    print(f"[*] Site Batch Sizing ({mode_desc}): {batch_size} sites/chunk ({num_chunks} chunk{'s' if num_chunks != 1 else ''} for {num_variable}/{L} variable codons)")
    
    d_dev = d.to(device)
    z_dev = z.to(device)
    tree_cache = model.precompute_tree_cache(d_dev, z_dev)
    
    lrts = np.zeros(L, dtype=np.float32)
    if num_variable > 0:
        with torch.no_grad():
            for start_idx in range(0, num_variable, batch_size):
                end_idx = min(start_idx + batch_size, num_variable)
                batch_site_idx = variable_indices[start_idx:end_idx]
                
                c_chunk = c[batch_site_idx].to(device)
                a_chunk = a[batch_site_idx].to(device)
                
                y_soft, _ = model.forward_cached(c_chunk, a_chunk, tree_cache)
                chunk_lrts = torch.clamp(y_soft.squeeze(-1), min=0.0).cpu().numpy().flatten()
                lrts[batch_site_idx] = chunk_lrts
                
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - t0
    
    pvals = lrt_to_pvals(lrts)
    qvals = benjamini_hochberg(pvals)
    
    sig_10 = (pvals <= 0.10).sum()
    sig_05 = (pvals <= 0.05).sum()
    fdr_05 = (qvals <= 0.05).sum()
    fdr_10 = (qvals <= 0.10).sum()
    
    print("\n" + "=" * 78)
    print(f"🎉 AxoMEME Selection Inference Complete in {elapsed:.3f} seconds!")
    print(f"   Taxa: {len(taxa)} | Codon Sites: {L} | Total Invariable: {inv.sum()}")
    print(f"   Nominal Significance: (p <= 0.05): {sig_05} | (p <= 0.10): {sig_10}")
    print(f"   FDR Significance:     (q <= 0.05): {fdr_05} | (q <= 0.10): {fdr_10}")
    print("=" * 78)
    
    top_indices = np.argsort(lrts)[::-1][:10]
    print("\nTop Candidate Sites for Episodic Positive Selection:")
    print(f"{'Codon':<8} {'LRT Score':<12} {'p-value':<12} {'FDR q-val':<12} {'Status':<15}")
    print("-" * 62)
    for idx in top_indices:
        l = lrts[idx]
        p = pvals[idx]
        q = qvals[idx]
        status = "p <= 0.05" if p <= 0.05 else ("p <= 0.10" if p <= 0.10 else "not significant")
        print(f"{idx+1:<8} {l:<12.3f} {p:<12.4e} {q:<12.4e} {status:<15}")
        
    results_list = [
        {
            "site": i + 1,
            "axomeme_lrt": float(lrts[i]),
            "p_value": float(pvals[i]),
            "q_value": float(qvals[i]),
            "is_invariable": bool(inv[i])
        }
        for i in range(L)
    ]
    
    tree_meta = args.tree if args.tree else "embedded_in_alignment"
    
    if args.output:
        ensure_parent_directory(args.output)
        with open(args.output, "w") as f:
            json.dump({
                "alignment": args.alignment,
                "tree": tree_meta,
                "taxa_count": len(taxa),
                "codon_count": L,
                "runtime_sec": elapsed,
                "sites": results_list
            }, f, indent=2)
        print(f"\n[✓] JSON results written to: {args.output}")
        
    if args.csv:
        ensure_parent_directory(args.csv)
        df = pd.DataFrame(results_list)
        df.to_csv(args.csv, index=False)
        print(f"[✓] CSV results written to: {args.csv}")

def cmd_busted(args):
    """
    Run Alignment-Wide Omnibus Selection Testing (BUSTED & BUSTED+S emulation).
    Combines site-level representations via ACAT Cauchy transformation and
    evaluates CORAL rank-consistent ordinal heads for exact calibrated selection calls.
    Supports single alignments (-a) or high-throughput batch directories (-d).
    """
    device = select_device(cpu=args.cpu)
    print(f"[*] Running HyphAeon BUSTED Omnibus Selection Inference on {device}...")
    t_global_start = time.time()
    
    # 1. Weights
    resolved_path = resolve_weights_path(args.weights, variant=args.variant)
    state_dict = load_weights(weights=resolved_path, variant=args.variant, map_location=device)
    arch_config = load_arch_config(weights=resolved_path, variant=args.variant)
    
    model = PhyloAxialTransformer(
        embed_dim=arch_config["embed_dim"],
        num_layers=arch_config["num_layers"],
        num_heads=arch_config["num_heads"],
        window_size=arch_config["window_size"],
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    busted_head = BustedMultiTaskHead(embed_dim=arch_config["embed_dim"]).to(device)
    busted_dict = {k.replace("head_busted.", ""): v for k, v in state_dict.items() if k.startswith("head_busted.")}
    if busted_dict:
        busted_head.load_state_dict(busted_dict, strict=False)
    busted_head.eval()

    # 2. Collect input files (single file or batch directory)
    input_files = []
    if getattr(args, "dir", None) and os.path.isdir(args.dir):
        patterns = getattr(args, "pattern", "*.aln,*.fa,*.fasta,*.nex,*.fna").split(',')
        for pat in patterns:
            pat = pat.strip()
            if pat:
                input_files.extend(glob.glob(os.path.join(args.dir, pat)))
        input_files = sorted(list(set(input_files)))
        print(f"[*] Discovered {len(input_files)} alignment files in '{args.dir}' for batch processing.")
    elif getattr(args, "alignment", None):
        if ',' in args.alignment:
            input_files = [f.strip() for f in args.alignment.split(',') if f.strip()]
        else:
            input_files = [args.alignment]
    else:
        print("[!] Error: You must specify either --alignment (-a) or --dir (-d).")
        sys.exit(1)

    if not input_files:
        print("[!] No matching alignment files found.")
        sys.exit(1)

    tree_suffix = getattr(args, "tree_suffix", ".raxml.bestTree")
    tree_dir = getattr(args, "tree_dir", None)
    prune_dups = not getattr(args, "no_prune_duplicates", False)

    batch_results = []
    total_sites_processed = 0

    is_batch = len(input_files) > 1
    if is_batch:
        print("\n" + "=" * 92)
        print(f"{'#':<4} {'Alignment':<20} {'Taxa':<6} {'Sites':<7} {'p_ACAT':<11} {'Prob':<8} {'w3':<8} {'Time':<8} {'Verdict'}")
        print("=" * 92)

    for file_idx, aln_path in enumerate(input_files, 1):
        gene_id = os.path.splitext(os.path.basename(aln_path))[0]
        tree_path = args.tree
        if tree_path is None:
            # Auto-locate matching tree
            if tree_dir:
                cand = os.path.join(tree_dir, os.path.basename(aln_path) + tree_suffix)
                if os.path.exists(cand): tree_path = cand
            else:
                cand1 = aln_path + tree_suffix
                cand2 = os.path.splitext(aln_path)[0] + ".nwk"
                cand3 = os.path.splitext(aln_path)[0] + ".tree"
                if os.path.exists(cand1): tree_path = cand1
                elif os.path.exists(cand2): tree_path = cand2
                elif os.path.exists(cand3): tree_path = cand3

        t0 = time.time()
        try:
            c, a, d, z, inv, taxa, L = load_alignment_and_tree(
                aln_path, tree_path, max_species=args.max_species, prune_duplicates=prune_dups
            )
        except Exception as e:
            if not is_batch:
                print(f"[!] Error loading {aln_path}: {e}")
            continue

        variable_indices = np.where(~inv)[0]
        num_variable = len(variable_indices)
        num_species = len(taxa)
        total_sites_processed += L
        
        batch_size = determine_adaptive_batch_size(num_species, max(1, num_variable), device, args.batch_size)
        tree_cache = model.precompute_tree_cache(d.to(device), z.to(device))
        
        lrts = np.zeros(L, dtype=np.float32)
        hidden_all = torch.zeros((1, L, arch_config["embed_dim"]), dtype=torch.float32)

        if num_variable > 0:
            with torch.no_grad():
                for start_idx in range(0, num_variable, batch_size):
                    end_idx = min(start_idx + batch_size, num_variable)
                    batch_site_idx = variable_indices[start_idx:end_idx]
                    c_chunk = c[batch_site_idx].to(device)
                    a_chunk = a[batch_site_idx].to(device)
                    y_soft, _, root_repr = model.forward_cached(c_chunk, a_chunk, tree_cache, return_hidden=True)
                    chunk_lrts = torch.clamp(y_soft.squeeze(-1), min=0.0).cpu().numpy().flatten()
                    lrts[batch_site_idx] = chunk_lrts
                    hidden_all[0, batch_site_idx] = root_repr.cpu()
                    
        if device.type == 'mps':
            torch.mps.synchronize()
        elif device.type == 'cuda':
            torch.cuda.synchronize()
            
        # 3. Neural BUSTED Head Evaluation
        with torch.no_grad():
            neural_out = busted_head(hidden_all.to(device))
            pred_prob_pos = float(neural_out["cls_prob"].item())
            pred_neural_lrt = float(neural_out["pred_lrt"].item()) if "pred_lrt" in neural_out else float((neural_out.get("sqrt_lrt", 0.0) ** 2).item())
            pred_syn_var = float(neural_out["syn_var"].item())
            pred_w3 = float(neural_out["pred_omega3"].item()) if "pred_omega3" in neural_out else float(neural_out.get("omega_vals", torch.tensor([1.0, 1.0, 1.0]))[2].item())
            pred_prop = neural_out["omega_prop"].squeeze().cpu().numpy()
            pred_omega = [0.10, 1.00, pred_w3]

        elapsed = time.time() - t0
        
        # 4. Asymptotic mixture p-values
        pvals = lrt_to_pvals(lrts.astype(np.float64))

        # 5. ACAT & Simes Combination
        var_p = pvals[variable_indices] if num_variable > 0 else pvals
        valid_p = np.clip(var_p, 1e-15, 1.0 - 1e-6)
        cauchy_terms = np.tan((0.5 - valid_p) * np.pi)
        t_acat = float(np.mean(cauchy_terms))
        p_acat = float(0.5 - (np.arctan(t_acat) / np.pi))
        p_acat = max(1e-15, min(1.0, p_acat))

        sorted_p = np.sort(pvals)
        ranks = np.arange(1, L + 1)
        p_simes = float(np.min((L / ranks) * sorted_p))
        p_simes = max(1e-15, min(1.0, p_simes))

        sig_sites_05 = int(np.sum(pvals < 0.05))
        sig_sites_10 = int(np.sum(pvals < 0.10))
        total_selection_energy = float(np.sum(lrts))
        omnibus_lrt = float(np.sum(np.maximum(0.0, lrts - 3.841)))
        is_significant = bool(p_acat < 0.05 or pred_prob_pos > 0.50)

        record = {
            "alignment": aln_path,
            "gene": gene_id,
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
            "elapsed_seconds": elapsed
        }
        batch_results.append(record)

        if is_batch:
            verdict_str = "✓ POSITIVE" if is_significant else "  Neutral"
            print(f"#{file_idx:<3d} {gene_id:<20s} {num_species:<6d} {L:<7d} {p_acat:<11.4e} {pred_prob_pos*100:<7.1f}% {pred_w3:<8.2f} {elapsed*1000:<6.1f}ms {verdict_str}")
        else:
            print("\n" + "=" * 78)
            print("                HYPHAEON BUSTED SELECTION INFERENCE RESULTS")
            print("=" * 78)
            print(f"  Alignment:                 {os.path.basename(aln_path)}")
            print(f"  Taxa Count:                {num_species}")
            print(f"  Codon Sites:               {L} ({num_variable} variable)")
            print(f"  Throughput:                {L / elapsed:.1f} sites/s ({elapsed * 1000:.1f} ms total)")
            print("-" * 78)
            print(f"  [Neural BUSTED Head]")
            print(f"    Positive Selection Prob: {pred_prob_pos * 100:.1f}%")
            print(f"    Predicted Gene LRT:      {pred_neural_lrt:.2f}")
            print(f"    Synonymous Variation:    Var(alpha) = {pred_syn_var:.4f}")
            print(f"  [Statistical Bridge (ACAT / Simes)]")
            print(f"    ACAT Omnibus p-value:    {p_acat:.4e}")
            print(f"    Simes Omnibus p-value:   {p_simes:.4e}")
            print(f"    Total Selection Energy:  {total_selection_energy:.2f}")
            print(f"    Significant Sites:       {sig_sites_05}/{L} (p<0.05), {sig_sites_10}/{L} (p<0.10)")
            print("-" * 78)
            print("  Inferred 3-Class Omega Mixture Distribution:")
            print(f"    Class 1 (Purifying):     omega_1 = {pred_omega[0]:.4f}  (proportion = {pred_prop[0]*100:.1f}%)")
            print(f"    Class 2 (Neutral):       omega_2 = {pred_omega[1]:.4f}  (proportion = {pred_prop[1]*100:.1f}%)")
            print(f"    Class 3 (Positive):      omega_3 = {pred_omega[2]:.4f}  (proportion = {pred_prop[2]*100:.1f}%)")
            print("-" * 78)
            if is_significant:
                print(f"  VERDICT: POSITIVE SELECTION DETECTED (Confidence: {max(pred_prob_pos*100, (1-p_acat)*100):.1f}%)")
            else:
                print("  VERDICT: No Evidence of Positive Selection (p >= 0.05)")
            print("=" * 78 + "\n")

    t_total = time.time() - t_global_start
    if is_batch:
        print("=" * 92)
        print(f"🎉 Batch Complete: Processed {len(batch_results)} alignments ({total_sites_processed} codons) in {t_total:.2f}s ({len(batch_results)/t_total:.1f} genes/s)")
        print("=" * 92)

    if args.output:
        ensure_parent_directory(args.output)
        with open(args.output, 'w') as f:
            json.dump(batch_results if is_batch else batch_results[0], f, indent=2)
        print(f"[✓] JSON results written to: {args.output}")

    if args.csv:
        ensure_parent_directory(args.csv)
        df_summary = pd.DataFrame([{
            "Gene": r["gene"],
            "Taxa": r["taxa"],
            "Sites": r["sites"],
            "p_ACAT": r["p_value_acat"],
            "p_Simes": r["p_value_simes"],
            "Selection_Prob": r["selection_probability"],
            "Pred_Gene_LRT": r["predicted_gene_lrt"],
            "Omnibus_LRT": r["omnibus_lrt"],
            "Omega_3": float(r["rate_distributions"]["omega_3"]),
            "Prop_Positive": float(r["rate_distributions"]["proportion_3"]),
            "Sig_Sites_p05": r["sig_sites_p05"],
            "Selected": r["positive_selection_detected"],
            "Time_ms": r["elapsed_seconds"] * 1000
        } for r in batch_results])
        df_summary.to_csv(args.csv, index=False)
        print(f"[✓] CSV summary written to: {args.csv}")

def list_models():
    """List available model variants from Hugging Face."""
    try:
        variants = list_available_variants()
    except Exception as e:
        print(f"[!] Could not fetch model list from Hugging Face: {e}")
        if "401" in str(e) or "Unauthorized" in str(e):
            print("    The model repo may be gated. Set HF_TOKEN env var to authenticate.")
            print("    Get a token at: https://huggingface.co/settings/tokens")
        return

    if not variants:
        print("No model variants found on Hugging Face.")
        return

    print(f"Available AxoMEME model variants ({HF_REPO_ID}):")
    print()
    for v in variants:
        default = " (default)" if v["variant"] == DEFAULT_VARIANT else ""
        print(f"  {v['variant']:15s}  {v['description']}{default}")
    print()
    print("Use with:  axomeme predict -a alignment.fa --model-variant <variant>")
    print(f"Default variant: {DEFAULT_VARIANT}")

def cmd_phenotype(args):
    print(f"[*] Executing Directional Phenotype-Genotype Mapping (PhyloWAS)...")
    print(f"[*] Alignment: {args.alignment}")
    if getattr(args, "tree", None):
        print(f"[*] Tree:      {args.tree}")
    else:
        print(f"[*] Tree:      (extracting from alignment)")
    
    t0 = time.time()
    try:
        res = run_phenotype_association(
            alignment_path=args.alignment,
            tree_path=getattr(args, "tree", None),
            weights_path=getattr(args, "weights", DEFAULT_WEIGHTS_ENV),
            variant=getattr(args, "variant", DEFAULT_VARIANT_ENV),
            preset=getattr(args, "preset", None),
            foreground=getattr(args, "foreground", None),
            background=getattr(args, "background", None),
            phenotype_file=getattr(args, "phenotype_file", None),
            trait_col=getattr(args, "trait_col", None),
            species_col=getattr(args, "species_col", None),
            continuous=getattr(args, "continuous", False),
            min_taxa_per_site=getattr(args, "min_taxa", 4),
            alpha=getattr(args, "alpha", 0.05),
            cpu=getattr(args, "cpu", False)
        )
    except Exception as e:
        print(f"\n[!] Phenotype Association Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    elapsed = time.time() - t0
    meta = res["phenotype_meta"]
    
    print("\n" + "=" * 88)
    print(f"🎉 PhyloWAS Directional Phenotype Association Complete in {elapsed:.3f} seconds!")
    print(f"   Target Trait: {meta['description']}")
    print(f"   Taxa: {res['taxa_count']} (Foreground: {meta.get('foreground_count', 'N/A')}) | Codon Sites: {res['codon_count']}")
    print(f"   Spectral Energy (Psi): {res['spectral_energy']:.4f} | Normalized Spectral Ratio: {res['norm_spectral_ratio']:.4f}")
    print(f"   Significant Sites (FDR q <= {args.alpha}): {res['significant_sites_count']}")
    print(f"   Compact PARS Signature: {res['compact_pars_signature']}")
    print("=" * 88)
    
    sites = res["sites"]
    top_sites = sites[:15]
    if top_sites:
        print("\nTop Trait-Associated Codon Sites (Ranked by Transformer Attribution Selection Score):")
        print(f"{'Site':<6} {'Ref':<5} {'Derived':<9} {'LRT':<7} {'Assoc (rho)':<12} {'Score':<8} {'p-value':<12} {'FDR q-val':<12} {'Fg %':<7} {'Bg %':<7}")
        print("-" * 92)
        for s in top_sites:
            q_str = f"{s.get('q_value', 1.0):.2e}" if s.get('q_value', 1.0) < 0.01 else f"{s.get('q_value', 1.0):.3f}"
            p_str = f"{s.get('p_value', 1.0):.2e}" if s.get('p_value', 1.0) < 0.01 else f"{s.get('p_value', 1.0):.3f}"
            print(f"{s['site']:<6d} {s['ref_aa']:<5s} {s['derived_aa']:<9s} {s['axomeme_lrt']:<7.2f} {s['association_rho']:<12.4f} {s.get('score', 0.0):<8.3f} {p_str:<12s} {q_str:<12s} {s['foreground_freq_pct']:<7.1f} {s['background_freq_pct']:<7.1f}")

    if args.output:
        ensure_parent_directory(args.output)
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n[✓] JSON results written to: {args.output}")

    if args.csv:
        ensure_parent_directory(args.csv)
        df = pd.DataFrame(sites)
        df.to_csv(args.csv, index=False)
        print(f"[✓] CSV results written to: {args.csv}")

def cmd_epistasis(args):
    print(f"[*] Executing Phylogenetic Branch Attribution, Co-Selection Networks & Selection DMS (ESSM)...")
    print(f"[*] Alignment: {args.alignment}")
    if args.tree:
        print(f"[*] Tree:      {args.tree}")
    else:
        print(f"[*] Tree:      (extracting from alignment)")
    
    t0 = time.time()
    try:
        res = run_epistasis_analysis(
            alignment_path=args.alignment,
            tree_path=args.tree,
            weights_path=args.weights,
            focal_taxon=getattr(args, "focal_taxon", None),
            min_sim=getattr(args, "min_sim", 0.30),
            min_shared=getattr(args, "min_shared", 2),
            max_fdr=getattr(args, "max_fdr", 0.05),
            min_lrt=getattr(args, "min_lrt", 1.0),
            min_clique_size=getattr(args, "min_clique_size", 3),
            max_overlap=getattr(args, "max_overlap", 0.50),
            run_dms=not getattr(args, "no_dms", False),
            cpu=getattr(args, "cpu", False)
        )
    except Exception as e:
        print(f"\n[!] Epistasis / ESSM Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0
    edges = res["edges"]
    sectors = res["sectors"]
    plasticity = res["plasticity"]
    
    print("\n" + "=" * 82)
    print(f"🎉 Epistatic Analysis Complete in {elapsed:.3f} seconds!")
    print(f"   Taxa: {res['taxa_count']} | Codons: {res['codon_count']} | Evaluated Branches: {res.get('branch_count', res.get('evaluated_branches', 0))}")
    max_fdr = getattr(args, "max_fdr", 0.05)
    print(f"   Co-Selection Edges (FDR q <= {max_fdr}): {len(edges)} | Discovered Sectors: {len(sectors)}")
    print("=" * 82)
    
    # 1. Top Co-Selection Pairs
    if edges:
        print("\nTop Phylogenetic Branch Co-Selection Pairs (Ranked by CESI):")
        print(f"{'Rank':<5} {'Residue Pair':<16} {'LRT 1/2':<12} {'Co-Sel':<9} {'Shared':<8} {'CESI':<10} {'FDR q-val':<12}")
        print("-" * 76)
        for idx, e in enumerate(edges[:12]):
            pair_str = f"{e['ref_u']}{e['site_u']} <-> {e['ref_v']}{e['site_v']}"
            lrt_str = f"{e['lrt_u']:.1f} / {e['lrt_v']:.1f}"
            q_str = f"{e['fdr_q']:.2e}" if e['fdr_q'] < 0.01 else f"{e['fdr_q']:.3f}"
            print(f"#{idx+1:<4d} {pair_str:<16s} {lrt_str:<12s} {e['similarity']:<9.4f} {e['shared_branches']:<8d} {e['cesi']:<10.3f} {q_str:<12s}")

    # 2. Epistatic Sectors
    if sectors:
        print("\nDiscovered Epistatic Sectors (Two-Stage Seed-and-Extend):")
        for sec in sectors[:8]:
            print(f"\nSector #{sec['sector_id']} (Size K = {sec['size']} residues): Sites {sec['sites']}")
            print(f"  • Spectral Coherence C(S): {sec['spectral_coherence']:.4f} | Mean LRT: {sec['mean_lrt']:.2f}")
            print(f"  • Consensus Signature: {sec['pars_signature']}")
            if sec.get("focal_taxon"):
                print(f"  • Focal Species ({sec['focal_taxon']}) Signature: {sec['focal_signature']}")
                if sec.get("focal_mutations"):
                    print(f"    - Focal Derived Shifts: {', '.join(sec['focal_mutations'])}")
                else:
                    print(f"    - Focal State: Invariant with consensus")

    # 3. Selection DMS Mutational Plasticity
    if plasticity:
        df_plas = pd.DataFrame(plasticity)
        top_plastic = df_plas.sort_values(by="intrinsic_plasticity", ascending=False).head(5)
        top_rigid = df_plas.sort_values(by="intrinsic_plasticity", ascending=True).head(5)
        
        print("\nSelection Deep Mutational Scanning (ESSM Intrinsic Plasticity):")
        print("  Top Permissive / Evolvable Sites (High Plasticity):")
        for _, r in top_plastic.iterrows():
            print(f"    • Site {int(r['site']):<4d} ({r['wt_aa']}): Plasticity = {r['intrinsic_plasticity']:.3f} | Baseline LRT = {r['baseline_lrt']:.2f} (p = {r['p_value']:.3e})")
            
        print("  Top Rigid / Catalytic Backbone Sites (Low Plasticity):")
        for _, r in top_rigid.iterrows():
            print(f"    • Site {int(r['site']):<4d} ({r['wt_aa']}): Plasticity = {r['intrinsic_plasticity']:.3f} | Baseline LRT = {r['baseline_lrt']:.2f} (p = {r['p_value']:.3e})")

    if getattr(args, "output", None):
        ensure_parent_directory(args.output)
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n[✓] JSON results written to: {args.output}")

    if getattr(args, "csv", None):
        ensure_parent_directory(args.csv)
        if getattr(args, "command", "") in ["dms", "essm", "digital-dms"] and plasticity:
            df_out = pd.DataFrame(plasticity)
        elif edges:
            df_out = pd.DataFrame(edges)
        elif plasticity:
            df_out = pd.DataFrame(plasticity)
        else:
            df_out = pd.DataFrame(sectors)
        df_out.to_csv(args.csv, index=False)
        print(f"[✓] CSV results written to: {args.csv}")

    if getattr(args, "graphml", None):
        ensure_parent_directory(args.graphml)
        G = nx.Graph()
        for e in edges:
            G.add_edge(
                str(e["site_u"]),
                str(e["site_v"]),
                weight=float(e["similarity"]),
                cesi=float(e["cesi"]),
                shared=int(e["shared_branches"]),
                fdr_q=float(e["fdr_q"])
            )
        nx.write_graphml(G, args.graphml)
        print(f"[✓] Co-selection network GraphML written to: {args.graphml}")

def main():
    parser = argparse.ArgumentParser(
        prog="axomeme",
        description="AxoMEME: Ultra-Fast Neural Selection Inference, Phenotype-Genotype Mapping, and Epistatic Sector Mining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Predict Subcommand
    pred_parser = subparsers.add_parser("predict", help="Run episodic positive selection inference (AxoMEME Transformer)")
    pred_parser.add_argument("-a", "--alignment", required=True, help="Path to in-frame codon FASTA or NEXUS alignment")
    pred_parser.add_argument("-t", "--tree", required=False, default=None, help="Path to Newick/NEXUS phylogenetic tree (optional if embedded)")
    pred_parser.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS_ENV, help="Path to local model weights file (overrides HF download). Can also be set via AXOMEME_WEIGHTS env var.")
    pred_parser.add_argument("--model-variant", default=DEFAULT_VARIANT_ENV, help=f"Model variant to download from Hugging Face (default: {DEFAULT_VARIANT})")
    pred_parser.add_argument("-b", "--batch-size", type=int, default=None, help="Site batch size (default: auto-selected)")
    pred_parser.add_argument("-s", "--max-species", type=int, default=None, help="Maximum number of species to include (PD downsampling)")
    pred_parser.add_argument("--no-prune-duplicates", action="store_true", help="Disable automatic collapsing of 100% identical sequence duplicates")
    pred_parser.add_argument("-o", "--output", help="Optional path to output JSON results")
    pred_parser.add_argument("-c", "--csv", help="Optional path to output CSV results")
    pred_parser.add_argument("--cpu", action="store_true", help="Force CPU inference")

    # 2. Phenotype Subcommand (PhyloWAS)
    pheno_parser = subparsers.add_parser("phenotype", aliases=["phylowas", "trait"], help="Run directional phenotype-genotype association & PARS signature extraction")
    pheno_parser.add_argument("-a", "--alignment", required=True, help="Path to in-frame codon FASTA or NEXUS alignment")
    pheno_parser.add_argument("-t", "--tree", default=None, help="Optional Newick/NEXUS phylogenetic tree (optional if embedded)")
    pheno_parser.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS_ENV, help="Path to local model weights file (overrides HF download)")
    pheno_parser.add_argument("--model-variant", dest="variant", default=DEFAULT_VARIANT_ENV, help=f"Model variant to download from HF (default: {DEFAULT_VARIANT})")
    pheno_parser.add_argument("-p", "--preset", choices=list(PRESETS.keys()), help=f"Curated phenotype preset: {', '.join(PRESETS.keys())}")
    pheno_parser.add_argument("-fg", "--foreground", help="Inline comma-separated list or regex pattern of foreground species")
    pheno_parser.add_argument("-bg", "--background", help="Optional explicit list of background control species")
    pheno_parser.add_argument("-pf", "--phenotype-file", help="Path to CSV/TSV metadata file mapping taxa to trait values")
    pheno_parser.add_argument("-tc", "--trait-col", help="Name of the trait column in phenotype file")
    pheno_parser.add_argument("-sc", "--species-col", help="Name of the species/taxa column in phenotype file")
    pheno_parser.add_argument("--continuous", action="store_true", help="Treat trait values as continuous phylogenetic contrasts")
    pheno_parser.add_argument("--min-taxa", type=int, default=4, help="Minimum sequenced taxa required per site")
    pheno_parser.add_argument("--alpha", type=float, default=0.05, help="FDR significance threshold")
    pheno_parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    pheno_parser.add_argument("-o", "--output", help="Optional path to output JSON results")
    pheno_parser.add_argument("-c", "--csv", help="Optional path to output CSV results")

    # 3. Epistasis Subcommand (Branch Co-Selection & Sectors)
    epi_parser = subparsers.add_parser("epistasis", aliases=["coselection", "sector", "network"], help="Run phylogenetic branch co-selection and epistatic sector mining")
    epi_parser.add_argument("-a", "--alignment", required=True, help="Path to in-frame codon FASTA or NEXUS alignment")
    epi_parser.add_argument("-t", "--tree", default=None, help="Optional Newick/NEXUS phylogenetic tree (optional if embedded)")
    epi_parser.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS_ENV, help="Path to local model weights file (overrides HF download)")
    epi_parser.add_argument("--focal-taxon", help="Focal taxon for in silico Selection DMS sweep (default: auto/consensus)")
    epi_parser.add_argument("--min-sim", type=float, default=0.30, help="Pairwise cosine similarity threshold for co-selection edges")
    epi_parser.add_argument("--min-shared", type=int, default=2, help="Minimum shared mutated phylogenetic branches")
    epi_parser.add_argument("--max-fdr", type=float, default=0.05, help="Benjamini-Hochberg FDR q-value threshold")
    epi_parser.add_argument("--min-lrt", type=float, default=1.0, help="Minimum site selection drive (LRT threshold)")
    epi_parser.add_argument("--min-clique-size", type=int, default=3, help="Minimum clique seed size for epistatic sectors")
    epi_parser.add_argument("--max-overlap", type=float, default=0.50, help="Maximum Jaccard overlap allowed between discovered sectors")
    epi_parser.add_argument("--no-dms", action="store_true", help="Skip 19-amino-acid in silico Selection DMS sweep")
    epi_parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    epi_parser.add_argument("-o", "--output", help="Optional path to output JSON results")
    epi_parser.add_argument("-c", "--csv", help="Optional path to output CSV results")
    epi_parser.add_argument("--graphml", help="Export co-selection network to GraphML for Cytoscape/Gephi")

    # 4. Digital DMS / ESSM Subcommand
    dms_parser = subparsers.add_parser("dms", aliases=["essm", "digital-dms"], help="Run in silico Selection Deep Mutational Scanning (Digital DMS / ESSM)")
    dms_parser.add_argument("-a", "--alignment", required=True, help="Path to in-frame codon FASTA or NEXUS alignment")
    dms_parser.add_argument("-t", "--tree", default=None, help="Optional Newick/NEXUS phylogenetic tree (optional if embedded)")
    dms_parser.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS_ENV, help="Path to local model weights file (overrides HF download)")
    dms_parser.add_argument("--focal-taxon", help="Focal taxon for in silico Selection DMS sweep (default: auto/consensus)")
    dms_parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    dms_parser.add_argument("-o", "--output", help="Optional path to output JSON results")
    dms_parser.add_argument("-c", "--csv", help="Optional path to output CSV results")

    # 5. BUSTED Omnibus Subcommand
    busted_parser = subparsers.add_parser("busted", aliases=["omnibus", "gene-selection"], help="Run alignment-wide omnibus episodic selection inference (BUSTED / BUSTED+S emulation)")
    busted_parser.add_argument("-a", "--alignment", default=None, help="Path to single in-frame codon alignment or comma-separated list")
    busted_parser.add_argument("-d", "--dir", default=None, help="Path to directory containing alignment files for high-throughput batch processing")
    busted_parser.add_argument("--pattern", default="*.aln,*.fa,*.fasta,*.nex,*.fna", help="Comma-separated glob patterns to match in --dir (default: *.aln,*.fa,*.fasta,*.nex,*.fna)")
    busted_parser.add_argument("-t", "--tree", default=None, help="Optional Newick/NEXUS phylogenetic tree (optional if embedded)")
    busted_parser.add_argument("--tree-suffix", default=".raxml.bestTree", help="Suffix to append to alignment filename to locate matching tree (default: .raxml.bestTree)")
    busted_parser.add_argument("--tree-dir", default=None, help="Optional directory containing corresponding phylogenetic trees")
    busted_parser.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS_ENV, help="Path to local model weights file (overrides HF download)")
    busted_parser.add_argument("--model-variant", dest="variant", default=DEFAULT_VARIANT_ENV, help=f"Model variant to download from HF (default: {DEFAULT_VARIANT})")
    busted_parser.add_argument("-s", "--max-species", type=int, default=512, help="Maximum number of taxa (Farthest-Point Traversal subsampling if exceeded)")
    busted_parser.add_argument("-b", "--batch-size", type=int, default=None, help="Number of codon sites to process in parallel (default: adaptive)")
    busted_parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    busted_parser.add_argument("-o", "--output", help="Optional path to output JSON results")
    busted_parser.add_argument("-c", "--csv", help="Optional path to output CSV results")

    # 6. List-models Subcommand
    list_parser = subparsers.add_parser("list-models", help="List available model variants from Hugging Face")

    args = parser.parse_args()
    if args.command == "predict":
        cmd_predict(args)
    elif args.command in ["busted", "omnibus", "gene-selection"]:
        cmd_busted(args)
    elif args.command in ["phenotype", "phylowas", "trait"]:
        cmd_phenotype(args)
    elif args.command in ["epistasis", "coselection", "sector", "network"]:
        cmd_epistasis(args)
    elif args.command in ["dms", "essm", "digital-dms"]:
        cmd_epistasis(args)
    elif args.command == "list-models":
        list_models()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
