"""
hyphaeon/cli.py
--------------
Command-line interface for HyphAeon:
1. 'meme': Ultra-Fast Neural Inference of Episodic Positive Selection (HyphAeon Transformer)
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
import scipy.stats as stats
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
from .phenotype import run_phenotype_association, PRESETS
from .epistasis import run_epistasis_analysis, run_epistatic_sector_mining
from .stats import pvals_from_lrt_meme, pvals_from_lrt_self_liang, benjamini_hochberg, cauchy_combination_p
from .inference import get_device, load_model, prepare_alignment, predict_site_lrts, compute_adaptive_safe_batch_size
from .io import ensure_parent_directory, write_json, write_csv, format_pq
from ._progress import ChunkProgress

DEFAULT_VARIANT_ENV = os.environ.get("HYPHAEON_VARIANT", DEFAULT_VARIANT)

# Default to package model.safetensors if it exists, otherwise check HYPHAEON_WEIGHTS
_local_repo_weights = Path(__file__).resolve().parent.parent / "model.safetensors"
DEFAULT_WEIGHTS_ENV = os.environ.get("HYPHAEON_WEIGHTS", str(_local_repo_weights) if _local_repo_weights.exists() else None)

def determine_adaptive_batch_size(num_species: int, total_sites: int, device: torch.device, user_batch_size: int = None) -> int:
    # DEPRECATED: retained for test/back-compat; delegates to compute_adaptive_safe_batch_size
    bs = compute_adaptive_safe_batch_size(num_species, user_batch_size=user_batch_size, device=device)
    return min(bs, total_sites)

def cmd_meme(args):
    device = get_device(cpu=getattr(args, "cpu", False))
    print(f"[*] Hardware device selected: {device.type.upper()}")

    try:
        model = load_model(weights=args.weights, variant=args.model_variant, device=device)
    except RuntimeError as e:
        print(f"[!] {e}")
        sys.exit(1)
    weights_path = resolve_weights_path(weights=args.weights, variant=args.model_variant)
    print(f"[*] Loading HyphAeon model from: {weights_path}")

    print(f"[*] Parsing Alignment: {args.alignment}")
    if args.tree:
        print(f"[*] Parsing Tree:      {args.tree}")
    else:
        print(f"[*] Tree argument not provided; extracting tree from alignment...")

    t0 = time.time()
    try:
        prune_dups = not getattr(args, "no_prune_duplicates", False)
        c, a, d, z, inv, taxa, L, tree_cache = prepare_alignment(
            args.alignment, args.tree, model=model, device=device,
            max_species=args.max_species, prune_duplicates=prune_dups
        )
    except Exception as e:
        print(f"\n[!] Error loading alignment and tree: {e}")
        sys.exit(1)

    variable_indices = np.where(~inv)[0]
    num_variable = len(variable_indices)

    batch_size = compute_adaptive_safe_batch_size(len(taxa), user_batch_size=args.batch_size, device=device)
    batch_size = min(batch_size, max(1, num_variable))
    num_chunks = (num_variable + batch_size - 1) // batch_size if num_variable > 0 else 0
    mode_desc = "manual override" if args.batch_size else "hardware adaptive"
    print(f"[*] Site Batch Sizing ({mode_desc}): {batch_size} sites/chunk ({num_chunks} chunk{'s' if num_chunks != 1 else ''} for {num_variable}/{L} variable codons)")

    lrts = predict_site_lrts(model, c, a, d, z, inv, tree_cache=tree_cache,
                             batch_size=batch_size, device=device, desc="Predict")
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - t0

    # MEME asymptotic mixture p-values + Benjamini-Hochberg FDR q-values
    pvals = pvals_from_lrt_meme(lrts).astype(np.float32)
    qvals = benjamini_hochberg(pvals).astype(np.float32)
    
    raw_sig_05 = int((pvals <= 0.05).sum())
    raw_sig_10 = int((pvals <= 0.10).sum())
    raw_fdr_05 = int((qvals <= 0.05).sum())
    raw_fdr_10 = int((qvals <= 0.10).sum())

    filtered_artifacts = []
    cleaned_alignment_path = None
    if getattr(args, "filter", False):
        print("\n[*] Running Automated Alignment Error Screening (Hypergeometric Patch + Counterfactual Outlier Attribution)...")
        from .dataset import parse_alignment_sequences, CODON_TO_AA
        from .filter import scan_hypergeometric_patches

        # 1. Hypergeometric Scan for selective patches
        patches = scan_hypergeometric_patches(
            pvals, alpha_site=0.05, min_k=3, max_span=35,
            p_local_thresh=getattr(args, "filter_p_thresh", 0.01)
        )

        if patches:
            raw_seqs = parse_alignment_sequences(args.alignment)
            cleaned_seqs = {t: list(raw_seqs[t]) for t in taxa if t in raw_seqs}
            
            # Compute consensus per site
            consensus_codons = []
            for s in range(L):
                scds = [raw_seqs[t][s*3:(s+1)*3].upper() for t in taxa if t in raw_seqs and len(raw_seqs[t]) >= (s+1)*3]
                valid = [cd for cd in scds if '-' not in cd and 'N' not in cd]
                consensus_codons.append(pd.Series(valid).mode().iloc[0] if valid else 'NNN')
                
            min_consec_thresh = getattr(args, "min_patch_consec", 3)
            for p in patches:
                p_start, p_end = p['start'], p['end']
                
                # Check each taxon for contiguous non-synonymous mutation runs
                tax_max_run = {}
                tax_total_muts = {}
                total_patch_muts = 0
                for t in taxa:
                    if t not in raw_seqs: continue
                    run, max_r, m_cnt = 0, 0, 0
                    for s in range(p_start, p_end + 1):
                        obs_cd = raw_seqs[t][s*3:(s+1)*3].upper()
                        con_cd = consensus_codons[s]
                        obs_aa = CODON_TO_AA.get(obs_cd, '-')
                        con_aa = CODON_TO_AA.get(con_cd, '-')
                        if obs_aa not in '-?' and con_aa not in '-?' and obs_aa != con_aa:
                            m_cnt += 1
                            run += 1
                            max_r = max(max_r, run)
                        else:
                            run = 0
                    tax_max_run[t] = max_r
                    tax_total_muts[t] = m_cnt
                    total_patch_muts += m_cnt
                    
                top_tax = max(taxa, key=lambda t: (tax_max_run.get(t, 0), tax_total_muts.get(t, 0)))
                top_run = tax_max_run.get(top_tax, 0)
                top_muts = tax_total_muts.get(top_tax, 0)
                oci = top_muts / (total_patch_muts + 1e-8)
                
                is_artifact = (top_run >= min_consec_thresh and oci >= 0.25) or (top_run >= 4)
                if is_artifact:
                    filtered_artifacts.append({
                        'start': p_start + 1,
                        'end': p_end + 1,
                        'span': p['d'],
                        'outlier_taxon': top_tax,
                        'consecutive_mismatches': top_run,
                        'oci': oci
                    })
                    # Mask the guilty outlier in the patch
                    for s in range(p_start, p_end + 1):
                        cleaned_seqs[top_tax][s*3 : (s+1)*3] = ['N', 'N', 'N']
                        
            if filtered_artifacts:
                # Re-evaluate cleaned alignment
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_fa:
                    for t in taxa:
                        if t in cleaned_seqs:
                            tmp_fa.write(f">{t}\n{''.join(cleaned_seqs[t])}\n")
                    cleaned_temp_path = tmp_fa.name
                    
                c_cl, a_cl, d_cl, z_cl, inv_cl, taxa_cl, L_cl = load_alignment_and_tree(
                    cleaned_temp_path, args.tree, max_species=args.max_species, prune_duplicates=prune_dups
                )
                lrts_cl = predict_site_lrts(model, c_cl, a_cl, d_cl, z_cl, inv_cl,
                                            tree_cache=tree_cache, batch_size=batch_size, device=device,
                                            desc="Predict (re-eval)")
                            
                pvals_cl = pvals_from_lrt_meme(lrts_cl).astype(np.float32)
                qvals_cl = benjamini_hochberg(pvals_cl).astype(np.float32)
                
                # Export cleaned alignment if requested
                if getattr(args, "filter_out_aln", None):
                    ensure_parent_directory(args.filter_out_aln)
                    with open(args.filter_out_aln, 'w') as out_f:
                        for t in taxa:
                            if t in cleaned_seqs:
                                out_f.write(f">{t}\n{''.join(cleaned_seqs[t])}\n")
                    cleaned_alignment_path = args.filter_out_aln
                    print(f"[✓] Cleaned in-frame alignment exported to: {args.filter_out_aln}")
                    
                if os.path.exists(cleaned_temp_path):
                    os.remove(cleaned_temp_path)
                    
                # Update inference arrays to cleaned version
                lrts = lrts_cl
                pvals = pvals_cl
                qvals = qvals_cl
                inv = inv_cl
                
    sig_10 = (pvals <= 0.10).sum()
    sig_05 = (pvals <= 0.05).sum()
    fdr_05 = (qvals <= 0.05).sum()
    fdr_10 = (qvals <= 0.10).sum()
    
    print("\n" + "=" * 78)
    print(f"🎉 HyphAeon Selection Inference Complete in {elapsed:.3f} seconds!")
    print(f"   Taxa: {len(taxa)} | Codon Sites: {L} | Total Invariable: {inv.sum()}")
    if getattr(args, "filter", False) and filtered_artifacts:
        print(f"   Automated Alignment Error Filtering: {len(filtered_artifacts)} artifact patch(es) surgically masked")
        for fa_item in filtered_artifacts:
            print(f"     ↳ Masked Codons {fa_item['start']}-{fa_item['end']} in [{fa_item['outlier_taxon']}] ({fa_item['consecutive_mismatches']} consec muts, OCI={fa_item['oci']*100:.1f}%)")
        print(f"   Raw vs Clean Significance: (p <= 0.05): {raw_sig_05} -> {sig_05} | (FDR q <= 0.10): {raw_fdr_10} -> {fdr_10}")
    elif getattr(args, "filter", False):
        print(f"   Automated Alignment Error Filtering: 0 artifacts detected (All patches authentic/clean)")
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
        
    attributions = {}
    if getattr(args, "attribute", False):
        print("\n[*] Running Mechanistic Feature Attribution (Single-Taxon Counterfactual Sensitivity)...")
        from .attribution import attribute_selection
        min_attr_lrt = getattr(args, "attribution_min_lrt", 3.84)
        attributions = attribute_selection(
            model, c, a, d, z, inv, taxa=taxa, min_lrt=min_attr_lrt, base_lrts=lrts, cache=tree_cache
        )
        if attributions:
            print(f"\nMechanistic Selection Attribution ({len(attributions)} sites with LRT >= {min_attr_lrt:.2f}):")
            print(f"{'Codon':<6} {'LRT':<7} {'Cons':<5} {'Epoch':<32} {'Top Driving Taxon & Mutation (ΔLRT, % explained)'}")
            print("-" * 105)
            for s_idx in sorted(attributions.keys()):
                rec = attributions[s_idx]
                pos = rec['site_1indexed']
                l = rec['predicted_lrt']
                cons = rec['consensus_aa']
                epoch = rec['when_selection_occurred']['evolutionary_epoch']
                top_sp = rec['driving_species'][:2]
                top_str = "; ".join([
                    f"{d['taxon'][:18]}: {cons}->{d['observed_aa']} (ΔLRT=+{d['delta_lrt']:.2f}, {d['pct_signal_explained']:.1f}%)"
                    for d in top_sp if d['delta_lrt'] > 0
                ]) or "Diffuse / Multi-Taxon Distributed"
                print(f"{pos:<6} {l:<7.2f} {cons:<5} {epoch:<32} {top_str}")
        else:
            print(f"[*] No sites met attribution threshold (LRT >= {min_attr_lrt:.2f})")

    results_list = []
    for i in range(L):
        site_entry = {
            "site": i + 1,
            "hyphaeon_lrt": float(lrts[i]),
            "p_value": float(pvals[i]),
            "q_value": float(qvals[i]),
            "is_invariable": bool(inv[i])
        }
        if i in attributions:
            rec = attributions[i]
            site_entry["evolutionary_epoch"] = rec['when_selection_occurred']['evolutionary_epoch']
            site_entry["adaptation_mode"] = rec['when_selection_occurred']['mode_of_adaptation']
            site_entry["top_driver"] = rec['driving_species'][0]['taxon'] if rec['driving_species'] else None
            site_entry["top_mutation"] = f"{rec['consensus_aa']}->{rec['driving_species'][0]['observed_aa']}" if rec['driving_species'] else None
            site_entry["attribution_details"] = rec
        results_list.append(site_entry)
    
    tree_meta = args.tree if args.tree else "embedded_in_alignment"
    
    if args.output:
        write_json(args.output, {
            "alignment": args.alignment,
            "tree": tree_meta,
            "taxa_count": len(taxa),
            "codon_count": L,
            "runtime_sec": elapsed,
            "filter_enabled": bool(getattr(args, "filter", False)),
            "artifacts_masked": filtered_artifacts,
            "attribution_enabled": bool(getattr(args, "attribute", False)),
            "attributions": {str(k+1): v for k, v in attributions.items()},
            "sites": results_list
        })

    if args.csv:
        # Flatten attribution details for clean CSV export
        csv_records = []
        for r in results_list:
            row_dict = {
                "site": r["site"],
                "hyphaeon_lrt": r["hyphaeon_lrt"],
                "p_value": r["p_value"],
                "q_value": r["q_value"],
                "is_invariable": r["is_invariable"],
            }
            if "evolutionary_epoch" in r:
                row_dict["evolutionary_epoch"] = r["evolutionary_epoch"]
                row_dict["adaptation_mode"] = r["adaptation_mode"]
                row_dict["top_driver"] = r["top_driver"]
                row_dict["top_mutation"] = r["top_mutation"]
            csv_records.append(row_dict)
        write_csv(args.csv, csv_records)

def cmd_busted(args):
    """
    Run Alignment-Wide Omnibus Selection Testing (BUSTED & BUSTED+S emulation).
    Combines site-level representations via ACAT Cauchy transformation and
    evaluates CORAL rank-consistent ordinal heads for exact calibrated selection calls.
    Supports single alignments (-a) or high-throughput batch directories (-d).
    """
    device = get_device(cpu=getattr(args, "cpu", False))
    print(f"[*] Running HyphAeon BUSTED Omnibus Selection Inference on {device}...")
    t_global_start = time.time()

    # 1. Weights
    resolved_path = resolve_weights_path(args.weights, variant=args.variant)
    state_dict = load_weights(weights=resolved_path, variant=args.variant)
    arch_config = load_arch_config(resolved_path, variant=args.variant)

    model = PhyloAxialTransformer(
        embed_dim=arch_config["embed_dim"],
        num_layers=arch_config["num_layers"],
        num_heads=arch_config["num_heads"],
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
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
        
        batch_size = compute_adaptive_safe_batch_size(num_species, user_batch_size=args.batch_size, device=device)
        batch_size = min(batch_size, max(1, num_variable))
        tree_cache = model.precompute_tree_cache(d.to(device), z.to(device))
        
        lrts = np.zeros(L, dtype=np.float32)
        hidden_all = torch.zeros((1, L, arch_config["embed_dim"]), dtype=torch.float32)

        if num_variable > 0:
            pb = ChunkProgress(num_variable, f'BUSTED {gene_id}', 'codon', enabled=(not is_batch and num_variable > 0))
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
                    pb.update(end_idx)
            pb.finish()

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

        # 4. Asymptotic mixture p-values (Self & Liang for BUSTED omnibus)
        pvals = pvals_from_lrt_self_liang(lrts)

        # 5. ACAT & Simes Combination
        var_p = pvals[variable_indices] if num_variable > 0 else pvals
        p_acat = cauchy_combination_p(var_p)

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
        write_json(args.output, batch_results if is_batch else batch_results[0])

    if args.csv:
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
        write_csv(args.csv, df_summary, label="CSV summary")

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

    print(f"Available HyphAeon model variants ({HF_REPO_ID}):")
    print()
    for v in variants:
        default = " (default)" if v["variant"] == DEFAULT_VARIANT else ""
        print(f"  {v['variant']:15s}  {v['description']}{default}")
    print()
    print("Use with:  hyphaeon meme -a alignment.fa --model-variant <variant>")
    print(f"Default variant: {DEFAULT_VARIANT}")

def cmd_phenotype(args):
    alignment_path = os.path.expanduser(args.alignment) if getattr(args, "alignment", None) else None
    tree_path = os.path.expanduser(args.tree) if getattr(args, "tree", None) else None
    weights_path = os.path.expanduser(args.weights) if getattr(args, "weights", None) else DEFAULT_WEIGHTS_ENV

    print(f"[*] Executing Directional Phenotype-Genotype Mapping (PhyloWAS)...")
    print(f"[*] Alignment: {alignment_path}")
    if tree_path:
        print(f"[*] Tree:      {tree_path}")
    else:
        print(f"[*] Tree:      (extracting from alignment)")
    
    t0 = time.time()
    try:
        res = run_phenotype_association(
            alignment_path=alignment_path,
            tree_path=tree_path,
            weights_path=weights_path,
            variant=getattr(args, "variant", DEFAULT_VARIANT_ENV),
            preset=getattr(args, "preset", None),
            foreground=getattr(args, "foreground", None),
            background=getattr(args, "background", None),
            phenotype_file=getattr(args, "phenotype_file", None),
            trait_col=getattr(args, "trait_col", None),
            species_col=getattr(args, "species_col", None),
            continuous=getattr(args, "continuous", False),
            permulations=getattr(args, "permulations", 0),
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
    if res.get("permulations_count", 0) > 0:
        p_str = f"{res['gene_p_value_perm']:.4e}" if res['gene_p_value_perm'] is not None else "N/A"
        print(f"   Phylogenetic Permulations: {res['permulations_count']} (Gene Empirical p = {p_str})")
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
            q_str = format_pq(s.get('q_value', 1.0))
            p_str = format_pq(s.get('p_value', 1.0))
            print(f"{s['site']:<6d} {s['ref_aa']:<5s} {s['derived_aa']:<9s} {s['hyphaeon_lrt']:<7.2f} {s['association_rho']:<12.4f} {s.get('score', 0.0):<8.3f} {p_str:<12s} {q_str:<12s} {s['foreground_freq_pct']:<7.1f} {s['background_freq_pct']:<7.1f}")

    trait_sectors = res.get("trait_sectors", [])
    if trait_sectors:
        print("\n" + "=" * 88)
        print(f"🧬 Inferred Epistatic Sectors Across Trait-Associated Sites ({len(trait_sectors)} passing C(S) >= 0.45):")
        for sec in trait_sectors:
            sec_sites_str = ", ".join([f"{sec['sites'][i]}" for i in range(len(sec['sites']))])
            print(f"  • Sector {sec['sector_id']}: Codons [ {sec_sites_str} ] | Size: {sec['size']} | Coherence C(S): {sec['spectral_coherence']:.3f} | Mean LRT: {sec['mean_lrt']:.2f}")

    coselection_pairs = res.get("coselection_pairs", [])
    if coselection_pairs:
        print("\nTop Co-Evolving Trait Pairs (Ranked by Composite Epistatic Selection Index CESI):")
        print(f"{'Pair':<16} {'Co-Sel (Sim)':<14} {'CESI':<8} {'Shared':<8} {'p-value':<12} {'FDR q-val':<12}")
        print("-" * 74)
        for p in coselection_pairs[:10]:
            pair_str = f"{p['ref_u']}{p['site_u']} - {p['ref_v']}{p['site_v']}"
            q_str = format_pq(p.get('q_value', 1.0))
            p_str = format_pq(p.get('p_value', 1.0))
            print(f"{pair_str:<16} {p['similarity']:<14.4f} {p['cesi']:<8.3f} {p['shared_branches']:<8d} {p_str:<12s} {q_str:<12s}")

    if args.output:
        write_json(args.output, res)

    if args.csv:
        write_csv(args.csv, sites)

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
            q_str = format_pq(e['fdr_q'])
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
        write_json(args.output, res)

    if getattr(args, "csv", None):
        if getattr(args, "command", "") in ["dms", "essm", "digital-dms"] and plasticity:
            df_out = pd.DataFrame(plasticity)
        elif edges:
            df_out = pd.DataFrame(edges)
        elif plasticity:
            df_out = pd.DataFrame(plasticity)
        else:
            df_out = pd.DataFrame(sectors)
        write_csv(args.csv, df_out)

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

def cmd_dms(args):
    print(f"[*] Executing in silico Selection Deep Mutational Scanning (Digital DMS / ESSM)...")
    print(f"[*] Alignment: {args.alignment}")
    if args.tree:
        print(f"[*] Tree:      {args.tree}")
    else:
        print(f"[*] Tree:      (extracting from alignment)")
        
    t0 = time.time()
    try:
        from .epistasis import run_digital_dms_analysis
        res = run_digital_dms_analysis(
            alignment_path=args.alignment,
            tree_path=args.tree,
            weights_path=args.weights,
            variant=getattr(args, "variant", None),
            focal_taxon=getattr(args, "focal_taxon", None),
            cpu=getattr(args, "cpu", False),
            progress=True
        )
    except Exception as e:
        print(f"\n[!] Digital DMS / ESSM Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    elapsed = time.time() - t0
    plasticity = res["plasticity"]
    
    print("\n" + "=" * 82)
    print(f"🎉 Digital DMS / ESSM Analysis Complete in {elapsed:.3f} seconds!")
    print(f"   Taxa: {res['taxa_count']} | Codons: {res['codon_count']} | Total Evaluated Single Mutants: {res['total_mutations']}")
    print("=" * 82)
    
    if plasticity:
        df_plas = pd.DataFrame(plasticity)
        top_plastic = df_plas.sort_values(by="intrinsic_plasticity", ascending=False).head(8)
        top_rigid = df_plas.sort_values(by="intrinsic_plasticity", ascending=True).head(8)
        
        print("\nTop Permissive / Evolvable Sites (High Plasticity Φ):")
        print(f"{'Site':<6} {'WT':<5} {'Plasticity Φ':<15} {'Baseline LRT':<15} {'p-value':<12} {'Max ΔLRT':<10}")
        print("-" * 65)
        for _, r in top_plastic.iterrows():
            p_str = format_pq(r['p_value'])
            print(f"{int(r['site']):<6d} {r['wt_aa']:<5s} {r['intrinsic_plasticity']:<15.4f} {r['baseline_lrt']:<15.2f} {p_str:<12s} {r['max_delta_lrt']:<10.2f}")
            
        print("\nTop Rigid / Catalytic Backbone Sites (Low Plasticity Φ):")
        print(f"{'Site':<6} {'WT':<5} {'Plasticity Φ':<15} {'Baseline LRT':<15} {'p-value':<12} {'Max ΔLRT':<10}")
        print("-" * 65)
        for _, r in top_rigid.iterrows():
            p_str = format_pq(r['p_value'])
            print(f"{int(r['site']):<6d} {r['wt_aa']:<5s} {r['intrinsic_plasticity']:<15.4f} {r['baseline_lrt']:<15.2f} {p_str:<12s} {r['max_delta_lrt']:<10.2f}")
            
    if getattr(args, "output", None):
        write_json(args.output, res)

    if getattr(args, "csv", None):
        df_out = pd.DataFrame(plasticity)
        if "mutant_deltas" in df_out.columns:
            df_out_csv = df_out.drop(columns=["mutant_deltas"])
        else:
            df_out_csv = df_out
        write_csv(args.csv, df_out_csv)

def cmd_disease(args):
    """Executes disease pathogenicity prediction for clinical variants."""
    from .disease import predict_disease_pathogenicity
    
    device = get_device(cpu=getattr(args, "cpu", False))
    print(f"[*] Running HyphAeon Disease Variant Pathogenicity Scoring on {device}...")
    
    resolved_path = resolve_weights_path(args.weights, variant=getattr(args, 'variant', DEFAULT_VARIANT))
    
    # Load canonical human sequence if provided
    canon_seq = None
    if getattr(args, "canonical_seq", None):
        if os.path.exists(args.canonical_seq):
            from Bio import SeqIO
            rec = next(SeqIO.parse(args.canonical_seq, "fasta"))
            canon_seq = str(rec.seq)
        else:
            canon_seq = args.canonical_seq.strip()
            
    df_res = predict_disease_pathogenicity(
        msa_path=args.alignment,
        mutations=args.mutations,
        canonical_human_seq=canon_seq,
        human_taxon=getattr(args, "human_taxon", None),
        weights_path=resolved_path,
        device=device,
        batch_size=getattr(args, "batch_size", None)
    )
    
    print("\n" + "=" * 90)
    print(f"{'Mutation':<12} {'Canonical Pos':<14} {'MSA Col':<9} {'Mapped':<8} {'Score':<10} {'Prediction'}")
    print("=" * 90)
    for _, r in df_res.head(25).iterrows():
        print(f"{r['mutation']:<12} {r['canonical_pos']:<14} {r['aln_col']:<9} {str(r['is_mapped']):<8} {r['pathogenicity_score']:<10.4f} {r['prediction']}")
    if len(df_res) > 25:
        print(f"... ({len(df_res) - 25} more mutations evaluated)")
    print("=" * 90)
    
    if getattr(args, "csv", None):
        write_csv(args.csv, df_res, label="CSV results")
    if getattr(args, "output", None):
        ensure_parent_directory(args.output)
        df_res.to_json(args.output, orient="records", indent=2)
        print(f"[*] Saved JSON results to: {args.output}")

def cmd_filter(args):
    """Executes automated alignment quality control, artifact detection, and surgical masking."""
    from .filter import run_alignment_filter
    
    print("[*] Executing Automated Alignment Error Detection & Surgical Masking...")
    print(f"[*] Alignment: {args.alignment}")
    if args.tree:
        print(f"[*] Tree:      {args.tree}")
        
    res = run_alignment_filter(
        alignment_path=args.alignment,
        tree_path=args.tree,
        weights_path=args.weights,
        model_variant=getattr(args, "variant", DEFAULT_VARIANT),
        output_alignment_path=args.output,
        audit_csv_path=args.csv,
        alpha_site=args.alpha_site,
        min_k=args.min_k,
        max_span=args.max_span,
        min_oci=args.min_oci,
        min_run_length=args.min_run_length,
        batch_size=getattr(args, "batch_size", None),
        max_species=getattr(args, "max_species", None),
        device=get_device(cpu=getattr(args, "cpu", False))
    )
    
    print("\n" + "=" * 80)
    print(f"🎉 Alignment Filtering Complete in {res['elapsed_seconds']:.3f} seconds!")
    print(f"   Taxa: {res['num_taxa']} | Codons: {res['num_codons']}")
    print(f"   Candidate Patches Detected: {res['num_patches_detected']} | Offender Artifacts Masked: {res['num_artifacts_masked']}")
    print(f"   Total Codons Surgically Masked (NNN): {res['masked_codons_count']}")
    print("=" * 80)
    
    raw = res['raw_metrics']
    cl = res['cleaned_metrics']
    print(f"\nMetric Comparison (Raw -> Cleaned):")
    print(f"  • Cauchy Omnibus p-value (CCT): {raw['cct_p_value']:.4e} -> {cl['cct_p_value']:.4e}")
    print(f"  • Significant Sites (p <= 0.05): {raw['sig_sites_p05']} -> {cl['sig_sites_p05']} (Suppressed: {res['suppressed_spurious_sites']})")
    print(f"  • Significant Sites (FDR q <= 0.10): {raw['sig_sites_q10']} -> {cl['sig_sites_q10']}")
    print(f"  • Mean Likelihood Ratio (LRT): {raw['mean_lrt']:.3f} -> {cl['mean_lrt']:.3f}")
    
    if res['artifacts']:
        print(f"\nSurgically Masked Frameshift / Sequencing Artifacts:")
        print(f"{'Patch (1-idx)':<15} {'Span':<6} {'k (p<=.05)':<12} {'p_hypergeom':<14} {'Outlier Taxon':<20} {'Run':<5} {'OCI':<6}")
        print("-" * 80)
        for a in res['artifacts']:
            p_range = f"{a['patch_start_1idx']}-{a['patch_end_1idx']}"
            print(f"{p_range:<15} {a['span_codons']:<6} {a['significant_sites_k']:<12} {a['p_hypergeom']:<14.2e} {a['outlier_taxon']:<20} {a['consecutive_mismatches']:<5} {a['outlier_contamination_index']:<6.2f}")
            
    if args.output:
        print(f"\n[✓] Cleaned alignment saved to: {args.output}")
    if args.csv:
        print(f"[✓] Artifact audit log written to: {args.csv}")

def main():

    parser = argparse.ArgumentParser(
        prog="hyphaeon",
        description="HyphAeon: Ultra-Fast Neural Selection Inference, Phenotype-Genotype Mapping, and Epistatic Sector Mining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. MEME Subcommand
    pred_parser = subparsers.add_parser("meme", aliases=["predict", "site-selection"], help="Run episodic positive selection inference (HyphAeon Transformer)")
    pred_parser.add_argument("-a", "--alignment", required=True, help="Path to in-frame codon FASTA or NEXUS alignment")
    pred_parser.add_argument("-t", "--tree", required=False, default=None, help="Path to Newick/NEXUS phylogenetic tree (optional if embedded)")
    pred_parser.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS_ENV, help="Path to local model weights file (overrides HF download). Can also be set via HYPHAEON_WEIGHTS env var.")
    pred_parser.add_argument("--model-variant", default=DEFAULT_VARIANT_ENV, help=f"Model variant to download from Hugging Face (default: {DEFAULT_VARIANT})")
    pred_parser.add_argument("-b", "--batch-size", type=int, default=None, help="Site batch size (default: auto-selected; very large values may be capped to a hardware-safe threshold to prevent GPU OOM)")
    pred_parser.add_argument("-s", "--max-species", type=int, default=None, help="Maximum number of species to include (PD downsampling)")
    pred_parser.add_argument("--no-prune-duplicates", action="store_true", help="Disable automatic collapsing of 100%% identical sequence duplicates")
    pred_parser.add_argument("-o", "--output", help="Optional path to output JSON results")
    pred_parser.add_argument("-c", "--csv", help="Optional path to output CSV results")
    pred_parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    pred_parser.add_argument("--filter", action="store_true", help="Enable automated dual-stage alignment error detection and surgical outlier masking (Hypergeometric Patch + Counterfactual Attribution)")
    pred_parser.add_argument("--filter-out-aln", help="Optional path to export cleaned in-frame codon FASTA alignment")
    pred_parser.add_argument("--filter-p-thresh", type=float, default=0.01, help="Hypergeometric local patch p-value threshold (default: 0.01)")
    pred_parser.add_argument("--min-patch-consec", type=int, default=3, help="Minimum consecutive radical mutations in single taxon to declare alignment artifact (default: 3)")
    pred_parser.add_argument("--attribute", action="store_true", help="Enable mechanistic feature attribution (single-taxon counterfactual sensitivity, Delta-LRT, and evolutionary epoch decomposition)")
    pred_parser.add_argument("--attribution-min-lrt", type=float, default=3.84, help="Minimum LRT threshold to run feature attribution (default: 3.84, corresponding to nominal p <= 0.05)")

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
    pheno_parser.add_argument("--permulations", type=int, default=0, help="Number of Brownian motion phylogenetic permulations for empirical p-values (RERconverge null model; default: 0 / parametric)")
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
    busted_parser.add_argument("-b", "--batch-size", type=int, default=None, help="Number of codon sites to process in parallel (default: adaptive; very large values may be capped to a hardware-safe threshold to prevent GPU OOM)")
    busted_parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    busted_parser.add_argument("-o", "--output", help="Optional path to output JSON results")
    busted_parser.add_argument("-c", "--csv", help="Optional path to output CSV results")

    # 6. Disease Pathogenicity Subcommand
    disease_parser = subparsers.add_parser("disease", aliases=["pathogenicity", "variant", "clinvar"], help="Predict disease variant effect and pathogenicity using HyphAeon Transformer")
    disease_parser.add_argument("-a", "--alignment", required=True, help="Path to in-frame codon FASTA or NEXUS alignment")
    disease_parser.add_argument("-m", "--mutations", required=True, help="List of mutations ('R175H,G245S'), CSV file, or Parquet path")
    disease_parser.add_argument("--canonical-seq", default=None, help="Optional canonical human reference sequence or FASTA path")
    disease_parser.add_argument("--human-taxon", default=None, help="Name of human reference taxon in alignment (default: auto-detected)")
    disease_parser.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS_ENV, help="Path to local model weights file (overrides HF download)")
    disease_parser.add_argument("--model-variant", dest="variant", default=DEFAULT_VARIANT_ENV, help=f"Model variant to download from HF (default: {DEFAULT_VARIANT})")
    disease_parser.add_argument("-b", "--batch-size", type=int, default=None, help="Batch size for site processing (default: adaptive hardware budget; very large values may be capped to a hardware-safe threshold to prevent GPU OOM)")
    disease_parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    disease_parser.add_argument("-o", "--output", help="Optional path to output JSON results")
    disease_parser.add_argument("-c", "--csv", help="Optional path to output CSV results")

    # 7. Alignment Filtering / Surgical Masking Subcommand
    filter_parser = subparsers.add_parser("filter", aliases=["mask", "qc", "clean"], help="Run automated alignment quality control, spatial artifact detection, and surgical masking")
    filter_parser.add_argument("-a", "--alignment", required=True, help="Path to in-frame codon FASTA or NEXUS alignment")
    filter_parser.add_argument("-t", "--tree", default=None, help="Optional Newick/NEXUS phylogenetic tree (optional if embedded)")
    filter_parser.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS_ENV, help="Path to local model weights file (overrides HF download)")
    filter_parser.add_argument("--model-variant", dest="variant", default=DEFAULT_VARIANT_ENV, help=f"Model variant to download from HF (default: {DEFAULT_VARIANT})")
    filter_parser.add_argument("-o", "--output", help="Path to write the cleaned, surgically masked alignment (FASTA format)")
    filter_parser.add_argument("-c", "--csv", help="Optional path to write artifact audit log (CSV format)")
    filter_parser.add_argument("-b", "--batch-size", type=int, default=None, help="Site batch size (default: adaptive hardware budget)")
    filter_parser.add_argument("-s", "--max-species", type=int, default=None, help="Maximum number of taxa to include (PD downsampling)")
    filter_parser.add_argument("--alpha-site", type=float, default=0.05, help="Site significance threshold for cluster scanning (default: 0.05)")
    filter_parser.add_argument("--min-k", type=int, default=3, help="Minimum significant sites within window (default: 3)")
    filter_parser.add_argument("--max-span", type=int, default=35, help="Maximum codon window span for spatial cluster (default: 35)")
    filter_parser.add_argument("--min-oci", type=float, default=0.25, help="Minimum Outlier Contamination Index threshold (default: 0.25)")
    filter_parser.add_argument("--min-run-length", type=int, default=3, help="Minimum consecutive mismatch run length in single taxon (default: 3)")
    filter_parser.add_argument("--cpu", action="store_true", help="Force CPU execution")

    # 8. List-models Subcommand
    list_parser = subparsers.add_parser("list-models", help="List available model variants from Hugging Face")

    # 7. Pooled HyphAeon-vs-MEME evaluation
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate folders of site predictions against matched HyPhy MEME results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    from .evaluation import configure_parser as configure_evaluation_parser
    configure_evaluation_parser(eval_parser)

    args = parser.parse_args()
    if args.command in ["meme", "predict", "site-selection"]:
        cmd_meme(args)
    elif args.command in ["busted", "omnibus", "gene-selection"]:
        cmd_busted(args)
    elif args.command in ["phenotype", "phylowas", "trait"]:
        cmd_phenotype(args)
    elif args.command in ["epistasis", "coselection", "sector", "network"]:
        cmd_epistasis(args)
    elif args.command in ["dms", "essm", "digital-dms"]:
        cmd_dms(args)
    elif args.command in ["disease", "pathogenicity", "variant", "clinvar"]:
        cmd_disease(args)
    elif args.command in ["filter", "mask", "qc", "clean"]:
        cmd_filter(args)
    elif args.command == "list-models":
        list_models()
    elif args.command == "evaluate":
        from .evaluation import EvaluationError, command as evaluate_command
        try:
            evaluate_command(args)
        except EvaluationError as exc:
            parser.error(str(exc))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()


