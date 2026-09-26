"""
rhizaeon.cli
============
Command-Line Interface for RhizAeon.
"""

import argparse
import sys
import json
import time
from pathlib import Path
import numpy as np

from rhizaeon.tensor import encode_alignment_matrix, PrefixDistanceEngine, SNPCompressedPrefixEngine, parse_fasta
from rhizaeon.embed import build_prefix_engine
from rhizaeon.segmentation import RhizAeonDetector
from rhizaeon.fda import run_recursive_partition_fda_screen
from rhizaeon.alluvial import render_alluvial_genome_river
from rhizaeon.export import (
    export_hyphy_partition_json,
    export_nexus_partitions,
    export_hyphy_batchfile,
    export_split_fastas
)
from rhizaeon.visualizer import generate_interactive_html


def main():
    parser = argparse.ArgumentParser(
        description="RhizAeon: Tree-Free Reticulate Evolution & Recombination Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: scan
    scan_p = subparsers.add_parser("scan", help="Scan FASTA alignment for recombination breakpoints")
    scan_p.add_argument("alignment", type=str, help="Path to input nucleotide/codon FASTA alignment")
    scan_p.add_argument("--codon", action="store_true", default=True, help="Treat alignment as in-frame codons")
    scan_p.add_argument("--nt", action="store_false", dest="codon", help="Treat alignment as raw nucleotides")
    scan_p.add_argument("--window", type=int, default=25, help="Sliding window size (units)")
    scan_p.add_argument(
        "--min-tract",
        default="auto",
        help="Minimum recombinant tract length in codons/nt (default: 'auto'). When 'auto', dynamically calculates the Poisson mutation information limit: L_min = max(15, ceil(3.0 / mean_divergence))."
    )
    scan_p.add_argument(
        "--calibration",
        type=str,
        default="calibrated",
        choices=["calibrated", "strict"],
        help="Principled calibration profile: 'calibrated' (optimal F1; Z=2.75, PIR=0.20, FPR<=0.5%%) or 'strict' (conservative; Z=3.00, PIR=0.25, FPR=0.0%%)."
    )
    scan_p.add_argument(
        "--ghost-z",
        type=float,
        default=None,
        help="Manual override for Ghost Node Z-score significance threshold (default: derived from --calibration profile; scaled by Thompson-Grubbs bound for small cohorts)."
    )
    scan_p.add_argument(
        "--pir",
        type=float,
        default=None,
        help="Manual override for L-PIR incongruence threshold (default: derived from --calibration profile: calibrated=0.20, strict=0.25)."
    )
    scan_p.add_argument("--json", type=str, default=None, help="Save detection results to JSON")
    scan_p.add_argument("--verbose", "-v", action="store_true", default=False, help="Display extended per-breakpoint inspection details")
    scan_p.add_argument("--summary-only", "-s", action="store_true", default=False, help="Display only executive cohort summary and mosaic architecture maps without full catalog table")
    scan_p.add_argument("--max-rows", type=int, default=None, help="Maximum number of rows to display in breakpoint catalog table (default: all)")
    scan_p.add_argument("--alluvial", type=str, default=None, help="Render Alluvial Genome River plot to file")
    scan_p.add_argument(
        "--engine",
        type=str,
        default="two-tier",
        choices=[
            "two-tier", "two-tier-static", "two-tier-contextual", "hybrid",
            "scalar", "embed-static", "static", "384d",
            "embed-contextual", "contextual", "neural"
        ],
        help="Prefix distance engine: 'two-tier' (default: Tier 1 scalar sieve -> Tier 2 static 384D embeddings), 'two-tier-contextual' (Tier 1 scalar -> Tier 2 contextual transformer), 'scalar' (Hamming/TN93), 'embed-static' (standalone 384D token embeddings), or 'embed-contextual' (standalone 384D transformer hidden states)"
    )
    scan_p.add_argument(
        "--track",
        type=str,
        default="joint",
        choices=["joint", "ds", "dn"],
        help="Embedding track for embed-static / two-tier: 'joint' (384D codon+AA), 'ds' (192D synonymous codon only), or 'dn' (192D non-synonymous AA only)"
    )
    scan_p.add_argument(
        "--concordance-tol",
        type=int,
        default=60,
        help="Maximum distance (in units) for spatial concordance between Tier 1 screening candidate and Tier 2 breakpoint"
    )
    scan_p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Custom path to HyphAeon transformer weights (.pt)"
    )
    scan_p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Hardware accelerator for embed-contextual / two-tier-contextual"
    )
    scan_p.add_argument("--html", type=str, nargs="?", const="AUTO", default="AUTO", help="Generate standard self-contained interactive HTML dashboard (default: <alignment>_rhizaeon.html)")
    scan_p.add_argument("--no-html", action="store_true", default=False, help="Disable generating interactive HTML dashboard")
    scan_p.add_argument("--export-nexus", type=str, default=None, help="Export multi-partition NEXUS alignment")
    scan_p.add_argument("--export-hyphy-json", type=str, default=None, help="Export HyPhy partition JSON")
    scan_p.add_argument("--export-hyphy-bf", type=str, default=None, help="Export HyPhy batch script (.bf)")
    scan_p.add_argument("--export-partitions", type=str, default=None, help="Directory to export sliced non-recombinant FASTA files")

    # Subcommand: rp-fda
    rpfda_p = subparsers.add_parser("rp-fda", help="Run Recursive Partitioning FDA (RP-FDA) with ML Breakpoint Polisher")
    rpfda_p.add_argument("alignment", type=str, help="Path to input nucleotide FASTA alignment")
    rpfda_p.add_argument("--min-len", type=int, default=60, help="Minimum segment length (nt, default: 60)")
    rpfda_p.add_argument("--max-depth", type=int, default=4, help="Maximum recursion tree depth (default: 4)")
    rpfda_p.add_argument(
        "--calibration",
        type=str,
        default="calibrated",
        choices=["calibrated", "strict", "custom"],
        help="Principled calibration profile: 'calibrated' (optimal F1; Z=2.75, PIR=0.20), 'strict' (conservative; Z=3.00, PIR=0.25), or 'custom' (respects manual --min-z / --min-pir)."
    )
    rpfda_p.add_argument("--min-z", type=float, default=None, help="Kinetic Z-score threshold (default: derived from --calibration profile; scaled by finite-sample Thompson/Grubbs bound for N <= 5)")
    rpfda_p.add_argument("--min-pir", type=float, default=None, help="L-PIR incongruence threshold (default: derived from --calibration profile)")
    rpfda_p.add_argument("--frobenius-triage", action="store_true", default=False, help="Enable bilateral Frobenius pre-triage (156x speedup)")
    rpfda_p.add_argument("--no-polish", action="store_false", dest="polish_ml", default=True, help="Disable ML breakpoint polisher")
    rpfda_p.add_argument("--compress-snps", action="store_true", default=False, help="Use SNP-compressed prefix engine (220x RAM reduction)")
    rpfda_p.add_argument("--json", type=str, default=None, help="Save detection results to JSON")
    rpfda_p.add_argument("--verbose", "-v", action="store_true", default=False, help="Display extended per-breakpoint inspection details")
    rpfda_p.add_argument("--summary-only", "-s", action="store_true", default=False, help="Display only executive cohort summary and mosaic architecture maps without full catalog table")
    rpfda_p.add_argument("--max-rows", type=int, default=None, help="Maximum number of rows to display in breakpoint catalog table (default: all)")
    rpfda_p.add_argument("--html", type=str, nargs="?", const="AUTO", default="AUTO", help="Generate standard self-contained interactive HTML dashboard (default: <alignment>_rhizaeon.html)")
    rpfda_p.add_argument("--no-html", action="store_true", default=False, help="Disable generating interactive HTML dashboard")
    rpfda_p.add_argument("--export-nexus", type=str, default=None, help="Export multi-partition NEXUS alignment")
    rpfda_p.add_argument("--export-hyphy-json", type=str, default=None, help="Export HyPhy partition JSON")
    rpfda_p.add_argument("--export-hyphy-bf", type=str, default=None, help="Export HyPhy batch script (.bf)")
    rpfda_p.add_argument("--export-partitions", type=str, default=None, help="Directory to export sliced non-recombinant FASTA files")
    rpfda_p.add_argument("--tier2", action="store_true", dest="tier2", default=True, help="Enable automatic Tier 2 Transformer handoff (default: enabled)")
    rpfda_p.add_argument("--no-tier2", action="store_false", dest="tier2", help="Disable Tier 2 Transformer handoff")
    rpfda_p.add_argument("--weights", type=str, default=None, help="Custom path to Tier 2 transformer weights (.pt)")
    rpfda_p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Hardware accelerator for Tier 2 transformer")

    # Subcommand: alluvial
    alluvial_p = subparsers.add_parser("alluvial", help="Render Alluvial Genome River plot")
    alluvial_p.add_argument("alignment", type=str, help="Path to input FASTA alignment")
    alluvial_p.add_argument("--output", "-o", type=str, default="genome_river.png", help="Output image file")
    alluvial_p.add_argument("--codon", action="store_true", default=True, help="Treat alignment as in-frame codons")
    alluvial_p.add_argument("--highlight", nargs="+", default=None, help="Taxa headers to highlight")

    # Subcommand: visualize
    viz_p = subparsers.add_parser("visualize", help="Generate standard self-contained interactive HTML dashboard")
    viz_p.add_argument("alignment", type=str, help="Path to input nucleotide FASTA alignment")
    viz_p.add_argument("--output", "-o", type=str, default="AUTO", help="Output HTML path (default: <alignment>_rhizaeon.html)")
    viz_p.add_argument("--html", type=str, nargs="?", const="AUTO", default=None, help="Output HTML path alias")
    viz_p.add_argument("--no-html", action="store_true", default=False, help="Disable generating interactive HTML dashboard")
    viz_p.add_argument("--title", type=str, default=None, help="Dashboard title")

    args = parser.parse_args()

    if args.command == "scan":
        t0 = time.time()
        print(f"[*] Loading alignment: {args.alignment}")
        if not args.codon and args.engine in ("two-tier", "two-tier-static", "hybrid", "default"):
            args.engine = "scalar"
        if not args.codon and args.window == 25:
            args.window = 300
            if args.min_tract == 35:
                args.min_tract = 150
        engine = build_prefix_engine(
            args.alignment,
            engine=args.engine,
            codon=args.codon,
            track=args.track,
            weights_path=args.weights,
            device=args.device
        )
        taxa = getattr(engine, "taxa", parse_fasta(args.alignment)[0])
        L = getattr(engine, "L", getattr(engine, "L_nt", engine.num_units * (3 if args.codon else 1)))
        L_desc = f"{engine.num_units} {'codons' if args.codon else 'nt'}"
        engine_desc = f"{args.engine}"
        if args.engine in ("two-tier", "two-tier-static", "hybrid", "default"):
            engine_desc = f"Two-Tier (Tier 1 Scalar Sieve -> Tier 2 Static 384D [track: {args.track}])"
        elif args.engine in ("two-tier-contextual", "two-tier-transformer", "two-tier-neural"):
            engine_desc = f"Two-Tier (Tier 1 Scalar Sieve -> Tier 2 Contextual 384D Transformer)"
        elif args.engine in ("embed-static", "static", "384d"):
            engine_desc += f" (track: {args.track})"
        print(f"[*] Alignment matrix: {len(taxa)} taxa, {L_desc} | Engine: {engine_desc}")

        detector = RhizAeonDetector(
            window_units=args.window,
            min_tract_units=args.min_tract,
            ghost_z_threshold=args.ghost_z,
            pir_threshold=args.pir,
            calibration=args.calibration,
            concordance_tolerance=args.concordance_tol
        )
        eff_tract = detector.get_effective_min_tract(engine)
        print(f"[*] Calibration Profile: {detector.calibration.capitalize()} (Z >= {detector.ghost_z_threshold:.2f}, PIR >= {detector.pir_threshold:.2f}, Min Tract: {eff_tract} units)")

        print("[*] Running Recursive Binary Segmentation in sequence manifold space...")
        events = detector.detect_recombination(engine, taxa)
        elapsed = time.time() - t0

        from rhizaeon.reporting import synthesize_inference_report, format_humanized_report
        report = synthesize_inference_report(
            events_or_bps=events,
            taxa=taxa,
            alignment_path=args.alignment,
            alignment_len=int(engine.num_units if args.codon else L),
            unit_type="codon" if args.codon else "nt",
            elapsed_sec=elapsed,
            calibration=detector.calibration,
            z_threshold=detector.ghost_z_threshold,
            pir_threshold=detector.pir_threshold,
            min_tract=eff_tract
        )
        print(f"\n{format_humanized_report(report, verbose=getattr(args, 'verbose', False), max_rows=getattr(args, 'max_rows', None), summary_only=getattr(args, 'summary_only', False))}\n")

        if args.json:
            def _to_json_safe(obj):
                import numpy as _np
                if isinstance(obj, (_np.integer, int)):
                    return int(obj)
                if isinstance(obj, (_np.floating, float)):
                    return float(obj)
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {k: _to_json_safe(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_to_json_safe(v) for v in obj]
                return obj

            with open(args.json, "w") as f:
                json.dump({
                    "alignment": str(args.alignment),
                    "num_taxa": len(taxa),
                    "length": int(L),
                    "units": int(engine.num_units),
                    "unit_type": "codon" if args.codon else "nucleotide",
                    "runtime_sec": float(elapsed),
                    "num_breakpoints": len(events),
                    "summary": report.to_summary_dict(),
                    "events": _to_json_safe(events)
                }, f, indent=2)
            print(f"[✓] Saved JSON report to: {args.json}")

        if args.alluvial:
            print(f"[*] Rendering Alluvial Genome River plot to: {args.alluvial}")
            render_alluvial_genome_river(
                engine, taxa,
                recombination_events=events,
                window_units=args.window,
                output_path=args.alluvial
            )
            print(f"[✓] River plot saved to: {args.alluvial}")

        bps = [ev["breakpoint"] for ev in events]

        if args.export_nexus:
            print(f"[*] Exporting multi-partition NEXUS alignment to: {args.export_nexus}")
            export_nexus_partitions(args.alignment, bps, args.export_nexus, is_codon=args.codon)
            print(f"[✓] NEXUS file saved to: {args.export_nexus}")

        if args.export_hyphy_json:
            print(f"[*] Exporting HyPhy partition JSON to: {args.export_hyphy_json}")
            export_hyphy_partition_json(engine.num_units, bps, args.export_hyphy_json, is_codon=args.codon)
            print(f"[✓] HyPhy partition JSON saved to: {args.export_hyphy_json}")

        if args.export_hyphy_bf:
            print(f"[*] Exporting HyPhy batch script (.bf) to: {args.export_hyphy_bf}")
            export_hyphy_batchfile(args.alignment, bps, args.export_hyphy_bf, is_codon=args.codon)
            print(f"[✓] HyPhy batch script saved to: {args.export_hyphy_bf}")

        if args.export_partitions:
            print(f"[*] Exporting non-recombinant FASTA partitions to: {args.export_partitions}")
            created = export_split_fastas(args.alignment, bps, args.export_partitions, is_codon=args.codon)
            print(f"[✓] Exported {len(created)} partition FASTA files.")

        if args.html and not args.no_html:
            html_out = f"{Path(args.alignment).stem}_rhizaeon.html" if args.html == "AUTO" else args.html
            print(f"[*] Generating standard interactive HTML dashboard to: {html_out}")
            generate_interactive_html(
                alignment_path=args.alignment,
                detection_results=events,
                output_html_path=html_out,
                title=f"RhizAeon Scan Analysis: {Path(args.alignment).stem}"
            )
            print(f"[✓] Dashboard saved to: {html_out}")

    elif args.command == "alluvial":
        seq_mat, taxa, L = encode_alignment_matrix(args.alignment)
        engine = PrefixDistanceEngine(seq_mat, codon_aligned=args.codon)
        print(f"[*] Rendering Alluvial Genome River plot to: {args.output}")
        render_alluvial_genome_river(
            engine, taxa,
            recombination_events=None,
            highlight_taxa=args.highlight,
            output_path=args.output
        )
        print(f"[✓] Done.")

    elif args.command == "rp-fda":
        t0 = time.time()
        print(f"[*] Loading alignment: {args.alignment}")
        seq_mat, taxa, L = encode_alignment_matrix(args.alignment)
        N = len(taxa)
        
        if args.compress_snps:
            print(f"[*] Initializing SNPCompressedPrefixEngine (220x memory reduction)...")
            engine = SNPCompressedPrefixEngine(seq_mat, codon_aligned=False)
            print(f"[*] Compressed {L:,} nt down to {len(engine.seg_indices):,} segregating SNPs.")
        else:
            engine = PrefixDistanceEngine(seq_mat, codon_aligned=False)
            
        print(f"[*] Alignment matrix: {N} taxa, {L:,} nt")
        print(f"[*] Running Recursive Partitioning FDA (RP-FDA) screen (Frobenius triage={args.frobenius_triage}, ML polish={args.polish_ml})...")
        
        eff_min_len = args.min_len if args.min_len is not None else 60
        cal_profile = getattr(args, "calibration", "calibrated")
        if cal_profile == "strict":
            eff_min_z = 3.00 if args.min_z is None else args.min_z
            eff_min_pir = 0.25 if args.min_pir is None else args.min_pir
        elif cal_profile == "calibrated":
            eff_min_z = 2.75 if args.min_z is None else args.min_z
            eff_min_pir = 0.20 if args.min_pir is None else args.min_pir
        else:
            eff_min_z = 1.8 if args.min_z is None else args.min_z
            eff_min_pir = 0.08 if args.min_pir is None else args.min_pir

        print(f"[*] Calibration Profile: {cal_profile.capitalize()} (Z >= {eff_min_z:.2f}, PIR >= {eff_min_pir:.2f}, Min Length: {eff_min_len} nt)")

        bps = run_recursive_partition_fda_screen(
            engine=engine,
            taxa_names=taxa,
            min_len=eff_min_len,
            max_depth=args.max_depth,
            min_z=eff_min_z,
            min_pir=eff_min_pir,
            frobenius_triage=args.frobenius_triage,
            crossover_validation=True,
            polish_ml=args.polish_ml
        )

        # Tier 2 Dual Architecture Handover (Section 4 & 10 of Manuscript)
        if getattr(args, "tier2", True):
            try:
                from rhizaeon.adaptive import DualArchitectureConfig, evaluate_tier2_trigger, dispatch_tier2_transformer
                from rhizaeon.fda import FDABreakpoint
                from rhizaeon.polisher import polish_breakpoint_ml
                cfg = DualArchitectureConfig(
                    weights_path=getattr(args, "weights", None),
                    device=getattr(args, "device", "auto")
                )
                # Handover 1: For each candidate detected by Tier 1, evaluate trigger
                for b in bps:
                    num_snps = getattr(b, "num_informative_sites", None)
                    if num_snps is None and b.parent_1 in taxa and b.parent_2 in taxa and seq_mat is not None:
                        p1_idx = taxa.index(b.parent_1)
                        p2_idx = taxa.index(b.parent_2)
                        w_start = max(0, b.breakpoint_nt - 150)
                        w_end = min(L, b.breakpoint_nt + 150)
                        s1 = seq_mat[p1_idx, w_start:w_end]
                        s2 = seq_mat[p2_idx, w_start:w_end]
                        valid = (s1 < 4) & (s2 < 4) & (s1 != s2)
                        num_snps = int(np.sum(valid))
                    if num_snps is None:
                        num_snps = (1 if b.flanking_p1_site is not None else 0) + (1 if b.flanking_p2_site is not None else 0)

                    should_t2, reason = evaluate_tier2_trigger(
                        plateau_width=b.plateau_width or 0,
                        num_snps=num_snps,
                        pir_val=b.l_pir,
                        config=cfg
                    )
                    if should_t2:
                        t2 = dispatch_tier2_transformer(
                            fasta_path=args.alignment,
                            candidate_nt=b.breakpoint_nt,
                            uncertainty_window_nt=(b.ci_left or b.breakpoint_nt, b.ci_right or b.breakpoint_nt),
                            recombinant_taxon=b.recombinant_taxon,
                            config=cfg
                        )
                        if t2:
                            setattr(b, 'tier2_result', t2)
                            setattr(b, 'tier2_reason', reason)
                            setattr(b, 'tier2_resolved', True)
                            b.tier2_resolved = True

                # Handover 2: If Tier 1 found 0 breakpoints, evaluate Low-Divergence Mutational Voids (Trigger 2)
                if len(bps) == 0:
                    D_full = engine.query_distance_matrix(0, L)
                    max_div = np.max(D_full)
                    if 0.0 < max_div < 0.05:  # Low divergence regime (< 5% max distance / > 95% identity)
                        # Dispatch Tier 2 over central candidate window to isolate mosaic lineage via attention drift
                        t2 = dispatch_tier2_transformer(
                            fasta_path=args.alignment,
                            candidate_nt=L // 2,
                            uncertainty_window_nt=(L // 4, 3 * L // 4),
                            config=cfg
                        )
                        if t2 and t2.fiedler_divergence >= cfg.min_fiedler_div and t2.taxon_drift >= cfg.min_taxon_drift:
                            rec_name = t2.top_recombinant_taxon
                            r_idx = taxa.index(rec_name)
                            refined_nt = t2.refined_breakpoint_nt

                            D_l = engine.query_distance_matrix(0, refined_nt)
                            D_r = engine.query_distance_matrix(refined_nt, L)
                            cand_p1 = [i for i in range(N) if i != r_idx]
                            best_p1 = min(cand_p1, key=lambda i: D_l[r_idx, i])
                            cand_p2 = [i for i in range(N) if i != r_idx and i != best_p1]
                            best_p2 = min(cand_p2, key=lambda i: D_r[r_idx, i])

                            pol = polish_breakpoint_ml(seq_mat, coarse_bp=refined_nt, r_idx=r_idx, p1_idx=best_p1, p2_idx=best_p2, search_window=L//2, taxa_names=taxa)
                            should_t2, reason = evaluate_tier2_trigger(
                                plateau_width=pol.plateau_width,
                                num_snps=pol.num_informative_sites,
                                pir_val=0.0,
                                config=cfg
                            )
                            fb = FDABreakpoint(
                                breakpoint_nt=pol.polished_bp,
                                recombinant_taxon=rec_name,
                                taxon_idx=r_idx,
                                kinetic_z=2.0,
                                l_pir=0.0,
                                parent_1=taxa[best_p1],
                                parent_2=taxa[best_p2],
                                jump_magnitude=t2.fiedler_divergence,
                                coarse_bp=pol.coarse_bp,
                                polished_bp=pol.polished_bp,
                                ci_left=pol.ci_left,
                                ci_right=pol.ci_right,
                                plateau_width=pol.plateau_width,
                                log_likelihood_gain=pol.log_likelihood_gain,
                                flanking_p1_site=pol.flanking_p1_site,
                                flanking_p2_site=pol.flanking_p2_site,
                                num_informative_sites=pol.num_informative_sites,
                                tier2_resolved=True
                            )
                            setattr(fb, 'tier2_result', t2)
                            setattr(fb, 'tier2_reason', reason)
                            setattr(fb, 'tier2_resolved', True)
                            bps.append(fb)
            except Exception:
                pass

        elapsed = time.time() - t0
        
        from rhizaeon.reporting import synthesize_inference_report, format_humanized_report
        report = synthesize_inference_report(
            events_or_bps=bps,
            taxa=taxa,
            alignment_path=args.alignment,
            alignment_len=int(L),
            unit_type="nt",
            elapsed_sec=elapsed,
            calibration=cal_profile,
            z_threshold=eff_min_z,
            pir_threshold=eff_min_pir,
            min_tract=eff_min_len
        )
        print(f"\n{format_humanized_report(report, verbose=getattr(args, 'verbose', False), max_rows=getattr(args, 'max_rows', None), summary_only=getattr(args, 'summary_only', False))}\n")

        bp_coords = [b.breakpoint_nt for b in bps]

        if args.json:
            out_data = {
                "alignment": str(args.alignment),
                "num_taxa": N,
                "length": int(L),
                "runtime_sec": float(elapsed),
                "num_breakpoints": len(bps),
                "summary": report.to_summary_dict(),
                "breakpoints": [
                    {
                        "breakpoint_nt": int(b.breakpoint_nt),
                        "coarse_bp": int(b.coarse_bp) if b.coarse_bp is not None else None,
                        "polished_bp": int(b.polished_bp) if b.polished_bp is not None else None,
                        "ci_left": int(b.ci_left) if b.ci_left is not None else None,
                        "ci_right": int(b.ci_right) if b.ci_right is not None else None,
                        "plateau_width": int(b.plateau_width) if b.plateau_width is not None else None,
                        "log_likelihood_gain": float(b.log_likelihood_gain) if b.log_likelihood_gain is not None else None,
                        "flanking_p1_site": int(b.flanking_p1_site) if b.flanking_p1_site is not None else None,
                        "flanking_p2_site": int(b.flanking_p2_site) if b.flanking_p2_site is not None else None,
                        "recombinant": b.recombinant_taxon,
                        "parent_1": b.parent_1,
                        "parent_2": b.parent_2,
                        "kinetic_z": float(b.kinetic_z),
                        "l_pir": float(b.l_pir),
                        "tier2_verified": getattr(b, 'tier2_result', None) is not None,
                        "tier2_fiedler_divergence": float(b.tier2_result.fiedler_divergence) if getattr(b, 'tier2_result', None) is not None else None,
                        "tier2_taxon_drift": float(b.tier2_result.taxon_drift) if getattr(b, 'tier2_result', None) is not None else None,
                        "tier2_refined_bp": int(b.tier2_result.refined_breakpoint_nt) if getattr(b, 'tier2_result', None) is not None else None,
                        "tier2_refined_codon": int(b.tier2_result.refined_breakpoint_codon) if getattr(b, 'tier2_result', None) is not None else None,
                        "tier2_reason": getattr(b, 'tier2_reason', None)
                    } for b in bps
                ]
            }
            with open(args.json, "w") as f:
                json.dump(out_data, f, indent=2)
            print(f"[✓] Saved JSON report to: {args.json}")

        if args.export_nexus:
            print(f"[*] Exporting multi-partition NEXUS alignment to: {args.export_nexus}")
            export_nexus_partitions(args.alignment, bp_coords, args.export_nexus, is_codon=False)
            print(f"[✓] NEXUS file saved to: {args.export_nexus}")

        if args.export_hyphy_json:
            print(f"[*] Exporting HyPhy partition JSON to: {args.export_hyphy_json}")
            export_hyphy_partition_json(len(engine.seq_matrix[0]), bp_coords, args.export_hyphy_json, is_codon=False)
            print(f"[✓] HyPhy partition JSON saved to: {args.export_hyphy_json}")

        if args.export_hyphy_bf:
            print(f"[*] Exporting HyPhy batch script (.bf) to: {args.export_hyphy_bf}")
            export_hyphy_batchfile(args.alignment, bp_coords, args.export_hyphy_bf, is_codon=False)
            print(f"[✓] HyPhy batch script saved to: {args.export_hyphy_bf}")

        if args.export_partitions:
            print(f"[*] Exporting non-recombinant FASTA partitions to: {args.export_partitions}")
            created = export_split_fastas(args.alignment, bp_coords, args.export_partitions, is_codon=False)
            print(f"[✓] Exported {len(created)} partition FASTA files.")

        if args.html and not args.no_html:
            html_out = f"{Path(args.alignment).stem}_rhizaeon.html" if args.html == "AUTO" else args.html
            print(f"[*] Generating standard interactive HTML dashboard to: {html_out}")
            generate_interactive_html(
                alignment_path=args.alignment,
                detection_results=bps,
                output_html_path=html_out,
                title=f"RhizAeon RP-FDA Analysis: {Path(args.alignment).stem}"
            )
            print(f"[✓] Dashboard saved to: {html_out}")

    elif args.command == "visualize":
        if args.no_html:
            print("[*] HTML visualization suppressed by --no-html.")
            return
        t0 = time.time()
        print(f"[*] Building standard interactive dashboard for: {args.alignment}")
        target_out = args.html if (args.html and args.html != "AUTO") else args.output
        html_out = f"{Path(args.alignment).stem}_rhizaeon.html" if target_out == "AUTO" else target_out
        out_path = generate_interactive_html(
            alignment_path=args.alignment,
            detection_results=None,
            output_html_path=html_out,
            title=args.title
        )
        print(f"[✓] Generated interactive dashboard in {time.time()-t0:.2f}s: {out_path}")


if __name__ == "__main__":
    main()

