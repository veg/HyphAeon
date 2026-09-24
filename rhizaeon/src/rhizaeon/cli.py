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
    scan_p.add_argument("--min-tract", type=int, default=35, help="Minimum recombinant tract length (units)")
    scan_p.add_argument("--ghost-z", type=float, default=3.0, help="Ghost Node Z-score significance threshold")
    scan_p.add_argument("--pir", type=float, default=0.25, help="L-PIR incongruence threshold")
    scan_p.add_argument("--json", type=str, default=None, help="Save detection results to JSON")
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
    rpfda_p.add_argument("--min-len", type=int, default=40, help="Minimum segment length (nt)")
    rpfda_p.add_argument("--max-depth", type=int, default=5, help="Maximum recursion tree depth")
    rpfda_p.add_argument("--min-z", type=float, default=1.8, help="Kinetic Z-score threshold")
    rpfda_p.add_argument("--min-pir", type=float, default=0.08, help="L-PIR incongruence threshold")
    rpfda_p.add_argument("--frobenius-triage", action="store_true", default=False, help="Enable bilateral Frobenius pre-triage (156x speedup)")
    rpfda_p.add_argument("--no-polish", action="store_false", dest="polish_ml", default=True, help="Disable ML breakpoint polisher")
    rpfda_p.add_argument("--compress-snps", action="store_true", default=False, help="Use SNP-compressed prefix engine (220x RAM reduction)")
    rpfda_p.add_argument("--json", type=str, default=None, help="Save detection results to JSON")
    rpfda_p.add_argument("--html", type=str, nargs="?", const="AUTO", default="AUTO", help="Generate standard self-contained interactive HTML dashboard (default: <alignment>_rhizaeon.html)")
    rpfda_p.add_argument("--no-html", action="store_true", default=False, help="Disable generating interactive HTML dashboard")
    rpfda_p.add_argument("--export-nexus", type=str, default=None, help="Export multi-partition NEXUS alignment")
    rpfda_p.add_argument("--export-hyphy-json", type=str, default=None, help="Export HyPhy partition JSON")
    rpfda_p.add_argument("--export-hyphy-bf", type=str, default=None, help="Export HyPhy batch script (.bf)")
    rpfda_p.add_argument("--export-partitions", type=str, default=None, help="Directory to export sliced non-recombinant FASTA files")

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
    viz_p.add_argument("--title", type=str, default=None, help="Dashboard title")

    args = parser.parse_args()

    if args.command == "scan":
        t0 = time.time()
        print(f"[*] Loading alignment: {args.alignment}")
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
            concordance_tolerance=args.concordance_tol
        )

        print("[*] Running Recursive Binary Segmentation in sequence manifold space...")
        events = detector.detect_recombination(engine, taxa)
        elapsed = time.time() - t0

        print(f"\n================================================================================")
        print(f"RHIZAEON INFERENCE REPORT ({elapsed:.3f} seconds)")
        print(f"================================================================================")
        if len(events) == 0:
            print("[✓] Alignment is CLONAL / NON-RECOMBINANT (0 breakpoints detected).")
        else:
            print(f"[!] Detected {len(events)} Recombination Breakpoint(s):")
            for idx, ev in enumerate(events, 1):
                unit_str = "Codon" if args.codon else "Nucleotide"
                tier_tag = " [Two-Tier Verified]" if ev.get("tier") == "two-tier" else ""
                print(f"  {idx}. Breakpoint: {unit_str} {ev['breakpoint']}{tier_tag}")
                print(f"     Recombinant Lineage: {ev['recombinant']}")
                print(f"     Parental Transition: {ev['parent_left']} ---> {ev['parent_right']}")
                print(f"     Metrics: L-PIR = {ev.get('refined_pir', ev['l_pir']):.4f} | Ghost Node Z = {ev['ghost_z']:.2f}")

        print(f"================================================================================\n")

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
        
        bps = run_recursive_partition_fda_screen(
            engine=engine,
            taxa_names=taxa,
            min_len=args.min_len,
            max_depth=args.max_depth,
            min_z=args.min_z,
            min_pir=args.min_pir,
            frobenius_triage=args.frobenius_triage,
            polish_ml=args.polish_ml
        )
        elapsed = time.time() - t0
        
        print(f"\n================================================================================")
        print(f"RHIZAEON RP-FDA + ML POLISHER INFERENCE REPORT ({elapsed:.3f} seconds)")
        print(f"================================================================================")
        if len(bps) == 0:
            print("[✓] Alignment is CLONAL / NON-RECOMBINANT (0 breakpoints detected).")
        else:
            print(f"[!] Detected {len(bps)} Recombination Breakpoint(s):")
            for idx, b in enumerate(bps, 1):
                if b.ci_left is not None and b.ci_right is not None:
                    p1_site = str(b.flanking_p1_site) if b.flanking_p1_site is not None else "None"
                    p2_site = str(b.flanking_p2_site) if b.flanking_p2_site is not None else "None"
                    print(f"  {idx}. Breakpoint: {b.breakpoint_nt} nt (Coarse: {b.coarse_bp} nt)")
                    print(f"     ML Plateau: [{b.ci_left}, {b.ci_right}] (Width Δ={b.plateau_width} nt) | LL Gain: +{b.log_likelihood_gain:.2f}")
                    print(f"     Flanking SNPs: {p1_site} -> {p2_site}")
                else:
                    print(f"  {idx}. Breakpoint: {b.breakpoint_nt} nt")
                print(f"     Recombinant: {b.recombinant_taxon}")
                print(f"     Parental Transition: {b.parent_1} ---> {b.parent_2}")
                print(f"     Significance: Kinetic Z = {b.kinetic_z:.2f} | L-PIR = {b.l_pir:.4f}")

        print(f"================================================================================\n")

        bp_coords = [b.breakpoint_nt for b in bps]

        if args.json:
            out_data = {
                "alignment": str(args.alignment),
                "num_taxa": N,
                "length": int(L),
                "runtime_sec": float(elapsed),
                "num_breakpoints": len(bps),
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
                        "l_pir": float(b.l_pir)
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
        t0 = time.time()
        print(f"[*] Building standard interactive dashboard for: {args.alignment}")
        html_out = f"{Path(args.alignment).stem}_rhizaeon.html" if args.output == "AUTO" else args.output
        out_path = generate_interactive_html(
            alignment_path=args.alignment,
            detection_results=None,
            output_html_path=html_out,
            title=args.title
        )
        print(f"[✓] Generated interactive dashboard in {time.time()-t0:.2f}s: {out_path}")


if __name__ == "__main__":
    main()

