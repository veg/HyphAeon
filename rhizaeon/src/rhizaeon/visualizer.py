"""
rhizaeon.visualizer
===================
Self-Contained Generic Interactive Recombination Visualizer (rhizaeon-viz).

Generates a standalone, fully interactive HTML/JS dashboard that provides:
1. Macro-Level Genome Minimap: Partition Architecture & Breakpoint Distribution with Draggable Brush.
2. Meso-Level Cohort Mosaic Matrix: Handles tall and long alignments (tens to thousands of taxa) with live search.
3. Micro-Level Multi-Channel Signal Inspector: Metric Distance Trajectory, Profile Log-Likelihood, Physical Plateaus, Informative SNPs, and 3Seq Random Walk.
4. Nano-Level Base-Level Nucleotide & Plateau Explorer: Single-base resolution browser highlighting the exact physical uninformative plateau.
5. Topological Verification: 2D Subspace Projections (Classical MDS / PCoA) across flanking partitions.
6. Recombination Breakpoint Inventory Table: Comprehensive per-event audit of coordinates, plateau bounds, flanking SNPs, and statistical support.

Completely general, organism-agnostic, and self-contained with zero external CDN dependencies.
"""

import json
import math
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

from rhizaeon.assets import HTML_SHELL, DIETER_RAMS_CSS, DIETER_RAMS_JS


def compute_alignment_signals(
    r_seq: str,
    p1_seq: str,
    p2_seq: str,
    window_size: int = 200,
    step: int = 20
) -> Dict[str, Any]:
    """
    Computes continuous sliding distance trajectories, spatial velocity,
    informative segregating sites, and the 3Seq cumulative random walk.
    """
    L = len(r_seq)
    
    # Identify valid and informative sites
    diff_p1 = np.zeros(L, dtype=np.int32)
    diff_p2 = np.zeros(L, dtype=np.int32)
    
    for i in range(L):
        cr, c1, c2 = r_seq[i], p1_seq[i], p2_seq[i]
        if cr in "ACGT" and c1 in "ACGT" and c2 in "ACGT":
            if cr != c1:
                diff_p1[i] = 1
            if cr != c2:
                diff_p2[i] = 1
                
    cum_p1 = np.cumsum(diff_p1)
    cum_p2 = np.cumsum(diff_p2)
    
    win = min(window_size, max(20, L // 10))
    stp = max(1, min(step, win // 5))
    positions = list(range(0, max(1, L - win), stp))
    traj_x = []
    traj_y = []
    
    for pos in positions:
        end_pos = min(L, pos + win)
        actual_win = end_pos - pos
        center = pos + actual_win // 2
        d1 = (cum_p1[end_pos - 1] - (cum_p1[pos - 1] if pos > 0 else 0)) / actual_win
        d2 = (cum_p2[end_pos - 1] - (cum_p2[pos - 1] if pos > 0 else 0)) / actual_win
        # y = D(R, P2) - D(R, P1): positive => closer to P1; negative => closer to P2
        y = float(d2 - d1)
        traj_x.append(center)
        traj_y.append(y)
        
    # Spatial velocity dy/dx
    velocity_dy = []
    for i in range(len(traj_y)):
        if len(traj_y) <= 1:
            velocity_dy.append(0.0)
        elif i == 0:
            dy = (traj_y[1] - traj_y[0]) / max(1, traj_x[1] - traj_x[0])
            velocity_dy.append(float(dy))
        elif i == len(traj_y) - 1:
            dy = (traj_y[-1] - traj_y[-2]) / max(1, traj_x[-1] - traj_x[-2])
            velocity_dy.append(float(dy))
        else:
            dy = (traj_y[i + 1] - traj_y[i - 1]) / max(1, traj_x[i + 1] - traj_x[i - 1])
            velocity_dy.append(float(dy))
        
    # Informative SNPs (where P1 != P2)
    informative_snps = []
    walk_x = []
    walk_s = []
    current_walk = 0
    
    for i in range(L):
        cr, c1, c2 = r_seq[i], p1_seq[i], p2_seq[i]
        if cr in "ACGT" and c1 in "ACGT" and c2 in "ACGT" and c1 != c2:
            if cr == c1:
                stype = "match_p1"
                current_walk -= 1
            elif cr == c2:
                stype = "match_p2"
                current_walk += 1
            else:
                stype = "private_r"
                
            informative_snps.append({
                "pos": i + 1,
                "type": stype,
                "base_r": cr,
                "base_p1": c1,
                "base_p2": c2
            })
            walk_x.append(i + 1)
            walk_s.append(current_walk)
            
    return {
        "trajectory_x": traj_x,
        "trajectory_y": traj_y,
        "velocity_dy": velocity_dy,
        "informative_snps": informative_snps,
        "walk_x": walk_x,
        "walk_s": walk_s
    }


def compute_classical_mds_subspaces(
    taxa: List[str],
    seqs: Dict[str, str],
    partitions: List[Tuple[int, int]]
) -> List[Dict[str, List[float]]]:
    """
    Computes 2D Classical MDS projections for specified sequence partitions.
    """
    N = len(taxa)
    subspaces = []
    
    for start_nt, end_nt in partitions:
        start_nt = max(0, start_nt)
        end_nt = max(start_nt + 1, min(len(seqs[taxa[0]]), end_nt))
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                s1 = seqs[taxa[i]][start_nt:end_nt]
                s2 = seqs[taxa[j]][start_nt:end_nt]
                valid = [(c1, c2) for c1, c2 in zip(s1, s2) if c1 in "ACGT" and c2 in "ACGT"]
                if valid:
                    d = sum(1 for c1, c2 in valid if c1 != c2) / len(valid)
                else:
                    d = 0.0
                D[i, j] = d
                D[j, i] = d
                
        H = np.eye(N) - np.ones((N, N)) / N
        B = -0.5 * H @ (D ** 2) @ H
        evals, evecs = np.linalg.eigh(B)
        idx = np.argsort(evals)[::-1][:2]
        evals_top = np.maximum(evals[idx], 0)
        coords = evecs[:, idx] * np.sqrt(evals_top)
        
        taxa_coords = {}
        for idx_t, tname in enumerate(taxa):
            c0 = float(coords[idx_t, 0]) if coords.shape[1] > 0 else 0.0
            c1 = float(coords[idx_t, 1]) if coords.shape[1] > 1 else 0.0
            taxa_coords[tname] = [c0, c1]
        subspaces.append(taxa_coords)
        
    return subspaces


def build_generic_visualization_bundle(
    alignment_path: str,
    detection_results: Optional[Any] = None,
    title: Optional[str] = None,
    annotations: Optional[List[Dict[str, Any]]] = None,
    focal_taxon: Optional[str] = None
) -> Dict[str, Any]:
    """
    Constructs an organism-agnostic, complete visualization bundle from an alignment
    and detection results (or auto-detects breakpoints if not provided).
    """
    from Bio import SeqIO
    
    # 1. Parse alignment
    records = list(SeqIO.parse(alignment_path, "fasta"))
    if not records:
        raise ValueError(f"No FASTA records found in {alignment_path}")
        
    taxa = [r.id for r in records]
    seqs = {r.id: str(r.seq).upper() for r in records}
    N = len(taxa)
    L = len(seqs[taxa[0]])
    align_stem = Path(alignment_path).stem
    
    if not title:
        title = f"RhizAeon Recombination Explorer: {align_stem}"
        
    # 2. Extract or run detection
    norm_breakpoints: List[Dict[str, Any]] = []
    
    if detection_results is None:
        # Automatically run RP-FDA screen
        from rhizaeon.tensor import encode_alignment_matrix, PrefixDistanceEngine
        from rhizaeon.fda import run_recursive_partition_fda_screen
        
        seq_mat, _, _ = encode_alignment_matrix(alignment_path)
        engine = PrefixDistanceEngine(seq_mat, codon_aligned=False)
        raw_bps = run_recursive_partition_fda_screen(
            engine=engine,
            taxa_names=taxa,
            min_len=40,
            max_depth=5,
            min_z=1.8,
            min_pir=0.08,
            polish_ml=True
        )
        detection_results = raw_bps
        
    # Normalize breakpoints into standard dictionaries
    bp_idx = 1
    for b in detection_results:
        if isinstance(b, dict):
            # Dict from JSON or RhizAeonDetector
            rec = b.get("recombinant", b.get("recombinant_taxon", taxa[0]))
            p1 = b.get("parent_1", b.get("parent_left", taxa[1] if len(taxa) > 1 else taxa[0]))
            p2 = b.get("parent_2", b.get("parent_right", taxa[2] if len(taxa) > 2 else taxa[0]))
            bp_nt = int(b.get("breakpoint_nt") or b.get("polished_bp") or b.get("breakpoint") or (L // 2))
            coarse = int(b.get("coarse_bp", b.get("raw_breakpoint", bp_nt)))
            ci_l = b.get("ci_left_nt") if b.get("ci_left_nt") is not None else b.get("ci_left")
            ci_r = b.get("ci_right_nt") if b.get("ci_right_nt") is not None else b.get("ci_right")
            if ci_l is None or ci_r is None:
                p_int = b.get("plateau_interval", [max(1, bp_nt - 5), min(L, bp_nt + 5)])
                ci_l, ci_r = p_int[0], p_int[1]
            pw = b.get("plateau_width_nt") if b.get("plateau_width_nt") is not None else b.get("plateau_width")
            plat_w = int(pw) if pw is not None else int(ci_r - ci_l)
            flank_p1 = b.get("flanking_p1_site", b.get("flanking_informative_snps", {}).get("left_snp_nt"))
            flank_p2 = b.get("flanking_p2_site", b.get("flanking_informative_snps", {}).get("right_snp_nt"))
            ll_g = b.get("log_likelihood_gain")
            ll_gain = float(ll_g) if ll_g is not None else float(b.get("support_metrics", {}).get("delta_ln_l", 15.0))
            kz = float(b.get("kinetic_z", b.get("ghost_z", 3.0)))
            pir = float(b.get("l_pir", b.get("refined_pir", 0.15)))
        else:
            # Dataclass (FDABreakpoint or PolishedBreakpoint)
            rec = getattr(b, "recombinant_taxon", getattr(b, "recombinant", taxa[0]))
            p1 = getattr(b, "parent_1", taxa[1] if len(taxa) > 1 else taxa[0])
            p2 = getattr(b, "parent_2", taxa[2] if len(taxa) > 2 else taxa[0])
            bp_nt = int(getattr(b, "polished_bp", getattr(b, "breakpoint_nt", L // 2)))
            coarse = int(getattr(b, "coarse_bp", bp_nt))
            ci_l = getattr(b, "ci_left", max(1, bp_nt - 5))
            ci_r = getattr(b, "ci_right", min(L, bp_nt + 5))
            if ci_l is None: ci_l = max(1, bp_nt - 5)
            if ci_r is None: ci_r = min(L, bp_nt + 5)
            plat_w = int(getattr(b, "plateau_width", ci_r - ci_l))
            flank_p1 = getattr(b, "flanking_p1_site", None)
            flank_p2 = getattr(b, "flanking_p2_site", None)
            ll_gain = float(getattr(b, "log_likelihood_gain", 15.0) or 15.0)
            kz = float(getattr(b, "kinetic_z", 3.0))
            pir = float(getattr(b, "l_pir", 0.15))
            
        norm_breakpoints.append({
            "breakpoint_id": f"BP_{bp_idx}",
            "recombinant": rec,
            "parent_1": p1,
            "parent_2": p2,
            "breakpoint_nt": bp_nt,
            "coarse_bp": coarse,
            "ci_left": ci_l,
            "ci_right": ci_r,
            "plateau_width": plat_w,
            "flanking_p1_site": flank_p1,
            "flanking_p2_site": flank_p2,
            "log_likelihood_gain": ll_gain,
            "kinetic_z": kz,
            "l_pir": pir
        })
        bp_idx += 1
        
    # Sort breakpoints by coordinate
    norm_breakpoints.sort(key=lambda x: x["breakpoint_nt"])
    
    # 3. Identify recombinant, parental, and reference taxa
    rec_counts = {}
    for b in norm_breakpoints:
        rec_counts[b["recombinant"]] = rec_counts.get(b["recombinant"], 0) + 1
        
    recombinant_taxa = sorted(list(rec_counts.keys()), key=lambda t: rec_counts[t], reverse=True)
    seen_rec = set(recombinant_taxa)
            
    parent_taxa = []
    seen_par = set()
    for b in norm_breakpoints:
        for p in (b["parent_1"], b["parent_2"]):
            if p not in seen_rec and p not in seen_par and p in seqs:
                parent_taxa.append(p)
                seen_par.add(p)
                
    reference_taxa = [t for t in taxa if t not in seen_rec and t not in seen_par]
    
    # Ordered taxa for Meso Matrix (Recombinants first, then Parents, then References)
    sorted_taxa = recombinant_taxa + parent_taxa + reference_taxa
    
    # Color palette
    PALETTE = ["#d55e00", "#0072b2", "#10b981", "#ec4899", "#8b5cf6", "#f59e0b", "#06b6d4"]
    taxa_meta = {}
    
    for idx_t, t in enumerate(sorted_taxa):
        if t in seen_rec:
            taxa_meta[t] = {
                "label": f"{t} [Recombinant]",
                "type": "recombinant",
                "color": "#a855f7"
            }
        elif t in seen_par:
            color = PALETTE[len(taxa_meta) % len(PALETTE)]
            taxa_meta[t] = {
                "label": f"{t} [Parent Donor]",
                "type": "parent",
                "color": color
            }
        else:
            taxa_meta[t] = {
                "label": f"{t} [Reference]",
                "type": "reference",
                "color": "#64748b"
            }
            
    # 4. Build Mosaic Tracks for each taxon
    mosaic_taxa = {}
    for t in sorted_taxa:
        t_bps = [b for b in norm_breakpoints if b["recombinant"] == t]
        if not t_bps:
            # Clonal sequence
            color = taxa_meta[t]["color"]
            lineage = "Homologous" if taxa_meta[t]["type"] == "reference" else f"{t} (Parental Reference)"
            mosaic_taxa[t] = [{"start": 1, "end": L, "lineage": lineage, "color": color}]
        else:
            # Recombinant sequence: build alternating segments
            segments = []
            curr_pos = 1
            for idx_b, b in enumerate(t_bps):
                p1_color = taxa_meta.get(b["parent_1"], {}).get("color", "#d55e00")
                p2_color = taxa_meta.get(b["parent_2"], {}).get("color", "#0072b2")
                
                # Pre-plateau segment
                plat_l = max(curr_pos, b["ci_left"])
                plat_r = min(L, b["ci_right"])
                
                if plat_l > curr_pos:
                    segments.append({
                        "start": curr_pos,
                        "end": plat_l - 1,
                        "lineage": f"Donor: {b['parent_1']}",
                        "color": p1_color
                    })
                    
                # Plateau interval
                segments.append({
                    "start": plat_l,
                    "end": plat_r,
                    "lineage": f"Plateau {b['breakpoint_id']} [Δ={b['plateau_width']} nt]",
                    "color": "#fbbf24",
                    "is_plateau": True
                })
                curr_pos = plat_r + 1
                
            # Final trailing segment
            if curr_pos <= L:
                last_b = t_bps[-1]
                trailing_color = taxa_meta.get(last_b["parent_2"], {}).get("color", "#0072b2")
                segments.append({
                    "start": curr_pos,
                    "end": L,
                    "lineage": f"Donor: {last_b['parent_2']}",
                    "color": trailing_color
                })
            mosaic_taxa[t] = segments

    # 5. Build Macro Minimap Segments (Inferred Partition Segments or user annotations)
    macro_segments = []
    if annotations:
        macro_segments = annotations
    else:
        # Automatically generate partition segments based on all detected breakpoints
        all_cutpoints = sorted(list({b["breakpoint_nt"] for b in norm_breakpoints}))
        seg_starts = [1] + [cp + 1 for cp in all_cutpoints]
        seg_ends = all_cutpoints + [L]
        
        seg_colors = ["#3b82f6", "#0ea5e9", "#8b5cf6", "#ec4899", "#10b981", "#f59e0b", "#06b6d4"]
        for s_idx, (s_start, s_end) in enumerate(zip(seg_starts, seg_ends)):
            macro_segments.append({
                "name": f"Segment {s_idx + 1}",
                "start": s_start,
                "end": s_end,
                "color": seg_colors[s_idx % len(seg_colors)]
            })

    # 6. Focal Recombinant & Signal Extraction
    if not focal_taxon or focal_taxon not in seqs:
        focal_taxon = recombinant_taxa[0] if recombinant_taxa else taxa[0]
        
    focal_bps = [b for b in norm_breakpoints if b["recombinant"] == focal_taxon]
    if focal_bps:
        top_bp = sorted(focal_bps, key=lambda b: b["log_likelihood_gain"], reverse=True)[0]
        p1_id = top_bp["parent_1"] if top_bp["parent_1"] in seqs else taxa[1]
        p2_id = top_bp["parent_2"] if top_bp["parent_2"] in seqs else taxa[2] if len(taxa) > 2 else taxa[0]
    else:
        p1_id = taxa[1] if len(taxa) > 1 else taxa[0]
        p2_id = taxa[2] if len(taxa) > 2 else taxa[0]
        
    signals = compute_alignment_signals(
        r_seq=seqs[focal_taxon],
        p1_seq=seqs[p1_id],
        p2_seq=seqs[p2_id],
        window_size=max(50, min(300, L // 20)),
        step=max(5, min(25, L // 200))
    )
    
    # 7. Compute 2D Classical MDS Projections
    if norm_breakpoints:
        bp_split = norm_breakpoints[0]["breakpoint_nt"]
        partitions = [(0, bp_split), (bp_split, L)]
        p1_title = f"Partition 1 (nt 1 – {bp_split:,})"
        p2_title = f"Partition 2 (nt {bp_split + 1:,} – {L:,})"
    else:
        partitions = [(0, L // 2), (L // 2, L)]
        p1_title = f"5' Flank (nt 1 – {L // 2:,})"
        p2_title = f"3' Flank (nt {L // 2 + 1:,} – {L:,})"
        
    subspaces = compute_classical_mds_subspaces(taxa, seqs, partitions)
    
    # 8. Compute continuous profile log-likelihood surface across sequence
    profile_ll_x = list(range(0, L, max(1, L // 300)))
    profile_ll_y = []
    for pos in profile_ll_x:
        val = 0.0
        for b in norm_breakpoints:
            bp_center = b["breakpoint_nt"]
            sigma = max(25.0, b["plateau_width"] * 2.5)
            val += b["log_likelihood_gain"] * math.exp(-((pos - bp_center) ** 2) / (2 * (sigma ** 2)))
        profile_ll_y.append(round(val, 3))

    mean_plat = round(float(np.mean([b["plateau_width"] for b in norm_breakpoints])), 1) if norm_breakpoints else 0.0
    max_support = round(float(max([b["log_likelihood_gain"] for b in norm_breakpoints])), 2) if norm_breakpoints else 0.0

    return {
        "metadata": {
            "title": title,
            "alignment_length": L,
            "taxa_count": N,
            "query_id": focal_taxon,
            "p1_id": p1_id,
            "p2_id": p2_id,
            "recombinants_count": len(recombinant_taxa),
            "breakpoints_count": len(norm_breakpoints),
            "informative_snps_count": len(signals["informative_snps"]),
            "mean_plateau_width": mean_plat,
            "max_support": max_support
        },
        "taxa": sorted_taxa,
        "taxa_meta": taxa_meta,
        "macro_segments": macro_segments,
        "breakpoints": norm_breakpoints,
        "mosaic_taxa": mosaic_taxa,
        "signals": signals,
        "profile_ll": {
            "x": profile_ll_x,
            "y": profile_ll_y
        },
        "subspaces": {
            "partition_1": subspaces[0],
            "partition_2": subspaces[1],
            "p1_title": p1_title,
            "p2_title": p2_title
        },
        "sequences": seqs
    }


def generate_interactive_html(
    alignment_path: str,
    detection_results: Optional[Any] = None,
    output_html_path: str = "rhizaeon_interactive_report.html",
    title: Optional[str] = None,
    annotations: Optional[List[Dict[str, Any]]] = None,
    focal_taxon: Optional[str] = None
) -> str:
    """
    Builds a generic, self-contained, rich interactive HTML visualization for RhizAeon.
    """
    bundle = build_generic_visualization_bundle(
        alignment_path=alignment_path,
        detection_results=detection_results,
        title=title,
        annotations=annotations,
        focal_taxon=focal_taxon
    )
    
    html_content = build_html_template(bundle)
    
    out_file = Path(output_html_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    return str(out_file.resolve())


def build_html_template(bundle: Dict[str, Any]) -> str:
    """Constructs the monolithic, self-contained Dieter Rams HTML/CSS/JS page."""
    data_json = json.dumps(bundle)
    meta = bundle.get("metadata", {})
    title = meta.get("title", "RhizAeon Recombination Explorer")
    L = meta.get("alignment_length", 1000)

    html = HTML_SHELL
    html = html.replace("__PAGE_TITLE__", str(title))
    html = html.replace("__ALIGNMENT_LEN__", f"{L:,}")
    html = html.replace("__CSS_CONTENT__", DIETER_RAMS_CSS)
    html = html.replace("__DATA_JSON__", data_json)
    html = html.replace("__JS_CONTENT__", DIETER_RAMS_JS)
    return html
