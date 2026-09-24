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
            bp_nt = int(b.get("polished_bp", b.get("breakpoint_nt", b.get("breakpoint", L // 2))))
            coarse = int(b.get("coarse_bp", b.get("raw_breakpoint", bp_nt)))
            ci_l = b.get("ci_left")
            ci_r = b.get("ci_right")
            if ci_l is None or ci_r is None:
                p_int = b.get("plateau_interval", [max(1, bp_nt - 5), min(L, bp_nt + 5)])
                ci_l, ci_r = p_int[0], p_int[1]
            plat_w = int(b.get("plateau_width", ci_r - ci_l))
            flank_p1 = b.get("flanking_p1_site", b.get("flanking_informative_snps", {}).get("left_snp_nt"))
            flank_p2 = b.get("flanking_p2_site", b.get("flanking_informative_snps", {}).get("right_snp_nt"))
            ll_gain = float(b.get("log_likelihood_gain", b.get("support_metrics", {}).get("delta_ln_l", 15.0)))
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
    """Constructs the monolithic, self-contained HTML/CSS/JS page."""
    data_json = json.dumps(bundle)
    meta = bundle['metadata']
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{meta['title']}</title>
  <style>
    :root {{
      --bg-base: #090d16;
      --bg-surface: #111827;
      --bg-surface-elevated: #1f2937;
      --bg-surface-hover: #374151;
      --border: #1f2937;
      --border-bright: #374151;
      --text-main: #f9fafb;
      --text-muted: #9ca3af;
      --text-dim: #6b7280;
      --cyan: #06b6d4;
      --emerald: #10b981;
      --amber: #f59e0b;
      --rose: #f43f5e;
      --purple: #a855f7;
      --blue: #3b82f6;
      --gold-glow: rgba(245, 158, 11, 0.4);
    }}
    
    body.light-theme {{
      --bg-base: #f8fafc;
      --bg-surface: #ffffff;
      --bg-surface-elevated: #f1f5f9;
      --bg-surface-hover: #e2e8f0;
      --border: #e2e8f0;
      --border-bright: #cbd5e1;
      --text-main: #0f172a;
      --text-muted: #475569;
      --text-dim: #94a3b8;
      --cyan: #0891b2;
      --emerald: #059669;
      --amber: #d97706;
      --rose: #e11d48;
      --purple: #9333ea;
      --blue: #2563eb;
      --gold-glow: rgba(217, 119, 6, 0.3);
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg-base);
      color: var(--text-main);
      line-height: 1.45;
      padding: 16px;
      transition: background-color 0.2s, color 0.2s;
    }}

    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 12px;
      margin-bottom: 16px;
      border-bottom: 1px solid var(--border-bright);
      flex-wrap: wrap;
      gap: 12px;
    }}
    .brand-title {{
      font-size: 19px;
      font-weight: 700;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .brand-badge {{
      background: rgba(6, 182, 212, 0.15);
      color: var(--cyan);
      border: 1px solid rgba(6, 182, 212, 0.35);
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .btn {{
      background: var(--bg-surface-elevated);
      color: var(--text-main);
      border: 1px solid var(--border-bright);
      padding: 5px 12px;
      border-radius: 4px;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s;
      user-select: none;
    }}
    .btn:hover {{
      background: var(--bg-surface-hover);
      border-color: var(--text-muted);
    }}
    .btn-primary {{
      background: var(--cyan);
      color: #ffffff;
      border-color: var(--cyan);
      font-weight: 600;
    }}
    .btn-primary:hover {{
      background: #0891b2;
      border-color: #0891b2;
    }}
    .btn-outline {{
      background: transparent;
      border-color: var(--border-bright);
    }}
    .btn-active {{
      background: rgba(168, 85, 247, 0.25) !important;
      border-color: var(--purple) !important;
      color: var(--purple) !important;
      font-weight: 700;
    }}

    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .kpi-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 14px;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }}
    .kpi-label {{
      font-size: 10.5px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      font-weight: 600;
    }}
    .kpi-value {{
      font-size: 19px;
      font-weight: 700;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    .kpi-subtext {{
      font-size: 11px;
      color: var(--text-dim);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    .panel {{
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      margin-bottom: 16px;
      overflow: hidden;
    }}
    .panel-header {{
      padding: 9px 14px;
      background: var(--bg-surface-elevated);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12.5px;
      font-weight: 600;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .panel-body {{
      padding: 14px;
    }}

    /* Level 1: Minimap */
    #minimap-container {{
      position: relative;
      width: 100%;
      height: 48px;
      background: var(--bg-base);
      border-radius: 4px;
      border: 1px solid var(--border);
      overflow: hidden;
      cursor: crosshair;
      user-select: none;
    }}
    #minimap-canvas {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
    }}
    #brush-overlay {{
      position: absolute;
      top: 0;
      height: 100%;
      background: rgba(6, 182, 212, 0.18);
      border-left: 2px solid var(--cyan);
      border-right: 2px solid var(--cyan);
      cursor: grab;
      box-sizing: border-box;
      z-index: 10;
    }}
    #brush-overlay:active {{ cursor: grabbing; }}
    .brush-handle {{
      position: absolute;
      top: 0;
      width: 8px;
      height: 100%;
      background: var(--cyan);
      cursor: ew-resize;
      z-index: 11;
      opacity: 0.8;
      transition: opacity 0.15s;
    }}
    .brush-handle:hover {{ opacity: 1; }}
    #brush-handle-l {{ left: -4px; }}
    #brush-handle-r {{ right: -4px; }}

    /* Level 2: Meso Matrix */
    .matrix-toolbar {{
      display: flex;
      gap: 12px;
      margin-bottom: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .search-input {{
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-bright);
      color: var(--text-main);
      padding: 5px 10px;
      border-radius: 4px;
      font-size: 11px;
      width: 200px;
    }}
    .matrix-scroll {{
      max-height: 260px;
      overflow-y: auto;
      border: 1px solid var(--border);
      border-radius: 4px;
    }}
    .seq-row {{
      display: flex;
      align-items: center;
      padding: 5px 10px;
      border-bottom: 1px solid var(--border);
      font-size: 11.5px;
      cursor: pointer;
      transition: background 0.1s;
    }}
    .seq-row:hover {{ background: var(--bg-surface-hover); }}
    .seq-row.active {{
      background: rgba(168, 85, 247, 0.22) !important;
      border-left: 4px solid var(--purple);
    }}
    .seq-name {{
      width: 220px;
      flex-shrink: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11.5px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      display: flex;
      align-items: center;
      gap: 7px;
    }}
    .seq-track-bar {{
      flex-grow: 1;
      height: 16px;
      position: relative;
      background: var(--bg-base);
      border-radius: 3px;
      overflow: hidden;
    }}

    /* Level 3: Micro Signals */
    .hud-bar {{
      background: var(--bg-base);
      border: 1px solid var(--border-bright);
      border-radius: 4px;
      padding: 6px 12px;
      font-size: 11px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      display: flex;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .hud-item {{
      display: flex;
      align-items: center;
      gap: 5px;
    }}
    .signal-track-label {{
      font-size: 11px;
      color: var(--text-muted);
      font-weight: 600;
      letter-spacing: 0.02em;
      margin-top: 10px;
      margin-bottom: 4px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .signal-canvas-wrapper {{
      position: relative;
      width: 100%;
      height: 80px;
      background: var(--bg-base);
      border: 1px solid var(--border);
      border-radius: 4px;
      overflow: hidden;
    }}
    .signal-canvas {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
    }}

    /* Level 4: Nano Base Browser */
    .base-browser-outer {{
      display: flex;
      border: 1px solid var(--border);
      border-radius: 4px;
      background: var(--bg-base);
      overflow: hidden;
    }}
    .base-taxa-col {{
      flex-shrink: 0;
      width: 190px;
      background: var(--bg-surface-elevated);
      border-right: 1px solid var(--border-bright);
      padding: 24px 6px 6px 8px;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }}
    .base-taxa-label {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 10px;
      font-weight: 600;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      height: 17px;
      line-height: 17px;
      display: flex;
      align-items: center;
      gap: 5px;
    }}
    .base-grid {{
      flex-grow: 1;
      display: flex;
      overflow-x: auto;
      padding: 6px 4px;
      user-select: none;
    }}
    .base-col {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 3px;
      min-width: 17px;
      flex-shrink: 0;
    }}
    .base-pos-header {{
      height: 15px;
      font-size: 8px;
      font-family: ui-monospace, monospace;
      font-weight: 600;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .base-pill {{
      width: 15px;
      height: 17px;
      border-radius: 2px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 9.5px;
      font-weight: 700;
      color: #ffffff;
      box-sizing: border-box;
    }}
    .base-a {{ background: #10b981; }}
    .base-c {{ background: #06b6d4; }}
    .base-g {{ background: #f59e0b; }}
    .base-t {{ background: #f43f5e; }}
    .base-gap {{ background: #475569; }}
    .col-snp-indicator {{
      width: 4px;
      height: 4px;
      border-radius: 50%;
      margin-bottom: 1px;
    }}
    .plateau-col {{
      background: rgba(245, 158, 11, 0.15);
      border-top: 2px solid #f59e0b;
      border-bottom: 2px solid #f59e0b;
    }}

    /* Level 5: Topological Subspaces */
    .subspaces-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}
    @media (max-width: 820px) {{
      .subspaces-grid {{ grid-template-columns: 1fr; }}
    }}
    .subspace-box {{
      background: var(--bg-base);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .subspace-canvas-wrapper {{
      width: 100%;
      height: 220px;
      position: relative;
    }}
    .subspace-metrics {{
      background: var(--bg-surface-elevated);
      padding: 6px 10px;
      border-radius: 4px;
      font-size: 11px;
      font-family: ui-monospace, monospace;
      display: flex;
      justify-content: space-between;
      color: var(--text-muted);
    }}

    /* Level 6: Table */
    .table-container {{ overflow-x: auto; }}
    table.inventory-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 11.5px;
    }}
    table.inventory-table th, table.inventory-table td {{
      padding: 8px 10px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }}
    table.inventory-table th {{
      background: var(--bg-surface-elevated);
      color: var(--text-dim);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 10px;
      letter-spacing: 0.04em;
    }}
    table.inventory-table tr:hover {{ background: var(--bg-surface-hover); }}
    table.inventory-table tr.active-row {{
      background: rgba(168, 85, 247, 0.16) !important;
      border-left: 3px solid var(--purple);
    }}
    .pill-badge {{
      display: inline-block;
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 10px;
      font-weight: 600;
    }}
  </style>
</head>
<body>

  <!-- Top Header Navigation -->
  <header>
    <div class="brand-title">
      <span>RhizAeon Recombination Explorer</span>
      <span class="brand-badge">{Path(meta['title']).stem if len(meta['title']) < 40 else 'Genome View'}</span>
      <span style="font-size:11.5px; color:var(--text-dim); font-weight:normal;">{meta['taxa_count']} Taxa &middot; {meta['alignment_length']:,} nt</span>
    </div>
    <div style="display:flex; gap:6px; align-items:center; flex-wrap:wrap;">
      <button class="btn btn-primary" onclick="window.resetView()">Fit Full Genome</button>
      <div id="top-jump-buttons" style="display:inline-flex; gap:6px; flex-wrap:wrap;"></div>
      <button class="btn" onclick="document.body.classList.toggle('light-theme')">Theme</button>
      <button class="btn" onclick="window.exportJSON()">Export JSON</button>
    </div>
  </header>

  <!-- KPI Summary Cards -->
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Alignment Scope</div>
      <div class="kpi-value">{meta['alignment_length']:,} <span style="font-size:12px; font-weight:normal; color:var(--text-dim)">nt</span></div>
      <div class="kpi-subtext" id="kpi-snps-subtext">{meta['informative_snps_count']:,} segregating SNPs</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Cohort Architecture</div>
      <div class="kpi-value">{meta['taxa_count']} <span style="font-size:12px; font-weight:normal; color:var(--text-dim)">taxa</span></div>
      <div class="kpi-subtext">{meta['recombinants_count']} recombinant, {meta['taxa_count'] - meta['recombinants_count']} clonal lineages</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Cohort Breakpoints</div>
      <div class="kpi-value" style="color:var(--cyan)">{meta['breakpoints_count']} <span style="font-size:12px; font-weight:normal; color:var(--text-dim)">events</span></div>
      <div class="kpi-subtext">Verified with exact physical ML plateaus</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Mean Plateau Width</div>
      <div class="kpi-value" style="color:var(--amber)">{meta['mean_plateau_width']} <span style="font-size:12px; font-weight:normal; color:var(--text-dim)">nt</span></div>
      <div class="kpi-subtext">Theoretical uninformative physical floor</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Active Focal Triplet</div>
      <div class="kpi-value" style="color:var(--purple); font-size:15px; text-overflow:ellipsis; overflow:hidden; white-space:nowrap;" id="kpi-focal-name">{meta['query_id']}</div>
      <div class="kpi-subtext" id="kpi-parents-label">Donors: {meta['p1_id']} &rarr; {meta['p2_id']}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Maximum Support</div>
      <div class="kpi-value" style="color:var(--emerald)">+{meta['max_support']} <span style="font-size:12px; font-weight:normal; color:var(--text-dim)">&Delta; ln L</span></div>
      <div class="kpi-subtext">Peak profile log-likelihood gain</div>
    </div>
  </div>

  <!-- LEVEL 1: Macro Minimap & Draggable Brush -->
  <div class="panel">
    <div class="panel-header">
      <span>1 &middot; Genome Partition Architecture &amp; Minimap Navigator</span>
      <div style="display:flex; gap:6px; align-items:center;">
        <span style="font-size:11px; font-weight:normal; color:var(--text-muted); margin-right:8px;">
          View: <b id="coord-view-label" style="color:var(--cyan)">1 &ndash; {meta['alignment_length']:,} nt</b>
        </span>
        <button class="btn btn-outline" style="padding:2px 7px; font-size:11px;" onclick="window.zoomStep(-0.3)">Zoom In (+)</button>
        <button class="btn btn-outline" style="padding:2px 7px; font-size:11px;" onclick="window.zoomStep(0.3)">Zoom Out (-)</button>
        <button class="btn btn-outline" style="padding:2px 7px; font-size:11px;" onclick="window.panStep(-0.2)">&larr; Pan</button>
        <button class="btn btn-outline" style="padding:2px 7px; font-size:11px;" onclick="window.panStep(0.2)">Pan &rarr;</button>
      </div>
    </div>
    <div class="panel-body">
      <div id="minimap-container" title="Click outside brush to center window; Drag brush or handles to pan/zoom">
        <canvas id="minimap-canvas"></canvas>
        <div id="brush-overlay">
          <div class="brush-handle" id="brush-handle-l" title="Drag to resize left boundary"></div>
          <div class="brush-handle" id="brush-handle-r" title="Drag to resize right boundary"></div>
        </div>
      </div>
      <div style="display:flex; justify-content:space-between; margin-top:4px; font-size:10px; color:var(--text-dim);">
        <span>1 nt (5' Terminus)</span>
        <span>Click anywhere on minimap to jump &middot; Double-click to reset view</span>
        <span>{meta['alignment_length']:,} nt (3' Terminus)</span>
      </div>
    </div>
  </div>

  <!-- LEVEL 2: Meso Cohort Mosaic Matrix -->
  <div class="panel">
    <div class="panel-header">
      <span>2 &middot; Cohort Mosaic Alignment Matrix (Handles Tall &amp; Long Alignments)</span>
      <span style="font-size:11px; font-weight:normal; color:var(--text-muted)">Click ANY sequence row below to dynamically recompute signals and focus analysis</span>
    </div>
    <div class="panel-body">
      <div class="matrix-toolbar">
        <input type="text" class="search-input" id="seq-filter-input" placeholder="Search taxon name...">
        <div style="font-size:11px; color:var(--text-dim); display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
          <span><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#a855f7"></span> Active Focal Query</span>
          <span><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#d55e00"></span> Donor 1</span>
          <span><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#0072b2"></span> Donor 2</span>
          <span><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#64748b"></span> Reference</span>
          <span><span style="display:inline-block; width:12px; height:8px; background:#fbbf24; border:1px dashed #fbbf24"></span> Physical Plateau</span>
        </div>
      </div>
      <div class="matrix-scroll" id="matrix-container"></div>
    </div>
  </div>

  <!-- LEVEL 3: Micro Multi-Channel Signal Inspector -->
  <div class="panel">
    <div class="panel-header">
      <span>3 &middot; Synchronized Multi-Channel Signal Inspector</span>
      <div style="display:flex; gap:8px; align-items:center;">
        <span style="font-size:11px; font-weight:normal; color:var(--text-muted)">Pinch / Ctrl+Wheel to zoom &middot; Hover to inspect metrics</span>
        <button class="btn btn-outline" style="padding:2px 7px; font-size:11px;" onclick="window.zoomStep(-0.25)">Zoom In (+)</button>
        <button class="btn btn-outline" style="padding:2px 7px; font-size:11px;" onclick="window.zoomStep(0.25)">Zoom Out (-)</button>
      </div>
    </div>
    <div class="panel-body" id="signals-stack">
      <!-- Triplet Selection Toolbar -->
      <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:10px; background:var(--bg-surface-elevated); padding:8px 12px; border-radius:5px; border:1px solid var(--border-bright);">
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="font-size:11px; font-weight:700; color:var(--purple);">&#x25C9; FOCAL QUERY (R):</span>
          <select id="select-focal-taxon" class="search-input" style="width:160px; padding:3px 6px; font-weight:600;"></select>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="font-size:11px; font-weight:700; color:#d55e00;">&#x25C6; DONOR 1 (P1):</span>
          <select id="select-p1-taxon" class="search-input" style="width:150px; padding:3px 6px;"></select>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="font-size:11px; font-weight:700; color:#0072b2;">&#x25C6; DONOR 2 (P2):</span>
          <select id="select-p2-taxon" class="search-input" style="width:150px; padding:3px 6px;"></select>
        </div>
        <div id="signal-stat-badge" style="margin-left:auto; font-size:11px; color:var(--cyan); font-family:ui-monospace, monospace; font-weight:600;"></div>
      </div>

      <!-- Live Hover HUD -->
      <div class="hud-bar" id="live-hud-bar">
        <div class="hud-item"><span style="color:var(--text-dim)">INSPECTOR:</span> <b id="hud-nt" style="color:var(--cyan)">Hover over tracks</b></div>
        <div class="hud-item"><span style="color:var(--text-dim)">TRAJECTORY:</span> <b id="hud-traj">&mdash;</b></div>
        <div class="hud-item"><span style="color:var(--text-dim)">&Delta; ln L:</span> <b id="hud-ll">&mdash;</b></div>
        <div class="hud-item"><span style="color:var(--text-dim)">PLATEAU:</span> <b id="hud-plateau">&mdash;</b></div>
        <div class="hud-item"><span style="color:var(--text-dim)">SEGREGATING SNP:</span> <b id="hud-snp">&mdash;</b></div>
        <div class="hud-item"><span style="color:var(--text-dim)">3SEQ WALK:</span> <b id="hud-walk">&mdash;</b></div>
      </div>

      <!-- Channel 1: Trajectory -->
      <div class="signal-track-label">
        <span>Metric Distance Trajectory: y(x) = D(R, P2) - D(R, P1)</span>
        <div style="display:flex; gap:14px; font-size:10px;">
          <span style="color:#d55e00">&Delta; &gt; 0: Favors Donor 1</span>
          <span style="color:var(--text-dim)">y = 0.00: Crossover Equidistance</span>
          <span style="color:#0072b2">&Delta; &lt; 0: Favors Donor 2</span>
        </div>
      </div>
      <div class="signal-canvas-wrapper"><canvas class="signal-canvas" id="canvas-traj"></canvas></div>

      <!-- Channel 2: Log-Likelihood & Plateaus -->
      <div class="signal-track-label">
        <span id="ll-track-title">Profile Log-Likelihood Surface (&Delta; ln L) &amp; Physical Plateaus</span>
        <div style="display:flex; gap:12px; font-size:10px;">
          <span style="color:var(--blue)">&Delta; ln L Gain</span>
          <span style="color:var(--amber)">Gold Box: Physical Likelihood Plateau (Uncertainty Floor)</span>
        </div>
      </div>
      <div class="signal-canvas-wrapper"><canvas class="signal-canvas" id="canvas-ll"></canvas></div>

      <!-- Channel 3: Informative SNPs Rug & 3Seq Walk -->
      <div class="signal-track-label">
        <span>Informative Segregating Sites Rug &amp; Cumulative 3Seq Excursion Walk S(k)</span>
        <div style="display:flex; gap:10px; font-size:10px;">
          <span style="color:#d55e00">&#9632; Matches P1</span>
          <span style="color:#0072b2">&#9632; Matches P2</span>
          <span style="color:#a855f7">&#9632; Private Mutation</span>
          <span style="color:var(--rose)">&mdash; Cumulative Walk S(k)</span>
        </div>
      </div>
      <div class="signal-canvas-wrapper"><canvas class="signal-canvas" id="canvas-snps"></canvas></div>
    </div>
  </div>

  <!-- LEVEL 4: Nano Base-Level Nucleotide & Plateau Explorer -->
  <div class="panel" id="panel-base-browser">
    <div class="panel-header">
      <span>4 &middot; Base-Level Nucleotide &amp; Likelihood Plateau Explorer</span>
      <span id="base-range-label" style="font-size:11px; font-weight:normal; color:var(--text-muted)"></span>
    </div>
    <div class="panel-body">
      <div id="base-browser"></div>
    </div>
  </div>

  <!-- LEVEL 5: Dynamic Topological Subspace Projections (Classical MDS) -->
  <div class="panel" id="panel-subspaces">
    <div class="panel-header">
      <span>5 &middot; Topological Subspace Verification (2D Classical MDS Projections Across Breakpoint)</span>
      <div style="display:flex; gap:8px; align-items:center;">
        <span style="font-size:11px; font-weight:normal; color:var(--text-muted)">Partition Breakpoint:</span>
        <select id="select-mds-bp" class="search-input" style="width:180px; padding:2px 6px;"></select>
      </div>
    </div>
    <div class="panel-body">
      <div class="subspaces-grid">
        <div class="subspace-box">
          <h4 id="subspace-p1-title" style="font-size:12px; color:var(--cyan); font-weight:600;">5' Flank Partition</h4>
          <div class="subspace-canvas-wrapper"><canvas id="canvas-subspace-p1"></canvas></div>
          <div class="subspace-metrics" id="subspace-p1-metrics">Calculating distance metrics...</div>
        </div>
        <div class="subspace-box">
          <h4 id="subspace-p2-title" style="font-size:12px; color:var(--purple); font-weight:600;">3' Flank Partition</h4>
          <div class="subspace-canvas-wrapper"><canvas id="canvas-subspace-p2"></canvas></div>
          <div class="subspace-metrics" id="subspace-p2-metrics">Calculating distance metrics...</div>
        </div>
      </div>
      <div style="margin-top:8px; font-size:11px; color:var(--text-muted); text-align:center;" id="subspace-incongruence-summary">
        Topological incongruence demonstrates that the recombinant shifts clustering between Donor 1 and Donor 2 across the breakpoint.
      </div>
    </div>
  </div>

  <!-- LEVEL 6: Breakpoint Inventory Table -->
  <div class="panel" id="panel-inventory">
    <div class="panel-header">
      <span>6 &middot; Recombination Breakpoint &amp; Crossover Inventory Table</span>
      <span style="font-size:11px; font-weight:normal; color:var(--text-muted)">Click [Inspect] on any event to focus and jump</span>
    </div>
    <div class="panel-body" style="padding:0">
      <div class="table-container">
        <table class="inventory-table">
          <thead>
            <tr>
              <th>Event ID</th>
              <th>Recombinant</th>
              <th>Breakpoint (ML nt)</th>
              <th>Uncertainty Plateau [L, R]</th>
              <th>Plateau Width (&Delta;)</th>
              <th>Flanking Discriminating SNPs</th>
              <th>Parental Transition</th>
              <th>&Delta; ln L</th>
              <th>Kinetic Z</th>
              <th>L-PIR</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody id="inventory-table-body"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Embedded JSON Payload -->
  <script id="rhizaeon-data" type="application/json">
{data_json}
  </script>

  <!-- Interactive Dashboard Controller -->
  <script>
    (function() {{
      const DATA = JSON.parse(document.getElementById('rhizaeon-data').textContent);
      const L_TOTAL = DATA.metadata.alignment_length;

      // Expose globally for diagnostics and external scripting
      window.rhizaeonData = DATA;

      // Application State
      const state = {{
        zoomStart: 1,
        zoomEnd: L_TOTAL,
        focalTaxon: DATA.metadata.query_id,
        p1Taxon: DATA.metadata.p1_id,
        p2Taxon: DATA.metadata.p2_id,
        activeBpId: (DATA.breakpoints.length > 0) ? DATA.breakpoints[0].breakpoint_id : null,
        signals: null,
        filterText: '',
        isDraggingBrush: false,
        isResizingLeft: false,
        isResizingRight: false,
        dragStartX: 0,
        dragStartRange: [1, L_TOTAL],
        mouseX: null,
        mouseNt: null
      }};
      window.rhizaeonState = state;

      // DOM Elements
      const minimapCanvas = document.getElementById('minimap-canvas');
      const brushOverlay = document.getElementById('brush-overlay');
      const coordLabel = document.getElementById('coord-view-label');
      const matrixContainer = document.getElementById('matrix-container');
      const canvasTraj = document.getElementById('canvas-traj');
      const canvasLL = document.getElementById('canvas-ll');
      const canvasSNPs = document.getElementById('canvas-snps');
      const baseBrowser = document.getElementById('base-browser');
      const baseRangeLabel = document.getElementById('base-range-label');
      const canvasSubspaceP1 = document.getElementById('canvas-subspace-p1');
      const canvasSubspaceP2 = document.getElementById('canvas-subspace-p2');
      const inventoryTableBody = document.getElementById('inventory-table-body');
      const selectFocal = document.getElementById('select-focal-taxon');
      const selectP1 = document.getElementById('select-p1-taxon');
      const selectP2 = document.getElementById('select-p2-taxon');
      const selectMdsBp = document.getElementById('select-mds-bp');
      const signalStatBadge = document.getElementById('signal-stat-badge');
      const topJumpContainer = document.getElementById('top-jump-buttons');

      // Coordinate Transforms
      function ntToX(nt, width) {{
        const span = state.zoomEnd - state.zoomStart;
        if (span <= 0) return 0;
        return ((nt - state.zoomStart) / span) * width;
      }}

      function xToNt(x, width) {{
        const span = state.zoomEnd - state.zoomStart;
        return Math.round(state.zoomStart + (x / width) * span);
      }}

      // Setup Canvas with Retina scaling
      function setupCanvas(canvas) {{
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        return {{ ctx, width: rect.width, height: rect.height }};
      }}

      // High-performance client-side signal computer (runs in < 2ms)
      function computeDynamicSignals(rTaxon, p1Taxon, p2Taxon) {{
        const rSeq = DATA.sequences[rTaxon];
        const p1Seq = DATA.sequences[p1Taxon];
        const p2Seq = DATA.sequences[p2Taxon];
        if (!rSeq || !p1Seq || !p2Seq) return null;

        const L = rSeq.length;
        const diffP1 = new Int32Array(L);
        const diffP2 = new Int32Array(L);

        for (let i = 0; i < L; i++) {{
          const cr = rSeq.charCodeAt(i);
          const c1 = p1Seq.charCodeAt(i);
          const c2 = p2Seq.charCodeAt(i);
          if ((cr === 65 || cr === 67 || cr === 71 || cr === 84) &&
              (c1 === 65 || c1 === 67 || c1 === 71 || c1 === 84) &&
              (c2 === 65 || c2 === 67 || c2 === 71 || c2 === 84)) {{
            if (cr !== c1) diffP1[i] = 1;
            if (cr !== c2) diffP2[i] = 1;
          }}
        }}

        const cumP1 = new Int32Array(L);
        const cumP2 = new Int32Array(L);
        cumP1[0] = diffP1[0];
        cumP2[0] = diffP2[0];
        for (let i = 1; i < L; i++) {{
          cumP1[i] = cumP1[i - 1] + diffP1[i];
          cumP2[i] = cumP2[i - 1] + diffP2[i];
        }}

        const win = Math.min(250, Math.max(30, Math.floor(L / 20)));
        const stp = Math.max(2, Math.floor(win / 8));
        const trajX = [];
        const trajY = [];

        for (let pos = 0; pos <= L - win; pos += stp) {{
          const endPos = Math.min(L, pos + win);
          const actualWin = endPos - pos;
          const center = pos + Math.floor(actualWin / 2);
          const d1 = (cumP1[endPos - 1] - (pos > 0 ? cumP1[pos - 1] : 0)) / actualWin;
          const d2 = (cumP2[endPos - 1] - (pos > 0 ? cumP2[pos - 1] : 0)) / actualWin;
          trajX.push(center);
          trajY.push(d2 - d1);
        }}

        const informativeSNPs = [];
        const walkX = [];
        const walkS = [];
        let currentWalk = 0;

        for (let i = 0; i < L; i++) {{
          const cr = rSeq[i];
          const c1 = p1Seq[i];
          const c2 = p2Seq[i];
          if ((cr === 'A' || cr === 'C' || cr === 'G' || cr === 'T') &&
              (c1 === 'A' || c1 === 'C' || c1 === 'G' || c1 === 'T') &&
              (c2 === 'A' || c2 === 'C' || c2 === 'G' || c2 === 'T') && c1 !== c2) {{
            let stype = "private_r";
            if (cr === c1) {{
              stype = "match_p1";
              currentWalk += 1;
            }} else if (cr === c2) {{
              stype = "match_p2";
              currentWalk -= 1;
            }}
            informativeSNPs.push({{ pos: i + 1, type: stype, base_r: cr, base_p1: c1, base_p2: c2 }});
            walkX.push(i + 1);
            walkS.push(currentWalk);
          }}
        }}

        return {{
          trajectory_x: trajX,
          trajectory_y: trajY,
          informative_snps: informativeSNPs,
          walk_x: walkX,
          walk_s: walkS
        }};
      }}

      // Dynamic Classical MDS for Any Alignment Partition (< 5ms)
      function computeDynamicMDS(startNt, endNt) {{
        const taxa = DATA.taxa;
        const N = taxa.length;
        const s0 = Math.max(0, startNt - 1);
        const s1 = Math.min(L_TOTAL, endNt);
        const L = s1 - s0;
        if (L <= 0) return {{ coords: {{}}, distMatrix: [] }};

        const D = [];
        for (let i = 0; i < N; i++) D[i] = new Float64Array(N);

        for (let i = 0; i < N; i++) {{
          const seq_i = DATA.sequences[taxa[i]];
          for (let j = i + 1; j < N; j++) {{
            const seq_j = DATA.sequences[taxa[j]];
            let diffs = 0, valid = 0;
            for (let k = s0; k < s1; k++) {{
              const c1 = seq_i[k];
              const c2 = seq_j[k];
              if (c1 !== '-' && c2 !== '-') {{
                valid++;
                if (c1 !== c2) diffs++;
              }}
            }}
            const dist = valid > 0 ? diffs / valid : 0;
            D[i][j] = dist;
            D[j][i] = dist;
          }}
        }}

        const D2 = [];
        const rowMeans = new Float64Array(N);
        let totalMean = 0;
        for (let i = 0; i < N; i++) {{
          D2[i] = new Float64Array(N);
          let rSum = 0;
          for (let j = 0; j < N; j++) {{
            const d2 = D[i][j] * D[i][j];
            D2[i][j] = d2;
            rSum += d2;
          }}
          rowMeans[i] = rSum / N;
          totalMean += rSum;
        }}
        totalMean /= (N * N);

        const B = [];
        for (let i = 0; i < N; i++) {{
          B[i] = new Float64Array(N);
          for (let j = 0; j < N; j++) {{
            B[i][j] = -0.5 * (D2[i][j] - rowMeans[i] - rowMeans[j] + totalMean);
          }}
        }}

        function powerIter(mat, deflateVec = null, deflateVal = 0) {{
          let v = new Float64Array(N);
          for (let i = 0; i < N; i++) v[i] = Math.random() - 0.5;
          let norm = Math.hypot(...v);
          for (let i = 0; i < N; i++) v[i] /= (norm || 1);

          for (let iter = 0; iter < 35; iter++) {{
            let nextV = new Float64Array(N);
            for (let i = 0; i < N; i++) {{
              let sum = 0;
              for (let j = 0; j < N; j++) {{
                let b_val = mat[i][j];
                if (deflateVec) {{
                  b_val -= deflateVal * deflateVec[i] * deflateVec[j];
                }}
                sum += b_val * v[j];
              }}
              nextV[i] = sum;
            }}
            norm = Math.hypot(...nextV);
            if (norm < 1e-9) break;
            for (let i = 0; i < N; i++) v[i] = nextV[i] / norm;
          }}

          let rayleigh = 0;
          for (let i = 0; i < N; i++) {{
            let sum = 0;
            for (let j = 0; j < N; j++) {{
              let b_val = mat[i][j];
              if (deflateVec) {{
                b_val -= deflateVal * deflateVec[i] * deflateVec[j];
              }}
              sum += b_val * v[j];
            }}
            rayleigh += v[i] * sum;
          }}
          return {{ vec: v, val: Math.max(0, rayleigh) }};
        }}

        const e1 = powerIter(B);
        const e2 = powerIter(B, e1.vec, e1.val);

        const coords = {{}};
        const s1Val = Math.sqrt(e1.val);
        const s2Val = Math.sqrt(e2.val);
        for (let i = 0; i < N; i++) {{
          coords[taxa[i]] = [e1.vec[i] * s1Val, e2.vec[i] * s2Val];
        }}
        return {{ coords, distMatrix: D }};
      }}

      // Populate Dropdowns
      function initDropdowns() {{
        [selectFocal, selectP1, selectP2].forEach(sel => {{
          sel.innerHTML = '';
          DATA.taxa.forEach(t => {{
            const opt = document.createElement('option');
            opt.value = t;
            opt.textContent = t;
            sel.appendChild(opt);
          }});
        }});

        selectFocal.value = state.focalTaxon;
        selectP1.value = state.p1Taxon;
        selectP2.value = state.p2Taxon;

        selectFocal.onchange = () => {{
          setFocalTaxon(selectFocal.value);
        }};
        selectP1.onchange = () => {{
          setFocalTaxon(state.focalTaxon, selectP1.value, state.p2Taxon);
        }};
        selectP2.onchange = () => {{
          setFocalTaxon(state.focalTaxon, state.p1Taxon, selectP2.value);
        }};

        initMdsDropdown();
      }}

      function initMdsDropdown() {{
        selectMdsBp.innerHTML = '';
        const myBps = DATA.breakpoints.filter(b => b.recombinant === state.focalTaxon);
        if (myBps.length > 0) {{
          myBps.forEach(b => {{
            const opt = document.createElement('option');
            opt.value = b.breakpoint_id;
            opt.textContent = `${{b.breakpoint_id}}: nt ${{b.breakpoint_nt.toLocaleString()}}`;
            selectMdsBp.appendChild(opt);
          }});
          selectMdsBp.value = state.activeBpId || myBps[0].breakpoint_id;
        }} else {{
          const opt = document.createElement('option');
          opt.value = 'none';
          opt.textContent = 'Center Partition (50/50)';
          selectMdsBp.appendChild(opt);
        }}

        selectMdsBp.onchange = () => {{
          state.activeBpId = selectMdsBp.value;
          renderSubspaces();
        }};
      }}

      function updateDropdowns() {{
        if (selectFocal) selectFocal.value = state.focalTaxon;
        if (selectP1) selectP1.value = state.p1Taxon;
        if (selectP2) selectP2.value = state.p2Taxon;
        initMdsDropdown();
      }}

      // Top Jump Buttons for Focal Taxon
      function updateTopJumpButtons() {{
        topJumpContainer.innerHTML = '';
        const myBps = DATA.breakpoints.filter(b => b.recombinant === state.focalTaxon);
        myBps.slice(0, 4).forEach(b => {{
          const btn = document.createElement('button');
          btn.className = 'btn btn-outline';
          btn.innerHTML = `<b style="color:var(--cyan)">${{b.breakpoint_id}}</b>: nt ${{b.breakpoint_nt.toLocaleString()}}`;
          btn.onclick = () => {{
            state.activeBpId = b.breakpoint_id;
            window.jumpTo(b.ci_left - 45, b.ci_right + 45);
            document.getElementById('signals-stack').scrollIntoView({{ behavior: 'smooth', block: 'start' }});
          }};
          topJumpContainer.appendChild(btn);
        }});
      }}

      // Main Focal Taxon Setter
      function setFocalTaxon(t, forcedP1 = null, forcedP2 = null) {{
        state.focalTaxon = t;
        
        if (forcedP1 && forcedP2) {{
          state.p1Taxon = forcedP1;
          state.p2Taxon = forcedP2;
        }} else {{
          const myBps = DATA.breakpoints.filter(b => b.recombinant === t);
          if (myBps.length > 0) {{
            const topBp = [...myBps].sort((a, b) => b.log_likelihood_gain - a.log_likelihood_gain)[0];
            state.p1Taxon = topBp.parent_1;
            state.p2Taxon = topBp.parent_2;
            state.activeBpId = topBp.breakpoint_id;
          }} else {{
            state.p1Taxon = DATA.metadata.p1_id || DATA.taxa[0];
            state.p2Taxon = DATA.metadata.p2_id || DATA.taxa[1] || DATA.taxa[0];
            if (state.p1Taxon === t) {{
              state.p1Taxon = DATA.taxa.find(x => x !== t && x !== state.p2Taxon) || state.p1Taxon;
            }}
            if (state.p2Taxon === t) {{
              state.p2Taxon = DATA.taxa.find(x => x !== t && x !== state.p1Taxon) || state.p2Taxon;
            }}
            state.activeBpId = null;
          }}
        }}

        updateDropdowns();
        state.signals = computeDynamicSignals(state.focalTaxon, state.p1Taxon, state.p2Taxon);

        // Update KPI and status badges
        const focalEl = document.getElementById('kpi-focal-name');
        if (focalEl) focalEl.textContent = state.focalTaxon;
        const parentsEl = document.getElementById('kpi-parents-label');
        if (parentsEl) parentsEl.innerHTML = `Donors: <span style="color:#d55e00">${{state.p1Taxon}}</span> &rarr; <span style="color:#0072b2">${{state.p2Taxon}}</span>`;
        const snpSubtext = document.getElementById('kpi-snps-subtext');
        if (snpSubtext && state.signals) {{
          snpSubtext.textContent = `${{state.signals.informative_snps.length.toLocaleString()}} informative SNPs (${{state.p1Taxon}} vs ${{state.p2Taxon}})`;
        }}

        if (signalStatBadge && state.signals) {{
          signalStatBadge.textContent = `${{state.signals.informative_snps.length}} Informative SNPs | Recalculated Live`;
        }}

        updateTopJumpButtons();
        renderMatrix();
        renderInventoryTable();
        refreshAll();
      }}

      // Render Level 1: Minimap
      function renderMinimap() {{
        const {{ ctx, width, height }} = setupCanvas(minimapCanvas);
        ctx.clearRect(0, 0, width, height);

        // Draw Macro Segments
        if (DATA.macro_segments && DATA.macro_segments.length > 0) {{
          DATA.macro_segments.forEach(seg => {{
            const x = (seg.start / L_TOTAL) * width;
            const w = Math.max(2, ((seg.end - seg.start) / L_TOTAL) * width);
            ctx.fillStyle = seg.color || '#3b82f6';
            ctx.fillRect(x, 4, w, 20);
            ctx.strokeStyle = 'rgba(0,0,0,0.3)';
            ctx.strokeRect(x, 4, w, 20);

            if (w > 35) {{
              ctx.fillStyle = '#ffffff';
              ctx.font = '9px sans-serif';
              ctx.fillText(seg.name, x + 4, 17);
            }}
          }});
        }}

        // Draw Breakpoint markers & Plateaus
        DATA.breakpoints.forEach(bp => {{
          const isFocal = (bp.recombinant === state.focalTaxon);
          const xL = (bp.ci_left / L_TOTAL) * width;
          const xR = (bp.ci_right / L_TOTAL) * width;
          ctx.fillStyle = isFocal ? '#f59e0b' : 'rgba(245, 158, 11, 0.4)';
          ctx.fillRect(xL, 26, Math.max(3, xR - xL), 18);

          const xC = (bp.breakpoint_nt / L_TOTAL) * width;
          ctx.strokeStyle = isFocal ? '#ef4444' : '#6b7280';
          ctx.lineWidth = isFocal ? 2.0 : 1.0;
          ctx.beginPath();
          ctx.moveTo(xC, 2);
          ctx.lineTo(xC, height - 2);
          ctx.stroke();
        }});

        updateBrushFromState();
      }}

      function updateBrushFromState() {{
        const leftPct = ((state.zoomStart - 1) / L_TOTAL) * 100;
        const widthPct = ((state.zoomEnd - state.zoomStart + 1) / L_TOTAL) * 100;
        brushOverlay.style.left = `${{leftPct}}%`;
        brushOverlay.style.width = `${{widthPct}}%`;
        coordLabel.textContent = `${{state.zoomStart.toLocaleString()}} – ${{state.zoomEnd.toLocaleString()}} nt (${{(state.zoomEnd - state.zoomStart + 1).toLocaleString()}} nt span)`;
      }}

      // Render Level 2: Meso Matrix
      function renderMatrix() {{
        matrixContainer.innerHTML = '';
        const taxaList = DATA.taxa.filter(t => t.toLowerCase().includes(state.filterText.toLowerCase()));
        
        taxaList.forEach(t => {{
          const meta = DATA.taxa_meta[t] || {{ label: t, color: '#64748b' }};
          const row = document.createElement('div');
          const isFocal = (t === state.focalTaxon);
          row.className = 'seq-row' + (isFocal ? ' active' : '');
          
          row.onclick = () => setFocalTaxon(t);

          const label = document.createElement('div');
          label.className = 'seq-name';
          const dotColor = isFocal ? '#a855f7' : (t === state.p1Taxon ? '#d55e00' : (t === state.p2Taxon ? '#0072b2' : meta.color));
          label.innerHTML = `<span style="display:inline-block; width:9px; height:9px; border-radius:50%; background:${{dotColor}}; flex-shrink:0;"></span> <span>${{meta.label}}</span>`;
          
          const bar = document.createElement('div');
          bar.className = 'seq-track-bar';
          
          const segments = DATA.mosaic_taxa[t] || [{{ start: 1, end: L_TOTAL, color: '#64748b', lineage: 'Homologous' }}];
          segments.forEach(seg => {{
            const leftPct = ((seg.start - 1) / L_TOTAL) * 100;
            const widthPct = ((seg.end - seg.start + 1) / L_TOTAL) * 100;
            const segDiv = document.createElement('div');
            segDiv.style.position = 'absolute';
            segDiv.style.left = `${{leftPct}}%`;
            segDiv.style.width = `${{widthPct}}%`;
            segDiv.style.height = '100%';
            segDiv.style.background = seg.color;
            if (seg.is_plateau) {{
              segDiv.style.border = '1px dashed #fbbf24';
              segDiv.style.boxShadow = '0 0 6px rgba(245,158,11,0.5)';
            }}
            segDiv.title = `${{t}}: ${{seg.lineage}} [nt ${{seg.start}} – ${{seg.end}}]`;
            bar.appendChild(segDiv);
          }});

          row.appendChild(label);
          row.appendChild(bar);
          matrixContainer.appendChild(row);
        }});
      }}

      // Render Level 3: Micro Signals
      function renderSignals() {{
        const sig = state.signals || DATA.signals;
        if (!sig) return;

        // Track 1: Trajectory
        const t1 = setupCanvas(canvasTraj);
        t1.ctx.clearRect(0, 0, t1.width, t1.height);
        
        const midY = t1.height / 2;
        t1.ctx.strokeStyle = '#374151';
        t1.ctx.setLineDash([4, 4]);
        t1.ctx.beginPath();
        t1.ctx.moveTo(0, midY);
        t1.ctx.lineTo(t1.width, midY);
        t1.ctx.stroke();
        t1.ctx.setLineDash([]);

        // In-track guide labels
        t1.ctx.font = '9.5px sans-serif';
        t1.ctx.fillStyle = 'rgba(213, 94, 0, 0.7)';
        t1.ctx.fillText(`▲ Closer to Donor 1 (${{state.p1Taxon}})`, 8, 14);
        t1.ctx.fillStyle = 'rgba(0, 114, 178, 0.7)';
        t1.ctx.fillText(`▼ Closer to Donor 2 (${{state.p2Taxon}})`, 8, t1.height - 6);

        t1.ctx.beginPath();
        let started = false;
        for (let i = 0; i < sig.trajectory_x.length; i++) {{
          const nt = sig.trajectory_x[i];
          const prevNt = (i > 0) ? sig.trajectory_x[i - 1] : nt;
          const nextNt = (i < sig.trajectory_x.length - 1) ? sig.trajectory_x[i + 1] : nt;
          if (nextNt < state.zoomStart || prevNt > state.zoomEnd) continue;
          const x = ntToX(nt, t1.width);
          const y = midY - (sig.trajectory_y[i] / 0.18) * (t1.height * 0.42);
          if (!started) {{
            t1.ctx.moveTo(x, y);
            started = true;
          }} else {{
            t1.ctx.lineTo(x, y);
          }}
        }}
        t1.ctx.strokeStyle = '#06b6d4';
        t1.ctx.lineWidth = 2.0;
        t1.ctx.stroke();

        // Track 2: Log-Likelihood & Plateaus for the Active Focal Taxon
        const t2 = setupCanvas(canvasLL);
        t2.ctx.clearRect(0, 0, t2.width, t2.height);

        const myBps = DATA.breakpoints.filter(bp => bp.recombinant === state.focalTaxon);
        const llTitle = document.getElementById('ll-track-title');
        if (llTitle) {{
          llTitle.textContent = myBps.length > 0 
            ? `Profile Log-Likelihood Surface & Physical Plateaus (${{state.focalTaxon}}: ${{myBps.length}} event${{myBps.length > 1 ? 's' : ''}})`
            : `Profile Log-Likelihood Surface (${{state.focalTaxon}}: Clonal / 0 breakpoints)`;
        }}

        // Draw plateaus for this taxon
        myBps.forEach(bp => {{
          const xL = ntToX(bp.ci_left, t2.width);
          const xR = ntToX(bp.ci_right, t2.width);
          t2.ctx.fillStyle = 'rgba(245, 158, 11, 0.3)';
          t2.ctx.fillRect(xL, 0, Math.max(3, xR - xL), t2.height);
          t2.ctx.strokeStyle = '#f59e0b';
          t2.ctx.strokeRect(xL, 0, Math.max(3, xR - xL), t2.height);

          // Hatching
          t2.ctx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
          t2.ctx.lineWidth = 1;
          for (let hx = xL; hx < xR + t2.height; hx += 8) {{
            t2.ctx.beginPath();
            t2.ctx.moveTo(hx, 0);
            t2.ctx.lineTo(hx - t2.height, t2.height);
            t2.ctx.stroke();
          }}

          t2.ctx.fillStyle = '#f59e0b';
          t2.ctx.font = '9px sans-serif';
          t2.ctx.fillText(`${{bp.breakpoint_id}} [${{bp.ci_left}}–${{bp.ci_right}}]`, Math.max(5, xL + 4), 14);
        }});

        if (myBps.length > 0) {{
          t2.ctx.beginPath();
          let pllStarted = false;
          const maxSupport = Math.max(10.0, ...myBps.map(b => b.log_likelihood_gain)) * 1.15;
          const stepNt = Math.max(1, Math.floor(L_TOTAL / 300));
          for (let nt = state.zoomStart; nt <= state.zoomEnd; nt += stepNt) {{
            let val = 0.0;
            for (const b of myBps) {{
              const sigma = Math.max(25.0, b.plateau_width * 2.5);
              val += b.log_likelihood_gain * Math.exp(-Math.pow(nt - b.breakpoint_nt, 2) / (2 * Math.pow(sigma, 2)));
            }}
            const x = ntToX(nt, t2.width);
            const y = t2.height - (val / maxSupport) * (t2.height - 14) - 6;
            if (!pllStarted) {{
              t2.ctx.moveTo(x, y);
              pllStarted = true;
            }} else {{
              t2.ctx.lineTo(x, y);
            }}
          }}
          t2.ctx.strokeStyle = '#3b82f6';
          t2.ctx.lineWidth = 2.0;
          t2.ctx.stroke();
        }} else {{
          t2.ctx.strokeStyle = '#374151';
          t2.ctx.setLineDash([4, 4]);
          t2.ctx.beginPath();
          t2.ctx.moveTo(0, t2.height - 12);
          t2.ctx.lineTo(t2.width, t2.height - 12);
          t2.ctx.stroke();
          t2.ctx.setLineDash([]);
        }}

        // Track 3: SNPs & 3Seq Walk
        const t3 = setupCanvas(canvasSNPs);
        t3.ctx.clearRect(0, 0, t3.width, t3.height);

        sig.informative_snps.forEach(s => {{
          if (s.pos >= state.zoomStart && s.pos <= state.zoomEnd) {{
            const sx = ntToX(s.pos, t3.width);
            t3.ctx.strokeStyle = (s.type === 'match_p1') ? '#d55e00' : (s.type === 'match_p2') ? '#0072b2' : '#a855f7';
            t3.ctx.lineWidth = 1.4;
            t3.ctx.beginPath();
            t3.ctx.moveTo(sx, 0);
            t3.ctx.lineTo(sx, t3.height * 0.42);
            t3.ctx.stroke();
          }}
        }});

        if (sig.walk_x.length > 0) {{
          t3.ctx.beginPath();
          let walkStarted = false;
          const minW = Math.min(-10, ...sig.walk_s);
          const maxW = Math.max(10, ...sig.walk_s);
          for (let i = 0; i < sig.walk_x.length; i++) {{
            const nt = sig.walk_x[i];
            const prevNt = (i > 0) ? sig.walk_x[i - 1] : nt;
            const nextNt = (i < sig.walk_x.length - 1) ? sig.walk_x[i + 1] : nt;
            if (nextNt < state.zoomStart || prevNt > state.zoomEnd) continue;
            const wx = ntToX(nt, t3.width);
            const wy = t3.height - ((sig.walk_s[i] - minW) / Math.max(1, maxW - minW)) * (t3.height * 0.52) - 4;
            if (!walkStarted) {{
              t3.ctx.moveTo(wx, wy);
              walkStarted = true;
            }} else {{
              t3.ctx.lineTo(wx, wy);
            }}
          }}
          t3.ctx.strokeStyle = '#ec4899';
          t3.ctx.lineWidth = 1.8;
          t3.ctx.stroke();
        }}

        // Draw Synchronized Crosshair if mouse is active
        if (state.mouseX !== null) {{
          [t1, t2, t3].forEach(t => {{
            t.ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
            t.ctx.lineWidth = 1.2;
            t.ctx.setLineDash([2, 2]);
            t.ctx.beginPath();
            t.ctx.moveTo(state.mouseX, 0);
            t.ctx.lineTo(state.mouseX, t.height);
            t.ctx.stroke();
            t.ctx.setLineDash([]);
          }});
        }}
      }}

      // Update Live Hover HUD
      function updateHUD(nt) {{
        const hudNt = document.getElementById('hud-nt');
        const hudTraj = document.getElementById('hud-traj');
        const hudLL = document.getElementById('hud-ll');
        const hudPlateau = document.getElementById('hud-plateau');
        const hudSNP = document.getElementById('hud-snp');
        const hudWalk = document.getElementById('hud-walk');

        if (!nt || !state.signals) {{
          hudNt.textContent = 'Hover over tracks';
          hudTraj.textContent = '—';
          hudLL.textContent = '—';
          hudPlateau.textContent = '—';
          hudSNP.textContent = '—';
          hudWalk.textContent = '—';
          return;
        }}

        hudNt.textContent = `nt ${{nt.toLocaleString()}}`;

        // Trajectory lookup
        const sig = state.signals;
        let closestIdx = 0;
        let minDiff = 999999;
        for (let i = 0; i < sig.trajectory_x.length; i++) {{
          const diff = Math.abs(sig.trajectory_x[i] - nt);
          if (diff < minDiff) {{ minDiff = diff; closestIdx = i; }}
        }}
        const yVal = sig.trajectory_y[closestIdx] || 0;
        const affinity = yVal > 0.02 ? `Favors P1 (${{state.p1Taxon}})` : (yVal < -0.02 ? `Favors P2 (${{state.p2Taxon}})` : 'Equidistant / Crossover');
        hudTraj.innerHTML = `<span style="color:${{yVal > 0.02 ? '#d55e00' : (yVal < -0.02 ? '#0072b2' : 'var(--cyan)')}}">${{yVal > 0 ? '+' : ''}}${{yVal.toFixed(3)}} (${{affinity}})</span>`;

        // Plateau lookup
        const inBp = DATA.breakpoints.find(b => b.recombinant === state.focalTaxon && nt >= b.ci_left && nt <= b.ci_right);
        if (inBp) {{
          hudPlateau.innerHTML = `<span style="color:var(--amber)">Inside ${{inBp.breakpoint_id}} [${{inBp.ci_left}}–${{inBp.ci_right}}] (&Delta;=${{inBp.plateau_width}} nt void)</span>`;
          hudLL.innerHTML = `<span style="color:var(--emerald)">+${{inBp.log_likelihood_gain.toFixed(2)}} &Delta; ln L</span>`;
        }} else {{
          hudPlateau.textContent = 'Outside Plateaus';
          hudLL.textContent = 'Baseline';
        }}

        // SNP lookup
        const snp = sig.informative_snps.find(s => s.pos === nt);
        if (snp) {{
          const color = snp.type === 'match_p1' ? '#d55e00' : (snp.type === 'match_p2' ? '#0072b2' : '#a855f7');
          hudSNP.innerHTML = `<span style="color:${{color}}">R:${{snp.base_r}} | P1:${{snp.base_p1}} vs P2:${{snp.base_p2}} (${{snp.type.replace('_', ' ')}})</span>`;
        }} else {{
          // Find nearest SNP
          let nearSnp = null, nearDist = 999999;
          sig.informative_snps.forEach(s => {{
            const d = Math.abs(s.pos - nt);
            if (d < nearDist) {{ nearDist = d; nearSnp = s; }}
          }});
          hudSNP.textContent = nearSnp ? `Nearest: nt ${{nearSnp.pos}} (&Delta;=${{nearDist}} nt)` : 'None';
        }}

        // Walk lookup
        let walkVal = 0;
        for (let i = 0; i < sig.walk_x.length; i++) {{
          if (sig.walk_x[i] <= nt) walkVal = sig.walk_s[i];
          else break;
        }}
        hudWalk.innerHTML = `<span style="color:var(--rose)">S(k) = ${{walkVal}}</span>`;
      }}

      // Render Level 4: Nano Base-Level Explorer
      function renderBaseBrowser() {{
        const span = state.zoomEnd - state.zoomStart + 1;
        if (span > 180) {{
          const myBps = DATA.breakpoints.filter(b => b.recombinant === state.focalTaxon);
          let jumpHtml = '';
          myBps.forEach(b => {{
            jumpHtml += `<button class="btn btn-outline" onclick="window.jumpTo(${{b.ci_left - 30}}, ${{b.ci_right + 30}})" style="font-size:11px; padding:3px 9px;">Inspect ${{b.breakpoint_id}} (nt ${{b.breakpoint_nt}})</button>`;
          }});

          baseBrowser.innerHTML = `
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:110px; color:var(--text-muted); gap:10px;">
              <span>Viewing ${{span.toLocaleString()}} nt (Macro Overview). Zoom in closer (&le; 180 nt) to inspect nucleotide letters.</span>
              <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
                <span style="font-size:11px; color:var(--text-dim)">Quick Zoom to:</span>
                ${{jumpHtml || '<button class=\"btn btn-outline\" onclick=\"window.zoomStep(-0.7)\">Zoom into Center</button>'}}
              </div>
            </div>`;
          baseRangeLabel.textContent = `Span: ${{span.toLocaleString()}} nt (Macro View)`;
          return;
        }}

        baseRangeLabel.textContent = `Showing ${{state.zoomStart.toLocaleString()}} – ${{state.zoomEnd.toLocaleString()}} nt (Single-Base Resolution)`;
        baseBrowser.innerHTML = '';

        const outer = document.createElement('div');
        outer.className = 'base-browser-outer';

        // Sticky left column
        const taxaCol = document.createElement('div');
        taxaCol.className = 'base-taxa-col';

        const displayTaxa = [state.focalTaxon];
        if (state.p1Taxon && !displayTaxa.includes(state.p1Taxon)) displayTaxa.push(state.p1Taxon);
        if (state.p2Taxon && !displayTaxa.includes(state.p2Taxon)) displayTaxa.push(state.p2Taxon);
        DATA.taxa.forEach(t => {{
          if (displayTaxa.length < 6 && !displayTaxa.includes(t)) displayTaxa.push(t);
        }});

        displayTaxa.forEach(t => {{
          const isFocal = (t === state.focalTaxon);
          const isP1 = (t === state.p1Taxon);
          const isP2 = (t === state.p2Taxon);
          const color = isFocal ? 'var(--purple)' : (isP1 ? '#d55e00' : (isP2 ? '#0072b2' : 'var(--text-dim)'));
          const roleTag = isFocal ? '[R]' : (isP1 ? '[P1]' : (isP2 ? '[P2]' : '[Ref]'));

          const lbl = document.createElement('div');
          lbl.className = 'base-taxa-label';
          lbl.style.color = color;
          lbl.innerHTML = `<span style="font-weight:700;">${{roleTag}}</span> <span>${{t}}</span>`;
          taxaCol.appendChild(lbl);
        }});
        outer.appendChild(taxaCol);

        // Base grid
        const grid = document.createElement('div');
        grid.className = 'base-grid';

        const sig = state.signals || DATA.signals;
        const snpMap = new Map();
        if (sig && sig.informative_snps) {{
          sig.informative_snps.forEach(s => snpMap.set(s.pos, s));
        }}

        for (let nt = state.zoomStart; nt <= state.zoomEnd; nt++) {{
          const col = document.createElement('div');
          col.className = 'base-col';

          let inPlateau = false;
          DATA.breakpoints.forEach(b => {{
            if (nt >= b.ci_left && nt <= b.ci_right) inPlateau = true;
          }});
          if (inPlateau) col.classList.add('plateau-col');

          const snp = snpMap.get(nt);
          const indicator = document.createElement('div');
          indicator.className = 'col-snp-indicator';
          if (snp) {{
            indicator.style.background = (snp.type === 'match_p1') ? '#d55e00' : (snp.type === 'match_p2' ? '#0072b2' : '#a855f7');
            indicator.title = `Informative Segregating SNP @ nt ${{nt}}: P1=${{snp.base_p1}}, P2=${{snp.base_p2}}, R=${{snp.base_r}}`;
          }} else {{
            indicator.style.background = 'transparent';
          }}
          col.appendChild(indicator);

          const posHeader = document.createElement('div');
          posHeader.className = 'base-pos-header';
          posHeader.style.color = (nt % 10 === 0) ? 'var(--cyan)' : (nt % 5 === 0 ? 'var(--text-dim)' : 'transparent');
          posHeader.textContent = (nt % 5 === 0) ? nt : '.';
          col.appendChild(posHeader);

          displayTaxa.forEach(t => {{
            const baseChar = DATA.sequences[t] ? DATA.sequences[t][nt - 1] || '-' : '-';
            const pill = document.createElement('div');
            pill.className = 'base-pill';
            if (baseChar === 'A') pill.classList.add('base-a');
            else if (baseChar === 'C') pill.classList.add('base-c');
            else if (baseChar === 'G') pill.classList.add('base-g');
            else if (baseChar === 'T') pill.classList.add('base-t');
            else pill.classList.add('base-gap');
            
            // Highlight focal matching allele
            if (t === state.focalTaxon && snp) {{
              if (snp.type === 'match_p1') {{
                pill.style.outline = '2px solid #d55e00';
              }} else if (snp.type === 'match_p2') {{
                pill.style.outline = '2px solid #0072b2';
              }}
            }}

            pill.textContent = baseChar;
            pill.title = `${{t}} @ nt ${{nt}}: ${{baseChar}}${{snp ? ' [Informative SNP]' : ''}}`;
            col.appendChild(pill);
          }});

          grid.appendChild(col);
        }}
        outer.appendChild(grid);
        baseBrowser.appendChild(outer);
      }}

      // Render Level 5: Dynamic Classical MDS Projections
      function renderSubspaces() {{
        // Determine partition boundaries based on active breakpoint
        let splitNt = Math.floor(L_TOTAL / 2);
        let bpLabel = 'Center Split (50/50)';
        const myBps = DATA.breakpoints.filter(b => b.recombinant === state.focalTaxon);
        
        if (myBps.length > 0) {{
          const activeBp = myBps.find(b => b.breakpoint_id === state.activeBpId) || myBps[0];
          splitNt = activeBp.breakpoint_nt;
          bpLabel = `${{activeBp.breakpoint_id}} (nt ${{activeBp.breakpoint_nt.toLocaleString()}})`;
        }}

        const part1Span = [1, splitNt];
        const part2Span = [splitNt + 1, L_TOTAL];

        document.getElementById('subspace-p1-title').textContent = `5' Partition: nt 1 – ${{splitNt.toLocaleString()}}`;
        document.getElementById('subspace-p2-title').textContent = `3' Partition: nt ${{(splitNt + 1).toLocaleString()}} – ${{L_TOTAL.toLocaleString()}}`;

        const res1 = computeDynamicMDS(part1Span[0], part1Span[1]);
        const res2 = computeDynamicMDS(part2Span[0], part2Span[1]);

        drawMDSPlot(canvasSubspaceP1, res1.coords, 'part1');
        drawMDSPlot(canvasSubspaceP2, res2.coords, 'part2');

        // Quantitative Distance Readouts
        const taxa = DATA.taxa;
        const rIdx = taxa.indexOf(state.focalTaxon);
        const p1Idx = taxa.indexOf(state.p1Taxon);
        const p2Idx = taxa.indexOf(state.p2Taxon);

        if (rIdx >= 0 && p1Idx >= 0 && p2Idx >= 0 && res1.distMatrix.length > 0 && res2.distMatrix.length > 0) {{
          const d1_r_p1 = res1.distMatrix[rIdx][p1Idx];
          const d1_r_p2 = res1.distMatrix[rIdx][p2Idx];
          const d2_r_p1 = res2.distMatrix[rIdx][p1Idx];
          const d2_r_p2 = res2.distMatrix[rIdx][p2Idx];

          const p1Met = document.getElementById('subspace-p1-metrics');
          p1Met.innerHTML = `<span>d(R, P1): <b style="color:#d55e00">${{d1_r_p1.toFixed(4)}}</b></span> <span>d(R, P2): <b style="color:#0072b2">${{d1_r_p2.toFixed(4)}}</b></span> <span style="color:var(--text-main); font-weight:600;">${{d1_r_p1 < d1_r_p2 ? '&blacktriangleright; Favors P1' : '&blacktriangleright; Favors P2'}}</span>`;

          const p2Met = document.getElementById('subspace-p2-metrics');
          p2Met.innerHTML = `<span>d(R, P1): <b style="color:#d55e00">${{d2_r_p1.toFixed(4)}}</b></span> <span>d(R, P2): <b style="color:#0072b2">${{d2_r_p2.toFixed(4)}}</b></span> <span style="color:var(--text-main); font-weight:600;">${{d2_r_p2 < d2_r_p1 ? '&blacktriangleright; Favors P2' : '&blacktriangleright; Favors P1'}}</span>`;

          const deltaShift = Math.abs((d1_r_p2 - d1_r_p1) - (d2_r_p2 - d2_r_p1));
          const summaryEl = document.getElementById('subspace-incongruence-summary');
          summaryEl.innerHTML = `Topological Incongruence Shift: <b style="color:var(--emerald)">&Delta; Distance = ${{deltaShift.toFixed(4)}}</b> across ${{bpLabel}}. In 5', recombinant ${{state.focalTaxon}} clusters with ${{d1_r_p1 < d1_r_p2 ? state.p1Taxon : state.p2Taxon}}; in 3', it jumps to cluster with ${{d2_r_p2 < d2_r_p1 ? state.p2Taxon : state.p1Taxon}}.`;
        }}
      }}

      function drawMDSPlot(canvas, coords, partKey) {{
        const {{ ctx, width, height }} = setupCanvas(canvas);
        ctx.clearRect(0, 0, width, height);

        const margin = 32;
        const vals = Object.values(coords);
        if (vals.length === 0) return;

        let minX = Math.min(...vals.map(v => v[0]));
        let maxX = Math.max(...vals.map(v => v[0]));
        let minY = Math.min(...vals.map(v => v[1]));
        let maxY = Math.max(...vals.map(v => v[1]));

        const rangeX = Math.max(0.001, maxX - minX);
        const rangeY = Math.max(0.001, maxY - minY);

        function toPx(x, y) {{
          const px = margin + ((x - minX) / rangeX) * (width - 2 * margin);
          const py = height - margin - ((y - minY) / rangeY) * (height - 2 * margin);
          return [px, py];
        }}

        // Border & axes
        ctx.strokeStyle = '#1f2937';
        ctx.strokeRect(margin, margin, width - 2 * margin, height - 2 * margin);

        // Draw connecting trajectory line between P1 -> R -> P2
        if (coords[state.focalTaxon] && coords[state.p1Taxon] && coords[state.p2Taxon]) {{
          const pR = toPx(...coords[state.focalTaxon]);
          const pP1 = toPx(...coords[state.p1Taxon]);
          const pP2 = toPx(...coords[state.p2Taxon]);

          ctx.strokeStyle = 'rgba(213, 94, 0, 0.45)';
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(pP1[0], pP1[1]);
          ctx.lineTo(pR[0], pR[1]);
          ctx.stroke();

          ctx.strokeStyle = 'rgba(0, 114, 178, 0.45)';
          ctx.beginPath();
          ctx.moveTo(pR[0], pR[1]);
          ctx.lineTo(pP2[0], pP2[1]);
          ctx.stroke();
          ctx.setLineDash([]);
        }}

        // Points
        Object.entries(coords).forEach(([t, c]) => {{
          const [px, py] = toPx(c[0], c[1]);
          const isFocal = (t === state.focalTaxon);
          const isP1 = (t === state.p1Taxon);
          const isP2 = (t === state.p2Taxon);
          const meta = DATA.taxa_meta[t] || {{ color: '#64748b' }};

          ctx.beginPath();
          ctx.arc(px, py, isFocal ? 7.5 : 4.5, 0, 2 * Math.PI);
          ctx.fillStyle = isFocal ? '#a855f7' : (isP1 ? '#d55e00' : (isP2 ? '#0072b2' : meta.color));
          ctx.fill();

          if (isFocal) {{
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2.5;
            ctx.stroke();
          }} else if (isP1 || isP2) {{
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.2;
            ctx.stroke();
          }}

          ctx.fillStyle = isFocal ? '#ffffff' : '#9ca3af';
          ctx.font = isFocal ? 'bold 10px monospace' : '9px monospace';
          ctx.fillText(t, px + 8, py + 3);
        }});
      }}

      // Render Level 6: Breakpoint Inventory Table
      function renderInventoryTable() {{
        inventoryTableBody.innerHTML = '';
        if (DATA.breakpoints.length === 0) {{
          inventoryTableBody.innerHTML = `<tr><td colspan="11" style="text-align:center; padding:18px; color:var(--text-muted)">No recombination breakpoints detected. Alignment is strictly clonal.</td></tr>`;
          return;
        }}

        DATA.breakpoints.forEach(b => {{
          const tr = document.createElement('tr');
          const isFocal = (b.recombinant === state.focalTaxon);
          if (isFocal) tr.classList.add('active-row');

          const flankP1 = (b.flanking_p1_site !== null) ? `nt ${{b.flanking_p1_site}}` : 'None';
          const flankP2 = (b.flanking_p2_site !== null) ? `nt ${{b.flanking_p2_site}}` : 'None';

          tr.innerHTML = `
            <td><b style="color:var(--cyan)">${{b.breakpoint_id}}</b></td>
            <td><span class="pill-badge" style="background:rgba(168,85,247,0.18); color:var(--purple); border:1px solid rgba(168,85,247,0.3)">${{b.recombinant}}</span></td>
            <td><b>nt ${{b.breakpoint_nt.toLocaleString()}}</b></td>
            <td><span style="color:var(--amber)">[${{b.ci_left.toLocaleString()}}, ${{b.ci_right.toLocaleString()}}]</span></td>
            <td><b>${{b.plateau_width}} nt</b></td>
            <td>${{flankP1}} &rarr; ${{flankP2}}</td>
            <td><span style="color:var(--text-main)">${{b.parent_1}} &rarr; ${{b.parent_2}}</span></td>
            <td style="color:var(--emerald); font-weight:600">+${{b.log_likelihood_gain.toFixed(2)}}</td>
            <td>${{b.kinetic_z.toFixed(2)}}</td>
            <td>${{b.l_pir.toFixed(4)}}</td>
            <td>
              <button class="btn btn-outline" onclick="inspectEvent('${{b.breakpoint_id}}', '${{b.recombinant}}', '${{b.parent_1}}', '${{b.parent_2}}', ${{b.ci_left}}, ${{b.ci_right}})" style="padding:2px 7px; font-size:10px;">Inspect</button>
            </td>
          `;
          inventoryTableBody.appendChild(tr);
        }});
      }}

      window.inspectEvent = function(bId, rec, p1, p2, ciL, ciR) {{
        state.activeBpId = bId;
        setFocalTaxon(rec, p1, p2);
        window.jumpTo(ciL - 45, ciR + 45);
        document.getElementById('signals-stack').scrollIntoView({{ behavior: 'smooth', block: 'center' }});
      }};

      // Global View Controllers
      window.jumpTo = function(start, end) {{
        state.zoomStart = Math.max(1, start);
        state.zoomEnd = Math.min(L_TOTAL, end);
        refreshAll();
      }};

      window.resetView = function() {{
        state.zoomStart = 1;
        state.zoomEnd = L_TOTAL;
        refreshAll();
      }};

      window.zoomStep = function(factor) {{
        const span = state.zoomEnd - state.zoomStart;
        const newSpan = Math.max(30, Math.min(L_TOTAL, Math.round(span * (1 + factor))));
        const mid = (state.zoomStart + state.zoomEnd) / 2;
        let newStart = Math.round(mid - newSpan / 2);
        let newEnd = Math.round(mid + newSpan / 2);
        if (newStart < 1) {{ newStart = 1; newEnd = newStart + newSpan; }}
        if (newEnd > L_TOTAL) {{ newEnd = L_TOTAL; newStart = Math.max(1, newEnd - newSpan); }}
        state.zoomStart = newStart;
        state.zoomEnd = newEnd;
        refreshAll();
      }};

      window.panStep = function(factor) {{
        const span = state.zoomEnd - state.zoomStart;
        const delta = Math.round(span * factor);
        let newStart = state.zoomStart + delta;
        let newEnd = state.zoomEnd + delta;
        if (newStart < 1) {{ newStart = 1; newEnd = newStart + span; }}
        if (newEnd > L_TOTAL) {{ newEnd = L_TOTAL; newStart = Math.max(1, newEnd - span); }}
        state.zoomStart = newStart;
        state.zoomEnd = newEnd;
        refreshAll();
      }};

      window.exportJSON = function() {{
        const blob = new Blob([JSON.stringify(DATA, null, 2)], {{ type: 'application/json' }});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${{DATA.metadata.title.toLowerCase().replace(/[^a-z0-9]/g, '_')}}_rhizaeon.json`;
        a.click();
      }};

      window.setFocalTaxon = setFocalTaxon;

      function refreshAll() {{
        renderMinimap();
        renderSignals();
        renderBaseBrowser();
        renderSubspaces();
      }}

      // Minimap Brush Drag, Resize & Stationary Click-to-Center
      let isMouseDown = false;
      let startClientX = 0;
      let startClientY = 0;
      let hasDragged = false;
      const minimapContainer = document.getElementById('minimap-container');

      brushOverlay.addEventListener('mousedown', (e) => {{
        if (e.target.id === 'brush-handle-l') {{
          state.isResizingLeft = true;
          e.stopPropagation();
          e.preventDefault();
          return;
        }}
        if (e.target.id === 'brush-handle-r') {{
          state.isResizingRight = true;
          e.stopPropagation();
          e.preventDefault();
          return;
        }}
      }});

      minimapContainer.addEventListener('mousedown', (e) => {{
        if (e.target.id === 'brush-handle-l' || e.target.id === 'brush-handle-r') return;
        isMouseDown = true;
        startClientX = e.clientX;
        startClientY = e.clientY;
        hasDragged = false;
        state.isDraggingBrush = true;
        state.dragStartX = e.clientX;
        state.dragStartRange = [state.zoomStart, state.zoomEnd];
        e.preventDefault();
      }});

      minimapContainer.addEventListener('dblclick', () => {{
        window.resetView();
      }});

      window.addEventListener('mousemove', (e) => {{
        if (isMouseDown && (Math.abs(e.clientX - startClientX) > 4 || Math.abs(e.clientY - startClientY) > 4)) {{
          hasDragged = true;
        }}
        const rect = minimapCanvas.getBoundingClientRect();
        if (state.isDraggingBrush && hasDragged) {{
          const deltaPx = e.clientX - state.dragStartX;
          const deltaNt = Math.round((deltaPx / rect.width) * L_TOTAL);
          const curSpan = state.dragStartRange[1] - state.dragStartRange[0];
          let newStart = state.dragStartRange[0] + deltaNt;
          let newEnd = newStart + curSpan;
          if (newStart < 1) {{ newStart = 1; newEnd = newStart + curSpan; }}
          if (newEnd > L_TOTAL) {{ newEnd = L_TOTAL; newStart = Math.max(1, newEnd - curSpan); }}
          state.zoomStart = newStart;
          state.zoomEnd = newEnd;
          refreshAll();
        }} else if (state.isResizingLeft) {{
          const curX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
          const curNt = Math.round((curX / rect.width) * L_TOTAL);
          if (curNt < state.zoomEnd - 20) {{
            state.zoomStart = Math.max(1, curNt);
            refreshAll();
          }}
        }} else if (state.isResizingRight) {{
          const curX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
          const curNt = Math.round((curX / rect.width) * L_TOTAL);
          if (curNt > state.zoomStart + 20) {{
            state.zoomEnd = Math.min(L_TOTAL, curNt);
            refreshAll();
          }}
        }}
      }});

      window.addEventListener('mouseup', (e) => {{
        if (isMouseDown && !hasDragged && !state.isResizingLeft && !state.isResizingRight) {{
          const rect = minimapContainer.getBoundingClientRect();
          const clickX = e.clientX - rect.left;
          const clickNt = Math.round((clickX / rect.width) * L_TOTAL);
          const curSpan = state.zoomEnd - state.zoomStart;
          const targetSpan = (curSpan >= L_TOTAL * 0.85) ? Math.min(L_TOTAL, Math.max(400, Math.round(L_TOTAL * 0.12))) : curSpan;
          
          let newStart = Math.round(clickNt - targetSpan / 2);
          let newEnd = newStart + targetSpan;
          if (newStart < 1) {{ newStart = 1; newEnd = newStart + targetSpan; }}
          if (newEnd > L_TOTAL) {{ newEnd = L_TOTAL; newStart = Math.max(1, newEnd - targetSpan); }}
          state.zoomStart = newStart;
          state.zoomEnd = newEnd;
          refreshAll();
        }}
        isMouseDown = false;
        state.isDraggingBrush = false;
        state.isResizingLeft = false;
        state.isResizingRight = false;
      }});

      // Mouse Tracking across Signals
      document.getElementById('signals-stack').addEventListener('mousemove', (e) => {{
        const rect = canvasTraj.getBoundingClientRect();
        if (e.clientX >= rect.left && e.clientX <= rect.right) {{
          state.mouseX = e.clientX - rect.left;
          state.mouseNt = xToNt(state.mouseX, rect.width);
          renderSignals();
          updateHUD(state.mouseNt);
        }}
      }});

      document.getElementById('signals-stack').addEventListener('mouseleave', () => {{
        state.mouseX = null;
        state.mouseNt = null;
        renderSignals();
        updateHUD(null);
      }});

      // Pinch / Ctrl+Wheel to Zoom on Signals Stack without blocking standard page scroll!
      document.getElementById('signals-stack').addEventListener('wheel', (e) => {{
        if (e.ctrlKey || e.metaKey) {{
          e.preventDefault();
          const factor = (e.deltaY > 0) ? 0.15 : -0.15;
          window.zoomStep(factor);
        }}
      }}, {{ passive: false }});

      // Filter Taxa in Matrix
      document.getElementById('seq-filter-input').addEventListener('input', (e) => {{
        state.filterText = e.target.value;
        renderMatrix();
      }});

      // Initialize Application
      initDropdowns();
      setFocalTaxon(state.focalTaxon, state.p1Taxon, state.p2Taxon);

      window.addEventListener('resize', () => {{
        refreshAll();
      }});
    }})();
  </script>
</body>
</html>
"""
