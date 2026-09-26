"""
rhizaeon.reporting
==================
Humanized, multi-layered inference reporting and terminal synthesis for RhizAeon.

Provides structured, hierarchical executive summaries, sequence-centric mosaic
architecture schematics, and coordinate-sorted breakpoint catalogs with principled
quality and detection-type classifications.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import sys


def shorten_taxon_name(name: str, max_len: int = 18) -> str:
    """
    Produces a concise, human-readable label for a taxon header.
    Specialized for LANL HIV, GISAID Influenza, and generic accessions.
    """
    if not name:
        return "-"
    name_clean = name.strip(".")
    parts = name_clean.split(".")
    # LANL HIV format: Subtype.Country.Year.Name.Accession (e.g. SN01_AE.TH.90.CM240.AF069670)
    if len(parts) >= 4:
        st = parts[0]
        if st.startswith("SN"):
            st = "CRF" + st[2:]
        iso = parts[3]
        short = f"{st} ({iso})"
        if len(short) <= max_len:
            return short
        return short[:max_len]

    # Influenza format: A/host/location/id/year (e.g. A/Great black-backed gull/MA/25HP00805/2025)
    if "/" in name_clean:
        f_parts = name_clean.split("/")
        if len(f_parts) >= 4:
            sub_id = f_parts[-2] if len(f_parts[-2]) <= 10 else f_parts[-2][:8]
            short = f"{f_parts[0]}/{sub_id}"
            if len(short) <= max_len:
                return short
            return short[:max_len]

    if len(name_clean) <= max_len:
        return name_clean
    return name_clean[: max_len - 2] + ".."


def shorten_parent_name(name: str, max_len: int = 7) -> str:
    """
    Extracts the minimal clade or lineage code for a parent.
    e.g., SN01_AE.TH.90.CM240 -> CRF01 or C01
          A1.UG.92.92UG037.AB2 -> A1
          B.FR.83.HXB2 -> B
    """
    if not name:
        return "?"
    name_clean = name.strip(".")
    parts = name_clean.split(".")
    st = parts[0]
    if st.startswith("SN"):
        # SN01_AE -> C01
        sub_num = st[2:].split("_")[0]
        return f"CRF{sub_num}" if len(f"CRF{sub_num}") <= max_len else f"C{sub_num}"
    if len(st) <= max_len:
        return st
    return st[:max_len]


@dataclass
class BreakpointRecord:
    idx: int
    coord: int  # primary coordinate in units (nt or codon)
    coord_nt: int  # coordinate in nucleotides
    unit_type: str  # "nt" or "codon"
    recombinant: str
    recombinant_short: str
    parent_left: str
    parent_right: str
    parent_left_short: str
    parent_right_short: str
    z_score: float
    pir: float
    tier: str  # "tier1" or "two-tier"
    bp_type: str  # "T1-Cross", "T2-Attn", "Micro", "Ghost"
    support: str  # "HIGH", "MOD", "LOW"
    stars: str  # "★★★", "★★☆", "★☆☆"
    ci_left: Optional[int] = None
    ci_right: Optional[int] = None
    ci_left_nt: Optional[int] = None
    ci_right_nt: Optional[int] = None
    plateau_width: Optional[int] = None
    plateau_width_nt: Optional[int] = None
    ll_gain: Optional[float] = None
    flanking_p1: Optional[str] = None
    flanking_p2: Optional[str] = None
    is_primary_mosaic: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecombinantLineage:
    taxon: str
    short_name: str
    is_primary: bool
    breakpoints: List[BreakpointRecord]
    confidence: str
    stars: str
    parent_proportions: Dict[str, float]
    mosaic_map: str


@dataclass
class InferenceReport:
    alignment_path: str
    total_taxa: int
    alignment_len: int
    unit_type: str
    elapsed_sec: float
    calibration: str
    z_threshold: float
    pir_threshold: float
    min_tract: int
    total_breakpoints: int
    primary_recombinants: List[RecombinantLineage]
    isolated_displacements: List[RecombinantLineage]
    clonal_taxa: List[str]
    catalog: List[BreakpointRecord]
    confidence_counts: Dict[str, int]
    type_counts: Dict[str, int]

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "total_taxa": self.total_taxa,
            "alignment_len": self.alignment_len,
            "unit_type": self.unit_type,
            "elapsed_sec": self.elapsed_sec,
            "calibration": self.calibration,
            "z_threshold": self.z_threshold,
            "pir_threshold": self.pir_threshold,
            "min_tract": self.min_tract,
            "total_breakpoints": self.total_breakpoints,
            "confidence_counts": self.confidence_counts,
            "type_counts": self.type_counts,
            "num_primary_mosaics": len(self.primary_recombinants),
            "primary_mosaics": [
                {
                    "taxon": r.taxon,
                    "short_name": r.short_name,
                    "num_breakpoints": len(r.breakpoints),
                    "confidence": r.confidence,
                    "parent_proportions": r.parent_proportions,
                    "mosaic_map": r.mosaic_map
                }
                for r in self.primary_recombinants
            ],
            "num_isolated_displacements": len(self.isolated_displacements),
            "isolated_displacements": [
                {
                    "taxon": r.taxon,
                    "short_name": r.short_name,
                    "breakpoint": r.breakpoints[0].coord if r.breakpoints else None,
                    "transition": f"{r.breakpoints[0].parent_left_short} -> {r.breakpoints[0].parent_right_short}" if r.breakpoints else None
                }
                for r in self.isolated_displacements
            ],
            "num_clonal_taxa": len(self.clonal_taxa),
            "clonal_taxa": self.clonal_taxa
        }


def classify_breakpoint(
    z_score: float,
    pir: float,
    plateau_width: Optional[int],
    tier: str = "tier1",
    has_tier2_result: bool = False,
    tier2_resolved: bool = False,
    is_micro: bool = False
) -> Tuple[str, str, str]:
    """
    Classifies a breakpoint into (bp_type, support_str, stars).
    
    Types:
      - T2-Attn: Tier 2 contextual transformer handoff
      - Micro: Micro-tract gene conversion (<150 nt / <50 codons)
      - Ghost: Unsampled donor introgression (high Z, low PIR)
      - T1-Cross: Tier 1 classical geometric crossover
      
    Confidence:
      - HIGH (★★★): Z >= 3.0, PIR >= 0.25, tight plateau (<=40 nt)
      - MOD  (★★☆): Z >= 2.75, PIR >= 0.18, plateau <=100 nt
      - LOW  (★☆☆): near threshold or wide uninformative plateau
    """
    # Type classification
    if tier2_resolved or (has_tier2_result and tier in ("two-tier", "tier2", "two-tier-attention")) or tier in ("two-tier", "tier2", "two-tier-attention"):
        bp_type = "T2-Attn"
    elif is_micro:
        bp_type = "Micro"
    elif pir < 0.15 and z_score >= 3.0:
        bp_type = "Ghost"
    else:
        bp_type = "T1-Cross"

    # Support tier
    p_width = plateau_width if plateau_width is not None else 0
    if (z_score >= 3.0 and pir >= 0.25 and p_width <= 60) or (z_score >= 2.50 and pir >= 0.40 and p_width <= 25):
        support = "HIGH"
        stars = "★★★"
    elif (z_score >= 2.60 and pir >= 0.18 and p_width <= 100) or (z_score >= 2.35 and pir >= 0.30 and p_width <= 50):
        support = "MOD"
        stars = "★★☆"
    else:
        support = "LOW"
        stars = "★☆☆"

    return bp_type, support, stars


def construct_mosaic_architecture(
    tbps: List[BreakpointRecord],
    alignment_len: int,
    unit_type: str = "nt"
) -> Tuple[str, Dict[str, float]]:
    """
    Constructs an ASCII 1-line genomic mosaic schematic and computes
    parental sequence percentages.
    
    Example output:
      map:  [A1]──2,254──[H*]──4,420──[A1]──6,810──[H*]
      comp: {'A1': 61.2, 'H*': 38.8}
    """
    if not tbps:
        return "[Clonal]", {}

    sorted_bps = sorted(tbps, key=lambda b: b.coord)
    segments = []
    parent_lengths: Dict[str, int] = {}

    # Initial segment from 0 to first breakpoint
    first_bp = sorted_bps[0]
    p_initial = first_bp.parent_left_short
    len_0 = first_bp.coord
    parent_lengths[p_initial] = parent_lengths.get(p_initial, 0) + len_0
    segments.append(f"[{p_initial}]")

    prev_coord = len_0
    for i, b in enumerate(sorted_bps):
        coord_val = b.coord
        if i > 0:
            seg_len = max(0, coord_val - prev_coord)
            p_curr = b.parent_left_short
            parent_lengths[p_curr] = parent_lengths.get(p_curr, 0) + seg_len

        p_next = b.parent_right_short
        coord_str = f"{coord_val:,}"
        segments.append(f"──{coord_str}──[{p_next}]")
        prev_coord = coord_val

    # Final segment from last breakpoint to end of alignment
    last_bp = sorted_bps[-1]
    p_final = last_bp.parent_right_short
    final_len = max(0, alignment_len - last_bp.coord)
    parent_lengths[p_final] = parent_lengths.get(p_final, 0) + final_len

    total_len = sum(parent_lengths.values()) or max(1, alignment_len)
    proportions = {p: (length / total_len) * 100.0 for p, length in parent_lengths.items()}

    mosaic_map_str = "".join(segments)
    return mosaic_map_str, proportions


def synthesize_inference_report(
    events_or_bps: List[Any],
    taxa: List[str],
    alignment_path: str,
    alignment_len: int,
    unit_type: str = "nt",
    elapsed_sec: float = 0.0,
    calibration: str = "calibrated",
    z_threshold: float = 2.75,
    pir_threshold: float = 0.20,
    min_tract: int = 15
) -> InferenceReport:
    """
    Harmonizes raw breakpoint events from either `scan` (dicts) or `rp-fda` (FDABreakpoints)
    into a structured, executive InferenceReport.
    """
    scale = 3 if unit_type == "codon" else 1
    raw_records: List[BreakpointRecord] = []

    for idx, ev in enumerate(events_or_bps, 1):
        if hasattr(ev, "breakpoint_nt"):
            # FDABreakpoint object from rp-fda
            coord = ev.breakpoint_nt if unit_type == "nt" else (ev.breakpoint_nt // scale)
            coord_nt = ev.breakpoint_nt
            rec_taxon = ev.recombinant_taxon
            p1 = ev.parent_1
            p2 = ev.parent_2
            z_val = float(ev.kinetic_z)
            pir_val = float(ev.l_pir)
            ci_l = ev.ci_left
            ci_r = ev.ci_right
            ci_l_nt = ev.ci_left
            ci_r_nt = ev.ci_right
            plat_w = ev.plateau_width
            plat_w_nt = ev.plateau_width
            ll_g = getattr(ev, "log_likelihood_gain", None)
            fp1 = getattr(ev, "flanking_p1_site", None)
            fp2 = getattr(ev, "flanking_p2_site", None)
            t2_res = bool(getattr(ev, "tier2_resolved", False))
            tier_val = "two-tier" if t2_res else "tier1"
            has_t2 = t2_res
        else:
            # Dictionary from scan / detect_recombination
            coord = ev.get("breakpoint", ev.get("breakpoint_nt", 0))
            coord_nt = ev.get("breakpoint_nt", coord * scale)
            rec_taxon = ev.get("recombinant", "Unknown")
            p1 = ev.get("parent_left", "?")
            p2 = ev.get("parent_right", "?")
            z_val = float(ev.get("ghost_z", ev.get("kinetic_z", 0.0)))
            pir_val = float(ev.get("refined_pir", ev.get("l_pir", 0.0)))
            ci_l = ev.get("ci_left")
            ci_r = ev.get("ci_right")
            ci_l_nt = ev.get("ci_left_nt", ci_l * scale if ci_l is not None else None)
            ci_r_nt = ev.get("ci_right_nt", ci_r * scale if ci_r is not None else None)
            plat_w = ev.get("plateau_width")
            plat_w_nt = ev.get("plateau_width_nt", plat_w * scale if plat_w is not None else None)
            ll_g = ev.get("log_likelihood_gain")
            fp1 = ev.get("flanking_p1_site")
            fp2 = ev.get("flanking_p2_site")
            t2_res = bool(ev.get("tier2_resolved", False))
            tier_val = ev.get("tier", "tier1")
            has_t2 = t2_res or tier_val in ("two-tier", "tier2")

        rec_short = shorten_taxon_name(rec_taxon, max_len=18)
        p1_short = shorten_parent_name(p1, max_len=7)
        p2_short = shorten_parent_name(p2, max_len=7)

        # Micro-tract check (tract length < 200 nt / < 50 codons)
        if hasattr(ev, "breakpoint_nt"):
            is_micro = bool(getattr(ev, "is_micro", False) or (getattr(ev, "scale_regime", "") == "micro"))
        else:
            is_micro = bool(ev.get("is_micro", False) or (ev.get("scale_regime") == "micro"))

        bp_type, support, stars = classify_breakpoint(
            z_score=z_val,
            pir=pir_val,
            plateau_width=plat_w_nt if plat_w_nt is not None else plat_w,
            tier=tier_val,
            has_tier2_result=has_t2,
            tier2_resolved=t2_res,
            is_micro=is_micro
        )

        rec = BreakpointRecord(
            idx=idx,
            coord=coord,
            coord_nt=coord_nt,
            unit_type=unit_type,
            recombinant=rec_taxon,
            recombinant_short=rec_short,
            parent_left=p1,
            parent_right=p2,
            parent_left_short=p1_short,
            parent_right_short=p2_short,
            z_score=z_val,
            pir=pir_val,
            tier=tier_val,
            bp_type=bp_type,
            support=support,
            stars=stars,
            ci_left=ci_l,
            ci_right=ci_r,
            ci_left_nt=ci_l_nt,
            ci_right_nt=ci_r_nt,
            plateau_width=plat_w,
            plateau_width_nt=plat_w_nt,
            ll_gain=ll_g,
            flanking_p1=str(fp1) if fp1 is not None else None,
            flanking_p2=str(fp2) if fp2 is not None else None
        )
        raw_records.append(rec)

    # Sort the global catalog strictly by genomic coordinate (5' to 3')
    catalog = sorted(raw_records, key=lambda r: r.coord)
    for i, r in enumerate(catalog, 1):
        r.idx = i

    # Group by recombinant taxon to classify Primary Mosaics vs Isolated Displacements
    by_taxon: Dict[str, List[BreakpointRecord]] = {}
    for r in catalog:
        by_taxon.setdefault(r.recombinant, []).append(r)

    primary_recombinants: List[RecombinantLineage] = []
    isolated_displacements: List[RecombinantLineage] = []

    for taxon, tbps in by_taxon.items():
        tbps_sorted = sorted(tbps, key=lambda b: b.coord)
        high_or_mod_count = sum(1 for b in tbps if b.support in ("HIGH", "MOD"))
        distinct_parents = len(set(b.parent_right_short for b in tbps).union(set(b.parent_left_short for b in tbps)))

        mosaic_map, proportions = construct_mosaic_architecture(
            tbps=tbps_sorted,
            alignment_len=alignment_len,
            unit_type=unit_type
        )
        sorted_props = sorted(proportions.values(), reverse=True)
        minor_parent_pct = sorted_props[1] if len(sorted_props) > 1 else 0.0
        top2_parents_pct = sum(sorted_props[:2])

        coords = [0] + [b.coord for b in tbps_sorted] + [alignment_len]
        seg_lens = [coords[i+1] - coords[i] for i in range(len(coords)-1)]
        min_seg = min(seg_lens) if seg_lens else 0
        scale_nt = 3 if unit_type == "codon" else 1

        # Primary Mosaic Criterion:
        # Authentic genomic mosaics alternate between consistent parental clades (distinct parents <= 3),
        # have balanced introgressions (minor parent >= 12% and top 2 >= 85%), and no isolated edge blips (min segment >= 200 nt).
        is_primary = (
            len(tbps) >= 1
            and distinct_parents <= 3
            and minor_parent_pct >= 12.0
            and top2_parents_pct >= 85.0
            and (min_seg * scale_nt) >= 200
            and any(b.support in ("HIGH", "MOD") for b in tbps)
        )

        # Check for micro-tract conversions between adjacent breakpoints (<200 nt)
        if len(tbps_sorted) >= 2:
            for i in range(len(tbps_sorted) - 1):
                inter_dist = (tbps_sorted[i+1].coord - tbps_sorted[i].coord) * scale_nt
                if inter_dist < 200:
                    if tbps_sorted[i].bp_type == "T1-Cross":
                        tbps_sorted[i].bp_type = "Micro"
                    if tbps_sorted[i+1].bp_type == "T1-Cross":
                        tbps_sorted[i+1].bp_type = "Micro"

        for b in tbps:
            b.is_primary_mosaic = is_primary

        # Lineage-level confidence summary
        if any(b.support == "HIGH" for b in tbps):
            lineage_conf = "HIGH"
            lineage_stars = "★★★"
        elif any(b.support == "MOD" for b in tbps):
            lineage_conf = "MOD"
            lineage_stars = "★★☆"
        else:
            lineage_conf = "LOW"
            lineage_stars = "★☆☆"

        lineage = RecombinantLineage(
            taxon=taxon,
            short_name=shorten_taxon_name(taxon, max_len=20),
            is_primary=is_primary,
            breakpoints=tbps_sorted,
            confidence=lineage_conf,
            stars=lineage_stars,
            parent_proportions=proportions,
            mosaic_map=mosaic_map
        )

        if is_primary:
            primary_recombinants.append(lineage)
        else:
            isolated_displacements.append(lineage)

    # Sort lineages: Primary by descending breakpoint count, Isolated by coordinate
    primary_recombinants.sort(key=lambda x: -len(x.breakpoints))
    isolated_displacements.sort(key=lambda x: x.breakpoints[0].coord if x.breakpoints else 0)

    # Identify clonal reference taxa (0 breakpoints detected)
    rec_taxa_set = set(by_taxon.keys())
    clonal_taxa = [t for t in taxa if t not in rec_taxa_set]

    # Global summary statistics
    confidence_counts = {
        "HIGH": sum(1 for r in catalog if r.support == "HIGH"),
        "MOD": sum(1 for r in catalog if r.support == "MOD"),
        "LOW": sum(1 for r in catalog if r.support == "LOW")
    }
    type_counts = {
        "T1-Cross": sum(1 for r in catalog if r.bp_type == "T1-Cross"),
        "T2-Attn": sum(1 for r in catalog if r.bp_type == "T2-Attn"),
        "Micro": sum(1 for r in catalog if r.bp_type == "Micro"),
        "Ghost": sum(1 for r in catalog if r.bp_type == "Ghost")
    }

    return InferenceReport(
        alignment_path=alignment_path,
        total_taxa=len(taxa),
        alignment_len=alignment_len,
        unit_type=unit_type,
        elapsed_sec=elapsed_sec,
        calibration=calibration,
        z_threshold=z_threshold,
        pir_threshold=pir_threshold,
        min_tract=min_tract,
        total_breakpoints=len(catalog),
        primary_recombinants=primary_recombinants,
        isolated_displacements=isolated_displacements,
        clonal_taxa=clonal_taxa,
        catalog=catalog,
        confidence_counts=confidence_counts,
        type_counts=type_counts
    )


def format_humanized_report(
    report: InferenceReport,
    verbose: bool = False,
    max_rows: Optional[int] = None,
    summary_only: bool = False
) -> str:
    """
    Renders the InferenceReport into a visually rich, human-readable terminal report.
    """
    lines = []
    width = 116
    u_str = report.unit_type

    # ─────────────────────────────────────────────────────────────────────────────
    # Layer 1: Executive Cohort Summary Box
    # ─────────────────────────────────────────────────────────────────────────────
    lines.append("┌" + "─" * (width - 2) + "┐")
    title_str = f"RHIZAEON INFERENCE REPORT ({report.elapsed_sec:.3f} seconds)"
    lines.append(f"│ {title_str[:width - 4]:<{width - 4}} │")
    lines.append("├" + "─" * (width - 2) + "┤")
    
    max_info_w = width - 7
    suffix = f" ({report.total_taxa} taxa, {report.alignment_len:,} {u_str}s)"
    align_info = f"Alignment: {report.alignment_path}{suffix}"
    if len(align_info) > max_info_w:
        avail = max_info_w - len("Alignment: ") - len(suffix)
        if avail > 10:
            align_info = f"Alignment: ...{str(report.alignment_path)[-(avail - 3):]}{suffix}"
        else:
            align_info = align_info[:max_info_w - 3] + "..."
    lines.append(f"│  • {align_info:<{max_info_w}}│")

    cal_info = f"Calibration: {report.calibration.capitalize()} (Z ≥ {report.z_threshold:.2f}, PIR ≥ {report.pir_threshold:.2f}, Min Tract: {report.min_tract} {u_str}s)"
    lines.append(f"│  • {cal_info[:max_info_w]:<{max_info_w}}│")

    num_rec = len(report.primary_recombinants) + len(report.isolated_displacements)
    pct_rec = (num_rec / report.total_taxa * 100.0) if report.total_taxa > 0 else 0.0
    pct_clonal = 100.0 - pct_rec
    cohort_str = (
        f"Cohort Structure: {num_rec}/{report.total_taxa} Recombinant ({pct_rec:.1f}%) | "
        f"{len(report.clonal_taxa)}/{report.total_taxa} Clonal Pure ({pct_clonal:.1f}%)"
    )
    lines.append(f"│  • {cohort_str[:max_info_w]:<{max_info_w}}│")

    if report.total_breakpoints == 0:
        msg = "Outcome: CLONAL ALIGNMENT — Zero recombination breakpoints detected."
        lines.append(f"│  • {msg:<{max_info_w}}│")
        lines.append("└" + "─" * (width - 2) + "┘")
        return "\n".join(lines)

    t2_tag = " [Two-Tier Verified]" if report.type_counts["T2-Attn"] > 0 else ""
    bp_word = "Breakpoint" if report.total_breakpoints == 1 else "Breakpoints"
    bp_cnt_str = f"Detected {report.total_breakpoints} Recombination {bp_word}{t2_tag}"
    c_high = report.confidence_counts['HIGH']
    c_mod = report.confidence_counts['MOD']
    c_low = report.confidence_counts['LOW']
    conf_str = (
        f"{bp_cnt_str} | "
        f"Confidence: {c_high} High (★★★), "
        f"{c_mod} Mod (★★☆), "
        f"{c_low} Low (★☆☆)"
    )
    lines.append(f"│  • {conf_str[:max_info_w]:<{max_info_w}}│")

    type_str = (
        f"Mechanisms:  {report.type_counts['T1-Cross']} T1-Cross, "
        f"{report.type_counts['T2-Attn']} T2-Attn, "
        f"{report.type_counts['Micro']} Micro, "
        f"{report.type_counts['Ghost']} Ghost"
    )
    lines.append(f"│  • {type_str[:max_info_w]:<{max_info_w}}│")
    lines.append("└" + "─" * (width - 2) + "┘")
    lines.append("")

    # ─────────────────────────────────────────────────────────────────────────────
    # Layer 2: Recombinant Lineages & Mosaic Architecture
    # ─────────────────────────────────────────────────────────────────────────────
    lines.append("━" * width)
    lines.append("RECOMBINANT LINEAGES & MOSAIC ARCHITECTURE")
    lines.append("━" * width)

    if report.primary_recombinants:
        lines.append(f"[+] Primary Mosaic Genomes ({len(report.primary_recombinants)} lineage{'s' if len(report.primary_recombinants) > 1 else ''}):")
        for i, rec in enumerate(report.primary_recombinants, 1):
            comp_strs = [f"{p} ({pct:.1f}%)" for p, pct in sorted(rec.parent_proportions.items(), key=lambda x: -x[1])[:3]]
            comp_display = ", ".join(comp_strs) if comp_strs else "Undefined"
            lines.append(f"  {i}. {rec.short_name}")
            lines.append(f"     • Events: {len(rec.breakpoints)} Breakpoints | Support: {rec.confidence} ({rec.stars}) | Clades: {comp_display}")
            lines.append(f"     • Architecture: {rec.mosaic_map}")
            lines.append("")
    else:
        lines.append("[-] No Primary Multi-Breakpoint Mosaics detected.")
        lines.append("")

    if report.isolated_displacements:
        lines.append(f"[?] Ambiguous / Isolated Displacements ({len(report.isolated_displacements)} lineage{'s' if len(report.isolated_displacements) > 1 else ''}):")
        for rec in report.isolated_displacements:
            n_ev = len(rec.breakpoints)
            if n_ev == 1:
                bp = rec.breakpoints[0]
                pos_str = f"{bp.coord:,} {u_str}"
                lines.append(f"  • {rec.short_name:<20} : 1 event at {pos_str} ({bp.parent_left_short} ➔ {bp.parent_right_short}, Z={bp.z_score:.2f}, PIR={bp.pir:.3f}, {bp.support} {bp.stars})")
            else:
                first_pos = f"{rec.breakpoints[0].coord:,} {u_str}"
                last_pos = f"{rec.breakpoints[-1].coord:,} {u_str}"
                lines.append(f"  • {rec.short_name:<20} : {n_ev} minor events between {first_pos} and {last_pos} (apparent/shadow displacements)")
        lines.append("")

    if report.clonal_taxa:
        lines.append(f"[✓] Clonal Reference Lineages ({len(report.clonal_taxa)} pure lineages, 0 breakpoints):")
        clonal_shorts = [shorten_taxon_name(t, max_len=16) for t in report.clonal_taxa]
        # Chunk into lines of ~6 taxa
        chunk_size = 6
        for c_idx in range(0, len(clonal_shorts), chunk_size):
            chunk = clonal_shorts[c_idx: c_idx + chunk_size]
            lines.append("    " + ", ".join(chunk))
        lines.append("")

    if summary_only:
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────────
    # Layer 3: Breakpoint Catalog Table (Sorted by Genomic Coordinate)
    # ─────────────────────────────────────────────────────────────────────────────
    lines.append("━" * width)
    lines.append("BREAKPOINT CATALOG (Sorted by Genomic Coordinate from 5' to 3')")
    lines.append("━" * width)

    # Column specifications: (Column Name, Max Field Width, Formatter Lambda)
    cols = [
        ("#", 3, lambda r: str(r.idx)),
        ("Position", 10, lambda r: f"{r.coord:,} {r.unit_type}"),
        ("Plateau (Δ)", 19, lambda r: f"[{r.ci_left}, {r.ci_right}] (Δ={r.plateau_width})" if r.ci_left is not None and r.ci_right is not None else "-"),
        ("Recombinant", 20, lambda r: ("● " if r.is_primary_mosaic else "○ ") + r.recombinant_short),
        ("Transition", 15, lambda r: f"{r.parent_left_short} ➔ {r.parent_right_short}"),
        ("Z-Score", 7, lambda r: f"Z={r.z_score:.2f}"),
        ("L-PIR", 6, lambda r: f"{r.pir:.3f}"),
        ("Type", 8, lambda r: r.bp_type),
        ("Support", 8, lambda r: f"{r.support} {r.stars}"),
    ]
    gutter = "  "

    lines.append("┌" + "─" * (width - 2) + "┐")
    header_line = gutter.join(f"{name:<{w}}" for name, w, _ in cols)
    lines.append("│ " + f"{header_line:<{width - 4}}" + " │")
    lines.append("├" + "─" * (width - 2) + "┤")

    display_catalog = report.catalog
    is_truncated = False
    if max_rows is not None and max_rows > 0 and len(display_catalog) > max_rows:
        display_catalog = report.catalog[:max_rows]
        is_truncated = True

    for r in display_catalog:
        row_cells = []
        for name, w, fn in cols:
            val = fn(r)
            trimmed = val[:w]
            row_cells.append(f"{trimmed:<{w}}")
        row_str = gutter.join(row_cells)
        lines.append("│ " + f"{row_str:<{width - 4}}" + " │")

        if verbose:
            # Extended multi-line details for this breakpoint
            details = []
            if r.ci_left_nt is not None and r.unit_type == "codon":
                details.append(f"nt [{r.ci_left_nt}, {r.ci_right_nt}] (Δ={r.plateau_width_nt} nt)")
            if r.ll_gain is not None:
                details.append(f"LL Gain: +{r.ll_gain:.2f}")
            if r.flanking_p1 and r.flanking_p1 != "None":
                details.append(f"Flanking SNPs: {r.flanking_p1} ➔ {r.flanking_p2}")
            if r.tier == "two-tier":
                details.append("Tier 2 Verified")

            if details:
                detail_str = "   └─ " + " | ".join(details)
                lines.append("│ " + f"{detail_str[:width - 4]:<{width - 4}}" + " │")

    lines.append("└" + "─" * (width - 2) + "┘")
    lines.append("Legend: ● Primary Mosaic  ○ Isolated Displacement | T1-Cross: Tier 1 Crossover, T2-Attn: Tier 2 Attention, Micro: Micro-Tract, Ghost: Unsampled Donor")

    if is_truncated:
        lines.append(f"[*] Showing first {max_rows} of {report.total_breakpoints} breakpoints. Use --verbose or --max-rows 0 to display all.")

    return "\n".join(lines)
