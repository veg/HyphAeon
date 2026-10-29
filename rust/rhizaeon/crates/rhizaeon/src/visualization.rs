use crate::jacobi::symmetric_jacobi_eigh;
use crate::types::{
    Alignment, CanonicalManifold, GowerPoint, InformativeSnpRecord, MacroSegment, MosaicSegment,
    RecombinationLandscape, RecombinationRegion, ScanResult, TaxonMeta, VisualizationBreakpoint,
    VisualizationDossier, VisualizationMetadata, VisualizationSignals, VisualizationSubspaces,
};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Embed the dashboard HTML template at compile time
pub const DASHBOARD_HTML_TEMPLATE: &str = include_str!("../assets/dashboard_template.html");

/// Standard categorical color palette for parents and clades
const PALETTE: &[&str] = &[
    "#d55e00", // Vermilion / Orange
    "#0072b2", // Blue
    "#10b981", // Emerald
    "#ec4899", // Pink
    "#8b5cf6", // Purple-indigo
    "#f59e0b", // Amber
    "#06b6d4", // Cyan
    "#059669", // Dark emerald
    "#2563eb", // Royal blue
];

/// Generates a rich, structured, organism-agnostic VisualizationDossier from an alignment and scan results.
pub fn generate_visualization_dossier(
    aln: &Alignment,
    scan_res: &ScanResult,
    title_opt: Option<&str>,
) -> VisualizationDossier {
    let n = aln.num_taxa;
    let l = aln.length;

    let default_title = format!("RhizAeon Scan Analysis ({} taxa, {} bp)", n, l);
    let title = title_opt.unwrap_or(&default_title).to_string();

    // 1. Identify recombinant, parental, and reference taxa first so we can prioritize them
    let mut taxon_idx_map: HashMap<&str, usize> = HashMap::with_capacity(n);
    for (idx, t) in aln.taxa.iter().enumerate() {
        taxon_idx_map.entry(t.as_str()).or_insert(idx);
    }

    let mut rec_indices = Vec::new();
    let mut rec_taxa_set = HashSet::new();
    let mut parent_taxa_set = HashSet::new();
    let mut parent_indices = Vec::new();

    for ev in &scan_res.events {
        for isolate in &ev.isolates {
            if !rec_taxa_set.contains(isolate) {
                if let Some(&idx) = taxon_idx_map.get(isolate.as_str()) {
                    rec_indices.push(idx);
                }
                rec_taxa_set.insert(isolate.clone());
            }
        }
        if !ev.home_name.is_empty() && ev.home_name != "Ghost" {
            if parent_taxa_set.insert(ev.home_name.clone()) {
                if let Some(&idx) = taxon_idx_map.get(ev.home_name.as_str()) {
                    parent_indices.push(idx);
                }
            }
        }
        if !ev.donor_name.is_empty() && ev.donor_name != "Ghost" && !ev.is_ghost_donor {
            if parent_taxa_set.insert(ev.donor_name.clone()) {
                if let Some(&idx) = taxon_idx_map.get(ev.donor_name.as_str()) {
                    parent_indices.push(idx);
                }
            }
        }
    }

    // Ordering: Recombinants first, then Parents, then References
    let mut sorted_taxa = Vec::with_capacity(n);
    for t in &aln.taxa {
        if rec_taxa_set.contains(t) {
            sorted_taxa.push(t.clone());
        }
    }
    for t in &aln.taxa {
        if !rec_taxa_set.contains(t) && parent_taxa_set.contains(t) {
            sorted_taxa.push(t.clone());
        }
    }
    for t in &aln.taxa {
        if !rec_taxa_set.contains(t) && !parent_taxa_set.contains(t) {
            sorted_taxa.push(t.clone());
        }
    }

    // 2. Reconstruct decoded nucleotide sequences
    // Store all sequences up to ~30 MB nucleotide budget for cohorts <= 3,000 taxa (so all 2,000 sequences are retained),
    // and cap for massive cohorts (e.g., 100K sequences) to prevent browser memory exhaustion.
    let mut sequences = BTreeMap::new();
    let max_store_len = 100_000;
    let store_l = l.min(max_store_len);
    let max_stored_seqs = if n <= 3_000 {
        (30_000_000 / store_l.max(1)).max(500)
    } else {
        (10_000_000 / store_l.max(1)).clamp(320, 2_000)
    };
    let mut stored_count = 0usize;
    for i in 0..n {
        let t_name = &aln.taxa[i];
        let is_priority = rec_taxa_set.contains(t_name) || parent_taxa_set.contains(t_name) || i < 16;
        if n > max_stored_seqs && !is_priority && stored_count >= max_stored_seqs {
            continue;
        }
        let mut seq_str = String::with_capacity(store_l);
        for u in 0..store_l {
            let ch = match aln.get(i, u) {
                1 => 'A',
                2 => 'C',
                3 => 'G',
                4 => 'T',
                _ => '-',
            };
            seq_str.push(ch);
        }
        sequences.insert(t_name.clone(), seq_str);
        stored_count += 1;
    }

    // 3. Taxa metadata and color mapping
    let mut lineage_order = Vec::new();
    let mut lineage_seen = HashSet::new();
    for ev in &scan_res.events {
        if lineage_seen.insert(ev.home_name.clone()) {
            lineage_order.push(ev.home_name.clone());
        }
        let d_name = if ev.is_ghost_donor {
            "Ghost (Unsampled)".to_string()
        } else {
            ev.donor_name.clone()
        };
        if lineage_seen.insert(d_name.clone()) {
            lineage_order.push(d_name);
        }
    }
    for t in &aln.taxa {
        if parent_taxa_set.contains(t) && lineage_seen.insert(t.clone()) {
            lineage_order.push(t.clone());
        }
    }

    let mut parent_colors: BTreeMap<String, String> = BTreeMap::new();
    for (idx, name) in lineage_order.iter().enumerate() {
        let col = if name == "Ghost (Unsampled)" {
            "#94a3b8".to_string()
        } else {
            PALETTE[idx % PALETTE.len()].to_string()
        };
        parent_colors.insert(name.clone(), col);
    }

    // Keep full taxa_meta and mosaic_taxa for all cohorts up to N = 10,000;
    // only use compact sparse representation for massive alignments (N > 10,000).
    let compact_large_n = n > 10_000;
    let mut taxa_meta = BTreeMap::new();
    for (sorted_idx, t) in sorted_taxa.iter().enumerate() {
        let input_order = taxon_idx_map.get(t.as_str()).copied().unwrap_or(0);
        if rec_taxa_set.contains(t) {
            let color = parent_colors
                .get(t)
                .cloned()
                .unwrap_or_else(|| "#a855f7".to_string());
            taxa_meta.insert(
                t.clone(),
                TaxonMeta {
                    label: format!("{} [Recombinant]", t),
                    taxon_type: "recombinant".to_string(),
                    color,
                    input_order,
                },
            );
        } else if parent_taxa_set.contains(t) {
            let color = parent_colors
                .get(t)
                .cloned()
                .unwrap_or_else(|| "#0072b2".to_string());
            taxa_meta.insert(
                t.clone(),
                TaxonMeta {
                    label: format!("{} [Parent Donor]", t),
                    taxon_type: "parent".to_string(),
                    color,
                    input_order,
                },
            );
        } else if !compact_large_n || sorted_idx < 500 {
            taxa_meta.insert(
                t.clone(),
                TaxonMeta {
                    label: format!("{} [Reference]", t),
                    taxon_type: "reference".to_string(),
                    color: "#64748b".to_string(), // Slate
                    input_order,
                },
            );
        }
    }

    // 4. Detailed Breakpoint Records and Uncertainty Plateaus
    let mut breakpoints = Vec::with_capacity(scan_res.events.len() * 2);
    let mut total_plateau_width = 0usize;
    let mut bp_counter = 1usize;

    for ev in &scan_res.events {
        let first_isolate = &ev.isolates[0];
        let cand_idx = taxon_idx_map
            .get(first_isolate.as_str())
            .copied()
            .unwrap_or(0);
        let home_idx = ev.home_idx.unwrap_or(0);
        let donor_idx_opt = ev.donor_idx;

        let donor_display_name = if ev.is_ghost_donor {
            "Ghost (Unsampled)".to_string()
        } else {
            ev.donor_name.clone()
        };

        // 4. Detailed Breakpoint Records and Uncertainty Plateaus
        let is_5p_crossover = ev.is_crossover && ev.u1.saturating_sub(1) < l.saturating_sub(ev.u2);

        if is_5p_crossover {
            // 5' Terminal Crossover: Donor from 1..ev.u2, Breakpoint at ev.u2 transitioning to Home
            let mut left_flank = None;
            let mut right_flank = None;

            if let Some(donor_idx) = donor_idx_opt {
                let search_start = ev.u2.saturating_sub(1);
                // Scan left inside tract for donor match (u < ev.u2)
                for u in (0..search_start).rev() {
                    let rc = aln.get(cand_idx, u);
                    let rh = aln.get(home_idx, u);
                    let rd = aln.get(donor_idx, u);
                    if rc > 0 && rh > 0 && rd > 0 && rh != rd && rc == rd {
                        left_flank = Some(u + 1); // 1-indexed
                        break;
                    }
                }
                // Scan right outside tract for home match (u >= ev.u2 - 1)
                for u in search_start..l {
                    let rc = aln.get(cand_idx, u);
                    let rh = aln.get(home_idx, u);
                    let rd = aln.get(donor_idx, u);
                    if rc > 0 && rh > 0 && rd > 0 && rh != rd && rc == rh {
                        right_flank = Some(u + 1); // 1-indexed
                        break;
                    }
                }
            }

            let ci_l = left_flank.unwrap_or_else(|| ev.u2.saturating_sub(5).max(1));
            let ci_r = right_flank.unwrap_or_else(|| (ev.u2 + 5).min(l));
            let plat_w = if ci_r >= ci_l { ci_r - ci_l } else { 0 };
            total_plateau_width += plat_w;

            breakpoints.push(VisualizationBreakpoint {
                breakpoint_id: format!("BP_{}", bp_counter),
                event_id: ev.event_id,
                isolates: ev.isolates.clone(),
                recombinant: ev.candidate_name.clone(),
                parent_1: donor_display_name.clone(),
                parent_2: ev.home_name.clone(),
                breakpoint_nt: ev.u2,
                coarse_bp: ev.u2_continuous.round() as usize,
                ci_left: ci_l,
                ci_right: ci_r,
                plateau_width: plat_w,
                tract_start: ev.u1,
                tract_end: ev.u2,
                flanking_p1_site: left_flank,
                flanking_p2_site: right_flank,
                log_likelihood_gain: (ev.z_phys * 2.5).max(0.0),
                kinetic_z: ev.z_phys,
                l_pir: ev.d_tract_donor,
                p_fisher: ev.p_fisher,
                is_crossover: true,
                is_ghost_donor: ev.is_ghost_donor,
            });
        } else {
            // 4A. 5' Breakpoint at ev.u1 (for 3' crossovers or cassettes)
            let mut left_flank = None;
            let mut right_flank = None;

            if let Some(donor_idx) = donor_idx_opt {
                // Scan left for home match (u < ev.u1)
                let search_start = ev.u1.saturating_sub(1);
                for u in (0..search_start).rev() {
                    let rc = aln.get(cand_idx, u);
                    let rh = aln.get(home_idx, u);
                    let rd = aln.get(donor_idx, u);
                    if rc > 0 && rh > 0 && rd > 0 && rh != rd && rc == rh {
                        left_flank = Some(u + 1); // 1-indexed
                        break;
                    }
                }
                // Scan right for donor match (u >= ev.u1 - 1)
                for u in search_start..l {
                    let rc = aln.get(cand_idx, u);
                    let rh = aln.get(home_idx, u);
                    let rd = aln.get(donor_idx, u);
                    if rc > 0 && rh > 0 && rd > 0 && rh != rd && rc == rd {
                        right_flank = Some(u + 1); // 1-indexed
                        break;
                    }
                }
            }

            let ci_l = left_flank.unwrap_or_else(|| ev.u1.saturating_sub(5).max(1));
            let ci_r = right_flank.unwrap_or_else(|| (ev.u1 + 5).min(l));
            let plat_w = if ci_r >= ci_l { ci_r - ci_l } else { 0 };
            total_plateau_width += plat_w;

            let bp_5p_id = if ev.is_crossover {
                format!("BP_{}", bp_counter)
            } else {
                format!("BP_{} (5')", bp_counter)
            };

            breakpoints.push(VisualizationBreakpoint {
                breakpoint_id: bp_5p_id,
                event_id: ev.event_id,
                isolates: ev.isolates.clone(),
                recombinant: ev.candidate_name.clone(),
                parent_1: ev.home_name.clone(),
                parent_2: donor_display_name.clone(),
                breakpoint_nt: ev.u1,
                coarse_bp: ev.u1_continuous.round() as usize,
                ci_left: ci_l,
                ci_right: ci_r,
                plateau_width: plat_w,
                tract_start: ev.u1,
                tract_end: ev.u2,
                flanking_p1_site: left_flank,
                flanking_p2_site: right_flank,
                log_likelihood_gain: (ev.z_phys * 2.5).max(0.0),
                kinetic_z: ev.z_phys,
                l_pir: ev.d_tract_donor,
                p_fisher: ev.p_fisher,
                is_crossover: ev.is_crossover,
                is_ghost_donor: ev.is_ghost_donor,
            });

            // 4B. 3' Breakpoint at ev.u2 (for cassettes/conversions)
            if !ev.is_crossover && ev.u2 > ev.u1 && ev.u2 < l {
                let mut left_flank_3p = None;
                let mut right_flank_3p = None;

                if let Some(donor_idx) = donor_idx_opt {
                    let search_3p = ev.u2.saturating_sub(1);
                    // Scan left inside tract for donor match (u < ev.u2)
                    for u in (0..search_3p).rev() {
                        let rc = aln.get(cand_idx, u);
                        let rh = aln.get(home_idx, u);
                        let rd = aln.get(donor_idx, u);
                        if rc > 0 && rh > 0 && rd > 0 && rh != rd && rc == rd {
                            left_flank_3p = Some(u + 1);
                            break;
                        }
                    }
                    // Scan right outside tract for home match (u >= ev.u2)
                    for u in search_3p..l {
                        let rc = aln.get(cand_idx, u);
                        let rh = aln.get(home_idx, u);
                        let rd = aln.get(donor_idx, u);
                        if rc > 0 && rh > 0 && rd > 0 && rh != rd && rc == rh {
                            right_flank_3p = Some(u + 1);
                            break;
                        }
                    }
                }

                let ci_l_3p = left_flank_3p.unwrap_or_else(|| ev.u2.saturating_sub(5).max(1));
                let ci_r_3p = right_flank_3p.unwrap_or_else(|| (ev.u2 + 5).min(l));
                let plat_w_3p = if ci_r_3p >= ci_l_3p {
                    ci_r_3p - ci_l_3p
                } else {
                    0
                };
                total_plateau_width += plat_w_3p;

                breakpoints.push(VisualizationBreakpoint {
                    breakpoint_id: format!("BP_{} (3')", bp_counter),
                    event_id: ev.event_id,
                    isolates: ev.isolates.clone(),
                    recombinant: ev.candidate_name.clone(),
                    parent_1: donor_display_name,
                    parent_2: ev.home_name.clone(),
                    breakpoint_nt: ev.u2,
                    coarse_bp: ev.u2_continuous.round() as usize,
                    ci_left: ci_l_3p,
                    ci_right: ci_r_3p,
                    plateau_width: plat_w_3p,
                    tract_start: ev.u1,
                    tract_end: ev.u2,
                    flanking_p1_site: left_flank_3p,
                    flanking_p2_site: right_flank_3p,
                    log_likelihood_gain: (ev.z_phys * 2.5).max(0.0),
                    kinetic_z: ev.z_phys,
                    l_pir: ev.d_tract_donor,
                    p_fisher: ev.p_fisher,
                    is_crossover: false,
                    is_ghost_donor: ev.is_ghost_donor,
                });
            }
        }
        bp_counter += 1;
    }

    // Helper to paint a mosaic segment onto an existing non-overlapping sorted list of segments
    let paint_mosaic_segment = |segments: &mut Vec<MosaicSegment>, patch: MosaicSegment| {
        if patch.start > patch.end {
            return;
        }
        let mut next_segs = Vec::with_capacity(segments.len() + 2);
        for seg in segments.drain(..) {
            if seg.end < patch.start || seg.start > patch.end {
                next_segs.push(seg);
            } else {
                if seg.start < patch.start {
                    let mut left_piece = seg.clone();
                    left_piece.end = patch.start - 1;
                    next_segs.push(left_piece);
                }
                if seg.end > patch.end {
                    let mut right_piece = seg;
                    right_piece.start = patch.end + 1;
                    next_segs.push(right_piece);
                }
            }
        }
        next_segs.push(patch);
        next_segs.sort_by_key(|s| s.start);
        *segments = next_segs;
    };

    // 5. Build Mosaic Tracks for each taxon (using HashMap index to avoid O(N * E) scans)
    let mut events_by_taxon: HashMap<&str, Vec<&crate::types::RecombinationEvent>> = HashMap::new();
    for ev in &scan_res.events {
        for iso in &ev.isolates {
            events_by_taxon.entry(iso.as_str()).or_default().push(ev);
        }
    }

    let mut mosaic_taxa = BTreeMap::new();
    for (sorted_idx, t) in sorted_taxa.iter().enumerate() {
        let mut t_events = events_by_taxon.get(t.as_str()).cloned().unwrap_or_default();
        // Paint broadest events first and narrower nested events on top so nested tracts are never skipped
        t_events.sort_by(|a, b| {
            let span_a = a.u2.saturating_sub(a.u1);
            let span_b = b.u2.saturating_sub(b.u1);
            span_b.cmp(&span_a).then_with(|| a.u1.cmp(&b.u1))
        });

        if t_events.is_empty() {
            if let Some(meta) = taxa_meta.get(t) {
                if !compact_large_n || meta.taxon_type != "reference" || sorted_idx < 500 {
                    let lineage = if meta.taxon_type == "parent" {
                        format!("{} (Parental Reference)", t)
                    } else {
                        "Homologous".to_string()
                    };
                    mosaic_taxa.insert(
                        t.clone(),
                        vec![MosaicSegment {
                            start: 1,
                            end: l,
                            lineage,
                            color: meta.color.clone(),
                            is_plateau: None,
                            breakpoint_id: None,
                            event_id: None,
                        }],
                    );
                }
            }
        } else {
            let home_name = &t_events[0].home_name;
            let home_col = parent_colors
                .get(home_name)
                .cloned()
                .unwrap_or_else(|| "#0072b2".to_string());

            let mut segments = vec![MosaicSegment {
                start: 1,
                end: l,
                lineage: format!("Parent: {}", home_name),
                color: home_col.clone(),
                is_plateau: None,
                breakpoint_id: None,
                event_id: None,
            }];

            for ev in &t_events {
                let donor_display = if ev.is_ghost_donor {
                    "Ghost (Unsampled)"
                } else {
                    &ev.donor_name
                };
                let mut donor_col = parent_colors
                    .get(donor_display)
                    .cloned()
                    .unwrap_or_else(|| "#d55e00".to_string());
                if donor_col == home_col && donor_display != home_name {
                    if let Some(alt) = PALETTE.iter().find(|&&c| c != home_col) {
                        donor_col = alt.to_string();
                    }
                }

                let is_5p_crossover =
                    ev.is_crossover && ev.u1.saturating_sub(1) < l.saturating_sub(ev.u2);

                if is_5p_crossover {
                    let bp_rec = breakpoints
                        .iter()
                        .find(|b| b.event_id == ev.event_id && b.breakpoint_nt == ev.u2);
                    let bp_id = bp_rec.map(|b| b.breakpoint_id.clone());
                    let tract_start = ev.u1.max(1);
                    let ci_l = bp_rec
                        .map(|b| b.ci_left)
                        .unwrap_or(ev.u2.saturating_sub(5).max(1))
                        .max(tract_start.saturating_add(1).min(ev.u2));
                    let ci_r = bp_rec
                        .map(|b| b.ci_right)
                        .unwrap_or((ev.u2 + 5).min(l))
                        .max(ci_l);

                    // Donor segment starting from tract_start up to ci_l - 1
                    if ci_l > tract_start {
                        paint_mosaic_segment(
                            &mut segments,
                            MosaicSegment {
                                start: tract_start,
                                end: ci_l - 1,
                                lineage: format!("Donor: {}", donor_display),
                                color: donor_col.clone(),
                                is_plateau: None,
                                breakpoint_id: bp_id.clone(),
                                event_id: Some(ev.event_id),
                            },
                        );
                    }

                    // Plateau around ev.u2
                    let plat_w = if ci_r >= ci_l { ci_r - ci_l } else { 0 };
                    paint_mosaic_segment(
                        &mut segments,
                        MosaicSegment {
                            start: ci_l,
                            end: ci_r,
                            lineage: format!("Breakpoint [nt {}, Δ={} nt]", ev.u2, plat_w),
                            color: "#fbbf24".to_string(),
                            is_plateau: Some(true),
                            breakpoint_id: bp_id,
                            event_id: Some(ev.event_id),
                        },
                    );
                } else {
                    let bp_5p = breakpoints
                        .iter()
                        .find(|b| b.event_id == ev.event_id && b.breakpoint_nt == ev.u1);
                    let bp_5p_id = bp_5p.map(|b| b.breakpoint_id.clone());
                    let mid_tract = (ev.u1 + ev.u2) / 2;
                    let ci_l_5p = bp_5p
                        .map(|b| b.ci_left)
                        .unwrap_or(ev.u1.saturating_sub(5).max(1))
                        .max(1);
                    let ci_r_5p = bp_5p
                        .map(|b| b.ci_right)
                        .unwrap_or((ev.u1 + 5).min(l))
                        .min(mid_tract.max(ci_l_5p))
                        .max(ci_l_5p);

                    // 5' Plateau
                    let plat_5p_w = if ci_r_5p >= ci_l_5p {
                        ci_r_5p - ci_l_5p
                    } else {
                        0
                    };
                    paint_mosaic_segment(
                        &mut segments,
                        MosaicSegment {
                            start: ci_l_5p,
                            end: ci_r_5p,
                            lineage: format!("Breakpoint 5' [nt {}, Δ={} nt]", ev.u1, plat_5p_w),
                            color: "#fbbf24".to_string(),
                            is_plateau: Some(true),
                            breakpoint_id: bp_5p_id.clone(),
                            event_id: Some(ev.event_id),
                        },
                    );

                    if ev.is_crossover {
                        let donor_end = ev.u2.max(ci_r_5p + 1).min(l);
                        if ci_r_5p + 1 <= donor_end {
                            paint_mosaic_segment(
                                &mut segments,
                                MosaicSegment {
                                    start: ci_r_5p + 1,
                                    end: donor_end,
                                    lineage: format!("Donor: {}", donor_display),
                                    color: donor_col,
                                    is_plateau: None,
                                    breakpoint_id: bp_5p_id.clone(),
                                    event_id: Some(ev.event_id),
                                },
                            );
                        }
                    } else {
                        let bp_3p = breakpoints
                            .iter()
                            .find(|b| b.event_id == ev.event_id && b.breakpoint_nt == ev.u2);
                        let bp_3p_id = bp_3p.map(|b| b.breakpoint_id.clone());
                        let tract_st = (ci_r_5p + 1).min(ev.u2);
                        let ci_l_3p = bp_3p
                            .map(|b| b.ci_left)
                            .unwrap_or(ev.u2.saturating_sub(5).max(1))
                            .max((mid_tract + 1).max(tract_st + 1).min(ev.u2));
                        let ci_r_3p = bp_3p
                            .map(|b| b.ci_right)
                            .unwrap_or((ev.u2 + 5).min(l))
                            .max(ci_l_3p);

                        // Donor tract between 5' and 3' plateaus
                        if ci_l_3p > tract_st {
                            paint_mosaic_segment(
                                &mut segments,
                                MosaicSegment {
                                    start: tract_st,
                                    end: ci_l_3p - 1,
                                    lineage: format!("Donor: {}", donor_display),
                                    color: donor_col,
                                    is_plateau: None,
                                    breakpoint_id: bp_5p_id.clone(),
                                    event_id: Some(ev.event_id),
                                },
                            );
                        }

                        // 3' Plateau
                        let plat_3p_w = if ci_r_3p >= ci_l_3p {
                            ci_r_3p - ci_l_3p
                        } else {
                            0
                        };
                        paint_mosaic_segment(
                            &mut segments,
                            MosaicSegment {
                                start: ci_l_3p,
                                end: ci_r_3p,
                                lineage: format!(
                                    "Breakpoint 3' [nt {}, Δ={} nt]",
                                    ev.u2, plat_3p_w
                                ),
                                color: "#fbbf24".to_string(),
                                is_plateau: Some(true),
                                breakpoint_id: bp_3p_id,
                                event_id: Some(ev.event_id),
                            },
                        );
                    }
                }
            }

            mosaic_taxa.insert(t.clone(), segments);
        }
    }

    // 5B. Region-Aware Ancestral Color Inheritance (Fix for Issue #13)
    // Ensures that if a sequence obtains a genomic region from a donor (which may itself be a recombinant),
    // the displayed color matches the exact ancestral color of the donor in that specific genomic interval.
    for _pass in 0..4 {
        let current_mosaic = mosaic_taxa.clone();
        let mut changed = false;

        for (taxon_name, segments) in mosaic_taxa.iter_mut() {
            if rec_taxa_set.contains(taxon_name) {
                // Collect primary home colors in this taxon for contrast preservation
                let mut home_colors: Vec<(String, String)> = Vec::new();
                for seg in segments.iter() {
                    if seg.lineage.starts_with("Parent: ") {
                        let h_name = seg.lineage.trim_start_matches("Parent: ");
                        let c = resolve_lineage_regional_color(
                            h_name,
                            seg.start,
                            seg.end,
                            &current_mosaic,
                            &parent_colors,
                        );
                        home_colors.push((h_name.to_string(), c));
                    }
                }

                for seg in segments.iter_mut() {
                    if seg.is_plateau == Some(true) || seg.color == "#fbbf24" {
                        continue;
                    }

                    if seg.lineage.starts_with("Donor: ") {
                        let donor_target = seg.lineage.trim_start_matches("Donor: ");
                        let mut resolved_col = resolve_lineage_regional_color(
                            donor_target,
                            seg.start,
                            seg.end,
                            &current_mosaic,
                            &parent_colors,
                        );

                        // Only switch color if a DIFFERENT home lineage name shares the same palette color
                        if home_colors
                            .iter()
                            .any(|(hn, hc)| hn != donor_target && hc == &resolved_col)
                        {
                            if let Some(alt) = PALETTE
                                .iter()
                                .find(|&&c| !home_colors.iter().any(|(_, hc)| hc == c))
                            {
                                resolved_col = alt.to_string();
                            }
                        }

                        if resolved_col != seg.color {
                            seg.color = resolved_col;
                            changed = true;
                        }
                    } else if seg.lineage.starts_with("Parent: ") {
                        let home_target = seg.lineage.trim_start_matches("Parent: ");
                        let resolved_col = resolve_lineage_regional_color(
                            home_target,
                            seg.start,
                            seg.end,
                            &current_mosaic,
                            &parent_colors,
                        );
                        if resolved_col != seg.color {
                            seg.color = resolved_col;
                            changed = true;
                        }
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    // 6. Macro genomic segments
    let mut macro_segments = Vec::new();
    if breakpoints.is_empty() {
        macro_segments.push(MacroSegment {
            name: "Segment 1".to_string(),
            start: 1,
            end: l,
            color: "#3b82f6".to_string(),
        });
    } else {
        let mut cutpoints: Vec<usize> = breakpoints.iter().map(|b| b.breakpoint_nt).collect();
        cutpoints.sort_unstable();
        cutpoints.dedup();

        let mut seg_starts = vec![1];
        for &cp in &cutpoints {
            seg_starts.push(cp + 1);
        }
        let mut seg_ends = cutpoints;
        seg_ends.push(l);

        let seg_colors = [
            "#3b82f6", "#0ea5e9", "#8b5cf6", "#ec4899", "#10b981", "#f59e0b", "#06b6d4",
        ];
        for (s_i, (&s_st, &s_en)) in seg_starts.iter().zip(seg_ends.iter()).enumerate() {
            if s_en >= s_st {
                macro_segments.push(MacroSegment {
                    name: format!("Segment {}", s_i + 1),
                    start: s_st,
                    end: s_en,
                    color: seg_colors[s_i % seg_colors.len()].to_string(),
                });
            }
        }
    }

    // 7. Focal Recombinant & Continuous Signals
    let (
        focal_cand_idx,
        focal_cand_name,
        focal_home_idx,
        focal_home_name,
        focal_donor_idx_opt,
        focal_donor_name,
    ) = if let Some(ev) = scan_res.events.first() {
        (
            taxon_idx_map
                .get(ev.isolates[0].as_str())
                .copied()
                .unwrap_or(0),
            ev.isolates[0].clone(),
            ev.home_idx.unwrap_or(0),
            ev.home_name.clone(),
            ev.donor_idx,
            ev.donor_name.clone(),
        )
    } else {
        (
            0,
            aln.taxa[0].clone(),
            if n > 1 { 1 } else { 0 },
            if n > 1 {
                aln.taxa[1].clone()
            } else {
                aln.taxa[0].clone()
            },
            if n > 2 { Some(2) } else { None },
            if n > 2 {
                aln.taxa[2].clone()
            } else {
                "None".to_string()
            },
        )
    };

    let focal_donor_idx = focal_donor_idx_opt.unwrap_or_else(|| if n > 2 { 2 } else { 0 });

    // Compute site differences for focal candidate vs home and donor
    let mut diff_p1 = vec![0i32; l];
    let mut diff_p2 = vec![0i32; l];
    let mut informative_snps = Vec::new();
    let mut walk_x = Vec::new();
    let mut walk_s = Vec::new();
    let mut current_walk = 0i32;

    for u in 0..l {
        let cr = aln.get(focal_cand_idx, u);
        let c1 = aln.get(focal_home_idx, u);
        let c2 = aln.get(focal_donor_idx, u);

        if cr > 0 && c1 > 0 && c2 > 0 {
            if cr != c1 {
                diff_p1[u] = 1;
            }
            if cr != c2 {
                diff_p2[u] = 1;
            }
            if c1 != c2 {
                let (match_type, walk_step) = if cr == c1 {
                    ("match_p1", -1)
                } else if cr == c2 {
                    ("match_p2", 1)
                } else {
                    ("divergent", 0)
                };

                let base_char = |b: u8| match b {
                    1 => 'A',
                    2 => 'C',
                    3 => 'G',
                    4 => 'T',
                    _ => '-',
                };

                informative_snps.push(InformativeSnpRecord {
                    pos: u + 1,
                    match_type: match_type.to_string(),
                    base_r: base_char(cr),
                    base_p1: base_char(c1),
                    base_p2: base_char(c2),
                });

                current_walk += walk_step;
                walk_x.push(u + 1);
                walk_s.push(current_walk);
            }
        }
    }

    // Cumulative sums
    let mut cum_p1 = vec![0i32; l + 1];
    let mut cum_p2 = vec![0i32; l + 1];
    for u in 0..l {
        cum_p1[u + 1] = cum_p1[u] + diff_p1[u];
        cum_p2[u + 1] = cum_p2[u] + diff_p2[u];
    }

    // Grid sampling (smooth 300-500 points for canvas rendering)
    let grid_step = (l / 400).max(5);
    let mut u_grid = Vec::new();
    let mut cur_u = 0usize;
    while cur_u <= l {
        u_grid.push(cur_u);
        cur_u += grid_step;
    }
    if *u_grid.last().unwrap() != l {
        u_grid.push(l);
    }

    // Ensure all breakpoint coordinates are explicitly represented in u_grid
    for b in &breakpoints {
        if !u_grid.contains(&b.breakpoint_nt) {
            u_grid.push(b.breakpoint_nt);
        }
    }
    u_grid.sort_unstable();
    u_grid.dedup();

    let mut trajectory_x = Vec::with_capacity(u_grid.len());
    let mut trajectory_y = Vec::with_capacity(u_grid.len());
    let mut running_max = 0.0f64;
    let mut drawdown = Vec::with_capacity(u_grid.len());
    let mut brownian_bridge = Vec::with_capacity(u_grid.len());

    let y_final = (cum_p2[l] - cum_p1[l]) as f64;

    for &u in &u_grid {
        let y = if u == 0 {
            0.0
        } else {
            (cum_p2[u] - cum_p1[u]) as f64
        };
        trajectory_x.push(u);
        trajectory_y.push(y);

        if y > running_max {
            running_max = y;
        }
        drawdown.push(running_max - y);

        let bridge = y - (u as f64 / l.max(1) as f64) * y_final;
        brownian_bridge.push(bridge);
    }

    // Instantaneous numerical velocity dy
    let mut velocity_dy = Vec::with_capacity(trajectory_y.len());
    let m = trajectory_y.len();
    for i in 0..m {
        if m < 2 {
            velocity_dy.push(0.0);
        } else if i == 0 {
            let du = (trajectory_x[1] - trajectory_x[0]).max(1) as f64;
            velocity_dy.push((trajectory_y[1] - trajectory_y[0]) / du);
        } else if i == m - 1 {
            let du = (trajectory_x[m - 1] - trajectory_x[m - 2]).max(1) as f64;
            velocity_dy.push((trajectory_y[m - 1] - trajectory_y[m - 2]) / du);
        } else {
            let du = (trajectory_x[i + 1] - trajectory_x[i - 1]).max(1) as f64;
            velocity_dy.push((trajectory_y[i + 1] - trajectory_y[i - 1]) / du);
        }
    }

    let signals = VisualizationSignals {
        trajectory_x,
        trajectory_y,
        velocity_dy,
        informative_snps,
        walk_x,
        walk_s,
        drawdown,
        brownian_bridge,
    };

    // 8. 2D Classical / Landmark MDS Metric Subspaces
    // For N <= 250, solve full N x N MDS exactly; for N > 250, solve anchor eigensystem on M = 200
    // representative anchors and project all taxa (up to 5,000, including all 2,000 sequences) via
    // exact Gower out-of-sample projection so 2D coordinates are populated for every taxon without
    // stalling for 78+ seconds on three O(N^2 * L) matrices.
    let mds_indices: Vec<usize> = if n <= 250 {
        (0..n).collect()
    } else {
        let max_anchors = 200usize;
        let mut idx_set = Vec::with_capacity(max_anchors);
        let mut seen = HashSet::with_capacity(max_anchors);
        for &imp in &[focal_cand_idx, focal_home_idx, focal_donor_idx] {
            if imp < n && seen.insert(imp) {
                idx_set.push(imp);
            }
        }
        for &r_idx in &rec_indices {
            if idx_set.len() >= max_anchors * 2 / 3 {
                break;
            }
            if r_idx < n && seen.insert(r_idx) {
                idx_set.push(r_idx);
            }
        }
        for &p_idx in &parent_indices {
            if idx_set.len() >= max_anchors * 4 / 5 {
                break;
            }
            if p_idx < n && seen.insert(p_idx) {
                idx_set.push(p_idx);
            }
        }
        let rem = max_anchors.saturating_sub(idx_set.len()).max(1);
        for s in 0..rem {
            let idx = (s * (n - 1)) / rem.saturating_sub(1).max(1);
            if seen.insert(idx) {
                idx_set.push(idx);
            }
        }
        idx_set.sort_unstable();
        idx_set
    };

    let focal_bp_nt = breakpoints
        .first()
        .map(|b| b.breakpoint_nt)
        .unwrap_or(l / 2);

    let p1_end = focal_bp_nt.saturating_sub(1).max(1);
    let p2_start = focal_bp_nt.min(l);

    let coords_p1 = compute_mds_2d(aln, 0, p1_end, &mds_indices);
    let coords_p2 = compute_mds_2d(aln, p2_start.saturating_sub(1), l, &mds_indices);

    let subspaces = VisualizationSubspaces {
        partition_1: coords_p1,
        partition_2: coords_p2,
        p1_title: format!("Partition 1 (nt 1 – {})", p1_end),
        p2_title: format!("Partition 2 (nt {} – {})", p2_start, l),
    };

    // 9. Time-invariant Canonical Metric Manifold & Chord Colinearity
    let canonical_manifold =
        compute_canonical_manifold(aln, focal_cand_idx, focal_home_idx, focal_donor_idx, &mds_indices);

    // 9B. Genome-wide Recombination Hotspot / Coldspot Flux & Cumulative Strain Landscape
    let recombination_landscape =
        compute_recombination_landscape(aln, scan_res, &breakpoints);

    // 10. Summary Metadata
    let mean_plat_w = if !breakpoints.is_empty() {
        (total_plateau_width as f64 / breakpoints.len() as f64 * 10.0).round() / 10.0
    } else {
        0.0
    };

    let max_support = scan_res
        .events
        .iter()
        .map(|e| e.z_phys)
        .fold(0.0f64, |a, b| a.max(b));

    let metadata = VisualizationMetadata {
        title,
        engine_version: env!("CARGO_PKG_VERSION").to_string(),
        alignment_length: l,
        taxa_count: n,
        query_id: focal_cand_name,
        p1_id: focal_home_name,
        p2_id: focal_donor_name,
        unique_events_count: scan_res.events.len(),
        recombinants_count: rec_taxa_set.len(),
        breakpoints_count: breakpoints.len(),
        informative_snps_count: scan_res.informative_sites,
        mean_plateau_width: mean_plat_w,
        max_support: (max_support * 100.0).round() / 100.0,
        run_time_ms: scan_res.run_time_ms,
        is_non_recombinant_h0: scan_res.is_non_recombinant_h0,
    };

    let input_taxa = if compact_large_n {
        aln.taxa.clone()
    } else {
        Vec::new()
    };

    VisualizationDossier {
        metadata,
        taxa: sorted_taxa,
        input_taxa,
        taxa_meta,
        macro_segments,
        breakpoints,
        mosaic_taxa,
        signals,
        subspaces,
        canonical_manifold: Some(canonical_manifold),
        recombination_landscape: Some(recombination_landscape),
        sequences,
    }
}

/// Fast top-2 eigenpair solver using exact Jacobi for small N (<= 80) and deflated power iteration for large N (> 80).
fn top_two_eigenpairs(b: &[f64], n: usize) -> ([f64; 2], Vec<f64>, Vec<f64>) {
    if n == 0 {
        return ([0.0, 0.0], Vec::new(), Vec::new());
    }
    if n <= 80 {
        let eigh = symmetric_jacobi_eigh(b, n);
        let lam0 = eigh.eigenvalues.first().copied().unwrap_or(0.0);
        let lam1 = if n > 1 {
            eigh.eigenvalues.get(1).copied().unwrap_or(0.0)
        } else {
            0.0
        };
        let mut v0 = vec![0.0f64; n];
        let mut v1 = vec![0.0f64; n];
        for i in 0..n {
            v0[i] = eigh.eigenvectors[i * n];
            if n > 1 {
                v1[i] = eigh.eigenvectors[i * n + 1];
            }
        }
        return ([lam0, lam1], v0, v1);
    }

    let power_step = |deflate_v: Option<&[f64]>, deflate_lam: f64, seed_offset: usize| -> (f64, Vec<f64>) {
        let mut v = vec![0.0f64; n];
        for i in 0..n {
            v[i] = (((i + seed_offset + 1) * 2654435761) % 1000) as f64 / 1000.0 - 0.5;
        }
        let mut norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for x in &mut v {
                *x /= norm;
            }
        }

        let mut next_v = vec![0.0f64; n];
        for _iter in 0..45 {
            let dot_def = if let Some(dv) = deflate_v {
                dv.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<f64>()
            } else {
                0.0
            };
            for i in 0..n {
                let row = &b[i * n..(i + 1) * n];
                let mut s = 0.0f64;
                for j in 0..n {
                    s += row[j] * v[j];
                }
                if let Some(dv) = deflate_v {
                    s -= deflate_lam * dv[i] * dot_def;
                }
                next_v[i] = s;
            }
            norm = next_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-12 {
                break;
            }
            for i in 0..n {
                v[i] = next_v[i] / norm;
            }
        }

        let dot_def = if let Some(dv) = deflate_v {
            dv.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<f64>()
        } else {
            0.0
        };
        let mut rayleigh = 0.0f64;
        for i in 0..n {
            let row = &b[i * n..(i + 1) * n];
            let mut s = 0.0f64;
            for j in 0..n {
                s += row[j] * v[j];
            }
            if let Some(dv) = deflate_v {
                s -= deflate_lam * dv[i] * dot_def;
            }
            rayleigh += v[i] * s;
        }
        (rayleigh.max(0.0), v)
    };

    let (lam0, v0) = power_step(None, 0.0, 0);
    let (lam1, v1) = if n > 1 {
        power_step(Some(&v0), lam0, 17)
    } else {
        (0.0, vec![0.0; n])
    };
    ([lam0, lam1], v0, v1)
}

/// Computes genome-wide Recombination Hotspot / Coldspot Flux Phi_bp(u) and Cumulative Reticulation Strain rho(u).
fn compute_recombination_landscape(
    aln: &Alignment,
    scan_res: &ScanResult,
    breakpoints: &[VisualizationBreakpoint],
) -> RecombinationLandscape {
    let l = aln.length.max(1);
    let n_grid = l.clamp(60, 400);
    let mut grid_x = Vec::with_capacity(n_grid);
    for g in 0..n_grid {
        let u = 1 + ((g * (l.saturating_sub(1))) / (n_grid.saturating_sub(1).max(1)));
        grid_x.push(u);
    }
    grid_x.dedup();
    let m = grid_x.len();

    let sigma = ((l as f64) / 45.0).clamp(15.0, 1500.0);
    let two_sigma_sq = 2.0 * sigma * sigma;

    let mut flux_phi = vec![0.0f64; m];
    let mut strain_density = vec![0.0f64; m];

    // 1. Gaussian-smoothed breakpoint flux Phi_bp(u) weighted by physical support Z_phys
    for bp in breakpoints {
        let bp_u = bp.breakpoint_nt as f64;
        let weight = bp.kinetic_z.max(1.0);
        let plat_sigma = (bp.plateau_width as f64 * 0.5).max(sigma);
        let denom = 2.0 * plat_sigma * plat_sigma;
        for (idx, &gx) in grid_x.iter().enumerate() {
            let d = (gx as f64) - bp_u;
            flux_phi[idx] += weight * (-d * d / denom).exp();
        }
    }

    // Also add tract-interior & edge strain density from verified events
    for ev in &scan_res.events {
        let u1 = ev.u1.min(l);
        let u2 = ev.u2.min(l).max(u1);
        let z_w = ev.z_phys.max(1.0) * (1.0 + (ev.isolates.len() as f64).ln());
        let span = (u2 - u1 + 1).max(1) as f64;
        for (idx, &gx) in grid_x.iter().enumerate() {
            if gx >= u1 && gx <= u2 {
                strain_density[idx] += z_w / span;
            }
            let d1 = (gx as f64) - (u1 as f64);
            let d2 = (gx as f64) - (u2 as f64);
            strain_density[idx] += 0.35 * (z_w / sigma) * ((-d1 * d1 / two_sigma_sq).exp() + (-d2 * d2 / two_sigma_sq).exp());
        }
    }

    // Normalize flux_phi to Z-density units and round for compact JSON
    for val in &mut flux_phi {
        *val = (*val * 1000.0).round() / 1000.0;
    }

    // 2. Cumulative reticulation strain rho(u) in [0, 1]
    let mut cumulative_strain = vec![0.0f64; m];
    let mut run_sum = 0.0f64;
    for (idx, &sd) in strain_density.iter().enumerate() {
        run_sum += sd.max(0.0);
        cumulative_strain[idx] = run_sum;
    }
    if run_sum > 1e-12 {
        for val in &mut cumulative_strain {
            *val = ((*val / run_sum) * 10000.0).round() / 10000.0;
        }
    }

    // 3. Identify statistically enriched Hotspot peaks and conserved Coldspot valleys
    let mean_flux = flux_phi.iter().sum::<f64>() / (m as f64);
    let var_flux = flux_phi
        .iter()
        .map(|&v| (v - mean_flux) * (v - mean_flux))
        .sum::<f64>()
        / (m as f64);
    let sd_flux = var_flux.sqrt();

    let max_flux = flux_phi.iter().cloned().fold(0.0f64, f64::max);
    let hotspot_threshold = if max_flux > 0.0 {
        (mean_flux + 0.85 * sd_flux)
            .max(0.25 * max_flux)
            .min(0.85 * max_flux)
    } else {
        1.0
    };
    let coldspot_threshold = if max_flux > 0.0 {
        (mean_flux * 0.20).min(0.05 * max_flux).max(0.02)
    } else {
        0.0
    };

    let mut hotspots = Vec::new();
    let mut coldspots = Vec::new();

    if !breakpoints.is_empty() && max_flux > 0.0 {
        // Segment contiguous hotspot intervals where flux_phi >= hotspot_threshold
        let mut i = 0usize;
        while i < m {
            if flux_phi[i] >= hotspot_threshold {
                let start_i = i;
                let mut peak_i = i;
                let mut sum_f = 0.0f64;
                while i < m && flux_phi[i] >= hotspot_threshold {
                    if flux_phi[i] > flux_phi[peak_i] {
                        peak_i = i;
                    }
                    sum_f += flux_phi[i];
                    i += 1;
                }
                let end_i = i - 1;
                let s_nt = grid_x[start_i];
                let e_nt = grid_x[end_i];
                let pk_nt = grid_x[peak_i];
                let pk_val = (flux_phi[peak_i] * 100.0).round() / 100.0;
                let mn_val = (sum_f / ((end_i - start_i + 1) as f64) * 100.0).round() / 100.0;
                let margin = (sigma as usize).max(25);
                let ev_cnt = breakpoints
                    .iter()
                    .filter(|b| {
                        b.breakpoint_nt + margin >= s_nt && b.breakpoint_nt <= e_nt + margin
                    })
                    .count()
                    .max(1);
                hotspots.push(RecombinationRegion {
                    start: s_nt,
                    end: e_nt,
                    peak_nt: pk_nt,
                    peak_flux: pk_val,
                    mean_flux: mn_val,
                    event_count: ev_cnt,
                    label: format!("Hotspot @ nt {}", pk_nt),
                });
            } else {
                i += 1;
            }
        }

        // Segment contiguous conserved coldspot basins where flux_phi <= coldspot_threshold
        // and no verified event breakpoint falls inside the core basin
        let min_cold_span = (l / 12).max(50);
        let mut j = 0usize;
        while j < m {
            if flux_phi[j] <= coldspot_threshold {
                let start_j = j;
                let mut min_j = j;
                let mut sum_f = 0.0f64;
                while j < m && flux_phi[j] <= coldspot_threshold {
                    if flux_phi[j] < flux_phi[min_j] {
                        min_j = j;
                    }
                    sum_f += flux_phi[j];
                    j += 1;
                }
                let end_j = j - 1;
                let s_nt = grid_x[start_j];
                let e_nt = grid_x[end_j];
                if e_nt.saturating_sub(s_nt) >= min_cold_span {
                    let mid_nt = (s_nt + e_nt) / 2;
                    let mn_val = (sum_f / ((end_j - start_j + 1) as f64) * 1000.0).round() / 1000.0;
                    coldspots.push(RecombinationRegion {
                        start: s_nt,
                        end: e_nt,
                        peak_nt: mid_nt,
                        peak_flux: (flux_phi[min_j] * 1000.0).round() / 1000.0,
                        mean_flux: mn_val,
                        event_count: 0,
                        label: format!("Coldspot [nt {}–{}]", s_nt, e_nt),
                    });
                }
            } else {
                j += 1;
            }
        }
    } else {
        coldspots.push(RecombinationRegion {
            start: 1,
            end: l,
            peak_nt: l / 2,
            peak_flux: 0.0,
            mean_flux: 0.0,
            event_count: 0,
            label: format!("Conserved Clonal Alignment [nt 1–{}]", l),
        });
    }

    RecombinationLandscape {
        grid_x,
        flux_phi,
        cumulative_strain,
        hotspot_threshold: (hotspot_threshold * 1000.0).round() / 1000.0,
        coldspot_threshold: (coldspot_threshold * 1000.0).round() / 1000.0,
        hotspots,
        coldspots,
    }
}

/// Computes 2D Classical / Landmark MDS coordinates over a sliced partition [start_site, end_site).
/// When `mds_indices.len() < aln.num_taxa`, solves the anchor eigensystem on `mds_indices` and projects
/// the remaining taxa (all taxa for N <= 5,000) via exact Gower out-of-sample projection.
fn compute_mds_2d(
    aln: &Alignment,
    start_site: usize,
    end_site: usize,
    mds_indices: &[usize],
) -> BTreeMap<String, Vec<f64>> {
    let n = aln.num_taxa;
    let m = mds_indices.len();
    let s_end = end_site.min(aln.length);
    let s_len = s_end.saturating_sub(start_site);

    let mut res = BTreeMap::new();
    if m < 2 || s_len == 0 {
        for &idx in mds_indices {
            res.insert(aln.taxa[idx].clone(), vec![0.0, 0.0]);
        }
        return res;
    }

    // Subsample sites when M^2 * s_len is large (using u64 to prevent 32-bit WASM usize overflow)
    let site_stride = if (m as u64) * (m as u64) * (s_len as u64) > 8_000_000 {
        (s_len / 500).max(1)
    } else {
        1
    };

    // Pairwise Hamming distance matrix on anchor taxa
    let mut d_sq = vec![0.0f64; m * m];
    for a in 0..m {
        let row_i = aln.row(mds_indices[a]);
        for b_idx in (a + 1)..m {
            let row_j = aln.row(mds_indices[b_idx]);
            let mut diff = 0usize;
            let mut valid = 0usize;
            let mut u = start_site;
            while u < s_end {
                let ci = row_i[u];
                let cj = row_j[u];
                if ci > 0 && cj > 0 {
                    valid += 1;
                    if ci != cj {
                        diff += 1;
                    }
                }
                u += site_stride;
            }
            let dist = if valid > 0 {
                diff as f64 / valid as f64
            } else {
                0.0
            };
            let d2 = dist * dist;
            d_sq[a * m + b_idx] = d2;
            d_sq[b_idx * m + a] = d2;
        }
    }

    // Double centering: B = -0.5 * H @ D_sq @ H
    let mut row_means = vec![0.0f64; m];
    let mut total_mean = 0.0f64;
    for a in 0..m {
        let mut sum = 0.0;
        for b_idx in 0..m {
            sum += d_sq[a * m + b_idx];
        }
        row_means[a] = sum / (m as f64);
        total_mean += sum;
    }
    total_mean /= (m * m) as f64;

    let mut b = vec![0.0f64; m * m];
    for a in 0..m {
        for b_idx in 0..m {
            b[a * m + b_idx] = -0.5 * (d_sq[a * m + b_idx] - row_means[a] - row_means[b_idx] + total_mean);
        }
    }

    let (evals, v0, v1) = top_two_eigenpairs(&b, m);
    let lam0 = evals[0].max(0.0).sqrt();
    let lam1 = if m > 1 {
        evals[1].max(0.0).sqrt()
    } else {
        0.0
    };

    for a in 0..m {
        let x = v0[a] * lam0;
        let y = if m > 1 { v1[a] * lam1 } else { 0.0 };
        res.insert(
            aln.taxa[mds_indices[a]].clone(),
            vec![
                (x * 10000.0).round() / 10000.0,
                (y * 10000.0).round() / 10000.0,
            ],
        );
    }

    // Project remaining taxa into the 2D MDS subspace via exact Gower out-of-sample projection
    // so all sequences (up to 5,000, including all 2,000 sequences) have 2D coordinates.
    if m < n {
        let mut is_anchor = vec![false; n];
        for &idx in mds_indices {
            if idx < n {
                is_anchor[idx] = true;
            }
        }
        let max_project = 5_000usize;
        let taxon_step = if n <= max_project {
            1
        } else {
            (n / max_project).max(1)
        };
        let proj_stride = (s_len / 200).max(site_stride).max(1);
        let inv_2lam0 = if lam0 > 1e-6 { 0.5 / lam0 } else { 0.0 };
        let inv_2lam1 = if lam1 > 1e-6 { 0.5 / lam1 } else { 0.0 };

        let mut idx = 0usize;
        while idx < n {
            if !is_anchor[idx] {
                let row_i = aln.row(idx);
                let mut sum0 = 0.0f64;
                let mut sum1 = 0.0f64;
                for b_idx in 0..m {
                    let row_b = aln.row(mds_indices[b_idx]);
                    let mut diff = 0usize;
                    let mut valid = 0usize;
                    let mut u = start_site;
                    while u < s_end {
                        let ci = row_i[u];
                        let cb = row_b[u];
                        if ci > 0 && cb > 0 {
                            valid += 1;
                            if ci != cb {
                                diff += 1;
                            }
                        }
                        u += proj_stride;
                    }
                    let dist = if valid > 0 {
                        diff as f64 / valid as f64
                    } else {
                        0.0
                    };
                    let d2 = dist * dist;
                    let delta = row_means[b_idx] - d2;
                    sum0 += v0[b_idx] * delta;
                    if m > 1 {
                        sum1 += v1[b_idx] * delta;
                    }
                }
                let x = sum0 * inv_2lam0;
                let y = sum1 * inv_2lam1;
                res.insert(
                    aln.taxa[idx].clone(),
                    vec![
                        (x * 10000.0).round() / 10000.0,
                        (y * 10000.0).round() / 10000.0,
                    ],
                );
            }
            idx += taxon_step;
        }
    }

    res
}

/// Finds the regional color of a lineage (donor or home) across a genomic interval [u_start, u_end].
/// If the lineage is a known taxon in mosaic_taxa, queries its non-plateau segment with maximum overlap.
/// Otherwise, falls back to parent_colors.
fn resolve_lineage_regional_color(
    lineage_name: &str,
    u_start: usize,
    u_end: usize,
    mosaic_taxa: &BTreeMap<String, Vec<MosaicSegment>>,
    parent_colors: &BTreeMap<String, String>,
) -> String {
    if lineage_name == "Ghost (Unsampled)" {
        return "#94a3b8".to_string();
    }

    if let Some(target_segments) = mosaic_taxa.get(lineage_name) {
        let mut best_overlap = 0usize;
        let mut best_color = None;

        for seg in target_segments {
            // Ignore breakpoint plateaus
            if seg.is_plateau == Some(true) || seg.color == "#fbbf24" {
                continue;
            }

            // Calculate overlap between [u_start, u_end] and [seg.start, seg.end]
            if seg.end >= u_start && seg.start <= u_end {
                let overlap = (seg.end.min(u_end) + 1).saturating_sub(seg.start.max(u_start));
                if overlap > best_overlap {
                    best_overlap = overlap;
                    best_color = Some(seg.color.clone());
                }
            }
        }

        if let Some(col) = best_color {
            return col;
        }
    }

    parent_colors
        .get(lineage_name)
        .cloned()
        .unwrap_or_else(|| "#d55e00".to_string())
}

/// Computes whole-gene time-invariant Canonical Metric Manifold (R^2),
/// out-of-sample continuous Gower trajectory, and chord colinearity cos(theta).
fn compute_canonical_manifold(
    aln: &Alignment,
    cand_idx: usize,
    home_idx: usize,
    donor_idx: usize,
    mds_indices: &[usize],
) -> CanonicalManifold {
    let n = aln.num_taxa;
    let m = mds_indices.len();
    let l = aln.length;

    let site_stride = if (m as u64) * (m as u64) * (l as u64) > 8_000_000 {
        (l / 500).max(1)
    } else {
        1
    };

    // 1. Whole-alignment pairwise distance matrix D_glob on anchor taxa
    let mut d_glob_sq = vec![0.0f64; m * m];
    for a in 0..m {
        let row_i = aln.row(mds_indices[a]);
        for b_idx in (a + 1)..m {
            let row_j = aln.row(mds_indices[b_idx]);
            let mut diff = 0usize;
            let mut valid = 0usize;
            let mut u = 0usize;
            while u < l {
                let ci = row_i[u];
                let cj = row_j[u];
                if ci > 0 && cj > 0 {
                    valid += 1;
                    if ci != cj {
                        diff += 1;
                    }
                }
                u += site_stride;
            }
            let dist = if valid > 0 {
                diff as f64 / valid as f64
            } else {
                0.0
            };
            let d2 = dist * dist;
            d_glob_sq[a * m + b_idx] = d2;
            d_glob_sq[b_idx * m + a] = d2;
        }
    }

    // Double centering: B = -0.5 * H @ D_glob_sq @ H
    let mut row_means = vec![0.0f64; m];
    let mut total_mean = 0.0f64;
    for a in 0..m {
        let mut sum = 0.0;
        for b_idx in 0..m {
            sum += d_glob_sq[a * m + b_idx];
        }
        row_means[a] = sum / (m as f64);
        total_mean += sum;
    }
    total_mean /= (m * m) as f64;

    let mut b = vec![0.0f64; m * m];
    for a in 0..m {
        for b_idx in 0..m {
            b[a * m + b_idx] = -0.5 * (d_glob_sq[a * m + b_idx] - row_means[a] - row_means[b_idx] + total_mean);
        }
    }

    let (evals, v0, v1) = top_two_eigenpairs(&b, m);
    let k_eff = m.min(2);
    let lam0 = evals[0].max(1e-9);
    let lam1 = if k_eff > 1 {
        evals[1].max(1e-9)
    } else {
        1e-9
    };

    let sqrt_lam0 = lam0.sqrt();
    let sqrt_lam1 = lam1.sqrt();
    let inv_sqrt_lam0 = 1.0 / sqrt_lam0;
    let inv_sqrt_lam1 = 1.0 / sqrt_lam1;

    let mut ref_coords = BTreeMap::new();
    for a in 0..m {
        let x = v0[a] * sqrt_lam0;
        let y = if k_eff > 1 {
            v1[a] * sqrt_lam1
        } else {
            0.0
        };
        ref_coords.insert(
            aln.taxa[mds_indices[a]].clone(),
            vec![
                (x * 10000.0).round() / 10000.0,
                (y * 10000.0).round() / 10000.0,
            ],
        );
    }

    // Project remaining non-anchor taxa (all taxa for N <= 5,000) into ref_coords via Gower projection
    if m < n {
        let mut is_anchor = vec![false; n];
        for &idx in mds_indices {
            if idx < n {
                is_anchor[idx] = true;
            }
        }
        let max_project = 5_000usize;
        let taxon_step = if n <= max_project {
            1
        } else {
            (n / max_project).max(1)
        };
        let proj_stride = (l / 200).max(site_stride).max(1);
        let mut idx = 0usize;
        while idx < n {
            if !is_anchor[idx] {
                let row_i = aln.row(idx);
                let mut sum0 = 0.0f64;
                let mut sum1 = 0.0f64;
                for b_idx in 0..m {
                    let row_b = aln.row(mds_indices[b_idx]);
                    let mut diff = 0usize;
                    let mut valid = 0usize;
                    let mut u = 0usize;
                    while u < l {
                        let ci = row_i[u];
                        let cb = row_b[u];
                        if ci > 0 && cb > 0 {
                            valid += 1;
                            if ci != cb {
                                diff += 1;
                            }
                        }
                        u += proj_stride;
                    }
                    let dist = if valid > 0 {
                        diff as f64 / valid as f64
                    } else {
                        0.0
                    };
                    let d2 = dist * dist;
                    let delta = row_means[b_idx] - d2;
                    sum0 += v0[b_idx] * delta;
                    if k_eff > 1 {
                        sum1 += v1[b_idx] * delta;
                    }
                }
                let x = 0.5 * inv_sqrt_lam0 * sum0;
                let y = 0.5 * inv_sqrt_lam1 * sum1;
                ref_coords.insert(
                    aln.taxa[idx].clone(),
                    vec![
                        (x * 10000.0).round() / 10000.0,
                        (y * 10000.0).round() / 10000.0,
                    ],
                );
            }
            idx += taxon_step;
        }
    }

    let home_coord = ref_coords
        .get(&aln.taxa[home_idx])
        .cloned()
        .unwrap_or_else(|| vec![0.0, 0.0]);
    let donor_coord = ref_coords
        .get(&aln.taxa[donor_idx])
        .cloned()
        .unwrap_or_else(|| vec![0.1, 0.0]);

    let chord_dx = donor_coord[0] - home_coord[0];
    let chord_dy = donor_coord[1] - home_coord[1];
    let chord_vector = vec![chord_dx, chord_dy];

    // 2. Continuous Out-of-sample Gower Trajectory
    // Slide a local window of length w = min(300, max(50, L / 20)) across alignment
    let win = (l / 20).clamp(50, 300);
    let step = (win / 4).max(5).max(l / 120);

    let mut gower_trajectory = Vec::new();
    let cand_row = aln.row(cand_idx);

    let mut p = 0usize;
    while p + win <= l {
        let u_mid = p + win / 2;

        // Compute local distance vector d_loc^2 from cand_idx to anchor taxa
        let mut d_loc_sq = vec![0.0f64; m];
        for b_idx in 0..m {
            let row_j = aln.row(mds_indices[b_idx]);
            let mut diff = 0usize;
            let mut valid = 0usize;
            for u in p..(p + win) {
                let cr = cand_row[u];
                let cj = row_j[u];
                if cr > 0 && cj > 0 {
                    valid += 1;
                    if cr != cj {
                        diff += 1;
                    }
                }
            }
            let d = if valid > 0 {
                diff as f64 / valid as f64
            } else {
                0.0
            };
            d_loc_sq[b_idx] = d * d;
        }

        // Exact Gower projection: z = 0.5 * Lambda^{-1/2} * V^T * (row_means - d_loc_sq)
        let mut sum0 = 0.0f64;
        let mut sum1 = 0.0f64;
        for b_idx in 0..m {
            let delta = row_means[b_idx] - d_loc_sq[b_idx];
            sum0 += v0[b_idx] * delta;
            if k_eff > 1 {
                sum1 += v1[b_idx] * delta;
            }
        }
        let z_x = 0.5 * inv_sqrt_lam0 * sum0;
        let z_y = 0.5 * inv_sqrt_lam1 * sum1;

        gower_trajectory.push(GowerPoint {
            x: (z_x * 10000.0).round() / 10000.0,
            y: (z_y * 10000.0).round() / 10000.0,
            u: u_mid,
        });

        p += step;
    }

    // 3. Directional Chord Colinearity cos(theta)
    let (cos_theta, theta_deg, is_colinear) = if gower_trajectory.len() >= 2 {
        let first_pt = gower_trajectory.first().unwrap();
        let last_pt = gower_trajectory.last().unwrap();
        let traj_dx = last_pt.x - first_pt.x;
        let traj_dy = last_pt.y - first_pt.y;

        let dot = traj_dx * chord_dx + traj_dy * chord_dy;
        let norm_traj = (traj_dx * traj_dx + traj_dy * traj_dy).sqrt();
        let norm_chord = (chord_dx * chord_dx + chord_dy * chord_dy).sqrt();

        if norm_traj > 1e-6 && norm_chord > 1e-6 {
            let cos_th = (dot / (norm_traj * norm_chord)).clamp(-1.0, 1.0);
            let th_rad = cos_th.acos();
            let th_deg = th_rad * 180.0 / std::f64::consts::PI;
            (cos_th, th_deg, cos_th >= 0.40)
        } else {
            (0.0, 90.0, false)
        }
    } else {
        (1.0, 0.0, true)
    };

    CanonicalManifold {
        ref_coords,
        gower_trajectory,
        home_coord,
        donor_coord,
        chord_vector,
        cos_theta: (cos_theta * 1000.0).round() / 1000.0,
        theta_deg: (theta_deg * 10.0).round() / 10.0,
        is_colinear,
    }
}

/// Renders a standalone, fully self-contained HTML interactive dashboard with embedded JSON data.
pub fn generate_standalone_html(dossier: &VisualizationDossier) -> Result<String, String> {
    let json_str = serde_json::to_string(dossier)
        .map_err(|e| format!("Failed to serialize VisualizationDossier to JSON: {}", e))?;

    let html = DASHBOARD_HTML_TEMPLATE
        .replace("__RHIZAEON_TITLE__", &dossier.metadata.title)
        .replace("__RHIZAEON_VERSION__", &dossier.metadata.engine_version)
        .replace("__RHIZAEON_DATA_JSON__", &json_str);

    Ok(html)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::RhizAeonEngine;
    use crate::fasta::parse_fasta;

    #[test]
    fn test_visualization_dossier_generation() {
        let fasta = "\
>R
AAAAAAAAGGGGGGGG
>P1
AAAAAAAAAAAAAAAA
>P2
GGGGGGGGGGGGGGGG
>O
TTTTTTTTTTTTTTTT
";
        let aln = parse_fasta(fasta).unwrap();
        let engine = RhizAeonEngine::new();
        let scan_res = engine.scan(&aln);

        let dossier = generate_visualization_dossier(&aln, &scan_res, Some("Test Recombination"));

        assert_eq!(dossier.metadata.taxa_count, 4);
        assert_eq!(dossier.metadata.alignment_length, 16);
        assert_eq!(dossier.taxa.len(), 4);
        assert_eq!(dossier.sequences.len(), 4);
        assert!(dossier.canonical_manifold.is_some());

        let html = generate_standalone_html(&dossier).unwrap();
        assert!(html.contains("<title>Test Recombination</title>"));
        assert!(html.contains("<script id=\"rhizaeon-data\" type=\"application/json\">"));
    }

    #[test]
    fn test_regional_ancestral_color_inheritance() {
        // Multi-generation introgression: P2 -> L -> H, with P1 as home
        let mut p1_seq = String::new();
        let mut p2_seq = String::new();
        let mut l_seq = String::new();
        let mut h_seq = String::new();
        let mut o_seq = String::new();

        for _ in 0..500 {
            p1_seq.push('A');
            p2_seq.push('C');
            l_seq.push('C'); // Tract 1..500 from P2
            h_seq.push('C'); // Tract 1..500 from L
            o_seq.push('T');
        }
        for _ in 0..500 {
            p1_seq.push('A');
            p2_seq.push('C');
            l_seq.push('A'); // Tract 501..1000 from P1
            h_seq.push('A'); // Tract 501..1000 from P1
            o_seq.push('T');
        }

        let fasta = format!(
            ">P1\n{}\n>P2\n{}\n>L\n{}\n>H\n{}\n>O\n{}\n",
            p1_seq, p2_seq, l_seq, h_seq, o_seq
        );
        let aln = parse_fasta(&fasta).unwrap();
        let engine = RhizAeonEngine::new();
        let scan_res = engine.scan(&aln);

        let dossier = generate_visualization_dossier(&aln, &scan_res, Some("Test Multi-Hop"));

        // If L and H have events where L is donor to H:
        if let (Some(l_tracks), Some(h_tracks)) =
            (dossier.mosaic_taxa.get("L"), dossier.mosaic_taxa.get("H"))
        {
            let l_donor_seg = l_tracks
                .iter()
                .find(|s| s.lineage.contains("Donor") && s.start <= 250);
            let h_donor_seg = h_tracks
                .iter()
                .find(|s| s.lineage.contains("Donor") && s.start <= 250);
            if let (Some(l_seg), Some(h_seg)) = (l_donor_seg, h_donor_seg) {
                // H's donor tract color MUST match L's donor tract color in that region!
                assert_eq!(h_seg.color, l_seg.color);
            }
        }
    }
}
