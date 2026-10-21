use crate::attention::{compute_attention_masks, AttentionMasks, PhyloAttentionEngine};
use crate::attribution::TrajectoryParentalResolver;
use crate::due_diligence::SequenceDueDiligence;
use crate::fasta::{
    compute_min_informative_sites, count_parsimony_informative_sites,
    extract_informative_snp_alignment, trim_coverage_envelope,
};
use crate::landmarks::select_landmarks;
use crate::polishing::CumulativeTrajectoryPolisher;
use crate::prior::PhyloBias;
use crate::screener::LocalFluxScreener;
use crate::soft_root::compute_consensus_soft_root;
use crate::tree_rope::TreeRoPE;
use crate::types::{Alignment, ScanResult, ScreeningResult};

pub struct RhizAeonEngine {
    pub target_landmarks: usize,
    pub dim: usize,
    pub epsilon_0: f64,
    pub decay_lambda: f64,
    pub root_sink_factor: f64,
    pub poisson_floor: usize,
    pub alpha: f64,
    pub tier2_stringency: f64,
    pub tier3_stringency: f64,
    /// Informative SNP compression mode:
    /// None = Auto-trigger (L > 50,000 bp OR (L > 10,000 bp AND M_inf / L < 0.10))
    /// Some(true) = Force SNP compression
    /// Some(false) = Force full physical nucleotide scan
    pub compress_snps: Option<bool>,
}

impl Default for RhizAeonEngine {
    fn default() -> Self {
        Self {
            target_landmarks: 16,
            dim: 8,
            epsilon_0: 0.25,
            decay_lambda: 4.0 / 3.0,
            root_sink_factor: 1.0,
            poisson_floor: 3,
            alpha: 0.05,
            tier2_stringency: 1.0,
            tier3_stringency: 1.0,
            compress_snps: None,
        }
    }
}

impl RhizAeonEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_landmarks(mut self, target_landmarks: usize) -> Self {
        self.target_landmarks = target_landmarks;
        self
    }

    pub fn with_compress_snps(mut self, compress: bool) -> Self {
        self.compress_snps = Some(compress);
        self
    }

    pub fn with_auto_compress_snps(mut self) -> Self {
        self.compress_snps = None;
        self
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_poisson_floor(mut self, floor: usize) -> Self {
        self.poisson_floor = floor;
        self
    }

    pub fn with_tier2_stringency(mut self, stringency: f64) -> Self {
        self.tier2_stringency = stringency;
        self
    }

    pub fn with_tier3_stringency(mut self, stringency: f64) -> Self {
        self.tier3_stringency = stringency;
        self
    }

    /// Scans an alignment for mosaic recombination events using the full PA-FF v2.1 pipeline
    pub fn scan(&self, raw_aln: &Alignment) -> ScanResult {
        let (aln_trimmed, u_offset) = trim_coverage_envelope(raw_aln);
        let n = aln_trimmed.num_taxa;
        let l_chromosomal = raw_aln.length;

        let m_inf = count_parsimony_informative_sites(&aln_trimmed);
        let base_m_min = compute_min_informative_sites(n);
        let m_min = if (self.tier3_stringency - 1.0).abs() < 1e-9 {
            base_m_min
        } else {
            ((base_m_min as f64 * self.tier3_stringency.min(1.0)).round() as usize).max(2)
        };

        // Early exit: non-recombinant H0 if informative sites below admission floor
        if m_inf < m_min {
            return ScanResult {
                num_taxa: n,
                length: l_chromosomal,
                informative_sites: m_inf,
                is_non_recombinant_h0: true,
                screening: ScreeningResult {
                    v_cont: vec![0.0; n],
                    v_root: vec![0.0; n],
                    r_root: vec![0.0; n],
                    gamma_root: vec![0.0; n],
                    rate_burst_indices: Vec::new(),
                    ghost_candidate_indices: Vec::new(),
                    candidate_recombinant_indices: Vec::new(),
                    clonal_backbone_indices: (0..n).collect(),
                    median_v_cont: 0.0,
                    is_non_recombinant_h0: true,
                },
                events: Vec::new(),
                run_time_ms: 0.0,
            };
        }

        // Auto-Trigger Threshold Policy:
        // Automatically compress to informative SNPs when L > 30,000 bp (bacterial/chromosomal scale)
        // or when L > 10,000 bp AND N >= 48, or when N * L > 600,000 matrix cells
        let should_compress = self.compress_snps.unwrap_or_else(|| {
            (aln_trimmed.length > 30_000 && n > 12)
                || (aln_trimmed.length > 50_000)
                || (aln_trimmed.length > 10_000 && n >= 48)
                || (n * aln_trimmed.length > 600_000)
        });

        let (aln, snp_map_opt) = if should_compress && m_inf >= m_min {
            let (snp_aln, map) = extract_informative_snp_alignment(&aln_trimmed);
            (snp_aln, Some(map))
        } else {
            (aln_trimmed, None)
        };
        let l = aln.length;

        // 1. Landmark Selection on Delta^K with adaptive scaling for large cohorts
        let effective_k = if self.target_landmarks == 16 {
            self.target_landmarks.max(n / 2).min(32).min(n)
        } else {
            self.target_landmarks.max(2).min(n).min(127)
        };
        let landmarks = select_landmarks(&aln, effective_k);

        // 2. Tree-RoPE & Prior
        let rope = TreeRoPE::new(self.dim);
        let m_matrix = rope.compute_query_landmark_rotary_factors(&aln, &landmarks);

        let prior = PhyloBias::new(self.epsilon_0, self.decay_lambda);
        let b_prior = prior.compute_prior_matrix(&aln, &landmarks);

        // 3. Ancestral Simplex Soft Root
        let root_seq = compute_consensus_soft_root(&aln, None);

        // 4. Attention & Zero-drift Force Field. Only the (N, K+1) sequence-wide mean and the
        // (N, K) masks are retained; the attention, force and momentum rows themselves are
        // streamed per taxon, so no (L, N, K+1) tensor is ever allocated.
        let attention = PhyloAttentionEngine::new(self.dim, self.root_sink_factor);
        let masks = compute_attention_masks(&aln, &landmarks);
        let mean_a = attention
            .compute_mean_attention(&aln, &landmarks, &m_matrix, &b_prior, &root_seq, &masks);

        // 5. Pass 1 Screener
        let screener = LocalFluxScreener::new().with_tier2_stringency(self.tier2_stringency);
        let screening = screener.screen(
            &attention, &m_matrix, &b_prior, &root_seq, &masks, &mean_a, &aln, &landmarks,
        );

        if screening.is_non_recombinant_h0 {
            return ScanResult {
                num_taxa: n,
                length: l,
                informative_sites: m_inf,
                is_non_recombinant_h0: true,
                screening,
                events: Vec::new(),
                run_time_ms: 0.0,
            };
        }

        // Pass 2: Retain full canonical landmark basis (DIR-RHIZ-PAFF-012)
        // Discarding candidate taxa from the landmark basis corrupts authentic parental lineages
        // (e.g. Subtype B in HIV-1) that exhibit legitimate contrast with recombinant children.
        let num_channels = landmarks.num_landmarks + 1;

        // 6. Pass 2 Polishing & Pass 3 Due Diligence on candidates
        let (d_sister, d_clade) =
            crate::landmarks::compute_scale_invariant_distance_thresholds(&aln);
        let compression_ratio = if snp_map_opt.is_some() {
            (l as f64) / (l_chromosomal.max(1) as f64)
        } else {
            1.0
        };
        let resolver = TrajectoryParentalResolver::new().with_stringency(self.tier2_stringency);
        let polisher = CumulativeTrajectoryPolisher::new()
            .with_stringency(self.tier2_stringency)
            .with_compression_ratio(compression_ratio);
        let due_diligence = SequenceDueDiligence::new(self.poisson_floor, self.alpha, 100_000)
            .with_stringency(self.tier3_stringency)
            .with_compression_ratio(compression_ratio)
            .with_d_sister(d_sister)
            .with_d_clade(d_clade);

        // Pass 2 regenerates each candidate's rows into buffers allocated once per scan rather
        // than slicing resident tensors, so residency is one row per candidate for any candidate
        // count. Regenerating rather than caching matters because the screener's recombination
        // sieve admits nearly all of N on large cohorts.
        let run_candidate_scan = |target_aln: &Alignment,
                                  target_masks: &AttentionMasks,
                                  target_mean_a: &[f64],
                                  target_landmarks: &crate::types::LandmarkSet,
                                  target_screening: &crate::types::ScreeningResult|
         -> Vec<crate::types::RecombinationEvent> {
            let mut pass_events = Vec::new();

            // Reused across candidates; the row providers rewrite every element, so no stale
            // tail can survive into a reduction.
            let mut a_row = vec![0.0f64; l * num_channels];
            let mut c_row = vec![0.0f64; l * num_channels];
            // The force row is a by-product here: Pass 2 consumes only A and C.
            let mut f_scratch = vec![0.0f64; l * num_channels];

            for &cand_idx in &target_screening.candidate_recombinant_indices {
                attention.compute_attention_row(
                    target_aln,
                    target_landmarks,
                    &m_matrix,
                    &b_prior,
                    &root_seq,
                    target_masks,
                    target_mean_a,
                    cand_idx,
                    &mut a_row,
                );
                attention.compute_force_and_momentum_row(
                    target_aln,
                    target_landmarks,
                    &m_matrix,
                    &b_prior,
                    &root_seq,
                    target_masks,
                    target_mean_a,
                    cand_idx,
                    &mut f_scratch,
                    &mut c_row,
                );

                let parental_pairs = resolver.resolve_all_reticulation_channels(
                    cand_idx,
                    &c_row,
                    &a_row,
                    l,
                    n,
                    target_landmarks,
                    target_aln,
                    &target_screening.ghost_candidate_indices,
                );

                let mut candidate_tracts = Vec::new();
                for pair in &parental_pairs {
                    let tracts =
                        polisher.deconvolve_candidate(cand_idx, pair, &c_row, l, n, num_channels);
                    candidate_tracts.extend(tracts);
                }

                let mut candidate_verified = Vec::new();
                for tract in candidate_tracts {
                    let local_parental = resolver.resolve_reticulation_channels_localized(
                        cand_idx,
                        tract.u1_discrete,
                        tract.u2_discrete,
                        &c_row,
                        l,
                        target_landmarks,
                        target_aln,
                        Some(tract.home_channel),
                        Some(tract.donor_channel),
                        &target_screening.candidate_recombinant_indices,
                        &target_screening.ghost_candidate_indices,
                    );

                    let mut tract_opt = due_diligence.verify_tract(
                        &tract,
                        &local_parental,
                        target_aln,
                        target_landmarks,
                        &target_screening.rate_burst_indices,
                        &target_screening.candidate_recombinant_indices,
                    );

                    if tract_opt.is_none()
                        && !local_parental.is_ghost_donor
                        && n >= 16
                        && (target_screening.ghost_candidate_indices.contains(&cand_idx)
                            || snp_map_opt.is_some())
                    {
                        let mut ghost_parental = local_parental.clone();
                        ghost_parental.donor_channel = 0;
                        ghost_parental.donor_landmark_idx = None;
                        ghost_parental.is_ghost_donor = true;
                        ghost_parental.donor_name = "Ghost".to_string();
                        ghost_parental.donor_global_idx = None;

                        tract_opt = due_diligence.verify_tract(
                            &tract,
                            &ghost_parental,
                            target_aln,
                            target_landmarks,
                            &target_screening.rate_burst_indices,
                            &target_screening.candidate_recombinant_indices,
                        );
                    }

                    if let Some(event) = tract_opt {
                        if event.is_verified {
                            candidate_verified.push(event);
                        }
                    }
                }

                let deduplicated_events =
                    deduplicate_candidate_events(candidate_verified, target_aln);
                pass_events.extend(deduplicated_events);
            }
            pass_events
        };

        // Pass 1: Initial forward scan on `aln`
        let mut verified_events = run_candidate_scan(&aln, &masks, &mean_a, &landmarks, &screening);
        verified_events =
            due_diligence.adjudicate_reticulation_graph(verified_events, &aln, &screening);

        // Pass 2: Forward Scrubbing (Peeling) Pass
        // Replace detected recombinant tracts with gap character `0` in `aln_scrubbed` and rescan
        // so secondary crossovers (e.g. triple crossovers) and split outer Russian-doll cassettes
        // can be resolved cleanly without interference from the primary tract.
        if !verified_events.is_empty() && l >= 300 {
            let mut aln_scrubbed = aln.clone();
            let mut any_scrubbed = false;

            for cand_idx in 0..n {
                let cand_evs: Vec<&crate::types::RecombinationEvent> = verified_events
                    .iter()
                    .filter(|e| e.candidate_idx == cand_idx)
                    .collect();
                if cand_evs.is_empty() {
                    continue;
                }

                // Check if an inner cassette from one donor clade is bracketed by two tracts from another donor clade
                let mut bracketed_inner_donor: Option<Option<usize>> = None;
                if cand_evs.len() >= 3 {
                    let mut sorted_evs = cand_evs.clone();
                    sorted_evs.sort_by_key(|e| e.u1);
                    for a_i in 0..sorted_evs.len() {
                        for b_i in (a_i + 2)..sorted_evs.len() {
                            let e_left = sorted_evs[a_i];
                            let e_right = sorted_evs[b_i];
                            let same_outer = match (e_left.donor_idx, e_right.donor_idx) {
                                (Some(d1), Some(d2)) => {
                                    d1 == d2
                                        || crate::landmarks::compute_pairwise_distance(&aln, d1, d2)
                                            <= d_clade
                                }
                                (None, None) => e_left.is_ghost_donor && e_right.is_ghost_donor,
                                _ => false,
                            };
                            if same_outer {
                                for m_i in (a_i + 1)..b_i {
                                    let e_mid = sorted_evs[m_i];
                                    let same_mid = match (e_left.donor_idx, e_mid.donor_idx) {
                                        (Some(d1), Some(d2)) => {
                                            d1 == d2
                                                || crate::landmarks::compute_pairwise_distance(
                                                    &aln, d1, d2,
                                                ) <= d_clade
                                        }
                                        (None, None) => {
                                            e_left.is_ghost_donor && e_mid.is_ghost_donor
                                        }
                                        _ => false,
                                    };
                                    if !same_mid {
                                        bracketed_inner_donor = Some(e_mid.donor_idx);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                let target_donor_idx = if let Some(inner_d) = bracketed_inner_donor {
                    inner_d
                } else {
                    let best_ev = cand_evs
                        .iter()
                        .min_by(|a, b| {
                            a.p_fisher
                                .partial_cmp(&b.p_fisher)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then_with(|| {
                                    b.z_phys
                                        .partial_cmp(&a.z_phys)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                })
                        })
                        .unwrap();
                    best_ev.donor_idx
                };

                for ev in &cand_evs {
                    let match_donor = match (ev.donor_idx, target_donor_idx) {
                        (Some(d1), Some(d2)) => {
                            d1 == d2
                                || crate::landmarks::compute_pairwise_distance(&aln, d1, d2)
                                    <= d_clade
                        }
                        (None, None) => true,
                        _ => false,
                    };
                    if match_donor {
                        let s_u = ev.u1.saturating_sub(1).min(l);
                        let e_u = ev.u2.min(l);
                        if e_u > s_u {
                            for t_idx in 0..n {
                                if t_idx == cand_idx
                                    || crate::landmarks::compute_pairwise_distance(
                                        &aln, cand_idx, t_idx,
                                    ) < d_sister
                                {
                                    for u in s_u..e_u {
                                        aln_scrubbed.matrix[t_idx * l + u] = 0;
                                    }
                                    any_scrubbed = true;
                                }
                            }
                        }
                    }
                }
            }

            if any_scrubbed {
                // The scrubbed alignment is a different alignment, so it carries its own masks
                // and its own sequence-wide mean, exactly as the whole-tensor path recomputed
                // both implicitly inside its second call.
                let masks_scrub = compute_attention_masks(&aln_scrubbed, &landmarks);
                let mean_a_scrub = attention.compute_mean_attention(
                    &aln_scrubbed,
                    &landmarks,
                    &m_matrix,
                    &b_prior,
                    &root_seq,
                    &masks_scrub,
                );
                let mut peeled_events = run_candidate_scan(
                    &aln_scrubbed,
                    &masks_scrub,
                    &mean_a_scrub,
                    &landmarks,
                    &screening,
                );
                if !peeled_events.is_empty() {
                    verified_events.append(&mut peeled_events);
                    let mut combined_by_cand = Vec::new();
                    for c_idx in 0..n {
                        let c_evs: Vec<crate::types::RecombinationEvent> = verified_events
                            .iter()
                            .filter(|e| e.candidate_idx == c_idx)
                            .cloned()
                            .collect();
                        if !c_evs.is_empty() {
                            combined_by_cand.extend(deduplicate_candidate_events(c_evs, &aln));
                        }
                    }
                    verified_events = due_diligence.adjudicate_reticulation_graph(
                        combined_by_cand,
                        &aln,
                        &screening,
                    );
                }
            }
        }

        // Pass 3: Backward "Leave-One-In" Verification Pass
        // Re-verify each event in isolation on an alignment where stronger disjoint/parental tracts
        // (and nested inner cassettes from distinct donors) are scrubbed to `0`. This eliminates:
        //   (a) Complementary flank echoes (where the true recombinant tract in the flank made the
        //       pure Home flanks look like a reverse Home<->Donor event), and
        //   (b) Parental mirror false positives (where a pure parent cited a recombinant as Home
        //       because the recombinant carried the parent's DNA inside a disjoint tract).
        if verified_events.len() > 1 {
            verified_events.sort_by(|a, b| {
                a.p_fisher
                    .partial_cmp(&b.p_fisher)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.s_informative.cmp(&a.s_informative))
                    .then_with(|| {
                        b.z_phys
                            .partial_cmp(&a.z_phys)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            });

            let snapshot_events = verified_events.clone();
            let mut loi_kept = Vec::with_capacity(snapshot_events.len());

            for (m_idx, ev_m) in snapshot_events.iter().enumerate() {
                let mut aln_loi = aln.clone();
                let mut modified_loi = false;

                for (k_idx, ev_k) in snapshot_events.iter().enumerate() {
                    if k_idx == m_idx {
                        continue;
                    }
                    let k_is_stronger = k_idx < m_idx;

                    let same_cand_clade = ev_k.candidate_idx == ev_m.candidate_idx
                        || crate::landmarks::compute_pairwise_distance(
                            &aln,
                            ev_k.candidate_idx,
                            ev_m.candidate_idx,
                        ) <= d_clade;

                    let ov_s = ev_k.u1.max(ev_m.u1);
                    let ov_e = ev_k.u2.min(ev_m.u2);
                    let ov_len = if ov_e >= ov_s { ov_e - ov_s + 1 } else { 0 };
                    let m_len = (ev_m.u2.saturating_sub(ev_m.u1) + 1).max(1);
                    let k_len = (ev_k.u2.saturating_sub(ev_k.u1) + 1).max(1);

                    let same_donor = match (ev_k.donor_idx, ev_m.donor_idx) {
                        (Some(d1), Some(d2)) => {
                            d1 == d2
                                || crate::landmarks::compute_pairwise_distance(&aln, d1, d2)
                                    <= d_clade
                        }
                        (None, None) => ev_k.is_ghost_donor && ev_m.is_ghost_donor,
                        _ => false,
                    };

                    // Case A1: Stronger disjoint event on the same candidate clade -> scrub it
                    // so complementary flank echoes lose their false flank reference.
                    let is_disjoint_on_same_cand =
                        k_is_stronger && same_cand_clade && (ov_len as f64) < 0.25 * (m_len as f64);

                    // Case A2: Strictly nested inner cassette from a DIFFERENT donor clade on the
                    // same candidate clade -> scrub the inner cassette when verifying the outer cassette.
                    let is_nested_inner_diff_donor = same_cand_clade
                        && !same_donor
                        && ev_k.u1 > ev_m.u1
                        && ev_k.u2 < ev_m.u2
                        && k_len * 2 <= m_len * 3;

                    // Case B: Stronger overlapping event on a DIFFERENT clade -> scrub it
                    // so pure parents cannot use a recombinant's tract as a false Home reference.
                    let is_overlapping_on_other_cand = k_is_stronger
                        && !same_cand_clade
                        && (ov_len as f64) >= 0.25 * (m_len as f64);

                    if is_disjoint_on_same_cand
                        || is_nested_inner_diff_donor
                        || is_overlapping_on_other_cand
                    {
                        let s_u = ev_k.u1.saturating_sub(1).min(l);
                        let e_u = ev_k.u2.min(l);
                        if e_u > s_u {
                            let cited_home_carries_k = if is_overlapping_on_other_cand
                                && ov_len < m_len
                            {
                                ev_m.home_idx.filter(|&h_idx| {
                                    let in_k_cand_clade =
                                        crate::landmarks::compute_pairwise_distance(
                                            &aln,
                                            ev_k.candidate_idx,
                                            h_idx,
                                        ) <= d_clade;
                                    let not_in_k_donor_clade = ev_k.donor_idx.map_or(true, |kd| {
                                        crate::landmarks::compute_pairwise_distance(&aln, h_idx, kd)
                                            > d_clade
                                    });
                                    if !in_k_cand_clade || !not_in_k_donor_clade {
                                        return false;
                                    }
                                    let k_row = aln.row(ev_k.candidate_idx);
                                    let h_row = aln.row(h_idx);
                                    let mut diffs = 0usize;
                                    let mut valid = 0usize;
                                    for u in s_u..e_u {
                                        let rk = k_row[u];
                                        let rh = h_row[u];
                                        if rk > 0 && rh > 0 {
                                            valid += 1;
                                            if rk != rh {
                                                diffs += 1;
                                            }
                                        }
                                    }
                                    valid > 0 && (diffs as f64) / (valid as f64) <= d_clade
                                })
                            } else {
                                None
                            };

                            for t_idx in 0..n {
                                if t_idx == ev_k.candidate_idx
                                    || Some(t_idx) == cited_home_carries_k
                                    || crate::landmarks::compute_pairwise_distance(
                                        &aln,
                                        ev_k.candidate_idx,
                                        t_idx,
                                    ) < d_sister
                                {
                                    for u in s_u..e_u {
                                        aln_loi.matrix[t_idx * l + u] = 0;
                                    }
                                    modified_loi = true;
                                }
                            }
                        }
                    }
                }

                if !modified_loi {
                    loi_kept.push(ev_m.clone());
                } else {
                    let probe_tract = crate::types::PhysicalTract {
                        candidate_idx: ev_m.candidate_idx,
                        home_channel: 0,
                        donor_channel: 0,
                        is_crossover: ev_m.is_crossover,
                        u1_discrete: ev_m.u1,
                        u2_discrete: ev_m.u2,
                        u1_continuous: ev_m.u1_continuous,
                        u2_continuous: ev_m.u2_continuous,
                        delta_c_max: 0.0,
                        z_phys: ev_m.z_phys,
                        z_crit: 0.0,
                        is_physically_significant: true,
                    };
                    let probe_parental = crate::attribution::TrajectoryParentalPair {
                        candidate_idx: ev_m.candidate_idx,
                        home_channel: 0,
                        donor_channel: 0,
                        home_landmark_idx: None,
                        donor_landmark_idx: None,
                        is_ghost_donor: ev_m.is_ghost_donor,
                        home_name: ev_m.home_name.clone(),
                        donor_name: ev_m.donor_name.clone(),
                        home_global_idx: ev_m.home_idx,
                        donor_global_idx: ev_m.donor_idx,
                        swing_home: 0.0,
                        swing_donor: 0.0,
                    };
                    if let Some(reverified) = due_diligence.verify_tract(
                        &probe_tract,
                        &probe_parental,
                        &aln_loi,
                        &landmarks,
                        &screening.rate_burst_indices,
                        &screening.candidate_recombinant_indices,
                    ) {
                        if reverified.is_verified {
                            let mut updated_ev = ev_m.clone();
                            updated_ev.p_fisher = updated_ev.p_fisher.min(reverified.p_fisher);
                            updated_ev.s_informative =
                                updated_ev.s_informative.max(reverified.s_informative);
                            loi_kept.push(updated_ev);
                        }
                    }
                }
            }

            verified_events = loi_kept;
        }

        // Map verified events from `aln` coordinates (1..=l) to original chromosomal coordinates
        for event in &mut verified_events {
            if let Some(ref snp_map) = snp_map_opt {
                let s_snp = event
                    .u1
                    .saturating_sub(1)
                    .min(snp_map.len().saturating_sub(1));
                let e_snp = event
                    .u2
                    .saturating_sub(1)
                    .min(snp_map.len().saturating_sub(1));
                let orig_u1 = snp_map[s_snp] + 1 + u_offset;
                let orig_u2 = snp_map[e_snp] + 1 + u_offset;
                event.u1 = orig_u1;
                event.u2 = orig_u2;
                event.u1_continuous = orig_u1 as f64;
                event.u2_continuous = orig_u2 as f64;
                event.tract_length = if orig_u2 >= orig_u1 {
                    orig_u2 - orig_u1 + 1
                } else {
                    0
                };
            } else {
                event.u1 += u_offset;
                event.u2 += u_offset;
                event.u1_continuous += u_offset as f64;
                event.u2_continuous += u_offset as f64;
            }
        }

        // Sort events by candidate_idx and u1
        verified_events.sort_by(|a, b| a.candidate_idx.cmp(&b.candidate_idx).then(a.u1.cmp(&b.u1)));

        // Deduplicate adjacent identical calls
        verified_events
            .dedup_by(|a, b| a.isolates[0] == b.isolates[0] && a.u1 == b.u1 && a.u2 == b.u2);

        // Adjudicate reticulation graph (DIR-RHIZ-PAFF-013.1)
        let mut verified_events =
            due_diligence.adjudicate_reticulation_graph(verified_events, &raw_aln, &screening);

        // Enforce single-backbone polarity consistency per candidate taxon
        prune_inverted_backbone_events(&mut verified_events, &raw_aln);

        // Resolve transitive donor & home chains for co-recombinant isolates
        resolve_donor_chains(&mut verified_events, &raw_aln);
        prune_inverted_backbone_events(&mut verified_events, &raw_aln);

        // Align multi-segment double-recombinants whose two donor tracts exceed 50% of the genome
        // with the dominant cohort backbone orientation when they share breakpoints with a multi-isolate clade
        reorient_cohort_complementary_mosaics(
            &mut verified_events,
            &raw_aln,
            &due_diligence,
            &landmarks,
            &screening,
        );
        prune_inverted_backbone_events(&mut verified_events, &raw_aln);

        // Snap parental attributions to the closest non-recombinant sequences in the full cohort
        snap_to_nearest_cohort_parents(&mut verified_events, &raw_aln);

        // Group ancestral events
        verified_events = group_ancestral_events(verified_events, &raw_aln);

        let is_h0 = verified_events.is_empty();

        ScanResult {
            num_taxa: n,
            length: raw_aln.length,
            informative_sites: m_inf,
            is_non_recombinant_h0: is_h0,
            screening,
            events: verified_events,
            run_time_ms: 0.0,
        }
    }
}

fn regional_or_global_same_clade(
    aln: &crate::types::Alignment,
    t1: usize,
    t2: usize,
    u1: usize,
    u2: usize,
    d_clade: f64,
) -> bool {
    if t1 == t2 || crate::landmarks::compute_pairwise_distance(aln, t1, t2) <= d_clade {
        return true;
    }
    let s_idx = u1.saturating_sub(1).min(aln.length);
    let e_idx = u2.min(aln.length);
    if e_idx <= s_idx {
        return false;
    }
    let r1 = aln.row(t1);
    let r2 = aln.row(t2);
    let mut diffs = 0usize;
    let mut valid = 0usize;
    for u in s_idx..e_idx {
        if r1[u] > 0 && r2[u] > 0 {
            valid += 1;
            if r1[u] != r2[u] {
                diffs += 1;
            }
        }
    }
    valid > 0 && (diffs as f64) / (valid as f64) <= d_clade
}

/// Enforces Single-Backbone Polarity Consistency & Un-Recombined Background Home Dominance on each candidate sequence:
/// 1. A single mosaic sequence has one clonal Home backbone and cannot simultaneously have
///    a primary event (Home = A, Donor = B) and a weaker disjoint event with inverted
///    parental clade polarity (Home = B, Donor = A).
/// 2. On the un-recombined background `B(C) = [0, L) \ U_{e in E(C)} [u1(e)-1, u2(e))`,
///    any sampled-donor event on `C` must prefer `Home` over `Donor` (`s_home_bg > s_donor_bg` and `d_bg_home < d_bg_donor`).
fn prune_inverted_backbone_events(
    events: &mut Vec<crate::types::RecombinationEvent>,
    aln: &crate::types::Alignment,
) {
    if events.is_empty() {
        return;
    }
    let (_d_sister, d_clade) = crate::landmarks::compute_scale_invariant_distance_thresholds(aln);
    let mut drop_flags = vec![false; events.len()];
    for i in 0..events.len() {
        if drop_flags[i] {
            continue;
        }
        for j in 0..events.len() {
            if i == j || drop_flags[j] {
                continue;
            }
            if events[i].candidate_idx != events[j].candidate_idx {
                continue;
            }
            // Check if events[j] is strictly stronger than events[i]
            let j_is_stronger = events[j].p_fisher < events[i].p_fisher
                || (events[j].p_fisher == events[i].p_fisher
                    && events[j].z_phys > events[i].z_phys);
            if !j_is_stronger {
                continue;
            }
            if let (Some(hi), Some(di), Some(hj), Some(dj)) = (
                events[i].home_idx,
                events[i].donor_idx,
                events[j].home_idx,
                events[j].donor_idx,
            ) {
                let h_matches_d =
                    regional_or_global_same_clade(aln, hi, dj, events[j].u1, events[j].u2, d_clade);
                let d_matches_h =
                    regional_or_global_same_clade(aln, di, hj, events[i].u1, events[i].u2, d_clade);
                if h_matches_d && d_matches_h {
                    drop_flags[i] = true;
                    break;
                }
            }
        }
    }

    // Build per-candidate un-recombined background intervals from non-dropped events
    let mut cand_tracts: Vec<Vec<(usize, usize)>> = vec![Vec::new(); aln.num_taxa];
    for (i, ev) in events.iter().enumerate() {
        if !drop_flags[i] && ev.candidate_idx < aln.num_taxa {
            let s = ev.u1.saturating_sub(1).min(aln.length);
            let e = ev.u2.min(aln.length);
            if e > s {
                cand_tracts[ev.candidate_idx].push((s, e));
            }
        }
    }

    for (i, ev) in events.iter().enumerate() {
        if drop_flags[i] || ev.is_ghost_donor {
            continue;
        }
        let c_idx = ev.candidate_idx;
        let (Some(hg), Some(dg)) = (ev.home_idx, ev.donor_idx) else {
            continue;
        };
        if c_idx >= aln.num_taxa || hg >= aln.num_taxa || dg >= aln.num_taxa {
            continue;
        }
        let c_row = aln.row(c_idx);
        let h_row = aln.row(hg);
        let d_row = aln.row(dg);
        let mut s_h_bg = 0usize;
        let mut s_d_bg = 0usize;
        let mut m_h_bg = 0usize;
        let mut v_h_bg = 0usize;
        let mut m_d_bg = 0usize;
        let mut v_d_bg = 0usize;
        for u in 0..aln.length {
            let in_any = cand_tracts[c_idx].iter().any(|&(ts, te)| u >= ts && u < te);
            if !in_any {
                let rc = c_row[u];
                let rh = h_row[u];
                let rd = d_row[u];
                if rc > 0 && rh > 0 {
                    v_h_bg += 1;
                    if rc != rh {
                        m_h_bg += 1;
                    }
                }
                if rc > 0 && rd > 0 {
                    v_d_bg += 1;
                    if rc != rd {
                        m_d_bg += 1;
                    }
                }
                if rc > 0 && rh > 0 && rd > 0 && rh != rd {
                    if rc == rh {
                        s_h_bg += 1;
                    } else if rc == rd {
                        s_d_bg += 1;
                    }
                }
            }
        }
        if v_h_bg > 0 && v_d_bg > 0 && (s_h_bg + s_d_bg > 0) {
            let d_bg_h = (m_h_bg as f64) / (v_h_bg as f64);
            let d_bg_d = (m_d_bg as f64) / (v_d_bg as f64);
            if s_h_bg <= s_d_bg || d_bg_h >= d_bg_d {
                drop_flags[i] = true;
            }
        }
    }

    let mut idx = 0usize;
    events.retain(|_| {
        let keep = !drop_flags[idx];
        idx += 1;
        keep
    });
}

/// Checks if two overlapping events with the same (Home, Donor) clade have a 5' or 3'
/// non-overlapping overhang where one candidate carries Donor and the other carries Home
/// (`d_overhang(c1, c2) > d_clade` and 2x2 Fisher exact split `p < 0.05 / L`), indicating
/// two distinct recombination breakpoints (e.g., an extended 3' event or a partial reversion)
/// rather than breakpoint estimation uncertainty.
fn has_divergent_overhang(
    aln: &crate::types::Alignment,
    c1: usize,
    u1_a: usize,
    u2_a: usize,
    c2: usize,
    u1_b: usize,
    u2_b: usize,
    home_idx: Option<usize>,
    donor_idx: Option<usize>,
    d_clade: f64,
) -> bool {
    if c1 == c2 {
        return false;
    }
    let (Some(hg), Some(dg)) = (home_idx, donor_idx) else {
        return false;
    };
    let alpha_split = 0.05 / (aln.length.max(1) as f64);
    let mut overhangs = Vec::new();
    if u1_a != u1_b && u1_a > 1 && u1_b > 1 {
        let s = u1_a.min(u1_b).saturating_sub(1);
        let e = u1_a.max(u1_b).saturating_sub(1).min(aln.length);
        let c_in = if u1_a < u1_b { c1 } else { c2 };
        let c_out = if u1_a < u1_b { c2 } else { c1 };
        overhangs.push((s, e, c_in, c_out));
    }
    if u2_a != u2_b && u2_a < aln.length && u2_b < aln.length {
        let s = u2_a.min(u2_b).min(aln.length);
        let e = u2_a.max(u2_b).min(aln.length);
        let c_in = if u2_a > u2_b { c1 } else { c2 };
        let c_out = if u2_a > u2_b { c2 } else { c1 };
        overhangs.push((s, e, c_in, c_out));
    }

    let h_row = aln.row(hg);
    let d_row = aln.row(dg);
    for (s, e, c_in, c_out) in overhangs {
        if e <= s {
            continue;
        }
        let r_in = aln.row(c_in);
        let r_out = aln.row(c_out);
        let mut diffs_12 = 0usize;
        let mut v_12 = 0usize;
        let mut in_d = 0usize;
        let mut in_h = 0usize;
        let mut out_d = 0usize;
        let mut out_h = 0usize;
        for u in s..e {
            let a = r_in[u];
            let b = r_out[u];
            let rh = h_row[u];
            let rd = d_row[u];
            if a > 0 && b > 0 {
                v_12 += 1;
                if a != b {
                    diffs_12 += 1;
                }
            }
            if a > 0 && b > 0 && rh > 0 && rd > 0 && rh != rd {
                if a == rd {
                    in_d += 1;
                } else if a == rh {
                    in_h += 1;
                }
                if b == rd {
                    out_d += 1;
                } else if b == rh {
                    out_h += 1;
                }
            }
        }
        if v_12 > 0 {
            let d_12 = (diffs_12 as f64) / (v_12 as f64);
            if d_12 > d_clade
                && in_d > in_h
                && out_h > out_d
                && crate::due_diligence::SequenceDueDiligence::fisher_exact_2x2_static(
                    in_d, in_h, out_d, out_h,
                ) < alpha_split
            {
                return true;
            }
        }
    }
    false
}

/// Counts informative sites where `h_row` and `d_row` differ in `[u_a.min(u_b), u_a.max(u_b))`.
fn count_inter_breakpoint_informative(
    h_row: &[u8],
    d_row: &[u8],
    u_a: usize,
    u_b: usize,
    l: usize,
) -> usize {
    let s = u_a.min(u_b).saturating_sub(1).min(l);
    let e = u_a.max(u_b).saturating_sub(1).min(l);
    let mut inf = 0usize;
    for u in s..e {
        let rh = h_row[u];
        let rd = d_row[u];
        if rh > 0 && rd > 0 && rh != rd {
            inf += 1;
        }
    }
    inf
}

/// Aligns multi-segment double-recombinants whose two donor tracts sum to >50% of the genome
/// with the dominant cohort backbone orientation when they share breakpoints with a multi-isolate clade.
fn reorient_cohort_complementary_mosaics(
    events: &mut Vec<crate::types::RecombinationEvent>,
    aln: &crate::types::Alignment,
    due_diligence: &crate::due_diligence::SequenceDueDiligence,
    landmarks: &crate::types::LandmarkSet,
    screening: &crate::types::ScreeningResult,
) {
    if events.len() <= 1 {
        return;
    }
    let (_d_sister, d_clade) = crate::landmarks::compute_scale_invariant_distance_thresholds(aln);
    let l = aln.length;
    let inf_tol = due_diligence.poisson_floor;

    let mut cand_indices: Vec<usize> = events.iter().map(|e| e.candidate_idx).collect();
    cand_indices.sort_unstable();
    cand_indices.dedup();

    let snapshot = events.clone();
    let mut replacements: Vec<(usize, Vec<crate::types::RecombinationEvent>)> = Vec::new();

    for &c_idx in &cand_indices {
        let mut c_evs: Vec<crate::types::RecombinationEvent> = snapshot
            .iter()
            .filter(|e| e.candidate_idx == c_idx && !e.is_ghost_donor)
            .cloned()
            .collect();
        if c_evs.is_empty() {
            continue;
        }
        c_evs.sort_by_key(|e| e.u1);

        let (Some(hc), Some(dc)) = (c_evs[0].home_idx, c_evs[0].donor_idx) else {
            continue;
        };
        let hc_row = aln.row(hc);
        let dc_row = aln.row(dc);

        let mut same_cands = Vec::new();
        let mut inv_cands = Vec::new();
        let mut inv_events = Vec::new();

        for other in &snapshot {
            if other.candidate_idx == c_idx || other.is_ghost_donor {
                continue;
            }
            let (Some(ho), Some(do_idx)) = (other.home_idx, other.donor_idx) else {
                continue;
            };
            let shares_bp = c_evs.iter().any(|ce| {
                count_inter_breakpoint_informative(hc_row, dc_row, ce.u1, other.u1, l) < inf_tol
                    || count_inter_breakpoint_informative(hc_row, dc_row, ce.u2, other.u2, l)
                        < inf_tol
                    || count_inter_breakpoint_informative(hc_row, dc_row, ce.u1, other.u2, l)
                        < inf_tol
                    || count_inter_breakpoint_informative(hc_row, dc_row, ce.u2, other.u1, l)
                        < inf_tol
            });
            if !shares_bp {
                continue;
            }

            let is_same_pol =
                regional_or_global_same_clade(aln, hc, ho, other.u1, other.u2, d_clade)
                    && regional_or_global_same_clade(aln, dc, do_idx, other.u1, other.u2, d_clade);
            if is_same_pol {
                if !same_cands.contains(&other.candidate_idx) {
                    same_cands.push(other.candidate_idx);
                }
            }

            let is_inv_pol =
                regional_or_global_same_clade(aln, dc, ho, other.u1, other.u2, d_clade)
                    && regional_or_global_same_clade(aln, hc, do_idx, other.u1, other.u2, d_clade);
            if is_inv_pol {
                if !inv_cands.contains(&other.candidate_idx) {
                    inv_cands.push(other.candidate_idx);
                }
                inv_events.push(other.clone());
            }
        }

        if inv_cands.len() <= same_cands.len() || inv_cands.len() < 2 {
            continue;
        }

        // Build complementary intervals of `c_evs` on `[1, l]`
        let mut blocked: Vec<(usize, usize)> = Vec::new();
        for ce in &c_evs {
            let s = if count_inter_breakpoint_informative(hc_row, dc_row, 1, ce.u1, l) < inf_tol {
                1
            } else {
                ce.u1
            };
            let e = if count_inter_breakpoint_informative(hc_row, dc_row, ce.u2, l + 1, l) < inf_tol
            {
                l
            } else {
                ce.u2
            };
            if let Some(last) = blocked.last_mut() {
                if s <= last.1 + 1 {
                    last.1 = last.1.max(e);
                    continue;
                }
            }
            blocked.push((s, e));
        }

        let mut comp_intervals: Vec<(usize, usize)> = Vec::new();
        let mut cursor = 1usize;
        for (bs, be) in &blocked {
            if *bs > cursor {
                comp_intervals.push((cursor, bs - 1));
            }
            cursor = be + 1;
        }
        if cursor <= l {
            comp_intervals.push((cursor, l));
        }

        if comp_intervals.is_empty() {
            continue;
        }

        let new_home_idx = dc;
        let new_home_name = aln.taxa[new_home_idx].clone();
        let mut verified_comps = Vec::new();

        for (r_idx, &(cu1, cu2)) in comp_intervals.iter().enumerate() {
            // Choose the cohort donor from `inv_events` overlapping `[cu1, cu2]` and sharing a breakpoint
            let mut best_donor_idx = hc;
            for ie in &inv_events {
                let ov_s = cu1.max(ie.u1);
                let ov_e = cu2.min(ie.u2);
                if ov_e >= ov_s {
                    let shares_edge =
                        count_inter_breakpoint_informative(hc_row, dc_row, cu1, ie.u1, l) < inf_tol
                            || count_inter_breakpoint_informative(hc_row, dc_row, cu2, ie.u2, l)
                                < inf_tol;
                    if shares_edge {
                        if let Some(id_idx) = ie.donor_idx {
                            best_donor_idx = id_idx;
                            break;
                        }
                    }
                }
            }
            let new_donor_name = aln.taxa[best_donor_idx].clone();

            // Mask the OTHER complementary intervals on `c_idx` so multi-tract flanks are clean
            let mut aln_comp = aln.clone();
            for (o_idx, &(ou1, ou2)) in comp_intervals.iter().enumerate() {
                if o_idx != r_idx {
                    for u in ou1.saturating_sub(1)..ou2.min(l) {
                        aln_comp.matrix[c_idx * l + u] = 0;
                    }
                }
            }

            let probe_tract = crate::types::PhysicalTract {
                candidate_idx: c_idx,
                home_channel: 0,
                donor_channel: 0,
                is_crossover: (cu1 == 1) || (cu2 == l),
                u1_discrete: cu1,
                u2_discrete: cu2,
                u1_continuous: cu1 as f64,
                u2_continuous: cu2 as f64,
                delta_c_max: 0.0,
                z_phys: c_evs[0].z_phys,
                z_crit: 0.0,
                is_physically_significant: true,
            };
            let probe_parental = crate::attribution::TrajectoryParentalPair {
                candidate_idx: c_idx,
                home_channel: 0,
                donor_channel: 0,
                home_landmark_idx: None,
                donor_landmark_idx: None,
                is_ghost_donor: false,
                home_name: new_home_name.clone(),
                donor_name: new_donor_name,
                home_global_idx: Some(new_home_idx),
                donor_global_idx: Some(best_donor_idx),
                swing_home: 0.0,
                swing_donor: 0.0,
            };

            if let Some(ev) = due_diligence.verify_tract(
                &probe_tract,
                &probe_parental,
                &aln_comp,
                landmarks,
                &screening.rate_burst_indices,
                &screening.candidate_recombinant_indices,
            ) {
                if ev.is_verified {
                    verified_comps.push(ev);
                }
            }
        }

        if !verified_comps.is_empty() {
            replacements.push((c_idx, verified_comps));
        }
    }

    for (c_idx, mut new_evs) in replacements {
        events.retain(|e| e.candidate_idx != c_idx);
        events.append(&mut new_evs);
    }
}

/// Resolves transitive donor chains and co-recombinant lineages across overlapping tracts.
/// When sequence A is detected with donor B, but B is itself a co-recombinant with donor C
/// sharing the same Home lineage over an overlapping tract, A inherits donor C from B.
/// Similarly, if sequence A cites B as Home, but B is a co-recombinant sharing the same donor
/// over an overlapping tract with Home C, A inherits Home C from B; or if A's cassette is nested
/// inside B's introgressed tract from Donor D, A's local flanking Home in B is resolved to D.
fn resolve_donor_chains(
    events: &mut [crate::types::RecombinationEvent],
    aln: &crate::types::Alignment,
) {
    if events.len() <= 1 {
        return;
    }

    let max_passes = events.len().min(10);
    for _ in 0..max_passes {
        let mut changed = false;

        for i in 0..events.len() {
            let e1_home = events[i].home_name.clone();
            let e1_donor = events[i].donor_name.clone();
            let e1_cand = events[i].candidate_name.clone();
            let e1_u1 = events[i].u1;
            let e1_u2 = events[i].u2;
            let e1_len = e1_u2.saturating_sub(e1_u1) + 1;

            // 1. Resolve Donor Chain: e1 -> e2 (e1 cited e2 as Donor)
            if !e1_donor.is_empty() && e1_donor != "Ghost" {
                let mut resolved_donor_idx = None;
                let mut resolved_donor_name = None;
                let mut resolved_is_ghost = false;
                let mut any_overlapping_donor_event = false;

                for (j, e2) in events.iter().enumerate() {
                    if i == j {
                        continue;
                    }

                    let e2_is_donor_isolate =
                        e2.candidate_name == e1_donor || e2.isolates.contains(&e1_donor);
                    if !e2_is_donor_isolate {
                        continue;
                    }

                    // Check tract overlap: >= 50% of the minimum length
                    let ov_start = e1_u1.max(e2.u1);
                    let ov_end = e1_u2.min(e2.u2);
                    if ov_end < ov_start {
                        continue;
                    }
                    let ov_len = ov_end - ov_start + 1;
                    let min_len = e1_len.min(e2.u2.saturating_sub(e2.u1) + 1);
                    if (ov_len as f64) < 0.25 * (min_len as f64) {
                        continue;
                    }
                    any_overlapping_donor_event = true;
                    if (ov_len as f64) < 0.5 * (min_len as f64) {
                        continue;
                    }

                    // Compatible Home lineage (or e1's tract is >= 80% inside e2's tract)
                    let e1_inside_e2 = (ov_len as f64) >= 0.80 * (e1_len as f64);
                    if e1_home != e2.home_name && !e1_inside_e2 {
                        continue;
                    }

                    // e2's donor must not be e1 or e1's Home (prevent cyclic attribution)
                    if e2.donor_name.is_empty()
                        || e2.donor_name == e1_cand
                        || e2.donor_name == e1_home
                        || events[i].isolates.contains(&e2.donor_name)
                    {
                        continue;
                    }

                    resolved_donor_name = Some(e2.donor_name.clone());
                    resolved_donor_idx = e2.donor_idx;
                    resolved_is_ghost = e2.is_ghost_donor;
                    break;
                }

                // Subcase C for Donor: e1 cited a recombinant `e1_donor` at a locus disjoint from
                // `e1_donor`'s recombinant tracts (i.e. inside `e1_donor`'s un-recombined Home backbone).
                // Resolve `e1_donor` to its pure non-recombinant Home parent `e2.home_name` if `e2.home_name`
                // is in the same clade as `e1_donor` at `e1`'s locus.
                if resolved_donor_name.is_none() && !any_overlapping_donor_event {
                    let (_d_sister, d_clade) =
                        crate::landmarks::compute_scale_invariant_distance_thresholds(aln);
                    for (j, e2) in events.iter().enumerate() {
                        if i == j {
                            continue;
                        }
                        let e2_is_donor_isolate =
                            e2.candidate_name == e1_donor || e2.isolates.contains(&e1_donor);
                        if !e2_is_donor_isolate {
                            continue;
                        }
                        let target_donor_is_recombinant = events.iter().any(|ev| {
                            ev.candidate_name == e2.home_name || ev.isolates.contains(&e2.home_name)
                        });
                        if e2.home_name.is_empty()
                            || e2.home_name == e1_cand
                            || e2.home_name == e1_home
                            || events[i].isolates.contains(&e2.home_name)
                            || target_donor_is_recombinant
                        {
                            continue;
                        }
                        if let (Some(d1_idx), Some(h2_idx)) = (events[i].donor_idx, e2.home_idx) {
                            if regional_or_global_same_clade(
                                aln, d1_idx, h2_idx, e1_u1, e1_u2, d_clade,
                            ) {
                                resolved_donor_name = Some(e2.home_name.clone());
                                resolved_donor_idx = e2.home_idx;
                                resolved_is_ghost = false;
                                break;
                            }
                        }
                    }
                }

                if let Some(new_donor) = resolved_donor_name {
                    if new_donor != events[i].donor_name {
                        if let Some((s_donor, d_tr_h, d_fl_h, d_tr_d, d_fl_d)) =
                            verify_carpet_polarity_for_pair(
                                aln,
                                events[i].candidate_idx,
                                events[i].home_idx,
                                resolved_donor_idx,
                                resolved_is_ghost,
                                events[i].u1,
                                events[i].u2,
                            )
                        {
                            events[i].donor_name = new_donor;
                            events[i].donor_idx = resolved_donor_idx;
                            events[i].is_ghost_donor = resolved_is_ghost;
                            events[i].s_informative = s_donor;
                            events[i].d_tract_home = d_tr_h;
                            events[i].d_flank_home = d_fl_h;
                            events[i].d_tract_donor = d_tr_d;
                            events[i].d_flank_donor = d_fl_d;
                            changed = true;
                        }
                    }
                }
            }

            // 2. Resolve Home Chain: e1 -> e2 (e1 cited e2 as Home)
            if !e1_home.is_empty() {
                let mut resolved_home_idx = None;
                let mut resolved_home_name = None;
                let mut any_overlapping_home_event = false;

                for (j, e2) in events.iter().enumerate() {
                    if i == j {
                        continue;
                    }

                    let e2_is_home_isolate =
                        e2.candidate_name == e1_home || e2.isolates.contains(&e1_home);
                    if !e2_is_home_isolate {
                        continue;
                    }

                    // Check tract overlap
                    let ov_start = e1_u1.max(e2.u1);
                    let ov_end = e1_u2.min(e2.u2);
                    if ov_end < ov_start {
                        continue;
                    }
                    let ov_len = ov_end - ov_start + 1;
                    let e2_len = e2.u2.saturating_sub(e2.u1) + 1;
                    let min_len = e1_len.min(e2_len);
                    if (ov_len as f64) < 0.25 * (min_len as f64) {
                        continue;
                    }
                    any_overlapping_home_event = true;
                    if (ov_len as f64) < 0.5 * (min_len as f64) {
                        continue;
                    }

                    if events[i].donor_name == e2.donor_name
                        || e2.is_ghost_donor
                        || events[i].is_ghost_donor
                    {
                        // Subcase A: Co-recombinant sisters sharing an overlapping event -> inherit e2's Home
                        if e2.home_name.is_empty()
                            || e2.home_name == e1_cand
                            || e2.home_name == events[i].donor_name
                            || events[i].isolates.contains(&e2.home_name)
                        {
                            continue;
                        }
                        resolved_home_name = Some(e2.home_name.clone());
                        resolved_home_idx = e2.home_idx;
                        break;
                    } else if (ov_len as f64) >= 0.80 * (e1_len as f64)
                        && e1_len * 2 <= e2_len
                        && !e2.is_ghost_donor
                        && e2.donor_idx.is_some()
                    {
                        // Subcase B: e1 is a nested cassette strictly inside e2's larger introgressed tract.
                        // Around e1's breakpoints, e2 is physically carrying e2.donor_name!
                        if e2.donor_name.is_empty()
                            || e2.donor_name == e1_cand
                            || e2.donor_name == events[i].donor_name
                            || events[i].isolates.contains(&e2.donor_name)
                        {
                            continue;
                        }
                        resolved_home_name = Some(e2.donor_name.clone());
                        resolved_home_idx = e2.donor_idx;
                        break;
                    }
                }

                // Subcase C: e1 cited a recombinant `e1_home` at a locus disjoint from `e1_home`'s
                // recombinant tracts (i.e. inside `e1_home`'s un-recombined Home backbone).
                // Resolve `e1_home` to its pure non-recombinant Home parent `e2.home_name` if `e2.home_name`
                // is in the same clade as `e1_home` at `e1`'s locus.
                if resolved_home_name.is_none() && !any_overlapping_home_event {
                    let (_d_sister, d_clade) =
                        crate::landmarks::compute_scale_invariant_distance_thresholds(aln);
                    for (j, e2) in events.iter().enumerate() {
                        if i == j {
                            continue;
                        }
                        let e2_is_home_isolate =
                            e2.candidate_name == e1_home || e2.isolates.contains(&e1_home);
                        if !e2_is_home_isolate {
                            continue;
                        }
                        let target_home_is_recombinant = events.iter().any(|ev| {
                            ev.candidate_name == e2.home_name || ev.isolates.contains(&e2.home_name)
                        });
                        if e2.home_name.is_empty()
                            || e2.home_name == e1_cand
                            || e2.home_name == events[i].donor_name
                            || events[i].isolates.contains(&e2.home_name)
                            || target_home_is_recombinant
                        {
                            continue;
                        }
                        if let (Some(h1_idx), Some(h2_idx)) = (events[i].home_idx, e2.home_idx) {
                            if regional_or_global_same_clade(
                                aln, h1_idx, h2_idx, e1_u1, e1_u2, d_clade,
                            ) {
                                resolved_home_name = Some(e2.home_name.clone());
                                resolved_home_idx = e2.home_idx;
                                break;
                            }
                        }
                    }
                }

                if let Some(new_home) = resolved_home_name {
                    if new_home != events[i].home_name {
                        if let Some((s_donor, d_tr_h, d_fl_h, d_tr_d, d_fl_d)) =
                            verify_carpet_polarity_for_pair(
                                aln,
                                events[i].candidate_idx,
                                resolved_home_idx,
                                events[i].donor_idx,
                                events[i].is_ghost_donor,
                                events[i].u1,
                                events[i].u2,
                            )
                        {
                            events[i].home_name = new_home;
                            events[i].home_idx = resolved_home_idx;
                            events[i].s_informative = s_donor;
                            events[i].d_tract_home = d_tr_h;
                            events[i].d_flank_home = d_fl_h;
                            events[i].d_tract_donor = d_tr_d;
                            events[i].d_flank_donor = d_fl_d;
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
}

/// Verifies that a proposed `(home_idx, donor_idx)` pair on `[u1, u2]` obeys the Carpet Polarity
/// Invariant (`s_donor > s_home_tract`, `d_tract_donor < d_tract_home`, and `d_tract_donor < d_flank_donor`
/// for sampled donors; `d_tract_home > d_flank_home` for ghost donors) before `resolve_donor_chains`
/// applies any post-hoc chain substitution.
fn verify_carpet_polarity_for_pair(
    aln: &crate::types::Alignment,
    cand_idx: usize,
    home_idx_opt: Option<usize>,
    donor_idx_opt: Option<usize>,
    is_ghost_donor: bool,
    u1: usize,
    u2: usize,
) -> Option<(usize, f64, f64, f64, f64)> {
    let hg = home_idx_opt?;
    if cand_idx >= aln.num_taxa || hg >= aln.num_taxa {
        return None;
    }
    let cand_row = aln.row(cand_idx);
    let home_row = aln.row(hg);
    let s_idx = u1.saturating_sub(1);
    let e_idx = u2.min(aln.length);

    let mut d_tr_h = 0usize;
    let mut v_tr_h = 0usize;
    let mut d_fl_h = 0usize;
    let mut v_fl_h = 0usize;
    for u in 0..aln.length {
        let rc = cand_row[u];
        let rh = home_row[u];
        if rc > 0 && rh > 0 {
            if u >= s_idx && u < e_idx {
                v_tr_h += 1;
                if rc != rh {
                    d_tr_h += 1;
                }
            } else {
                v_fl_h += 1;
                if rc != rh {
                    d_fl_h += 1;
                }
            }
        }
    }
    if v_tr_h == 0 {
        return None;
    }
    let d_tract_home = (d_tr_h as f64) / (v_tr_h as f64);
    let d_flank_home = if v_fl_h > 0 {
        (d_fl_h as f64) / (v_fl_h as f64)
    } else {
        0.0
    };

    if is_ghost_donor || donor_idx_opt.is_none() {
        if d_tract_home <= d_flank_home || d_tr_h < 4 {
            return None;
        }
        return Some((d_tr_h, d_tract_home, d_flank_home, 0.0, 0.0));
    }

    let dg = donor_idx_opt?;
    if dg >= aln.num_taxa {
        return None;
    }
    let donor_row = aln.row(dg);

    let mut s_donor = 0usize;
    let mut s_home_tract = 0usize;
    let mut s_donor_flank = 0usize;
    let mut s_home_flank = 0usize;
    let mut d_tr_d = 0usize;
    let mut v_tr_d = 0usize;
    let mut d_fl_d = 0usize;
    let mut v_fl_d = 0usize;

    for u in 0..aln.length {
        let rc = cand_row[u];
        let rh = home_row[u];
        let rd = donor_row[u];
        if u >= s_idx && u < e_idx {
            if rc > 0 && rd > 0 {
                v_tr_d += 1;
                if rc != rd {
                    d_tr_d += 1;
                }
            }
            if rc > 0 && rh > 0 && rd > 0 && rh != rd {
                if rc == rd {
                    s_donor += 1;
                } else if rc == rh {
                    s_home_tract += 1;
                }
            }
        } else {
            if rc > 0 && rd > 0 {
                v_fl_d += 1;
                if rc != rd {
                    d_fl_d += 1;
                }
            }
            if rc > 0 && rh > 0 && rd > 0 && rh != rd {
                if rc == rd {
                    s_donor_flank += 1;
                } else if rc == rh {
                    s_home_flank += 1;
                }
            }
        }
    }

    if v_tr_d == 0 {
        return None;
    }
    let d_tract_donor = (d_tr_d as f64) / (v_tr_d as f64);
    let d_flank_donor = if v_fl_d > 0 {
        (d_fl_d as f64) / (v_fl_d as f64)
    } else {
        d_tract_donor
    };

    // Synthetic monomorphic unit-test alignment (all sequences identical):
    if d_tr_h == 0 && d_fl_h == 0 && d_tr_d == 0 && d_fl_d == 0 {
        return Some((0, 0.0, 0.0, 0.0, 0.0));
    }

    if s_donor <= s_home_tract
        || d_tract_donor >= d_tract_home
        || (v_fl_d > 0 && d_tract_donor >= d_flank_donor)
        || (v_fl_h > 0 && d_tract_home <= d_flank_home)
        || ((s_donor_flank > 0 || s_home_flank > 0)
            && (s_home_flank == 0 || s_home_tract * s_donor_flank >= s_donor * s_home_flank))
    {
        return None;
    }

    Some((
        s_donor,
        d_tract_home,
        d_flank_home,
        d_tract_donor,
        d_flank_donor,
    ))
}

/// Snaps `(home_idx, donor_idx)` of each verified event to the closest pure non-recombinant
/// sequence in the full cohort (`0..aln.num_taxa`) when Farthest-Point Landmark Sampling (`K < N`)
/// or pre-verification screening excluded a significantly closer parental sequence on the carpet.
fn snap_to_nearest_cohort_parents(
    events: &mut [crate::types::RecombinationEvent],
    aln: &crate::types::Alignment,
) {
    if events.is_empty() || aln.num_taxa < 4 || aln.length == 0 {
        return;
    }
    let (d_sister, _d_clade) = crate::landmarks::compute_scale_invariant_distance_thresholds(aln);
    let mut is_recombinant = vec![false; aln.num_taxa];
    for ev in events.iter() {
        if ev.candidate_idx < aln.num_taxa {
            is_recombinant[ev.candidate_idx] = true;
        }
        for iso in &ev.isolates {
            if let Some(idx) = aln.taxa.iter().position(|t| t == iso) {
                is_recombinant[idx] = true;
            }
        }
    }

    for ev in events.iter_mut() {
        let c_idx = ev.candidate_idx;
        if c_idx >= aln.num_taxa {
            continue;
        }
        let s_idx = ev.u1.saturating_sub(1).min(aln.length);
        let e_idx = ev.u2.min(aln.length);
        if e_idx <= s_idx {
            continue;
        }
        let c_row = aln.row(c_idx);

        // 1. Snap Donor (for sampled-donor events) to the closest pure non-recombinant sequence inside [s_idx, e_idx)
        if !ev.is_ghost_donor {
            if let (Some(hg), Some(dg)) = (ev.home_idx, ev.donor_idx) {
                if hg < aln.num_taxa && dg < aln.num_taxa {
                    let d_row = aln.row(dg);
                    let mut cur_diffs = 0usize;
                    let mut cur_valid = 0usize;
                    for u in s_idx..e_idx {
                        if c_row[u] > 0 && d_row[u] > 0 {
                            cur_valid += 1;
                            if c_row[u] != d_row[u] {
                                cur_diffs += 1;
                            }
                        }
                    }
                    let cur_d_tr = if cur_valid > 0 {
                        (cur_diffs as f64) / (cur_valid as f64)
                    } else {
                        1.0
                    };
                    let dg_is_rec = is_recombinant[dg];

                    if cur_diffs >= 5 || dg_is_rec {
                        let mut best_d = dg;
                        let mut best_d_tr = if dg_is_rec { f64::INFINITY } else { cur_d_tr };
                        let mut best_diffs = if dg_is_rec { usize::MAX } else { cur_diffs };
                        let mut best_metrics: Option<(usize, f64, f64, f64, f64)> = None;

                        for t in 0..aln.num_taxa {
                            if t == c_idx
                                || t == hg
                                || t == dg
                                || is_recombinant[t]
                                || crate::landmarks::is_true_sister_pair(aln, c_idx, t, d_sister)
                            {
                                continue;
                            }
                            let t_row = aln.row(t);
                            let h_row = aln.row(hg);
                            let mut t_diffs = 0usize;
                            let mut t_valid = 0usize;
                            let mut th_diffs = 0usize;
                            let mut th_valid = 0usize;
                            for u in s_idx..e_idx {
                                if c_row[u] > 0 && t_row[u] > 0 {
                                    t_valid += 1;
                                    if c_row[u] != t_row[u] {
                                        t_diffs += 1;
                                    }
                                }
                                if h_row[u] > 0 && t_row[u] > 0 {
                                    th_valid += 1;
                                    if h_row[u] != t_row[u] {
                                        th_diffs += 1;
                                    }
                                }
                            }
                            if t_valid == 0 || th_valid == 0 {
                                continue;
                            }
                            let t_d_tr = (t_diffs as f64) / (t_valid as f64);
                            let th_d_tr = (th_diffs as f64) / (th_valid as f64);
                            if th_d_tr <= t_d_tr {
                                continue;
                            }
                            let improves = if dg_is_rec {
                                t_d_tr <= cur_d_tr + 0.005
                            } else {
                                t_d_tr + 0.005 < 0.70 * cur_d_tr && cur_diffs >= t_diffs + 5
                            };
                            if improves
                                && (t_d_tr < best_d_tr
                                    || (t_d_tr == best_d_tr && t_diffs < best_diffs))
                                && verify_carpet_polarity_for_pair(
                                    aln,
                                    t,
                                    Some(hg),
                                    Some(dg),
                                    false,
                                    ev.u1,
                                    ev.u2,
                                )
                                .is_none()
                            {
                                if let Some(m) = verify_carpet_polarity_for_pair(
                                    aln,
                                    c_idx,
                                    Some(hg),
                                    Some(t),
                                    false,
                                    ev.u1,
                                    ev.u2,
                                ) {
                                    if m.0 >= 4 && (dg_is_rec || m.0 * 3 >= ev.s_informative) {
                                        best_d = t;
                                        best_d_tr = t_d_tr;
                                        best_diffs = t_diffs;
                                        best_metrics = Some(m);
                                    }
                                }
                            }
                        }

                        if best_d != dg {
                            if let Some((s_d, d_tr_h, d_fl_h, d_tr_d, d_fl_d)) = best_metrics {
                                ev.donor_idx = Some(best_d);
                                ev.donor_name = aln.taxa[best_d].clone();
                                ev.s_informative = s_d;
                                ev.d_tract_home = d_tr_h;
                                ev.d_flank_home = d_fl_h;
                                ev.d_tract_donor = d_tr_d;
                                ev.d_flank_donor = d_fl_d;
                            }
                        }
                    }
                }
            }
        }

        // 2. Snap Home to the closest pure non-recombinant sequence outside [s_idx, e_idx)
        if let Some(hg) = ev.home_idx {
            if hg < aln.num_taxa {
                let h_row = aln.row(hg);
                let mut cur_diffs = 0usize;
                let mut cur_valid = 0usize;
                for u in 0..aln.length {
                    if (u < s_idx || u >= e_idx) && c_row[u] > 0 && h_row[u] > 0 {
                        cur_valid += 1;
                        if c_row[u] != h_row[u] {
                            cur_diffs += 1;
                        }
                    }
                }
                let cur_d_out = if cur_valid > 0 {
                    (cur_diffs as f64) / (cur_valid as f64)
                } else {
                    1.0
                };
                let hg_is_rec = is_recombinant[hg];

                if cur_diffs >= 5 || hg_is_rec {
                    let mut best_h = hg;
                    let mut best_d_out = if hg_is_rec { f64::INFINITY } else { cur_d_out };
                    let mut best_diffs = if hg_is_rec { usize::MAX } else { cur_diffs };
                    let mut best_metrics: Option<(usize, f64, f64, f64, f64)> = None;

                    for t in 0..aln.num_taxa {
                        if t == c_idx
                            || t == hg
                            || Some(t) == ev.donor_idx
                            || is_recombinant[t]
                            || crate::landmarks::is_true_sister_pair(aln, c_idx, t, d_sister)
                        {
                            continue;
                        }
                        let t_row = aln.row(t);
                        let mut t_diffs = 0usize;
                        let mut t_valid = 0usize;
                        for u in 0..aln.length {
                            if (u < s_idx || u >= e_idx) && c_row[u] > 0 && t_row[u] > 0 {
                                t_valid += 1;
                                if c_row[u] != t_row[u] {
                                    t_diffs += 1;
                                }
                            }
                        }
                        if t_valid == 0 {
                            continue;
                        }
                        let t_d_out = (t_diffs as f64) / (t_valid as f64);
                        let improves = if hg_is_rec {
                            t_d_out <= cur_d_out + 0.005
                        } else {
                            t_d_out + 0.005 < 0.70 * cur_d_out && cur_diffs >= t_diffs + 5
                        };
                        if improves
                            && (t_d_out < best_d_out
                                || (t_d_out == best_d_out && t_diffs < best_diffs))
                            && verify_carpet_polarity_for_pair(
                                aln,
                                t,
                                Some(hg),
                                ev.donor_idx,
                                ev.is_ghost_donor,
                                ev.u1,
                                ev.u2,
                            )
                            .is_none()
                        {
                            if let Some(m) = verify_carpet_polarity_for_pair(
                                aln,
                                c_idx,
                                Some(t),
                                ev.donor_idx,
                                ev.is_ghost_donor,
                                ev.u1,
                                ev.u2,
                            ) {
                                if m.0 >= 4 {
                                    best_h = t;
                                    best_d_out = t_d_out;
                                    best_diffs = t_diffs;
                                    best_metrics = Some(m);
                                }
                            }
                        }
                    }

                    if best_h != hg {
                        if let Some((s_d, d_tr_h, d_fl_h, d_tr_d, d_fl_d)) = best_metrics {
                            ev.home_idx = Some(best_h);
                            ev.home_name = aln.taxa[best_h].clone();
                            ev.s_informative = s_d;
                            ev.d_tract_home = d_tr_h;
                            ev.d_flank_home = d_fl_h;
                            ev.d_tract_donor = d_tr_d;
                            ev.d_flank_donor = d_fl_d;
                        }
                    }
                }
            }
        }
    }
}

/// Prunes secondary events that remain unresolved recombinant-parent mirrors after chain resolution
/// and full-cohort snapping:
/// 1. Single Pure Taxon Dominates Both Inside and Outside: if a single pure non-recombinant taxon `t`
///    is significantly closer (`< 0.70x` and `>= 5` fewer diffs) to `c` than `Donor` inside `[u1, u2]`
///    AND significantly closer (`< 0.70x` and `>= 10` fewer diffs) to `c` than `Home` outside `[u1, u2]`,
///    then `c` is a clonal relative of `t` across both regions rather than a recombinant between `Home` and `Donor`.
/// 2. Unresolved Overlapping Recombinant Parent: if a weaker event still cites an isolate of a strictly
///    stronger overlapping event (`overlap > 100` and `>= 20%` of min length) as its Home or Donor because
///    no pure non-recombinant parent in the cohort satisfies `verify_carpet_polarity_for_pair`, drop the weaker mirror.
fn prune_unresolved_recombinant_parent_mirrors(
    events: &mut Vec<crate::types::RecombinationEvent>,
    aln: &crate::types::Alignment,
) {
    if events.is_empty() || aln.num_taxa < 4 || aln.length == 0 {
        return;
    }

    for _iter in 0..4 {
        resolve_donor_chains(events, aln);
        snap_to_nearest_cohort_parents(events, aln);
        prune_inverted_backbone_events(events, aln);
        if events.is_empty() {
            break;
        }

        events.sort_by(|a, b| {
            a.p_fisher
                .partial_cmp(&b.p_fisher)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.z_phys
                        .partial_cmp(&a.z_phys)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        let mut is_recombinant = vec![false; aln.num_taxa];
        for ev in events.iter() {
            if ev.candidate_idx < aln.num_taxa {
                is_recombinant[ev.candidate_idx] = true;
            }
            for iso in &ev.isolates {
                if let Some(idx) = aln.taxa.iter().position(|t| t == iso) {
                    is_recombinant[idx] = true;
                }
            }
        }

        let mut drop_flags = vec![false; events.len()];
        for i in 0..events.len() {
            if drop_flags[i] {
                continue;
            }
            let c_idx = events[i].candidate_idx;
            let s_idx = events[i].u1.saturating_sub(1).min(aln.length);
            let e_idx = events[i].u2.min(aln.length);
            if c_idx < aln.num_taxa && e_idx > s_idx && !events[i].is_ghost_donor {
                if let (Some(hg), Some(dg)) = (events[i].home_idx, events[i].donor_idx) {
                    if hg < aln.num_taxa && dg < aln.num_taxa {
                        let c_row = aln.row(c_idx);
                        let h_row = aln.row(hg);
                        let d_row = aln.row(dg);
                        let mut d_diffs_in = 0usize;
                        let mut d_val_in = 0usize;
                        let mut h_diffs_out = 0usize;
                        let mut h_val_out = 0usize;
                        for u in 0..aln.length {
                            if u >= s_idx && u < e_idx {
                                if c_row[u] > 0 && d_row[u] > 0 {
                                    d_val_in += 1;
                                    if c_row[u] != d_row[u] {
                                        d_diffs_in += 1;
                                    }
                                }
                            } else if c_row[u] > 0 && h_row[u] > 0 {
                                h_val_out += 1;
                                if c_row[u] != h_row[u] {
                                    h_diffs_out += 1;
                                }
                            }
                        }
                        if d_val_in > 0 && h_val_out > 0 && h_diffs_out >= 10 {
                            let d_dist_in = (d_diffs_in as f64) / (d_val_in as f64);
                            let h_dist_out = (h_diffs_out as f64) / (h_val_out as f64);
                            let mut dominated_by_single_pure = false;
                            for t in 0..aln.num_taxa {
                                if t == c_idx || t == hg || t == dg || is_recombinant[t] {
                                    continue;
                                }
                                let t_row = aln.row(t);
                                let mut t_diffs_in = 0usize;
                                let mut t_val_in = 0usize;
                                let mut t_diffs_out = 0usize;
                                let mut t_val_out = 0usize;
                                for u in 0..aln.length {
                                    if u >= s_idx && u < e_idx {
                                        if c_row[u] > 0 && t_row[u] > 0 {
                                            t_val_in += 1;
                                            if c_row[u] != t_row[u] {
                                                t_diffs_in += 1;
                                            }
                                        }
                                    } else if c_row[u] > 0 && t_row[u] > 0 {
                                        t_val_out += 1;
                                        if c_row[u] != t_row[u] {
                                            t_diffs_out += 1;
                                        }
                                    }
                                }
                                if (t_val_in as f64) > 0.8 * (d_val_in as f64)
                                    && (t_val_out as f64) > 0.8 * (h_val_out as f64)
                                {
                                    let t_dist_in = (t_diffs_in as f64) / (t_val_in as f64);
                                    let t_dist_out = (t_diffs_out as f64) / (t_val_out as f64);
                                    let closer_home_outside = t_diffs_out + 10 <= h_diffs_out
                                        && t_dist_out < 0.70 * h_dist_out;
                                    let closer_or_equal_in = (t_diffs_in <= d_diffs_in
                                        && t_dist_in <= d_dist_in + 1e-9)
                                        || (d_diffs_in >= 5
                                            && t_diffs_in + 5 <= d_diffs_in
                                            && t_dist_in < 0.70 * d_dist_in);
                                    let no_mosaic_vs_t = t_dist_in <= t_dist_out + 0.005
                                        || verify_carpet_polarity_for_pair(
                                            aln,
                                            c_idx,
                                            Some(t),
                                            Some(dg),
                                            false,
                                            events[i].u1,
                                            events[i].u2,
                                        )
                                        .is_none();
                                    if closer_home_outside && closer_or_equal_in && no_mosaic_vs_t {
                                        dominated_by_single_pure = true;
                                        break;
                                    }
                                }
                            }
                            if dominated_by_single_pure {
                                drop_flags[i] = true;
                                continue;
                            }
                        }
                    }
                }
            }

            // Check B: Unresolved overlapping recombinant parent on weaker event
            for j in 0..i {
                if drop_flags[j] {
                    continue;
                }
                let ov_s = events[i].u1.max(events[j].u1);
                let ov_e = events[i].u2.min(events[j].u2);
                if ov_e >= ov_s {
                    let ov = ov_e - ov_s + 1;
                    let min_len = (events[i].u2.saturating_sub(events[i].u1) + 1)
                        .min(events[j].u2.saturating_sub(events[j].u1) + 1)
                        .max(1);
                    if ov > 100 && (ov as f64) >= 0.20 * (min_len as f64) {
                        let cites_j_as_home = events[j].isolates.contains(&events[i].home_name)
                            || events[j].candidate_name == events[i].home_name;
                        let cites_j_as_donor = !events[i].is_ghost_donor
                            && (events[j].isolates.contains(&events[i].donor_name)
                                || events[j].candidate_name == events[i].donor_name);
                        if cites_j_as_home || cites_j_as_donor {
                            drop_flags[i] = true;
                            break;
                        }
                    }
                }
            }
        }

        if !drop_flags.iter().any(|&d| d) {
            break;
        }
        let mut idx = 0usize;
        events.retain(|_| {
            let keep = !drop_flags[idx];
            idx += 1;
            keep
        });
    }
}

fn group_ancestral_events(
    mut events: Vec<crate::types::RecombinationEvent>,
    aln: &crate::types::Alignment,
) -> Vec<crate::types::RecombinationEvent> {
    let (_d_sister, d_clade) = crate::landmarks::compute_scale_invariant_distance_thresholds(aln);
    let mut grouped: Vec<crate::types::RecombinationEvent> = Vec::new();
    events.sort_by_key(|e| (e.home_idx, e.donor_idx, e.u1));
    for e in events {
        let mut found = false;
        for g in &mut grouped {
            // Direct compatibility: Home~Home and Donor~Donor within `d_clade`
            let home_matches = g.home_name == e.home_name
                || match (g.home_idx, e.home_idx) {
                    (Some(gh), Some(eh)) => {
                        crate::landmarks::compute_pairwise_distance(aln, gh, eh) <= d_clade
                    }
                    _ => false,
                };

            let donor_matches = g.donor_name == e.donor_name
                || match (g.donor_idx, e.donor_idx) {
                    (Some(gd), Some(ed)) => {
                        crate::landmarks::compute_pairwise_distance(aln, gd, ed) <= d_clade
                    }
                    _ => false,
                };

            if home_matches && donor_matches {
                let overlap_start = g.u1.max(e.u1);
                let overlap_end = g.u2.min(e.u2);
                if overlap_end >= overlap_start {
                    let overlap_len = overlap_end - overlap_start + 1;
                    let g_len = g.u2.saturating_sub(g.u1) + 1;
                    let e_len = e.u2.saturating_sub(e.u1) + 1;
                    let min_len = g_len.min(e_len);
                    let union_len = (g.u2.max(e.u2) - g.u1.min(e.u1) + 1).max(1);
                    let iou = (overlap_len as f64) / (union_len as f64);
                    let g_pair_check = verify_carpet_polarity_for_pair(
                        aln,
                        e.candidate_idx,
                        g.home_idx,
                        g.donor_idx,
                        g.is_ghost_donor,
                        g.u1.min(e.u1),
                        g.u2.max(e.u2),
                    );
                    let fit_compatible = match g_pair_check {
                        Some((_, _, d_fl_h_g, d_tr_d_g, _)) => {
                            let h_ok =
                                g.home_idx == e.home_idx || d_fl_h_g <= e.d_flank_home + d_clade;
                            let d_ok = g.donor_idx == e.donor_idx
                                || g.is_ghost_donor
                                || d_tr_d_g <= e.d_tract_donor + d_clade;
                            h_ok && d_ok
                        }
                        None => false,
                    };
                    if (iou >= 0.50
                        || (overlap_len as f64 >= 0.70 * min_len as f64
                            && min_len * 2 >= union_len))
                        && fit_compatible
                        && !has_divergent_overhang(
                            aln,
                            g.candidate_idx,
                            g.u1,
                            g.u2,
                            e.candidate_idx,
                            e.u1,
                            e.u2,
                            g.home_idx,
                            g.donor_idx,
                            d_clade,
                        )
                    {
                        g.isolates.extend(e.isolates.clone());
                        g.isolates.sort();
                        g.isolates.dedup();
                        g.u1 = g.u1.min(e.u1);
                        g.u2 = g.u2.max(e.u2);
                        g.u1_continuous = g.u1_continuous.min(e.u1_continuous);
                        g.u2_continuous = g.u2_continuous.max(e.u2_continuous);
                        g.tract_length = if g.u2 >= g.u1 { g.u2 - g.u1 + 1 } else { 0 };
                        if e.p_fisher < g.p_fisher {
                            g.p_fisher = e.p_fisher;
                        }
                        if e.z_phys > g.z_phys {
                            g.z_phys = e.z_phys;
                        }
                        g.s_informative = g.s_informative.max(e.s_informative);
                        found = true;
                        break;
                    }
                }
            }
        }
        if !found {
            grouped.push(e);
        }
    }

    // Resolve donor/home chains after merging identical primary events so merged isolate rosters
    // resolve any remaining co-recombinant parental citations before cohort recruitment
    resolve_donor_chains(&mut grouped, aln);
    snap_to_nearest_cohort_parents(&mut grouped, aln);

    // Sort grouped events by statistical significance before recruitment so the strongest
    // ancestral events recruit their mosaic cohort isolates first.
    grouped.sort_by(|a, b| {
        a.p_fisher
            .partial_cmp(&b.p_fisher)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.z_phys
                    .partial_cmp(&a.z_phys)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    // Track verified and recruited isolate assignments to:
    // (a) prevent recruiting an isolate into a complementary event with inverted Home<->Donor polarity
    // (b) mask disjoint recombinant tracts of multi-event mosaics from global flank accumulators
    let mut active_isolate_events: Vec<(Option<usize>, Option<usize>, usize, usize, Vec<String>)> =
        grouped
            .iter()
            .map(|ev| (ev.home_idx, ev.donor_idx, ev.u1, ev.u2, ev.isolates.clone()))
            .collect();

    // Cohort Roster Recruitment (DIR-RHIZ-PAFF-013.9)
    // Once an ancestral event is verified between Home and Donor over [u1, u2],
    // enroll all cohort isolates that carry this verified mosaic haplotype into the event roster.
    // Governed strictly by the canonical physical polarity invariants across BOTH the global flanks
    // and the local flanking windows immediately adjacent to [u1, u2], plus strict disjointness
    // from the designated Home and Donor parental references:
    //   1. d_flank(S, H) < d_flank(S, D) and d_local_flank(S, H) < d_local_flank(S, D)
    //   2. d_tract(S, D) < d_tract(S, H)
    //   3. d_flank(S, D) > d_tract(S, D) and d_local_flank(S, D) > d_tract(S, D) (positive recoil towards Donor)
    //   4. d_tract(S, H) > d_flank(S, H) and d_tract(S, H) > d_local_flank(S, H) (positive departure from Home)
    for (g_idx, g) in grouped.iter_mut().enumerate() {
        if let Some(hg) = g.home_idx {
            let s_idx = g.u1.saturating_sub(1);
            let e_idx = g.u2.min(aln.length);
            if e_idx <= s_idx {
                continue;
            }
            let home_row = aln.row(hg);
            let donor_row_opt = g.donor_idx.map(|dg| aln.row(dg));
            let w_loc = (e_idx - s_idx).clamp(300, 1000);
            let loc_l_start = s_idx.saturating_sub(w_loc);
            let loc_r_end = (e_idx + w_loc).min(aln.length);

            for t_idx in 0..aln.num_taxa {
                let t_name = &aln.taxa[t_idx];
                if g.isolates.contains(t_name)
                    || *t_name == g.home_name
                    || *t_name == g.donor_name
                    || Some(t_idx) == g.home_idx
                    || Some(t_idx) == g.donor_idx
                {
                    continue;
                }
                // Do not recruit `t_name` into `g` if `t_name` already has a verified or recruited
                // event with inverted Home<->Donor clade polarity, or an overlapping same-clade
                // event with a distinct divergent overhang
                if let Some(dg) = g.donor_idx {
                    let has_conflicting_event =
                        active_isolate_events
                            .iter()
                            .any(|(oh_opt, od_opt, ou1, ou2, isos)| {
                                if !isos.contains(t_name) {
                                    return false;
                                }
                                match (oh_opt, od_opt) {
                                    (Some(oh), Some(od)) => {
                                        let is_inv = regional_or_global_same_clade(
                                            aln, *oh, dg, g.u1, g.u2, d_clade,
                                        ) && regional_or_global_same_clade(
                                            aln, *od, hg, *ou1, *ou2, d_clade,
                                        );
                                        if is_inv {
                                            return true;
                                        }
                                        let ov_s = g.u1.max(*ou1);
                                        let ov_e = g.u2.min(*ou2);
                                        if ov_e >= ov_s {
                                            let is_same = regional_or_global_same_clade(
                                                aln, *oh, hg, g.u1, g.u2, d_clade,
                                            ) && regional_or_global_same_clade(
                                                aln, *od, dg, g.u1, g.u2, d_clade,
                                            );
                                            if is_same
                                                && has_divergent_overhang(
                                                    aln,
                                                    g.candidate_idx,
                                                    g.u1,
                                                    g.u2,
                                                    t_idx,
                                                    *ou1,
                                                    *ou2,
                                                    g.home_idx,
                                                    g.donor_idx,
                                                    d_clade,
                                                )
                                            {
                                                return true;
                                            }
                                        }
                                        false
                                    }
                                    _ => false,
                                }
                            });
                    if has_conflicting_event {
                        continue;
                    }
                }
                // Collect any disjoint recombinant tracts already carried by `t_name`
                // so they do not contaminate `t_name`'s global Home-background flank accumulators
                let disjoint_tracts: Vec<(usize, usize)> = active_isolate_events
                    .iter()
                    .filter_map(|(_oh, _od, ou1, ou2, isos)| {
                        if isos.contains(t_name) && (*ou2 < g.u1 || *ou1 > g.u2) {
                            Some((ou1.saturating_sub(1), (*ou2).min(aln.length)))
                        } else {
                            None
                        }
                    })
                    .collect();

                let t_row = aln.row(t_idx);

                let mut d_tr_h = 0usize;
                let mut v_tr_h = 0usize;
                let mut d_fl_h = 0usize;
                let mut v_fl_h = 0usize;
                let mut d_loc_l_h = 0usize;
                let mut v_loc_l_h = 0usize;
                let mut d_loc_r_h = 0usize;
                let mut v_loc_r_h = 0usize;
                for u in 0..aln.length {
                    let rt = t_row[u];
                    let rh = home_row[u];
                    if rt > 0 && rh > 0 {
                        if u >= s_idx && u < e_idx {
                            v_tr_h += 1;
                            if rt != rh {
                                d_tr_h += 1;
                            }
                        } else {
                            let in_other_tract =
                                disjoint_tracts.iter().any(|&(os, oe)| u >= os && u < oe);
                            if !in_other_tract {
                                v_fl_h += 1;
                                if rt != rh {
                                    d_fl_h += 1;
                                }
                            }
                            if u >= loc_l_start && u < s_idx {
                                v_loc_l_h += 1;
                                if rt != rh {
                                    d_loc_l_h += 1;
                                }
                            } else if u >= e_idx && u < loc_r_end {
                                v_loc_r_h += 1;
                                if rt != rh {
                                    d_loc_r_h += 1;
                                }
                            }
                        }
                    }
                }
                let dist_tr_h = if v_tr_h > 0 {
                    (d_tr_h as f64) / (v_tr_h as f64)
                } else {
                    1.0
                };
                let dist_fl_h = if v_fl_h > 0 {
                    (d_fl_h as f64) / (v_fl_h as f64)
                } else {
                    1.0
                };
                let v_loc_h = v_loc_l_h + v_loc_r_h;
                let dist_loc_fl_h = if v_loc_h > 0 {
                    ((d_loc_l_h + d_loc_r_h) as f64) / (v_loc_h as f64)
                } else {
                    dist_fl_h
                };

                if let Some(donor_row) = donor_row_opt {
                    let mut d_tr_d = 0usize;
                    let mut v_tr_d = 0usize;
                    let mut d_fl_d = 0usize;
                    let mut v_fl_d = 0usize;
                    let mut d_loc_l_d = 0usize;
                    let mut v_loc_l_d = 0usize;
                    let mut d_loc_r_d = 0usize;
                    let mut v_loc_r_d = 0usize;
                    let mut d_tr_hd = 0usize;
                    let mut v_tr_hd = 0usize;
                    let mut d_fl_hd = 0usize;
                    let mut v_fl_hd = 0usize;
                    for u in 0..aln.length {
                        let rt = t_row[u];
                        let rd = donor_row[u];
                        let rh = home_row[u];
                        let in_other_tract = !(u >= s_idx && u < e_idx)
                            && disjoint_tracts.iter().any(|&(os, oe)| u >= os && u < oe);
                        if rt > 0 && rd > 0 {
                            if u >= s_idx && u < e_idx {
                                v_tr_d += 1;
                                if rt != rd {
                                    d_tr_d += 1;
                                }
                            } else {
                                if !in_other_tract {
                                    v_fl_d += 1;
                                    if rt != rd {
                                        d_fl_d += 1;
                                    }
                                }
                                if u >= loc_l_start && u < s_idx {
                                    v_loc_l_d += 1;
                                    if rt != rd {
                                        d_loc_l_d += 1;
                                    }
                                } else if u >= e_idx && u < loc_r_end {
                                    v_loc_r_d += 1;
                                    if rt != rd {
                                        d_loc_r_d += 1;
                                    }
                                }
                            }
                        }
                        if rh > 0 && rd > 0 {
                            if u >= s_idx && u < e_idx {
                                v_tr_hd += 1;
                                if rh != rd {
                                    d_tr_hd += 1;
                                }
                            } else if !in_other_tract {
                                v_fl_hd += 1;
                                if rh != rd {
                                    d_fl_hd += 1;
                                }
                            }
                        }
                    }
                    let dist_tr_d = if v_tr_d > 0 {
                        (d_tr_d as f64) / (v_tr_d as f64)
                    } else {
                        1.0
                    };
                    let dist_fl_d = if v_fl_d > 0 {
                        (d_fl_d as f64) / (v_fl_d as f64)
                    } else {
                        1.0
                    };
                    let v_loc_d = v_loc_l_d + v_loc_r_d;
                    let dist_loc_fl_d = if v_loc_d > 0 {
                        ((d_loc_l_d + d_loc_r_d) as f64) / (v_loc_d as f64)
                    } else {
                        dist_fl_d
                    };
                    let dist_tr_hd = if v_tr_hd > 0 {
                        (d_tr_hd as f64) / (v_tr_hd as f64)
                    } else {
                        1.0
                    };
                    let dist_fl_hd = if v_fl_hd > 0 {
                        (d_fl_hd as f64) / (v_fl_hd as f64)
                    } else {
                        1.0
                    };

                    let both_arms_ok =
                        if v_loc_l_h >= 50 && v_loc_l_d >= 50 && v_loc_r_h >= 50 && v_loc_r_d >= 50
                        {
                            let l_h = (d_loc_l_h as f64) / (v_loc_l_h as f64);
                            let l_d = (d_loc_l_d as f64) / (v_loc_l_d as f64);
                            let r_h = (d_loc_r_h as f64) / (v_loc_r_h as f64);
                            let r_d = (d_loc_r_d as f64) / (v_loc_r_d as f64);
                            l_h < l_d || r_h < r_d
                        } else {
                            true
                        };

                    let matches_mosaic = dist_fl_h < dist_fl_d
                        && dist_loc_fl_h < dist_loc_fl_d
                        && both_arms_ok
                        && dist_tr_d < dist_tr_h
                        && dist_fl_d > dist_tr_d
                        && dist_loc_fl_d > dist_tr_d
                        && dist_tr_h > dist_fl_h
                        && dist_tr_h > dist_loc_fl_h
                        && dist_fl_h < 0.75 * dist_fl_hd
                        && dist_tr_d < 0.75 * dist_tr_hd
                        && verify_carpet_polarity_for_pair(
                            aln,
                            t_idx,
                            g.home_idx,
                            g.donor_idx,
                            false,
                            g.u1,
                            g.u2,
                        )
                        .is_some();

                    if matches_mosaic {
                        g.isolates.push(t_name.clone());
                    }
                } else if g.is_ghost_donor {
                    let cand_row = aln.row(g.candidate_idx);
                    let mut d_tr_c = 0usize;
                    let mut v_tr_c = 0usize;
                    for u in s_idx..e_idx {
                        let rt = t_row[u];
                        let rc = cand_row[u];
                        if rt > 0 && rc > 0 {
                            v_tr_c += 1;
                            if rt != rc {
                                d_tr_c += 1;
                            }
                        }
                    }
                    let dist_tr_c = if v_tr_c > 0 {
                        (d_tr_c as f64) / (v_tr_c as f64)
                    } else {
                        1.0
                    };
                    let (_d_sister, d_clade) =
                        crate::landmarks::compute_scale_invariant_distance_thresholds(aln);
                    if dist_tr_c < dist_tr_h
                        && dist_tr_h > dist_fl_h
                        && dist_tr_h > dist_loc_fl_h
                        && dist_fl_h < d_clade
                    {
                        g.isolates.push(t_name.clone());
                    }
                }
            }
            // Enforce strict disjointness between recombinant isolates and parental references
            g.isolates
                .retain(|iso| *iso != g.home_name && *iso != g.donor_name);
            if g.isolates.is_empty() {
                g.isolates.push(g.candidate_name.clone());
            }
            g.isolates.sort();
            g.isolates.dedup();
            active_isolate_events[g_idx].4 = g.isolates.clone();
        }
    }

    // Re-run donor/home chain resolution and mirror pruning after cohort recruitment so any event citing a
    // recruited co-recombinant isolate resolves to its primary non-recombinant parent
    prune_unresolved_recombinant_parent_mirrors(&mut grouped, aln);
    for g in &mut grouped {
        g.isolates
            .retain(|iso| *iso != g.home_name && *iso != g.donor_name);
        if g.isolates.is_empty() {
            g.isolates.push(g.candidate_name.clone());
        }
    }

    // Sort grouped events by statistical significance (lowest p-value first, highest z-phys tiebreaker)
    grouped.sort_by(|a, b| {
        a.p_fisher
            .partial_cmp(&b.p_fisher)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.z_phys
                    .partial_cmp(&a.z_phys)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    // Post-recruitment consolidation: merge redundant or contiguous events that recruited the same
    // recipient cohort via sister donors/homes from the same lineage cluster.
    let mut consolidated: Vec<crate::types::RecombinationEvent> = Vec::new();
    for ev in grouped {
        let mut merged_into_existing = false;
        for existing in &mut consolidated {
            // Recipient roster overlap
            let shared_isolates = ev
                .isolates
                .iter()
                .filter(|iso| existing.isolates.contains(iso))
                .count();
            let min_iso = ev.isolates.len().min(existing.isolates.len()).max(1);
            let union_iso = (ev.isolates.len() + existing.isolates.len() - shared_isolates).max(1);
            let jaccard_iso = (shared_isolates as f64) / (union_iso as f64);
            let subset_iso = (shared_isolates as f64) / (min_iso as f64);

            if jaccard_iso < 0.50 && subset_iso < 0.75 {
                continue;
            }

            let same_home_clade = existing.home_name == ev.home_name
                || match (existing.home_idx, ev.home_idx) {
                    (Some(h1), Some(h2)) => {
                        crate::landmarks::compute_pairwise_distance(aln, h1, h2) <= d_clade
                    }
                    _ => false,
                };
            let same_donor_clade = existing.donor_name == ev.donor_name
                || (existing.is_ghost_donor && ev.is_ghost_donor)
                || match (existing.donor_idx, ev.donor_idx) {
                    (Some(d1), Some(d2)) => {
                        crate::landmarks::compute_pairwise_distance(aln, d1, d2) <= d_clade
                    }
                    _ => false,
                };

            if !same_home_clade || !same_donor_clade {
                continue;
            }

            let ov_start = ev.u1.max(existing.u1);
            let ov_end = ev.u2.min(existing.u2);
            let is_overlapping_or_contiguous = if ov_end >= ov_start {
                let ov_len = ov_end - ov_start + 1;
                let union_len = (ev.u2.max(existing.u2) - ev.u1.min(existing.u1) + 1).max(1);
                (ov_len as f64) / (union_len as f64) >= 0.50 || jaccard_iso >= 0.75
            } else {
                let gap = ov_start - ov_end;
                gap <= 350 && jaccard_iso >= 0.75
            };

            let merged_u1 = existing.u1.min(ev.u1);
            let merged_u2 = existing.u2.max(ev.u2);

            if is_overlapping_or_contiguous
                && verify_carpet_polarity_for_pair(
                    aln,
                    existing.candidate_idx,
                    existing.home_idx,
                    existing.donor_idx,
                    existing.is_ghost_donor,
                    merged_u1,
                    merged_u2,
                )
                .is_some()
                && verify_carpet_polarity_for_pair(
                    aln,
                    ev.candidate_idx,
                    existing.home_idx,
                    existing.donor_idx,
                    existing.is_ghost_donor,
                    merged_u1,
                    merged_u2,
                )
                .is_some()
                && !has_divergent_overhang(
                    aln,
                    existing.candidate_idx,
                    existing.u1,
                    existing.u2,
                    ev.candidate_idx,
                    ev.u1,
                    ev.u2,
                    existing.home_idx,
                    existing.donor_idx,
                    d_clade,
                )
            {
                existing.u1 = merged_u1;
                existing.u2 = merged_u2;
                existing.u1_continuous = existing.u1 as f64;
                existing.u2_continuous = existing.u2 as f64;
                existing.tract_length = existing.u2.saturating_sub(existing.u1) + 1;
                existing.isolates.extend(ev.isolates.clone());
                existing
                    .isolates
                    .retain(|iso| *iso != existing.home_name && *iso != existing.donor_name);
                if existing.isolates.is_empty() {
                    existing.isolates.push(existing.candidate_name.clone());
                }
                existing.isolates.sort();
                existing.isolates.dedup();
                if ev.p_fisher < existing.p_fisher {
                    existing.p_fisher = ev.p_fisher;
                }
                if ev.z_phys > existing.z_phys {
                    existing.z_phys = ev.z_phys;
                }
                existing.s_informative = existing.s_informative.max(ev.s_informative);
                merged_into_existing = true;
                break;
            }
        }
        if !merged_into_existing {
            consolidated.push(ev);
        }
    }

    // Assign 1-based sequential event ID and sort by it
    for (i, g) in consolidated.iter_mut().enumerate() {
        g.event_id = i + 1;
    }
    consolidated
}

/// Deduplicates overlapping verified events across different parental channels for a single candidate taxon.
/// Preserves distinct non-overlapping tracts and nested multi-way mosaics (e.g. Russian dolls from distinct donors)
/// while suppressing redundant sister-channel duplicates (IoU >= 0.50 or overlap >= 70% for same-donor clade).
fn deduplicate_candidate_events(
    mut events: Vec<crate::types::RecombinationEvent>,
    aln: &crate::types::Alignment,
) -> Vec<crate::types::RecombinationEvent> {
    if events.len() <= 1 {
        return events;
    }

    let (_d_sister, d_clade) = crate::landmarks::compute_scale_invariant_distance_thresholds(aln);

    // Sort by statistical significance (p_fisher ascending, s_informative descending, z_phys descending)
    // so complete coalesced tracts take priority over split sub-fragments.
    events.sort_by(|a, b| {
        a.p_fisher
            .partial_cmp(&b.p_fisher)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.s_informative.cmp(&a.s_informative))
            .then_with(|| {
                b.z_phys
                    .partial_cmp(&a.z_phys)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let mut kept: Vec<crate::types::RecombinationEvent> = Vec::new();

    for ev in events {
        let mut is_redundant = false;
        let l_ev = (ev.u2.saturating_sub(ev.u1) + 1) as f64;

        for existing in &kept {
            let s_ov = ev.u1.max(existing.u1);
            let e_ov = ev.u2.min(existing.u2);

            if e_ov >= s_ov {
                let inter = (e_ov - s_ov + 1) as f64;
                let l_exist = (existing.u2.saturating_sub(existing.u1) + 1) as f64;
                let union_span = (ev.u2.max(existing.u2) - ev.u1.min(existing.u1) + 1) as f64;
                let iou = inter / union_span.max(1.0);
                let frac_ev = inter / l_ev.max(1.0);
                let frac_exist = inter / l_exist.max(1.0);

                let is_same_donor = match (ev.donor_idx, existing.donor_idx) {
                    (Some(d1), Some(d2)) => {
                        d1 == d2
                            || crate::landmarks::compute_pairwise_distance(aln, d1, d2) < d_clade
                    }
                    (None, None) => true,
                    _ => false,
                };

                if is_same_donor {
                    // For the same donor clade, any substantial overlap is redundant
                    if iou >= 0.50 || frac_ev >= 0.70 || frac_exist >= 0.70 {
                        is_redundant = true;
                        break;
                    }
                } else {
                    // For distinct donor clades, only compete if they claim the exact same span (IoU >= 0.50)
                    // Nested Russian dolls (IoU < 0.50) from distinct donors are preserved.
                    if iou >= 0.50 {
                        is_redundant = true;
                        break;
                    }
                }
            }
        }

        if !is_redundant {
            kept.push(ev);
        }
    }

    // Restore coordinate order
    kept.sort_by_key(|e| e.u1);
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Alignment;

    #[test]
    fn test_deduplicate_candidate_events() {
        let make_event =
            |u1: usize, u2: usize, donor: &str, z: f64| crate::types::RecombinationEvent {
                event_id: 1,
                isolates: vec!["R".to_string()],
                candidate_idx: 0,
                candidate_name: "R".to_string(),
                home_idx: Some(1),
                home_name: "P1".to_string(),
                donor_idx: Some(2),
                donor_name: donor.to_string(),
                is_ghost_donor: false,
                is_crossover: false,
                u1,
                u2,
                u1_continuous: u1 as f64,
                u2_continuous: u2 as f64,
                tract_length: u2 - u1 + 1,
                s_informative: 20,
                p_fisher: 1e-10,
                z_phys: z,
                d_flank_home: 0.005,
                d_flank_donor: 0.05,
                d_tract_home: 0.05,
                d_tract_donor: 0.005,
                is_verified: true,
            };

        let t1 = make_event(100, 500, "P2", 12.0);
        let t2 = make_event(110, 490, "P3", 9.0);
        let t3 = make_event(800, 1200, "P4", 11.0);

        let dummy_aln = Alignment {
            taxa: vec![
                "R".into(),
                "P1".into(),
                "P2".into(),
                "P3".into(),
                "P4".into(),
            ],
            num_taxa: 5,
            length: 1500,
            matrix: vec![1u8; 5 * 1500],
        };
        let dedup = deduplicate_candidate_events(vec![t1, t2, t3], &dummy_aln);
        assert_eq!(
            dedup.len(),
            2,
            "Expected 2 events after deduplicating redundant overlap"
        );
        assert_eq!(dedup[0].u1, 100);
        assert_eq!(dedup[0].u2, 500);
        assert_eq!(dedup[0].donor_name, "P2", "Stronger event should survive");
        assert_eq!(dedup[1].u1, 800);
        assert_eq!(dedup[1].u2, 1200);
        assert_eq!(dedup[1].donor_name, "P4");
    }

    #[test]
    fn test_multiway_mosaic_detection() {
        let l = 2000;
        let taxa = vec![
            "R".into(),
            "P1".into(),
            "P2".into(),
            "P3".into(),
            "O".into(),
        ];
        let n = taxa.len();

        let mut data = vec![1u8; n * l];

        // Seed distinctive patterns
        // P1 is mostly state 1
        // P2 has state 2 every 6 sites
        // P3 has state 3 every 6 sites
        // O has state 4 every 4 sites
        for u in 0..l {
            if u % 6 == 0 {
                data[2 * l + u] = 2; // P2
            }
            if u % 6 == 3 {
                data[3 * l + u] = 3; // P3
            }
            if u % 4 == 0 {
                data[4 * l + u] = 4; // O
            }
        }

        // Construct R as multiway mosaic:
        // 0..400: P1
        // 400..800: P2
        // 800..1200: P1
        // 1200..1600: P3
        // 1600..2000: P1
        for u in 0..l {
            if u >= 400 && u < 800 {
                data[0 * l + u] = data[2 * l + u];
            } else if u >= 1200 && u < 1600 {
                data[0 * l + u] = data[3 * l + u];
            } else {
                data[0 * l + u] = data[1 * l + u];
            }
        }

        let aln = Alignment::new(taxa, l, data);
        let engine = RhizAeonEngine::new();
        let res = engine.scan(&aln);

        assert!(
            !res.is_non_recombinant_h0,
            "Expected multiway mosaic to be detected"
        );
        println!("Detected events: {:?}", res.events.len());
        for ev in &res.events {
            println!(
                "Event: {} in [{}, {}] Home: {} Donor: {}",
                ev.candidate_name, ev.u1, ev.u2, ev.home_name, ev.donor_name
            );
        }

        // Verify that R is the candidate and both tracts are detected
        let r_events: Vec<_> = res
            .events
            .iter()
            .filter(|e| e.candidate_name == "R")
            .collect();
        assert!(
            r_events.len() >= 2,
            "Expected at least 2 events for multi-way mosaic child R"
        );

        let donors: Vec<_> = r_events.iter().map(|e| e.donor_name.as_str()).collect();
        assert!(
            donors.contains(&"P2"),
            "Expected P2 to be identified as donor"
        );
        assert!(
            donors.contains(&"P3"),
            "Expected P3 to be identified as donor"
        );
    }

    #[test]
    fn test_resolve_donor_chains_and_grouping() {
        let taxa = vec!["H".into(), "L".into(), "X".into(), "I".into()];
        let l = 1000;
        let data = vec![1u8; 4 * l];
        let aln = Alignment::new(taxa, l, data);

        let e1 = crate::types::RecombinationEvent {
            event_id: 1,
            isolates: vec!["H".into()],
            candidate_idx: 0,
            candidate_name: "H".into(),
            home_idx: Some(3),
            home_name: "I".into(),
            donor_idx: Some(1),
            donor_name: "L".into(),
            is_ghost_donor: false,
            is_crossover: false,
            u1: 1,
            u2: 500,
            u1_continuous: 1.0,
            u2_continuous: 500.0,
            tract_length: 500,
            s_informative: 50,
            p_fisher: 1e-10,
            z_phys: 8.0,
            d_flank_home: 0.05,
            d_flank_donor: 0.30,
            d_tract_home: 0.30,
            d_tract_donor: 0.02,
            is_verified: true,
        };

        let e2 = crate::types::RecombinationEvent {
            event_id: 2,
            isolates: vec!["L".into()],
            candidate_idx: 1,
            candidate_name: "L".into(),
            home_idx: Some(3),
            home_name: "I".into(),
            donor_idx: Some(2),
            donor_name: "X".into(),
            is_ghost_donor: false,
            is_crossover: false,
            u1: 1,
            u2: 500,
            u1_continuous: 1.0,
            u2_continuous: 500.0,
            tract_length: 500,
            s_informative: 60,
            p_fisher: 1e-15,
            z_phys: 9.0,
            d_flank_home: 0.04,
            d_flank_donor: 0.35,
            d_tract_home: 0.35,
            d_tract_donor: 0.01,
            is_verified: true,
        };

        let mut events = vec![e1, e2];
        resolve_donor_chains(&mut events, &aln);

        // Both should now cite donor X
        assert_eq!(events[0].donor_name, "X");
        assert_eq!(events[1].donor_name, "X");

        // Grouping should merge H and L into a single ancestral event
        let grouped = group_ancestral_events(events, &aln);
        assert_eq!(
            grouped.len(),
            1,
            "Expected H and L to merge into a single co-recombinant event"
        );
        assert_eq!(grouped[0].isolates, vec!["H", "L"]);
        assert_eq!(grouped[0].home_name, "I");
        assert_eq!(grouped[0].donor_name, "X");
        assert_eq!(grouped[0].u1, 1);
        assert_eq!(grouped[0].u2, 500);
        assert_eq!(
            grouped[0].p_fisher, 1e-15,
            "Strongest p-value should be preserved"
        );
        assert_eq!(
            grouped[0].z_phys, 9.0,
            "Highest z-score should be preserved"
        );
    }

    #[test]
    fn test_tier2_and_tier3_stringency_controls() {
        let l = 2000;
        let taxa = vec![
            "R".into(),
            "P1".into(),
            "P2".into(),
            "P3".into(),
            "O".into(),
        ];
        let n = taxa.len();
        let mut data = vec![1u8; n * l];

        for u in 0..l {
            if u % 6 == 0 {
                data[2 * l + u] = 2; // P2
            }
            if u % 6 == 3 {
                data[3 * l + u] = 3; // P3
            }
            if u % 4 == 0 {
                data[4 * l + u] = 4; // O
            }
        }
        // R has a primary 400 bp tract from P2 [400..800] and a borderline 48 bp micro-conversion from P3 [1200..1248]
        for u in 0..l {
            if u >= 400 && u < 800 {
                data[0 * l + u] = data[2 * l + u];
            } else if u >= 1200 && u < 1248 {
                data[0 * l + u] = data[3 * l + u];
            } else {
                data[0 * l + u] = data[1 * l + u];
            }
        }
        let aln = Alignment::new(taxa, l, data);

        let res_relaxed = RhizAeonEngine::new()
            .with_tier2_stringency(0.40)
            .with_tier3_stringency(0.40)
            .scan(&aln);

        let res_default = RhizAeonEngine::new().scan(&aln);

        let res_strict = RhizAeonEngine::new()
            .with_tier2_stringency(2.50)
            .with_tier3_stringency(2.50)
            .scan(&aln);

        assert!(
            res_relaxed.events.len() >= res_default.events.len()
                && res_default.events.len() >= res_strict.events.len(),
            "Event count must be non-increasing with stringency: relaxed={}, default={}, strict={}",
            res_relaxed.events.len(),
            res_default.events.len(),
            res_strict.events.len()
        );
        assert!(
            res_relaxed.events.len() > res_strict.events.len(),
            "Relaxed stringency must retain more tracts than high stringency (relaxed={}, strict={})",
            res_relaxed.events.len(),
            res_strict.events.len()
        );
    }
}
