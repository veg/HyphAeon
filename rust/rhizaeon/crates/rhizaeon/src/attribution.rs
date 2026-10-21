use crate::types::LandmarkSet;

#[derive(Debug, Clone)]
pub struct TrajectoryParentalPair {
    pub candidate_idx: usize,
    pub home_channel: usize,
    pub donor_channel: usize,
    pub home_landmark_idx: Option<usize>,
    pub donor_landmark_idx: Option<usize>,
    pub is_ghost_donor: bool,
    pub home_name: String,
    pub donor_name: String,
    pub home_global_idx: Option<usize>,
    pub donor_global_idx: Option<usize>,
    pub swing_home: f64,
    pub swing_donor: f64,
}

pub struct TrajectoryParentalResolver {
    pub min_root_swing: f64,
    pub stringency: f64,
}

impl Default for TrajectoryParentalResolver {
    fn default() -> Self {
        Self {
            min_root_swing: 3.0,
            stringency: 1.0,
        }
    }
}

impl TrajectoryParentalResolver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_stringency(mut self, stringency: f64) -> Self {
        self.stringency = stringency;
        self
    }

    /// Resolves active reticulation dipole (P_Home, P_Donor) for candidate R (backward-compatible)
    pub fn resolve_reticulation_channels(
        &self,
        candidate_idx: usize,
        c_row: &[f64],
        a_row: &[f64],
        l: usize,
        n: usize,
        landmarks: &LandmarkSet,
        aln: &crate::types::Alignment,
    ) -> TrajectoryParentalPair {
        let all_pairs = self.resolve_all_reticulation_channels(
            candidate_idx,
            c_row,
            a_row,
            l,
            n,
            landmarks,
            aln,
            &[],
        );
        all_pairs.into_iter().next().unwrap()
    }

    /// Resolves all significant reticulation dipoles (P_Home, P_Donor_k) for candidate R
    /// enabling Multi-Way Mosaicism resolution across distinct parental channels.
    ///
    /// `c_row` and `a_row` are the candidate's own momentum and (Pass 2 patched) attention rows
    /// of shape (L, K+1), indexed `u * num_channels + ch`.
    pub fn resolve_all_reticulation_channels(
        &self,
        candidate_idx: usize,
        c_row: &[f64],
        a_row: &[f64],
        l: usize,
        n: usize,
        landmarks: &LandmarkSet,
        aln: &crate::types::Alignment,
        ghost_candidate_indices: &[usize],
    ) -> Vec<TrajectoryParentalPair> {
        let k = landmarks.num_landmarks;
        let num_channels = k + 1;
        let inv_l = 1.0 / (l as f64);
        debug_assert_eq!(c_row.len(), l * num_channels);
        debug_assert_eq!(a_row.len(), l * num_channels);

        // 1. Sequence-wide mean attention for candidate
        let mut mean_a = vec![0.0f64; num_channels];
        for u in 0..l {
            let offset = u * num_channels;
            for ch in 0..num_channels {
                mean_a[ch] += a_row[offset + ch];
            }
        }
        for ch in 0..num_channels {
            mean_a[ch] *= inv_l;
        }

        // 2. Swings for each channel
        let mut swings = vec![0.0f64; num_channels];
        for ch in 0..num_channels {
            let mut min_c = f64::INFINITY;
            let mut max_c = f64::NEG_INFINITY;
            for u in 0..l {
                let c = c_row[u * num_channels + ch];
                if c < min_c {
                    min_c = c;
                }
                if c > max_c {
                    max_c = c;
                }
            }
            swings[ch] = (max_c - min_c).max(0.0);
        }

        // 2b. Standardized Brownian bridge excursions and scale-invariant parameters
        let mut z_swings = vec![0.0f64; num_channels];
        for ch in 0..num_channels {
            let mut sum_sq_a = 0.0;
            for u in 0..l {
                let a_val = a_row[u * num_channels + ch];
                let diff = a_val - mean_a[ch];
                sum_sq_a += diff * diff;
            }
            let sigma_a = (sum_sq_a * inv_l).sqrt().max(1e-6);
            z_swings[ch] = swings[ch] / ((l as f64).sqrt() * sigma_a);
        }
        let uniform_share = 1.0 / (num_channels as f64);
        let km_channel = uniform_share;
        let k_crit_att = (-0.5 * (0.05 / (2.0 * (num_channels as f64))).ln())
            .sqrt()
            .max(1.645);
        let (d_sister, d_clade) =
            crate::landmarks::compute_scale_invariant_distance_thresholds(aln);

        // 3. Channel-contiguous trajectory buffer for SIMD cache-friendly dot products
        let mut c_traj = vec![0.0f64; num_channels * l];
        for u in 0..l {
            let offset = u * num_channels;
            for ch in 0..num_channels {
                c_traj[ch * l + u] = c_row[offset + ch];
            }
        }

        let mut mean_c = vec![0.0f64; num_channels];
        for ch in 0..num_channels {
            let slice = &c_traj[ch * l..(ch + 1) * l];
            mean_c[ch] = slice.iter().sum::<f64>() * inv_l;
        }

        // Covariance matrix of C_R in shape (num_channels, num_channels)
        let mut cov_c = vec![0.0f64; num_channels * num_channels];
        for a in 0..num_channels {
            let slice_a = &c_traj[a * l..(a + 1) * l];
            let mean_a = mean_c[a];
            for b in a..num_channels {
                let slice_b = &c_traj[b * l..(b + 1) * l];
                let mut sum_ab = 0.0f64;
                for u in 0..l {
                    sum_ab += slice_a[u] * slice_b[u];
                }
                let cov = sum_ab * inv_l - mean_a * mean_c[b];
                cov_c[a * num_channels + b] = cov;
                cov_c[b * num_channels + a] = cov;
            }
        }

        // 4. Eligible contemporary landmark channels
        let mut contemporary_channels = Vec::new();
        for p in 0..k {
            let l_global = landmarks.indices[p];
            if l_global != candidate_idx {
                contemporary_channels.push(p + 1);
            }
        }
        if contemporary_channels.is_empty() {
            contemporary_channels.push(1);
        }

        // Check whether candidate has at least one landmark in its own clade (d <= d_clade)
        // and at least one landmark outside its clade (d > d_clade)
        let has_intra_clade_lm = contemporary_channels.iter().any(|&ch| {
            let l_idx = landmarks.indices[ch - 1];
            crate::landmarks::compute_pairwise_distance(aln, candidate_idx, l_idx) <= d_clade
        });
        let has_inter_clade_lm = contemporary_channels.iter().any(|&ch| {
            let l_idx = landmarks.indices[ch - 1];
            crate::landmarks::compute_pairwise_distance(aln, candidate_idx, l_idx) > d_clade
        });

        // 5. Find optimal primary pair across all contemporary pairs
        let mut best_score = -1.0;
        let mut best_pair = (contemporary_channels[0], contemporary_channels[0]);
        let mut best_is_sister_a = false;
        let mut best_is_sister_b = false;

        for i in 0..contemporary_channels.len() {
            let ch_a = contemporary_channels[i];
            let l_a = landmarks.indices[ch_a - 1];
            let d_ca = crate::landmarks::compute_pairwise_distance(aln, candidate_idx, l_a);
            let is_sister_a = d_ca < d_sister;
            let in_clade_a = d_ca <= d_clade;
            let var_a = cov_c[ch_a * num_channels + ch_a].max(1e-12);

            for j in (i + 1)..contemporary_channels.len() {
                let ch_b = contemporary_channels[j];
                let l_b = landmarks.indices[ch_b - 1];
                let d_cb = crate::landmarks::compute_pairwise_distance(aln, candidate_idx, l_b);
                let is_sister_b = d_cb < d_sister;
                let in_clade_b = d_cb <= d_clade;

                // Both channels cannot be sisters of candidate (must be different clades)
                if is_sister_a && is_sister_b {
                    continue;
                }
                // When both intra-clade and inter-clade landmarks exist, disallow two intra-clade channels
                // from forming a mirror dipole inside the candidate's own lineage cluster
                if has_intra_clade_lm && has_inter_clade_lm && in_clade_a && in_clade_b {
                    continue;
                }

                let var_b = cov_c[ch_b * num_channels + ch_b].max(1e-12);
                let cov_ab = cov_c[ch_a * num_channels + ch_b];

                let diff_var = (var_a + var_b - 2.0 * cov_ab).max(0.0);
                let rho_ab = (cov_ab / (var_a * var_b).sqrt()).clamp(-1.0, 1.0);
                let opposition = 1.0 - rho_ab;

                let max_mean_a = mean_a[ch_a].max(mean_a[ch_b]);
                let attention_support = max_mean_a / (max_mean_a + km_channel);

                let score = diff_var * opposition * attention_support;
                if score > best_score {
                    best_score = score;
                    best_pair = (ch_a, ch_b);
                    best_is_sister_a = is_sister_a || (in_clade_a && !in_clade_b);
                    best_is_sister_b = is_sister_b || (in_clade_b && !in_clade_a);
                }
            }
        }

        let (mut home_ch, primary_donor_ch) = if best_is_sister_a {
            (best_pair.0, best_pair.1)
        } else if best_is_sister_b {
            (best_pair.1, best_pair.0)
        } else if mean_a[best_pair.0] >= mean_a[best_pair.1] {
            (best_pair.0, best_pair.1)
        } else {
            (best_pair.1, best_pair.0)
        };

        // Co-recombinant sisters (d < d_sister) cannot serve as Home parental reference
        // when non-sister contemporary landmarks (d >= d_sister) exist:
        let non_sister_channels: Vec<usize> = contemporary_channels
            .iter()
            .copied()
            .filter(|&ch| {
                let l_idx = landmarks.indices[ch - 1];
                !crate::landmarks::is_true_sister_pair(aln, candidate_idx, l_idx, d_sister)
            })
            .collect();

        if !non_sister_channels.is_empty() {
            let h_idx = landmarks.indices[home_ch.saturating_sub(1)];
            if crate::landmarks::is_true_sister_pair(aln, candidate_idx, h_idx, d_sister) {
                let mut best_ns_home = non_sister_channels[0];
                let mut best_ns_mean = -1.0f64;
                for &ch in &non_sister_channels {
                    if mean_a[ch] > best_ns_mean {
                        best_ns_mean = mean_a[ch];
                        best_ns_home = ch;
                    }
                }
                home_ch = best_ns_home;
            }
            for &ch in &non_sister_channels {
                if mean_a[ch] > 1.3 * mean_a[home_ch] && swings[ch] < swings[home_ch] {
                    home_ch = ch;
                }
            }
        } else {
            for &ch in &contemporary_channels {
                if mean_a[ch] > 1.3 * mean_a[home_ch] && swings[ch] < swings[home_ch] {
                    home_ch = ch;
                }
            }
        }

        // 6. Score candidate donors against home_ch
        struct CandidateDonor {
            channel: usize,
            is_ghost: bool,
            score: f64,
        }

        let mut candidate_donors = Vec::new();

        // Always include primary donor
        candidate_donors.push(CandidateDonor {
            channel: primary_donor_ch,
            is_ghost: false,
            score: best_score.max(1e-6),
        });

        // Test other contemporary channels against home_ch
        let home_l_idx = landmarks.indices[home_ch.saturating_sub(1)];
        for &d in &contemporary_channels {
            if d == home_ch || d == primary_donor_ch {
                continue;
            }
            let l_d = landmarks.indices[d - 1];
            let d_cd = crate::landmarks::compute_pairwise_distance(aln, candidate_idx, l_d);
            let d_hd = crate::landmarks::compute_pairwise_distance(aln, home_l_idx, l_d);
            if d_cd < d_sister
                || (n >= 8 && has_inter_clade_lm && d_cd <= d_clade && d_hd <= d_clade)
            {
                continue;
            }

            let var_h = cov_c[home_ch * num_channels + home_ch].max(1e-12);
            let var_d = cov_c[d * num_channels + d].max(1e-12);
            let cov_hd = cov_c[home_ch * num_channels + d];

            let diff_var = (var_h + var_d - 2.0 * cov_hd).max(0.0);
            let rho_hd = (cov_hd / (var_h * var_d).sqrt()).clamp(-1.0, 1.0);
            let opposition = 1.0 - rho_hd;

            let max_mean = mean_a[home_ch].max(mean_a[d]);
            let attention_support = max_mean / (max_mean + km_channel);

            let score_d = diff_var * opposition * attention_support;

            // Donor selection criterion:
            // Needs meaningful contrast with Home and non-trivial attention/swing
            let s2 = self.stringency.max(0.1);
            if score_d >= 0.05 * s2 * best_score.max(1e-6)
                && (mean_a[d] >= 0.5 * s2 * uniform_share || z_swings[d] >= 1.0 * s2)
                && (z_swings[d] >= k_crit_att * s2 || swings[d] >= 0.25 * s2)
            {
                candidate_donors.push(CandidateDonor {
                    channel: d,
                    is_ghost: false,
                    score: score_d,
                });
            }
        }

        // Channel 0: Ghost / Unsampled Donor evaluation (DIR-RHIZ-PAFF-012.3)
        // Autapomorphic rate bursts naturally pool into [ROOT], preventing false donor attribution.
        let var_h = cov_c[home_ch * num_channels + home_ch].max(1e-12);
        let var_0 = cov_c[0].max(1e-12);
        let cov_h0 = cov_c[home_ch * num_channels + 0];

        let diff_var_0 = (var_h + var_0 - 2.0 * cov_h0).max(0.0);
        let rho_h0 = (cov_h0 / (var_h * var_0).sqrt()).clamp(-1.0, 1.0);
        let opposition_0 = 1.0 - rho_h0;

        let max_mean_0 = mean_a[home_ch].max(mean_a[0]);
        let attention_support_0 = max_mean_0 / (max_mean_0 + km_channel);
        let score_0 = diff_var_0 * opposition_0 * attention_support_0;

        let s2 = self.stringency.max(0.1);
        let allow_ghost = ghost_candidate_indices.contains(&candidate_idx);
        if allow_ghost
            && score_0 >= 0.05 * s2 * best_score.max(1e-6)
            && (mean_a[0] >= 0.5 * s2 * uniform_share || z_swings[0] >= 1.0 * s2)
            && (z_swings[0] >= k_crit_att * s2 || swings[0] >= 0.25 * s2)
        {
            candidate_donors.push(CandidateDonor {
                channel: 0,
                is_ghost: true,
                score: score_0,
            });
        }

        // Sort candidate donors by score descending
        candidate_donors.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Deduplicate candidate donors across clades (`d <= d_clade`) first so a single donor population
        // cannot consume all 4 slots and crowd out secondary donor populations in multi-way mosaicism;
        // then fill remaining slots (up to 4) with non-sister channels (`d >= d_sister`).
        let mut distinct_donors: Vec<CandidateDonor> = Vec::new();
        let mut fallback_clade_sisters: Vec<CandidateDonor> = Vec::new();
        for cd in candidate_donors {
            if cd.is_ghost {
                distinct_donors.push(cd);
                continue;
            }
            let dk = cd.channel.saturating_sub(1);
            if let Some(dg) = landmarks.indices.get(dk).copied() {
                let is_sister = distinct_donors.iter().any(|existing| {
                    if existing.is_ghost {
                        return false;
                    }
                    let ex_k = existing.channel.saturating_sub(1);
                    if let Some(ex_g) = landmarks.indices.get(ex_k).copied() {
                        crate::landmarks::compute_pairwise_distance(aln, dg, ex_g) < d_sister
                    } else {
                        false
                    }
                });
                if is_sister {
                    continue;
                }
                let is_same_clade = distinct_donors.iter().any(|existing| {
                    if existing.is_ghost {
                        return false;
                    }
                    let ex_k = existing.channel.saturating_sub(1);
                    if let Some(ex_g) = landmarks.indices.get(ex_k).copied() {
                        crate::landmarks::compute_pairwise_distance(aln, dg, ex_g) <= d_clade
                    } else {
                        false
                    }
                });
                if !is_same_clade {
                    distinct_donors.push(cd);
                } else {
                    fallback_clade_sisters.push(cd);
                }
            } else {
                distinct_donors.push(cd);
            }
        }
        for cd in fallback_clade_sisters {
            if distinct_donors.len() >= 4 {
                break;
            }
            distinct_donors.push(cd);
        }
        distinct_donors.truncate(4);
        let candidate_donors = distinct_donors;

        // 7. Construct TrajectoryParentalPairs
        let home_k = home_ch.saturating_sub(1);
        let home_name = landmarks
            .taxa
            .get(home_k)
            .cloned()
            .unwrap_or_else(|| "Home".to_string());
        let home_global = landmarks.indices.get(home_k).copied();

        let mut pairs = Vec::with_capacity(candidate_donors.len() + 4);
        for cd in &candidate_donors {
            let (donor_k, donor_name, donor_global) = if cd.is_ghost {
                (None, "Ghost".to_string(), None)
            } else {
                let dk = cd.channel.saturating_sub(1);
                (
                    Some(dk),
                    landmarks
                        .taxa
                        .get(dk)
                        .cloned()
                        .unwrap_or_else(|| "Donor".to_string()),
                    landmarks.indices.get(dk).copied(),
                )
            };

            pairs.push(TrajectoryParentalPair {
                candidate_idx,
                home_channel: home_ch,
                donor_channel: cd.channel,
                home_landmark_idx: Some(home_k),
                donor_landmark_idx: donor_k,
                is_ghost_donor: cd.is_ghost,
                home_name: home_name.clone(),
                donor_name,
                home_global_idx: home_global,
                donor_global_idx: donor_global,
                swing_home: swings[home_ch],
                swing_donor: swings[cd.channel],
            });
        }

        // Clonal Medoid Reference Augmentation:
        // In `select_landmarks_from_pool`, the first landmarks are the pure clonal medoids of each lineage cluster.
        // If `home_ch` is a partially mosaic sub-lineage sister rather than the purest clade medoid (minimum mean
        // intra-clade distance to candidate's clade), also evaluate dipoles anchored on the purest intra-clade medoid
        // against the top external clade medoids so shared sub-lineage tracts (e.g. 1169-1269 in pop2) are not masked!
        let mut best_medoid_ch = None;
        let mut min_intra_d = f64::INFINITY;
        for &ch in &contemporary_channels {
            let l_idx = landmarks.indices[ch - 1];
            let d_cl = crate::landmarks::compute_pairwise_distance(aln, candidate_idx, l_idx);
            if d_cl <= d_clade && d_cl >= d_sister {
                // Check mean distance of l_idx to all landmarks in the same clade
                let mut sum_d = 0.0;
                let mut cnt = 0usize;
                for &ch2 in &contemporary_channels {
                    let l2 = landmarks.indices[ch2 - 1];
                    if crate::landmarks::compute_pairwise_distance(aln, l_idx, l2) <= d_clade {
                        sum_d += crate::landmarks::compute_pairwise_distance(aln, l_idx, l2);
                        cnt += 1;
                    }
                }
                let mean_d = if cnt > 0 { sum_d / (cnt as f64) } else { d_cl };
                if mean_d < min_intra_d {
                    min_intra_d = mean_d;
                    best_medoid_ch = Some(ch);
                }
            }
        }
        if let Some(med_h_ch) = best_medoid_ch {
            if med_h_ch != home_ch {
                let mh_k = med_h_ch - 1;
                let mh_name = landmarks
                    .taxa
                    .get(mh_k)
                    .cloned()
                    .unwrap_or_else(|| "Home".to_string());
                let mh_global = landmarks.indices.get(mh_k).copied();
                for cd in &candidate_donors {
                    if cd.is_ghost {
                        continue;
                    }
                    let dk = cd.channel.saturating_sub(1);
                    pairs.push(TrajectoryParentalPair {
                        candidate_idx,
                        home_channel: med_h_ch,
                        donor_channel: cd.channel,
                        home_landmark_idx: Some(mh_k),
                        donor_landmark_idx: Some(dk),
                        is_ghost_donor: false,
                        home_name: mh_name.clone(),
                        donor_name: landmarks
                            .taxa
                            .get(dk)
                            .cloned()
                            .unwrap_or_else(|| "Donor".to_string()),
                        home_global_idx: mh_global,
                        donor_global_idx: landmarks.indices.get(dk).copied(),
                        swing_home: swings[med_h_ch],
                        swing_donor: swings[cd.channel],
                    });
                }
            }
        }

        // Multi-Way Mosaicism: Also evaluate secondary reticulation dipoles between distinct donor clades
        // (e.g. nested Russian dolls P2 -> P3, or sequential cascade transitions)
        if candidate_donors.len() >= 2 {
            for i in 0..candidate_donors.len() {
                let d1 = candidate_donors[i].channel;
                if d1 == 0 {
                    continue;
                } // Ghost handled separately

                for j in (i + 1)..candidate_donors.len() {
                    let d2 = candidate_donors[j].channel;
                    if d2 == 0 {
                        continue;
                    }

                    let var_1 = cov_c[d1 * num_channels + d1].max(1e-12);
                    let var_2 = cov_c[d2 * num_channels + d2].max(1e-12);
                    let cov_12 = cov_c[d1 * num_channels + d2];

                    let diff_var = (var_1 + var_2 - 2.0 * cov_12).max(0.0);
                    let rho_12 = (cov_12 / (var_1 * var_2).sqrt()).clamp(-1.0, 1.0);
                    let opposition = 1.0 - rho_12;

                    let max_m = mean_a[d1].max(mean_a[d2]);
                    let att_sup = max_m / (max_m + km_channel);
                    let score_12 = diff_var * opposition * att_sup;

                    if score_12 >= 0.05 * best_score.max(1e-6)
                        && (z_swings[d1] >= k_crit_att || swings[d1] >= 0.25)
                        && (z_swings[d2] >= k_crit_att || swings[d2] >= 0.25)
                    {
                        let (h_sec, d_sec) = if mean_a[d1] >= mean_a[d2] {
                            (d1, d2)
                        } else {
                            (d2, d1)
                        };

                        let h_sec_k = h_sec.saturating_sub(1);
                        let d_sec_k = d_sec.saturating_sub(1);

                        pairs.push(TrajectoryParentalPair {
                            candidate_idx,
                            home_channel: h_sec,
                            donor_channel: d_sec,
                            home_landmark_idx: Some(h_sec_k),
                            donor_landmark_idx: Some(d_sec_k),
                            is_ghost_donor: false,
                            home_name: landmarks
                                .taxa
                                .get(h_sec_k)
                                .cloned()
                                .unwrap_or_else(|| "Home".to_string()),
                            donor_name: landmarks
                                .taxa
                                .get(d_sec_k)
                                .cloned()
                                .unwrap_or_else(|| "Donor".to_string()),
                            home_global_idx: landmarks.indices.get(h_sec_k).copied(),
                            donor_global_idx: landmarks.indices.get(d_sec_k).copied(),
                            swing_home: swings[h_sec],
                            swing_donor: swings[d_sec],
                        });
                    }
                }
            }
        }

        // Unsampled Outgroup / Ghost Cassette Dipole:
        // When a candidate is screened as a Ghost candidate (`allow_ghost`), ensure Channel 0 (Root)
        // is evaluated against the candidate's clonal clade medoid (or Home channel).
        if !pairs.iter().any(|p| p.is_ghost_donor)
            && allow_ghost
            && (z_swings[0] >= k_crit_att * s2 || swings[0] >= self.min_root_swing * s2)
        {
            if let Some(ghost_home_ch) = best_medoid_ch.or(Some(home_ch)) {
                let gh_k = ghost_home_ch.saturating_sub(1);
                pairs.push(TrajectoryParentalPair {
                    candidate_idx,
                    home_channel: ghost_home_ch,
                    donor_channel: 0,
                    home_landmark_idx: Some(gh_k),
                    donor_landmark_idx: None,
                    is_ghost_donor: true,
                    home_name: landmarks
                        .taxa
                        .get(gh_k)
                        .cloned()
                        .unwrap_or_else(|| "Home".to_string()),
                    donor_name: "Ghost".to_string(),
                    home_global_idx: landmarks.indices.get(gh_k).copied(),
                    donor_global_idx: None,
                    swing_home: swings[ghost_home_ch],
                    swing_donor: swings[0],
                });
            }
        }

        if pairs.is_empty() {
            pairs.push(TrajectoryParentalPair {
                candidate_idx,
                home_channel: home_ch,
                donor_channel: primary_donor_ch,
                home_landmark_idx: Some(home_k),
                donor_landmark_idx: Some(primary_donor_ch.saturating_sub(1)),
                is_ghost_donor: false,
                home_name,
                donor_name: landmarks
                    .taxa
                    .get(primary_donor_ch.saturating_sub(1))
                    .cloned()
                    .unwrap_or_else(|| "Donor".to_string()),
                home_global_idx: home_global,
                donor_global_idx: landmarks
                    .indices
                    .get(primary_donor_ch.saturating_sub(1))
                    .copied(),
                swing_home: swings[home_ch],
                swing_donor: swings[primary_donor_ch],
            });
        }

        pairs
    }

    /// Localized Attractive Work Integration (DIR-RHIZ-PAFF-012.0)
    ///
    /// `c_row` is the candidate's own momentum row of shape (L, K+1), indexed
    /// `u * num_channels + ch`.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve_reticulation_channels_localized(
        &self,
        candidate_idx: usize,
        u1: usize,
        u2: usize,
        c_row: &[f64],
        l: usize,
        landmarks: &LandmarkSet,
        aln: &crate::types::Alignment,
        preferred_home_ch: Option<usize>,
        preferred_donor_ch: Option<usize>,
        candidate_indices: &[usize],
        ghost_candidate_indices: &[usize],
    ) -> TrajectoryParentalPair {
        let k = landmarks.num_landmarks;
        let num_channels = k + 1;
        let (d_sister, d_clade) =
            crate::landmarks::compute_scale_invariant_distance_thresholds(aln);

        let u1_idx = (u1.saturating_sub(1)).min(l - 1);
        let u2_idx = (u2.saturating_sub(1)).min(l - 1);
        let (s_idx, e_idx) = if u1_idx <= u2_idx {
            (u1_idx, u2_idx)
        } else {
            (u2_idx, u1_idx)
        };

        let mut w_tract = vec![0.0f64; num_channels];
        let mut w_flank = vec![0.0f64; num_channels];

        for ch in 0..num_channels {
            if ch > 0 {
                let k_idx = ch - 1;
                if landmarks.indices[k_idx] == candidate_idx {
                    continue;
                }
            }

            let c_u1 = if s_idx > 0 {
                c_row[(s_idx - 1) * num_channels + ch]
            } else {
                0.0
            };
            let c_u2 = c_row[e_idx * num_channels + ch];
            w_tract[ch] = c_u2 - c_u1;

            let c_0 = 0.0;
            let c_l = c_row[(l - 1) * num_channels + ch];
            w_flank[ch] = (c_u1 - c_0) + (c_l - c_u2);
        }

        let mut contemporary_channels = Vec::new();
        for p in 0..k {
            let l_global = landmarks.indices[p];
            if l_global != candidate_idx {
                contemporary_channels.push(p + 1);
            }
        }
        if contemporary_channels.is_empty() {
            contemporary_channels.push(1);
        }

        // Test for Ghost Donor
        let mut is_ghost = false;
        let mut max_w_tr_cont = f64::NEG_INFINITY;
        for &ch in &contemporary_channels {
            if w_tract[ch] > max_w_tr_cont {
                max_w_tr_cont = w_tract[ch];
            }
        }

        if preferred_donor_ch == Some(0)
            || (preferred_donor_ch.is_none()
                && ghost_candidate_indices.contains(&candidate_idx)
                && w_tract[0] >= self.min_root_swing
                && w_tract[0] > max_w_tr_cont)
        {
            is_ghost = true;
        }

        if is_ghost {
            let mut home_candidates: Vec<usize> = contemporary_channels
                .iter()
                .copied()
                .filter(|&ch| {
                    let l_idx = landmarks.indices[ch - 1];
                    crate::landmarks::compute_pairwise_distance(aln, candidate_idx, l_idx)
                        >= d_sister
                })
                .collect();
            if home_candidates.is_empty() {
                home_candidates = contemporary_channels.clone();
            }
            let mut best_home_ch = home_candidates[0];
            let mut max_w_fl = f64::NEG_INFINITY;
            for &ch in &home_candidates {
                if w_flank[ch] > max_w_fl {
                    max_w_fl = w_flank[ch];
                    best_home_ch = ch;
                }
            }

            let mut final_home_ch = best_home_ch;
            if home_candidates.len() > 1 {
                let best_w = w_flank[best_home_ch];
                let mut runner_ups = Vec::new();
                for &ch in &home_candidates {
                    if (best_w - w_flank[ch]) < 0.25 * best_w.abs().max(1.0) {
                        runner_ups.push(ch);
                    }
                }
                if runner_ups.len() > 1 {
                    let cand_row = aln.row(candidate_idx);
                    let mut best_dist = f64::INFINITY;
                    for ch in runner_ups {
                        let k_idx = ch - 1;
                        let t_idx = landmarks.indices[k_idx];
                        let t_row = aln.row(t_idx);
                        let mut mismatches = 0usize;
                        let mut valid = 0usize;
                        for u in 0..l {
                            if u < s_idx || u > e_idx {
                                let rc = cand_row[u];
                                let rt = t_row[u];
                                if rc > 0 && rt > 0 {
                                    valid += 1;
                                    if rc != rt {
                                        mismatches += 1;
                                    }
                                }
                            }
                        }
                        let d = if valid > 0 {
                            (mismatches as f64) / (valid as f64)
                        } else {
                            1.0
                        };
                        if d < best_dist {
                            best_dist = d;
                            final_home_ch = ch;
                        }
                    }
                }
            }

            let home_k = final_home_ch.saturating_sub(1);
            let home_name = landmarks
                .taxa
                .get(home_k)
                .cloned()
                .unwrap_or_else(|| "Home".to_string());
            let home_global = landmarks.indices.get(home_k).copied();

            return TrajectoryParentalPair {
                candidate_idx,
                home_channel: final_home_ch,
                donor_channel: 0,
                home_landmark_idx: Some(home_k),
                donor_landmark_idx: None,
                is_ghost_donor: true,
                home_name,
                donor_name: "Ghost".to_string(),
                home_global_idx: home_global,
                donor_global_idx: None,
                swing_home: w_flank[final_home_ch],
                swing_donor: w_tract[0],
            };
        }

        // Contemporary Donor Resolution:
        // By physical definition of a reticulation dipole, an authentic donor landmark must exhibit
        // positive recoil contrast relative to the candidate (d_flank > d_tract).
        // Channels with d_flank <= d_tract (such as co-recombinant sisters or Home sisters) do not
        // diverge in the flanks and cannot act as external donor parents.
        let cand_row = aln.row(candidate_idx);
        let mut recoil_channels = Vec::new();
        for &ch in &contemporary_channels {
            let k_idx = ch - 1;
            let t_idx = landmarks.indices[k_idx];
            let t_row = aln.row(t_idx);
            let mut df = 0usize;
            let mut vf = 0usize;
            let mut dt = 0usize;
            let mut vt = 0usize;
            for u in 0..l {
                let rc = cand_row[u];
                let rt = t_row[u];
                if rc > 0 && rt > 0 {
                    if u < s_idx || u > e_idx {
                        vf += 1;
                        if rc != rt {
                            df += 1;
                        }
                    } else {
                        vt += 1;
                        if rc != rt {
                            dt += 1;
                        }
                    }
                }
            }
            let d_fl = if vf > 0 {
                (df as f64) / (vf as f64)
            } else {
                0.0
            };
            let d_tr = if vt > 0 {
                (dt as f64) / (vt as f64)
            } else {
                1.0
            };
            if d_fl > d_tr {
                recoil_channels.push((ch, d_fl));
            }
        }

        let mut non_candidate_recoil = Vec::new();
        for &(ch, d_fl) in &recoil_channels {
            let k_idx = ch - 1;
            let t_idx = landmarks.indices[k_idx];
            // Disqualify co-recombinant sisters sharing the candidate's home flank (d_fl < d_sister),
            // while allowing authentic donor parents from the opposite clade:
            let is_corecombinant_sister = candidate_indices.contains(&t_idx) && d_fl < d_sister;
            if !is_corecombinant_sister {
                non_candidate_recoil.push(ch);
            }
        }

        let eligible_donor_channels = if !non_candidate_recoil.is_empty() {
            non_candidate_recoil
        } else if !recoil_channels.is_empty() {
            recoil_channels.into_iter().map(|(c, _)| c).collect()
        } else {
            contemporary_channels.clone()
        };

        let mut max_w_tr_elig = f64::NEG_INFINITY;
        let mut best_elig_donor = eligible_donor_channels[0];
        for &ch in &eligible_donor_channels {
            if w_tract[ch] > max_w_tr_elig {
                max_w_tr_elig = w_tract[ch];
                best_elig_donor = ch;
            }
        }

        let best_donor_ch = if let Some(pref_d) = preferred_donor_ch {
            if pref_d > 0
                && eligible_donor_channels.contains(&pref_d)
                && w_tract[pref_d] >= 0.5 * max_w_tr_elig
            {
                pref_d
            } else {
                best_elig_donor
            }
        } else {
            best_elig_donor
        };

        // Intra-Clade Sister Disambiguation via normalized Hamming distance inside tract
        let mut final_donor_ch = best_donor_ch;
        if eligible_donor_channels.len() > 1 {
            let best_w = w_tract[best_donor_ch];
            let mut runner_ups = Vec::new();
            for &ch in &eligible_donor_channels {
                let is_sister = if let Some(pref_d) = preferred_donor_ch {
                    let k_pref = pref_d.saturating_sub(1);
                    let k_ch = ch.saturating_sub(1);
                    if let (Some(&t_pref), Some(&t_ch)) =
                        (landmarks.indices.get(k_pref), landmarks.indices.get(k_ch))
                    {
                        crate::landmarks::compute_pairwise_distance(aln, t_pref, t_ch) < d_clade
                    } else {
                        ch == pref_d
                    }
                } else {
                    true
                };

                if is_sister && (best_w - w_tract[ch]) < 0.25 * best_w.abs().max(1.0) {
                    runner_ups.push(ch);
                }
            }
            if runner_ups.len() > 1 {
                let cand_row = aln.row(candidate_idx);
                let mut best_dist = f64::INFINITY;
                for ch in runner_ups {
                    let k_idx = ch - 1;
                    let t_idx = landmarks.indices[k_idx];
                    let t_row = aln.row(t_idx);
                    let mut mismatches = 0usize;
                    let mut valid = 0usize;
                    for u in s_idx..=e_idx {
                        let rc = cand_row[u];
                        let rt = t_row[u];
                        if rc > 0 && rt > 0 {
                            valid += 1;
                            if rc != rt {
                                mismatches += 1;
                            }
                        }
                    }
                    let d = if valid > 0 {
                        (mismatches as f64) / (valid as f64)
                    } else {
                        1.0
                    };
                    if d < best_dist {
                        best_dist = d;
                        final_donor_ch = ch;
                    }
                }
            }
        }

        // Home Resolution in Flanks
        let mut home_candidates: Vec<usize> = contemporary_channels
            .iter()
            .cloned()
            .filter(|&ch| {
                if ch == final_donor_ch {
                    return false;
                }
                let l_idx = landmarks.indices[ch - 1];
                !candidate_indices.contains(&l_idx)
                    || !crate::landmarks::is_true_sister_pair(aln, candidate_idx, l_idx, d_sister)
            })
            .collect();
        if home_candidates.is_empty() {
            home_candidates = contemporary_channels
                .iter()
                .cloned()
                .filter(|&ch| ch != final_donor_ch)
                .collect();
        }
        if home_candidates.is_empty() {
            home_candidates = contemporary_channels.clone();
        }

        let mut max_w_fl = f64::NEG_INFINITY;
        let mut best_home_ch = home_candidates[0];
        for &ch in &home_candidates {
            if w_flank[ch] > max_w_fl {
                max_w_fl = w_flank[ch];
                best_home_ch = ch;
            }
        }

        if let Some(pref_h) = preferred_home_ch {
            if home_candidates.contains(&pref_h) && w_flank[pref_h] >= 0.5 * max_w_fl {
                best_home_ch = pref_h;
            }
        }

        let mut final_home_ch = best_home_ch;
        if home_candidates.len() > 1 {
            let best_w = w_flank[best_home_ch];
            let mut runner_ups = Vec::new();
            for &ch in &home_candidates {
                let is_sister = if let Some(pref_h) = preferred_home_ch {
                    let k_pref = pref_h.saturating_sub(1);
                    let k_ch = ch.saturating_sub(1);
                    if let (Some(&t_pref), Some(&t_ch)) =
                        (landmarks.indices.get(k_pref), landmarks.indices.get(k_ch))
                    {
                        crate::landmarks::compute_pairwise_distance(aln, t_pref, t_ch) < d_clade
                    } else {
                        ch == pref_h
                    }
                } else {
                    true
                };

                if is_sister && (best_w - w_flank[ch]) < 0.25 * best_w.abs().max(1.0) {
                    runner_ups.push(ch);
                }
            }
            if runner_ups.len() > 1 {
                let cand_row = aln.row(candidate_idx);
                let mut best_dist = f64::INFINITY;
                for ch in runner_ups {
                    let k_idx = ch - 1;
                    let t_idx = landmarks.indices[k_idx];
                    let t_row = aln.row(t_idx);
                    let mut mismatches = 0usize;
                    let mut valid = 0usize;
                    for u in 0..l {
                        if u < s_idx || u > e_idx {
                            let rc = cand_row[u];
                            let rt = t_row[u];
                            if rc > 0 && rt > 0 {
                                valid += 1;
                                if rc != rt {
                                    mismatches += 1;
                                }
                            }
                        }
                    }
                    let d = if valid > 0 {
                        (mismatches as f64) / (valid as f64)
                    } else {
                        1.0
                    };
                    if d < best_dist {
                        best_dist = d;
                        final_home_ch = ch;
                    }
                }
            }
        }

        // Polarity Verification: In tract, candidate must be closer to Donor than in flank
        let cand_row = aln.row(candidate_idx);
        let donor_k_temp = final_donor_ch - 1;
        let home_k_temp = final_home_ch - 1;
        let d_row = aln.row(landmarks.indices[donor_k_temp]);
        let h_row = aln.row(landmarks.indices[home_k_temp]);

        let mut d_tr_d = 0usize;
        let mut v_tr = 0usize;
        let mut d_fl_d = 0usize;
        let mut v_fl = 0usize;

        let mut d_tr_h = 0usize;
        let mut d_fl_h = 0usize;

        for u in 0..l {
            let rc = cand_row[u];
            let rd = d_row[u];
            let rh = h_row[u];
            if rc > 0 && rd > 0 && rh > 0 {
                if u >= s_idx && u <= e_idx {
                    v_tr += 1;
                    if rc != rd {
                        d_tr_d += 1;
                    }
                    if rc != rh {
                        d_tr_h += 1;
                    }
                } else {
                    v_fl += 1;
                    if rc != rd {
                        d_fl_d += 1;
                    }
                    if rc != rh {
                        d_fl_h += 1;
                    }
                }
            }
        }

        let dist_tr_d = if v_tr > 0 {
            (d_tr_d as f64) / (v_tr as f64)
        } else {
            0.0
        };
        let dist_fl_d = if v_fl > 0 {
            (d_fl_d as f64) / (v_fl as f64)
        } else {
            0.0
        };
        let dist_tr_h = if v_tr > 0 {
            (d_tr_h as f64) / (v_tr as f64)
        } else {
            0.0
        };
        let dist_fl_h = if v_fl > 0 {
            (d_fl_h as f64) / (v_fl as f64)
        } else {
            0.0
        };

        // If inverted polarity (closer to Donor in flank and closer to Home in tract), swap them
        if (dist_fl_d < dist_tr_d) && (dist_tr_h < dist_fl_h) {
            std::mem::swap(&mut final_home_ch, &mut final_donor_ch);
        }

        let home_k = final_home_ch - 1;
        let home_name = landmarks
            .taxa
            .get(home_k)
            .cloned()
            .unwrap_or_else(|| "Home".to_string());
        let home_global = landmarks.indices.get(home_k).copied();

        let donor_k = final_donor_ch - 1;
        let donor_name = landmarks
            .taxa
            .get(donor_k)
            .cloned()
            .unwrap_or_else(|| "Donor".to_string());
        let donor_global = landmarks.indices.get(donor_k).copied();

        TrajectoryParentalPair {
            candidate_idx,
            home_channel: final_home_ch,
            donor_channel: final_donor_ch,
            home_landmark_idx: Some(home_k),
            donor_landmark_idx: Some(donor_k),
            is_ghost_donor: false,
            home_name,
            donor_name,
            home_global_idx: home_global,
            donor_global_idx: donor_global,
            swing_home: w_flank[final_home_ch],
            swing_donor: w_tract[final_donor_ch],
        }
    }
}
