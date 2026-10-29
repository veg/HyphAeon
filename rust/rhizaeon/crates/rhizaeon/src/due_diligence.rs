use crate::attribution::TrajectoryParentalPair;
use crate::types::{Alignment, LandmarkSet, PhysicalTract, RecombinationEvent};

pub struct SequenceDueDiligence {
    pub poisson_floor: usize,
    pub alpha: f64,
    pub stringency: f64,
    pub compression_ratio: f64,
    pub d_sister: Option<f64>,
    pub d_clade: Option<f64>,
    log_factorials: Vec<f64>,
}

impl Default for SequenceDueDiligence {
    fn default() -> Self {
        Self::new(3, 0.05, 100_000)
    }
}

impl SequenceDueDiligence {
    pub fn new(poisson_floor: usize, alpha: f64, max_fact: usize) -> Self {
        let mut log_facts = Vec::with_capacity(max_fact + 1);
        log_facts.push(0.0); // 0!
        let mut sum_ln = 0.0;
        for i in 1..=max_fact {
            sum_ln += (i as f64).ln();
            log_facts.push(sum_ln);
        }
        Self {
            poisson_floor,
            alpha,
            stringency: 1.0,
            compression_ratio: 1.0,
            d_sister: None,
            d_clade: None,
            log_factorials: log_facts,
        }
    }

    pub fn with_stringency(mut self, stringency: f64) -> Self {
        self.stringency = stringency;
        self
    }

    pub fn with_compression_ratio(mut self, ratio: f64) -> Self {
        self.compression_ratio = ratio.clamp(0.005, 1.0);
        self
    }

    pub fn with_d_sister(mut self, d_sister: f64) -> Self {
        self.d_sister = Some(d_sister);
        self
    }

    pub fn with_d_clade(mut self, d_clade: f64) -> Self {
        self.d_clade = Some(d_clade);
        self
    }

    #[inline(always)]
    fn log_factorial(&self, n: usize) -> f64 {
        if n < self.log_factorials.len() {
            self.log_factorials[n]
        } else {
            // Stirling's approximation
            let nf = n as f64;
            0.5 * (2.0 * std::f64::consts::PI * nf).ln() + nf * (nf / std::f64::consts::E).ln()
        }
    }

    /// Exact Fisher 2x2 one-sided test for table:
    /// [a, b]
    /// [c, d]
    /// Testing if a/(a+b) > c/(c+d)
    pub fn fisher_exact_2x2(&self, a: usize, b: usize, c: usize, d: usize) -> f64 {
        let n = a + b + c + d;
        if n == 0 {
            return 1.0;
        }

        let r1 = a + b;
        let r2 = c + d;
        let c1 = a + c;
        let c2 = b + d;

        let log_denom = self.log_factorial(n);
        let log_margins = self.log_factorial(r1)
            + self.log_factorial(r2)
            + self.log_factorial(c1)
            + self.log_factorial(c2);

        let max_k = r1.min(c1);
        let mut p_sum = 0.0;

        // Sum hypergeometric probabilities for k >= a
        for k in a..=max_k {
            let cell_b = r1 - k;
            let cell_c = c1 - k;
            let cell_d = r2 - cell_c;

            let log_cells = self.log_factorial(k)
                + self.log_factorial(cell_b)
                + self.log_factorial(cell_c)
                + self.log_factorial(cell_d);

            let log_prob = log_margins - log_denom - log_cells;
            p_sum += log_prob.exp();
        }

        p_sum.min(1.0).max(0.0)
    }

    /// Static exact Fisher 2x2 one-sided test for external callers without a preallocated table
    pub fn fisher_exact_2x2_static(a: usize, b: usize, c: usize, d: usize) -> f64 {
        let n = a + b + c + d;
        if n == 0 {
            return 1.0;
        }
        fn lfact(k: usize) -> f64 {
            if k <= 1 {
                0.0
            } else if k <= 32 {
                (2..=k).map(|x| (x as f64).ln()).sum()
            } else {
                let kf = k as f64;
                0.5 * (2.0 * std::f64::consts::PI * kf).ln()
                    + kf * (kf / std::f64::consts::E).ln()
                    + 1.0 / (12.0 * kf)
            }
        }
        let r1 = a + b;
        let r2 = c + d;
        let c1 = a + c;
        let c2 = b + d;
        let log_denom = lfact(n);
        let log_margins = lfact(r1) + lfact(r2) + lfact(c1) + lfact(c2);
        let max_k = r1.min(c1);
        let mut p_sum = 0.0;
        for k in a..=max_k {
            let cell_b = r1 - k;
            let cell_c = c1 - k;
            let cell_d = r2 - cell_c;
            let log_cells = lfact(k) + lfact(cell_b) + lfact(cell_c) + lfact(cell_d);
            p_sum += (log_margins - log_denom - log_cells).exp();
        }
        p_sum.min(1.0).max(0.0)
    }

    /// Verifies physical tract via orthogonal sequence due diligence
    pub fn verify_tract(
        &self,
        tract: &PhysicalTract,
        parental: &TrajectoryParentalPair,
        aln: &Alignment,
        _landmarks: &LandmarkSet,
        rate_burst_indices: &[usize],
        candidate_indices: &[usize],
    ) -> Option<RecombinationEvent> {
        let s3 = self.stringency.max(0.1);
        let cand_idx = parental.candidate_idx;
        let cand_name = aln.taxa[cand_idx].clone();
        let l = aln.length;

        let u1 = tract.u1_discrete;
        let u2 = tract.u2_discrete;
        let tract_len = if u2 >= u1 { u2 - u1 + 1 } else { 0 };

        let base_min_len = if (s3 - 1.0).abs() < 1e-9 {
            20.0
        } else {
            (20.0 * s3).clamp(8.0, 150.0)
        };
        let min_len = if self.compression_ratio < 0.999 {
            ((base_min_len * self.compression_ratio).round() as usize).max(4)
        } else {
            base_min_len.round() as usize
        };
        if tract_len < min_len || !tract.is_physically_significant {
            return None;
        }

        // Bounded cassette check for ghost donors: reject full genome spans
        if parental.is_ghost_donor && (u1 <= 5 && u2 >= l - 5) {
            return None;
        }

        let home_global = parental.home_global_idx.unwrap_or(0);
        // Autapomorphic rate burst lineages cannot serve as Home parental reference
        if rate_burst_indices.contains(&home_global) {
            return None;
        }

        let (d_sister, d_clade) = match (self.d_sister, self.d_clade) {
            (Some(ds), Some(dc)) => (ds, dc),
            (Some(ds), None) => {
                let (_, dc) = crate::landmarks::compute_scale_invariant_distance_thresholds(aln);
                (ds, dc)
            }
            _ => crate::landmarks::compute_scale_invariant_distance_thresholds(aln),
        };

        // Precompute lineage peers at `d <= d_clade` once per tract
        let mut lineage_peers = Vec::new();
        if aln.num_taxa >= 6 {
            for t_idx in 0..aln.num_taxa {
                if crate::landmarks::compute_pairwise_distance(aln, cand_idx, t_idx) <= d_clade {
                    lineage_peers.push(t_idx);
                }
            }
        }

        // Co-recombinant sisters (d < d_sister) cannot serve as Home parental reference for ghost donors.
        // In addition, if `home_global` is in `candidate_indices`, it must come from a multi-isolate
        // cluster (`|L_C| >= 4`) and match the candidate on the un-recombined flank (`d_flank_home <= d_clade`).
        if parental.is_ghost_donor {
            let d_ch = crate::landmarks::compute_pairwise_distance(aln, cand_idx, home_global);
            if d_ch < d_sister {
                return None;
            }
            if candidate_indices.contains(&home_global) {
                let mut home_peers = 0usize;
                for t_idx in 0..aln.num_taxa {
                    if crate::landmarks::compute_pairwise_distance(aln, home_global, t_idx)
                        <= d_clade
                    {
                        home_peers += 1;
                    }
                }
                if home_peers < 4 {
                    return None;
                }
            }
        }

        let cand_row = aln.row(cand_idx);
        let home_row = aln.row(home_global);

        let (donor_row_opt, donor_name) = if parental.is_ghost_donor {
            (None, "Ghost".to_string())
        } else if let Some(dg) = parental.donor_global_idx {
            // Co-recombinant sister taxa or candidate's own sister cannot serve as donor
            let d_cd = crate::landmarks::compute_pairwise_distance(aln, cand_idx, dg);
            let d_hd = crate::landmarks::compute_pairwise_distance(aln, home_global, dg);
            if d_cd < d_sister || d_hd < d_sister {
                return None;
            }
            (Some(aln.row(dg)), parental.donor_name.clone())
        } else {
            (None, "Ghost".to_string())
        };

        let mut u1 = u1;
        let mut u2 = u2;

        // Boundary Refinement via Exact 2x2 Fisher Contingency Splits:
        // 1. Outward Donor-Carpet Extension: When a coarse landmark tract `[s0, e0)` stops inside
        //    a contiguous Donor-dominant run (`rc == rd != rh`), extend `s0` leftward and/or `e0`
        //    rightward to the peak of cumulative Donor excess `S(u) = sum [I(R_u = D_u != H_u) - I(R_u = H_u != D_u)]`
        //    provided the walk never dips into a Home gap (`diff >= -2`) and the extension segment
        //    forms a statistically significant 2x2 Fisher split against the remaining outer flank.
        // 2. Inward Prefix/Suffix Home-Reversion Trimming: When `[s0, e0)` over-extends into a
        //    flanking Home-dominant prefix or suffix (`rc == rh != rd`), trim at the global extremum
        //    of `S(u)` when significant at the Bonferroni threshold `alpha_split = 0.05 / L`.
        let alpha_split = 0.05 / (l.max(1) as f64);
        if let Some(donor_row) = donor_row_opt {
            let s0 = u1 - 1;
            let mut e0 = u2.min(l);

            // Step A1: Rightward contiguous Donor-carpet extension `[e0, u_ext_r]`
            if e0 < l {
                let mut r_d = 0usize;
                let mut r_h = 0usize;
                let mut max_r_diff = 0i64;
                let mut best_r = None;
                for u in e0..l {
                    let rc = cand_row[u];
                    let rh = home_row[u];
                    let rd = donor_row[u];
                    if rc > 0 && rh > 0 && rd > 0 && rh != rd {
                        if rc == rd {
                            r_d += 1;
                        } else if rc == rh {
                            r_h += 1;
                        }
                        let diff = (r_d as i64) - (r_h as i64);
                        if diff < -2 {
                            break;
                        }
                        if diff > 0 && diff >= max_r_diff {
                            max_r_diff = diff;
                            best_r = Some((u + 1, r_d, r_h));
                        }
                    }
                }
                if let Some((cand_e, d_ext, h_ext)) = best_r {
                    if d_ext >= 6 && d_ext >= h_ext * 3 {
                        let mut d_rem = 0usize;
                        let mut h_rem = 0usize;
                        for u in (0..s0).chain(cand_e..l) {
                            let rc = cand_row[u];
                            let rh = home_row[u];
                            let rd = donor_row[u];
                            if rc > 0 && rh > 0 && rd > 0 && rh != rd {
                                if rc == rd {
                                    d_rem += 1;
                                } else if rc == rh {
                                    h_rem += 1;
                                }
                            }
                        }
                        if h_rem > d_rem
                            && self.fisher_exact_2x2(d_ext, h_ext, d_rem, h_rem) < self.alpha
                        {
                            e0 = cand_e;
                        }
                    }
                }
            }

            // Step B: Inward Prefix/Suffix Home-Reversion Trimming
            if e0 > s0 + min_len {
                let span_len = e0 - s0;
                let mut cum_d = Vec::with_capacity(span_len);
                let mut cum_h = Vec::with_capacity(span_len);
                let mut d_acc = 0usize;
                let mut h_acc = 0usize;
                let mut max_diff = i64::MIN;
                let mut u_max = e0 - 1;

                for u in s0..e0 {
                    let rc = cand_row[u];
                    let rh = home_row[u];
                    let rd = donor_row[u];
                    if rc > 0 && rh > 0 && rd > 0 && rh != rd {
                        if rc == rd {
                            d_acc += 1;
                        } else if rc == rh {
                            h_acc += 1;
                        }
                    }
                    cum_d.push(d_acc);
                    cum_h.push(h_acc);
                    let diff = (d_acc as i64) - (h_acc as i64);
                    if diff >= max_diff {
                        max_diff = diff;
                        u_max = u;
                    }
                }

                // 1. Trailing 3' Home-reversion suffix `(u_max, e0)`
                let mut e_trimmed = e0;
                if u_max + 1 < e0 && (u_max + 1 - s0) >= min_len {
                    let idx_max = u_max - s0;
                    let d_core = cum_d[idx_max];
                    let h_core = cum_h[idx_max];
                    let d_tail = d_acc - d_core;
                    let h_tail = h_acc - h_core;
                    let allow_tail_trim = if e0 == l || h_acc >= d_acc {
                        h_tail > d_tail
                    } else {
                        h_tail >= 8 && h_tail >= d_tail * 4
                    };
                    if d_core > h_core
                        && allow_tail_trim
                        && self.fisher_exact_2x2(d_core, h_core, d_tail, h_tail) < alpha_split
                    {
                        e_trimmed = u_max + 1;
                    }
                }

                // 2. Leading 5' Home-reversion prefix `[s0, u_min]` when `h_acc >= d_acc` within `[s0, e_trimmed)`
                let mut s_trimmed = s0;
                let end_rel = e_trimmed - s0;
                if end_rel > 0 && cum_h[end_rel - 1] >= cum_d[end_rel - 1] {
                    let mut min_diff = 0i64;
                    let mut u_min_rel = None;
                    for idx in 0..end_rel {
                        let diff = (cum_d[idx] as i64) - (cum_h[idx] as i64);
                        if diff < min_diff {
                            min_diff = diff;
                            u_min_rel = Some(idx);
                        }
                    }
                    if let Some(m_rel) = u_min_rel {
                        let cand_s = s0 + m_rel + 1;
                        if cand_s < e_trimmed && (e_trimmed - cand_s) >= min_len {
                            let d_pref = cum_d[m_rel];
                            let h_pref = cum_h[m_rel];
                            let d_core = cum_d[end_rel - 1] - d_pref;
                            let h_core = cum_h[end_rel - 1] - h_pref;
                            if h_pref > d_pref
                                && d_core > h_core
                                && self.fisher_exact_2x2(d_core, h_core, d_pref, h_pref)
                                    < alpha_split
                            {
                                s_trimmed = cand_s;
                            }
                        }
                    }
                }

                u1 = s_trimmed + 1;
                u2 = e_trimmed;
            }
        }

        let tract_len = if u2 >= u1 { u2 - u1 + 1 } else { 0 };

        // Site counters
        let mut s_donor = 0usize;
        let mut s_home_tract = 0usize;
        let mut s_donor_flank = 0usize;
        let mut s_home_flank = 0usize;

        let mut mismatch_tract_donor = 0usize;
        let mut mismatch_tract_home = 0usize;
        let mut mismatch_tract_hd = 0usize;
        let mut valid_tract = 0usize;

        let mut mismatch_flank_donor = 0usize;
        let mut mismatch_flank_home = 0usize;
        let mut mismatch_flank_hd = 0usize;
        let mut valid_flank = 0usize;

        let mut valid_left_flank = 0usize;
        let mut valid_right_flank = 0usize;
        let mut info_left_flank = 0usize;
        let mut info_right_flank = 0usize;
        let mut s_triplet_priv = 0usize;

        let s_idx = u1 - 1;
        let e_idx = u2;

        for u in 0..l {
            let rc = cand_row[u];
            let rh = home_row[u];
            let in_tract = u >= s_idx && u < e_idx;

            if let Some(donor_row) = donor_row_opt {
                let rd = donor_row[u];
                if rc > 0 && rh > 0 && rd > 0 {
                    if in_tract {
                        valid_tract += 1;
                        if rc != rd {
                            mismatch_tract_donor += 1;
                        }
                        if rc != rh {
                            mismatch_tract_home += 1;
                        }
                        if rc != rd && rc != rh {
                            s_triplet_priv += 1;
                        }
                        // Informative site where Home and Donor differ
                        if rh != rd {
                            mismatch_tract_hd += 1;
                            if rc == rd {
                                s_donor += 1;
                            }
                            if rc == rh {
                                s_home_tract += 1;
                            }
                        }
                    } else {
                        valid_flank += 1;
                        if u < s_idx {
                            valid_left_flank += 1;
                        } else {
                            valid_right_flank += 1;
                        }
                        if rc != rd {
                            mismatch_flank_donor += 1;
                        }
                        if rc != rh {
                            mismatch_flank_home += 1;
                        }
                        if rh != rd {
                            mismatch_flank_hd += 1;
                            if u < s_idx {
                                info_left_flank += 1;
                            } else {
                                info_right_flank += 1;
                            }
                            if rc == rd {
                                s_donor_flank += 1;
                            }
                            if rc == rh {
                                s_home_flank += 1;
                            }
                        }
                    }
                }
            } else {
                // Ghost donor: test against home divergence
                if rc > 0 && rh > 0 {
                    if in_tract {
                        valid_tract += 1;
                        if rc != rh {
                            mismatch_tract_home += 1;
                            s_donor += 1;
                        }
                    } else {
                        valid_flank += 1;
                        if u < s_idx {
                            valid_left_flank += 1;
                            info_left_flank += 1;
                        } else {
                            valid_right_flank += 1;
                            info_right_flank += 1;
                        }
                        if rc != rh {
                            mismatch_flank_home += 1;
                        } else {
                            s_home_flank += 1;
                        }
                    }
                }
            }
        }

        // Effective Coverage Terminus Classification (Single Crossover vs. Cassette):
        // If either the 5' flank (`0..s_idx`) or 3' flank (`e_idx..l`) reaches the alignment
        // boundary or has zero informative sites (e.g. 5'- or 3'-truncated sequences), only one
        // physical breakpoint is observable, so the tract is a Single Crossover (`is_crossover = true`).
        // Conversely, if both flanks contain informative sites, both breakpoints are bounded inside
        // the alignment (`is_crossover = false`).
        let is_crossover = u1 == 1
            || u2 == l
            || valid_left_flank == 0
            || valid_right_flank == 0
            || info_left_flank == 0
            || info_right_flank == 0;

        // Parental Collapse & Partition Polarity Invariant:
        // In a genuine recombination event, Home and Donor are distinct parental lineages inside the
        // recombinant tract (`d_tract_hd > d_clade` and `d_tract_cd < d_tract_hd`), and Candidate must
        // genuinely switch from Home on the adjacent flanks to Donor inside the tract (`s_donor > s_home_tract`,
        // `s_donor * 2 > mismatch_tract_hd`, and `d_tract_cd < d_tract_ch`).
        if let Some(donor_row) = donor_row_opt {
            if valid_tract > 0 {
                let d_tract_hd = (mismatch_tract_hd as f64) / (valid_tract as f64);
                let d_tract_cd = (mismatch_tract_donor as f64) / (valid_tract as f64);
                let d_tract_ch = (mismatch_tract_home as f64) / (valid_tract as f64);
                let d_flank_hd = if valid_flank > 0 {
                    (mismatch_flank_hd as f64) / (valid_flank as f64)
                } else {
                    d_tract_hd
                };
                let d_flank_ch = if valid_flank > 0 {
                    (mismatch_flank_home as f64) / (valid_flank as f64)
                } else {
                    d_tract_ch
                };
                let d_flank_cd = if valid_flank > 0 {
                    (mismatch_flank_donor as f64) / (valid_flank as f64)
                } else {
                    d_tract_cd
                };
                if (d_tract_hd <= d_clade
                    && (d_tract_hd < 0.5 * d_flank_hd || d_tract_hd < 0.5 * d_tract_cd))
                    || (d_tract_hd < 0.40 * d_flank_hd
                        && d_tract_ch < 2.0 * d_flank_ch
                        && (d_flank_hd - d_tract_hd > 2.0 * (d_tract_ch - d_flank_ch)))
                    || d_tract_cd >= 0.90 * d_tract_hd
                    || s_donor * 2 <= mismatch_tract_hd
                    || s_donor <= s_home_tract
                    || d_tract_cd >= d_tract_ch
                    || (d_tract_cd > d_clade
                        && mismatch_tract_donor > s_donor
                        && s_triplet_priv * 2 >= s_donor)
                {
                    return None;
                }

                // If the pooled global flank prefers Donor over Home (`d_flank_cd <= d_flank_ch` or
                // `s_donor_flank >= s_home_flank`, as can occur in a multi-event mosaic when another
                // event occupies the opposite end of the genome), walk outward from `s_idx` (leftward)
                // and `e_idx` (rightward) to the peak of cumulative Home excess `h - d` and require
                // that at least one contiguous flanking segment adjacent to the tract prefers `Home`
                // and forms a statistically significant 2x2 Fisher split (`p < alpha_split`) against the tract.
                if valid_flank > 0 && (d_flank_cd <= d_flank_ch || s_donor_flank >= s_home_flank) {
                    if s_home_flank < self.poisson_floor || s_home_flank * 2 <= s_donor_flank {
                        return None;
                    }
                    let mut l_h_acc = 0usize;
                    let mut l_d_acc = 0usize;
                    let mut best_l_diff = 0i64;
                    let mut best_l = (0usize, 0usize);
                    for u in (0..s_idx).rev() {
                        let rc = cand_row[u];
                        let rh = home_row[u];
                        let rd = donor_row[u];
                        if rc > 0 && rh > 0 && rd > 0 && rh != rd {
                            if rc == rh {
                                l_h_acc += 1;
                            } else if rc == rd {
                                l_d_acc += 1;
                            }
                            let diff = (l_h_acc as i64) - (l_d_acc as i64);
                            if diff > best_l_diff {
                                best_l_diff = diff;
                                best_l = (l_h_acc, l_d_acc);
                            }
                        }
                    }

                    let mut r_h_acc = 0usize;
                    let mut r_d_acc = 0usize;
                    let mut best_r_diff = 0i64;
                    let mut best_r = (0usize, 0usize);
                    for u in e_idx..l {
                        let rc = cand_row[u];
                        let rh = home_row[u];
                        let rd = donor_row[u];
                        if rc > 0 && rh > 0 && rd > 0 && rh != rd {
                            if rc == rh {
                                r_h_acc += 1;
                            } else if rc == rd {
                                r_d_acc += 1;
                            }
                            let diff = (r_h_acc as i64) - (r_d_acc as i64);
                            if diff > best_r_diff {
                                best_r_diff = diff;
                                best_r = (r_h_acc, r_d_acc);
                            }
                        }
                    }

                    let left_valid = best_l.0 >= self.poisson_floor
                        && best_l.0 > best_l.1
                        && self.fisher_exact_2x2(s_donor, s_home_tract, best_l.1, best_l.0)
                            < alpha_split;
                    let right_valid = best_r.0 >= self.poisson_floor
                        && best_r.0 > best_r.1
                        && self.fisher_exact_2x2(s_donor, s_home_tract, best_r.1, best_r.0)
                            < alpha_split;
                    if !left_valid && !right_valid {
                        return None;
                    }
                }
            }
        }

        if parental.is_ghost_donor && candidate_indices.contains(&home_global) {
            let d_fl_h = if valid_flank > 0 {
                (mismatch_flank_home as f64) / (valid_flank as f64)
            } else {
                1.0
            };
            if d_fl_h > d_clade {
                return None;
            }
        }

        // Informative site floor check
        let eff_poisson_floor = if (s3 - 1.0).abs() < 1e-9 {
            self.poisson_floor
        } else {
            ((self.poisson_floor as f64 * s3).round() as usize).max(1)
        };
        if s_donor < eff_poisson_floor {
            return None;
        }

        // Lineage Majority Polarity Invariant:
        // When a candidate C belongs to a multi-isolate lineage cluster (|L_C| >= 4 at d <= d_clade)
        // and its putative Home parent H (or Donor D) is a mosaic member of the same lineage cluster,
        // an inverted mirror artifact occurs if C actually carries the majority clonal consensus of L_C
        // inside [s_idx, e_idx] while H (or D) is the minority mosaic recombinant inside [s_idx, e_idx].
        // A true recent recombinant departs from its flank-consistent lineage peers inside the tract; a
        // pure clonal strain remains close to its flank-consistent lineage peers inside the tract.
        if aln.num_taxa >= 6 {
            if lineage_peers.len() >= 4 {
                // Compute candidate's and Home's mean distance to all lineage peers inside the tract
                let mut sum_tr_cand = 0.0f64;
                let mut sum_tr_home = 0.0f64;
                let mut peer_count = 0usize;

                for &p_idx in &lineage_peers {
                    let p_row = aln.row(p_idx);
                    let mut mc = 0usize;
                    let mut mh = 0usize;
                    let mut v = 0usize;
                    for u in s_idx..e_idx {
                        let rp = p_row[u];
                        let rc = cand_row[u];
                        let rh = home_row[u];
                        if rp > 0 && rc > 0 && rh > 0 {
                            v += 1;
                            if rc != rp {
                                mc += 1;
                            }
                            if rh != rp {
                                mh += 1;
                            }
                        }
                    }
                    if v > 0 {
                        sum_tr_cand += (mc as f64) / (v as f64);
                        sum_tr_home += (mh as f64) / (v as f64);
                        peer_count += 1;
                    }
                }

                if peer_count >= 2 {
                    let mean_tr_cand = sum_tr_cand / (peer_count as f64);
                    let mean_tr_home = sum_tr_home / (peer_count as f64);
                    let d_ch =
                        crate::landmarks::compute_pairwise_distance(aln, cand_idx, home_global);
                    let d_fl_h = if valid_flank > 0 {
                        (mismatch_flank_home as f64) / (valid_flank as f64)
                    } else {
                        d_ch
                    };

                    // Compute left-flank and right-flank distance to Home for multi-way mosaics
                    let min_arm_sites = 10usize.min(aln.length / 6).max(3);
                    let mut m_left_h = 0usize;
                    let mut v_left_h = 0usize;
                    for u in 0..s_idx {
                        let rc = cand_row[u];
                        let rh = home_row[u];
                        if rc > 0 && rh > 0 {
                            v_left_h += 1;
                            if rc != rh {
                                m_left_h += 1;
                            }
                        }
                    }
                    let d_left_h = if v_left_h >= min_arm_sites {
                        (m_left_h as f64) / (v_left_h as f64)
                    } else {
                        f64::INFINITY
                    };

                    let mut m_right_h = 0usize;
                    let mut v_right_h = 0usize;
                    for u in e_idx..aln.length {
                        let rc = cand_row[u];
                        let rh = home_row[u];
                        if rc > 0 && rh > 0 {
                            v_right_h += 1;
                            if rc != rh {
                                m_right_h += 1;
                            }
                        }
                    }
                    let d_right_h = if v_right_h >= min_arm_sites {
                        (m_right_h as f64) / (v_right_h as f64)
                    } else {
                        f64::INFINITY
                    };

                    // 1. Home Clade Membership Invariant:
                    // When a candidate belongs to a multi-isolate lineage cluster (|L_C| >= 4 at d <= d_clade)
                    // in a pervasive-recombination cohort, its authentic Home backbone parent must match the
                    // candidate (<= d_clade) either globally, across the un-scrubbed flank, or across at least
                    // one flanking arm (left or right). This blocks pure parental clades (P1, P2, Surr) from
                    // citing foreign clades as Home while preserving multi-way crossovers and nested cassettes.
                    let all_screened_as_candidates =
                        candidate_indices.len() + rate_burst_indices.len() >= aln.num_taxa;
                    if all_screened_as_candidates
                        && d_ch > d_clade
                        && d_fl_h > d_clade
                        && d_left_h > d_clade
                        && d_right_h > d_clade
                    {
                        return None;
                    }

                    // 2. Inverted Home Mirror: Home is in Candidate's lineage cluster, and inside the tract
                    // Candidate is closer to the lineage cluster consensus than Home is (i.e. Home is the mosaic outlier!).
                    // Note: When Candidate genuinely converges to a sampled external Donor (`d_fl_d > d_clade`,
                    // `d_tr_hd > d_clade`, `d_tr_d < 0.25 * d_fl_d`) and `mean_tr_cand > d_sister`, `mean_tr_cand < mean_tr_home`
                    // can occur simply because >= 50% of `lineage_peers` are co-recombinants sharing the ancestral event.
                    // In that case, it is only an Inverted Home Mirror if `Home` itself jumped into an external clade
                    // or lost all clade support in the tract.
                    if d_ch <= d_clade
                        && (mean_tr_cand + d_sister < mean_tr_home
                            || (mean_tr_home >= 1.4 * mean_tr_cand && mean_tr_home > d_sister))
                    {
                        let d_tr_d = if valid_tract > 0 {
                            (mismatch_tract_donor as f64) / (valid_tract as f64)
                        } else {
                            1.0
                        };
                        let d_fl_d = if valid_flank > 0 {
                            (mismatch_flank_donor as f64) / (valid_flank as f64)
                        } else {
                            0.0
                        };
                        let d_tr_hd = if valid_tract > 0 {
                            (mismatch_tract_hd as f64) / (valid_tract as f64)
                        } else {
                            0.0
                        };
                        let cand_has_ext_donor_jump = donor_row_opt.is_some()
                            && d_fl_d > d_clade
                            && d_tr_hd > d_clade
                            && d_tr_d <= d_clade
                            && d_tr_d < 0.25 * d_fl_d
                            && mean_tr_cand > d_sister;

                        let is_inverted_home_mirror = if cand_has_ext_donor_jump {
                            let mut home_jumped_ext = false;
                            let mut home_clade_support = 0usize;
                            for x_idx in 0..aln.num_taxa {
                                if x_idx != home_global && x_idx != cand_idx {
                                    let d_glob_hx = crate::landmarks::compute_pairwise_distance(
                                        aln,
                                        home_global,
                                        x_idx,
                                    );
                                    let x_row = aln.row(x_idx);
                                    let mut m_hx = 0usize;
                                    let mut v_hx = 0usize;
                                    for u in s_idx..e_idx {
                                        let rh = home_row[u];
                                        let rx = x_row[u];
                                        if rh > 0 && rx > 0 {
                                            v_hx += 1;
                                            if rh != rx {
                                                m_hx += 1;
                                            }
                                        }
                                    }
                                    if v_hx > 0 {
                                        let d_tr_hx = (m_hx as f64) / (v_hx as f64);
                                        if d_glob_hx > d_clade {
                                            if d_tr_hx <= d_clade && d_tr_hx < 0.5 * d_glob_hx {
                                                home_jumped_ext = true;
                                                break;
                                            }
                                        } else if d_tr_hx <= d_clade {
                                            home_clade_support += 1;
                                        }
                                    }
                                }
                            }
                            home_jumped_ext || home_clade_support == 0
                        } else {
                            true
                        };

                        if is_inverted_home_mirror {
                            return None;
                        }
                    }

                    // 3. Intra-lineage Donor Mirror: Donor is in Candidate's own lineage cluster (d_cd <= d_clade),
                    // which represents an internal sub-lineage split rather than external introgression when external clades exist
                    if let Some(dg) = parental.donor_global_idx {
                        let has_external_clade = lineage_peers.len() < aln.num_taxa;
                        let d_cd = crate::landmarks::compute_pairwise_distance(aln, cand_idx, dg);
                        if has_external_clade && d_cd <= d_clade {
                            return None;
                        }
                    } else if parental.is_ghost_donor {
                        // 4. Ghost Donor Authenticity & Ghost-Mirror Polarity Test:
                        // (a) An event is only an unsampled Ghost donor if NO external contemporary landmark
                        //     matches the candidate inside the tract significantly better than Home does.
                        // (b) Conversely, if Home (`home_row`) matches an external landmark `l_idx` inside the
                        //     tract much closer than the candidate does (`d_tr_home_l <= d_clade` or
                        //     `< 0.5 * d_tr`), then Home is the mosaic recombinant that jumped into `l_idx`
                        //     and Candidate's Ghost call is an inverted parental mirror!
                        let mut best_ext_tr_d = f64::INFINITY;
                        let mut max_ext_match_frac = 0.0f64;
                        let mut home_matches_external_in_tract = false;
                        for &l_idx in &_landmarks.indices {
                            if l_idx != cand_idx
                                && crate::landmarks::compute_pairwise_distance(aln, cand_idx, l_idx)
                                    > d_clade
                            {
                                let l_row = aln.row(l_idx);
                                let mut m = 0usize;
                                let mut v = 0usize;
                                let mut m_hl = 0usize;
                                let mut v_hl = 0usize;
                                let mut m_diff = 0usize;
                                let mut v_diff = 0usize;
                                for u in s_idx..e_idx {
                                    let rc = cand_row[u];
                                    let rl = l_row[u];
                                    let rh = home_row[u];
                                    if rc > 0 && rl > 0 {
                                        v += 1;
                                        if rc != rl {
                                            m += 1;
                                        }
                                        if rh > 0 && rc != rh {
                                            v_diff += 1;
                                            if rl == rc {
                                                m_diff += 1;
                                            }
                                        }
                                    }
                                    if rh > 0 && rl > 0 {
                                        v_hl += 1;
                                        if rh != rl {
                                            m_hl += 1;
                                        }
                                    }
                                }
                                let d_tr = if v > 0 {
                                    (m as f64) / (v as f64)
                                } else {
                                    f64::INFINITY
                                };
                                let d_tr_hl = if v_hl > 0 {
                                    (m_hl as f64) / (v_hl as f64)
                                } else {
                                    f64::INFINITY
                                };
                                let d_global_hl = crate::landmarks::compute_pairwise_distance(
                                    aln,
                                    home_global,
                                    l_idx,
                                );
                                if d_global_hl > d_clade
                                    && (d_tr_hl <= d_clade
                                        || (d_tr_hl < 0.5 * d_tr && d_tr_hl < 0.5 * d_global_hl))
                                {
                                    home_matches_external_in_tract = true;
                                }
                                if d_tr < best_ext_tr_d {
                                    best_ext_tr_d = d_tr;
                                }
                                let d_tr_h_local = if valid_tract > 0 {
                                    (mismatch_tract_home as f64) / (valid_tract as f64)
                                } else {
                                    1.0
                                };
                                if v_diff > 0 && d_tr <= 0.85 * d_tr_h_local {
                                    let frac = (m_diff as f64) / (v_diff as f64);
                                    if frac > max_ext_match_frac {
                                        max_ext_match_frac = frac;
                                    }
                                }
                            }
                        }
                        let d_tr_h = if valid_tract > 0 {
                            (mismatch_tract_home as f64) / (valid_tract as f64)
                        } else {
                            1.0
                        };
                        if home_matches_external_in_tract
                            || best_ext_tr_d < 0.5 * d_tr_h
                            || max_ext_match_frac >= 0.50
                        {
                            return None;
                        }
                    }
                }
            }
        }

        // Private Autapomorphy Ratio Sieve (DIR-RHIZ-PAFF-012.8 Requirement 1.3)
        // Rejects lineage-specific substitution rate acceleration (Darren's heterotachy)
        // Checks both Candidate AND Home to prevent phantom mirror events from an accelerated sister.
        if aln.num_taxa >= 3 {
            let mut n_priv_cand = 0usize;
            let mut n_priv_home = 0usize;
            let mut n_total_mut = 0usize;

            for u in s_idx..e_idx {
                let rc = cand_row[u];
                let rh = home_row[u];
                if rc > 0 && rh > 0 && rc != rh {
                    n_total_mut += 1;
                    let mut is_cand_priv = true;
                    let mut is_home_priv = true;
                    for t_idx in 0..aln.num_taxa {
                        let rt = aln.row(t_idx)[u];
                        if t_idx != cand_idx && rt == rc {
                            is_cand_priv = false;
                        }
                        if t_idx != home_global && rt == rh {
                            is_home_priv = false;
                        }
                        if !is_cand_priv && !is_home_priv {
                            break;
                        }
                    }
                    if is_cand_priv {
                        n_priv_cand += 1;
                    }
                    if is_home_priv {
                        n_priv_home += 1;
                    }
                }
            }

            if n_total_mut >= 4 {
                let rho_priv_cand = (n_priv_cand as f64) / (n_total_mut as f64);
                let rho_priv_home = (n_priv_home as f64) / (n_total_mut as f64);
                if (rho_priv_cand >= 0.50 && n_priv_cand >= 3)
                    || (aln.num_taxa >= 6 && rho_priv_home >= 0.50 && n_priv_home >= 3)
                {
                    return None;
                }
            }

            if let Some(donor_row) = donor_row_opt {
                if let Some(dg) = parental.donor_global_idx {
                    let mut n_priv_donor = 0usize;
                    let mut n_total_donor = 0usize;
                    for u in s_idx..e_idx {
                        let rd = donor_row[u];
                        let rh = home_row[u];
                        if rd > 0 && rh > 0 && rd != rh {
                            n_total_donor += 1;
                            let mut is_donor_priv = true;
                            for t_idx in 0..aln.num_taxa {
                                if t_idx != dg && aln.row(t_idx)[u] == rd {
                                    is_donor_priv = false;
                                    break;
                                }
                            }
                            if is_donor_priv {
                                n_priv_donor += 1;
                            }
                        }
                    }
                    if n_total_donor >= 4 {
                        let rho_priv_donor = (n_priv_donor as f64) / (n_total_donor as f64);
                        if rho_priv_donor >= 0.50 && n_priv_donor >= 3 {
                            return None;
                        }
                    }
                }
            }
        }

        // Localized Autapomorphic Rate Acceleration Contrast Test on Putative Parental Lineages:
        // In genuine recombination, Home is a stable parental lineage that did not undergo
        // an episodic rate burst of private autapomorphic mutations specifically within this tract.
        // If Home's private autapomorphy density in the tract jumps significantly above its flank
        // baseline (r_priv_tract >= r_priv_flank + 0.020 with n_priv >= 4), Home is an accelerated
        // lineage whose terminal mutations create a phantom mirror effect on innocent taxa.
        // This check requires N >= 4.
        if aln.num_taxa >= 4 {
            let mut cand_sisters = vec![false; aln.num_taxa];
            cand_sisters[cand_idx] = true;
            for &p_idx in &lineage_peers {
                if p_idx < aln.num_taxa {
                    cand_sisters[p_idx] = true;
                }
            }
            for t_idx in 0..aln.num_taxa {
                if t_idx != cand_idx
                    && crate::landmarks::compute_pairwise_distance(aln, cand_idx, t_idx) < d_sister
                {
                    cand_sisters[t_idx] = true;
                }
            }

            let mut n_priv_home_tract = 0usize;
            let mut valid_home_tract = 0usize;
            for u in s_idx..e_idx {
                let rh = home_row[u];
                if rh > 0 {
                    valid_home_tract += 1;
                    let mut is_priv = true;
                    for t_idx in 0..aln.num_taxa {
                        if t_idx != home_global && !cand_sisters[t_idx] && aln.row(t_idx)[u] == rh {
                            is_priv = false;
                            break;
                        }
                    }
                    if is_priv {
                        n_priv_home_tract += 1;
                    }
                }
            }

            let mut n_priv_home_flank = 0usize;
            let mut valid_home_flank = 0usize;
            for u in 0..l {
                if u < s_idx || u >= e_idx {
                    let rh = home_row[u];
                    if rh > 0 {
                        valid_home_flank += 1;
                        let mut is_priv = true;
                        for t_idx in 0..aln.num_taxa {
                            if t_idx != home_global
                                && !cand_sisters[t_idx]
                                && aln.row(t_idx)[u] == rh
                            {
                                is_priv = false;
                                break;
                            }
                        }
                        if is_priv {
                            n_priv_home_flank += 1;
                        }
                    }
                }
            }

            if valid_home_tract >= 20 && valid_home_flank >= 20 {
                let n1 = valid_home_tract as f64;
                let n2 = valid_home_flank as f64;
                let x1 = n_priv_home_tract as f64;
                let x2 = n_priv_home_flank as f64;
                let p1 = x1 / n1;
                let p2 = x2 / n2;
                if p1 >= p2 + 0.020 && n_priv_home_tract >= 4 {
                    let p_pool = (x1 + x2) / (n1 + n2);
                    if p_pool > 0.0 && p_pool < 1.0 {
                        let se_pool = (p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)).sqrt();
                        let z_stat = (p1 - p2) / se_pool.max(1e-9);
                        let z_crit = crate::screener::normal_ppf(
                            1.0 - self.alpha / (aln.num_taxa as f64).max(1.0),
                        );
                        if z_stat >= z_crit.max(2.326) {
                            return None;
                        }
                    }
                }
            }

            // Same localized autapomorphy test for Donor (if sampled)
            if let Some(donor_row) = donor_row_opt {
                if let Some(dg) = parental.donor_global_idx {
                    let mut n_priv_donor_tract = 0usize;
                    let mut valid_donor_tract = 0usize;
                    for u in s_idx..e_idx {
                        let rd = donor_row[u];
                        if rd > 0 {
                            valid_donor_tract += 1;
                            let mut is_priv = true;
                            for t_idx in 0..aln.num_taxa {
                                if t_idx != dg && !cand_sisters[t_idx] && aln.row(t_idx)[u] == rd {
                                    is_priv = false;
                                    break;
                                }
                            }
                            if is_priv {
                                n_priv_donor_tract += 1;
                            }
                        }
                    }

                    let mut n_priv_donor_flank = 0usize;
                    let mut valid_donor_flank = 0usize;
                    for u in 0..l {
                        if u < s_idx || u >= e_idx {
                            let rd = donor_row[u];
                            if rd > 0 {
                                valid_donor_flank += 1;
                                let mut is_priv = true;
                                for t_idx in 0..aln.num_taxa {
                                    if t_idx != dg
                                        && !cand_sisters[t_idx]
                                        && aln.row(t_idx)[u] == rd
                                    {
                                        is_priv = false;
                                        break;
                                    }
                                }
                                if is_priv {
                                    n_priv_donor_flank += 1;
                                }
                            }
                        }
                    }

                    if valid_donor_tract >= 20 && valid_donor_flank >= 20 {
                        let n1 = valid_donor_tract as f64;
                        let n2 = valid_donor_flank as f64;
                        let x1 = n_priv_donor_tract as f64;
                        let x2 = n_priv_donor_flank as f64;
                        let p1 = x1 / n1;
                        let p2 = x2 / n2;
                        if p1 >= p2 + 0.020 && n_priv_donor_tract >= 4 {
                            let p_pool = (x1 + x2) / (n1 + n2);
                            if p_pool > 0.0 && p_pool < 1.0 {
                                let se_pool =
                                    (p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)).sqrt();
                                let z_stat = (p1 - p2) / se_pool.max(1e-9);
                                let z_crit = crate::screener::normal_ppf(
                                    1.0 - self.alpha / (aln.num_taxa as f64).max(1.0),
                                );
                                if z_stat >= z_crit.max(2.326) {
                                    return None;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Canonical Fisher exact test contingency table
        let (a, b, c, d) = if donor_row_opt.is_some() {
            (s_donor, s_home_tract, s_donor_flank, s_home_flank)
        } else {
            (
                mismatch_tract_home,
                valid_tract.saturating_sub(mismatch_tract_home),
                mismatch_flank_home,
                valid_flank.saturating_sub(mismatch_flank_home),
            )
        };

        // Mosaic Odds Ratio check:
        // In recombination, the odds of matching Donor vs Home in the tract must exceed the flank
        if donor_row_opt.is_some() {
            if b * c >= a * d {
                return None;
            }
        }

        let p_fisher = self.fisher_exact_2x2(a, b, c, d);

        // Bonferroni critical threshold over candidate triplets in alignment: alpha / binom(N, 3)
        let num_triplets =
            (((aln.num_taxa * aln.num_taxa.saturating_sub(1) * aln.num_taxa.saturating_sub(2)) / 6)
                .max(1)) as f64;
        let base_p_crit = self.alpha / num_triplets;
        let p_crit = if (s3 - 1.0).abs() < 1e-9 {
            base_p_crit
        } else if s3 < 1.0 {
            (base_p_crit * 10.0_f64.powf(4.0 * (1.0 - s3))).min(0.25)
        } else {
            base_p_crit * 10.0_f64.powf(-6.0 * (s3 - 1.0))
        };

        let d_tract_donor = if valid_tract > 0 {
            (mismatch_tract_donor as f64) / (valid_tract as f64)
        } else {
            0.0
        };
        let d_tract_home = if valid_tract > 0 {
            (mismatch_tract_home as f64) / (valid_tract as f64)
        } else {
            0.0
        };
        let d_flank_donor = if valid_flank > 0 {
            (mismatch_flank_donor as f64) / (valid_flank as f64)
        } else {
            0.0
        };
        let d_flank_home = if valid_flank > 0 {
            (mismatch_flank_home as f64) / (valid_flank as f64)
        } else {
            0.0
        };

        // Relative recoil check (when donor is sampled)
        if !parental.is_ghost_donor {
            let delta_donor_recoil = d_flank_donor - d_tract_donor;
            let delta_home_recoil = d_tract_home - d_flank_home;
            let delta_net_recoil = delta_donor_recoil + delta_home_recoil;

            if delta_donor_recoil <= 0.0 {
                return None;
            }

            let n_tr = valid_tract.max(1) as f64;
            let n_fl = valid_flank.max(1) as f64;
            let se_recoil = ((d_tract_donor * (1.0 - d_tract_donor) / n_tr)
                + (d_flank_donor * (1.0 - d_flank_donor) / n_fl))
                .sqrt();
            let z_crit_recoil_base = crate::screener::normal_ppf(1.0 - self.alpha);
            let z_crit_recoil = z_crit_recoil_base * s3;

            // In genuine mosaicism, candidate departs from home parent in tract
            if delta_home_recoil < -(z_crit_recoil_base * se_recoil / s3).max(0.0) {
                return None;
            }

            if delta_net_recoil < (z_crit_recoil * se_recoil).max(0.0) {
                return None;
            }

            let ratio_donor = (d_tract_donor + 1e-4) / (d_flank_donor + 1e-4);
            let ratio_home = (d_tract_home + 1e-4) / (d_flank_home + 1e-4);
            if ratio_donor >= ratio_home {
                return None;
            }

            // Buneman 4-Point Metric and Topological Invariant (DIR-RHIZ-PAFF-013.4 / 013.7)
            if aln.num_taxa >= 4 {
                let donor_global = parental.donor_global_idx.unwrap();
                let mut best_outgroup = None;
                let mut max_out_div = -1.0;

                let d_hd =
                    crate::landmarks::compute_pairwise_distance(aln, home_global, donor_global);
                for o_idx in 0..aln.num_taxa {
                    if o_idx != cand_idx
                        && o_idx != home_global
                        && o_idx != donor_global
                        && !candidate_indices.contains(&o_idx)
                        && !rate_burst_indices.contains(&o_idx)
                    {
                        let d_oh =
                            crate::landmarks::compute_pairwise_distance(aln, o_idx, home_global);
                        let d_od =
                            crate::landmarks::compute_pairwise_distance(aln, o_idx, donor_global);
                        // An authentic phylogenetic outgroup must be external to the {H, D} split:
                        if d_oh >= 0.40 * d_hd && d_od >= 0.40 * d_hd {
                            let sum_d = d_oh + d_od;
                            if sum_d > max_out_div {
                                max_out_div = sum_d;
                                best_outgroup = Some(o_idx);
                            }
                        }
                    }
                }

                if let Some(o_idx) = best_outgroup {
                    let o_row = aln.row(o_idx);
                    let mut d_tr_oh = 0usize;
                    let mut d_tr_od = 0usize;
                    let mut d_tr_ro = 0usize;
                    let mut d_tr_hd = 0usize;
                    let mut v_tr_o = 0usize;

                    let mut d_fl_oh = 0usize;
                    let mut d_fl_od = 0usize;
                    let mut d_fl_ro = 0usize;
                    let mut d_fl_hd = 0usize;
                    let mut v_fl_o = 0usize;

                    for u in 0..l {
                        let ro = o_row[u];
                        let rh = home_row[u];
                        let rd = donor_row_opt.unwrap()[u];
                        let rc = cand_row[u];

                        if ro > 0 && rh > 0 && rd > 0 && rc > 0 {
                            if u >= s_idx && u < e_idx {
                                v_tr_o += 1;
                                if ro != rh {
                                    d_tr_oh += 1;
                                }
                                if ro != rd {
                                    d_tr_od += 1;
                                }
                                if rc != ro {
                                    d_tr_ro += 1;
                                }
                                if rh != rd {
                                    d_tr_hd += 1;
                                }
                            } else {
                                v_fl_o += 1;
                                if ro != rh {
                                    d_fl_oh += 1;
                                }
                                if ro != rd {
                                    d_fl_od += 1;
                                }
                                if rc != ro {
                                    d_fl_ro += 1;
                                }
                                if rh != rd {
                                    d_fl_hd += 1;
                                }
                            }
                        }
                    }

                    if v_tr_o > 0 && v_fl_o > 0 {
                        let tr_oh = (d_tr_oh as f64) / (v_tr_o as f64);
                        let tr_od = (d_tr_od as f64) / (v_tr_o as f64);
                        let tr_ro = (d_tr_ro as f64) / (v_tr_o as f64);
                        let tr_hd = (d_tr_hd as f64) / (v_tr_o as f64);

                        let fl_oh = (d_fl_oh as f64) / (v_fl_o as f64);
                        let fl_od = (d_fl_od as f64) / (v_fl_o as f64);
                        let fl_ro = (d_fl_ro as f64) / (v_fl_o as f64);
                        let fl_hd = (d_fl_hd as f64) / (v_fl_o as f64);

                        // Unrooted quartet topology check (DIR-RHIZ-PAFF-012.7):
                        // S1 = D(R, H) + D(D, O)  [Split 0: (RH)|(DO)]
                        // S2 = D(R, D) + D(H, O)  [Split 1: (RD)|(HO)]
                        // S3 = D(R, O) + D(H, D)  [Split 2: (RO)|(HD)]
                        let s1_tr = d_tract_home + tr_od;
                        let s2_tr = d_tract_donor + tr_oh;
                        let s3_tr = tr_ro + tr_hd;

                        let s1_fl = d_flank_home + fl_od;
                        let s2_fl = d_flank_donor + fl_oh;
                        let s3_fl = fl_ro + fl_hd;

                        let z_crit_q_base = crate::screener::normal_ppf(1.0 - self.alpha);

                        // Standard error of quartet split differences under neutral binomial sampling
                        let se_fl = ((d_flank_home * (1.0 - d_flank_home) / (v_fl_o as f64))
                            + (d_flank_donor * (1.0 - d_flank_donor) / (v_fl_o as f64))
                            + (fl_oh * (1.0 - fl_oh) / (v_fl_o as f64))
                            + (fl_od * (1.0 - fl_od) / (v_fl_o as f64)))
                            .sqrt();

                        let se_tr = ((d_tract_home * (1.0 - d_tract_home) / (v_tr_o as f64))
                            + (d_tract_donor * (1.0 - d_tract_donor) / (v_tr_o as f64))
                            + (tr_oh * (1.0 - tr_oh) / (v_tr_o as f64))
                            + (tr_od * (1.0 - tr_od) / (v_tr_o as f64)))
                            .sqrt();

                        // Flank topology: check if internal branch is statistically distinguishable from a star tree
                        let mut sums_fl = [(s1_fl, 0i32), (s2_fl, 1i32), (s3_fl, 2i32)];
                        sums_fl.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                        let delta_s_fl = sums_fl[1].0 - sums_fl[0].0;
                        let topo_fl = if delta_s_fl > z_crit_q_base * se_fl {
                            sums_fl[0].1
                        } else {
                            -1 // Unresolved polytomy / star tree
                        };

                        // Tract topology: check if internal branch is statistically distinguishable from a star tree
                        let mut sums_tr = [(s1_tr, 0i32), (s2_tr, 1i32), (s3_tr, 2i32)];
                        sums_tr.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                        let delta_s_tr = sums_tr[1].0 - sums_tr[0].0;
                        let topo_tr = if delta_s_tr > z_crit_q_base * se_tr {
                            sums_tr[0].1
                        } else {
                            -1 // Unresolved polytomy / star tree
                        };

                        // If both flanks and tract have the SAME resolved split, reject (no recombination)
                        if topo_tr >= 0 && topo_fl >= 0 && topo_tr == topo_fl {
                            return None;
                        }

                        // Split 0 in tract: ((R, H) | (D, O))
                        // Rejects ONLY if R is actually closer to Home than to Donor in the tract (i.e. R stayed in Home).
                        // If R is closer to Donor (d_tract_donor < d_tract_home), then R is in the Donor clade,
                        // and Split 0 is merely an artifact of D and O being sisters in the Donor clade.
                        if topo_tr == 0 && d_tract_home <= d_tract_donor {
                            return None;
                        }

                        // Differential Buneman Polarization:
                        let delta_q_flank = (d_flank_donor + fl_oh) - (d_flank_home + fl_od);
                        let delta_q_tract = (d_tract_donor + tr_oh) - (d_tract_home + tr_od);
                        let delta_q_diff = delta_q_tract - delta_q_flank;

                        // Under terminal heterotachy / rate burst, delta_q_diff >= 0.0
                        if delta_q_diff > (z_crit_q_base * se_tr / s3).max(0.0) {
                            return None;
                        }

                        // Topological Double Difference:
                        let delta_flank = (d_flank_home - d_flank_donor) - (fl_oh - fl_od);
                        let delta_tract = (d_tract_home - d_tract_donor) - (tr_oh - tr_od);
                        let delta_topo = (delta_tract - delta_flank).abs();

                        let se_topo = ((d_tract_home * (1.0 - d_tract_home) / (v_tr_o as f64))
                            + (d_flank_home * (1.0 - d_flank_home) / (v_fl_o as f64)))
                            .sqrt();
                        let resolution_floor = (1.0 / ((v_tr_o.min(v_fl_o)) as f64)) * s3;
                        let z_crit_topo = crate::screener::normal_ppf(1.0 - self.alpha) * s3;
                        let thresh = (z_crit_topo * se_topo).max(resolution_floor);

                        if delta_topo < thresh {
                            return None;
                        }
                    }
                }
            }
        } else {
            // For ghost donor, tract divergence to home must strictly exceed flank divergence to home
            if d_tract_home <= d_flank_home {
                return None;
            }
        }

        let is_verified = p_fisher <= p_crit && s_donor >= eff_poisson_floor;

        Some(RecombinationEvent {
            event_id: 0,
            isolates: vec![cand_name.clone()],
            candidate_idx: cand_idx,
            candidate_name: cand_name,
            home_idx: parental.home_global_idx,
            home_name: parental.home_name.clone(),
            donor_idx: parental.donor_global_idx,
            donor_name,
            is_ghost_donor: parental.is_ghost_donor,
            is_crossover,
            u1,
            u2,
            u1_continuous: u1 as f64,
            u2_continuous: u2 as f64,
            tract_length: tract_len,
            s_informative: s_donor,
            p_fisher,
            z_phys: tract.z_phys,
            d_flank_home,
            d_flank_donor,
            d_tract_home,
            d_tract_donor,
            is_verified,
        })
    }

    /// Adjudicates reticulation graph to eliminate reflected mosaic illusions (DIR-RHIZ-PAFF-013.1)
    pub fn adjudicate_reticulation_graph(
        &self,
        events: Vec<RecombinationEvent>,
        _aln: &Alignment,
        _screening: &crate::types::ScreeningResult,
    ) -> Vec<RecombinationEvent> {
        if events.len() <= 1 {
            return events;
        }

        let n_events = events.len();
        let mut pruned = vec![false; n_events];
        let (d_sister, d_clade) =
            crate::landmarks::compute_scale_invariant_distance_thresholds(_aln);

        // Pass 1: Mutual 2-Cycle Adjudication (C_1 <-> C_2)
        // Resolves cases where candidate C1 cites C2 as parent, and candidate C2 cites C1 as parent.
        for i in 0..n_events {
            if pruned[i] {
                continue;
            }
            let e1 = &events[i];
            let c1_name = &e1.isolates[0];

            for j in (i + 1)..n_events {
                if pruned[j] {
                    continue;
                }
                let e2 = &events[j];
                let c2_name = &e2.isolates[0];

                if c1_name == c2_name {
                    continue;
                }

                // Co-recombinant sisters share the same Home parent and Donor parent.
                // They descend from the same ancestral event and NEVER form a mutual 2-cycle reflection.
                // However, if one candidate cites the other as Home or Donor, they are in a mutual cycle relationship!
                let c1_is_h2 = (e2.home_name == *c1_name)
                    || (e2.home_idx.map_or(false, |h| {
                        crate::landmarks::compute_pairwise_distance(_aln, h, e1.candidate_idx)
                            < d_clade
                    }));
                let c2_is_h1 = (e1.home_name == *c2_name)
                    || (e1.home_idx.map_or(false, |h| {
                        crate::landmarks::compute_pairwise_distance(_aln, h, e2.candidate_idx)
                            < d_clade
                    }));
                let c1_is_d2 = (e2.donor_name == *c1_name)
                    || (e2.donor_idx.map_or(false, |d| {
                        crate::landmarks::compute_pairwise_distance(_aln, d, e1.candidate_idx)
                            < d_clade
                    }));
                let c2_is_d1 = (e1.donor_name == *c2_name)
                    || (e1.donor_idx.map_or(false, |d| {
                        crate::landmarks::compute_pairwise_distance(_aln, d, e2.candidate_idx)
                            < d_clade
                    }));

                if !c1_is_h2 && !c2_is_h1 && !c1_is_d2 && !c2_is_d1 {
                    let d_inter = match (e1.home_idx, e1.donor_idx) {
                        (Some(h), Some(d)) => {
                            crate::landmarks::compute_pairwise_distance(_aln, h, d)
                        }
                        _ => 0.05,
                    };
                    let same_home = e1.home_name == e2.home_name
                        || match (e1.home_idx, e2.home_idx) {
                            (Some(h1), Some(h2)) => {
                                d_inter > d_sister
                                    && crate::landmarks::compute_pairwise_distance(_aln, h1, h2)
                                        < 0.5 * d_inter
                            }
                            _ => false,
                        };
                    let same_donor = e1.donor_name == e2.donor_name
                        || match (e1.donor_idx, e2.donor_idx) {
                            (Some(d1), Some(d2)) => {
                                d_inter > d_sister
                                    && crate::landmarks::compute_pairwise_distance(_aln, d1, d2)
                                        < 0.5 * d_inter
                            }
                            _ => false,
                        };
                    if same_home && same_donor {
                        continue; // Co-recombinant sisters; preserve for group_ancestral_events
                    }
                }

                // Mutual cycle adjudication requires the two events to be homologous in genomic space:
                // either overlapping tracts (for cassettes/conversions) or complementary partitions (for crossovers).
                let ov_start = e1.u1.max(e2.u1);
                let ov_end = e1.u2.min(e2.u2);
                let overlap = if ov_end >= ov_start {
                    ov_end - ov_start + 1
                } else {
                    0
                };
                let min_len =
                    (e1.u2.saturating_sub(e1.u1) + 1).min(e2.u2.saturating_sub(e2.u1) + 1);

                let is_overlapping = (overlap as f64) >= 0.25 * (min_len as f64);
                let is_complementary_crossover = (e1.is_crossover
                    || e1.u1 <= 5
                    || e1.u2 >= _aln.length.saturating_sub(5))
                    && (e2.is_crossover || e2.u1 <= 5 || e2.u2 >= _aln.length.saturating_sub(5))
                    && (e1.u2.abs_diff(e2.u1) <= 10 || e2.u2.abs_diff(e1.u1) <= 10);

                if !is_overlapping && !is_complementary_crossover {
                    continue;
                }

                // Check if e1 cites c2 (as Home or Donor) AND e2 cites c1 (as Home or Donor):
                // Do NOT use broad `< d_clade` proximity to the other candidate's Home or Donor here,
                // because if e1 (pop1 -> pop3) cites pop3.22 as Donor and e2 (pop3 -> pop2) cites pop3.3 as Home,
                // pop3.22 and pop3.3 are both in pop3 (< d_clade), which would falsely link e1 and e2 into a 2-cycle!
                // A true mutual 2-cycle requires e1 to cite c2 (or c2's sister clone < d_sister) AND e2 to cite c1 (or c1's sister clone < d_sister).
                let e1_cites_c2 = (e1.home_name == *c2_name)
                    || (e1.donor_name == *c2_name)
                    || e2.isolates.contains(&e1.home_name)
                    || e2.isolates.contains(&e1.donor_name)
                    || (e1.home_idx.map_or(false, |h| {
                        crate::landmarks::compute_pairwise_distance(_aln, h, e2.candidate_idx)
                            < d_sister
                    }))
                    || (e1.donor_idx.map_or(false, |d| {
                        crate::landmarks::compute_pairwise_distance(_aln, d, e2.candidate_idx)
                            < d_sister
                    }));

                let e2_cites_c1 = (e2.home_name == *c1_name)
                    || (e2.donor_name == *c1_name)
                    || e1.isolates.contains(&e2.home_name)
                    || e1.isolates.contains(&e2.donor_name)
                    || (e2.home_idx.map_or(false, |h| {
                        crate::landmarks::compute_pairwise_distance(_aln, h, e1.candidate_idx)
                            < d_sister
                    }))
                    || (e2.donor_idx.map_or(false, |d| {
                        crate::landmarks::compute_pairwise_distance(_aln, d, e1.candidate_idx)
                            < d_sister
                    }));

                if e1_cites_c2 && e2_cites_c1 {
                    // Mutual 2-cycle detected!
                    // Check external donor authenticity:
                    // A candidate whose donor is internal (close to candidate or home) does not have an authentic external donor
                    let e1_donor_internal = e1.donor_idx.map_or(false, |d| {
                        crate::landmarks::compute_pairwise_distance(_aln, d, e1.candidate_idx)
                            < d_clade
                            || e1.home_idx.map_or(false, |h| {
                                crate::landmarks::compute_pairwise_distance(_aln, d, h) < d_clade
                            })
                    });
                    let e2_donor_internal = e2.donor_idx.map_or(false, |d| {
                        crate::landmarks::compute_pairwise_distance(_aln, d, e2.candidate_idx)
                            < d_clade
                            || e2.home_idx.map_or(false, |h| {
                                crate::landmarks::compute_pairwise_distance(_aln, d, h) < d_clade
                            })
                    });

                    // If one event is a Ghost donor introgression (donor_idx.is_none()) and the other event
                    // cites the Ghost recombinant as its Home parent (a phantom mirror event where a clonal
                    // parent appears to depart from the Ghost recombinant inside the Ghost tract), do NOT
                    // treat the phantom mirror event's contemporary donor as superior to the Ghost donor:
                    let e1_cites_c2_as_home = (e1.home_name == *c2_name)
                        || e2.isolates.contains(&e1.home_name)
                        || e1.home_idx.map_or(false, |h| {
                            crate::landmarks::compute_pairwise_distance(_aln, h, e2.candidate_idx)
                                < d_sister
                        });
                    let e2_cites_c1_as_home = (e2.home_name == *c1_name)
                        || e1.isolates.contains(&e2.home_name)
                        || e2.home_idx.map_or(false, |h| {
                            crate::landmarks::compute_pairwise_distance(_aln, h, e1.candidate_idx)
                                < d_sister
                        });

                    let e1_is_ghost_home_mirror = e2.is_ghost_donor
                        && e1_cites_c2_as_home
                        && !e1.is_ghost_donor
                        && e1.d_tract_donor > d_clade
                        && e1.d_tract_donor >= 0.75 * e1.d_flank_donor;
                    let e2_is_ghost_home_mirror = e1.is_ghost_donor
                        && e2_cites_c1_as_home
                        && !e2.is_ghost_donor
                        && e2.d_tract_donor > d_clade
                        && e2.d_tract_donor >= 0.75 * e2.d_flank_donor;

                    let e1_has_ext_donor = e1.donor_idx.is_some()
                        && e1.donor_name != *c2_name
                        && !e1_donor_internal
                        && !e1_is_ghost_home_mirror;
                    let e2_has_ext_donor = e2.donor_idx.is_some()
                        && e2.donor_name != *c1_name
                        && !e2_donor_internal
                        && !e2_is_ghost_home_mirror;

                    if e2_is_ghost_home_mirror || (e1_has_ext_donor && !e2_has_ext_donor) {
                        // e1 has authentic external donor (or e2 is a stationary mirror of Ghost recombinant e1)
                        pruned[j] = true;
                    } else if e1_is_ghost_home_mirror || (e2_has_ext_donor && !e1_has_ext_donor) {
                        // e2 has authentic external donor (or e1 is a stationary mirror of Ghost recombinant e2)
                        pruned[i] = true;
                        break;
                    } else {
                        // Compute physical transition contrast and home backbone disruption for each candidate:
                        // A true recombinant child undergoes a large home disruption (d_tract_home >> d_flank_home),
                        // moving away from its home clade into the donor clade during the tract.
                        // A stationary clonal parent never leaves its home clade (d_tract_home ~ d_flank_home).
                        let delta_h1 = (e1.d_tract_home - e1.d_flank_home).max(0.0);
                        let delta_h2 = (e2.d_tract_home - e2.d_flank_home).max(0.0);
                        let delta_d1 = (e1.d_flank_donor - e1.d_tract_donor).max(0.0);
                        let delta_d2 = (e2.d_flank_donor - e2.d_tract_donor).max(0.0);
                        let trans1 = delta_h1 + delta_d1;
                        let trans2 = delta_h2 + delta_d2;

                        if (trans1 - trans2).abs() > 1e-4 {
                            if trans1 > trans2 {
                                pruned[j] = true;
                            } else {
                                pruned[i] = true;
                                break;
                            }
                        } else if (delta_h1 - delta_h2).abs() > 1e-4 {
                            if delta_h1 > delta_h2 {
                                pruned[j] = true;
                            } else {
                                pruned[i] = true;
                                break;
                            }
                        } else if e1.p_fisher != e2.p_fisher {
                            if e1.p_fisher < e2.p_fisher {
                                pruned[j] = true;
                            } else {
                                pruned[i] = true;
                                break;
                            }
                        } else if e1.z_phys != e2.z_phys {
                            if e1.z_phys > e2.z_phys {
                                pruned[j] = true;
                            } else {
                                pruned[i] = true;
                                break;
                            }
                        } else if e1.s_informative >= e2.s_informative {
                            pruned[j] = true;
                        } else {
                            pruned[i] = true;
                            break;
                        }
                    }
                }
            }
        }

        // Pass 2: Asymmetric Reflected Parental Adjudication
        // Resolves cases where candidate e2 cites candidate e1 as its Home parent
        // across an overlapping tract (causing e2 to appear to depart from e1 when in fact e1 departed!),
        // OR where e1 and e2 are in the same clade and one cites the other as Donor.
        for i in 0..n_events {
            if pruned[i] {
                continue;
            }
            let e1 = &events[i];
            let c1_name = &e1.isolates[0];

            for j in 0..n_events {
                if i == j || pruned[j] || pruned[i] {
                    continue;
                }
                let e2 = &events[j];
                let c2_name = &e2.isolates[0];

                if c1_name == c2_name {
                    continue;
                }

                // Check tract overlap
                let ov_start = e1.u1.max(e2.u1);
                let ov_end = e1.u2.min(e2.u2);
                let overlap = if ov_end >= ov_start {
                    ov_end - ov_start + 1
                } else {
                    0
                };
                let min_len =
                    (e1.u2.saturating_sub(e1.u1) + 1).min(e2.u2.saturating_sub(e2.u1) + 1);
                let union_span = (e1.u2.max(e2.u2) - e1.u1.min(e2.u1) + 1) as f64;
                let iou = (overlap as f64) / union_span.max(1.0);
                let frac = (overlap as f64) / (min_len as f64).max(1.0);

                let is_overlapping = iou >= 0.50 || frac >= 0.70;
                if !is_overlapping {
                    continue;
                }

                // Check if e2 cites e1 (or e1's isolates) as Home (an inverted Home mirror where e1 is actually the recombinant),
                // OR cites e1 as Donor when e1 and e2 belong to the same clade / swap roles.
                // Note: If e2 is in clade A and e1 is in clade B (d(c1, c2) > d_clade), and e2 cites e1 as Donor,
                // e1 can legitimately be a member of the donor clade B even if e1 had a different event elsewhere or from another clade,
                // UNLESS e1's own event over this overlapping tract comes from e2's clade A!
                let e2_cites_e1_as_home =
                    (e2.home_name == *c1_name) || e1.isolates.contains(&e2.home_name);
                let e2_cites_e1_as_donor =
                    (e2.donor_name == *c1_name) || e1.isolates.contains(&e2.donor_name);
                let e1_donor_from_e2_clade = match (e1.donor_idx, e2.home_idx) {
                    (Some(d1), Some(h2)) => {
                        crate::landmarks::compute_pairwise_distance(_aln, d1, e2.candidate_idx)
                            <= d_clade
                            || crate::landmarks::compute_pairwise_distance(_aln, d1, h2) <= d_clade
                    }
                    _ => false,
                };
                let same_cand_clade = crate::landmarks::compute_pairwise_distance(
                    _aln,
                    e1.candidate_idx,
                    e2.candidate_idx,
                ) <= d_clade;
                let same_donor_clade = e1.donor_name == e2.donor_name
                    || (e1.is_ghost_donor && e2.is_ghost_donor)
                    || match (e1.donor_idx, e2.donor_idx) {
                        (Some(d1), Some(d2)) => {
                            crate::landmarks::compute_pairwise_distance(_aln, d1, d2) <= d_clade
                        }
                        _ => false,
                    };

                // Note: In population-scale multi-lineage cohorts (N >= 16), if e1 and e2 are in the same
                // candidate clade and share the same external donor clade over this overlapping tract,
                // they are co-recombinant sisters (to be resolved by `resolve_donor_chains` and `group_ancestral_events`),
                // NOT an asymmetric reflected illusion!
                let is_corecombinant_sister_pair =
                    _aln.num_taxa >= 16 && same_cand_clade && same_donor_clade;
                let e2_cites_e1 = (e2_cites_e1_as_home && !is_corecombinant_sister_pair)
                    || (e2_cites_e1_as_donor && (same_cand_clade || e1_donor_from_e2_clade));

                if e2_cites_e1 {
                    // e2 is citing an active recombinant candidate as parent.
                    let e1_donor_internal = e1.donor_idx.map_or(false, |d| {
                        crate::landmarks::compute_pairwise_distance(_aln, d, e1.candidate_idx)
                            < d_clade
                            || e1.home_idx.map_or(false, |h| {
                                crate::landmarks::compute_pairwise_distance(_aln, d, h) < d_clade
                            })
                    });
                    let e2_donor_internal = e2.donor_idx.map_or(false, |d| {
                        crate::landmarks::compute_pairwise_distance(_aln, d, e2.candidate_idx)
                            < d_clade
                            || e2.home_idx.map_or(false, |h| {
                                crate::landmarks::compute_pairwise_distance(_aln, d, h) < d_clade
                            })
                    });

                    let e2_is_ghost_home_mirror = e1.is_ghost_donor
                        && e2_cites_e1_as_home
                        && !e2.is_ghost_donor
                        && e2.d_tract_donor > d_clade
                        && e2.d_tract_donor >= 0.75 * e2.d_flank_donor;

                    let e1_has_ext_donor = e1.donor_idx.is_some() && !e1_donor_internal;
                    let e2_has_ext_donor =
                        e2.donor_idx.is_some() && !e2_donor_internal && !e2_is_ghost_home_mirror;

                    if e2_is_ghost_home_mirror || (e1_has_ext_donor && !e2_has_ext_donor) {
                        pruned[j] = true;
                    } else if e2_has_ext_donor && !e1_has_ext_donor {
                        pruned[i] = true;
                        break;
                    } else {
                        let delta_h1 = (e1.d_tract_home - e1.d_flank_home).max(0.0);
                        let delta_h2 = (e2.d_tract_home - e2.d_flank_home).max(0.0);
                        let delta_d1 = (e1.d_flank_donor - e1.d_tract_donor).max(0.0);
                        let delta_d2 = (e2.d_flank_donor - e2.d_tract_donor).max(0.0);
                        let trans1 = delta_h1 + delta_d1;
                        let trans2 = delta_h2 + delta_d2;

                        if (trans1 - trans2).abs() > 1e-4 {
                            if trans1 > trans2 {
                                pruned[j] = true;
                            } else {
                                pruned[i] = true;
                                break;
                            }
                        } else if e1.p_fisher <= e2.p_fisher {
                            pruned[j] = true;
                        } else {
                            pruned[i] = true;
                            break;
                        }
                    }
                }
            }
        }

        events
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| !pruned[*idx])
            .map(|(_, e)| e)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fisher_exact_2x2_basic() {
        let dd = SequenceDueDiligence::new(3, 0.05, 1000);
        // Table:
        // [10, 0]
        // [ 0, 10]
        // Strongly significant
        let p = dd.fisher_exact_2x2(10, 0, 0, 10);
        assert!(p < 1e-4);

        // Null table:
        // [5, 5]
        // [5, 5]
        let p_null = dd.fisher_exact_2x2(5, 5, 5, 5);
        assert!(p_null > 0.4);
    }

    #[test]
    fn test_adjudicate_reticulation_graph_cycle() {
        let dd = SequenceDueDiligence::new(3, 0.05, 1000);
        let aln = Alignment::new(
            vec!["R".into(), "H".into(), "D".into()],
            1000,
            vec![1; 3000],
        );

        let screening = crate::types::ScreeningResult {
            v_cont: vec![5.0, 3.0, 1.0],
            v_root: vec![0.0; 3],
            r_root: vec![0.0; 3],
            gamma_root: vec![0.0; 3],
            rate_burst_indices: vec![],
            ghost_candidate_indices: vec![],
            candidate_recombinant_indices: vec![0, 1],
            clonal_backbone_indices: vec![2],
            median_v_cont: 3.0,
            is_non_recombinant_h0: false,
        };

        // Event 1: R with Home H (1), Donor D (2)
        let e1 = RecombinationEvent {
            event_id: 1,
            isolates: vec!["R".into()],
            candidate_idx: 0,
            candidate_name: "R".into(),
            home_idx: Some(1),
            home_name: "H".into(),
            donor_idx: Some(2),
            donor_name: "D".into(),
            is_ghost_donor: false,
            is_crossover: true,
            u1: 1,
            u2: 500,
            u1_continuous: 1.0,
            u2_continuous: 500.0,
            tract_length: 500,
            s_informative: 50,
            p_fisher: 1e-20,
            z_phys: 10.0,
            d_flank_home: 0.01,
            d_flank_donor: 0.10,
            d_tract_home: 0.10,
            d_tract_donor: 0.01,
            is_verified: true,
        };

        // Event 2: H with Home D (2), Donor R (0) [Reflected illusion!]
        let e2 = RecombinationEvent {
            event_id: 2,
            isolates: vec!["H".into()],
            candidate_idx: 1,
            candidate_name: "H".into(),
            home_idx: Some(2),
            home_name: "D".into(),
            donor_idx: Some(0),
            donor_name: "R".into(),
            is_ghost_donor: false,
            is_crossover: true,
            u1: 501,
            u2: 1000,
            u1_continuous: 501.0,
            u2_continuous: 1000.0,
            tract_length: 500,
            s_informative: 50,
            p_fisher: 1e-20,
            z_phys: 8.0,
            d_flank_home: 0.10,
            d_flank_donor: 0.10,
            d_tract_home: 0.10,
            d_tract_donor: 0.01,
            is_verified: true,
        };

        let adjudicated = dd.adjudicate_reticulation_graph(vec![e1, e2], &aln, &screening);
        assert_eq!(adjudicated.len(), 1);
        assert_eq!(adjudicated[0].isolates[0], "R");
        assert_eq!(adjudicated[0].home_name, "H");
        assert_eq!(adjudicated[0].donor_name, "D");
    }
}
