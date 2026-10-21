use crate::attention::{AttentionMasks, PhyloAttentionEngine};
use crate::types::{Alignment, LandmarkSet, ScreeningResult};

pub struct LocalFluxScreener {
    pub root_ratio_threshold: f64,
    pub spatial_coherence_threshold: f64,
    pub coherence_window: usize,
    pub variance_multiplier: f64,
    pub alpha: f64,
    pub epsilon: f64,
    pub tier2_stringency: f64,
}

impl Default for LocalFluxScreener {
    fn default() -> Self {
        Self {
            root_ratio_threshold: 0.50,
            spatial_coherence_threshold: 0.30,
            coherence_window: 50,
            variance_multiplier: 1.80,
            alpha: 0.05,
            epsilon: 1e-12,
            tier2_stringency: 1.0,
        }
    }
}

impl LocalFluxScreener {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tier2_stringency(mut self, tier2_stringency: f64) -> Self {
        self.tier2_stringency = tier2_stringency;
        self
    }

    /// Evaluates spatial coherence of the Root flux stream over one taxon's force row
    /// `f_row` of shape (L, K+1), indexed `u * num_channels + ch`.
    fn compute_spatial_coherence(&self, f_row: &[f64], l: usize, num_channels: usize) -> f64 {
        let mut total_abs = 0.0f64;
        let w = self.coherence_window.min(l);
        if w == 0 {
            return 0.0;
        }

        // Circular buffer avoids allocating full L-length vector on heap
        let mut window_buf = vec![0.0f64; w];
        let mut current_sum = 0.0f64;
        let mut max_window_sum = 0.0f64;

        for u in 0..l {
            let val = f_row[u * num_channels + 0];
            total_abs += val.abs();
            let pos_val = val.max(0.0);

            if u < w {
                window_buf[u] = pos_val;
                current_sum += pos_val;
                if current_sum > max_window_sum {
                    max_window_sum = current_sum;
                }
            } else {
                let slot = u % w;
                let old_val = window_buf[slot];
                window_buf[slot] = pos_val;
                current_sum += pos_val - old_val;
                if current_sum > max_window_sum {
                    max_window_sum = current_sum;
                }
            }
        }

        if total_abs < self.epsilon {
            0.0
        } else {
            max_window_sum / total_abs
        }
    }

    /// Computes Spectral Participation Ratio (PR) of contemporary channels for a candidate taxon.
    /// Measures effective dimensional complexity of contemporary flux to distinguish
    /// multidirectional heterotachy bursts (high PR) from low-rank reticulation dipoles (PR ~ 1).
    ///
    /// `c_row` is the candidate's own momentum row of shape (L, K+1), indexed
    /// `u * num_channels + ch`.
    #[allow(dead_code)]
    fn compute_participation_ratio(
        &self,
        c_row: &[f64],
        l: usize,
        num_channels: usize,
        home_ch: Option<usize>,
        inv_l: f64,
    ) -> f64 {
        let mut active_channels = Vec::with_capacity(num_channels);
        for ch in 1..num_channels {
            if Some(ch) != home_ch {
                active_channels.push(ch);
            }
        }
        let m = active_channels.len();
        if m < 2 {
            return 1.0;
        }

        // 1. Single pass to compute channel means
        let mut sum_a = vec![0.0f64; m];
        for u in 0..l {
            let base_idx = u * num_channels;
            for (a_idx, &cha) in active_channels.iter().enumerate() {
                sum_a[a_idx] += c_row[base_idx + cha];
            }
        }
        let mut means = vec![0.0f64; m];
        for a in 0..m {
            means[a] = sum_a[a] * inv_l;
        }

        // 2. Single pass to compute symmetric covariance matrix
        let mut cov = vec![0.0f64; m * m];
        let mut diff = vec![0.0f64; m];
        for u in 0..l {
            let base_idx = u * num_channels;
            for (a_idx, &cha) in active_channels.iter().enumerate() {
                diff[a_idx] = c_row[base_idx + cha] - means[a_idx];
            }
            for a in 0..m {
                let da = diff[a];
                for b in a..m {
                    cov[a * m + b] += da * diff[b];
                }
            }
        }

        // Symmetrize and scale by 1/L
        for a in 0..m {
            cov[a * m + a] *= inv_l;
            for b in (a + 1)..m {
                let val = cov[a * m + b] * inv_l;
                cov[a * m + b] = val;
                cov[b * m + a] = val;
            }
        }

        let mut tr1 = 0.0f64;
        for a in 0..m {
            tr1 += cov[a * m + a];
        }

        let mut tr2 = 0.0f64;
        for a in 0..m {
            for b in 0..m {
                tr2 += cov[a * m + b] * cov[b * m + a];
            }
        }

        if tr2 > 1e-12 {
            (tr1 * tr1) / tr2
        } else {
            1.0
        }
    }

    /// Executes Pass 1 O(NL) Screening.
    ///
    /// Each taxon's force and momentum rows are regenerated on demand into two reusable
    /// (L, K+1) scratch buffers rather than read out of resident (L, N, K+1) tensors. This
    /// loop is a sequential per-taxon reduction: every value crossing the `i` boundary is a
    /// per-taxon scalar, the channel accumulators are consumed before the next taxon, and
    /// `running_c` restarts at zero per row, so each prefix sum is independent of every other
    /// taxon.
    #[allow(clippy::too_many_arguments)]
    pub fn screen(
        &self,
        attention: &PhyloAttentionEngine,
        m_matrix: &[f64],
        b_prior: &[f64],
        root_sequence: &[u8],
        masks: &AttentionMasks,
        mean_a: &[f64],
        aln: &Alignment,
        landmarks: &LandmarkSet,
    ) -> ScreeningResult {
        let n = aln.num_taxa;
        let l = aln.length;
        let k = landmarks.num_landmarks;
        let num_channels = k + 1;

        let mut v_root = vec![0.0f64; n];
        let mut root_pulse = vec![0.0f64; n];
        let mut root_pulse_stat = vec![0.0f64; n];
        let mut v_cont = vec![0.0f64; n];
        let mut cont_pulse = vec![0.0f64; n];
        let mut r_root = vec![0.0f64; n];
        let mut r_dim = vec![0.0f64; n];
        let mut gamma_root = vec![0.0f64; n];

        let inv_l = 1.0 / (l as f64);

        // Per-taxon scratch rows, allocated once outside both loops
        let mut f_row = vec![0.0f64; l * num_channels];
        let mut c_row = vec![0.0f64; l * num_channels];

        // 1. Compute variances and peak-to-trough pulses in a single cache-friendly pass per taxon
        for i in 0..n {
            attention.compute_force_and_momentum_row(
                aln,
                landmarks,
                m_matrix,
                b_prior,
                root_sequence,
                masks,
                mean_a,
                i,
                &mut f_row,
                &mut c_row,
            );

            let mut sum_c = vec![0.0f64; num_channels];
            let mut sum_sq_c = vec![0.0f64; num_channels];
            let mut min_c = vec![f64::INFINITY; num_channels];
            let mut max_c = vec![f64::NEG_INFINITY; num_channels];

            for u in 0..l {
                let base_idx = u * num_channels;
                for ch in 0..num_channels {
                    let c = c_row[base_idx + ch];
                    sum_c[ch] += c;
                    sum_sq_c[ch] += c * c;
                    if c < min_c[ch] {
                        min_c[ch] = c;
                    }
                    if c > max_c[ch] {
                        max_c[ch] = c;
                    }
                }
            }

            // Root channel 0: step variance and standardized Brownian bridge excursion
            let mean_c0 = sum_c[0] * inv_l;
            let var_c0 = (sum_sq_c[0] * inv_l - mean_c0 * mean_c0).max(0.0);
            v_root[i] = var_c0;
            let raw_pulse = (max_c[0] - min_c[0]).max(0.0);
            root_pulse[i] = raw_pulse;

            let mut sum_sq_f0 = 0.0;
            for u in 0..l {
                let f0 = f_row[u * num_channels + 0];
                sum_sq_f0 += f0 * f0;
            }
            let sigma_f0 = (sum_sq_f0 * inv_l).sqrt().max(1e-6);
            root_pulse_stat[i] = raw_pulse / ((l as f64).sqrt() * sigma_f0);

            // Contemporary channels 1..=K
            let mut sum_var_cont = 0.0;
            let mut max_pulse = 0.0f64;

            for ch in 1..num_channels {
                let mean_c = sum_c[ch] * inv_l;
                let var_c = (sum_sq_c[ch] * inv_l - mean_c * mean_c).max(0.0);
                sum_var_cont += var_c;

                let pulse = (max_c[ch] - min_c[ch]).max(0.0);
                if pulse > max_pulse {
                    max_pulse = pulse;
                }
            }

            v_cont[i] = sum_var_cont;
            cont_pulse[i] = max_pulse;

            let v_cont_mean = sum_var_cont / (k.max(1) as f64);
            r_root[i] = var_c0 / (sum_var_cont + var_c0 + self.epsilon);
            r_dim[i] = var_c0 / (v_cont_mean + var_c0 + self.epsilon);

            // Spatial coherence
            gamma_root[i] = self.compute_spatial_coherence(&f_row, l, num_channels);
        }

        // 2. Heterotachy logit outlier screening
        let mut logit_r = Vec::with_capacity(n);
        let eps_logit = 1e-4;
        for &r in &r_root {
            let clamped = r.clamp(eps_logit, 1.0 - eps_logit);
            logit_r.push((clamped / (1.0 - clamped)).ln());
        }

        let med_logit = median(&logit_r);
        let mad_logit = median(
            &logit_r
                .iter()
                .map(|&x| (x - med_logit).abs())
                .collect::<Vec<_>>(),
        );
        let scale_logit = (1.4826 * mad_logit).max(1.0 / (l as f64).sqrt());

        let z_alpha_r = 1.645f64; // norm.ppf(0.95)

        let mut rate_burst_indices = Vec::new();
        let mut ghost_candidate_indices = Vec::new();
        let mut is_rate_burst = vec![false; n];
        let mut is_ghost_candidate = vec![false; n];

        let (d_sister, d_clade) =
            crate::landmarks::compute_scale_invariant_distance_thresholds(aln);
        let k_crit_root = (-0.5f64 * (self.alpha / (2.0f64 * (n as f64).max(1.0f64))).ln())
            .sqrt()
            .max(1.645f64);

        // Sister adjacency in a single upper-triangular pass: each pair distance
        // is evaluated at most once, and pairs where both leaves already have a
        // sister are skipped entirely.
        let mut has_sister = vec![false; n];
        for i in 0..n {
            for j in (i + 1)..n {
                if has_sister[i] && has_sister[j] {
                    continue;
                }
                if crate::landmarks::compute_pairwise_distance(aln, i, j) < d_sister {
                    has_sister[i] = true;
                    has_sister[j] = true;
                }
            }
        }

        let med_v_root = median(&v_root);
        let mad_v_root = median(
            &v_root
                .iter()
                .map(|&x| (x - med_v_root).abs())
                .collect::<Vec<_>>(),
        );
        let scale_v_root = (1.4826 * mad_v_root).max(0.05);

        for i in 0..n {
            let z_r = (logit_r[i] - med_logit) / scale_logit;
            let z_v_root = (v_root[i] - med_v_root) / scale_v_root;

            let is_isolated_leaf = !has_sister[i];
            if z_r > z_alpha_r
                && r_root[i] >= self.root_ratio_threshold
                && gamma_root[i] <= self.spatial_coherence_threshold
                && is_isolated_leaf
            {
                is_rate_burst[i] = true;
                rate_burst_indices.push(i);
            } else if (z_r > z_alpha_r || r_dim[i] > self.root_ratio_threshold || z_v_root > 2.0)
                && (root_pulse_stat[i] >= k_crit_root || root_pulse[i] >= 3.0)
                && (!is_isolated_leaf || gamma_root[i] > self.spatial_coherence_threshold)
            {
                is_ghost_candidate[i] = true;
                ghost_candidate_indices.push(i);
            }
        }

        // 3. Pool of non-burst taxa
        let mut pool_indices: Vec<usize> = (0..n).filter(|&i| !is_rate_burst[i]).collect();
        if pool_indices.is_empty() {
            pool_indices = (0..n).collect();
        }

        let pool_v: Vec<f64> = pool_indices.iter().map(|&i| v_cont[i]).collect();
        let pool_pulse: Vec<f64> = pool_indices.iter().map(|&i| cont_pulse[i]).collect();

        let med_v = median(&pool_v);
        let med_pulse = median(&pool_pulse);

        let lower_v: Vec<f64> = pool_v.iter().cloned().filter(|&x| x <= med_v).collect();
        let lower_pulse: Vec<f64> = pool_pulse
            .iter()
            .cloned()
            .filter(|&x| x <= med_pulse)
            .collect();

        let (med_low_v, mad_v) = if !lower_v.is_empty() {
            let med_low = median(&lower_v);
            let mad = median(
                &lower_v
                    .iter()
                    .map(|&x| (x - med_low).abs())
                    .collect::<Vec<_>>(),
            );
            (med_low, mad)
        } else {
            (med_v, 0.0)
        };
        let (med_low_pulse, mad_pulse) = if !lower_pulse.is_empty() {
            let med_low = median(&lower_pulse);
            let mad = median(
                &lower_pulse
                    .iter()
                    .map(|&x| (x - med_low).abs())
                    .collect::<Vec<_>>(),
            );
            (med_low, mad)
        } else {
            (med_pulse, 0.0)
        };

        // Robust baseline scaling from uncontaminated lower quartile (75% breakdown point against recombinant swarms):
        let s2 = self.tier2_stringency.max(0.1);
        let z_crit_cand = 2.0f64 * s2;

        let floor_v = (med_low_v / (l as f64).sqrt()).max(1.0 / (l as f64).sqrt());
        let floor_pulse = (med_low_pulse / (l as f64).sqrt()).max(1.0 / (l as f64).sqrt());

        let scale_v = (1.4826 * mad_v).max(floor_v);
        let scale_pulse = (1.4826 * mad_pulse).max(floor_pulse);

        let thresh_v = (med_low_v + z_crit_cand * scale_v).max(med_v + 1.5 * s2 * scale_v);
        let thresh_pulse =
            (med_low_pulse + z_crit_cand * scale_pulse).max(med_pulse + 1.5 * s2 * scale_pulse);

        let mut candidate_mask = vec![false; n];
        for &i in &pool_indices {
            if s2 <= 0.65
                || v_cont[i] > thresh_v
                || cont_pulse[i] > thresh_pulse
                || is_ghost_candidate[i]
            {
                candidate_mask[i] = true;
            }
        }

        // Requirement 1 (DIR-RHIZ-PAFF-012.7): Patristic-Distance Gated Screener Admission
        // Small-cohort high-sensitivity admission in Pass 1 must ONLY activate
        // when the alignment is genuinely in a low-divergence regime (d_clade <= 3/sqrt(L)).
        let is_low_div = d_clade <= (3.0 / (l as f64).sqrt()).max(2.0 * d_sister);

        // Small-cohort dispersion admission (N <= 8, DIR-RHIZ-PAFF-012.6 / DIR-RHIZ-PAFF-012.7)
        if n <= 8 && is_low_div && !pool_indices.is_empty() {
            let max_v = pool_v.iter().cloned().fold(0.0f64, f64::max);
            let disp_ratio = max_v / med_v.max(1e-6);
            let r_crit_disp = ((n as f64).max(2.0) / self.alpha).ln().max(3.0);
            if disp_ratio <= r_crit_disp || pool_v.iter().all(|&x| x < 3.0) {
                for &i in &pool_indices {
                    if !is_rate_burst[i] {
                        candidate_mask[i] = true;
                    }
                }
            }
        }

        // Pervasive Recombination Admission Sieve (DIR-RHIZ-PAFF-013.3 Requirement 2)
        // In cohorts with pervasive recombination (e.g. viral swarms, potyviruses, CRFs),
        // the cohort-wide variance is orders of magnitude above the clonal floor.
        // In this regime, median-based outlier thresholds suffer from breakdown point masking.
        if !pool_indices.is_empty() {
            let mean_v_cohort: f64 = pool_v.iter().sum::<f64>() / (pool_v.len().max(1) as f64);
            let v_clonal_floor = 10.0 * ((l as f64) / 1000.0).sqrt().max(1.0);
            let v_pervasive_ceiling = 3.0 * v_clonal_floor;
            if mean_v_cohort >= v_pervasive_ceiling || med_v >= v_pervasive_ceiling {
                for &i in &pool_indices {
                    if !is_rate_burst[i]
                        && (v_cont[i] >= v_clonal_floor || cont_pulse[i] >= 1.5 * v_clonal_floor)
                    {
                        candidate_mask[i] = true;
                    }
                }
            }
        }

        let mut candidate_recombinant_indices = Vec::new();
        let mut clonal_backbone_indices = Vec::new();

        for i in 0..n {
            if candidate_mask[i] {
                candidate_recombinant_indices.push(i);
            } else if !is_rate_burst[i] {
                clonal_backbone_indices.push(i);
            }
        }

        let is_non_recombinant_h0 = candidate_recombinant_indices.is_empty();

        ScreeningResult {
            v_cont,
            v_root,
            r_root,
            gamma_root,
            rate_burst_indices,
            ghost_candidate_indices,
            candidate_recombinant_indices,
            clonal_backbone_indices,
            median_v_cont: med_v,
            is_non_recombinant_h0,
        }
    }
}

/// Delegates to `aeon_manifold::median`; the empty-slice guard preserves
/// this module's historical contract of returning 0.0 rather than NaN.
fn median(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    aeon_manifold::median(vals)
}

/// Machine-precision Acklam rational approximation of the standard normal
/// percent point function (probit).
///
/// Delegates to `aeon_manifold::norm_ppf`; the clamp preserves this wrapper's
/// historical contract of never returning an infinite quantile.
pub fn normal_ppf(p: f64) -> f64 {
    aeon_manifold::norm_ppf(p.clamp(1e-15, 1.0 - 1e-15))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_odd() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
    }

    #[test]
    fn test_median_even() {
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    #[test]
    fn test_median_empty_returns_zero() {
        assert_eq!(median(&[]), 0.0);
    }

    #[test]
    fn test_median_single() {
        assert_eq!(median(&[42.0]), 42.0);
    }

    #[test]
    fn test_normal_ppf_known_values() {
        // scipy.stats.norm.ppf references
        // Acklam approximation error is ~1.6e-9; 1e-8 matches aeon-manifold
        // golden tolerances so this is a regression net, not a precision oracle.
        assert!((normal_ppf(0.5) - 0.0).abs() < 1e-8);
        assert!((normal_ppf(0.975) - 1.9599639845400545).abs() < 1e-8);
        assert!((normal_ppf(0.025) - -1.9599639845400545).abs() < 1e-8);
        assert!((normal_ppf(0.001) - -3.090232306167813).abs() < 1e-7);
    }

    #[test]
    fn test_normal_ppf_symmetric() {
        let a = normal_ppf(0.05);
        let b = normal_ppf(0.95);
        assert!((a + b).abs() < 1e-12);
    }

    #[test]
    fn test_normal_ppf_clamps_at_boundaries() {
        // Historical contract: never returns +-infinity. p<=0 and p>=1 are
        // clamped to [1e-15, 1-1e-15] before delegating to aeon-manifold.
        let lo = normal_ppf(0.0);
        let hi = normal_ppf(1.0);
        assert!(lo.is_finite() && hi.is_finite());
        assert!(lo < -7.0 && hi > 7.0);
        // Out-of-range inputs match the clamped boundary quantiles.
        assert_eq!(normal_ppf(-1.0), normal_ppf(0.0));
        assert_eq!(normal_ppf(2.0), normal_ppf(1.0));
    }
}
