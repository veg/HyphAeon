use crate::landmarks::{compute_pairwise_distance, jukes_cantor_distance};
use crate::types::{Alignment, LandmarkSet};

pub struct PhyloBias {
    pub epsilon_0: f64,
    pub decay_lambda: f64,
}

impl PhyloBias {
    pub fn new(epsilon_0: f64, decay_lambda: f64) -> Self {
        Self {
            epsilon_0,
            decay_lambda,
        }
    }

    #[inline(always)]
    pub fn retention_probability(&self, distance: f64) -> f64 {
        self.epsilon_0 + (1.0 - self.epsilon_0) * (-self.decay_lambda * distance).exp()
    }

    /// Computes B^prior matrix of shape (N, K+1)
    pub fn compute_prior_matrix(&self, aln: &Alignment, landmarks: &LandmarkSet) -> Vec<f64> {
        let n = aln.num_taxa;
        let k = landmarks.num_landmarks;
        let num_channels = k + 1;

        let mut p_full = vec![0.0f64; n * num_channels];
        let mut d_landmarks = vec![0.0f64; n * k];

        // 1. Landmark retention probabilities (columns 1..=K) and Ancestral Root (column 0)
        let mut is_landmark = vec![None; n];
        for (pos, &idx) in landmarks.indices.iter().enumerate() {
            if idx < n {
                is_landmark[idx] = Some(pos);
            }
        }
        let mut hap_cache: std::collections::HashMap<&[u8], usize> = if n >= 500 {
            std::collections::HashMap::with_capacity(n.min(4096))
        } else {
            std::collections::HashMap::new()
        };

        for i in 0..n {
            if n >= 500 && is_landmark[i].is_none() {
                let row_slice = aln.row(i);
                if let Some(&rep) = hap_cache.get(row_slice) {
                    for ch in 0..num_channels {
                        p_full[i * num_channels + ch] = p_full[rep * num_channels + ch];
                    }
                    continue;
                }
            }
            for (k_idx, &l_global) in landmarks.indices.iter().enumerate() {
                let dist = compute_pairwise_distance(aln, i, l_global);
                let jc = jukes_cantor_distance(dist);
                d_landmarks[i * k + k_idx] = jc;

                let p = self.retention_probability(jc);
                p_full[i * num_channels + (k_idx + 1)] = p;
            }
            let t_query = if let Some(pos) = is_landmark[i] {
                landmarks.root_to_tip.get(pos).copied().unwrap_or(0.1)
            } else {
                let mut min_proj = f64::INFINITY;
                for p in 0..k {
                    let l_t = landmarks.root_to_tip.get(p).copied().unwrap_or(0.1);
                    let d = d_landmarks[i * k + p];
                    let proj = l_t + d;
                    if proj < min_proj {
                        min_proj = proj;
                    }
                }
                min_proj
            };
            p_full[i * num_channels + 0] = self.retention_probability(t_query);
            if n >= 500 && is_landmark[i].is_none() {
                hap_cache.insert(aln.row(i), i);
            }
        }

        // 3. Construct logit bias B^prior
        let mut b_prior = vec![0.0f64; n * num_channels];
        for idx in 0..(n * num_channels) {
            let p = p_full[idx].max(1e-15);
            b_prior[idx] = p.ln();
        }

        // Mask self-attention: if query i is landmark k, column k+1 is -inf
        for (k_idx, &i_idx) in landmarks.indices.iter().enumerate() {
            b_prior[i_idx * num_channels + (k_idx + 1)] = f64::NEG_INFINITY;
        }

        b_prior
    }
}
