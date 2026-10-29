use crate::landmarks::compute_pairwise_distance;
use crate::types::{Alignment, LandmarkSet};
use std::f64::consts::PI;

pub struct TreeRoPE {
    pub dim: usize,
    pub num_blocks: usize,
}

impl TreeRoPE {
    pub fn new(dim: usize) -> Self {
        assert_eq!(dim % 2, 0, "TreeRoPE dim must be even");
        Self {
            dim,
            num_blocks: dim / 2,
        }
    }

    /// Computes asymmetric rotary factor matrix M of shape (N, K+1)
    /// Column 0 is the Root factor; Columns 1..=K are the Landmark factors.
    pub fn compute_query_landmark_rotary_factors(
        &self,
        aln: &Alignment,
        landmarks: &LandmarkSet,
    ) -> Vec<f64> {
        let n = aln.num_taxa;
        let k = landmarks.num_landmarks;
        let num_channels = k + 1;

        let n_t = self.num_blocks / 2;
        let n_th = self.num_blocks - n_t;

        let max_t = landmarks.root_to_tip.iter().cloned().fold(1.0f64, f64::max);
        let t_scale = max_t.max(1e-4);

        let mut omega_t = Vec::with_capacity(n_t);
        for m in 0..n_t {
            let exponent = (m as f64) / ((n_t.max(2) - 1) as f64);
            omega_t.push((2.0 * PI / t_scale) * 0.5f64.powf(exponent));
        }

        let mut omega_theta = Vec::with_capacity(n_th);
        for m in 0..n_th {
            omega_theta.push(0.5 * (PI / 2.0).powi(m as i32));
        }

        // Derive query coordinates in R^{N x 2} via Barycentric Simplex Projection
        let mut query_coords = vec![0.0f64; n * 2];

        for i in 0..n {
            if let Some(pos) = landmarks.indices.iter().position(|&idx| idx == i) {
                // Exact landmark coordinates
                let l_idx = (pos + 1) * 2;
                query_coords[i * 2 + 0] = landmarks.coords[l_idx + 0];
                query_coords[i * 2 + 1] = landmarks.coords[l_idx + 1];
            } else {
                // Compute distances to each landmark
                let mut dists = Vec::with_capacity(k);
                for &l_idx in &landmarks.indices {
                    dists.push(compute_pairwise_distance(aln, i, l_idx));
                }

                // Adaptive temperature from median distance
                let mut sorted_dists = dists.clone();
                sorted_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median_d = sorted_dists[sorted_dists.len() / 2];
                let tau = median_d.max(1.0 / (aln.length as f64));

                let mut max_logit = f64::NEG_INFINITY;
                let mut logits = Vec::with_capacity(k);
                for &d in &dists {
                    let logit = -d / tau;
                    if logit > max_logit {
                        max_logit = logit;
                    }
                    logits.push(logit);
                }

                let mut sum_exp = 0.0;
                let mut weights = Vec::with_capacity(k);
                for logit in logits {
                    let w = (logit - max_logit).exp();
                    weights.push(w);
                    sum_exp += w;
                }
                for w in &mut weights {
                    *w /= sum_exp;
                }

                let mut t_proj = 0.0;
                let mut x_proj = 0.0;
                let mut y_proj = 0.0;

                for p in 0..k {
                    let l_idx = (p + 1) * 2;
                    let rad = landmarks.coords[l_idx + 0];
                    let ang = landmarks.coords[l_idx + 1];
                    let w = weights[p];

                    t_proj += w * rad;
                    x_proj += w * ang.cos();
                    y_proj += w * ang.sin();
                }

                let mut th_proj = y_proj.atan2(x_proj);
                if th_proj < 0.0 {
                    th_proj += 2.0 * PI;
                }

                query_coords[i * 2 + 0] = t_proj;
                query_coords[i * 2 + 1] = th_proj;
            }
        }

        // Pairwise inner products with landmarks (columns 1..=K) and root (column 0)
        let mut m_matrix = vec![0.0f64; n * num_channels];

        for i in 0..n {
            let q_t = query_coords[i * 2 + 0];
            let q_th = query_coords[i * 2 + 1];

            for ch in 0..num_channels {
                let (l_t, l_th) = if ch == 0 {
                    (0.0, 0.0) // Root at origin
                } else {
                    let l_idx = ch * 2;
                    (landmarks.coords[l_idx + 0], landmarks.coords[l_idx + 1])
                };

                let delta_t = q_t - l_t;
                let mut delta_th = q_th - l_th;
                // Circular wrap to [-PI, PI]
                delta_th = (delta_th + PI).rem_euclid(2.0 * PI) - PI;

                let mut sum_m = 0.0;
                for &wt in &omega_t {
                    sum_m += (delta_t * wt).cos();
                }
                for &wth in &omega_theta {
                    sum_m += (delta_th * wth).cos();
                }

                m_matrix[i * num_channels + ch] = sum_m;
            }
        }

        m_matrix
    }
}
