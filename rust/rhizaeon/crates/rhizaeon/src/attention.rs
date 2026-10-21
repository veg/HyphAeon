use crate::types::{Alignment, LandmarkSet};

pub struct PhyloAttentionEngine {
    pub dim: usize,
    pub sqrt_d: f64,
    pub root_sink_factor: f64,
}

/// Self- and twin-clone masking matrices of shape (N, K), a pure function of
/// (`aln`, `landmarks`) and independent of the genomic coordinate `u`. Hoisted out of Pass 1
/// because the geodesic-interior classification is O(K^2 * N) pairwise-distance work that must
/// be paid once per alignment, never once per streamed row.
pub struct AttentionMasks {
    /// Landmark p is masked for query i: self-attention, or a suppressed twin clone.
    pub mask_landmark: Vec<bool>,
    /// Landmark p is a non-sister (external) landmark for query i.
    pub is_non_sister_landmark: Vec<bool>,
}

/// Computes the self- and twin-clone masking matrices (n x k):
/// Mask landmark p for query i if i == l_global OR if l_global is an identical/sister clone
/// of i (d < d_sister) AND there is at least one other landmark in i's clade (d <= d_clade)
/// AND the sister sub-cluster is not the strict majority consensus of the clade.
/// This prevents minority co-recombinant twins (e.g. pop1.21 and pop1.23, or Rec_0..K) from
/// absorbing 100% of each other's attention while preserving majority parental clades.
pub fn compute_attention_masks(aln: &Alignment, landmarks: &LandmarkSet) -> AttentionMasks {
    let n = aln.num_taxa;
    let k = landmarks.num_landmarks;

    let (d_sister, d_clade) = crate::landmarks::compute_scale_invariant_distance_thresholds(aln);
    let mut mask_landmark = vec![false; n * k];
    let mut is_non_sister_landmark = vec![false; n * k];
    for i in 0..n {
        let mut sister_landmarks = Vec::new();
        let mut ext_landmarks = Vec::new();
        let mut non_sister_clade_landmarks = 0usize;
        for p in 0..k {
            let l_global = landmarks.indices[p];
            if l_global == i {
                mask_landmark[i * k + p] = true;
            } else {
                let d = crate::landmarks::compute_pairwise_distance(aln, i, l_global);
                if crate::landmarks::is_true_sister_pair(aln, i, l_global, d_sister) {
                    sister_landmarks.push(p);
                } else {
                    ext_landmarks.push((p, l_global, d));
                    is_non_sister_landmark[i * k + p] = true;
                    if d <= d_clade {
                        non_sister_clade_landmarks += 1;
                    }
                }
            }
        }
        if !sister_landmarks.is_empty() && !ext_landmarks.is_empty() {
            let mut my_size = 1usize;
            for j in 0..n {
                if j != i && crate::landmarks::compute_pairwise_distance(aln, i, j) < d_sister {
                    my_size += 1;
                }
            }

            // 1. Metric Geodesic Interior Condition (Contemporary / Surrogate Donor):
            // A mosaic recombinant R = (1 - a)*P1 + a*P2 lies strictly inside the metric interval
            // between its parental landmarks (p1, p2): d(p1, p2) > max(d(i, p1), d(i, p2)) + 0.5*d_sister
            // and d(i, p1) + d(i, p2) <= 1.15 * d(p1, p2).
            let mut is_metric_interior = false;
            for a_idx in 0..ext_landmarks.len() {
                let (_, l1, di1) = ext_landmarks[a_idx];
                let mut l1_size = 1usize;
                for j in 0..n {
                    if j != l1 && crate::landmarks::compute_pairwise_distance(aln, l1, j) < d_sister
                    {
                        l1_size += 1;
                    }
                }
                for b_idx in (a_idx + 1)..ext_landmarks.len() {
                    let (_, l2, di2) = ext_landmarks[b_idx];
                    let d12 = crate::landmarks::compute_pairwise_distance(aln, l1, l2);
                    let is_geodesic_interior = (di1 + di2) <= 1.15 * d12
                        || (di1.max(di2) <= 1.35 * di1.min(di2) && (di1 + di2) <= 1.65 * d12);
                    if d12 > di1.max(di2) + 0.5 * d_sister && is_geodesic_interior {
                        let mut l2_size = 1usize;
                        for j in 0..n {
                            if j != l2
                                && crate::landmarks::compute_pairwise_distance(aln, l2, j)
                                    < d_sister
                            {
                                l2_size += 1;
                            }
                        }
                        let is_asymmetric_outgroup_mirror =
                            my_size > l1_size.min(l2_size) && di1.max(di2) > 1.4 * di1.min(di2);
                        if !is_asymmetric_outgroup_mirror {
                            is_metric_interior = true;
                            break;
                        }
                    }
                }
                if is_metric_interior {
                    break;
                }
            }

            // 2. Minority Intra-Clade Sub-Cluster Fallback:
            let mut non_sister_clade_aln_count = 0usize;
            for j in 0..n {
                if j != i {
                    let d = crate::landmarks::compute_pairwise_distance(aln, i, j);
                    if d >= d_sister && d <= d_clade {
                        non_sister_clade_aln_count += 1;
                    }
                }
            }
            let is_minority_subclade =
                non_sister_clade_landmarks > 0 && (my_size - 1) <= non_sister_clade_aln_count;

            if is_metric_interior || is_minority_subclade {
                for &p in &sister_landmarks {
                    mask_landmark[i * k + p] = true;
                }
            }
        }
    }

    AttentionMasks {
        mask_landmark,
        is_non_sister_landmark,
    }
}

impl PhyloAttentionEngine {
    pub fn new(dim: usize, root_sink_factor: f64) -> Self {
        Self {
            dim,
            sqrt_d: (dim as f64).sqrt(),
            root_sink_factor,
        }
    }

    /// Pass 1 softmax attention of taxon `i` at site `u` over the N+1 nodes (channel 0 = [ROOT]).
    ///
    /// Pure function of (i, u): every table it reads is either site-independent (`m_matrix`,
    /// `b_prior`, `masks`, `landmarks.indices`) or indexed strictly at `u` (`aln`,
    /// `root_sequence`), and `max_s`, `logits`, `sum_exp` and `any_non_sister_match` are all
    /// cell-local. A row can therefore be regenerated on demand in any loop order with
    /// bit-identical results. Sole copy of the Pass 1 arithmetic; every consumer routes here.
    ///
    /// Writes A(i, u, 0..num_channels) into `out` and returns `valid_q`, so callers reproduce
    /// the gapped-query skip and the `valid_pair_counts` increments without duplicating the
    /// predicate. Masked channels are left at 0.0, mirroring the whole-tensor path, where they
    /// are never assigned. Deliberately touches no accumulator: keeping `sum_a` and
    /// `valid_pair_counts` with the caller is what makes the i/u reorder bit-exact.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn softmax_row_at(
        &self,
        aln: &Alignment,
        landmarks: &LandmarkSet,
        m_matrix: &[f64],
        b_prior: &[f64],
        root_sequence: &[u8],
        masks: &AttentionMasks,
        i: usize,
        u: usize,
        num_channels: usize,
        out: &mut [f64],
    ) -> bool {
        let k = landmarks.num_landmarks;
        let shift_d = (self.dim as f64) / 2.0;
        let root_c = root_sequence.get(u).copied().unwrap_or(0);

        let qi_c = aln.get(i, u);
        let valid_q = qi_c > 0;
        let in_offset = i * num_channels;

        // Masked and gapped cells stay 0.0, as in the vec![0.0]-initialized tensor.
        for ch in 0..num_channels {
            out[ch] = 0.0;
        }

        if !valid_q {
            return false;
        }

        let mut max_s = f64::NEG_INFINITY;
        let mut logits = [0.0f64; 128]; // Max K+1 <= 128
        assert!(num_channels <= 128, "K+1 exceeds stack buffer limit");

        // Landmark channels 1..=K
        let mut any_non_sister_match = false;
        for p in 0..k {
            let ch = p + 1;
            let l_global = landmarks.indices[p];
            let l_c = aln.get(l_global, u);

            if masks.mask_landmark[i * k + p] {
                // Mask self-attention and twin-clone attention
                logits[ch] = f64::NEG_INFINITY;
                continue;
            }

            let m_val = m_matrix[in_offset + ch];
            let match_aff = m_val + shift_d;
            let is_match = valid_q && qi_c == l_c;
            if is_match && masks.is_non_sister_landmark[i * k + p] {
                any_non_sister_match = true;
            }
            let dot = if is_match { match_aff } else { 0.0 };

            let s = b_prior[in_offset + ch] + (dot / self.sqrt_d);
            logits[ch] = s;
            if s > max_s {
                max_s = s;
            }
        }

        // Root channel 0:
        // Matches the ancestral consensus allele (`qi_c == root_c`), AND acts as the
        // autapomorphy / unsampled-outgroup sink (`root_affinity * root_sink_factor`)
        // when a valid query base mismatches all non-sister contemporary landmarks.
        let m_root = m_matrix[in_offset + 0];
        let root_affinity = m_root + shift_d;
        let root_dot = if root_c > 0 && qi_c == root_c {
            root_affinity
        } else if !any_non_sister_match {
            root_affinity * self.root_sink_factor
        } else {
            0.0
        };
        let s_root = b_prior[in_offset + 0] + (root_dot / self.sqrt_d);
        logits[0] = s_root;
        if s_root > max_s {
            max_s = s_root;
        }

        // Softmax
        let mut sum_exp = 0.0;
        for ch in 0..num_channels {
            let s = logits[ch];
            let exp_s = if s.is_finite() {
                (s - max_s).exp()
            } else {
                0.0
            };
            logits[ch] = exp_s;
            sum_exp += exp_s;
        }

        let inv_denom = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
        out[0] = logits[0] * inv_denom;

        for p in 0..k {
            let ch = p + 1;
            if !masks.mask_landmark[i * k + p] {
                out[ch] = logits[ch] * inv_denom;
            }
        }

        true
    }

    /// Empirical sequence-wide mean attention over valid (un-scrubbed) sites: mean_A in (N, K+1).
    ///
    /// Irreducibly global: the per-slot (i, ch) denominator is the number of valid pairs that
    /// contributed to that slot, so the whole (N, L) sweep must complete before any force or
    /// momentum row can be formed. Runs taxon-major rather than site-major; within a fixed slot
    /// the only free index is `u` and both accumulation predicates are visit-order independent,
    /// so each slot still receives its addends in ascending `u` and the f64 addition chain is
    /// bit-identical to the whole-tensor path.
    pub fn compute_mean_attention(
        &self,
        aln: &Alignment,
        landmarks: &LandmarkSet,
        m_matrix: &[f64],
        b_prior: &[f64],
        root_sequence: &[u8],
        masks: &AttentionMasks,
    ) -> Vec<f64> {
        let n = aln.num_taxa;
        let k = landmarks.num_landmarks;
        let l = aln.length;
        let num_channels = k + 1;
        assert!(num_channels <= 128, "K+1 exceeds stack buffer limit");

        let mut sum_a = vec![0.0f64; n * num_channels];
        let mut valid_pair_counts = vec![0usize; n * num_channels];

        // Single cell buffer, hoisted above both loops: zero heap allocation per site.
        let mut cell = [0.0f64; 128];

        for i in 0..n {
            let in_offset = i * num_channels;
            for u in 0..l {
                let valid_q = self.softmax_row_at(
                    aln,
                    landmarks,
                    m_matrix,
                    b_prior,
                    root_sequence,
                    masks,
                    i,
                    u,
                    num_channels,
                    &mut cell[..num_channels],
                );
                if !valid_q {
                    continue;
                }

                sum_a[in_offset + 0] += cell[0];
                valid_pair_counts[in_offset + 0] += 1;

                for p in 0..k {
                    let ch = p + 1;
                    if !masks.mask_landmark[i * k + p] {
                        sum_a[in_offset + ch] += cell[ch];
                        valid_pair_counts[in_offset + ch] += 1;
                    }
                }
            }
        }

        let mut mean_a = vec![0.0f64; n * num_channels];
        for idx in 0..(n * num_channels) {
            let cnt = valid_pair_counts[idx];
            if cnt > 0 {
                mean_a[idx] = sum_a[idx] / (cnt as f64);
            }
        }
        mean_a
    }

    /// Fills one taxon's attention row A_i of shape (L, K+1), indexed `u * num_channels + ch`.
    ///
    /// Returns the *patched* row the whole-tensor path holds once Pass 2 has run, not the raw
    /// softmax: every scrubbed, gapped or masked cell carries mean_A for its channel. Parental
    /// attribution reads exactly this patched A, so the patch is load-bearing. `row` must be
    /// exactly `L * (K+1)`; every element is written.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_attention_row(
        &self,
        aln: &Alignment,
        landmarks: &LandmarkSet,
        m_matrix: &[f64],
        b_prior: &[f64],
        root_sequence: &[u8],
        masks: &AttentionMasks,
        mean_a: &[f64],
        i: usize,
        row: &mut [f64],
    ) {
        let k = landmarks.num_landmarks;
        let l = aln.length;
        let num_channels = k + 1;
        assert_eq!(row.len(), l * num_channels, "attention row buffer mismatch");

        let in_offset = i * num_channels;

        for u in 0..l {
            let cell_offset = u * num_channels;
            let valid_q = self.softmax_row_at(
                aln,
                landmarks,
                m_matrix,
                b_prior,
                root_sequence,
                masks,
                i,
                u,
                num_channels,
                &mut row[cell_offset..cell_offset + num_channels],
            );

            // Pass 2 patch: an invalid or masked cell carries the channel mean, not raw softmax.
            if !valid_q {
                row[cell_offset + 0] = mean_a[in_offset + 0];
            }
            for p in 0..k {
                let ch = p + 1;
                let valid_pair = valid_q && !masks.mask_landmark[i * k + p];
                if !valid_pair {
                    row[cell_offset + ch] = mean_a[in_offset + ch];
                }
            }
        }
    }

    /// Fills one taxon's zero-drift force row F_i = A_i - mean_A_i (with F = 0 on scrubbed,
    /// gapped and masked cells) and cumulative momentum row C_i, both of shape (L, K+1) indexed
    /// `u * num_channels + ch`.
    ///
    /// `mean_a` is the (N, K+1) sequence-wide mean from [`Self::compute_mean_attention`]. The
    /// whole-tensor Pass 2 is already taxon-major with `running_c` reset per taxon, so streaming
    /// one row changes nothing but the index arithmetic. Both buffers must be exactly
    /// `L * (K+1)`; every element is written.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_force_and_momentum_row(
        &self,
        aln: &Alignment,
        landmarks: &LandmarkSet,
        m_matrix: &[f64],
        b_prior: &[f64],
        root_sequence: &[u8],
        masks: &AttentionMasks,
        mean_a: &[f64],
        i: usize,
        f_row: &mut [f64],
        c_row: &mut [f64],
    ) {
        let k = landmarks.num_landmarks;
        let l = aln.length;
        let num_channels = k + 1;
        assert_eq!(f_row.len(), l * num_channels, "force row buffer mismatch");
        assert_eq!(
            c_row.len(),
            l * num_channels,
            "momentum row buffer mismatch"
        );
        assert!(num_channels <= 128, "K+1 exceeds stack buffer limit");

        let in_offset = i * num_channels;
        // Hoisted above the u loop: zero heap allocation per site.
        let mut running_c = [0.0f64; 128];
        let mut a_cell = [0.0f64; 128];

        for u in 0..l {
            let cell_offset = u * num_channels;
            let valid_q = self.softmax_row_at(
                aln,
                landmarks,
                m_matrix,
                b_prior,
                root_sequence,
                masks,
                i,
                u,
                num_channels,
                &mut a_cell[..num_channels],
            );

            // Channel 0 (Root)
            let f_root = if valid_q {
                a_cell[0] - mean_a[in_offset + 0]
            } else {
                0.0
            };
            f_row[cell_offset + 0] = f_root;
            running_c[0] += f_root;
            c_row[cell_offset + 0] = running_c[0];

            // Landmark channels 1..=K
            for p in 0..k {
                let ch = p + 1;
                let valid_pair = valid_q && !masks.mask_landmark[i * k + p];
                let f_val = if valid_pair {
                    a_cell[ch] - mean_a[in_offset + ch]
                } else {
                    0.0
                };
                f_row[cell_offset + ch] = f_val;
                running_c[ch] += f_val;
                c_row[cell_offset + ch] = running_c[ch];
            }
        }
    }

    /// Computes A_tensor, F_tensor and C_tensor of shape (L, N, K+1).
    /// Flat indexing: `u * (N * num_channels) + i * num_channels + ch`
    ///
    /// Allocates 3 * L * N * (K+1) f64 up front, so production paths stream single taxon rows
    /// via [`Self::compute_mean_attention`], [`Self::compute_attention_row`] and
    /// [`Self::compute_force_and_momentum_row`] instead. Retained, routed through the same
    /// [`Self::softmax_row_at`] kernel so exactly one copy of the arithmetic exists and cannot
    /// drift, and serves as the reference oracle the row providers are pinned against (see the
    /// bit-identity tests below).
    pub fn compute_attention_and_force_tensors(
        &self,
        aln: &Alignment,
        landmarks: &LandmarkSet,
        m_matrix: &[f64],
        b_prior: &[f64],
        root_sequence: &[u8],
        masks: &AttentionMasks,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = aln.num_taxa;
        let k = landmarks.num_landmarks;
        let l = aln.length;
        let num_channels = k + 1;
        let total_size = l * n * num_channels;

        let mut a_tensor = vec![0.0f64; total_size];
        let mut f_tensor = vec![0.0f64; total_size];
        let mut c_tensor = vec![0.0f64; total_size];

        let mut sum_a = vec![0.0f64; n * num_channels];
        let mut valid_pair_counts = vec![0usize; n * num_channels];

        // Pass 1: Site-by-site softmax attention
        for u in 0..l {
            let u_offset = u * n * num_channels;

            for i in 0..n {
                let in_offset = i * num_channels;
                let out_offset = u_offset + in_offset;

                let valid_q = self.softmax_row_at(
                    aln,
                    landmarks,
                    m_matrix,
                    b_prior,
                    root_sequence,
                    masks,
                    i,
                    u,
                    num_channels,
                    &mut a_tensor[out_offset..out_offset + num_channels],
                );
                if !valid_q {
                    continue;
                }

                sum_a[in_offset + 0] += a_tensor[out_offset + 0];
                valid_pair_counts[in_offset + 0] += 1;

                for p in 0..k {
                    let ch = p + 1;
                    if !masks.mask_landmark[i * k + p] {
                        sum_a[in_offset + ch] += a_tensor[out_offset + ch];
                        valid_pair_counts[in_offset + ch] += 1;
                    }
                }
            }
        }

        // Empirical sequence-wide mean attention over valid (un-scrubbed) sites: mean_A in (N, K+1)
        let mut mean_a = vec![0.0f64; n * num_channels];
        for idx in 0..(n * num_channels) {
            let cnt = valid_pair_counts[idx];
            if cnt > 0 {
                mean_a[idx] = sum_a[idx] / (cnt as f64);
            }
        }

        // Pass 2: Zero-drift Force Field F = A - mean_A (with F = 0 on scrubbed/gapped sites) and cumulative momentum C
        for i in 0..n {
            let in_offset = i * num_channels;
            let mut running_c = vec![0.0f64; num_channels];

            for u in 0..l {
                let valid_q = aln.get(i, u) > 0;
                let cell_offset = u * n * num_channels + in_offset;

                // Channel 0 (Root)
                let f_root = if valid_q {
                    a_tensor[cell_offset + 0] - mean_a[in_offset + 0]
                } else {
                    a_tensor[cell_offset + 0] = mean_a[in_offset + 0];
                    0.0
                };
                f_tensor[cell_offset + 0] = f_root;
                running_c[0] += f_root;
                c_tensor[cell_offset + 0] = running_c[0];

                // Landmark channels 1..=K
                for p in 0..k {
                    let ch = p + 1;
                    let valid_pair = valid_q && !masks.mask_landmark[i * k + p];
                    let f_val = if valid_pair {
                        a_tensor[cell_offset + ch] - mean_a[in_offset + ch]
                    } else {
                        a_tensor[cell_offset + ch] = mean_a[in_offset + ch];
                        0.0
                    };
                    f_tensor[cell_offset + ch] = f_val;
                    running_c[ch] += f_val;
                    c_tensor[cell_offset + ch] = running_c[ch];
                }
            }
        }

        (a_tensor, f_tensor, c_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::landmarks::select_landmarks;
    use crate::prior::PhyloBias;
    use crate::soft_root::compute_consensus_soft_root;
    use crate::tree_rope::TreeRoPE;

    /// Deterministic clade-structured alignment with gaps and twin clones, so the masking,
    /// `!valid_q` skip and `valid_pair_counts` paths are all genuinely exercised rather than
    /// left dead. `num_clades` founder sequences are mutated at a low per-site rate to form
    /// clades; the last taxa are near-identical twins of their clade founder.
    fn structured_alignment(n: usize, l: usize, num_clades: usize) -> Alignment {
        let mut state: u64 = 0x5EED_1234_ABCD_0001;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };

        let mut founders = vec![vec![0u8; l]; num_clades];
        for founder in founders.iter_mut() {
            for cell in founder.iter_mut() {
                *cell = (next() % 4 + 1) as u8;
            }
        }

        let mut matrix = vec![0u8; n * l];
        for i in 0..n {
            let clade = i % num_clades;
            // The trailing taxa are twins: no divergence from their clade founder at all.
            let is_twin = i >= n.saturating_sub(2 * num_clades);
            for u in 0..l {
                let mut c = founders[clade][u];
                if !is_twin && next() % 100 < 4 {
                    c = (next() % 4 + 1) as u8;
                }
                // Sparse gaps drive the `!valid_q` skip and make cnt < L.
                if next() % 500 == 0 {
                    c = 0;
                }
                matrix[i * l + u] = c;
            }
        }

        Alignment {
            taxa: (0..n).map(|i| format!("t{}", i)).collect(),
            num_taxa: n,
            length: l,
            matrix,
        }
    }

    struct Fixture {
        aln: Alignment,
        landmarks: LandmarkSet,
        masks: AttentionMasks,
        m_matrix: Vec<f64>,
        b_prior: Vec<f64>,
        root_seq: Vec<u8>,
        engine: PhyloAttentionEngine,
    }

    fn fixture(n: usize, l: usize, k: usize) -> Fixture {
        let aln = structured_alignment(n, l, 4);
        let landmarks = select_landmarks(&aln, k);
        let masks = compute_attention_masks(&aln, &landmarks);
        let rope = TreeRoPE::new(64);
        let m_matrix = rope.compute_query_landmark_rotary_factors(&aln, &landmarks);
        let prior = PhyloBias::new(0.05, 1.0);
        let b_prior = prior.compute_prior_matrix(&aln, &landmarks);
        let root_seq = compute_consensus_soft_root(&aln, None);
        Fixture {
            aln,
            landmarks,
            masks,
            m_matrix,
            b_prior,
            root_seq,
            engine: PhyloAttentionEngine::new(64, 1.0),
        }
    }

    /// Pins the bit-identity of the row providers against the whole-tensor oracle across a
    /// small cohort, N = 65 (just above the K-saturation boundary) and the K = 32 saturated
    /// path. Exact `to_bits()` equality is deliberate: an epsilon tolerance would pass on
    /// precisely the arithmetic drift these tests exist to catch.
    #[test]
    fn test_row_providers_bit_identical_to_tensors() {
        for &(n, l, k) in &[
            (9usize, 140usize, 8usize),
            (65, 90, 32),
            (40, 110, 32),
            (7, 200, 6),
        ] {
            let f = fixture(n, l, k);
            let num_channels = f.landmarks.num_landmarks + 1;

            let (a_tensor, f_tensor, c_tensor) = f.engine.compute_attention_and_force_tensors(
                &f.aln,
                &f.landmarks,
                &f.m_matrix,
                &f.b_prior,
                &f.root_seq,
                &f.masks,
            );

            let mean_a = f.engine.compute_mean_attention(
                &f.aln,
                &f.landmarks,
                &f.m_matrix,
                &f.b_prior,
                &f.root_seq,
                &f.masks,
            );

            let mut a_row = vec![0.0f64; l * num_channels];
            let mut f_row = vec![0.0f64; l * num_channels];
            let mut c_row = vec![0.0f64; l * num_channels];

            for i in 0..n {
                f.engine.compute_attention_row(
                    &f.aln,
                    &f.landmarks,
                    &f.m_matrix,
                    &f.b_prior,
                    &f.root_seq,
                    &f.masks,
                    &mean_a,
                    i,
                    &mut a_row,
                );
                f.engine.compute_force_and_momentum_row(
                    &f.aln,
                    &f.landmarks,
                    &f.m_matrix,
                    &f.b_prior,
                    &f.root_seq,
                    &f.masks,
                    &mean_a,
                    i,
                    &mut f_row,
                    &mut c_row,
                );

                for u in 0..l {
                    let cell = u * n * num_channels + i * num_channels;
                    let row_cell = u * num_channels;
                    for ch in 0..num_channels {
                        assert_eq!(
                            a_row[row_cell + ch].to_bits(),
                            a_tensor[cell + ch].to_bits(),
                            "a_row drift at (i={}, u={}, ch={}) for n={} l={} k={}",
                            i,
                            u,
                            ch,
                            n,
                            l,
                            k
                        );
                        assert_eq!(
                            f_row[row_cell + ch].to_bits(),
                            f_tensor[cell + ch].to_bits(),
                            "f_row drift at (i={}, u={}, ch={}) for n={} l={} k={}",
                            i,
                            u,
                            ch,
                            n,
                            l,
                            k
                        );
                        assert_eq!(
                            c_row[row_cell + ch].to_bits(),
                            c_tensor[cell + ch].to_bits(),
                            "c_row drift at (i={}, u={}, ch={}) for n={} l={} k={}",
                            i,
                            u,
                            ch,
                            n,
                            l,
                            k
                        );
                    }
                }
            }
        }
    }

    /// The whole-tensor oracle derives mean_A under the site-major Pass 1 nest and
    /// `compute_mean_attention` under the taxon-major one. Every (i, ch) accumulator slot must
    /// still receive the same addends in the same ascending-`u` order, so the two f64 addition
    /// chains are bit-identical.
    #[test]
    fn test_mean_attention_invariant_to_pass1_loop_order() {
        for &(n, l, k_req) in &[(9usize, 140usize, 8usize), (65, 90, 32), (40, 110, 32)] {
            let f = fixture(n, l, k_req);
            // `select_landmarks` may return more or fewer than requested, so the stride of
            // `mask_landmark` is the realized K, never the requested one.
            let k = f.landmarks.num_landmarks;
            let num_channels = k + 1;

            // Site-major reference: the oracle's A is left unpatched on exactly the cells that
            // contributed to Pass 1, so re-reducing it site-major recovers the genuine sums.
            let (a_tensor, _f_tensor, _c_tensor) = f.engine.compute_attention_and_force_tensors(
                &f.aln,
                &f.landmarks,
                &f.m_matrix,
                &f.b_prior,
                &f.root_seq,
                &f.masks,
            );
            let mut sum_a = vec![0.0f64; n * num_channels];
            let mut counts = vec![0usize; n * num_channels];
            for u in 0..l {
                for i in 0..n {
                    if f.aln.get(i, u) == 0 {
                        continue;
                    }
                    let cell = u * n * num_channels + i * num_channels;
                    let slot = i * num_channels;
                    sum_a[slot + 0] += a_tensor[cell + 0];
                    counts[slot + 0] += 1;
                    for p in 0..k {
                        let ch = p + 1;
                        if !f.masks.mask_landmark[i * k + p] {
                            sum_a[slot + ch] += a_tensor[cell + ch];
                            counts[slot + ch] += 1;
                        }
                    }
                }
            }

            let mean_a = f.engine.compute_mean_attention(
                &f.aln,
                &f.landmarks,
                &f.m_matrix,
                &f.b_prior,
                &f.root_seq,
                &f.masks,
            );

            let mut fully_masked_slots = 0usize;
            let mut partial_count_slots = 0usize;
            for idx in 0..(n * num_channels) {
                let reference = if counts[idx] > 0 {
                    if counts[idx] < l {
                        partial_count_slots += 1;
                    }
                    sum_a[idx] / (counts[idx] as f64)
                } else {
                    fully_masked_slots += 1;
                    0.0
                };
                assert_eq!(
                    mean_a[idx].to_bits(),
                    reference.to_bits(),
                    "mean_a drift at slot {} for n={} l={} k_req={}",
                    idx,
                    n,
                    l,
                    k_req
                );
            }

            // Guards the fixture, not the engine: without a masked channel and without a
            // gapped taxon, the masked-channel and cnt < L branches would go untested.
            assert!(
                fully_masked_slots > 0,
                "fixture n={} l={} k_req={} exercises no fully masked channel",
                n,
                l,
                k_req
            );
            assert!(
                partial_count_slots > 0,
                "fixture n={} l={} k_req={} exercises no cnt < L denominator",
                n,
                l,
                k_req
            );
        }
    }

    /// Buffer reuse across taxa is the one new aliasing hazard the row-streaming design
    /// introduces: a stale tail from a previous row must not survive into a reduction.
    #[test]
    fn test_row_buffers_fully_overwritten_on_reuse() {
        let f = fixture(12, 80, 8);
        let num_channels = f.landmarks.num_landmarks + 1;
        let l = f.aln.length;
        let mean_a = f.engine.compute_mean_attention(
            &f.aln,
            &f.landmarks,
            &f.m_matrix,
            &f.b_prior,
            &f.root_seq,
            &f.masks,
        );

        let mut clean_a = vec![0.0f64; l * num_channels];
        let mut clean_f = vec![0.0f64; l * num_channels];
        let mut clean_c = vec![0.0f64; l * num_channels];
        f.engine.compute_attention_row(
            &f.aln,
            &f.landmarks,
            &f.m_matrix,
            &f.b_prior,
            &f.root_seq,
            &f.masks,
            &mean_a,
            5,
            &mut clean_a,
        );
        f.engine.compute_force_and_momentum_row(
            &f.aln,
            &f.landmarks,
            &f.m_matrix,
            &f.b_prior,
            &f.root_seq,
            &f.masks,
            &mean_a,
            5,
            &mut clean_f,
            &mut clean_c,
        );

        let mut dirty_a = vec![f64::NAN; l * num_channels];
        let mut dirty_f = vec![f64::NAN; l * num_channels];
        let mut dirty_c = vec![f64::NAN; l * num_channels];
        for &i in &[11usize, 5] {
            f.engine.compute_attention_row(
                &f.aln,
                &f.landmarks,
                &f.m_matrix,
                &f.b_prior,
                &f.root_seq,
                &f.masks,
                &mean_a,
                i,
                &mut dirty_a,
            );
            f.engine.compute_force_and_momentum_row(
                &f.aln,
                &f.landmarks,
                &f.m_matrix,
                &f.b_prior,
                &f.root_seq,
                &f.masks,
                &mean_a,
                i,
                &mut dirty_f,
                &mut dirty_c,
            );
        }

        for idx in 0..clean_f.len() {
            assert_eq!(dirty_a[idx].to_bits(), clean_a[idx].to_bits());
            assert_eq!(dirty_f[idx].to_bits(), clean_f[idx].to_bits());
            assert_eq!(dirty_c[idx].to_bits(), clean_c[idx].to_bits());
        }
    }
}
