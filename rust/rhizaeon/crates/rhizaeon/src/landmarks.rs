use crate::jacobi::symmetric_jacobi_eigh;
use crate::types::{Alignment, LandmarkSet};

/// Computes normalized Hamming distance between two sequence rows, excluding non-informative / gap sites
pub fn compute_pairwise_distance(aln: &Alignment, i: usize, j: usize) -> f64 {
    if i == j {
        return 0.0;
    }
    let row_i = aln.row(i);
    let row_j = aln.row(j);
    let mut valid_count = 0usize;
    let mut diff_count = 0usize;

    let site_stride = if aln.num_taxa > 500 && aln.length > 512 {
        (aln.length / 256).max(1)
    } else {
        1
    };

    let mut u = 0usize;
    while u < aln.length {
        let ci = row_i[u];
        let cj = row_j[u];
        if ci > 0 && cj > 0 {
            valid_count += 1;
            if ci != cj {
                diff_count += 1;
            }
        }
        u += site_stride;
    }

    if valid_count == 0 {
        1.0
    } else {
        (diff_count as f64) / (valid_count as f64)
    }
}

/// Jukes-Cantor distance correction: d = -3/4 * ln(1 - 4/3 * p)
#[inline(always)]
pub fn jukes_cantor_distance(p: f64) -> f64 {
    let clamped_p = p.min(0.7499).max(0.0);
    -0.75 * (1.0 - (4.0 / 3.0) * clamped_p).ln()
}

/// Tests whether two sequences (i, j) are true phylogenetic sisters / co-recombinant twins
/// rather than a (Recombinant, Pure Home) pair separated by a localized donor cassette.
/// True sisters share the same mosaic history and only differ by independent terminal mutations,
/// so their distances to any external taxon k != i, j satisfy |d(i, k) - d(j, k)| <= 0.45 * d(i, j).
pub fn is_true_sister_pair(aln: &Alignment, i: usize, j: usize, d_sister: f64) -> bool {
    if i == j {
        return true;
    }
    let dij = compute_pairwise_distance(aln, i, j);
    if dij >= d_sister {
        return false;
    }
    if dij <= 1e-9 || aln.num_taxa <= 2 {
        return true;
    }
    let mut max_asym = 0.0f64;
    let mut min_signed = f64::INFINITY;
    let mut max_signed = f64::NEG_INFINITY;
    let stride = (aln.num_taxa / 48).max(1);
    let mut k = 0usize;
    while k < aln.num_taxa {
        if k != i && k != j {
            let dik = compute_pairwise_distance(aln, i, k);
            let djk = compute_pairwise_distance(aln, j, k);
            let signed = dik - djk;
            let asym = signed.abs();
            if asym > max_asym {
                max_asym = asym;
            }
            if signed < min_signed {
                min_signed = signed;
            }
            if signed > max_signed {
                max_signed = signed;
            }
        }
        k += stride;
    }
    let spread = 0.5 * (max_signed - min_signed).max(0.0);
    max_asym <= 0.45 * dij + 1e-6 || (aln.num_taxa > 32 && spread <= 0.30 * dij + 1e-6)
}

/// Computes empirical scale-invariant sister and clade distance thresholds from the alignment.
/// - d_sister: Poisson transmission drift floor, bounded below by 3 / L
/// - d_clade: Spectral gap separating intra-lineage from inter-lineage distances (or 10th percentile fallback), bounded below by 5 / L
pub fn compute_scale_invariant_distance_thresholds(aln: &Alignment) -> (f64, f64) {
    let l_f64 = aln.length.max(1) as f64;
    let sister_floor = 0.008f64.max(3.0 / l_f64);
    let clade_floor = 0.015f64.max(5.0 / l_f64);

    let n = aln.num_taxa;
    if n <= 1 {
        return (sister_floor, clade_floor);
    }

    let max_samples = 2048usize;
    let mut dists = Vec::with_capacity(max_samples + 16);
    let mut min_pair = (0usize, 0usize, f64::INFINITY);

    if n <= 128 {
        let total_pairs = (n * (n - 1)) / 2;
        let stride = (total_pairs / max_samples).max(1);
        let mut count = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                count += 1;
                if count % stride == 0 {
                    let d = compute_pairwise_distance(aln, i, j);
                    if d > 0.0 {
                        dists.push(d);
                        if d < min_pair.2 {
                            min_pair = (i, j, d);
                        }
                    }
                }
            }
        }
    } else {
        let mut sample_indices = Vec::with_capacity(96);
        let mut seen_haps: std::collections::HashSet<&[u8]> = std::collections::HashSet::with_capacity(96);
        for i in 0..n.min(24) {
            if seen_haps.insert(aln.row(i)) {
                sample_indices.push(i);
            }
        }
        let step = (n / 96).max(1);
        let mut idx = 0usize;
        while idx < n && sample_indices.len() < 96 {
            if seen_haps.insert(aln.row(idx)) {
                sample_indices.push(idx);
            }
            idx += step;
        }
        if sample_indices.len() < 48 {
            for i in 0..n {
                if sample_indices.len() >= 96 {
                    break;
                }
                if seen_haps.insert(aln.row(i)) {
                    sample_indices.push(i);
                }
            }
        }
        for a in 0..sample_indices.len() {
            let i = sample_indices[a];
            for b in (a + 1)..sample_indices.len() {
                let j = sample_indices[b];
                let d = compute_pairwise_distance(aln, i, j);
                if d > 0.0 {
                    dists.push(d);
                    if d < min_pair.2 {
                        min_pair = (i, j, d);
                    }
                }
            }
        }
    }

    if dists.is_empty() {
        return (sister_floor, clade_floor);
    }

    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx_75 = (0.75 * (dists.len() as f64)).floor() as usize;
    let d_p75 = dists[idx_75.min(dists.len() - 1)];
    let eff_sister_floor = (3.0 / l_f64).max(sister_floor.min(0.45 * d_p75));
    let eff_clade_floor = (5.0 / l_f64).max(clade_floor.min(0.75 * d_p75));

    let idx_01 = (0.01 * (dists.len() as f64)).floor() as usize;
    let d_p01 = dists[idx_01.min(dists.len() - 1)];
    // On small cohorts (`idx_01 == 0`), `dists[0]` is the single closest pair `(min_i, min_j)`.
    // Only include `dists[0]` inside the strict `< d_sister` threshold if `(min_i, min_j)` are
    // true phylogenetic sisters sharing the same distance profile to all other taxa
    // (`max_{k != i,j} |d(i,k) - d(j,k)| <= 0.35 * d(i,j)`), rather than `(Recombinant, Home)`.
    let is_true_sister_min_pair = if idx_01 == 0 && d_p01 <= 0.45 * d_p75 && n >= 4 {
        let (mi, mj, md) = min_pair;
        let mut max_prof_diff = 0.0f64;
        let k_step = (n / 64).max(1);
        let mut k = 0usize;
        while k < n {
            if k != mi && k != mj {
                let dik = compute_pairwise_distance(aln, mi, k);
                let djk = compute_pairwise_distance(aln, mj, k);
                let diff = (dik - djk).abs();
                if diff > max_prof_diff {
                    max_prof_diff = diff;
                }
            }
            k += k_step;
        }
        max_prof_diff <= 0.35 * md
    } else {
        false
    };
    let mut d_sister = if is_true_sister_min_pair {
        (d_p01 + 1e-9).max(eff_sister_floor)
    } else {
        d_p01.max(eff_sister_floor)
    };

    let idx_10 = (0.10 * (dists.len() as f64)).floor() as usize;
    let raw_p10 = dists[idx_10.min(dists.len() - 1)];
    let d_p10 = raw_p10
        .max(eff_clade_floor)
        .max(1.5 * d_sister);

    // Search for a multimodal inter-lineage spectral gap in [p05, p50]
    let mut d_clade = d_p10;
    if dists.len() >= 28 {
        let s_idx = ((0.05 * (dists.len() as f64)).floor() as usize).max(1);
        let e_idx = ((0.50 * (dists.len() as f64)).floor() as usize).min(dists.len() - 1);
        if e_idx > s_idx {
            let mut max_gap = 0.0f64;
            let mut best_k = s_idx;
            let span_ref = (dists[e_idx] - dists[s_idx]).max(1e-9);
            let upper_mode_bound = if n > 64 {
                (2.2 * raw_p10.max(d_sister)).min(3.0 * d_p10)
            } else {
                3.0 * d_p10
            };
            for k in s_idx..e_idx {
                if dists[k] >= d_sister && dists[k] <= upper_mode_bound {
                    let gap = dists[k + 1] - dists[k];
                    if gap > max_gap {
                        max_gap = gap;
                        best_k = k;
                    }
                }
            }
            // Adopt the spectral gap midpoint when a clear inter-population jump exists
            let rel_ratio = dists[best_k + 1] / dists[best_k].max(1e-9);
            if max_gap >= 0.15 * span_ref && rel_ratio >= 1.25 {
                let gap_mid = 0.5 * (dists[best_k] + dists[best_k + 1]);
                d_clade = gap_mid.max(d_p10);
                if n > 64 && rel_ratio >= 1.35 {
                    d_sister = d_sister.max(dists[best_k] + 1e-6);
                }
            }
        }
    }

    (d_sister, d_clade)
}

/// Partitions alignment taxa into connected lineage clusters at distance threshold `d_clade`
pub fn partition_lineage_clusters(aln: &Alignment, d_clade: f64) -> Vec<Vec<usize>> {
    let n = aln.num_taxa;
    let mut visited = vec![false; n];
    let mut clusters = Vec::new();

    for i in 0..n {
        if !visited[i] {
            let mut comp = Vec::new();
            let mut queue = vec![i];
            visited[i] = true;
            while let Some(curr) = queue.pop() {
                comp.push(curr);
                for j in 0..n {
                    if !visited[j] && compute_pairwise_distance(aln, curr, j) <= d_clade {
                        visited[j] = true;
                        queue.push(j);
                    }
                }
            }
            comp.sort_unstable();
            clusters.push(comp);
        }
    }
    clusters
}

/// Builds Full-Cohort LandmarkSet when N <= 64 (zero-subsampling manifold tracking)
pub fn build_full_cohort_landmarks(alignment: &Alignment) -> LandmarkSet {
    let n = alignment.num_taxa;
    let selected_indices: Vec<usize> = (0..n).collect();
    let landmark_taxa: Vec<String> = alignment.taxa.clone();

    let mut d_sq = vec![0.0f64; n * n];
    for p in 0..n {
        for q in 0..n {
            let dist = if p == q {
                0.0
            } else {
                compute_pairwise_distance(alignment, p, q)
            };
            let jc = jukes_cantor_distance(dist);
            d_sq[p * n + q] = jc * jc;
        }
    }

    let mut row_means = vec![0.0f64; n];
    let mut total_sum = 0.0f64;
    for p in 0..n {
        let mut s = 0.0;
        for q in 0..n {
            s += d_sq[p * n + q];
        }
        row_means[p] = s / (n as f64);
        total_sum += s;
    }
    let grand_mean = total_sum / ((n * n) as f64);

    let mut b = vec![0.0f64; n * n];
    for p in 0..n {
        for q in 0..n {
            let val = -0.5 * (d_sq[p * n + q] - row_means[p] - row_means[q] + grand_mean);
            b[p * n + q] = val;
        }
    }

    let eigh = symmetric_jacobi_eigh(&b, n);
    let lam0 = eigh
        .eigenvalues
        .get(0)
        .copied()
        .unwrap_or(0.0)
        .max(1e-9)
        .sqrt();
    let lam1 = eigh
        .eigenvalues
        .get(1)
        .copied()
        .unwrap_or(0.0)
        .max(1e-9)
        .sqrt();

    let mut coords = vec![0.0f64; (n + 1) * 2];
    let mut root_to_tip = Vec::with_capacity(n);

    for p in 0..n {
        let x = eigh.eigenvectors[p * n + 0] * lam0;
        let y = if n > 1 {
            eigh.eigenvectors[p * n + 1] * lam1
        } else {
            0.0
        };

        let radius = (x * x + y * y).sqrt().max(1e-6);
        let mut angle = y.atan2(x);
        if angle < 0.0 {
            angle += 2.0 * std::f64::consts::PI;
        }

        root_to_tip.push(radius);

        let out_idx = (p + 1) * 2;
        coords[out_idx] = radius;
        coords[out_idx + 1] = angle;
    }

    LandmarkSet {
        indices: selected_indices,
        taxa: landmark_taxa,
        num_landmarks: n,
        coords,
        root_to_tip,
    }
}

/// Selects K landmarks and builds exact LandmarkSet with uncorrupted tree coordinates
pub fn select_landmarks(alignment: &Alignment, target_k: usize) -> LandmarkSet {
    select_landmarks_from_pool(alignment, target_k, None)
}

/// Selects K landmarks strictly from an allowed pool of indices (e.g. verified clonal backbone)
pub fn select_landmarks_from_pool(
    alignment: &Alignment,
    target_k: usize,
    pool_opt: Option<&[usize]>,
) -> LandmarkSet {
    let n = alignment.num_taxa;

    // Full-cohort resolution for small/medium cohorts when unconstrained
    if n <= 64 && pool_opt.is_none() {
        return build_full_cohort_landmarks(alignment);
    }

    let raw_pool: Vec<usize> = match pool_opt {
        Some(p) if !p.is_empty() => p.to_vec(),
        _ => (0..n).collect(),
    };

    let pool: Vec<usize> = if raw_pool.len() > 256 {
        let mut sub = Vec::with_capacity(256);
        let mut seen_haps: std::collections::HashSet<&[u8]> = std::collections::HashSet::with_capacity(256);
        for &idx in raw_pool.iter().take(32) {
            if seen_haps.insert(alignment.row(idx)) {
                sub.push(idx);
            }
        }
        let step = (raw_pool.len() / 256).max(1);
        for &idx in raw_pool.iter().step_by(step) {
            if sub.len() >= 256 {
                break;
            }
            if seen_haps.insert(alignment.row(idx)) {
                sub.push(idx);
            }
        }
        if sub.len() < 128 {
            for &idx in &raw_pool {
                if sub.len() >= 256 {
                    break;
                }
                if seen_haps.insert(alignment.row(idx)) {
                    sub.push(idx);
                }
            }
        }
        if sub.len() < 16 {
            for &idx in raw_pool.iter().step_by(step) {
                if sub.len() >= 256 {
                    break;
                }
                if !sub.contains(&idx) {
                    sub.push(idx);
                }
            }
        }
        sub.sort_unstable();
        sub
    } else {
        raw_pool
    };

    let pool_len = pool.len();
    let max_k = target_k.min(pool_len).max(1);
    let (d_sister, d_clade) = compute_scale_invariant_distance_thresholds(alignment);

    // 1. Seed with the clonal medoid of each lineage cluster present in the pool
    // Pure non-recombinant strains minimize mean intra-cluster distance, ensuring
    // every population has a clean clonal representative before farthest-point sampling.
    let clusters = if n <= 256 {
        partition_lineage_clusters(alignment, d_clade)
    } else {
        let mut visited = vec![false; pool.len()];
        let mut sub_clusters = Vec::new();
        for i_pos in 0..pool.len() {
            if !visited[i_pos] {
                let mut comp = Vec::new();
                let mut queue = vec![i_pos];
                visited[i_pos] = true;
                while let Some(curr_pos) = queue.pop() {
                    let curr = pool[curr_pos];
                    comp.push(curr);
                    for j_pos in 0..pool.len() {
                        if !visited[j_pos]
                            && compute_pairwise_distance(alignment, curr, pool[j_pos]) <= d_clade
                        {
                            visited[j_pos] = true;
                            queue.push(j_pos);
                        }
                    }
                }
                comp.sort_unstable();
                sub_clusters.push(comp);
            }
        }
        sub_clusters
    };
    let mut selected_indices = Vec::with_capacity(max_k);

    for comp in &clusters {
        if selected_indices.len() >= max_k {
            break;
        }
        let comp_in_pool: Vec<usize> = comp.iter().copied().filter(|i| pool.contains(i)).collect();
        if comp_in_pool.is_empty() || (comp_in_pool.len() < 2 && pool_len > max_k) {
            continue;
        }
        let mut best_medoid = comp_in_pool[0];
        let mut best_mean_d = f64::INFINITY;
        let mut best_valid = 0usize;
        let stride_m = (comp_in_pool.len() / 24).max(1);
        for (idx_i, &i) in comp_in_pool.iter().enumerate() {
            if comp_in_pool.len() > 48 && idx_i % stride_m != 0 {
                continue;
            }
            let mut sum_d = 0.0f64;
            let mut cnt_j = 0usize;
            for (idx_j, &j) in comp_in_pool.iter().enumerate() {
                if idx_j % stride_m == 0 {
                    if i != j {
                        sum_d += compute_pairwise_distance(alignment, i, j);
                    }
                    cnt_j += 1;
                }
            }
            let mean_d = sum_d / (cnt_j.max(1) as f64);
            let valid = aln_row_valid_count(alignment, i);
            if mean_d < best_mean_d - 1e-12
                || ((mean_d - best_mean_d).abs() <= 1e-12 && valid > best_valid)
            {
                best_mean_d = mean_d;
                best_valid = valid;
                best_medoid = i;
            }
        }
        if !selected_indices.contains(&best_medoid) {
            selected_indices.push(best_medoid);
        }
    }

    if selected_indices.is_empty() {
        let mut best_first_idx = pool[0];
        let mut max_valid = 0;
        for &i in &pool {
            let row = aln_row_valid_count(alignment, i);
            if row > max_valid {
                max_valid = row;
                best_first_idx = i;
            }
        }
        selected_indices.push(best_first_idx);
    }

    if max_k > selected_indices.len() {
        let mut min_dists: Vec<f64> = pool
            .iter()
            .map(|&i| {
                selected_indices
                    .iter()
                    .map(|&s| compute_pairwise_distance(alignment, i, s))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        while selected_indices.len() < max_k {
            let mut best_cand = pool[0];
            let mut max_min_d = -1.0;
            for (idx, &i) in pool.iter().enumerate() {
                if !selected_indices.contains(&i) && min_dists[idx] > max_min_d {
                    max_min_d = min_dists[idx];
                    best_cand = i;
                }
            }
            // Stop adding duplicate haplotypes or noisy terminal-branch clones once all
            // distinct lineages outside d_sister are covered (with at least 3 landmarks).
            if max_min_d <= 1e-12 || (selected_indices.len() >= 3 && max_min_d <= d_sister) {
                break;
            }
            selected_indices.push(best_cand);

            for (idx, &i) in pool.iter().enumerate() {
                let d = compute_pairwise_distance(alignment, i, best_cand);
                if d < min_dists[idx] {
                    min_dists[idx] = d;
                }
            }
        }
    }

    let k = selected_indices.len();
    let landmark_taxa: Vec<String> = selected_indices
        .iter()
        .map(|&i| alignment.taxa[i].clone())
        .collect();

    // 2. Compute Classical MDS on the K landmarks to get 2D polar tree coordinates
    // Pairwise distance matrix D_k
    let mut d_sq = vec![0.0f64; k * k];
    for p in 0..k {
        for q in 0..k {
            let p_idx = selected_indices[p];
            let q_idx = selected_indices[q];
            let dist = if p == q {
                0.0
            } else {
                compute_pairwise_distance(alignment, p_idx, q_idx)
            };
            let jc = jukes_cantor_distance(dist);
            d_sq[p * k + q] = jc * jc;
        }
    }

    // Double centering: B = -0.5 * H * D^2 * H
    // Row means and col means
    let mut row_means = vec![0.0f64; k];
    let mut total_sum = 0.0f64;
    for p in 0..k {
        let mut s = 0.0;
        for q in 0..k {
            s += d_sq[p * k + q];
        }
        row_means[p] = s / (k as f64);
        total_sum += s;
    }
    let grand_mean = total_sum / ((k * k) as f64);

    let mut b = vec![0.0f64; k * k];
    for p in 0..k {
        for q in 0..k {
            let val = -0.5 * (d_sq[p * k + q] - row_means[p] - row_means[q] + grand_mean);
            b[p * k + q] = val;
        }
    }

    let eigh = symmetric_jacobi_eigh(&b, k);

    // Embed into 2D: coord[p] = [ V[p, 0] * sqrt(max(lambda_0, 0)), V[p, 1] * sqrt(max(lambda_1, 0)) ]
    let lam0 = eigh
        .eigenvalues
        .get(0)
        .copied()
        .unwrap_or(0.0)
        .max(1e-9)
        .sqrt();
    let lam1 = eigh
        .eigenvalues
        .get(1)
        .copied()
        .unwrap_or(0.0)
        .max(1e-9)
        .sqrt();

    // Root is at index 0 with coords [0.0, 0.0]
    // Landmarks are at indices 1..=K with polar coords: [t (radius), theta (angle)]
    let mut coords = vec![0.0f64; (k + 1) * 2];
    let mut root_to_tip = Vec::with_capacity(k);

    for p in 0..k {
        let x = eigh.eigenvectors[p * k + 0] * lam0;
        let y = if k > 1 {
            eigh.eigenvectors[p * k + 1] * lam1
        } else {
            0.0
        };

        let radius = (x * x + y * y).sqrt().max(1e-6);
        let mut angle = y.atan2(x);
        if angle < 0.0 {
            angle += 2.0 * std::f64::consts::PI;
        }

        root_to_tip.push(radius);

        let out_idx = (p + 1) * 2;
        coords[out_idx] = radius;
        coords[out_idx + 1] = angle;
    }

    LandmarkSet {
        indices: selected_indices,
        taxa: landmark_taxa,
        num_landmarks: k,
        coords,
        root_to_tip,
    }
}

fn aln_row_valid_count(aln: &Alignment, i: usize) -> usize {
    let row = aln.row(i);
    row.iter().filter(|&&c| c > 0).count()
}
