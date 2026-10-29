use crate::types::Alignment;

/// Computes the column-frequency consensus ancestral soft root sequence in O(N * L) time.
/// Returns a Vec<u8> of length L where root[u] is the consensus nucleotide (1..4) or 0 if all gap.
pub fn compute_consensus_soft_root(aln: &Alignment, clonal_indices: Option<&[usize]>) -> Vec<u8> {
    let mut root_seq = vec![0u8; aln.length];

    for u in 0..aln.length {
        let mut counts = [0usize; 5];

        if let Some(clones) = clonal_indices {
            for &i in clones {
                let c = aln.get(i, u) as usize;
                counts[c] += 1;
            }
        } else {
            for i in 0..aln.num_taxa {
                let c = aln.get(i, u) as usize;
                counts[c] += 1;
            }
        }

        // Find character state 1..4 with maximum frequency
        let mut best_c = 0u8;
        let mut max_count = 0usize;
        for c in 1..=4 {
            if counts[c] > max_count {
                max_count = counts[c];
                best_c = c as u8;
            }
        }

        root_seq[u] = best_c;
    }

    root_seq
}

/// Computes exact mean evolutionary distances to the cohort root in O(N * L) time
/// via the column-frequency identity:
/// d_bar_i = (1 / (N * L)) * sum_{u=1}^L (N - C_u[s_i(u)])
pub fn compute_column_frequency_distances(aln: &Alignment) -> Vec<f64> {
    let n = aln.num_taxa;
    let l = aln.length;
    let mut freq_table = vec![[0usize; 5]; l];

    for u in 0..l {
        for i in 0..n {
            let c = aln.get(i, u) as usize;
            freq_table[u][c] += 1;
        }
    }

    let mut mean_dists = vec![0.0f64; n];
    let normalizer = (n as f64) * (l as f64);

    for i in 0..n {
        let mut sum_diff = 0u64;
        for u in 0..l {
            let c = aln.get(i, u) as usize;
            if c > 0 {
                sum_diff += (n - freq_table[u][c]) as u64;
            } else {
                sum_diff += n as u64; // Gap penalized
            }
        }
        mean_dists[i] = (sum_diff as f64) / normalizer;
    }

    mean_dists
}
