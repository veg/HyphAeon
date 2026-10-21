use clap::Parser;
use rhizaeon::attention::{compute_attention_masks, PhyloAttentionEngine};
use rhizaeon::fasta::parse_fasta;
use rhizaeon::landmarks::{compute_pairwise_distance, jukes_cantor_distance, select_landmarks};
use rhizaeon::prior::PhyloBias;
use rhizaeon::soft_root::compute_consensus_soft_root;
use rhizaeon::tree_rope::TreeRoPE;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "export_figure2_data")]
#[command(
    about = "Exports authentic Tree-RoPE, Markov Prior, and Attention Flux data for Figure 2"
)]
struct Args {
    /// Path to input alignment FASTA (expected taxa prefixes: B., C.,
    /// CRF07, CRF08 prototypes — see paper/figures in veg/rhizaeon-code)
    #[arg(short, long, required = true)]
    input: PathBuf,

    /// Output JSON path (default: fig2_rhizaeon_authentic_data.json in cwd)
    #[arg(short, long, default_value = "fig2_rhizaeon_authentic_data.json")]
    output: PathBuf,

    #[arg(short, long, default_value_t = 8)]
    dim: usize,

    #[arg(short, long, default_value_t = 16)]
    landmarks: usize,
}

#[derive(Serialize, Deserialize, Debug)]
struct Figure2Payload {
    alignment_length: usize,
    num_taxa: usize,
    taxa: Vec<String>,
    b_indices: Vec<usize>,
    c_indices: Vec<usize>,
    crf07_idx: usize,
    crf08_idx: usize,
    num_landmarks: usize,
    landmark_taxa: Vec<String>,
    landmark_subtypes: Vec<String>,
    landmark_indices: Vec<usize>,

    // Panel A: Tree-RoPE data
    pairwise_distances: Vec<f64>,
    pairwise_rotary_factors: Vec<f64>,
    landmark_coords_2d: Vec<Vec<f64>>,
    landmark_phase_angles: Vec<Vec<f64>>,

    // Panel B: Markov substitution prior
    jc_distances: Vec<f64>,
    retention_probs: Vec<f64>,
    epsilon_0: f64,
    decay_lambda: f64,
    num_invariant_sites: usize,
    num_synapomorphic_sites: usize,
    num_crf07_private_sites: usize,
    site_categories: Vec<u8>,
    crf07_sequence: String,
    b_consensus_sequence: String,
    c_consensus_sequence: String,
    root_sequence_str: String,

    // Panel C: Directed Attention Map along the chromosome for CRF07 (AF286230)
    stride: usize,
    u_coords: Vec<usize>,
    crf07_attention_root: Vec<f64>,
    crf07_attention_b: Vec<f64>,
    crf07_attention_c: Vec<f64>,
    crf07_channel_attention: Vec<Vec<f64>>,

    // Panel D: Cumulative Attention Trajectories and Differential Bridge
    crf07_cumsum_b: Vec<f64>,
    crf07_cumsum_c: Vec<f64>,
    crf07_cumsum_root: Vec<f64>,
    crf07_diff_bridge: Vec<f64>,
    crf08_cumsum_b: Vec<f64>,
    crf08_cumsum_c: Vec<f64>,
    crf08_cumsum_root: Vec<f64>,
    crf08_diff_bridge: Vec<f64>,

    // Resolved breakpoints from cumulative turning points
    crf07_bp1: usize,
    crf07_bp2: usize,
    crf08_bp1: usize,
    crf08_bp2: usize,
}

fn main() {
    let args = Args::parse();

    println!("================================================================================");
    println!("  RHIZAEON PA-FF: Authentic Figure 2 Tree-RoPE & Attention Flux Exporter");
    println!("================================================================================");
    println!("  Input alignment: {:?}", args.input);
    println!("  Output JSON:     {:?}", args.output);

    let fasta_content = fs::read_to_string(&args.input).expect("Failed to read FASTA file");
    let aln = parse_fasta(&fasta_content).expect("Failed to parse input FASTA");
    let n = aln.num_taxa;
    let l = aln.length;
    println!("  Loaded {} taxa, {} bp length", n, l);

    let mut b_indices = Vec::new();
    let mut c_indices = Vec::new();
    let mut crf07_indices = Vec::new();
    let mut crf08_indices = Vec::new();

    for (i, name) in aln.taxa.iter().enumerate() {
        if name.starts_with("B.") {
            b_indices.push(i);
        } else if name.starts_with("C.") {
            c_indices.push(i);
        } else if name.contains("CRF07") || name.contains("AF286230") {
            crf07_indices.push(i);
        } else if name.contains("CRF08") || name.contains("AY008715") {
            crf08_indices.push(i);
        }
    }

    let af286230_idx = aln
        .taxa
        .iter()
        .position(|t| t.contains("AF286230"))
        .expect("AF286230 prototype not found");
    let ay008715_idx = aln
        .taxa
        .iter()
        .position(|t| t.contains("AY008715"))
        .expect("AY008715 prototype not found");

    println!(
        "  Taxa composition: {} B, {} C, {} CRF07, {} CRF08",
        b_indices.len(),
        c_indices.len(),
        crf07_indices.len(),
        crf08_indices.len()
    );

    // 1. Select Reference Landmarks on Delta^K
    let effective_k = args.landmarks.min(n);
    let landmarks = select_landmarks(&aln, effective_k);
    let k = landmarks.num_landmarks;
    let num_channels = k + 1;

    let mut landmark_taxa = Vec::with_capacity(k);
    let mut landmark_subtypes = Vec::with_capacity(k);
    for &idx in &landmarks.indices {
        let name = aln.taxa[idx].clone();
        let subtype = if name.starts_with("B.") {
            "Subtype_B".to_string()
        } else if name.starts_with("C.") {
            "Subtype_C".to_string()
        } else {
            "Recombinant".to_string()
        };
        landmark_taxa.push(name);
        landmark_subtypes.push(subtype);
    }

    // 2. Tree-RoPE Rotary Factors and Phase Angles
    let rope = TreeRoPE::new(args.dim);
    let m_matrix = rope.compute_query_landmark_rotary_factors(&aln, &landmarks);

    // Pairwise distances and rotary attenuation across landmarks
    let mut pairwise_distances = Vec::with_capacity(k * k);
    let mut pairwise_rotary_factors = Vec::with_capacity(k * k);

    for i in 0..k {
        let g_i = landmarks.indices[i];
        for j in 0..k {
            let g_j = landmarks.indices[j];
            let p = compute_pairwise_distance(&aln, g_i, g_j);
            let d = jukes_cantor_distance(p);
            pairwise_distances.push(d);

            // M-matrix rotary factor from taxon g_i to landmark j+1
            let rot_factor = m_matrix[g_i * num_channels + (j + 1)];
            pairwise_rotary_factors.push(rot_factor);
        }
    }

    // 2D Coordinates and Phase Angles for Landmarks
    let mut landmark_coords_2d = Vec::with_capacity(k);
    let mut landmark_phase_angles = Vec::with_capacity(k);
    for i in 0..k {
        let c0 = landmarks.coords[(i + 1) * 2 + 0];
        let c1 = landmarks.coords[(i + 1) * 2 + 1];
        landmark_coords_2d.push(vec![c0, c1]);

        let theta0 = c0.atan2(c1);
        let theta1 = (c0 * c0 + c1 * c1).sqrt();
        landmark_phase_angles.push(vec![theta0, theta1]);
    }

    // 3. Continuous-Time Markov Substitution Prior
    let epsilon_0 = 0.25;
    let decay_lambda = 4.0 / 3.0;
    let prior = PhyloBias::new(epsilon_0, decay_lambda);
    let b_prior = prior.compute_prior_matrix(&aln, &landmarks);

    let mut jc_distances = Vec::new();
    let mut retention_probs = Vec::new();
    for step in 0..100 {
        let d = (step as f64) * 0.005; // 0.0 to 0.50
        jc_distances.push(d);
        retention_probs.push(prior.retention_probability(d));
    }

    // 4. Site Categories across HIV-1 pol
    let mut site_categories = vec![0u8; l];
    let mut num_invariant_sites = 0;
    let mut num_synapomorphic_sites = 0;
    let mut num_crf07_private_sites = 0;

    let ref_indices: Vec<usize> = b_indices.iter().chain(c_indices.iter()).copied().collect();

    for u in 0..l {
        // Collect reference bases
        let mut ref_bases = Vec::new();
        for &idx in &ref_indices {
            let b = aln.get(idx, u);
            if b > 0 {
                ref_bases.push(b);
            }
        }

        if ref_bases.is_empty() {
            continue;
        }

        // Check if invariant
        let first_b = ref_bases[0];
        let is_inv = ref_bases.iter().all(|&b| b == first_b);

        let crf_b = aln.get(af286230_idx, u);

        if is_inv {
            num_invariant_sites += 1;
            site_categories[u] = 0; // Invariant
        } else {
            // Check Subtype B vs C majority
            let mut b_bases = Vec::new();
            for &idx in &b_indices {
                let b = aln.get(idx, u);
                if b > 0 {
                    b_bases.push(b);
                }
            }
            let mut c_bases = Vec::new();
            for &idx in &c_indices {
                let b = aln.get(idx, u);
                if b > 0 {
                    c_bases.push(b);
                }
            }

            let mode_b = most_frequent(&b_bases);
            let mode_c = most_frequent(&c_bases);

            if mode_b.is_some() && mode_c.is_some() && mode_b != mode_c {
                if crf_b == mode_b.unwrap() {
                    num_synapomorphic_sites += 1;
                    site_categories[u] = 1; // Synapomorphy B match
                } else if crf_b == mode_c.unwrap() {
                    num_synapomorphic_sites += 1;
                    site_categories[u] = 2; // Synapomorphy C match
                } else {
                    num_crf07_private_sites += 1;
                    site_categories[u] = 3; // Private autapomorphy
                }
            } else if !ref_bases.contains(&crf_b) && crf_b > 0 {
                num_crf07_private_sites += 1;
                site_categories[u] = 3; // Private autapomorphy
            }
        }
    }

    println!(
        "  Site Classification: {} invariant, {} synapomorphic, {} private",
        num_invariant_sites, num_synapomorphic_sites, num_crf07_private_sites
    );

    // 5. Ancestral Simplex Soft Root & Attention Tensors
    let root_seq = compute_consensus_soft_root(&aln, None);

    // Sequence character strings
    let mut crf07_sequence = String::with_capacity(l);
    let mut b_consensus_sequence = String::with_capacity(l);
    let mut c_consensus_sequence = String::with_capacity(l);
    let mut root_sequence_str = String::with_capacity(l);

    for u in 0..l {
        crf07_sequence.push(base_to_char(aln.get(af286230_idx, u)));
        root_sequence_str.push(base_to_char(root_seq[u]));

        let mut b_cnt = [0usize; 5];
        for &idx in &b_indices {
            let b = aln.get(idx, u) as usize;
            if b <= 4 {
                b_cnt[b] += 1;
            }
        }
        let mut best_b = 0u8;
        let mut mc_b = 0usize;
        for c in 1..=4 {
            if b_cnt[c] > mc_b {
                mc_b = b_cnt[c];
                best_b = c as u8;
            }
        }
        b_consensus_sequence.push(base_to_char(best_b));

        let mut c_cnt = [0usize; 5];
        for &idx in &c_indices {
            let b = aln.get(idx, u) as usize;
            if b <= 4 {
                c_cnt[b] += 1;
            }
        }
        let mut best_c = 0u8;
        let mut mc_c = 0usize;
        for c in 1..=4 {
            if c_cnt[c] > mc_c {
                mc_c = c_cnt[c];
                best_c = c as u8;
            }
        }
        c_consensus_sequence.push(base_to_char(best_c));
    }

    let attention_engine = PhyloAttentionEngine::new(args.dim, 1.0);
    let masks = compute_attention_masks(&aln, &landmarks);
    let mean_a = attention_engine
        .compute_mean_attention(&aln, &landmarks, &m_matrix, &b_prior, &root_seq, &masks);

    // Only the two taxa this figure plots are materialized, one (L, K+1) row each, instead of
    // the three (L, N, K+1) tensors. The cumulative-sum loop below reads both taxa interleaved,
    // so both momentum rows are live at once.
    let mut a_row_07 = vec![0.0f64; l * num_channels];
    let mut f_scratch = vec![0.0f64; l * num_channels];
    let mut c_row_07 = vec![0.0f64; l * num_channels];
    let mut c_row_08 = vec![0.0f64; l * num_channels];

    attention_engine.compute_attention_row(
        &aln,
        &landmarks,
        &m_matrix,
        &b_prior,
        &root_seq,
        &masks,
        &mean_a,
        af286230_idx,
        &mut a_row_07,
    );
    attention_engine.compute_force_and_momentum_row(
        &aln,
        &landmarks,
        &m_matrix,
        &b_prior,
        &root_seq,
        &masks,
        &mean_a,
        af286230_idx,
        &mut f_scratch,
        &mut c_row_07,
    );
    attention_engine.compute_force_and_momentum_row(
        &aln,
        &landmarks,
        &m_matrix,
        &b_prior,
        &root_seq,
        &masks,
        &mean_a,
        ay008715_idx,
        &mut f_scratch,
        &mut c_row_08,
    );

    // Identify landmark subtype subsets
    let mut b_landmarks = Vec::new();
    let mut c_landmarks = Vec::new();
    for (k_idx, &g_idx) in landmarks.indices.iter().enumerate() {
        let ch = k_idx + 1;
        if b_indices.contains(&g_idx) {
            b_landmarks.push(ch);
        } else if c_indices.contains(&g_idx) {
            c_landmarks.push(ch);
        }
    }

    // Extract CRF07 (AF286230) and CRF08 (AY008715) attention curves at full single-site resolution (stride = 1)
    let stride = 1;
    let mut u_coords = Vec::with_capacity(l);
    let mut crf07_attention_root = Vec::with_capacity(l);
    let mut crf07_attention_b = Vec::with_capacity(l);
    let mut crf07_attention_c = Vec::with_capacity(l);
    let mut crf07_channel_attention = Vec::with_capacity(l);

    for u in 0..l {
        u_coords.push(u);
        let base_idx = u * num_channels;

        let a_root = a_row_07[base_idx + 0];
        let mut a_b = 0.0;
        for &ch in &b_landmarks {
            a_b += a_row_07[base_idx + ch];
        }
        let mut a_c = 0.0;
        for &ch in &c_landmarks {
            a_c += a_row_07[base_idx + ch];
        }

        crf07_attention_root.push(a_root);
        crf07_attention_b.push(a_b);
        crf07_attention_c.push(a_c);

        let mut row = Vec::with_capacity(num_channels);
        for ch in 0..num_channels {
            row.push(a_row_07[base_idx + ch]);
        }
        crf07_channel_attention.push(row);
    }

    // Cumulative sums and differential bridges (full resolution)
    let mut crf07_cumsum_b = Vec::with_capacity(l);
    let mut crf07_cumsum_c = Vec::with_capacity(l);
    let mut crf07_cumsum_root = Vec::with_capacity(l);
    let mut crf07_diff_bridge = Vec::with_capacity(l);

    let mut crf08_cumsum_b = Vec::with_capacity(l);
    let mut crf08_cumsum_c = Vec::with_capacity(l);
    let mut crf08_cumsum_root = Vec::with_capacity(l);
    let mut crf08_diff_bridge = Vec::with_capacity(l);

    for u in 0..l {
        let base = u * num_channels;

        // CRF07
        let c_root07 = c_row_07[base + 0];
        let mut c_b07 = 0.0;
        for &ch in &b_landmarks {
            c_b07 += c_row_07[base + ch];
        }
        let mut c_c07 = 0.0;
        for &ch in &c_landmarks {
            c_c07 += c_row_07[base + ch];
        }
        crf07_cumsum_b.push(c_b07);
        crf07_cumsum_c.push(c_c07);
        crf07_cumsum_root.push(c_root07);
        crf07_diff_bridge.push(c_b07 - c_c07);

        // CRF08
        let c_root08 = c_row_08[base + 0];
        let mut c_b08 = 0.0;
        for &ch in &b_landmarks {
            c_b08 += c_row_08[base + ch];
        }
        let mut c_c08 = 0.0;
        for &ch in &c_landmarks {
            c_c08 += c_row_08[base + ch];
        }
        crf08_cumsum_b.push(c_b08);
        crf08_cumsum_c.push(c_c08);
        crf08_cumsum_root.push(c_root08);
        crf08_diff_bridge.push(c_b08 - c_c08);
    }

    // Find turning points (local extrema in differential attention bridge)
    // RT cassette is around 850 - 1300 nt
    let crf07_bp1 = find_local_min(&crf07_diff_bridge, 700, 1100);
    let crf07_bp2 = find_local_max(&crf07_diff_bridge, 1100, 1400);

    let crf08_bp1 = find_local_min(&crf08_diff_bridge, 700, 1050);
    let crf08_bp2 = find_local_max(&crf08_diff_bridge, 1100, 1400);

    println!(
        "  Differential Turning Points: CRF07 = [{}, {}], CRF08 = [{}, {}]",
        crf07_bp1, crf07_bp2, crf08_bp1, crf08_bp2
    );

    let payload = Figure2Payload {
        alignment_length: l,
        num_taxa: n,
        taxa: aln.taxa.clone(),
        b_indices,
        c_indices,
        crf07_idx: af286230_idx,
        crf08_idx: ay008715_idx,
        num_landmarks: k,
        landmark_taxa,
        landmark_subtypes,
        landmark_indices: landmarks.indices,
        pairwise_distances,
        pairwise_rotary_factors,
        landmark_coords_2d,
        landmark_phase_angles,
        jc_distances,
        retention_probs,
        epsilon_0,
        decay_lambda,
        num_invariant_sites,
        num_synapomorphic_sites,
        num_crf07_private_sites,
        site_categories,
        crf07_sequence,
        b_consensus_sequence,
        c_consensus_sequence,
        root_sequence_str,
        stride,
        u_coords,
        crf07_attention_root,
        crf07_attention_b,
        crf07_attention_c,
        crf07_channel_attention,
        crf07_cumsum_b,
        crf07_cumsum_c,
        crf07_cumsum_root,
        crf07_diff_bridge,
        crf08_cumsum_b,
        crf08_cumsum_c,
        crf08_cumsum_root,
        crf08_diff_bridge,
        crf07_bp1,
        crf07_bp2,
        crf08_bp1,
        crf08_bp2,
    };

    let json_str = serde_json::to_string_pretty(&payload).expect("Serialization failed");
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent).expect("Failed to create output directory");
    }
    fs::write(&args.output, json_str).expect("Failed to write output JSON");

    println!("  [+] Authentic Figure 2 data saved to {:?}", args.output);
    println!("================================================================================");
}

fn most_frequent(bases: &[u8]) -> Option<u8> {
    if bases.is_empty() {
        return None;
    }
    let mut counts = [0usize; 5];
    for &b in bases {
        if b < 5 {
            counts[b as usize] += 1;
        }
    }
    let mut best_b = 0;
    let mut best_c = 0;
    for b in 1..5 {
        if counts[b] > best_c {
            best_c = counts[b];
            best_b = b as u8;
        }
    }
    if best_c > 0 {
        Some(best_b)
    } else {
        None
    }
}

fn find_local_min(arr: &[f64], start: usize, end: usize) -> usize {
    let s = start.min(arr.len() - 1);
    let e = end.min(arr.len());
    let mut min_val = f64::INFINITY;
    let mut min_idx = s;
    for i in s..e {
        if arr[i] < min_val {
            min_val = arr[i];
            min_idx = i;
        }
    }
    min_idx
}

fn find_local_max(arr: &[f64], start: usize, end: usize) -> usize {
    let s = start.min(arr.len() - 1);
    let e = end.min(arr.len());
    let mut max_val = f64::NEG_INFINITY;
    let mut max_idx = s;
    for i in s..e {
        if arr[i] > max_val {
            max_val = arr[i];
            max_idx = i;
        }
    }
    max_idx
}

fn base_to_char(b: u8) -> char {
    match b {
        1 => 'A',
        2 => 'C',
        3 => 'G',
        4 => 'T',
        _ => '-',
    }
}
