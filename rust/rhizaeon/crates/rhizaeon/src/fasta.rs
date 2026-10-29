use crate::types::Alignment;

/// Encodes a nucleotide ASCII character into integer state:
/// 1: A, 2: C, 3: G, 4: T/U, 0: gap/ambiguity.
///
/// Delegates to `aeon_manifold::encode_char` — the canonical IUPAC table's
/// ambiguity-collapse projection; identical semantics.
#[inline(always)]
pub fn encode_char(b: u8) -> u8 {
    aeon_manifold::encode_char(b)
}

/// Parses a FASTA format string into an Alignment struct.
///
/// Tokenization and the strict equal-length policy delegate to
/// `aeon_manifold::fasta`; a header with no name token defaults to "taxon"
/// (historical contract), and every sequence must match the first
/// sequence's length.
pub fn parse_fasta(fasta_str: &str) -> Result<Alignment, String> {
    use aeon_manifold::fasta::{normalize_lengths, tokenize_fasta, ParseMode};

    let mut records =
        tokenize_fasta(fasta_str).map_err(|_| "No sequences found in FASTA input".to_string())?;
    let length = normalize_lengths(&mut records, ParseMode::Strict)?;

    let taxa: Vec<String> = records
        .iter()
        .map(|r| {
            if r.name.is_empty() {
                "taxon".to_string()
            } else {
                r.name.clone()
            }
        })
        .collect();

    let num_taxa = taxa.len();
    let mut matrix = Vec::with_capacity(num_taxa * length);
    for rec in &records {
        matrix.extend(rec.seq.iter().map(|&b| encode_char(b)));
    }

    Ok(Alignment::new(taxa, length, matrix))
}

/// Counts parsimony-informative sites in an alignment
/// A site is informative if at least two distinct valid character states (1..4) each appear >= 2 times
pub fn count_parsimony_informative_sites(aln: &Alignment) -> usize {
    if aln.num_taxa < 4 || aln.length == 0 {
        return 0;
    }

    let mut count = 0;
    for u in 0..aln.length {
        let mut char_counts = [0usize; 5];
        for i in 0..aln.num_taxa {
            let c = aln.get(i, u) as usize;
            char_counts[c] += 1;
        }

        // Check valid characters 1..4
        let mut states_with_ge_2 = 0;
        for c in 1..=4 {
            if char_counts[c] >= 2 {
                states_with_ge_2 += 1;
            }
        }

        if states_with_ge_2 >= 2 {
            count += 1;
        }
    }
    count
}

/// Computes divergence-adaptive informative site admission floor:
/// M_min(N) = max(6, ceil(2.5 * ln(N)))
pub fn compute_min_informative_sites(num_taxa: usize) -> usize {
    let n = (num_taxa.max(2)) as f64;
    let min_val = (2.5 * n.ln()).ceil() as usize;
    min_val.max(6)
}

/// Trims ragged boundaries where coverage < min(3, N)
pub fn trim_coverage_envelope(aln: &Alignment) -> (Alignment, usize) {
    let k_min = aln.num_taxa.min(3);
    let mut first_col = None;
    let mut last_col = None;

    for u in 0..aln.length {
        let mut valid_taxa = 0;
        for i in 0..aln.num_taxa {
            if aln.get(i, u) > 0 {
                valid_taxa += 1;
            }
        }
        if valid_taxa >= k_min {
            if first_col.is_none() {
                first_col = Some(u);
            }
            last_col = Some(u);
        }
    }

    match (first_col, last_col) {
        (Some(start), Some(end)) if start > 0 || end + 1 < aln.length => {
            let new_len = end - start + 1;
            let mut new_matrix = Vec::with_capacity(aln.num_taxa * new_len);
            for i in 0..aln.num_taxa {
                let row = aln.row(i);
                new_matrix.extend_from_slice(&row[start..=end]);
            }
            (Alignment::new(aln.taxa.clone(), new_len, new_matrix), start)
        }
        _ => (aln.clone(), 0),
    }
}

/// Extracts parsimony-informative sites into a compressed Alignment, returning the sub-alignment
/// and a vector mapping each compressed column index (0-based) to its original 0-based coordinate.
/// A site is parsimony-informative if at least two distinct valid nucleotide states (1..4) each appear in >= 2 taxa.
pub fn extract_informative_snp_alignment(aln: &Alignment) -> (Alignment, Vec<usize>) {
    let mut snp_map = Vec::new();

    for u in 0..aln.length {
        let mut char_counts = [0usize; 5];
        for i in 0..aln.num_taxa {
            let c = aln.get(i, u) as usize;
            char_counts[c] += 1;
        }

        let retain_site = if aln.num_taxa <= 32 {
            let mut distinct_states = 0;
            for c in 1..=4 {
                if char_counts[c] >= 1 {
                    distinct_states += 1;
                }
            }
            distinct_states >= 2
        } else {
            let mut states_with_ge_2 = 0;
            for c in 1..=4 {
                if char_counts[c] >= 2 {
                    states_with_ge_2 += 1;
                }
            }
            states_with_ge_2 >= 2
        };

        if retain_site {
            snp_map.push(u);
        }
    }

    let m = snp_map.len();
    let num_taxa = aln.num_taxa;
    let mut new_matrix = Vec::with_capacity(num_taxa * m);

    for i in 0..num_taxa {
        for &u in &snp_map {
            new_matrix.push(aln.get(i, u));
        }
    }

    (Alignment::new(aln.taxa.clone(), m, new_matrix), snp_map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_char_canonical_bases() {
        assert_eq!(encode_char(b'A'), 1);
        assert_eq!(encode_char(b'C'), 2);
        assert_eq!(encode_char(b'G'), 3);
        assert_eq!(encode_char(b'T'), 4);
        assert_eq!(encode_char(b'U'), 4); // RNA folds to T
    }

    #[test]
    fn test_encode_char_case_insensitive() {
        assert_eq!(encode_char(b'a'), 1);
        assert_eq!(encode_char(b'c'), 2);
        assert_eq!(encode_char(b'g'), 3);
        assert_eq!(encode_char(b't'), 4);
        assert_eq!(encode_char(b'u'), 4);
    }

    #[test]
    fn test_encode_char_ambiguity_collapses_to_zero() {
        // Every IUPAC ambiguity code, gap, and unknown byte maps to 0.
        // This is rhizaeon's contract: ambiguity is NOT preserved.
        for &b in b"RYSWKMBDHVNryswkmbdhvn-.Xx?*".iter() {
            assert_eq!(encode_char(b), 0, "byte {} should encode to 0", b as char);
        }
    }

    #[test]
    fn test_parse_fasta_basic() {
        let aln = parse_fasta(">a\nACGT\n>b\nTGCA\n").unwrap();
        assert_eq!(aln.num_taxa, 2);
        assert_eq!(aln.length, 4);
        assert_eq!(aln.taxa, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(aln.get(0, 0), 1); // A
        assert_eq!(aln.get(1, 3), 1); // last base of TGCA = A
    }

    #[test]
    fn test_parse_fasta_multiline_sequence() {
        let aln = parse_fasta(">a\nAC\nGT\n>b\nTG\nCA\n").unwrap();
        assert_eq!(aln.length, 4);
        assert_eq!(aln.get(0, 2), 3); // 'G' is third base of a's joined seq
    }

    #[test]
    fn test_parse_fasta_header_first_token() {
        // Canonical taxon name is the first whitespace-delimited token.
        let aln = parse_fasta(">seq1 some description here\nACGT\n").unwrap();
        assert_eq!(aln.taxa[0], "seq1");
    }

    #[test]
    fn test_parse_fasta_ragged_is_an_error() {
        // rhizaeon's contract: ragged alignments are REJECTED (not padded).
        let err = parse_fasta(">a\nACGT\n>b\nACG\n").unwrap_err();
        assert!(err.contains("length"), "expected length error, got: {err}");
    }

    #[test]
    fn test_parse_fasta_empty_is_an_error() {
        assert!(parse_fasta("").is_err());
        assert!(parse_fasta("   \n\n").is_err());
    }

    #[test]
    fn test_parse_fasta_ambiguity_encodes_zero() {
        let aln = parse_fasta(">a\nACGN\n>b\nACG-\n").unwrap();
        assert_eq!(aln.get(0, 3), 0); // N
        assert_eq!(aln.get(1, 3), 0); // gap
    }

    #[test]
    fn test_parse_fasta_blank_lines_skipped() {
        let aln = parse_fasta(">a\nAC\n\nGT\n>b\n\nTGCA\n").unwrap();
        assert_eq!(aln.num_taxa, 2);
        assert_eq!(aln.length, 4);
    }

    #[test]
    fn test_extract_informative_snp_alignment_map() {
        // Alignment columns: AACC / AACC / BATG / BATG (B collapses to 0).
        // site0: A A A A -> not informative
        // site1: C C 0 0 -> only C has >=2 -> not informative
        // site2: C C T T -> C:2 and T:2 -> informative
        // site3: C C G G -> informative
        let aln = parse_fasta(">t1\nAACC\n>t2\nAACC\n>t3\nBATG\n>t4\nBATG\n").unwrap();
        let (sub, map) = extract_informative_snp_alignment(&aln);
        assert_eq!(map, vec![2, 3]);
        assert_eq!(sub.length, 2);
        assert_eq!(sub.num_taxa, 4);
    }
}
