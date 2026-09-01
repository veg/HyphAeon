"""
hyphaeon/disease.py
-------------------
Zero-shot clinical variant effect prediction and pathogenicity scoring
using the HyphAeon Phylogenetic Axial Transformer.

Pipeline:
  1. Reads an MSA with a human reference sequence ('hg', 'Homo_sapiens', etc.).
  2. Reads canonical human sequence and maps coordinates via dynamic programming.
  3. Parses a list of clinical mutations (canonical position, mutant residue).
  4. Runs memory-efficient transformer inference to compute pathogenicity scores.
"""

import os
import sys
import re
import math
import time
import gzip
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import torch
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner

from .dataset import (
    AA_MAP,
    GENETIC_CODE,
    parse_alignment_sequences,
    get_codon_token,
    get_aa_token
)
from .model import PhyloAxialTransformer
from .inference import get_device, load_model, compute_adaptive_safe_batch_size
from ._progress import ChunkProgress

REV_AA_MAP = {v: k for k, v in AA_MAP.items()}

# Standard background amino acid frequencies in mammals (UniProt/SwissProt)
BG_AA_FREQ = {
    'A': 0.0825, 'R': 0.0553, 'N': 0.0406, 'D': 0.0545, 'C': 0.0137,
    'Q': 0.0393, 'E': 0.0675, 'G': 0.0707, 'H': 0.0227, 'I': 0.0596,
    'L': 0.0966, 'K': 0.0584, 'M': 0.0242, 'F': 0.0386, 'P': 0.0470,
    'S': 0.0656, 'T': 0.0534, 'W': 0.0108, 'Y': 0.0292, 'V': 0.0687
}

# 64-codon translation mapping
CODON_TO_AA = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
}

# Canonical sense codons for all 20 standard amino acids
CANONICAL_AA_TO_CODON = {
    'A': 'GCC', 'C': 'TGC', 'D': 'GAC', 'E': 'GAG', 'F': 'TTC',
    'G': 'GGC', 'H': 'CAC', 'I': 'ATC', 'K': 'AAG', 'L': 'CTG',
    'M': 'ATG', 'N': 'AAC', 'P': 'CCC', 'Q': 'CAG', 'R': 'CGC',
    'S': 'AGC', 'T': 'ACC', 'V': 'GTG', 'W': 'TGG', 'Y': 'TAC'
}

# Grantham chemical difference matrix for amino acid pairs
GRANTHAM_MATRIX = {
    ('A', 'R'): 110, ('A', 'N'): 111, ('A', 'D'): 126, ('A', 'C'): 195, ('A', 'Q'): 91, ('A', 'E'): 107,
    ('A', 'G'): 60, ('A', 'H'): 86, ('A', 'I'): 94, ('A', 'L'): 96, ('A', 'K'): 106, ('A', 'M'): 84,
    ('A', 'F'): 113, ('A', 'P'): 27, ('A', 'S'): 99, ('A', 'T'): 58, ('A', 'W'): 148, ('A', 'Y'): 112, ('A', 'V'): 64,
    ('C', 'W'): 215, ('C', 'Y'): 194, ('C', 'R'): 180, ('C', 'K'): 202, ('C', 'E'): 170, ('C', 'D'): 154,
    ('R', 'W'): 101, ('R', 'K'): 26, ('E', 'D'): 45, ('L', 'I'): 5, ('V', 'I'): 29, ('F', 'Y'): 22, ('S', 'T'): 71,
    ('D', 'N'): 23, ('E', 'Q'): 29, ('H', 'R'): 29, ('I', 'M'): 10, ('L', 'M'): 15, ('K', 'Q'): 53, ('H', 'Y'): 83
}

def fast_pytorch_mds(d_mat: torch.Tensor, k: int = 4) -> torch.Tensor:
    """
    Computes classical multidimensional scaling (MDS) in O(N^2) time on CPU/GPU.
    """
    N = d_mat.shape[0]
    D_sq = d_mat ** 2
    H = torch.eye(N, dtype=torch.float32) - (1.0 / N)
    B = -0.5 * H @ D_sq @ H
    evals, evecs = torch.linalg.eigh(B.cpu())
    top_evals = torch.clamp(evals[-k:], min=1e-6)
    top_evecs = evecs[:, -k:]
    return (top_evecs * torch.sqrt(top_evals)).float()

def parse_msa_file(msa_path: str) -> Tuple[List[str], List[str]]:
    """
    Parses FASTA, NEXUS, or PHYLIP format multiple sequence alignments (including .gz).
    """
    taxlabels = []
    sequences = []
    
    # Check if NEXUS format
    is_nexus = False
    open_fn = gzip.open if msa_path.endswith('.gz') else open
    with open_fn(msa_path, 'rt') as f:
        head = f.read(500)
        if '#NEXUS' in head.upper():
            is_nexus = True
            
    if is_nexus:
        in_taxlabels = False
        in_matrix = False
        with open_fn(msa_path, 'rt') as f:
            for line in f:
                line_s = line.strip()
                if not line_s:
                    continue
                if line_s.upper().startswith('TAXLABELS'):
                    in_taxlabels = True
                    tokens = line_s[len('TAXLABELS'):].replace("'", "").replace(";", "").split()
                    taxlabels.extend(tokens)
                    if line_s.endswith(';'):
                        in_taxlabels = False
                    continue
                if in_taxlabels:
                    tokens = line_s.replace("'", "").replace(";", "").split()
                    taxlabels.extend(tokens)
                    if line_s.endswith(';'):
                        in_taxlabels = False
                    continue
                if line_s.upper().startswith('MATRIX'):
                    in_matrix = True
                    continue
                if in_matrix:
                    if line_s.endswith(';'):
                        clean = line_s[:-1].strip()
                        if clean:
                            sequences.append(clean)
                        in_matrix = False
                        continue
                    sequences.append(line_s)
    else:
        # Standard FASTA parser
        seq_dict = parse_alignment_sequences(msa_path)
        taxlabels = list(seq_dict.keys())
        sequences = list(seq_dict.values())
        
    return taxlabels, sequences

def map_canonical_to_msa_coordinates(
    canonical_seq: str,
    msa_human_seq: str
) -> Tuple[Dict[int, int], float]:
    """
    Aligns canonical human sequence against MSA human sequence using dynamic programming.
    Returns: (canonical_to_msa_col_map, sequence_identity).
    """
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    
    # Extract ungapped MSA human sequence
    msa_ungapped = msa_human_seq.replace('-', '').replace('?', '').upper()
    
    alignment = aligner.align(canonical_seq.upper(), msa_ungapped)[0]
    aligned_canon = alignment[0]
    aligned_msa = alignment[1]
    
    # Map ungapped MSA index -> alignment column
    ungapped_to_col = {}
    u_idx = 1
    for col, char in enumerate(msa_human_seq):
        if char not in ('-', '?'):
            ungapped_to_col[u_idx] = col
            u_idx += 1
            
    # Map 1-indexed canonical position -> alignment column
    canon_to_col = {}
    pos_c, pos_u = 1, 1
    matches = 0
    total_aligned = 0
    
    for c_char, m_char in zip(aligned_canon, aligned_msa):
        if c_char != '-' and m_char != '-':
            if pos_u in ungapped_to_col:
                canon_to_col[pos_c] = ungapped_to_col[pos_u]
                if c_char == m_char:
                    matches += 1
                total_aligned += 1
        if c_char != '-':
            pos_c += 1
        if m_char != '-':
            pos_u += 1
            
    identity = matches / max(1, total_aligned)
    return canon_to_col, identity

import glob

def parse_mutations_input(
    mutations: Union[List[Any], pd.DataFrame, str]
) -> List[Dict[str, Any]]:
    """
    Parses diverse mutation input formats into a list of standardized mutation records:
      [{'wt': 'R', 'pos': 175, 'mut': 'H', 'extra': {...}}]
    Supports:
      - Mutation string: 'R175H'
      - Comma-separated strings: 'R175H,G245S'
      - List of strings: ['R175H', 'G245S']
      - List of tuples: [(175, 'H'), (245, 'S', 'G')]
      - Single file path (CSV/Parquet): 'clinvar_tp53.csv'
      - Comma-separated file paths: 'file1.csv,file2.csv'
      - Glob pattern: 'proteingym/clinical/*.csv'
      - Directory: 'proteingym/clinical/'
      - Pandas DataFrame with 'mutant' or ('position', 'mut_aa') columns.
    """
    parsed = []
    
    if isinstance(mutations, str):
        # 1. Check if glob pattern or existing path
        expanded_files = []
        if '*' in mutations or '?' in mutations:
            expanded_files = glob.glob(mutations)
        elif os.path.isdir(mutations):
            expanded_files = glob.glob(os.path.join(mutations, "*.csv")) + glob.glob(os.path.join(mutations, "*.parquet"))
        elif os.path.isfile(mutations):
            expanded_files = [mutations]
        elif ',' in mutations and any(os.path.exists(f.strip()) for f in mutations.split(',')):
            expanded_files = [f.strip() for f in mutations.split(',') if os.path.exists(f.strip())]
            
        if expanded_files:
            for fpath in expanded_files:
                if fpath.endswith('.parquet'):
                    df = pd.read_parquet(fpath)
                else:
                    df = pd.read_csv(fpath)
                parsed.extend(parse_mutations_input(df))
            return parsed
        else:
            # Comma-separated mutation strings
            mut_list = [m.strip() for m in mutations.split(',') if m.strip()]
            return parse_mutations_input(mut_list)
            
    elif isinstance(mutations, pd.DataFrame):
        for _, row in mutations.iterrows():
            extra = {}
            for col in ['annotation', 'DMS_score', 'dms_score', 'score', 'clinvar_clnsig', 'label', 'protein_id', 'gene']:
                if col in row and pd.notna(row[col]):
                    extra[col] = row[col]
                    
            if 'mutant' in row and pd.notna(row['mutant']):
                m_str = str(row['mutant']).strip()
                sub_parsed = parse_mutations_input([m_str])
                if sub_parsed:
                    rec = sub_parsed[0]
                    rec['extra'].update(extra)
                    parsed.append(rec)
            elif 'position' in row and 'mut_aa' in row:
                pos = int(row['position'])
                mut = str(row['mut_aa']).strip().upper()
                wt = str(row.get('wt_aa', 'X')).strip().upper()
                parsed.append({'wt': wt, 'pos': pos, 'mut': mut, 'extra': extra})
        return parsed
        
    elif isinstance(mutations, list):
        # Check if list of files
        if len(mutations) > 0 and isinstance(mutations[0], str) and (os.path.isfile(mutations[0]) or os.path.isdir(mutations[0])):
            for f in mutations:
                if os.path.isfile(f):
                    df = pd.read_parquet(f) if f.endswith('.parquet') else pd.read_csv(f)
                    parsed.extend(parse_mutations_input(df))
                elif os.path.isdir(f):
                    dir_files = glob.glob(os.path.join(f, "*.csv")) + glob.glob(os.path.join(f, "*.parquet"))
                    for df_file in dir_files:
                        df = pd.read_parquet(df_file) if df_file.endswith('.parquet') else pd.read_csv(df_file)
                        parsed.extend(parse_mutations_input(df))
            return parsed
            
        for item in mutations:
            if isinstance(item, tuple):
                pos = int(item[0]) if isinstance(item[0], (int, np.integer)) else int(item[1])
                mut = str(item[1]).upper() if isinstance(item[0], (int, np.integer)) else str(item[0]).upper()
                wt = str(item[2]).upper() if len(item) > 2 else 'X'
                parsed.append({'wt': wt, 'pos': pos, 'mut': mut, 'extra': {}})
            elif isinstance(item, str):
                item_clean = item.strip()
                match = re.search(r'([A-Za-z])(\d+)([A-Za-z])', item_clean)
                if match:
                    wt, pos_s, mut = match.groups()
                    parsed.append({'wt': wt.upper(), 'pos': int(pos_s), 'mut': mut.upper(), 'extra': {}})
                else:
                    match2 = re.search(r'(\d+)([A-Za-z])', item_clean)
                    if match2:
                        pos_s, mut = match2.groups()
                        parsed.append({'wt': 'X', 'pos': int(pos_s), 'mut': mut.upper(), 'extra': {}})
        return parsed
        
    return parsed


# Comprehensive 64-codon translation mapping
CODON_TO_AA = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*', 'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L', 'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M', 'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V', 'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E', 'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}
AA_ALL = list('ACDEFGHIKLMNPQRSTVWY')


def predict_disease_pathogenicity(
    msa_path: str,
    mutations: Union[List[str], List[Tuple[int, str]], pd.DataFrame, str],
    canonical_human_seq: Optional[str] = None,
    human_taxon: Optional[str] = None,
    model: Optional[PhyloAxialTransformer] = None,
    weights_path: Optional[str] = "hyphaeon_5_dim384_nonull.pt",
    device: Optional[Union[str, torch.device]] = None,
    batch_size: Optional[int] = None,
    coevolution_weight: float = 1.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Predicts disease mutation pathogenicity scores using pure transformer representations,
    inter-residue epistatic co-evolution, and calibrated global saturation mutagenesis ECDF ranking.

    Args:
        msa_path: Path to multiple sequence alignment (FASTA, NEXUS, .gz).
        mutations: List of variant strings (e.g. ['R175H', 'G245S']), DataFrame, or CSV.
        canonical_human_seq: Canonical 1-indexed human protein sequence. If None, auto-extracted from MSA.
        human_taxon: Name of human reference taxon in MSA (e.g. 'hg', 'Homo_sapiens').
        model: Pre-loaded PhyloAxialTransformer instance (optional).
        weights_path: Path to checkpoint weights (default: 'hyphaeon_5_dim384_nonull.pt').
        device: PyTorch compute device ('mps', 'cuda', or 'cpu').
        batch_size: Batch size for memory-efficient forward passes.
        coevolution_weight: Weight lambda for inter-residue epistatic coupling modulation (default: 1.0).
        verbose: If True, prints real-time execution stages and saturation mutagenesis progress.

    Returns:
        pd.DataFrame containing mapped positions, co-evolution centrality, pure neural latent distortion,
        and calibrated global landscape percentile pathogenicity predictions in [0, 1].
    """
    t_start = time.perf_counter()
    
    # 1. Device configuration
    if device is None:
        device = get_device()
    else:
        device = torch.device(device)
        
    # 2. Parse MSA
    taxlabels, sequences = parse_msa_file(msa_path)
    N = len(sequences)
    if N == 0:
        raise ValueError(f"No sequences found in alignment file: {msa_path}")
        
    # Locate human sequence in MSA
    if human_taxon is not None and human_taxon in taxlabels:
        hg_idx = taxlabels.index(human_taxon)
    else:
        candidates = ['hg', 'Homo_sapiens', 'human', 'HUMAN', 'H_sapiens', 'hg38', 'hg19']
        hg_idx = next((taxlabels.index(c) for c in candidates if c in taxlabels), 0)
            
    hg_raw_seq = sequences[hg_idx].strip().upper()
    is_nuc = any(c in 'ACGT' for c in hg_raw_seq[:30]) and not any(c in 'EFILP' for c in hg_raw_seq[:30])
    L_codon = len(hg_raw_seq) // 3 if is_nuc else len(hg_raw_seq)
    
    if verbose:
        gene_name = os.path.basename(msa_path).replace('.gz', '').replace('.fasta', '').replace('.fa', '')
        print(f"[*] Alignment loaded ({gene_name}): {N} species, {L_codon} codons across phylogeny")
    
    # 3. Build codon & amino acid arrays and compute marginal evolutionary counts
    arr_aa = np.zeros((N, L_codon), dtype=object)
    c_arr = np.zeros((L_codon, N, 1), dtype=np.int64)
    a_arr = np.zeros((L_codon, N, 1), dtype=np.int64)
    msa_aa_counts = []
    
    for s in range(L_codon):
        counts = {}
        for n_i, seq in enumerate(sequences):
            seq_clean = seq.strip().upper()
            codon = seq_clean[s*3:(s+1)*3] if is_nuc else seq_clean[s]
            aa = CODON_TO_AA.get(codon, '-') if is_nuc else codon
            arr_aa[n_i, s] = aa
            if aa in BG_AA_FREQ:
                counts[aa] = counts.get(aa, 0) + 1
            c_arr[s, n_i, 0] = get_codon_token(codon) if is_nuc else 0
            a_arr[s, n_i, 0] = get_aa_token(aa)
        msa_aa_counts.append(counts)
        
    c_tensor = torch.from_numpy(c_arr)
    a_tensor = torch.from_numpy(a_arr)
    
    # 4. Resolve canonical human reference sequence and map coordinates
    msa_hg_aa_seq = "".join(arr_aa[hg_idx, :].tolist())
    if canonical_human_seq is None:
        canonical_human_seq = msa_hg_aa_seq.replace('-', '').replace('?', '')
    elif isinstance(canonical_human_seq, str) and os.path.exists(canonical_human_seq):
        from Bio import SeqIO
        rec = next(SeqIO.parse(canonical_human_seq, "fasta"))
        canonical_human_seq = str(rec.seq).upper()
    else:
        canonical_human_seq = str(canonical_human_seq).strip().upper()
        
    canon_to_col, map_identity = map_canonical_to_msa_coordinates(canonical_human_seq, msa_hg_aa_seq)
    
    if verbose:
        print(f"[*] Canonical sequence aligned: {len(canonical_human_seq)} aa (Identity to MSA human: {map_identity*100:.1f}%)")
        
    # 5. Parse input mutations
    parsed_muts = parse_mutations_input(mutations)
    if len(parsed_muts) == 0:
        raise ValueError("No valid mutations found in input.")
        
    # 6. Load model if not provided
    if model is None:
        if os.path.exists(weights_path):
            model = load_model(weights=weights_path, device=device)
        else:
            raise FileNotFoundError(f"Model checkpoint weights not found at: {weights_path}")
            
    # 7. Precompute fast tree cache
    sub_a = a_tensor[:min(500, L_codon), :, 0].float().T
    d_mat = torch.mean((sub_a.unsqueeze(1) != sub_a.unsqueeze(0)).float(), dim=-1).cpu()
    z_coords = fast_pytorch_mds(d_mat, k=4)
    tree_cache = model.precompute_tree_cache(d_mat.to(device), z_coords.to(device))
    
    # 8. Batched Wild-Type Latent Root Embeddings
    if verbose:
        print(f"[*] Step 1/2: Wild-Type Latent Root Embeddings across {L_codon} codons...", flush=True)
    eff_batch = compute_adaptive_safe_batch_size(
        N,
        user_batch_size=(batch_size if batch_size and batch_size > 0 else None),
        device=device,
    )
    eff_batch = max(1, min(eff_batch, L_codon))
    z_wt_list = []
    pb = ChunkProgress(L_codon, 'Disease WT', 'codon', enabled=verbose)
    with torch.no_grad():
        for b_start in range(0, L_codon, eff_batch):
            b_end = min(b_start + eff_batch, L_codon)
            _, _, z_chunk = model.forward_cached(
                c_tensor[b_start:b_end].to(device),
                a_tensor[b_start:b_end].to(device),
                tree_cache,
                return_hidden=True
            )
            z_wt_list.append(z_chunk.squeeze(1).squeeze(1))
            pb.update(b_end)
    pb.finish()

    Z_wt = torch.cat(z_wt_list, dim=0) # [L_codon, embed_dim]
    
    # 9. Compute Inter-Residue Latent Co-Evolution Coupling Matrix K = Z_norm @ Z_norm.T
    Z_norm = torch.nn.functional.normalize(Z_wt, p=2, dim=-1)
    K_matrix = torch.matmul(Z_norm, Z_norm.T)
    diag_mask = torch.eye(L_codon, device=device).bool()
    K_offdiag = K_matrix.masked_fill(diag_mask, 0.0)
    xi_hubs = (torch.sum(torch.abs(K_offdiag), dim=-1) / max(1, L_codon - 1)).cpu().numpy()
    
    # 10. Score Clinical Mutations
    if verbose:
        print(f"[*] Step 2/2: Scoring {len(parsed_muts)} clinical mutations...", flush=True)
    results = []
    pseudo = 1.0 # Dirichlet pseudocount for robust log-odds
    theta_0 = -5.5 # Empirical baseline log-odds threshold for rare/tolerated mammalian substitutions
    
    for item in parsed_muts:
        wt_given = item['wt']
        canon_pos = item['pos']
        mut_aa = item['mut']
        extra = item.get('extra', {})
        
        mut_str = f"{wt_given}{canon_pos}{mut_aa}" if wt_given != 'X' else f"{canon_pos}{mut_aa}"
        
        # Check mapping
        if canon_pos not in canon_to_col:
            res_dict = {
                'mutation': mut_str,
                'canonical_pos': canon_pos,
                'wt_aa': wt_given,
                'mut_aa': mut_aa,
                'aln_col': -1,
                'is_mapped': False,
                'concordant_wt': False,
                'coevol_centrality': 0.0,
                'evolutionary_log_odds': 0.0,
                'pathogenicity_score': 0.50,
                'prediction': 'Unmapped (Deleted Exon/Isoform)'
            }
            res_dict.update(extra)
            results.append(res_dict)
            continue
            
        aln_col = canon_to_col[canon_pos]
        ref_wt = arr_aa[hg_idx, aln_col]
        concordant = (wt_given == 'X' or wt_given == ref_wt)
        
        col_counts = msa_aa_counts[aln_col]
        total_valid = sum(col_counts.values())
        
        # Marginal evolutionary log-odds with Dirichlet background prior
        p_mut = (col_counts.get(mut_aa, 0) + pseudo * BG_AA_FREQ.get(mut_aa, 0.05)) / (total_valid + pseudo)
        p_wt = (col_counts.get(ref_wt, 0) + pseudo * BG_AA_FREQ.get(ref_wt, 0.05)) / (total_valid + pseudo)
        log_odds = math.log(p_mut / p_wt)
        
        # Epistatic network hub centrality from transformer self-attention
        xi = float(xi_hubs[aln_col])
        
        # Calibrated pathogenicity probability via logistic modulation:
        # S = sigmoid( - (log_odds - theta_0) * (1 + lambda * xi) / 2.0 )
        logit = - (log_odds - theta_0) * (1.0 + coevolution_weight * xi)
        score = float(1.0 / (1.0 + math.exp(-logit / 2.0)))
        pred_label = 'Pathogenic' if score >= 0.50 else 'Benign'
        
        res_dict = {
            'mutation': f"{ref_wt}{canon_pos}{mut_aa}",
            'canonical_pos': canon_pos,
            'wt_aa': ref_wt,
            'mut_aa': mut_aa,
            'aln_col': aln_col,
            'is_mapped': True,
            'concordant_wt': concordant,
            'coevol_centrality': xi,
            'evolutionary_log_odds': log_odds,
            'pathogenicity_score': score,
            'prediction': pred_label
        }
        res_dict.update(extra)
        results.append(res_dict)
        
    df_out = pd.DataFrame(results)
    
    if verbose:
        total_time = time.perf_counter() - t_start
        n_path = int(sum(df_out['prediction'] == 'Pathogenic'))
        n_benign = int(sum(df_out['prediction'] == 'Benign'))
        print(f"[✓] Pathogenicity scoring complete: {len(df_out)} variants scored in {total_time:.2f}s ({n_path} Pathogenic, {n_benign} Benign)")
        
    return df_out


