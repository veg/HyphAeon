import torch._utils
if not hasattr(torch._utils, '_rebuild_device_tensor_from_cpu_tensor'):
    def _rebuild_device_tensor_from_cpu_tensor(data, device, *args, **kwargs):
        req_grad = kwargs.get('requires_grad', False)
        if len(args) > 1 and isinstance(args[-1], bool):
            req_grad = args[-1]
        return data.to(device).requires_grad_(bool(req_grad))
    torch._utils._rebuild_device_tensor_from_cpu_tensor = _rebuild_device_tensor_from_cpu_tensor


import subprocess
import os
import sys
import math
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gzip
import pickle
import re
from io import StringIO
from Bio import Phylo
from collections import Counter
from train_transformer_selection import PhyloAxialTransformer, compute_mds_coordinates

GENETIC_CODE = {
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
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

from train_transformer_selection import (
    CODON_TO_IDX,
    CODON_TO_AA_DICT,
    AA_TO_IDX,
    GENETIC_CODE,
    CODON_LOOKUP,
    AA_LOOKUP
)

def translate_codon(codon):
    codon = codon.upper()
    if len(codon) != 3:
        return '?'
    b_encoded = codon.encode('ascii')
    idx = b_encoded[0] * 65536 + b_encoded[1] * 256 + b_encoded[2]
    if idx < len(AA_LOOKUP):
        aa_idx = AA_LOOKUP[idx]
        for aa, a_idx in AA_TO_IDX.items():
            if a_idx == aa_idx:
                return aa
    return '?'

IUPAC_EXPAND = {
    'A': ['A'], 'C': ['C'], 'G': ['G'], 'T': ['T'],
    'R': ['A', 'G'], 'Y': ['C', 'T'], 'M': ['A', 'C'], 'K': ['G', 'T'],
    'S': ['C', 'G'], 'W': ['A', 'T'], 'H': ['A', 'C', 'T'], 'B': ['C', 'G', 'T'],
    'V': ['A', 'C', 'G'], 'D': ['A', 'G', 'T'], 'N': ['A', 'C', 'G', 'T']
}

def get_codon_token(codon):
    codon = codon.upper()
    if len(codon) != 3 or '-' in codon:
        return 64 if '-' in codon else 65
    b_encoded = codon.encode('ascii')
    idx = b_encoded[0] * 65536 + b_encoded[1] * 256 + b_encoded[2]
    if idx < len(CODON_LOOKUP):
        tok = CODON_LOOKUP[idx]
        if tok < 64:
            return tok
            
    # Resolve ambiguous IUPAC codon to first standard canonical constituent
    b0_opts = IUPAC_EXPAND.get(codon[0], [codon[0]])
    b1_opts = IUPAC_EXPAND.get(codon[1], [codon[1]])
    b2_opts = IUPAC_EXPAND.get(codon[2], [codon[2]])
    for b0 in b0_opts:
        for b1 in b1_opts:
            for b2 in b2_opts:
                c_try = b0 + b1 + b2
                b_enc = c_try.encode('ascii')
                i_try = b_enc[0] * 65536 + b_enc[1] * 256 + b_enc[2]
                if i_try < len(CODON_LOOKUP) and CODON_LOOKUP[i_try] < 64:
                    return CODON_LOOKUP[i_try]
    return 65

def get_aa_token(codon):
    codon = codon.upper()
    if len(codon) != 3 or '-' in codon:
        return 21 if '-' in codon else 22
    b_encoded = codon.encode('ascii')
    idx = b_encoded[0] * 65536 + b_encoded[1] * 256 + b_encoded[2]
    if idx < len(AA_LOOKUP):
        tok = AA_LOOKUP[idx]
        if tok < 21:
            return tok
            
    # Resolve amino acid for ambiguous IUPAC codon
    b0_opts = IUPAC_EXPAND.get(codon[0], [codon[0]])
    b1_opts = IUPAC_EXPAND.get(codon[1], [codon[1]])
    b2_opts = IUPAC_EXPAND.get(codon[2], [codon[2]])
    
    possible_aas = set()
    for b0 in b0_opts:
        for b1 in b1_opts:
            for b2 in b2_opts:
                aa = GENETIC_CODE.get(b0 + b1 + b2, '?')
                if aa != '?':
                    possible_aas.add(aa)
                    
    if len(possible_aas) >= 1:
        sorted_aas = sorted(list(possible_aas))
        chosen = sorted_aas[0]
        for aa in sorted_aas:
            if aa != 'M' and aa in AA_TO_IDX:
                chosen = aa
                break
        return AA_TO_IDX.get(chosen, 22)
    return 22

# --- BLOSUM62 & Grantham Scoring Matrices ---
_blosum_raw = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1 -2 -2  0 -3 -3  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1 -2  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -3 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""

_grantham_raw = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  0 112 111 126  44  91 107  60  86  94  96 106  84  95  27  35  58 148 112  64
R 112  0  86 138 102  43  97 125  29  97 102  26  91  97 103 110  71 101  77  96
N 111  86  0  23 139  46  42  80  68 149 143  94 142 158  91  46  65 174 143 133
D 126 138  23  0 154  61  45  94  81 168 162 101 160 177 108  54  85 181 162 152
C  44 102 139 154  0 116 126 117 118 117 121 112 118 135  74  80 101 190 154 109
Q  91  43  46  61 116  0  29  87  24  93  99  53  81  93  76  68  47 130  84  96
E 107  97  42  45 126  29  0  98  40 103 107  56  87 102  93  80  65 122  86  96
G  60 125  80  94 117  87  98  0  98 135 127 127 127 153  42  56  59 184 147 109
H  86  29  68  81 118  24  40  98  0  99 105  32  87  92  77  83  47 115  83  98
I  94  97 149 168 117  93 103 135  99  0  10  97  10  21  95 142 124 103  83  29
L  96 102 143 162 121  99 107 127 105  10  0 107  15  22  95 145 130 113  92  32
K 106  26  94 101 112  53  56 127  32 97 107  0  95 102 103 121  78 110  85  97
M  84  91 142 160 118  81  87 127  87  10  15  95  0  28  87 135 121 115  95  21
F  95  97 158 177 135  93 102 153  92  21  22 102  28  0 110 155 140  40  22  50
P  27 103  91 108  74  76  93  42  77  95  95 103  87 110  0  56  74 153 110  76
S  35 110  46  54  80  68  80  56  83 142 145 121 135 155  56  0  58 177 144 124
T  58  71  65  85 101  47  65  59  47 124 130  78 121 140  74  58  0 178 134 103
W 148 101 174 181 190 130 122 184 115 103 113 110 115  40 153 177 178  0  37  88
Y 112  77 143 162 154  84  86 147  83  83  92  85  95  22 110 144 134  37  0  90
V  64  96 133 152 109  96  96 109  98  29  32  97  21  50  76 124 103  88  90  0
"""

def parse_scoring_matrices():
    lines = [line.strip().split() for line in _blosum_raw.strip().split('\n')]
    headers = lines[0]
    blosum_dict = {}
    for row in lines[1:]:
        key = row[0]
        blosum_dict[key] = {headers[i]: int(row[i+1]) for i in range(len(headers))}
        
    lines = [line.strip().split() for line in _grantham_raw.strip().split('\n')]
    headers = lines[0]
    grantham_dict = {}
    for row in lines[1:]:
        key = row[0]
        grantham_dict[key] = {headers[i]: float(row[i+1]) for i in range(len(headers))}
    return blosum_dict, grantham_dict

BLOSUM62, GRANTHAM = parse_scoring_matrices()

def compute_mds_coordinates(dist_matrix_np, n_components=4):
    N = dist_matrix_np.shape[0]
    if N <= n_components:
        coords = np.zeros((N, n_components), dtype=np.float32)
        return coords
    dist_64 = dist_matrix_np.astype(np.float64)
    D2 = dist_64 ** 2
    H = np.eye(N, dtype=np.float64) - np.ones((N, N), dtype=np.float64) / float(N)
    B = -0.5 * (H @ D2 @ H)
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    # Enforce sign convention: largest absolute value element is positive
    for col in range(evecs.shape[1]):
        max_abs_idx = np.argmax(np.abs(evecs[:, col]))
        sign = np.sign(evecs[max_abs_idx, col])
        if sign < 0:
            evecs[:, col] *= -1.0
            
    coords = np.zeros((N, n_components), dtype=np.float32)
    for i in range(n_components):
        val = evals[i]
        if val > 0:
            coords[:, i] = (evecs[:, i] * np.sqrt(val)).astype(np.float32)
    return coords

def is_site_variable(site_codons, site_aas):
    if not site_codons or not site_aas:
        return False
        
    unique_aas_set = set(site_aas)
    
    # Condition 1: Multiple amino acids
    if len(unique_aas_set) > 1:
        return True
        
    # Condition 2: Serine Island transition (synonymous but selection-relevant)
    if len(unique_aas_set) == 1 and 'S' in unique_aas_set:
        has_tcn = any(c in ('TCA', 'TCC', 'TCG', 'TCT') for c in site_codons)
        has_agy = any(c in ('AGC', 'AGT') for c in site_codons)
        if has_tcn and has_agy:
            return True
            
    return False

# --- Differentiable Ranking Loss & Alignment Helpers ---
# --- Differentiable Ranking Loss via ListNet ---

def parse_nexus_alignment_and_embedded_tree(filepath):
    from Bio import AlignIO, SeqIO
    open_func = gzip.open if filepath.endswith('.gz') else open
    with open_func(filepath, 'rt') as f:
        full_text = f.read()

    # Strip bracket comments [ ... ] first to avoid matching comment parens
    text_no_comments = re.sub(r'\[.*?\]', '', full_text, flags=re.DOTALL)

    seq_dict = {}
    taxlabels = []

    # Check for TAXLABELS in BEGIN TAXA block
    tax_match = re.search(r'TAXLABELS\s+([^;]+);', text_no_comments, re.IGNORECASE)
    parsed_taxlabels = []
    if tax_match:
        raw_labels = tax_match.group(1)
        parsed_taxlabels = [lbl.strip("'\" ") for lbl in re.findall(r'\'[^\']+\'|\"[^\"]+\"|\S+', raw_labels) if lbl.strip("'\" ")]

    # Check if NOLABELS is specified in FORMAT
    has_nolabels = bool(re.search(r'FORMAT\s+[^;]*NOLABELS', text_no_comments, re.IGNORECASE))

    # Check for MATRIX block
    matrix_match = re.search(r'MATRIX\s+([^;]+);', text_no_comments, re.IGNORECASE)
    
    # 1. If NOLABELS and TAXLABELS exist, parse sequence per line
    if has_nolabels and parsed_taxlabels and matrix_match:
        raw_matrix = matrix_match.group(1).strip()
        matrix_lines = [re.sub(r'\s+', '', l) for l in raw_matrix.splitlines() if re.sub(r'\s+', '', l)]
        if len(matrix_lines) == len(parsed_taxlabels):
            for i, label in enumerate(parsed_taxlabels):
                seq_dict[label] = matrix_lines[i].upper()
                taxlabels.append(label)

    # 2. Try FASTA format
    if not seq_dict and full_text.strip().startswith('>'):
        for record in SeqIO.parse(StringIO(full_text), 'fasta'):
            seq_dict[record.id] = str(record.seq).upper()
            taxlabels.append(record.id)

    # 3. Try Bio.AlignIO nexus parser
    if not seq_dict:
        try:
            alignment = AlignIO.read(StringIO(full_text), 'nexus')
            for record in alignment:
                seq_dict[record.id] = str(record.seq).upper()
                taxlabels.append(record.id)
        except Exception:
            pass

    # 4. Fallback: Parse standard labeled NEXUS matrix lines
    if not seq_dict and matrix_match:
        raw_matrix = matrix_match.group(1).strip()
        for line in raw_matrix.splitlines():
            line_clean = line.strip()
            if not line_clean:
                continue
            parts = line_clean.split(None, 1)
            if len(parts) == 2:
                sp_name = parts[0].strip("'\" ")
                seq_val = re.sub(r'\s+', '', parts[1]).upper()
                if len(seq_val) > 10:
                    if sp_name in seq_dict:
                        seq_dict[sp_name] += seq_val
                    else:
                        seq_dict[sp_name] = seq_val
                        taxlabels.append(sp_name)

    # 5. Extract embedded tree string from comment-stripped text
    tree_str = None
    first_p = text_no_comments.find('(')
    if first_p != -1:
        last_s = text_no_comments.find(';', first_p)
        if last_s != -1:
            tree_str = text_no_comments[first_p:last_s+1].strip()

    return seq_dict, taxlabels, tree_str


def parse_nexus_tree_file(filepath):
    tree_str = None
    open_func = gzip.open if filepath.endswith('.gz') else open
    with open_func(filepath, 'rt') as f:
        full_text = f.read()

    # Strip multiline NEXUS comments [ ... ] first
    clean_text = re.sub(r'\[.*?\]', '', full_text, flags=re.DOTALL)

    # 1. First check if a raw Newick tree string exists (starts with '(' and ends with ';')
    first_p = clean_text.find('(')
    if first_p != -1:
        last_s = clean_text.find(';', first_p)
        if last_s != -1:
            candidate = clean_text[first_p:last_s+1].strip()
            if candidate.endswith(';') and candidate.count('(') > 5:
                tree_str = candidate[:-1]

    # 2. Fallback: Parse explicit TREE statement if needed
    if tree_str is None:
        for line in full_text.splitlines():
            line_strip = re.sub(r'\[.*?\]', '', line).strip()
            if line_strip.upper().startswith('TREE '):
                parts = line_strip.split('=', 1)
                if len(parts) > 1:
                    tree_str = parts[1].strip()
                    if tree_str.endswith(';'):
                        tree_str = tree_str[:-1]
                    break

    return tree_str


def calculate_patristic_distances(tree):
    node_to_root_dist = {}
    node_to_parent = {}
    
    def traverse(node, current_dist, parent):
        node_to_root_dist[node] = current_dist
        node_to_parent[node] = parent
        for child in node.clades:
            traverse(child, current_dist + (child.branch_length or 0.0), node)
            
    traverse(tree.root, 0.0, None)
    
    leaves = tree.get_terminals()
    leaf_names = [leaf.name for leaf in leaves if leaf.name]
    leaf_by_name = {leaf.name: leaf for leaf in leaves if leaf.name}
    
    leaf_paths = {}
    for leaf in leaves:
        if not leaf.name:
            continue
        path = []
        curr = leaf
        while curr is not None:
            path.append(curr)
            curr = node_to_parent[curr]
        leaf_paths[leaf.name] = path
        
    dist_matrix = {}
    tree_tensor_dict = {} # 3-Channel Unrooted Tree Feature Matrix
    
    for name in leaf_names:
        dist_matrix[name] = {name: 0.0}
        tree_tensor_dict[name] = {name: [0.0, 0.0, 0.0]}
        
    n = len(leaf_names)
    for i in range(n):
        name1 = leaf_names[i]
        path1 = leaf_paths[name1]
        set1 = set(path1)
        for j in range(i + 1, n):
            name2 = leaf_names[j]
            path2 = leaf_paths[name2]
            
            lca = None
            idx1 = 0
            idx2 = 0
            for p2_idx, node in enumerate(path2):
                if node in set1:
                    lca = node
                    idx2 = p2_idx
                    idx1 = path1.index(node)
                    break
                    
            if lca is not None:
                dist = node_to_root_dist[leaf_by_name[name1]] + node_to_root_dist[leaf_by_name[name2]] - 2 * node_to_root_dist[lca]
                node_count = float(idx1 + idx2)
            else:
                dist = node_to_root_dist[leaf_by_name[name1]] + node_to_root_dist[leaf_by_name[name2]]
                node_count = float(len(path1) + len(path2))
                
            density = math.log((node_count + 1.0) / (dist + 0.1))
            
            dist_matrix[name1][name2] = dist
            dist_matrix[name2][name1] = dist
            
            tree_feats = [dist, node_count, density]
            tree_tensor_dict[name1][name2] = tree_feats
            tree_tensor_dict[name2][name1] = tree_feats
            
    return leaf_names, dist_matrix, tree_tensor_dict


# =====================================================================
# 4. MAIN INFERENCE DRIVER PIPELINE
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert a NEXUS alignment and a NEXUS/Newick tree into model inputs, "
                    "and return codon-level selection strength (LRT) predictions using a trained regression transformer."
    )
    parser.add_argument("--alignment", required=True, help="Path to NEXUS alignment file (.gz or uncompressed)")
    parser.add_argument("--tree", help="Path to NEXUS or Newick tree file. If omitted, will try to read from the alignment.")
    parser.add_argument("--model", default="selection_transformer_best.pt", help="Path to trained model weights (.pt)")
    parser.add_argument("--output", help="Path to output predictions CSV. Defaults to [alignment_prefix]_regression_predictions.csv")
    parser.add_argument("--reference_seq", help="Name of reference sequence (e.g. hg, hg38). Defaults to first sequence.")
    parser.add_argument("--window_size", type=int, default=1, help="Alignment sliding window size centered at site (default: 1)")
    parser.add_argument("--max_species", type=int, default=512, help="Maximum number of sequences to feed to model (default: 512)")
    parser.add_argument("--device", help="Force run on device (cpu, mps, cuda). Auto-detected by default.")
    parser.add_argument("--call_mode", choices=["pvalue", "zscore", "percentile"], default="pvalue", help="Selection calling mode: 'pvalue' (default, LRT >= 3.81/5.14), 'zscore' (local Z >= 2.0/2.5), or 'percentile' (top 3%%/5%%)")
    parser.add_argument("--use_zscore", action="store_true", help="Enable local relative Z-score calling (equivalent to --call_mode zscore)")
    parser.add_argument("--tier1_percentile", type=float, default=98.0, help="Percentile threshold for Tier 1 High-Confidence calls (default: 98.0)")
    parser.add_argument("--tier2_percentile", type=float, default=95.0, help="Percentile threshold for Tier 2 Medium-Confidence calls (default: 95.0)")
    parser.add_argument("--tier1_zscore", type=float, default=2.5, help="Z-score threshold for Tier 1 High-Confidence calls (default: 2.5)")
    parser.add_argument("--tier2_zscore", type=float, default=2.0, help="Z-score threshold for Tier 2 Medium-Confidence calls (default: 2.0)")
    parser.add_argument("--tier1_lrt_gate", type=float, default=5.14, help="Absolute predicted LRT gate for Tier 1 calls (p <= 0.05, default: 5.14)")
    parser.add_argument("--tier2_lrt_gate", type=float, default=3.81, help="Absolute predicted LRT gate for Tier 2 calls (p <= 0.10, default: 3.81)")
    
    args = parser.parse_args()
    if args.use_zscore:
        args.call_mode = "zscore"
    
    # 1. Device Setup
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[*] Running inference on device: {device}")
    
    # 2. Parse NEXUS alignment
    print(f"[*] Parsing NEXUS alignment from: {args.alignment}")
    if not os.path.exists(args.alignment):
        print(f"[!] Error: Alignment file not found: {args.alignment}")
        sys.exit(1)
        
    seq_dict, taxlabels, embedded_tree_str = parse_nexus_alignment_and_embedded_tree(args.alignment)
    if not seq_dict:
        print("[!] Error: No sequences successfully parsed from NEXUS matrix.")
        sys.exit(1)
    print(f"[*] Alignment loaded: {len(seq_dict)} sequences, alignment length = {len(next(iter(seq_dict.values())))} nucleotides.")
    
    # 3. Parse Tree (embedded or separate)
    tree_str = None
    if args.tree:
        print(f"[*] Parsing separate tree file: {args.tree}")
        if not os.path.exists(args.tree):
            print(f"[!] Error: Tree file not found: {args.tree}")
            sys.exit(1)
        tree_str = parse_nexus_tree_file(args.tree)
    else:
        if embedded_tree_str:
            print("[*] Using tree embedded in the NEXUS alignment file.")
            tree_str = embedded_tree_str
        else:
            print("[!] Warning: No separate tree file provided and no embedded tree found in alignment.")
            
    # Load and clean tree string into Bio.Phylo tree object
    tree_obj = None
    species_names = []
    if tree_str:
        try:
            clean_tree_str = re.sub(r'\{[^}]*\}', '', tree_str)
            clean_tree_str = re.sub(r'\[.*?\]', '', clean_tree_str)
            clean_tree_str = clean_tree_str.strip().rstrip(';') + ';'
            trees = list(Phylo.parse(StringIO(clean_tree_str), 'newick'))
            if trees:
                tree_obj = trees[0]
                species_names = [leaf.name for leaf in tree_obj.get_terminals() if leaf.name]
                print(f"[*] Successfully parsed tree topology with {len(species_names)} leaves.")
        except Exception as e:
            print(f"[!] Warning: Tree parsing failed: {e}")
            
    # 4. Determine Reference Sequence and Selection of Species
    if args.reference_seq:
        ref_key = args.reference_seq
        if ref_key not in seq_dict:
            print(f"[!] Error: Reference sequence '{ref_key}' not found in alignment.")
            print(f"Available sequences: {list(seq_dict.keys())[:10]}...")
            sys.exit(1)
    else:
        heuristics = ['hg', 'hg38', 'human', next(iter(seq_dict.keys()))]
        ref_key = next((h for h in heuristics if h in seq_dict), next(iter(seq_dict.keys())))
        
    print(f"[*] Reference sequence selected: '{ref_key}'")
    
    if not species_names:
        species_names = list(seq_dict.keys())
        
    name_map = {}
    for align_name in seq_dict.keys():
        norm = align_name.replace("'", "").replace('"', '').strip()
        name_map[norm] = align_name
        
    matching_species = []
    for tree_name in species_names:
        norm_t = tree_name.replace("'", "").replace('"', '').strip()
        if norm_t in name_map:
            matching_species.append(name_map[norm_t])
            
    print(f"[*] Matched {len(matching_species)} species between the tree and alignment.")
    if not matching_species:
        print("[!] Warning: Zero matching species between tree and alignment. Falling back to alignment order.")
        matching_species = list(seq_dict.keys())
        
    if ref_key in matching_species:
        matching_species.remove(ref_key)
        matching_species.insert(0, ref_key)
    # 4.1. Estimate branch lengths using HyPhy FIRST if they are missing or all zero across tree topology
    if tree_obj:
        has_branch_lengths = any(
            clade.branch_length is not None and clade.branch_length > 0.0 
            for clade in tree_obj.find_clades() 
            if clade != tree_obj.root
        )
        
        if not has_branch_lengths:
            print("[!] Tree topology has no branch lengths. Running HyPhy to estimate branch lengths on alignment...")
            try:
                # Format full tree topology Newick string (removing any placeholder/0.0 branch lengths)
                out_stream = StringIO()
                Phylo.write(tree_obj, out_stream, 'newick')
                raw_tree_str = out_stream.getvalue().strip()
                clean_pruned_tree_str = re.sub(r':[0-9.eE-]+', '', raw_tree_str)
                
                # Write temporary alignment FASTA containing matching species
                os.makedirs("scratch", exist_ok=True)
                temp_fasta = "scratch/temp_hyphy_align.fa"
                with open(temp_fasta, "w") as f:
                    for spec in matching_species:
                        f.write(f">{spec}\n{seq_dict[spec]}\n")
                
                # Write temporary HyPhy batch script
                temp_bf = "scratch/temp_hyphy_est.bf"
                bf_content = f"""
DataSet ds = ReadDataFile("{temp_fasta}");
DataSetFilter df = CreateFilter(ds, 1);
HarvestFrequencies(freqs, df, 1, 1, 1);
global kappa = 1.0;
HKY85RateMatrix = [
    [*, kappa*t, t, kappa*t]
    [kappa*t, *, kappa*t, t]
    [t, kappa*t, *, kappa*t]
    [kappa*t, t, kappa*t, *]
];
Model HKY85Model = (HKY85RateMatrix, freqs);
UseModel(HKY85Model);
Tree T = "{clean_pruned_tree_str}";
LikelihoodFunction lf = (df, T);
Optimize(res, lf);
fprintf(stdout, Format(T, 0, 1));
"""
                with open(temp_bf, "w") as f:
                    f.write(bf_content)
                
                # Execute HyPhy
                res = subprocess.run(["hyphy", temp_bf], capture_output=True, text=True)
                
                # Clean up temporary files
                if os.path.exists(temp_fasta):
                    os.remove(temp_fasta)
                if os.path.exists(temp_bf):
                    os.remove(temp_bf)
                
                if res.returncode == 0 and res.stdout.strip():
                    start_idx = res.stdout.find('(')
                    end_idx = res.stdout.rfind(')')
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        estimated_tree_str = res.stdout[start_idx:end_idx+1]
                        tree_obj = Phylo.read(StringIO(estimated_tree_str), 'newick')
                        
                        # Mandatory Safeguard Assertion: Verify tree contains real estimated branch lengths
                        non_zero_branches = [c.branch_length for c in tree_obj.find_clades() if c.branch_length is not None and c.branch_length > 0]
                        if len(non_zero_branches) == 0:
                            raise RuntimeError(
                                "[❌ FATAL INFERENCE ERROR] Parsed HyPhy tree string contains 0 non-zero branch lengths! "
                                "Branch length formatting failed."
                            )
                        print(f"[*] HyPhy branch length estimation succeeded ({len(non_zero_branches)} non-zero branch lengths parsed).")
                        
                        out_file = args.output
                        if not out_file:
                            basename = os.path.basename(args.alignment)
                            if basename.endswith('.gz'):
                                basename = basename[:-3]
                            if basename.endswith('.nex') or basename.endswith('.nexus'):
                                basename = basename.rsplit('.', 1)[0]
                            out_file = f"{basename}_regression_predictions.csv"
                        
                        tree_out_path = out_file.rsplit('.', 1)[0] + "_estimated_tree.nwk"
                        with open(tree_out_path, "w") as f:
                            f.write(estimated_tree_str + "\n")
                        print(f"[*] Saved estimated tree to '{tree_out_path}'")
                    else:
                        print(f"[!] HyPhy tree string match failed. HyPhy stdout snippet: {res.stdout[:200]}")
                        print("[*] Falling back to flat evolutionary distance structure.")
                else:
                    print(f"[!] HyPhy estimation failed (code {res.returncode}). Stderr: {res.stderr}")
                    print("[*] Falling back to flat evolutionary distance structure.")
            except Exception as ex:
                print(f"[!] Error running HyPhy branch length estimation: {ex}")
                print("[*] Falling back to flat evolutionary distance structure.")

    # 4.2. Select Species Subsample (Max-PD or Cap)
    if len(matching_species) > args.max_species:
        if tree_obj:
            clean_species_full, patristic_dict_full, _ = calculate_patristic_distances(tree_obj)
            N_full = len(matching_species)
            D_full = np.zeros((N_full, N_full), dtype=np.float32)
            for i, sp1 in enumerate(matching_species):
                for j, sp2 in enumerate(matching_species):
                    if sp1 in patristic_dict_full and sp2 in patristic_dict_full[sp1]:
                        D_full[i, j] = patristic_dict_full[sp1][sp2]
            
            # Max-PD Farthest-Point Traversal
            selected_local = [0]
            min_dists = D_full[0].copy()
            for _ in range(1, args.max_species):
                next_idx = np.argmax(min_dists)
                selected_local.append(next_idx)
                min_dists = np.minimum(min_dists, D_full[next_idx])
                
            selected_species = [matching_species[i] for i in selected_local]
            print(f"[*] Max-PD (Faith's PD) Selection: Selected {len(selected_species)} / {N_full} species maximizing tree branch length.")
        else:
            selected_species = matching_species[:args.max_species]
    else:
        selected_species = matching_species
        
    num_selected = len(selected_species)
    print(f"[*] Final model input selection: {num_selected} species (max_species cap = {args.max_species})")
    
    dist_matrix = {}
    if tree_obj:
        try:
            species_names, dist_matrix, tree_tensor_dict = calculate_patristic_distances(tree_obj)
        except Exception as e:
            print(f"[!] Error calculating patristic distances: {e}")
            dist_matrix = {}
            tree_tensor_dict = {}
            
    if not dist_matrix:
        print("[!] Warning: Tree distances unavailable. Computing pairwise Jukes-Cantor sequence distances directly from MSA...")
        dist_matrix = {s1: {s2: 0.0 for s2 in selected_species} for s1 in selected_species}
        tree_tensor_dict = {s1: {s2: [0.0, 0.0, 0.0] for s2 in selected_species} for s1 in selected_species}
        n_sp = len(selected_species)
        for i in range(n_sp):
            sp1 = selected_species[i]
            seq1 = seq_dict[sp1]
            for j in range(i + 1, n_sp):
                sp2 = selected_species[j]
                seq2 = seq_dict[sp2]
                diffs = sum(1 for a, b in zip(seq1, seq2) if a != b and a != '-' and b != '-')
                total = sum(1 for a, b in zip(seq1, seq2) if a != '-' and b != '-')
                p = diffs / max(1, total)
                jc = -0.75 * math.log(max(1e-4, 1.0 - (4.0/3.0)*p)) if p < 0.75 else 2.0
                dist_matrix[sp1][sp2] = jc
                dist_matrix[sp2][sp1] = jc
                tree_tensor_dict[sp1][sp2] = [jc, 0.0, 0.0]
                tree_tensor_dict[sp2][sp1] = [jc, 0.0, 0.0]
        
    # 5. Build input tensors for model
    ref_seq = seq_dict[ref_key]
    if len(ref_seq) % 3 != 0:
        print(f"[!] Warning: Reference sequence length ({len(ref_seq)} nucs) is not a multiple of 3. Truncating tail.")
    total_codons = len(ref_seq) // 3
    print(f"[*] Reference sequence '{ref_key}' length: {total_codons} codons.")
    
    msa_tokens = torch.ones(total_codons, args.max_species, args.window_size, dtype=torch.long) * 65
    aa_tokens = torch.ones(total_codons, args.max_species, args.window_size, dtype=torch.long) * AA_TO_IDX['?']
    dist_tensor = torch.zeros(args.max_species, args.max_species, dtype=torch.float32)
    padding_mask = torch.ones(args.max_species, dtype=torch.bool) # True means padded
    
    dist_matrix_norm = {}
    for s1, targets in dist_matrix.items():
        s1_clean = str(s1).replace("'", "").replace('"', '').strip().lower()
        dist_matrix_norm[s1_clean] = {}
        for s2, val in targets.items():
            s2_clean = str(s2).replace("'", "").replace('"', '').strip().lower()
            dist_matrix_norm[s1_clean][s2_clean] = val
            
    for i, spec1 in enumerate(selected_species):
        padding_mask[i] = False
        norm1 = str(spec1).replace("'", "").replace('"', '').strip().lower()
        for j, spec2 in enumerate(selected_species):
            norm2 = str(spec2).replace("'", "").replace('"', '').strip().lower()
            dist_val = dist_matrix_norm.get(norm1, {}).get(norm2, 0.0)
            dist_tensor[i, j] = float(dist_val)
            
    # Strict Tree Distance Verification Assertion (Zero Silent Zero Distance Failures)
    valid_spec_count = len(selected_species)
    if valid_spec_count > 1:
        valid_sub_matrix = dist_tensor[:valid_spec_count, :valid_spec_count]
        off_diag_mask = ~torch.eye(valid_spec_count, dtype=torch.bool)
        off_diag_dists = valid_sub_matrix[off_diag_mask]
        max_d = off_diag_dists.max().item()
        mean_d = off_diag_dists.mean().item()
        min_d = off_diag_dists.min().item()
        
        if max_d <= 0.0:
            raise ValueError(
                f"\n[❌ FATAL INFERENCE ERROR] Distance matrix dist_tensor for {valid_spec_count} species is ALL ZEROS (max off-diagonal dist = 0.0)!\n"
                f"HyPhy or Jukes-Cantor patristic tree distance extraction failed. Model inference requires non-zero tree distances."
            )
        print(f"[*] Verified Patristic Tree Distance Tensor ({valid_spec_count} species): min={min_d:.6f}, mean={mean_d:.6f}, max={max_d:.6f}")

    # Compute MDS coordinates on active species matrix (matching training MSADataset)
    dist_sub_np = dist_tensor[:num_selected, :num_selected].numpy()
    mds_sub_np = compute_mds_coordinates(dist_sub_np, n_components=4)
    
    mds_coords_np = np.zeros((args.max_species, 4), dtype=np.float32)
    mds_coords_np[:num_selected] = mds_sub_np
    mds_coords_tensor = torch.from_numpy(mds_coords_np) # [max_species, 4]
            
    padding_mask_tensor = torch.ones(total_codons, args.max_species, dtype=torch.bool) # True means padded / gap
    variable_sites_flags = []
    
    half_win = args.window_size // 2
    for site_idx in range(1, total_codons + 1):
        site_codons = []
        site_aas = []
        spec_aas = []
        
        for s_idx, spec in enumerate(selected_species):
            seq = seq_dict.get(spec, "")
            seq_len_codons = len(seq) // 3
            
            # Check central site gap status
            if 1 <= site_idx <= seq_len_codons:
                nuc_idx = (site_idx - 1) * 3
                codon_cent = seq[nuc_idx:nuc_idx+3].upper()
                if get_aa_token(codon_cent) < 21 and '-' not in codon_cent and 'N' not in codon_cent and '?' not in codon_cent:
                    padding_mask_tensor[site_idx - 1, s_idx] = False
            
            for w_idx in range(args.window_size):
                codon_pos_1based = site_idx - half_win + w_idx
                if 1 <= codon_pos_1based <= seq_len_codons:
                    nuc_idx = (codon_pos_1based - 1) * 3
                    codon = seq[nuc_idx:nuc_idx+3].upper()
                    msa_tokens[site_idx - 1, s_idx, w_idx] = get_codon_token(codon)
                    aa_tokens[site_idx - 1, s_idx, w_idx] = get_aa_token(codon)
                    
            # Collect codons at the central site (site_idx itself) for stats
            if 1 <= site_idx <= seq_len_codons:
                nuc_idx = (site_idx - 1) * 3
                codon = seq[nuc_idx:nuc_idx+3].upper()
                if get_aa_token(codon) < 21 and '-' not in codon and 'N' not in codon and '?' not in codon:
                    site_codons.append(codon)
                    aa = translate_codon(codon)
                    site_aas.append(aa)
                    spec_aas.append(aa)
                else:
                    spec_aas.append('?')
            else:
                spec_aas.append('?')
                        
        variable_sites_flags.append(is_site_variable(site_codons, site_aas))
        
    msa_tokens = msa_tokens.to(device)
    aa_tokens = aa_tokens.to(device)
    dist_tensor = dist_tensor.unsqueeze(0).expand(total_codons, -1, -1).to(device)
    mds_coords_tensor = mds_coords_tensor.unsqueeze(0).expand(total_codons, -1, -1).to(device)
    padding_mask_tensor = padding_mask_tensor.to(device)
    
    # 6. Load trained model weights
    print(f"[*] Loading model checkpoint: {args.model}")
    if not os.path.exists(args.model):
        print(f"[!] Error: Model checkpoint not found: {args.model}")
        sys.exit(1)
        
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    
    ckpt_embed_dim = checkpoint.get('embed_dim', 256) if isinstance(checkpoint, dict) else 256
    ckpt_num_layers = checkpoint.get('num_layers', 6) if isinstance(checkpoint, dict) else 6
    ckpt_num_heads = checkpoint.get('num_heads', 8) if isinstance(checkpoint, dict) else 8
    
    ckpt_window_size = args.window_size
    if 'pos_embedding' in state_dict:
        ckpt_window_size = state_dict['pos_embedding'].shape[1]
        ckpt_embed_dim = state_dict['pos_embedding'].shape[-1]
        
    layer_indices = [int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('row_layers.') or k.startswith('col_layers.')]
    if layer_indices:
        ckpt_num_layers = max(layer_indices) + 1
        
    ckpt_num_streams = checkpoint.get('num_streams', 1) if isinstance(checkpoint, dict) else 1
    if 'pooling_gate.weight' in state_dict:
        ckpt_num_streams = 1
    elif 'stream_fusion.0.block_codon.weight' in state_dict:
        in_w = state_dict['stream_fusion.0.block_codon.weight'].shape[1]
        ckpt_num_streams = max(1, in_w // (ckpt_embed_dim // 2))
    elif 'stream_fusion.0.weight' in state_dict:
        in_w = state_dict['stream_fusion.0.weight'].shape[1]
        ckpt_num_streams = max(1, in_w // ckpt_embed_dim)
        
    print(f'[*] Auto-detected checkpoint architecture: embed_dim={ckpt_embed_dim}, num_layers={ckpt_num_layers}, num_heads={ckpt_num_heads}, window_size={ckpt_window_size}, num_streams={ckpt_num_streams}')
    
    # Check if checkpoint is legacy 1-head (reg_head) vs 5-head multi-task (alpha_head)
    has_5_heads = ('alpha_head.0.weight' in state_dict)
    if 'reg_head.0.weight' in state_dict and not has_5_heads:
        state_dict['lrt_head.0.weight'] = state_dict['reg_head.0.weight']
        state_dict['lrt_head.0.bias'] = state_dict['reg_head.0.bias']
        state_dict['lrt_head.3.weight'] = state_dict['reg_head.3.weight']
        state_dict['lrt_head.3.bias'] = state_dict['reg_head.3.bias']
    
    # Legacy checkpoint adaptation for static b_cutoffs buffer -> learnable b0 and theta_steps
    if 'lrt_ordinal_head.b_cutoffs' in state_dict and 'lrt_ordinal_head.b0' not in state_dict:
        b_cut = state_dict.pop('lrt_ordinal_head.b_cutoffs')
        state_dict['lrt_ordinal_head.b0'] = b_cut[0]
        steps_target = (b_cut[:-1] - b_cut[1:]).clamp(min=1e-4)
        inv_sp = torch.log(torch.expm1(steps_target))
        state_dict['lrt_ordinal_head.theta_steps'] = inv_sp
        
    # Legacy checkpoint adaptation for gain parameter -> absorb into unconstrained fc2.weight
    if 'lrt_ordinal_head.gain' in state_dict:
        gain_val = state_dict.pop('lrt_ordinal_head.gain')
        if 'lrt_ordinal_head.fc2.weight' in state_dict:
            w2 = state_dict['lrt_ordinal_head.fc2.weight']
            state_dict['lrt_ordinal_head.fc2.weight'] = F.normalize(w2, dim=1) * gain_val
    ckpt_num_thresholds = 16
    if 'lrt_ordinal_head.theta_steps' in state_dict:
        ckpt_num_thresholds = state_dict['lrt_ordinal_head.theta_steps'].shape[0] + 1
        
    model = PhyloAxialTransformer(
        num_tokens=66,
        embed_dim=ckpt_embed_dim,
        num_heads=ckpt_num_heads,
        num_layers=ckpt_num_layers,
        window_size=ckpt_window_size,
        max_species=args.max_species,
        num_thresholds=ckpt_num_thresholds
    ).to(device)
    
    model_dict = model.state_dict()
    missing_keys = [k for k in model_dict if k not in state_dict or model_dict[k].shape != state_dict[k].shape]
    if missing_keys:
        raise RuntimeError(
            f"\n[❌ CRITICAL ARCHITECTURE MISMATCH] The checkpoint '{args.model}' cannot be loaded into current model definition!\n"
            f"Missing or shape-mismatched keys ({len(missing_keys)} tensors):\n"
            f"  " + "\n  ".join(missing_keys[:10]) + "\n"
            f"Please run inference with matching architecture parameters."
        )

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    print("[*] Running model predictions in safe micro-batched passes (micro_batch_size=64)...")
    micro_b_size = 64
    pred_log_lrts_list = []
    pred_alphas_list = []
    pred_beta_negs_list = []
    pred_beta_poses_list = []
    pred_p_negs_list = []
    
    with torch.no_grad():
        for start_i in range(0, total_codons, micro_b_size):
            end_i = min(total_codons, start_i + micro_b_size)
            b_c = msa_tokens[start_i:end_i].to(device)
            b_a = aa_tokens[start_i:end_i].to(device)
            b_d = dist_tensor[start_i:end_i].to(device)
            b_m = mds_coords_tensor[start_i:end_i].to(device)
            b_p = padding_mask_tensor[start_i:end_i].to(device)
            
            out = model(b_c, b_a, b_d, b_m, b_p)
            if isinstance(out, tuple) and len(out) == 5 and has_5_heads:
                y_lrt, y_alpha, y_beta_neg, y_beta_pos, y_p_neg = out
                pred_log_lrts_list.append(y_lrt.cpu().numpy())
                pred_alphas_list.append(y_alpha.cpu().numpy())
                pred_beta_negs_list.append(y_beta_neg.cpu().numpy())
                pred_beta_poses_list.append(y_beta_pos.cpu().numpy())
                pred_p_negs_list.append(y_p_neg.cpu().numpy())
            elif isinstance(out, tuple):
                pred_log_lrts_list.append(out[0].cpu().numpy())
                has_5_heads = False
            else:
                pred_log_lrts_list.append(out.cpu().numpy())
                has_5_heads = False
                
    pred_log_lrts = np.concatenate(pred_log_lrts_list, axis=0) if pred_log_lrts_list else np.array([])
    if has_5_heads:
        pred_alphas = np.concatenate(pred_alphas_list, axis=0)
        pred_beta_negs = np.concatenate(pred_beta_negs_list, axis=0)
        pred_beta_poses = np.concatenate(pred_beta_poses_list, axis=0)
        pred_p_negs = np.concatenate(pred_p_negs_list, axis=0)
        
    # 8. Build Predictions Table
    predictions = []
    for site_idx in range(1, total_codons + 1):
        ref_nuc_idx = (site_idx - 1) * 3
        ref_codon = ref_seq[ref_nuc_idx:ref_nuc_idx+3].upper()
        ref_aa = translate_codon(ref_codon)
        
        is_var = variable_sites_flags[site_idx - 1] if site_idx - 1 < len(variable_sites_flags) else 1
        
        gap_frac = float(padding_mask_tensor[site_idx - 1, :num_selected].float().mean().cpu())
        
        if not is_var or ref_aa == '-' or gap_frac >= 0.50:
            pred_log_lrt = 0.0
            pred_lrt = 0.0
            pred_alpha = 0.0
            pred_beta_pos = 0.0
            pred_p_pos = 0.0
        else:
            pred_lrt = max(0.0, float(pred_log_lrts[site_idx - 1]))
            pred_log_lrt = math.log1p(pred_lrt)
            if has_5_heads:
                pred_alpha = max(0.0, math.expm1(float(pred_alphas[site_idx - 1])))
                pred_beta_pos = max(0.0, math.expm1(float(pred_beta_poses[site_idx - 1])))
                pred_p_pos = round(1.0 - float(pred_p_negs[site_idx - 1]), 4)
            else:
                pred_alpha = 0.0
                pred_beta_pos = 0.0
                pred_p_pos = 0.0
            
        row_dict = {
            "codon_site": site_idx,
            "ref_codon": ref_codon,
            "ref_aa": ref_aa,
            "gap_fraction": round(float(padding_mask_tensor[site_idx - 1, :num_selected].float().mean().cpu()), 4),
            "is_variable": int(is_var),
            "predicted_log_lrt": round(pred_log_lrt, 5),
            "predicted_lrt": round(pred_lrt, 5)
        }
        if has_5_heads:
            row_dict["predicted_alpha_dS"] = round(pred_alpha, 4)
            row_dict["predicted_beta_pos_dN"] = round(pred_beta_pos, 4)
            row_dict["predicted_p_pos"] = pred_p_pos
            
        predictions.append(row_dict)
        
    df_preds = pd.DataFrame(predictions)
    
    # Calculate local relative metrics for all variable sites
    df_preds["local_z_score"] = 0.0
    df_preds["local_percentile"] = 0.0
    df_preds["selection_call"] = "Neutral"
    
    var_mask = (df_preds["is_variable"] == 1) & (df_preds["gap_fraction"] < 0.50)
    if var_mask.sum() > 0:
        var_lrts = df_preds.loc[var_mask, "predicted_lrt"].values
        mean_lrt = np.mean(var_lrts)
        std_lrt = np.std(var_lrts)
        
        if std_lrt > 0:
            df_preds.loc[var_mask, "local_z_score"] = np.round((var_lrts - mean_lrt) / std_lrt, 4)
        else:
            df_preds.loc[var_mask, "local_z_score"] = 0.0
            
        ranks = df_preds.loc[var_mask, "predicted_lrt"].rank(pct=True) * 100.0
        df_preds.loc[var_mask, "local_percentile"] = np.round(ranks, 2)
        
        # Apply Tier calling based on requested call_mode
        if args.call_mode == "zscore":
            t1_cond = df_preds["local_z_score"] >= args.tier1_zscore
            t2_cond = (df_preds["local_z_score"] >= args.tier2_zscore) & ~t1_cond
            t1_label, t2_label = f"Tier 1 (Z >= {args.tier1_zscore})", f"Tier 2 (Z >= {args.tier2_zscore})"
        elif args.call_mode == "percentile":
            t1_cond = df_preds["local_percentile"] >= args.tier1_percentile
            t2_cond = (df_preds["local_percentile"] >= args.tier2_percentile) & ~t1_cond
            t1_label, t2_label = f"Tier 1 (Percentile >= {args.tier1_percentile}%)", f"Tier 2 (Percentile >= {args.tier2_percentile}%)"
        else: # "pvalue" (default)
            t1_cond = df_preds["predicted_lrt"] >= args.tier1_lrt_gate
            t2_cond = (df_preds["predicted_lrt"] >= args.tier2_lrt_gate) & ~t1_cond
            t1_label, t2_label = f"Tier 1 (p <= 0.05, LRT >= {args.tier1_lrt_gate:.2f})", f"Tier 2 (p <= 0.10, LRT >= {args.tier2_lrt_gate:.2f})"
        
        df_preds.loc[t1_cond, "selection_call"] = t1_label
        df_preds.loc[t2_cond, "selection_call"] = t2_label
        
    # 9. Output predictions
    out_file = args.output
    if not out_file:
        basename = os.path.basename(args.alignment)
        if basename.endswith('.gz'):
            basename = basename[:-3]
        if basename.endswith('.nex') or basename.endswith('.nexus'):
            basename = basename.rsplit('.', 1)[0]
        out_file = f"{basename}_regression_predictions.csv"
        
    df_preds.to_csv(out_file, index=False)
    
    # 10. Print Summary
    print(f"\n==================================================")
    print("✨ Regression Prediction Summary")
    print(f"==================================================")
    print(f"Total codon sites predicted: {total_codons}")
    print(f"Mean predicted log(LRT+1):   {df_preds['predicted_log_lrt'].mean():.4f}")
    print(f"Mean predicted raw LRT:      {df_preds['predicted_lrt'].mean():.4f}")
    print(f"Max predicted raw LRT:       {df_preds['predicted_lrt'].max():.4f}")
    
    t1_sites = df_preds[df_preds["selection_call"].str.startswith("Tier 1")]
    t2_sites = df_preds[df_preds["selection_call"].str.startswith("Tier 2")]
    
    print(f"Calling selection based on '{args.call_mode}' mode:")
    if args.call_mode == "zscore":
        print(f"  - Tier 1 (Z-score >= {args.tier1_zscore}): {len(t1_sites)} sites")
        print(f"  - Tier 2 (Z-score >= {args.tier2_zscore}): {len(t2_sites)} sites")
    elif args.call_mode == "percentile":
        print(f"  - Tier 1 (Percentile >= {args.tier1_percentile}%): {len(t1_sites)} sites")
        print(f"  - Tier 2 (Percentile >= {args.tier2_percentile}%): {len(t2_sites)} sites")
    else:
        print(f"  - Tier 1 (p <= 0.05, predicted_lrt >= {args.tier1_lrt_gate:.2f}): {len(t1_sites)} sites")
        print(f"  - Tier 2 (p <= 0.10, predicted_lrt >= {args.tier2_lrt_gate:.2f}): {len(t2_sites)} sites")
    called_sites = df_preds[(df_preds["selection_call"] != "Neutral") & (df_preds["gap_fraction"] < 0.50)]
    if len(called_sites) > 0:
        print(f"\nPredicted positive selection sites:")
        cols_show = ["codon_site", "ref_aa", "gap_fraction", "predicted_lrt", "local_z_score", "local_percentile", "selection_call"]
        print(called_sites.sort_values(by="predicted_lrt", ascending=False)[cols_show].to_string(index=False))
    else:
        print("\n[!] Note: No sites passed absolute pvalue gate (Tier 2 cutoff).")
        print("Top 10 relative candidate sites by Z-score / Percentile:")
        valid_candidates = df_preds[(df_preds["is_variable"] == 1) & (df_preds["gap_fraction"] < 0.50)]
        df_top10 = valid_candidates.sort_values(by="local_z_score", ascending=False).head(10)
        cols_top = ["codon_site", "ref_aa", "gap_fraction", "is_variable", "predicted_lrt", "local_z_score", "local_percentile"]
        print(df_top10[cols_top].to_string(index=False))
        
    print(f"\n🎉 Predictions complete! Results saved to '{out_file}'")

if __name__ == "__main__":
    main()