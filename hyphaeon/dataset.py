"""
hyphaeon/dataset.py
------------------
Data preprocessing, tokenization, tree patristic distance calculation,
classical 4D MDS embedding, alignment & tree parsing, embedded tree extraction,
branch length validation, and HyPhy branch length estimation.
"""

import os
import sys
import gzip
import shutil
import tempfile
import subprocess
import re
from io import StringIO
from typing import Optional, Tuple, Dict, List

import numpy as np
import scipy.stats as stats
import torch
from Bio import SeqIO, Phylo

GENETIC_CODE = {
    'TTT': 0, 'TTC': 1, 'TTA': 2, 'TTG': 3, 'TCT': 4, 'TCC': 5, 'TCA': 6, 'TCG': 7,
    'TAT': 8, 'TAC': 9, 'TAA': 64, 'TAG': 64, 'TGT': 10, 'TGC': 11, 'TGA': 64, 'TGG': 12,
    'CTT': 13, 'CTC': 14, 'CTA': 15, 'CTG': 16, 'CCT': 17, 'CCC': 18, 'CCA': 19, 'CCG': 20,
    'CAT': 21, 'CAC': 22, 'CAA': 23, 'CAG': 24, 'CGT': 25, 'CGC': 26, 'CGA': 27, 'CGG': 28,
    'ATT': 29, 'ATC': 30, 'ATA': 31, 'ATG': 32, 'ACT': 33, 'ACC': 34, 'ACA': 35, 'ACG': 36,
    'AAT': 37, 'AAC': 38, 'AAA': 39, 'AAG': 40, 'AGT': 41, 'AGC': 42, 'AGA': 43, 'AGG': 44,
    'GTT': 45, 'GTC': 46, 'GTA': 47, 'GTG': 48, 'GCT': 49, 'GCC': 50, 'GCA': 51, 'GCG': 52,
    'GAT': 53, 'GAC': 54, 'GAA': 55, 'GAG': 56, 'GGT': 57, 'GGC': 58, 'GGA': 59, 'GGG': 60
}

AA_MAP = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}

CODON_TO_AA = {
    'TTT':'F', 'TTC':'F', 'TTA':'L', 'TTG':'L', 'TCT':'S', 'TCC':'S', 'TCA':'S', 'TCG':'S',
    'TAT':'Y', 'TAC':'Y', 'TAA':'*', 'TAG':'*', 'TGT':'C', 'TGC':'C', 'TGA':'*', 'TGG':'W',
    'CTT':'L', 'CTC':'L', 'CTA':'L', 'CTG':'L', 'CCT':'P', 'CCC':'P', 'CCA':'P', 'CCG':'P',
    'CAT':'H', 'CAC':'H', 'CAA':'Q', 'CAG':'Q', 'CGT':'R', 'CGC':'R', 'CGA':'R', 'CGG':'R',
    'ATT':'I', 'ATC':'I', 'ATA':'I', 'ATG':'M', 'ACT':'T', 'ACC':'T', 'ACA':'T', 'ACG':'T',
    'AAT':'N', 'AAC':'N', 'AAA':'K', 'AAG':'K', 'AGT':'S', 'AGC':'S', 'AGA':'R', 'AGG':'R',
    'GTT':'V', 'GTC':'V', 'GTA':'V', 'GTG':'V', 'GCT':'A', 'GCC':'A', 'GCA':'A', 'GCG':'A',
    'GAT':'D', 'GAC':'D', 'GAA':'E', 'GAG':'E', 'GGT':'G', 'GGC':'G', 'GGA':'G', 'GGG':'G'
}

def get_codon_token(codon: str) -> int:
    return GENETIC_CODE.get(codon.upper(), 64)

def get_aa_token(codon: str) -> int:
    aa = CODON_TO_AA.get(codon.upper(), '-')
    return AA_MAP.get(aa, 20)

def parse_alignment_sequences(filepath: str) -> Dict[str, str]:
    """
    Parses FASTA, NEXUS, or PHYLIP (sequential/interleaved) format alignments (including compressed .gz files).
    """
    filepath = os.path.expanduser(filepath)
    open_func = gzip.open if filepath.endswith('.gz') else open
    with open_func(filepath, 'rt') as f:
        full_text = f.read()

    # 1. PHYLIP / Sequential / Interleaved format (starts with 'ntaxa nsites' header)
    lines_nonempty = [l.strip() for l in full_text.splitlines() if l.strip()]
    if lines_nonempty:
        first_tokens = lines_nonempty[0].split()
        if len(first_tokens) == 2 and first_tokens[0].isdigit() and first_tokens[1].isdigit():
            seq_dict = {}
            curr_name = None
            curr_seq = []
            for line in lines_nonempty[1:]:
                parts = line.split(None, 1)
                if len(line) <= 35 and not any(c in line for c in '- ') and (line.isalnum() or '_' in line):
                    if curr_name:
                        seq_dict[curr_name] = "".join(curr_seq).upper().replace('U', 'T')
                    curr_name = line
                    curr_seq = []
                elif len(parts) == 2 and (parts[0].isalnum() or '_' in parts[0]) and len(parts[0]) <= 35 and len(parts[1]) > 10:
                    if curr_name:
                        seq_dict[curr_name] = "".join(curr_seq).upper().replace('U', 'T')
                    curr_name = parts[0]
                    curr_seq = [parts[1].replace(' ', '')]
                else:
                    curr_seq.append(line.replace(' ', ''))
            if curr_name:
                seq_dict[curr_name] = "".join(curr_seq).upper().replace('U', 'T')
            if seq_dict and len(seq_dict) >= int(first_tokens[0]):
                return seq_dict

    # 2. Fast direct FASTA format parsing
    if full_text.strip().startswith('>'):
        seq_dict = {}
        curr_id = None
        curr_chunks = []
        for line in full_text.splitlines():
            l_strip = line.strip()
            if not l_strip:
                continue
            if l_strip.startswith('>'):
                if curr_id is not None:
                    seq_dict[curr_id] = "".join(curr_chunks).upper().replace('U', 'T')
                curr_id = l_strip[1:].split()[0].strip("'\"")
                curr_chunks = []
            elif l_strip.startswith('(') or l_strip.lower().startswith('tree ') or l_strip.lower().startswith('begin '):
                break
            else:
                curr_chunks.append(l_strip.replace(' ', ''))
        if curr_id is not None:
            seq_dict[curr_id] = "".join(curr_chunks).upper().replace('U', 'T')
        return seq_dict

    # NEXUS format parsing
    clean_nexus_text = re.sub(r'\[[^\]]*\]', '', full_text)
    taxlabels = []
    tax_match = re.search(r'taxlabels\s+(.*?)\s*;', clean_nexus_text, re.IGNORECASE | re.DOTALL)
    if tax_match:
        tokens = re.findall(r"'([^']+)'|\"([^\"]+)\"|(\S+)", tax_match.group(1))
        for t in tokens:
            name = t[0] or t[1] or t[2]
            if name:
                taxlabels.append(name.strip())

    format_match = re.search(r'format\s+(.*?)\s*;', clean_nexus_text, re.IGNORECASE | re.DOTALL)
    is_nolabels = bool(format_match and 'nolabels' in format_match.group(1).lower())

    matrix_match = re.search(r'matrix\s+(.*?)\s*;', clean_nexus_text, re.IGNORECASE | re.DOTALL)
    seq_dict = {}
    if matrix_match:
        matrix_lines = [l.strip() for l in matrix_match.group(1).splitlines() if l.strip()]
        if is_nolabels and taxlabels:
            for idx, line in enumerate(matrix_lines):
                if idx < len(taxlabels):
                    seq = line.replace(' ', '').replace('\t', '').upper().replace('U', 'T')
                    seq_dict[taxlabels[idx]] = seq
        else:
            for line in matrix_lines:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    name = parts[0].replace("'", "").replace('"', '').strip()
                    seq = parts[1].replace(' ', '').replace('\t', '').strip().upper().replace('U', 'T')
                    if name in seq_dict:
                        seq_dict[name] += seq
                    else:
                        seq_dict[name] = seq
    else:
        # Fallback for plain matrices without explicit block wrapper
        for line in clean_nexus_text.splitlines():
            line_clean = line.strip()
            parts = line_clean.split(None, 1)
            if len(parts) == 2 and len(parts[1].replace(' ', '')) > 20:
                name = parts[0].replace("'", "").replace('"', '').strip()
                seq_dict[name] = parts[1].replace(' ', '').upper().replace('U', 'T')

    return seq_dict

def extract_tree_from_string_or_file(source: str) -> Optional[Phylo.BaseTree.Tree]:
    """
    Parses a tree from a file path or raw string. Supports Newick, Nexus, and embedded trees.
    """
    if isinstance(source, str):
        source_exp = os.path.expanduser(source)
        if os.path.exists(source_exp):
            source = source_exp
    if os.path.exists(source):
        open_func = gzip.open if source.endswith('.gz') else open
        with open_func(source, 'rt') as f:
            content = f.read()
    else:
        content = source

    # 1. Search for explicit Nexus / HyPhy TREE command
    tree_match = re.search(r'tree\s+[^=]+=\s*(\([^;]+;)', content, re.IGNORECASE)
    if tree_match:
        raw_nwk = tree_match.group(1)
        clean_nwk = re.sub(r'\{[^}]*\}', '', raw_nwk) # strip HyPhy comments like {Foreground}
        clean_nwk = re.sub(r'\[[^\]]*\]', '', clean_nwk) # strip Nexus comments
        try:
            return Phylo.read(StringIO(clean_nwk), 'newick')
        except Exception:
            pass

    # 2. Search for any standard Newick string starting with '('
    for line in content.splitlines():
        line_clean = line.strip()
        if line_clean.startswith('(') and line_clean.count('(') >= 2:
            if not line_clean.endswith(';'):
                line_clean += ';'
            clean_nwk = re.sub(r'\{[^}]*\}', '', line_clean)
            clean_nwk = re.sub(r'\[[^\]]*\]', '', clean_nwk)
            try:
                return Phylo.read(StringIO(clean_nwk), 'newick')
            except Exception:
                pass

    # 3. Direct parse attempt on whole content
    clean_all = content.strip()
    if clean_all.startswith('(') and clean_all.count('(') >= 2:
        if not clean_all.endswith(';'):
            clean_all += ';'
        clean_nwk = re.sub(r'\{[^}]*\}', '', clean_all)
        clean_nwk = re.sub(r'\[[^\]]*\]', '', clean_nwk)
        try:
            return Phylo.read(StringIO(clean_nwk), 'newick')
        except Exception:
            pass

    return None

def has_nonzero_branch_lengths(tree: Phylo.BaseTree.Tree) -> bool:
    """
    Returns True if the tree contains valid, positive branch lengths across most branches.
    """
    branches = [c.branch_length for c in tree.find_clades() if c != tree.root]
    if not branches:
        return False
    pos_branches = [b for b in branches if b is not None and b > 0.0]
    return len(pos_branches) > 0 and (len(pos_branches) / len(branches) >= 0.5)

def estimate_tree_branch_lengths_hyphy(seq_dict: Dict[str, str], tree_obj: Phylo.BaseTree.Tree) -> Optional[Phylo.BaseTree.Tree]:
    """
    Estimates tree branch lengths using HyPhy under the HKY85 model.
    """
    hyphy_path = shutil.which('hyphy')
    if not hyphy_path:
        return None

    # Prune tree to only taxa present in seq_dict
    seq_keys = set(seq_dict.keys()) | {k.strip("'\"") for k in seq_dict.keys()}
    to_prune = [term for term in tree_obj.get_terminals() if term.name and term.name.strip("'\"") not in seq_keys]
    for t in to_prune:
        try:
            tree_obj.prune(t)
        except Exception:
            pass

    # Get clean topology string without branch lengths
    out_stream = StringIO()
    Phylo.write(tree_obj, out_stream, 'newick')
    raw_tree_str = out_stream.getvalue().strip()
    clean_tree_str = re.sub(r':[0-9.eE-]+', '', raw_tree_str)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_fa = os.path.join(tmpdir, 'align.fa')
            with open(temp_fa, 'w') as f:
                for name, seq in seq_dict.items():
                    f.write(f'>{name}\n{seq}\n')

            temp_bf = os.path.join(tmpdir, 'est.bf')
            bf_content = f'''
DataSet ds = ReadDataFile("{temp_fa}");
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
Tree T = "{clean_tree_str}";
LikelihoodFunction lf = (df, T);
Optimize(res, lf);
fprintf(stdout, Format(T, 0, 1));
'''
            with open(temp_bf, 'w') as f:
                f.write(bf_content)

            res = subprocess.run([hyphy_path, temp_bf], capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                start_idx = res.stdout.find('(')
                end_idx = res.stdout.rfind(')')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    est_tree_str = res.stdout[start_idx:end_idx+1]
                    tree_res = Phylo.read(StringIO(est_tree_str), 'newick')
                    return tree_res
    except Exception as e:
        print(f"[!] HyPhy branch length estimation encountered an issue: {e}")

    return None

def enforce_nonzero_branch_lengths(tree_obj: Phylo.BaseTree.Tree, min_len: float = 1e-4, default_missing: float = 1e-3) -> Phylo.BaseTree.Tree:
    """
    Ensures all clades in the tree (except root) have valid, strictly positive branch lengths.
    """
    for clade in tree_obj.find_clades():
        if clade == tree_obj.root:
            continue
        if clade.branch_length is None:
            clade.branch_length = default_missing
        elif clade.branch_length < min_len:
            clade.branch_length = min_len
    return tree_obj

def compute_fast_dist_matrix(tree: Phylo.BaseTree.Tree, taxa: List[str]) -> np.ndarray:
    """
    Computes all-pairs patristic distance matrix across taxa from phylogenetic tree.
    """
    n = len(taxa)
    dist_mat = np.zeros((n, n), dtype=np.float32)
    taxa_set = set(taxa)
    terminals = {t.name.strip("'\""): t for t in tree.get_terminals() if t.name and t.name.strip("'\"") in taxa_set}
    
    root = tree.root
    depths = {}
    def calc_depths(node, current_depth=0.0):
        depths[id(node)] = current_depth
        for child in node.clades:
            bl = child.branch_length if child.branch_length is not None else 0.0
            calc_depths(child, current_depth + bl)
    calc_depths(root)
    
    parents = {}
    def get_parents(node):
        for child in node.clades:
            parents[id(child)] = node
            get_parents(child)
    get_parents(root)
    
    for i in range(n):
        ti_name = taxa[i]
        ti = terminals.get(ti_name)
        if ti is None: continue
        
        path_i = [id(ti)]
        curr = id(ti)
        while curr in parents:
            curr = id(parents[curr])
            path_i.append(curr)
        path_i_set = set(path_i)
        
        for j in range(i, n):
            tj_name = taxa[j]
            tj = terminals.get(tj_name)
            if tj is None: continue
            
            curr = id(tj)
            while curr not in path_i_set:
                curr = id(parents[curr])
            lca_id = curr
            d = depths[id(ti)] + depths[id(tj)] - 2.0 * depths[lca_id]
            dist_mat[i, j] = d
            dist_mat[j, i] = d
            
    return dist_mat

def compute_mds_coordinates(dist_matrix: np.ndarray, n_components: int = 4) -> np.ndarray:
    """
    Classical Multidimensional Scaling (MDS) embedding into 4D continuous coordinate space.
    Uses fast Truncated Lanczos spectral solver for N > 500, with dense eigh fallback.
    """
    n = dist_matrix.shape[0]
    if n > 500:
        try:
            import scipy.sparse.linalg as sla
            D2 = dist_matrix ** 2
            def matvec(v):
                hv = v - np.mean(v)
                dhv = D2.dot(hv)
                return -0.5 * (dhv - np.mean(dhv))
            from scipy.sparse.linalg import LinearOperator
            B_op = LinearOperator((n, n), matvec=matvec, dtype=np.float32)
            eigvals, eigvecs = sla.eigsh(B_op, k=n_components, which='LA', maxiter=300)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            pos_eigvals = np.maximum(eigvals[:n_components], 0)
            coords = eigvecs[:, :n_components] * np.sqrt(pos_eigvals)
            if coords.shape[1] < n_components:
                pad = np.zeros((n, n_components - coords.shape[1]))
                coords = np.hstack([coords, pad])
            return coords.astype(np.float32)
        except Exception:
            pass # Fallback to standard dense eigh

    H = np.eye(n, dtype=np.float32) - (1.0 / n)
    B = -0.5 * H.dot(dist_matrix ** 2).dot(H)
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    pos_eigvals = np.maximum(eigvals[:n_components], 0)
    coords = eigvecs[:, :n_components] * np.sqrt(pos_eigvals)
    if coords.shape[1] < n_components:
        pad = np.zeros((n, n_components - coords.shape[1]))
        coords = np.hstack([coords, pad])
    return coords.astype(np.float32)

def downsample_taxa_faith_pd(dist_mat: np.ndarray, taxa: List[str], max_species: int) -> Tuple[np.ndarray, List[str]]:
    """
    Greedily selects max_species taxa maximizing Faith's Phylogenetic Diversity (PD) / tree spread
    using farthest-point traversal on the patristic distance matrix.
    """
    n = len(taxa)
    if max_species >= n or max_species <= 0:
        return dist_mat, taxa

    # Step 1: Start with pair of most distant taxa
    idx1, idx2 = np.unravel_index(np.argmax(dist_mat), dist_mat.shape)
    selected_indices = [idx1, idx2]
    min_dists = np.minimum(dist_mat[idx1], dist_mat[idx2])

    # Step 2: Greedily add taxon with maximum distance to the current set
    for _ in range(2, max_species):
        next_idx = int(np.argmax(min_dists))
        selected_indices.append(next_idx)
        min_dists = np.minimum(min_dists, dist_mat[next_idx])

    selected_taxa = [taxa[i] for i in selected_indices]
    sub_dist_mat = dist_mat[np.ix_(selected_indices, selected_indices)]
    return sub_dist_mat, selected_taxa

def prune_identical_sequences(seq_dict: Dict[str, str], taxa: List[str]) -> Tuple[List[str], Dict[str, List[str]], int]:
    """
    Identifies and collapses 100% identical sequence duplicates among matching taxa.
    Retains 1 representative taxon per unique haplotype and prunes redundant duplicate leaves.
    Returns (unique_taxa, dup_map, num_pruned).
    """
    seen_seqs = {}
    unique_taxa = []
    dup_map = {}
    
    for t in taxa:
        seq = seq_dict[t]
        if seq not in seen_seqs:
            seen_seqs[seq] = t
            unique_taxa.append(t)
            dup_map[t] = []
        else:
            rep = seen_seqs[seq]
            dup_map[rep].append(t)
            
    num_pruned = len(taxa) - len(unique_taxa)
    return unique_taxa, dup_map, num_pruned

def load_alignment_and_tree(fa_path: str, nwk_path: Optional[str] = None, max_species: Optional[int] = None, prune_duplicates: bool = True):
    """
    Parses alignment (FASTA or NEXUS) and phylogenetic tree (from nwk_path or embedded in alignment).
    Enforces non-zero branch lengths (estimating them via HyPhy if available and missing).
    Automatically prunes identical sequence duplicates and trims tree accordingly if prune_duplicates=True.
    Optionally applies greedy Faith's PD species downsampling if max_species is specified.
    Returns PyTorch tensors (c, a, d, z), invariable mask, taxa list, and codon length L.
    """
    # 1. Parse alignment sequences
    seq_dict = parse_alignment_sequences(fa_path)
    if not seq_dict:
        raise ValueError(f"Could not parse any sequences from alignment file: '{fa_path}'")

    # 2. Extract or load tree
    tree_obj = None
    tree_source_desc = ""
    if nwk_path is not None:
        tree_obj = extract_tree_from_string_or_file(nwk_path)
        if tree_obj is None:
            raise ValueError(f"Could not parse phylogenetic tree from specified path: '{nwk_path}'")
        tree_source_desc = f"external file ({nwk_path})"
    else:
        # Attempt to find tree embedded in alignment
        tree_obj = extract_tree_from_string_or_file(fa_path)
        if tree_obj is None:
            raise ValueError(
                f"No tree specified (--tree), and no embedded phylogenetic tree found in alignment '{fa_path}'. "
                f"Please provide a tree via -t / --tree."
            )
        tree_source_desc = f"embedded in alignment ({fa_path})"

    # 3. Branch length validation / HyPhy estimation
    if not has_nonzero_branch_lengths(tree_obj):
        if shutil.which("hyphy"):
            print(f"[*] Tree ({tree_source_desc}) has no branch lengths. Estimating via HyPhy (HKY85)...")
            est_tree = estimate_tree_branch_lengths_hyphy(seq_dict, tree_obj)
            if est_tree is not None and has_nonzero_branch_lengths(est_tree):
                tree_obj = est_tree
                print(f"[✓] HyPhy branch length estimation succeeded.")
            else:
                print(f"[!] HyPhy estimation unsuccessful; enforcing minimum positive branch lengths.")
        else:
            print(f"[*] Tree ({tree_source_desc}) lacks branch lengths (HyPhy not found); enforcing default positive branch lengths.")

    # Guarantee all branch lengths strictly positive
    enforce_nonzero_branch_lengths(tree_obj, min_len=1e-4)

    # 4. Match taxa between tree and alignment
    tree_taxa = [term.name.strip("'\"") for term in tree_obj.get_terminals() if term.name]
    taxa = [t for t in tree_taxa if t in seq_dict]
    if not taxa:
        # Try matching with stripped names
        seq_keys_clean = {k.strip("'\""): k for k in seq_dict.keys()}
        taxa = [seq_keys_clean[t] for t in tree_taxa if t in seq_keys_clean]
        if not taxa:
            # Try case-insensitive matching
            seq_keys_lower = {k.strip("'\"").lower(): k for k in seq_dict.keys()}
            matched_keys = []
            for t in tree_taxa:
                t_clean = t.strip("'\"").lower()
                if t_clean in seq_keys_lower:
                    matched_keys.append(seq_keys_lower[t_clean])
            taxa = matched_keys
            if not taxa:
                raise ValueError(
                    f"No matching taxa found between tree terminals ({tree_taxa[:5]}...) "
                    f"and alignment sequences ({list(seq_dict.keys())[:5]}...)."
                )

    dropped_aln = len(seq_dict) - len(taxa)
    dropped_tree = len(tree_taxa) - len(taxa)
    if dropped_aln > 0 or dropped_tree > 0:
        print(f"[*] Taxon Matching: Retained {len(taxa)} shared taxa ({dropped_aln} alignment sequences, {dropped_tree} tree terminals unshared).")
    else:
        print(f"[*] Taxon Matching: 100% concordance ({len(taxa)} shared taxa).")

    # Automated Duplicate Sequence & Tree Pruning
    if prune_duplicates and len(taxa) > 1:
        unique_taxa, dup_map, num_pruned = prune_identical_sequences(seq_dict, taxa)
        if num_pruned > 0:
            print(f"[*] Duplicate Taxon Pruning: Collapsed {num_pruned} identical duplicate sequence(s) ({len(taxa)} -> {len(unique_taxa)} unique haplotypes).")
            taxa = unique_taxa

    # Sequence length & reading frame validation
    seq_lengths = {sp: len(seq_dict[sp]) for sp in taxa}
    unique_lens = set(seq_lengths.values())
    if len(unique_lens) > 1:
        print(f"[!] Warning: Unequal sequence lengths detected in alignment: {unique_lens}. Padding shorter sequences with gaps.")

    first_seq = seq_dict[taxa[0]]
    raw_len = len(first_seq)
    if raw_len < 3:
        raise ValueError(f"Alignment sequence length ({raw_len} bp) is less than 1 codon (3 bp).")

    if raw_len % 3 != 0:
        print(f"[!] Notice: Alignment length ({raw_len} bp) is not divisible by 3. Trimming {raw_len % 3} trailing nucleotide(s).")
    
    L = raw_len // 3
    n_taxa = len(taxa)

    # 5. Fast pre-downsampling for massive sequence collections (N > 300)
    if max_species is not None and len(taxa) > max_species:
        # Uniformly stride downsample to 2 * max_species before computing distance matrix
        stride = max(1, len(taxa) // (max_species * 2))
        taxa = taxa[::stride][:max_species * 2]

    # Compute distance matrix
    dist_mat = compute_fast_dist_matrix(tree_obj, taxa)

    # If tree branch lengths are raw mutation counts (> 10.0) rather than substitutions per site,
    # normalize by alignment codon length L to bring distances into standard evolutionary scale
    if dist_mat.max() > 10.0:
        dist_mat = dist_mat / L

    if max_species is not None and len(taxa) > max_species:
        dist_mat, taxa = downsample_taxa_faith_pd(dist_mat, taxa, max_species)
        print(f"[*] Faith's PD Species Downsampling: Selected {len(taxa)} taxa maximizing tree diversity.")

    n_taxa = len(taxa)
    mds_coords = compute_mds_coordinates(dist_mat, n_components=4)

    c_all = np.zeros((L, n_taxa, 1), dtype=np.int64)
    a_all = np.zeros((L, n_taxa, 1), dtype=np.int64)
    unknown_codons = 0
    stop_codons = 0
    total_codons = L * n_taxa

    for i, sp in enumerate(taxa):
        seq = seq_dict.get(sp, '-' * (L * 3))
        for site in range(L):
            codon = seq[site*3 : (site+1)*3].upper()
            c_tok = get_codon_token(codon)
            a_tok = get_aa_token(codon)
            if c_tok == 64:
                if codon in ('TAA', 'TAG', 'TGA'):
                    stop_codons += 1
                else:
                    unknown_codons += 1
            c_all[site, i, 0] = c_tok
            a_all[site, i, 0] = a_tok

    if unknown_codons > 0:
        frac_unk = unknown_codons / max(1, total_codons)
        if frac_unk > 0.05:
            print(f"[!] Diagnostic Notice: {unknown_codons}/{total_codons} ({frac_unk:.1%}) codons contain gaps, ambiguities, or unrecognized bases.")

    if stop_codons > 0:
        print(f"[!] Notice: Found {stop_codons} in-frame stop codon(s) across alignment matrix.")

    is_aa_invariable = np.zeros(L, dtype=bool)
    for site in range(L):
        aa_col = a_all[site, :, 0]
        valid_aa = aa_col[aa_col < 20]
        if len(np.unique(valid_aa)) <= 1:
            is_aa_invariable[site] = True

    c_tensor = torch.tensor(c_all, dtype=torch.long)
    a_tensor = torch.tensor(a_all, dtype=torch.long)
    d_tensor = torch.tensor(dist_mat, dtype=torch.float32).unsqueeze(0)  # [1, N, N] broadcastable
    z_tensor = torch.tensor(mds_coords, dtype=torch.float32).unsqueeze(0)  # [1, N, 4] broadcastable

    return c_tensor, a_tensor, d_tensor, z_tensor, is_aa_invariable, taxa, L
