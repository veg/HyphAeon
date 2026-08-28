import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
import time
import re
import gzip
import math
import sqlite3
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Standard genetic code dictionary
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
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*', 'TGA':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
}

# Codon vocabulary setup (64 codons + gap + unknown)
codons_list = [a+b+c for a in "TCAG" for b in "TCAG" for c in "TCAG"]
CODON_TO_IDX = {c: i for i, c in enumerate(codons_list)}
CODON_TO_IDX['-'] = 64
CODON_TO_IDX['?'] = 65

def get_codon_token(codon):
    codon = codon.upper()
    if '-' in codon:
        return 64
    if len(codon) != 3 or 'N' in codon:
        return 65
    return CODON_TO_IDX.get(codon, 65)

AA_LIST = "ACDEFGHIKLMNPQRSTVWY*-?"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

def get_aa_token(codon):
    codon = codon.upper()
    if '-' in codon:
        return 21  # AA_TO_IDX['-']
    if len(codon) != 3 or 'N' in codon or '?' in codon:
        return 22  # AA_TO_IDX['?']
    aa = GENETIC_CODE.get(codon, '?')
    return AA_TO_IDX.get(aa, 22)

# Precomputed mapping from codon to amino acid index directly
CODON_TO_AA_IDX = {}
for c in codons_list:
    aa = GENETIC_CODE.get(c, '?')
    CODON_TO_AA_IDX[c] = AA_TO_IDX.get(aa, 22)

# Create the 16.7M elements lookup tables (base 256)
CODON_LOOKUP = np.ones(256 * 256 * 256, dtype=np.int8) * 65
AA_LOOKUP = np.ones(256 * 256 * 256, dtype=np.int8) * 22

# We find base-256 byte values
grid_arr = np.arange(256, dtype=np.int32)
b0_grid = grid_arr[:, None, None] * 65536
b1_grid = grid_arr[None, :, None] * 256
b2_grid = grid_arr[None, None, :]
flat_indices = (b0_grid + b1_grid + b2_grid).ravel()

b0_val = (flat_indices // 65536)
b1_val = (flat_indices // 256) % 256
b2_val = flat_indices % 256

# Gaps: contains '-' (45)
gap_mask = (b0_val == 45) | (b1_val == 45) | (b2_val == 45)
CODON_LOOKUP[gap_mask] = 64
AA_LOOKUP[gap_mask] = 21

# Unknown: contains 'N' (78), 'n' (110), '?' (63)
unknown_mask = (b0_val == 78) | (b1_val == 78) | (b2_val == 78) | \
               (b0_val == 110) | (b1_val == 110) | (b2_val == 110) | \
               (b0_val == 63) | (b1_val == 63) | (b2_val == 63)
unknown_mask = unknown_mask & (~gap_mask)
CODON_LOOKUP[unknown_mask] = 65
AA_LOOKUP[unknown_mask] = 22

# Overwrite with exact valid codons
for codon, c_idx in CODON_TO_IDX.items():
    if len(codon) == 3 and '-' not in codon and '?' not in codon:
        for c_str in [codon.upper(), codon.lower()]:
            b_encoded = c_str.encode('ascii')
            idx = b_encoded[0] * 65536 + b_encoded[1] * 256 + b_encoded[2]
            CODON_LOOKUP[idx] = c_idx
            aa = GENETIC_CODE.get(codon.upper(), '?')
            AA_LOOKUP[idx] = AA_TO_IDX.get(aa, 22)


# --- Lightweight Newick Parser and Distance Calculator ---
class TreeNode:
    def __init__(self, name=None, length=0.0):
        self.name = name
        self.length = length
        self.children = []
        self.parent = None

def parse_newick(newick_str):
    newick_str = re.sub(r'\{[^}]*\}', '', newick_str.strip())  # remove annotations
    newick_str = re.sub(r'\[.*?\]', '', newick_str)            # remove comments
    
    tokens = []
    i = 0
    while i < len(newick_str):
        c = newick_str[i]
        if c in '(),;':
            tokens.append(c)
            i += 1
        elif c == ':':
            i += 1
            start = i
            while i < len(newick_str) and newick_str[i] not in '(),;':
                i += 1
            tokens.append(('length', float(newick_str[start:i])))
        else:
            start = i
            while i < len(newick_str) and newick_str[i] not in '(),;:':
                i += 1
            tokens.append(('name', newick_str[start:i].strip()))
            
    root = TreeNode()
    current = root
    for t in tokens:
        if t == '(':
            child = TreeNode()
            child.parent = current
            current.children.append(child)
            current = child
        elif t == ',':
            current = current.parent
            child = TreeNode()
            child.parent = current
            current.children.append(child)
            current = child
        elif t == ')':
            current = current.parent
        elif isinstance(t, tuple) and t[0] == 'name':
            current.name = t[1]
        elif isinstance(t, tuple) and t[0] == 'length':
            current.length = t[1]
            
    return root

def get_path_to_root(node):
    path = []
    curr = node
    while curr is not None:
        path.append(curr)
        curr = curr.parent
    return path

def get_leaves(node):
    if not node.children:
        return [node]
    leaves = []
    for c in node.children:
        leaves.extend(get_leaves(c))
    return leaves

def get_patristic_distances(root):
    leaves = get_leaves(root)
    leaf_paths = {}
    leaf_len_to_root = {}
    for leaf in leaves:
        if not leaf.name:
            continue
        path = get_path_to_root(leaf)
        leaf_paths[leaf.name] = path
        length = 0.0
        for node in path[:-1]:
            length += node.length
        leaf_len_to_root[leaf.name] = length
        
    names = list(leaf_len_to_root.keys())
    n = len(names)
    dist_matrix = {}
    for name in names:
        dist_matrix[name] = {name: 0.0}
        
    for i in range(n):
        name1 = names[i]
        path1 = leaf_paths[name1]
        set1 = set(path1)
        for j in range(i + 1, n):
            name2 = names[j]
            path2 = leaf_paths[name2]
            lca = None
            for node in path2:
                if node in set1:
                    lca = node
                    break
            
            if lca is not None:
                lca_path = get_path_to_root(lca)
                lca_len = sum(node.length for node in lca_path[:-1])
                dist = leaf_len_to_root[name1] + leaf_len_to_root[name2] - 2 * lca_len
            else:
                dist = leaf_len_to_root[name1] + leaf_len_to_root[name2]
                
            dist_matrix[name1][name2] = dist
            dist_matrix[name2][name1] = dist
            
    return names, dist_matrix

def select_diverse_species_matrix(dist_matrix_np, k, keep_idx=0):
    n = dist_matrix_np.shape[0]
    if n <= k:
        return list(range(n))
        
    selected = [keep_idx]
    selected_set = {keep_idx}
    min_dists = dist_matrix_np[keep_idx].copy()
    
    for _ in range(k - 1):
        farthest_val = -1.0
        farthest_idx = -1
        for i in range(n):
            if i not in selected_set:
                if min_dists[i] > farthest_val:
                    farthest_val = min_dists[i]
                    farthest_idx = i
        if farthest_idx == -1:
            break
        selected.append(farthest_idx)
        selected_set.add(farthest_idx)
        min_dists = np.minimum(min_dists, dist_matrix_np[farthest_idx])
        
    return sorted(selected)

def compute_mds_coordinates(dist_matrix_np, n_components=4):
    N = dist_matrix_np.shape[0]
    if N <= n_components:
        coords = np.zeros((N, n_components), dtype=np.float32)
        return coords
    D2 = dist_matrix_np ** 2
    H = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * (H @ D2 @ H)
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    coords = np.zeros((N, n_components), dtype=np.float32)
    for i in range(n_components):
        val = evals[i]
        if val > 0:
            coords[:, i] = evecs[:, i] * np.sqrt(val)
    return coords

def is_site_variable(site_codons, site_aas):
    if not site_codons or not site_aas:
        return False
    unique_aas_set = set(site_aas)
    # Condition 1: Multiple amino acids
    if len(unique_aas_set) > 1:
        return True
    # Condition 2: Serine Island transition
    if len(unique_aas_set) == 1 and 'S' in unique_aas_set:
        has_tcn = any(c in ('TCA', 'TCC', 'TCG', 'TCT') for c in site_codons)
        has_agy = any(c in ('AGC', 'AGT') for c in site_codons)
        if has_tcn and has_agy:
            return True
    return False

def precompute_variable_sites_worker(task):
    gene, msa_path = task
    if not os.path.exists(msa_path):
        return gene, []
    taxlabels = []
    sequences = []
    in_taxlabels = False
    in_matrix = False
    try:
        with gzip.open(msa_path, 'rt') as f:
            for line in f:
                line_strip = line.strip()
                if not line_strip:
                    continue
                if line_strip.upper().startswith("TAXLABELS"):
                    in_taxlabels = True
                    content = line_strip[len("TAXLABELS"):].strip()
                    tokens = content.replace("'", "").replace(";", "").split()
                    taxlabels.extend(tokens)
                    if line_strip.endswith(";"):
                        in_taxlabels = False
                    continue
                if in_taxlabels:
                    tokens = line_strip.replace("'", "").replace(";", "").split()
                    taxlabels.extend(tokens)
                    if line_strip.endswith(";"):
                        in_taxlabels = False
                    continue
                if line_strip.upper().startswith("MATRIX"):
                    in_matrix = True
                    continue
                if in_matrix:
                    if line_strip == ";":
                        in_matrix = False
                        continue
                    if line_strip.endswith(";"):
                        sequences.append(line_strip[:-1].strip())
                        in_matrix = False
                        continue
                    sequences.append(line_strip)
                    continue
    except:
        return gene, []
        
    seq_dict = {}
    for label, seq in zip(taxlabels, sequences):
        seq_dict[label] = seq
        
    any_seq = next(iter(seq_dict.values())) if seq_dict else ""
    num_codons = len(any_seq) // 3
    
    species_codons = []
    for spec in taxlabels:
        seq = seq_dict.get(spec, "")
        codons = [seq[idx*3:idx*3+3].upper() for idx in range(num_codons)]
        species_codons.append(codons)
        
    variable_sites = [False] * (num_codons + 1)
    for site_idx in range(1, num_codons + 1):
        c_idx = site_idx - 1
        site_codons = []
        site_aas = []
        for s_idx in range(len(taxlabels)):
            if s_idx < len(species_codons):
                codon = species_codons[s_idx][c_idx]
                if '-' not in codon and 'N' not in codon and '?' not in codon and len(codon) == 3:
                    site_codons.append(codon)
                    aa = GENETIC_CODE.get(codon, '?')
                    if aa != '?':
                        site_aas.append(aa)
        variable_sites[site_idx] = is_site_variable(site_codons, site_aas)
    return gene, variable_sites


# --- PyTorch Dataset Loader ---
class MSADataset(Dataset):
    def __init__(self, db_path, msa_dir, sites_list, window_size=1, max_species=256, cache_dict=None, cache_dir=None, cache_size_limit=0, subsample_species=False):
        self.db_path = db_path
        self.msa_dir = msa_dir
        self.sites = sites_list
        self.window_size = window_size
        self.max_species = max_species
        self.cache_dict = cache_dict
        self.cache_dir = cache_dir
        self.cache_size_limit = cache_size_limit
        self.subsample_species = subsample_species
        self.alignment_cache = {}

    def __len__(self):
        return len(self.sites)
        
    def _load_msa(self, gene_name):
        if gene_name in self.alignment_cache:
            return self.alignment_cache[gene_name]
            
        if self.cache_dir and os.path.isdir(self.cache_dir):
            npz_path = os.path.join(self.cache_dir, f"{gene_name}.npz")
            if os.path.exists(npz_path):
                try:
                    data = np.load(npz_path, allow_pickle=True)
                    dist_tensor_cached = torch.from_numpy(data["dist_arr"])
                    if torch.isnan(dist_tensor_cached).any() or torch.isinf(dist_tensor_cached).any():
                        dist_tensor_cached = torch.nan_to_num(dist_tensor_cached, nan=0.0, posinf=0.0, neginf=0.0)
                    mds_coords_cached = torch.from_numpy(data["mds_coords"])
                    if torch.isnan(mds_coords_cached).any() or torch.isinf(mds_coords_cached).any():
                        mds_coords_cached = torch.nan_to_num(mds_coords_cached, nan=0.0, posinf=0.0, neginf=0.0)
                        
                    selected_indices = data["selected_indices"].tolist()
                    if len(selected_indices) > self.max_species:
                        selected_indices = select_diverse_species_matrix(data["dist_arr"], self.max_species, keep_idx=0)
                    self.alignment_cache[gene_name] = (
                        data["codon_ids_matrix"],
                        data["aa_ids_matrix"],
                        data["species_names"].tolist(),
                        dist_tensor_cached,
                        mds_coords_cached,
                        data["variable_sites"].tolist(),
                        selected_indices,
                        data["valid_seqs"]
                    )
                    # Limit cache size if cache_size_limit > 0 to prevent memory bloat (especially with multiprocessing dataloading)
                    if self.cache_size_limit > 0 and len(self.alignment_cache) > self.cache_size_limit:
                        first_key = next(iter(self.alignment_cache))
                        self.alignment_cache.pop(first_key)
                        
                    return self.alignment_cache[gene_name]
                except Exception as e:
                    print(f"Error loading NPZ cache for {gene_name}: {e}")
                    
        if self.cache_dict and gene_name in self.cache_dict:
            data = self.cache_dict[gene_name]
            
            # Check if this is the new compact pre-tokenized cache format
            if "codon_ids_matrix" in data:
                dist_tensor_cached = torch.from_numpy(data["dist_arr"])
                if torch.isnan(dist_tensor_cached).any() or torch.isinf(dist_tensor_cached).any():
                    dist_tensor_cached = torch.nan_to_num(dist_tensor_cached, nan=0.0, posinf=0.0, neginf=0.0)
                mds_coords_cached = torch.from_numpy(data["mds_coords"])
                if torch.isnan(mds_coords_cached).any() or torch.isinf(mds_coords_cached).any():
                    mds_coords_cached = torch.nan_to_num(mds_coords_cached, nan=0.0, posinf=0.0, neginf=0.0)
                
                selected_indices = data["selected_indices"]
                if len(selected_indices) > self.max_species:
                    selected_indices = select_diverse_species_matrix(data["dist_arr"], self.max_species, keep_idx=0)
                self.alignment_cache[gene_name] = (
                    data["codon_ids_matrix"],
                    data["aa_ids_matrix"],
                    data["species_names"],
                    dist_tensor_cached,
                    mds_coords_cached,
                    data["variable_sites"],
                    selected_indices,
                    data["valid_seqs"]
                )
                return self.alignment_cache[gene_name]
                
            if "seq_dict" in data and "species_names" in data and "dist_arr" in data:
                dist_tensor_cached = torch.from_numpy(data["dist_arr"])
                if torch.isnan(dist_tensor_cached).any() or torch.isinf(dist_tensor_cached).any():
                    dist_tensor_cached = torch.nan_to_num(dist_tensor_cached, nan=0.0, posinf=0.0, neginf=0.0)
                
                mds_coords_cached = torch.from_numpy(data.get("mds_coords")) if "mds_coords" in data else None
                if mds_coords_cached is not None:
                    if torch.isnan(mds_coords_cached).any() or torch.isinf(mds_coords_cached).any():
                        mds_coords_cached = torch.nan_to_num(mds_coords_cached, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    mds_coords_cached = torch.from_numpy(compute_mds_coordinates(data["dist_arr"], n_components=4))
                    
                var_sites = data.get("variable_sites")
                
                # Precompute diversity selection indices once per gene
                n_spec = len(data["species_names"])
                if n_spec > self.max_species:
                    selected_indices = select_diverse_species_matrix(data["dist_arr"], self.max_species, keep_idx=0)
                else:
                    selected_indices = list(range(n_spec))
                    
                # Pre-tokenize all sequences for this gene into 2D matrices once
                codon_ids_list = []
                aa_ids_list = []
                species_names = data["species_names"]
                seq_dict = data["seq_dict"]
                
                any_seq = next(iter(seq_dict.values())) if seq_dict else ""
                seq_len_codons = len(any_seq) // 3
                
                valid_seqs = np.zeros(n_spec, dtype=bool)
                for i, spec in enumerate(species_names):
                    seq = seq_dict.get(spec, "")
                    if seq:
                        valid_seqs[i] = True
                        seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
                        if len(seq_bytes) == seq_len_codons * 3:
                            codons_bytes = seq_bytes.reshape(seq_len_codons, 3)
                            indices = codons_bytes[:, 0].astype(np.int32) * 65536 + codons_bytes[:, 1].astype(np.int32) * 256 + codons_bytes[:, 2].astype(np.int32)
                            c_ids = CODON_LOOKUP[indices]
                            a_ids = AA_LOOKUP[indices]
                        else:
                            c_ids = np.ones(seq_len_codons, dtype=np.int8) * 65
                            a_ids = np.ones(seq_len_codons, dtype=np.int8) * 22
                    else:
                        c_ids = np.ones(seq_len_codons, dtype=np.int8) * 65
                        a_ids = np.ones(seq_len_codons, dtype=np.int8) * 22
                    codon_ids_list.append(c_ids)
                    aa_ids_list.append(a_ids)
                    
                codon_ids_matrix = np.stack(codon_ids_list, axis=0)
                aa_ids_matrix = np.stack(aa_ids_list, axis=0)
                
                self.alignment_cache[gene_name] = (
                    codon_ids_matrix,
                    aa_ids_matrix,
                    species_names,
                    dist_tensor_cached,
                    mds_coords_cached,
                    var_sites,
                    selected_indices,
                    valid_seqs
                )
                return self.alignment_cache[gene_name]
                
        align_path = os.path.join(self.msa_dir, f"{gene_name}.gz")
        if not os.path.exists(align_path):
            return None
            
        taxlabels = []
        sequences = []
        tree_str = None
        in_taxlabels = False
        in_matrix = False
        in_trees = False
        
        try:
            with gzip.open(align_path, 'rt') as f:
                for line in f:
                    line_strip = line.strip()
                    if not line_strip:
                        continue
                    if line_strip.upper().startswith("TAXLABELS"):
                        in_taxlabels = True
                        content = line_strip[len("TAXLABELS"):].strip()
                        tokens = content.replace("'", "").replace(";", "").split()
                        taxlabels.extend(tokens)
                        if line_strip.endswith(";"):
                            in_taxlabels = False
                        continue
                    if in_taxlabels:
                        tokens = line_strip.replace("'", "").replace(";", "").split()
                        taxlabels.extend(tokens)
                        if line_strip.endswith(";"):
                            in_taxlabels = False
                        continue
                    if line_strip.upper().startswith("MATRIX"):
                        in_matrix = True
                        continue
                    if in_matrix:
                        if line_strip == ";":
                            in_matrix = False
                            continue
                        if line_strip.endswith(";"):
                            sequences.append(line_strip[:-1].strip())
                            in_matrix = False
                            continue
                        sequences.append(line_strip)
                        continue
                    if line_strip.upper().startswith("BEGIN TREES"):
                        in_trees = True
                        continue
                    if in_trees:
                        if line_strip.upper().startswith("TREE "):
                            parts = line_strip.split('=', 1)
                            if len(parts) > 1:
                                tree_str = parts[1].strip()
                        if line_strip == "END;":
                            in_trees = False
        except Exception as e:
            print(f"Error loading MSA for {gene_name}: {e}")
            return None
            
        seq_dict = {}
        for label, seq in zip(taxlabels, sequences):
            seq_dict[label] = seq
            
        dist_matrix = {}
        species_names = []
        if tree_str:
            try:
                tree_root = parse_newick(tree_str)
                species_names, dist_matrix = get_patristic_distances(tree_root)
            except Exception as e:
                pass
                
        if not species_names:
            species_names = list(seq_dict.keys())
            dist_matrix = {s1: {s2: 0.0 for s2 in species_names} for s1 in species_names}
            
        n_spec = len(species_names)
        dist_arr = np.zeros((n_spec, n_spec), dtype=np.float32)
        for i, spec1 in enumerate(species_names):
            for j, spec2 in enumerate(species_names):
                dist_arr[i, j] = dist_matrix.get(spec1, {}).get(spec2, 0.0)
        dist_tensor_cached = torch.from_numpy(dist_arr)
        if torch.isnan(dist_tensor_cached).any() or torch.isinf(dist_tensor_cached).any():
            dist_tensor_cached = torch.nan_to_num(dist_tensor_cached, nan=0.0, posinf=0.0, neginf=0.0)
            
        mds_coords_np = compute_mds_coordinates(dist_arr, n_components=4)
        mds_coords_cached = torch.from_numpy(mds_coords_np)
        if torch.isnan(mds_coords_cached).any() or torch.isinf(mds_coords_cached).any():
            mds_coords_cached = torch.nan_to_num(mds_coords_cached, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Variable sites computation
        any_seq = next(iter(seq_dict.values())) if seq_dict else ""
        num_codons = len(any_seq) // 3
        species_codons = []
        for spec in species_names:
            seq = seq_dict.get(spec, "")
            codons = [seq[idx*3:idx*3+3].upper() for idx in range(num_codons)]
            species_codons.append(codons)
            
        variable_sites = [False] * (num_codons + 1)
        for site_idx in range(1, num_codons + 1):
            c_idx = site_idx - 1
            site_codons = []
            site_aas = []
            for s_idx in range(n_spec):
                codon = species_codons[s_idx][c_idx]
                if '-' not in codon and 'N' not in codon and '?' not in codon and len(codon) == 3:
                    site_codons.append(codon)
                    aa = GENETIC_CODE.get(codon, '?')
                    if aa != '?':
                        site_aas.append(aa)
            variable_sites[site_idx] = is_site_variable(site_codons, site_aas)
            
        # Precompute diversity selection indices once per gene
        if n_spec > self.max_species:
            selected_indices = select_diverse_species_matrix(dist_arr, self.max_species, keep_idx=0)
        else:
            selected_indices = list(range(n_spec))
            
        # Pre-tokenize all sequences for this gene once into 2D matrices
        codon_ids_list = []
        aa_ids_list = []
        valid_seqs = np.zeros(n_spec, dtype=bool)
        for i, spec in enumerate(species_names):
            seq = seq_dict.get(spec, "")
            if seq:
                valid_seqs[i] = True
                seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
                if len(seq_bytes) == num_codons * 3:
                    codons_bytes = seq_bytes.reshape(num_codons, 3)
                    indices = codons_bytes[:, 0].astype(np.int32) * 65536 + codons_bytes[:, 1].astype(np.int32) * 256 + codons_bytes[:, 2].astype(np.int32)
                    c_ids = CODON_LOOKUP[indices]
                    a_ids = AA_LOOKUP[indices]
                else:
                    c_ids = np.ones(num_codons, dtype=np.int8) * 65
                    a_ids = np.ones(num_codons, dtype=np.int8) * 22
            else:
                c_ids = np.ones(num_codons, dtype=np.int8) * 65
                a_ids = np.ones(num_codons, dtype=np.int8) * 22
            codon_ids_list.append(c_ids)
            aa_ids_list.append(a_ids)
            
        codon_ids_matrix = np.stack(codon_ids_list, axis=0)
        aa_ids_matrix = np.stack(aa_ids_list, axis=0)
            
        self.alignment_cache[gene_name] = (
            codon_ids_matrix,
            aa_ids_matrix,
            species_names,
            dist_tensor_cached,
            mds_coords_cached,
            variable_sites,
            selected_indices,
            valid_seqs
        )
        num_cached = len(self.alignment_cache)
        if num_cached % 100 == 0:
            print(f"    [DataLoader Cache] Loaded and parsed {num_cached} unique gene alignments...", flush=True)
        return self.alignment_cache[gene_name]
        
    def __getitem__(self, idx):
        gene_name, site_idx, label = self.sites[idx]
        
        data = self._load_msa(gene_name)
        if data is None:
            dummy_codon = torch.ones(self.max_species, self.window_size, dtype=torch.long) * 65
            dummy_aa = torch.ones(self.max_species, self.window_size, dtype=torch.long) * 22
            dummy_dist = torch.zeros(self.max_species, self.max_species)
            dummy_mds = torch.zeros(self.max_species, 4, dtype=torch.float32)
            dummy_mask = torch.ones(self.max_species, dtype=torch.bool)
            return dummy_codon, dummy_aa, dummy_dist, dummy_mds, dummy_mask, torch.tensor(label, dtype=torch.float)
            
        codon_ids_matrix, aa_ids_matrix, species_names, dist_tensor_cached, mds_coords_cached, _, selected_indices, valid_seqs = data
        
        # Subsample species count using precomputed selected_indices (precomputed once per gene)
        if self.subsample_species and len(selected_indices) > 10:
            target_n = random.randint(10, len(selected_indices))
            sampled_idx = [selected_indices[0]] + sorted(random.sample(selected_indices[1:], target_n - 1))
        else:
            sampled_idx = selected_indices
            
        sub_dist = dist_tensor_cached[sampled_idx][:, sampled_idx]
        sub_mds = mds_coords_cached[sampled_idx]
            
        codon_tokens = torch.ones(self.max_species, self.window_size, dtype=torch.long) * 65
        aa_tokens = torch.ones(self.max_species, self.window_size, dtype=torch.long) * 22
        dist_tensor = torch.zeros(self.max_species, self.max_species)
        mds_coords = torch.zeros(self.max_species, 4, dtype=torch.float32)
        padding_mask = torch.ones(self.max_species, dtype=torch.bool) # True means padded
        
        n_sel = len(sampled_idx)
        sampled_valid = valid_seqs[sampled_idx]
        padding_mask[:n_sel] = torch.from_numpy(~sampled_valid)
        
        half_win = self.window_size // 2
        
        # Precomputed fast vectorized slice lookup
        start_1based = site_idx - half_win
        end_1based = site_idx + half_win
        start_idx = start_1based - 1
        end_idx = end_1based
        
        w_start = 0
        w_end = self.window_size
        
        seq_len_codons = codon_ids_matrix.shape[1]
        
        if start_idx < 0:
            w_start = -start_idx
            start_idx = 0
        if end_idx > seq_len_codons:
            w_end = w_end - (end_idx - seq_len_codons)
            end_idx = seq_len_codons
            
        if start_idx < end_idx:
            c_slice = codon_ids_matrix[sampled_idx, start_idx:end_idx]
            a_slice = aa_ids_matrix[sampled_idx, start_idx:end_idx]
            codon_tokens[:n_sel, w_start:w_end] = torch.from_numpy(c_slice.astype(np.int64))
            aa_tokens[:n_sel, w_start:w_end] = torch.from_numpy(a_slice.astype(np.int64))
            
        dist_tensor[:n_sel, :n_sel] = sub_dist
        mds_coords[:n_sel, :] = sub_mds
        
        return codon_tokens, aa_tokens, dist_tensor, mds_coords, padding_mask, torch.tensor(label, dtype=torch.float)


# --- Stable Attention Module ---
class StableAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        
    def forward(self, query, key, value, key_padding_mask=None):
        batch_size, q_seq_len, _ = query.shape
        k_seq_len = key.shape[1]
        
        w_q, w_k, w_v = torch.chunk(self.in_proj_weight, 3, dim=0)
        b_q, b_k, b_v = torch.chunk(self.in_proj_bias, 3, dim=0)
        
        w_q, w_k, w_v = w_q.contiguous(), w_k.contiguous(), w_v.contiguous()
        b_q, b_k, b_v = b_q.contiguous(), b_k.contiguous(), b_v.contiguous()
        
        q_proj = F.linear(query, w_q, b_q)
        k_proj = F.linear(key, w_k, b_k)
        v_proj = F.linear(value, w_v, b_v)
        
        q_h = q_proj.view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_h = k_proj.view(batch_size, k_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_h = v_proj.view(batch_size, k_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e9)
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v_h)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.embed_dim)
        return self.out_proj(out)


# --- Custom Row Attention with Learnable Phylogenetic Bias ---
class PhyloRowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.phylo_scale = nn.Parameter(torch.zeros(num_heads, 1, 1))
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, dist_matrix, padding_mask=None):
        batch_size, num_species, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_species, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias = dist_matrix.unsqueeze(1)
        scores = scores - torch.exp(self.phylo_scale.clamp(max=10.0)) * bias
        
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e9)
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_species, self.embed_dim)
        return self.out_proj(out)


# --- Custom Stable Transformer Encoder Layer ---
class StableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = StableAttention(d_model, nhead, dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src):
        attn_out = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout1(attn_out))
        
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff_out))
        return src


# --- Nano-scale Axial Transformer Architecture ---
class PhyloAxialTransformer(nn.Module):
    def __init__(self, num_tokens=66, embed_dim=128, num_heads=8, num_layers=4, window_size=1, max_species=256, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.max_species = max_species
        
        self.codon_embedding = nn.Embedding(num_tokens, embed_dim // 2)
        self.aa_embedding = nn.Embedding(23, embed_dim // 2)  # 23 amino acid tokens
        
        self.pos_embedding = nn.Parameter(torch.zeros(1, window_size, embed_dim))
        self.mds_proj = nn.Linear(4, embed_dim)
        
        self.col_layers = nn.ModuleList([
            StableTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=2*embed_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.row_layers = nn.ModuleList([
            PhyloRowAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.row_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        self.pool_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pool_attn = StableAttention(embed_dim, num_heads=num_heads, dropout=dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )
        
    def forward(self, msa_codons, msa_aas, dist_matrix, mds_coords, padding_mask=None):
        batch_size, num_species, window_size = msa_codons.shape
        
        codon_emb = self.codon_embedding(msa_codons)
        aa_emb = self.aa_embedding(msa_aas)
        
        x = torch.cat([codon_emb, aa_emb], dim=-1)
        x = x + self.pos_embedding.unsqueeze(1)
        
        phylo_pos = self.mds_proj(mds_coords)
        x = x + phylo_pos.unsqueeze(2)
        
        for i in range(len(self.col_layers)):
            col_in = x.reshape(batch_size * num_species, window_size, self.embed_dim)
            col_out = self.col_layers[i](col_in)
            x = col_out.reshape(batch_size, num_species, window_size, self.embed_dim)
            
            row_in = x.transpose(1, 2).contiguous().view(batch_size * window_size, num_species, self.embed_dim)
            dist_dup = dist_matrix.unsqueeze(1).expand(-1, window_size, -1, -1).contiguous().view(batch_size * window_size, num_species, num_species)
            
            if padding_mask is not None:
                padding_mask_dup = padding_mask.unsqueeze(1).expand(-1, window_size, -1).contiguous().view(batch_size * window_size, num_species)
            else:
                padding_mask_dup = None
                
            row_out = self.row_layers[i](row_in, dist_dup, padding_mask_dup)
            row_out = self.row_norms[i](row_in + row_out)
            x = row_out.reshape(batch_size, window_size, num_species, self.embed_dim).transpose(1, 2)
            
        central_idx = window_size // 2
        site_repr = x[:, :, central_idx, :]
        
        q = self.pool_query.expand(batch_size, -1, -1).contiguous()
        pooled_repr = self.pool_attn(q, site_repr, site_repr, key_padding_mask=padding_mask)
        pooled_repr = pooled_repr.squeeze(1)
        
        logits = self.mlp(pooled_repr)
        return logits.squeeze(1)


# --- Differentiable Ranking Loss via ListNet ---
class ListNetLoss(nn.Module):
    def __init__(self, target_transform='none'):
        super().__init__()
        self.target_transform = target_transform

    def forward(self, y_pred, y_true):
        y_true = torch.clamp(y_true, min=0.0)
        if self.target_transform == 'log1p':
            y_true_trans = torch.log1p(y_true)
        else:
            y_true_trans = y_true
            
        p_true = F.softmax(y_true_trans, dim=-1)
        log_p_pred = F.log_softmax(y_pred, dim=-1)
        
        loss = -torch.sum(p_true * log_p_pred, dim=-1)
        return loss


# --- Weighted Huber Loss for regression ---
class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0, rank_weight=0.5, power=2.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction='none')
        self.rank_weight = rank_weight
        self.power = power

    def forward(self, y_pred, y_true):
        raw_loss = self.huber(y_pred, y_true)
        # Apply power-law target weighting (default exponent = 2.0)
        weights = torch.pow(y_true + 1.0, self.power)
        loss_regression = (raw_loss * weights).mean()
        
        if self.rank_weight > 0.0 and y_pred.shape[0] > 1:
            # Pairwise ranking loss inside the batch to directly optimize rank correlations
            logits_diff = y_pred.unsqueeze(0) - y_pred.unsqueeze(1)
            labels_diff = y_true.unsqueeze(0) - y_true.unsqueeze(1)
            
            # Select pairs where labels[i] > labels[j] + margin
            margin = 0.1
            mask = labels_diff > margin
            
            if mask.sum() > 0:
                ranking_margin = 0.1
                loss_rank = torch.relu(-logits_diff[mask] + ranking_margin).mean()
            else:
                loss_rank = torch.tensor(0.0, device=y_pred.device)
        else:
            loss_rank = torch.tensor(0.0, device=y_pred.device)
            
        return loss_regression + self.rank_weight * loss_rank


# --- Full Training & Validation Routine ---
def train_full_model(db_path="meme_results.db", msa_dir="msa", epochs=5, batch_size=128, lr=2e-4, subsample_limit=None, loss_type="huber", target_transform="none", device_override=None, cache_path=None, epoch_size_limit=100000, num_workers=0, max_species=256, cache_size_limit=0, window_size=1, subsample_species=True, rank_weight=0.5, weights_path=None, subsample_val_limit=None, save_path="/Users/sergei/Projects/TOGA_MEME/MEME_transformer_joint.pt"):
    print("=" * 80)
    print("🚀 STEP 1: INITIALIZING HARDWARE ACCELERATION")
    print("=" * 80)
    
    loaded_cache = None
    cache_is_dir = False
    
    # Auto-resolve cache_path if not specified but the default NPZ directory exists locally
    if not cache_path and os.path.isdir("msa_cache_npz"):
        print(" -> Cache path not specified, but 'msa_cache_npz' directory found locally. Auto-selecting it to prevent slow fallback.")
        cache_path = "msa_cache_npz"
        
    if cache_path and os.path.exists(cache_path):
        if os.path.isdir(cache_path):
            print(f" -> Utilizing directory-based NPZ cache from {cache_path}...")
            cache_is_dir = True
        else:
            print(f" -> Loading precomputed alignment cache from {cache_path}...")
            t_cache0 = time.time()
            with gzip.open(cache_path, "rb") as f:
                loaded_cache = pickle.load(f)
            print(f" -> Successfully loaded cache for {len(loaded_cache):,} genes in {time.time()-t_cache0:.2f}s.")

    if device_override:
        device = torch.device(device_override)
        print(f" -> Forcing device selection: {device}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(" -> Detected Apple Silicon GPU! Utilizing MPS backend.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(" -> Detected NVIDIA GPU! Utilizing CUDA backend.")
    else:
        device = torch.device("cpu")
        print(" -> GPU acceleration not found. Training will run on CPU.")
        
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found!")
        return
        
    print("\n" + "=" * 80)
    print("🚀 STEP 2: EXTRACTING AND SPLITTING DATASET (PREVENTING DATA LEAKAGE)")
    print("=" * 80)
    print(" -> Querying SQLite database for all labeled site-level results (excluding synthetic nulls)...")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT gene_name, site_index, lrt 
        FROM site_results 
        WHERE lrt IS NOT NULL
          AND gene_name NOT LIKE 'null_%'
          AND gene_name NOT LIKE '%replicate%'
          AND gene_name NOT LIKE '%sim%'
    """)
    raw_sites = c.fetchall()
    conn.close()
    print(f" -> Successfully fetched {len(raw_sites):,} genuine biological sites (filtered out synthetic nulls).")
    
    # Shuffle and split by gene to avoid homology data leakage
    all_genes = sorted(list(set(row[0] for row in raw_sites)))
    random.seed(42)
    random.shuffle(all_genes)
    
    split_idx = int(len(all_genes) * 0.8)
    train_genes = set(all_genes[:split_idx])
    val_genes = set(all_genes[split_idx:])
    
    train_raw = [row for row in raw_sites if row[0] in train_genes]
    val_raw = [row for row in raw_sites if row[0] in val_genes]
    
    if subsample_limit:
        print(f" -> NOTE: Subsampling training dataset to {subsample_limit} sites.")
        train_raw = random.sample(train_raw, min(len(train_raw), subsample_limit))
        if not subsample_val_limit:
            val_raw = random.sample(val_raw, min(len(val_raw), subsample_limit // 4))
            
    # We do NOT subsample val_raw here to allow randomly selecting a different subset each epoch
        
    gene_to_variable_sites = {}
    if loaded_cache:
        for gene, cache_data in loaded_cache.items():
            if "variable_sites" in cache_data:
                gene_to_variable_sites[gene] = cache_data["variable_sites"]
        print(f"    - Variable site flags resolved from cache for {len(gene_to_variable_sites):,} genes.")
    elif cache_is_dir:
        # Load variable_sites flags. Try loading precompiled pkl first for speed.
        t_var0 = time.time()
        pkl_path = os.path.join(cache_path, "variable_sites.pkl")
        if os.path.exists(pkl_path):
            print(f" -> Loading precompiled variable site flags from {pkl_path}...")
            try:
                with open(pkl_path, "rb") as f_pkl:
                    gene_to_variable_sites = pickle.load(f_pkl)
                print(f"    - Successfully loaded variable site flags in {time.time()-t_var0:.2f}s.")
            except Exception as e:
                print(f"    - Error loading precompiled flags: {e}. Falling back to directory scan.")
                gene_to_variable_sites = {}
        
        if not gene_to_variable_sites:
            unique_genes_in_split = list(set(row[0] for row in raw_sites))
            print(f" -> Loading variable site flags on-the-fly from NPZ directory for {len(unique_genes_in_split):,} genes...")
            for idx, gene in enumerate(unique_genes_in_split):
                npz_path = os.path.join(cache_path, f"{gene}.npz")
                if os.path.exists(npz_path):
                    try:
                        data = np.load(npz_path, allow_pickle=True)
                        gene_to_variable_sites[gene] = data["variable_sites"].tolist()
                    except:
                        pass
            print(f"    - Successfully loaded variable site flags in {time.time()-t_var0:.2f}s.")
    else:
        print(f" -> Precomputing variable site flags for {len(all_genes):,} unique genes in parallel...")
        t_start_precompute = time.time()
        tasks = [(gene, os.path.join(msa_dir, f"{gene}.gz")) for gene in all_genes]
        from multiprocessing import Pool, cpu_count
        pool_workers = max(1, cpu_count() - 2)
        print(f"    - Utilizing {pool_workers} CPU worker processes...")
        with Pool(processes=pool_workers) as pool:
            for gene, var_sites in pool.imap_unordered(precompute_variable_sites_worker, tasks, chunksize=20):
                gene_to_variable_sites[gene] = var_sites
        print(f"    - Completed precomputing variable site flags in {time.time() - t_start_precompute:.2f}s.")
        
    print(" -> Filtering out invariable sites (unique_aas <= 1, excluding Serine islands)...")
    train_sites = []
    skipped_train = 0
    for gene, site, lrt in train_raw:
        variable_sites = gene_to_variable_sites.get(gene)
        if variable_sites is None:
            skipped_train += 1
            continue
        if 1 <= site < len(variable_sites) and variable_sites[site]:
            y = math.log(max(0.0, lrt) + 1.0)
            train_sites.append((gene, site, y))
        else:
            skipped_train += 1
            
    val_sites = []
    skipped_val = 0
    for gene, site, lrt in val_raw:
        variable_sites = gene_to_variable_sites.get(gene)
        if variable_sites is None:
            skipped_val += 1
            continue
        if 1 <= site < len(variable_sites) and variable_sites[site]:
            y = math.log(max(0.0, lrt) + 1.0)
            val_sites.append((gene, site, y))
        else:
            skipped_val += 1
            
    print(f" -> Variable Site Filter Complete:")
    print(f"    - Training sites kept: {len(train_sites):,} (skipped {skipped_train:,} invariable sites)")
    print(f"    - Validation sites kept: {len(val_sites):,} (skipped {skipped_val:,} invariable sites)")
    
    print(f" -> Split Stats:")
    print(f"    - Total unique genes: {len(all_genes)}")
    print(f"    - Training: {len(train_genes)} genes ({len(train_sites):,} sites)")
    print(f"    - Validation: {len(val_genes)} genes ({len(val_sites):,} sites)")
    
    train_active = [s for s in train_sites if s[2] > 0.0]
    train_neutral = [s for s in train_sites if s[2] <= 0.0]
    print(f" -> Balanced Sampling Setup:")
    print(f"    - Active training sites: {len(train_active):,}")
    print(f"    - Neutral training sites: {len(train_neutral):,}")
    
    print("\n" + "=" * 80)
    print("🚀 STEP 3: CREATING PYTORCH DATA LOADERS")
    print("=" * 80)
    train_dataset = MSADataset(db_path, msa_dir, train_sites, window_size=window_size, max_species=max_species, cache_dict=loaded_cache if not cache_is_dir else None, cache_dir=cache_path if cache_is_dir else None, cache_size_limit=cache_size_limit, subsample_species=subsample_species)
    # Save the full validation set for dynamic per-epoch sampling
    val_sites_all = val_sites
    if subsample_val_limit and len(val_sites_all) > subsample_val_limit:
        val_sites_init = random.sample(val_sites_all, subsample_val_limit)
        print(f" -> Validation evaluation will be subsampled dynamically to {subsample_val_limit} random sites per epoch.")
    else:
        val_sites_init = val_sites_all
        
    val_dataset = MSADataset(db_path, msa_dir, val_sites_init, window_size=window_size, max_species=max_species, cache_dict=loaded_cache if not cache_is_dir else None, cache_dir=cache_path if cache_is_dir else None, cache_size_limit=cache_size_limit, subsample_species=False)
    
    drop_last_train = len(train_dataset) >= batch_size
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if device.type not in ('cpu', 'mps') else False)
    print(f" -> Initial Validation Batches: {len(val_loader)} (Batch Size: {batch_size})")
    
    print("\n" + "=" * 80)
    print("🚀 STEP 4: INSTANTIATING MODEL, OPTIMIZER, AND LOSS FUNCTION")
    print("=" * 80)
    model = PhyloAxialTransformer(embed_dim=128, num_heads=8, num_layers=4, window_size=window_size, max_species=max_species).to(device)
    
    # Load existing weights if provided
    if weights_path and os.path.exists(weights_path):
        print(f" -> Loading existing model weights from '{weights_path}'...")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Filter state dict to handle potential mismatches gracefully
        model_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        if len(filtered_state_dict) < len(state_dict):
            print(f"    - Warning: Only loaded {len(filtered_state_dict)}/{len(state_dict)} tensors due to size or key mismatch.")
        model_dict.update(filtered_state_dict)
        model.load_state_dict(model_dict)
        print("    - Model weights loaded successfully.")
    elif weights_path:
        print(f" -> Warning: weights_path '{weights_path}' specified but file not found. Starting from scratch.")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    if loss_type == "listnet":
        print(f" -> Utilizing ListNet Loss (ranking) with target_transform='{target_transform}'")
        criterion = ListNetLoss(target_transform=target_transform)
    else:
        if rank_weight > 0.0:
            print(f" -> Utilizing Weighted Huber Loss (regression) + Pairwise Ranking Loss (weight: {rank_weight})")
        else:
            print(" -> Utilizing Weighted Huber Loss (regression only)")
        criterion = WeightedHuberLoss(rank_weight=rank_weight, power=loss_power)
    
    best_spearman = -1.0
    
    print("\n" + "=" * 80)
    print("🚀 STEP 5: TRAINING LOOP STARTING")
    print("=" * 80)
    for epoch in range(1, epochs + 1):
        print(f"\n--- 🌟 STARTING EPOCH {epoch}/{epochs} ---")
        
        # Resample balanced sites for this epoch
        if len(train_active) > 0 and len(train_neutral) > 0:
            sampled_neutral = random.sample(train_neutral, min(len(train_neutral), len(train_active)))
            epoch_sites = train_active + sampled_neutral
            
            # Limit the epoch size to prevent extremely long epochs on local machines
            if epoch_size_limit and len(epoch_sites) > epoch_size_limit:
                epoch_sites = random.sample(epoch_sites, epoch_size_limit)
                
            random.shuffle(epoch_sites)
            train_dataset.sites = epoch_sites
            print(f" -> Epoch {epoch} Balanced Dataset Size: {len(epoch_sites):,} sites ({len(epoch_sites)//batch_size} batches)")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last_train, num_workers=num_workers, pin_memory=True if device.type not in ('cpu', 'mps') else False)
        
        model.train()
        train_loss = 0.0
        
        # Let the user know the alignment cache is warming up
        if epoch == 1:
            print(" -> [Cache Warmup] Loading alignment files into memory for the first time. This may take 1-3 minutes...", flush=True)
            
        for batch_idx, (codon_tokens, aa_tokens, dist, mds_coords, padding_mask, labels) in enumerate(train_loader):
            codon_tokens = codon_tokens.to(device)
            aa_tokens = aa_tokens.to(device)
            dist = dist.to(device)
            mds_coords = mds_coords.to(device)
            padding_mask = padding_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(codon_tokens, aa_tokens, dist, mds_coords, padding_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print batch metrics frequently (especially in the beginning) to verify the script is not stuck
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader) or batch_idx < 10:
                phylo_scales = torch.exp(model.row_layers[0].phylo_scale).detach().cpu().squeeze().tolist()
                scales_str = ", ".join(f"{s:.3f}" for s in (phylo_scales if isinstance(phylo_scales, list) else [phylo_scales]))
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Tree Scales: [{scales_str}]", flush=True)
                
        avg_train_loss = train_loss / max(1, len(train_loader))
        print(f"\n -> Epoch {epoch} training complete. Running validation evaluation (loading validation alignments into memory cache)...", flush=True)
        
        model.eval()
        
        # Resample different validation sites dynamically for this epoch if subsample_val_limit is specified
        if subsample_val_limit and len(val_sites_all) > subsample_val_limit:
            epoch_val_sites = random.sample(val_sites_all, subsample_val_limit)
            val_dataset.sites = epoch_val_sites
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if device.type not in ('cpu', 'mps') else False)
            print(f" -> Dynamically selected a new validation subset of {len(epoch_val_sites):,} sites ({len(val_loader)} batches)...", flush=True)
            
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for codon_tokens, aa_tokens, dist, mds_coords, padding_mask, labels in val_loader:
                codon_tokens = codon_tokens.to(device)
                aa_tokens = aa_tokens.to(device)
                dist = dist.to(device)
                mds_coords = mds_coords.to(device)
                padding_mask = padding_mask.to(device)
                
                logits = model(codon_tokens, aa_tokens, dist, mds_coords, padding_mask)
                val_preds.extend(logits.cpu().tolist())
                val_targets.extend(labels.tolist())
                
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        mse = np.mean((val_preds - val_targets) ** 2)
        
        if len(np.unique(val_preds)) > 1 and len(np.unique(val_targets)) > 1:
            val_pearson = pearsonr(val_preds, val_targets)[0]
            val_spearman = spearmanr(val_preds, val_targets)[0]
            if np.isnan(val_spearman):
                val_spearman = 0.0
            if np.isnan(val_pearson):
                val_pearson = 0.0
        else:
            val_pearson, val_spearman = 0.0, 0.0
            
        print(f"\n📈 Epoch {epoch} Metrics Summary:")
        print(f"  - Average Training Loss: {avg_train_loss:.4f}")
        print(f"  - Validation MSE: {mse:.4f}")
        print(f"  - Validation Pearson r:  {val_pearson:.4f}")
        print(f"  - Validation Spearman rho: {val_spearman:.4f}")
        
        if val_spearman > best_spearman:
            best_spearman = val_spearman
            checkpoint_path = save_path
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_type': loss_type,
                'best_spearman_rho': best_spearman,
                'val_mse': mse,
                'val_pearson_r': val_pearson
            }, checkpoint_path)
            print(f"  💾 SUCCESS: Saved new best model checkpoint to '{checkpoint_path}' (Spearman rho improved to {best_spearman:.4f})")
        else:
            print(f"  ℹ️ Checkpoint not saved (Validation Spearman rho {val_spearman:.4f} did not exceed best of {best_spearman:.4f})")
            
    print("\n" + "=" * 80)
    print("🎉 MODEL TRAINING COMPLETE!")
    print(f"   Best Validation Spearman rho: {best_spearman:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Continuous Selection Transformer Model")
    parser.add_argument("--db_path", default="/Users/sergei/Dropbox/TOGA2026/meme_results.db", help="Path to SQLite database")
    parser.add_argument("--msa_dir", default="/Users/sergei/Dropbox/TOGA2026/msa", help="Path to compressed MSAs directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--subsample_limit", type=int, default=None, help="Limit number of training sites loaded (for testing)")
    parser.add_argument("--subsample_val_limit", type=int, default=None, help="Limit number of validation sites loaded to speed up evaluation")
    parser.add_argument("--loss_type", default="huber", choices=["huber", "listnet"], help="Loss function (huber or listnet)")
    parser.add_argument("--target_transform", default="none", choices=["none", "log1p"], help="Transformation for targets (listnet only)")
    parser.add_argument("--device", default=None, help="Force run on device (cpu, mps, cuda)")
    parser.add_argument("--cache_path", default=None, help="Path to precomputed MSA cache file (.pkl.gz)")
    parser.add_argument("--epoch_size_limit", type=int, default=100000, help="Max training sites per epoch to speed up local training (default: 100000)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for DataLoader (default: 0)")
    parser.add_argument("--max_species", type=int, default=256, help="Maximum number of species to subsample (default: 256)")
    parser.add_argument("--cache_size_limit", type=int, default=0, help="Maximum cache size for dataset alignments in memory (default: 0 = unlimited. Set to 256 for low RAM local execution)")
    parser.add_argument("--window_size", type=int, default=1, help="Window size around target site (default: 1)")
    parser.add_argument("--subsample_species", type=lambda x: (str(x).lower() == 'true'), default=True, help="Dynamically subsample species count during training (default: True)")
    parser.add_argument("--rank_weight", type=float, default=0.5, help="Pairwise ranking loss weight (default: 0.5)")
    parser.add_argument("--weights_path", default=None, help="Path to existing model checkpoint weights (.pt) to resume training from")
    parser.add_argument("--save_path", default="/Users/sergei/Projects/TOGA_MEME/MEME_transformer_joint.pt", help="Path to save the best trained model checkpoint weights (.pt)")
    args = parser.parse_args()
    
    train_full_model(
        db_path=args.db_path,
        msa_dir=args.msa_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        subsample_limit=args.subsample_limit,
        loss_type=args.loss_type,
        target_transform=args.target_transform,
        device_override=args.device,
        cache_path=args.cache_path,
        epoch_size_limit=args.epoch_size_limit,
        num_workers=args.num_workers,
        max_species=args.max_species,
        cache_size_limit=args.cache_size_limit,
        window_size=args.window_size,
        subsample_species=args.subsample_species,
        rank_weight=args.rank_weight,
        weights_path=args.weights_path,
        subsample_val_limit=args.subsample_val_limit,
        save_path=args.save_path,
        loss_power=args.loss_power
    )
