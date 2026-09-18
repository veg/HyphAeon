"""
aeon_core/dataset.py
--------------------
Data preprocessing, tokenization, tree patristic distance calculation,
classical 4D MDS embedding, alignment & tree parsing, embedded tree extraction,
branch length validation, and HyPhy branch length estimation.
"""

import os
import sys
import gzip
import warnings
import shutil
import tempfile
import subprocess
import re
import csv
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any, Union
import xml.etree.ElementTree as ET

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


def _parse_numeric_or_calendar_date(v_str: str) -> Optional[float]:
    """Helper to parse decimal year or ISO calendar dates into float timestamps."""
    if not v_str:
        return None
    s = str(v_str).strip().strip("'\"")
    try:
        return float(s)
    except ValueError:
        pass
    # YYYY-MM-DD
    m = re.match(r'^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$', s)
    if m:
        y, mth, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return float(y) + (mth - 1.0) / 12.0 + (day - 1.0) / 365.25
    # YYYY-MM
    m2 = re.match(r'^(\d{4})[-/](\d{1,2})$', s)
    if m2:
        y, mth = int(m2.group(1)), int(m2.group(2))
        return float(y) + (mth - 0.5) / 12.0
    return None


def parse_beast_xml(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Parses BEAST 1.x and BEAST 2.x XML configuration files.

    Extracts:
      - Multiple sequence alignments (nucleotide / coding sequences)
      - Heterochronous tip sampling dates (from <date> tags or TraitSet attributes)
      - Embedded starting tree / topology (if present as <newick>, <tree>, or <init>)
      - Reconciled taxon labels

    Returns a dictionary with keys:
      'sequences': Dict[str, str] (taxon -> sequence string)
      'dates': Dict[str, float] (taxon -> sampling timestamp)
      'tree_newick': Optional[str] (clean Newick string without rate annotations)
      'taxa': List[str] (taxa list in alignment order)
      'version': str ('BEAST 1', 'BEAST 2', or 'BEAST XML')
    """
    filepath_str = str(filepath)
    if os.path.exists(filepath_str):
        open_func = gzip.open if filepath_str.endswith('.gz') else open
        with open_func(filepath_str, 'rt', encoding='utf-8', errors='replace') as f:
            xml_content = f.read()
    else:
        xml_content = filepath_str

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse BEAST XML content from '{filepath_str[:60]}...': {e}")

    # Version heuristic
    is_beast2 = (
        'spec' in root.attrib or
        'namespace' in root.attrib.get('spec', '').lower() or
        any('spec' in el.attrib for el in root.iter())
    )
    detected_version = "BEAST 2" if is_beast2 else ("BEAST 1" if 'beast' in root.tag.lower() else "BEAST XML")

    # 1. Extract Sequences
    align_nodes = root.findall('.//alignment') + root.findall('.//data')
    candidates = []
    for node in align_nodes:
        seq_elems = node.findall('.//sequence')
        if not seq_elems:
            continue
        cur_dict = {}
        for s in seq_elems:
            t_child = s.find('./taxon')
            t_name = None
            if t_child is not None:
                t_name = t_child.attrib.get('idref') or t_child.attrib.get('id')
            if not t_name:
                t_name = s.attrib.get('taxon') or s.attrib.get('id')

            seq = ''
            if 'value' in s.attrib:
                seq = s.attrib['value']
            elif s.text and s.text.strip():
                seq = s.text.strip()
            elif t_child is not None and t_child.tail and t_child.tail.strip():
                seq = t_child.tail.strip()

            seq = re.sub(r'\s+', '', seq).upper().replace('U', 'T')
            if t_name and seq:
                cur_dict[t_name.strip("'\"")] = seq
        if cur_dict:
            candidates.append(cur_dict)

    seq_dict = max(candidates, key=len) if candidates else {}

    # 2. Extract Sampling Dates
    dates_map: Dict[str, float] = {}

    # BEAST 1.x format: <taxa><taxon id="..."><date value="..." .../></taxon></taxa>
    for tx in root.findall('.//taxon'):
        t_id = tx.attrib.get('id') or tx.attrib.get('idref')
        if not t_id:
            continue
        t_clean = t_id.strip("'\"")
        d = tx.find('./date')
        val_str = None
        if d is not None:
            val_str = d.attrib.get('value') or (d.text.strip() if d.text else None)
        elif 'date' in tx.attrib:
            val_str = tx.attrib['date']

        if val_str is not None:
            parsed_d = _parse_numeric_or_calendar_date(val_str)
            if parsed_d is not None:
                dates_map[t_clean] = parsed_d

    # BEAST 2.x format: <trait spec="beast.evolution.tree.TraitSet" traitname="date" value="...">
    for tr in root.findall('.//trait'):
        t_attr = (tr.attrib.get('traitname') or tr.attrib.get('name') or '').lower()
        if 'date' in t_attr:
            val_text = tr.attrib.get('value', tr.text or '')
            entries = re.split(r'[,;\n\r]+', val_text.strip())
            for entry in entries:
                if '=' in entry:
                    parts = entry.split('=', 1)
                    k = parts[0].strip().strip("'\"")
                    v_str = parts[1].strip()
                    parsed_d = _parse_numeric_or_calendar_date(v_str)
                    if parsed_d is not None:
                        dates_map[k] = parsed_d

    # 3. Taxon Name Reconciliation
    # Handle 'seq_' prefix or differences between sequence id and trait taxon names
    reconciled_seqs = {}
    for k, v in seq_dict.items():
        if k in dates_map:
            reconciled_seqs[k] = v
        elif k.startswith('seq_') and k[4:] in dates_map:
            reconciled_seqs[k[4:]] = v
        elif f"seq_{k}" in dates_map:
            reconciled_seqs[k] = v
            dates_map[k] = dates_map[f"seq_{k}"]
        else:
            reconciled_seqs[k] = v

    # 4. Starting Tree Extraction
    tree_newick = None
    for el in root.iter():
        if 'newick' in el.tag.lower() and el.text and el.text.strip().startswith('('):
            tree_newick = el.text.strip()
            break
        for attr_k, attr_v in el.attrib.items():
            if 'newick' in attr_k.lower() and isinstance(attr_v, str) and attr_v.strip().startswith('('):
                tree_newick = attr_v.strip()
                break
        if el.text and el.text.strip().startswith('(') and el.text.strip().endswith(';'):
            tree_newick = el.text.strip()
            break
        if tree_newick:
            break

    if tree_newick:
        # Strip BEAST comments [&rate=...] and HyPhy annotations
        tree_newick = re.sub(r'\[&[^\]]*\]', '', tree_newick)
        tree_newick = re.sub(r'\{[^}]*\}', '', tree_newick)
        if not tree_newick.endswith(';'):
            tree_newick += ';'

    return {
        'version': detected_version,
        'sequences': reconciled_seqs,
        'dates': dates_map,
        'tree_newick': tree_newick,
        'taxa': list(reconciled_seqs.keys())
    }


def _warn_internal_stops(seq_dict: Dict[str, str], L: int, taxa: List[str]) -> None:
    """Warn if any sequence has internal stop codons, which may indicate a frameshift."""
    worst_taxon = None
    worst_stops = 0
    internal_limit = max(0, (L - 1) * 3)
    for t in taxa:
        seq = seq_dict.get(t, "")
        stops = sum(1 for i in range(0, min(len(seq), internal_limit), 3)
                    if CODON_TO_AA.get(seq[i:i+3].upper(), '') == '*')
        if stops > worst_stops:
            worst_stops = stops
            worst_taxon = t
    if worst_stops >= 1:
        print(f"[!] Warning: {worst_stops} internal stop codon(s) found in '{worst_taxon}'. "
              f"This may indicate a frameshift or pseudogene. "
              f"Verify that the alignment is in-frame (codon-aligned).")

def parse_alignment_sequences(filepath: str) -> Dict[str, str]:
    """
    Parses FASTA, NEXUS, PHYLIP (sequential/interleaved), or BEAST XML format alignments (including compressed .gz files).
    """
    filepath = os.path.expanduser(filepath)
    open_func = gzip.open if filepath.endswith('.gz') else open
    with open_func(filepath, 'rt') as f:
        full_text = f.read()

    # 0. BEAST 1.x or BEAST 2.x XML format
    if filepath.lower().endswith(('.xml', '.xml.gz')) or (
        full_text.lstrip().startswith('<') and (
            '<beast' in full_text[:2000].lower() or
            '<alignment' in full_text[:2000].lower() or
            '<data' in full_text[:2000].lower()
        )
    ):
        beast_data = parse_beast_xml(filepath)
        if beast_data and beast_data.get('sequences'):
            return beast_data['sequences']

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

    # 0. Check for BEAST XML starting tree
    if (isinstance(source, str) and (source.lower().endswith(('.xml', '.xml.gz')) or (content.lstrip().startswith('<') and ('<beast' in content[:2000].lower() or '<newick' in content.lower())))):
        try:
            beast_data = parse_beast_xml(source)
            if beast_data.get('tree_newick'):
                clean_nwk = re.sub(r'\{[^}]*\}', '', beast_data['tree_newick'])
                clean_nwk = re.sub(r'\[[^\]]*\]', '', clean_nwk)
                return Phylo.read(StringIO(clean_nwk), 'newick')
        except Exception:
            pass

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

def has_nonzero_branch_lengths(tree: Phylo.BaseTree.Tree, min_pos_ratio: float = 0.05) -> bool:
    """
    Returns True if the tree contains valid, numeric branch lengths.
    A tree is considered to have branch lengths if the vast majority of branches
    have numeric lengths (not None) and at least some branches have positive divergence
    (default >= 5%, accommodating dense outbreak trees with many identical isolates).
    """
    branches = [c.branch_length for c in tree.find_clades() if c != tree.root]
    if not branches:
        return False
    numeric_branches = [b for b in branches if b is not None]
    if len(numeric_branches) < len(branches) * 0.5:
        return False
    pos_branches = [b for b in numeric_branches if b > 0.0]
    return len(pos_branches) > 0 and (len(pos_branches) / len(numeric_branches) >= min_pos_ratio)

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

def compute_fast_dist_matrix(tree: Phylo.BaseTree.Tree, taxa: List[str], strict: bool = False) -> np.ndarray:
    """
    Computes all-pairs patristic distance matrix across taxa from phylogenetic tree.

    Any taxon requested in ``taxa`` but absent from ``tree`` cannot be assigned a
    patristic distance and would otherwise be left as an all-zero row/column,
    silently telling the model it sits at distance 0 from every other taxon. To
    avoid that silent corruption, missing taxa are reported: a ``UserWarning`` is
    emitted by default, or a ``ValueError`` is raised when ``strict=True``.
    """
    n = len(taxa)
    dist_mat = np.zeros((n, n), dtype=np.float32)
    taxa_set = set(taxa)
    terminals = {t.name.strip("'\""): t for t in tree.get_terminals() if t.name and t.name.strip("'\"") in taxa_set}

    missing = [t for t in taxa if t not in terminals]
    if missing:
        preview = ", ".join(missing[:10]) + ("..." if len(missing) > 10 else "")
        msg = (
            f"{len(missing)} of {n} requested taxa are absent from the tree and will be "
            f"left at distance 0 from all others (silent corruption): {preview}"
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

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
            # Deterministic sign canonicalization: ensure max-magnitude component is positive
            for j in range(coords.shape[1]):
                max_idx = np.argmax(np.abs(coords[:, j]))
                if coords[max_idx, j] < 0:
                    coords[:, j] *= -1.0
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
    # Deterministic sign canonicalization: ensure max-magnitude component is positive
    for j in range(coords.shape[1]):
        max_idx = np.argmax(np.abs(coords[:, j]))
        if coords[max_idx, j] < 0:
            coords[:, j] *= -1.0
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

def compute_tn93_distance_matrix(
    seq_dict: Dict[str, str],
    taxa: List[str],
    fa_path: Optional[str] = None,
    threshold: float = 100.0
) -> np.ndarray:
    """
    Computes pairwise Tamura-Nei 93 (TN93) genetic distance matrix directly from sequences.
    Prefers the high-speed compiled 'tn93' C binary if available in PATH, falling back
    to the Python 'tn93' package (from tn93.tn93 import TN93).

    threshold: Distance cutoff for TN93 reporting. Set to a high value (default 100.0) so that
    no divergent pairwise distances are prematurely omitted from deep alignments. If the installed
    tn93 binary strictly bounds distance to [0, 1] (e.g. stock <= v1.0.15), it automatically retries
    with threshold=1.0.
    """
    n = len(taxa)
    dist_mat = np.full((n, n), -1.0, dtype=np.float32)
    np.fill_diagonal(dist_mat, 0.0)
    if n <= 1:
        return dist_mat

    taxa_idx = {t: i for i, t in enumerate(taxa)}
    tn93_bin = shutil.which("tn93")
    used_binary = False

    if tn93_bin:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_fa = os.path.join(tmpdir, "subset.fa")
            out_csv = os.path.join(tmpdir, "distances.csv")
            with open(tmp_fa, "w") as f:
                for t in taxa:
                    # Sanitize non-standard gap/missing characters (e.g. LANL MASE '*' -> '-')
                    s_clean = seq_dict[t].replace('*', '-')
                    f.write(f">{t}\n{s_clean}\n")
            
            # Call tn93 with higher threshold so divergent pairs are not omitted
            cmd = [tn93_bin, "-t", f"{threshold:.1f}", "-l", "1", "-q", "-o", out_csv, tmp_fa]
            binary_success = False
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                binary_success = True
            except subprocess.CalledProcessError:
                if threshold > 1.0:
                    # Stock tn93 binaries (<= v1.0.15) enforce distance threshold in [0, 1].
                    # Gracefully retry with threshold 1.0 (the maximum allowed by stock builds).
                    cmd_fallback = [tn93_bin, "-t", "1.0", "-l", "1", "-q", "-o", out_csv, tmp_fa]
                    try:
                        subprocess.run(cmd_fallback, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        binary_success = True
                    except Exception as e:
                        print(f"[!] Warning: tn93 binary execution failed ({e}); falling back to Python TN93.")
            except Exception as e:
                print(f"[!] Warning: tn93 binary execution failed ({e}); falling back to Python TN93.")

            if binary_success and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
                # Vectorized parse of the tn93 pairwise CSV (O(pairs) Python loop -> numpy scatter).
                import pandas as pd
                df = pd.read_csv(out_csv, usecols=["ID1", "ID2", "Distance"], dtype={"ID1": str, "ID2": str})
                ii = df["ID1"].map(taxa_idx).to_numpy(dtype="float64")
                jj = df["ID2"].map(taxa_idx).to_numpy(dtype="float64")
                dd = pd.to_numeric(df["Distance"], errors="coerce").to_numpy()
                valid = ~(np.isnan(ii) | np.isnan(jj) | np.isnan(dd))
                ii = ii[valid].astype(np.intp)
                jj = jj[valid].astype(np.intp)
                dd = dd[valid].astype(np.float32)
                dist_mat[ii, jj] = dd
                dist_mat[jj, ii] = dd
                used_binary = True

    if not used_binary:
        try:
            from tn93.tn93 import TN93
            tn = TN93()
            for i in range(n):
                for j in range(i + 1, n):
                    counts = tn.get_counts(seq_dict[taxa[i]], seq_dict[taxa[j]], "resolve")
                    nuc_freq = tn.get_nucleotide_frequency(counts)
                    try:
                        d = tn.calculate_distance(counts, nuc_freq)
                    except (ValueError, OverflowError):
                        # TN93 can throw math domain errors on very short or
                        # saturated sequences (e.g. log of a negative number);
                        # treat as a maximally distant pair.
                        d = 1.0
                    if d is None or d == "-" or d < 0 or np.isnan(d):
                        d = 1.0
                    d = float(d)
                    dist_mat[i, j] = d
                    dist_mat[j, i] = d
        except ImportError:
            raise ImportError(
                "The 'tn93' tool or python package is required to skip the tree and compute TN93 distances. "
                "Please install it via 'pip install tn93' or 'pip install hyphaeon[tn93]' "
                "(or install the tn93 binary from https://github.com/veg/tn93)."
            )

    # Impute missing off-diagonal distances (vectorized).
    # Identical sequences (distance == 0.0) are preserved.
    # Any negative entry (< 0.0) represents a divergent pair whose distance
    # exceeded the threshold or was omitted -> impute with max observed distance or 1.0.
    np.fill_diagonal(dist_mat, 0.0)
    missing = (dist_mat < 0.0)
    max_d = float(dist_mat.max()) if dist_mat.max() > 0 else 1.0
    dist_mat[missing] = max(1.0, max_d)
    np.fill_diagonal(dist_mat, 0.0)
    return dist_mat


def compute_tn93_cross_distance_matrix(
    seq_dict: Dict[str, str],
    taxa_all: List[str],
    taxa_landmarks: List[str],
    threshold: float = 100.0
) -> np.ndarray:
    """
    Computes rectangular (N x M) pairwise TN93 genetic distance matrix between all N taxa
    and M landmark taxa directly from sequences, enabling O(NM) landmark/Nystrom spectral
    deconvolution for ultra-large cohorts (N up to 50,000+) without O(N^2) memory footprint.
    """
    n = len(taxa_all)
    m = len(taxa_landmarks)
    dist_mat = np.full((n, m), -1.0, dtype=np.float32)
    if n == 0 or m == 0:
        return dist_mat

    taxa_idx = {t: i for i, t in enumerate(taxa_all)}
    lm_idx = {t: j for j, t in enumerate(taxa_landmarks)}
    tn93_bin = shutil.which("tn93")
    used_binary = False

    if tn93_bin:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_all = os.path.join(tmpdir, "all.fa")
            tmp_lm = os.path.join(tmpdir, "landmarks.fa")
            out_csv = os.path.join(tmpdir, "cross_distances.csv")

            with open(tmp_all, "w") as f:
                for t in taxa_all:
                    s_clean = seq_dict[t].replace('*', '-')
                    f.write(f">{t}\n{s_clean}\n")
            with open(tmp_lm, "w") as f:
                for t in taxa_landmarks:
                    s_clean = seq_dict[t].replace('*', '-')
                    f.write(f">{t}\n{s_clean}\n")

            cmd = [tn93_bin, "-s", tmp_lm, "-t", f"{threshold:.1f}", "-l", "1", "-q", "-o", out_csv, tmp_all]
            binary_success = False
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                binary_success = True
            except subprocess.CalledProcessError:
                if threshold > 1.0:
                    cmd_fallback = [tn93_bin, "-s", tmp_lm, "-t", "1.0", "-l", "1", "-q", "-o", out_csv, tmp_all]
                    try:
                        subprocess.run(cmd_fallback, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        binary_success = True
                    except Exception:
                        pass
            except Exception:
                pass

            if binary_success and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
                import pandas as pd
                chunksize = 2_000_000
                for chunk in pd.read_csv(out_csv, usecols=["ID1", "ID2", "Distance"], dtype={"ID1": str, "ID2": str}, chunksize=chunksize):
                    ii = chunk["ID1"].map(taxa_idx).to_numpy(dtype="float64")
                    jj = chunk["ID2"].map(lm_idx).to_numpy(dtype="float64")
                    dd = pd.to_numeric(chunk["Distance"], errors="coerce").to_numpy()
                    valid = ~(np.isnan(ii) | np.isnan(jj) | np.isnan(dd))
                    ii = ii[valid].astype(np.intp)
                    jj = jj[valid].astype(np.intp)
                    dd = dd[valid].astype(np.float32)
                    dist_mat[ii, jj] = dd
                used_binary = True

    if not used_binary:
        try:
            from tn93.tn93 import TN93
            tn = TN93()
            for j, lm in enumerate(taxa_landmarks):
                seq_lm = seq_dict[lm]
                for i, t in enumerate(taxa_all):
                    if t == lm:
                        dist_mat[i, j] = 0.0
                        continue
                    counts = tn.get_counts(seq_dict[t], seq_lm, "resolve")
                    nuc_freq = tn.get_nucleotide_frequency(counts)
                    try:
                        d = tn.calculate_distance(counts, nuc_freq)
                    except (ValueError, OverflowError):
                        d = 1.0
                    if d is None or d == "-" or d < 0 or np.isnan(d):
                        d = 1.0
                    dist_mat[i, j] = float(d)
        except ImportError:
            raise ImportError(
                "The 'tn93' tool or python package is required to compute TN93 distances."
            )

    # Fill self-distances
    for j, lm in enumerate(taxa_landmarks):
        if lm in taxa_idx:
            dist_mat[taxa_idx[lm], j] = 0.0

    # Impute missing entries
    missing = (dist_mat < 0.0)
    max_d = float(dist_mat.max()) if dist_mat.max() > 0 else 1.0
    dist_mat[missing] = max(1.0, max_d)
    for j, lm in enumerate(taxa_landmarks):
        if lm in taxa_idx:
            dist_mat[taxa_idx[lm], j] = 0.0

    return dist_mat


def load_alignment_and_tree(
    fa_path: str,
    nwk_path: Optional[str] = None,
    max_species: Optional[int] = None,
    prune_duplicates: bool = True,
    use_tn93: bool = False
):
    """
    Parses alignment (FASTA or NEXUS) and phylogenetic tree (from nwk_path or embedded in alignment).
    If use_tn93 is True (or nwk_path == "tn93"), skips the tree and estimates pairwise
    evolutionary distances directly from the alignment using TN93.
    Enforces non-zero branch lengths (estimating them via HyPhy if available and missing).
    Automatically prunes identical sequence duplicates and trims tree accordingly if prune_duplicates=True.
    Optionally applies greedy Faith's PD species downsampling if max_species is specified.
    Returns PyTorch tensors (c, a, d, z), invariable mask, taxa list, and codon length L.
    """
    # 1. Parse alignment sequences
    seq_dict = parse_alignment_sequences(fa_path)
    if not seq_dict:
        raise ValueError(f"Could not parse any sequences from alignment file: '{fa_path}'")

    if use_tn93 or (nwk_path is not None and str(nwk_path).strip().lower() in ("tn93", "none", "skip")):
        print("[*] Tree skipped: Estimating pairwise sequence distances directly via TN93...")
        taxa = list(seq_dict.keys())

        # Automated Duplicate Sequence Pruning
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
        _warn_internal_stops(seq_dict, L, taxa)

        # Fast pre-downsampling for massive sequence collections (N > 300)
        if max_species is not None and len(taxa) > max_species:
            stride = max(1, len(taxa) // (max_species * 2))
            taxa = taxa[::stride][:max_species * 2]

        dist_mat = compute_tn93_distance_matrix(seq_dict, taxa, fa_path=fa_path)

        if max_species is not None and len(taxa) > max_species:
            dist_mat, taxa = downsample_taxa_faith_pd(dist_mat, taxa, max_species)
            print(f"[*] Distance Diversity Downsampling: Selected {len(taxa)} taxa maximizing sequence diversity.")
    else:
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
                    f"Please provide a tree via -t / --tree, or use --no-tree / --use-tn93 to estimate distances via TN93."
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
        _warn_internal_stops(seq_dict, L, taxa)
        n_taxa = len(taxa)

        # 5. Fast pre-downsampling for massive sequence collections (N > 300)
        if max_species is not None and len(taxa) > max_species:
            # Uniformly stride downsample to 2 * max_species before computing distance matrix
            stride = max(1, len(taxa) // (max_species * 2))
            taxa = taxa[::stride][:max_species * 2]

        # Compute distance matrix
        dist_mat = compute_fast_dist_matrix(tree_obj, taxa)

        # If tree branch lengths are raw mutation counts (> 10.0) rather than substitutions per site,
        # normalize by alignment codon length L to bring distances into standard evolutionary scale.
        # This heuristic also rescales a legitimately long-branch tree (max patristic distance > 10
        # subs/site), so warn rather than rescaling silently.
        if dist_mat.max() > 10.0:
            warnings.warn(
                f"Max patristic distance {dist_mat.max():.3g} exceeds 10.0; assuming branch lengths "
                f"are raw mutation counts and rescaling by alignment length L={L}. If this tree is "
                f"genuinely long-branch (distances already in subs/site), this rescaling is incorrect.",
                UserWarning,
                stacklevel=2,
            )
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
