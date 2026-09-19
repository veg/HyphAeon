"""
chronaeon/dating_io.py
----------------------
Input validation, consensus generation, and timestamp parsing for
molecular clock dating.

Handles strict in-frame coding alignment validation (L_nt % 3 == 0,
triplet-gap check, stop codon audit), time-decay weighted consensus
sequence generation, and flexible timestamp ingestion from FASTA
headers, CSV/TSV metadata, Nextstrain Auspice JSON v2, and BEAST XML.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

from aeon_core.dataset import parse_beast_xml
from aeon_core.temporal import parse_date_to_decimal, parse_dates_from_auspice_json, extract_date_from_string



# =========================================================================
# 1. In-Frame Coding Alignment Validation
# =========================================================================

def verify_coding_alignment(
    seq_dict: Dict[str, str],
    allow_stop_codons: bool = True,
    auto_trim_trailing: bool = True
) -> Tuple[int, int]:
    """
    Validates that a nucleotide MSA is suitable for HyphAeon codon modeling.
    Requirements:
    - Non-empty alignment.
    - All sequences must possess identical aligned lengths.
    - Aligned sequence length must be a multiple of 3 (L_nt % 3 == 0). Auto-trims trailing nucleotides if auto_trim_trailing=True.
    - Checks for internal stop codons (TAA, TAG, TGA).

    Returns:
        (num_taxa, num_codons)
    Raises:
        ValueError: If sequence lengths differ, are non-coding, or violate codon constraints.
    """
    if not seq_dict:
        raise ValueError("Provided alignment is empty.")

    taxa = list(seq_dict.keys())
    first_taxon = taxa[0]
    first_seq = seq_dict[first_taxon].upper().replace('U', 'T')
    l_nt = len(first_seq)

    if l_nt == 0:
        raise ValueError(f"Sequence '{first_taxon}' has length 0.")

    if l_nt % 3 != 0:
        rem = l_nt % 3
        if auto_trim_trailing:
            print(f"[!] Notice: Alignment length ({l_nt} nt) is not divisible by 3. Auto-trimming {rem} trailing nucleotide(s).")
            for t in taxa:
                seq_dict[t] = seq_dict[t][:-rem]
            l_nt = len(seq_dict[first_taxon])
        else:
            raise ValueError(
                f"HyphAeon is a codon-level foundation model and strictly requires in-frame coding sequences. "
                f"Sequence '{first_taxon}' has length {l_nt} nt ({rem} remainder modulo 3). "
                f"Please verify open reading frames and remove non-coding flanking regions or frameshift indels."
            )

    num_codons = l_nt // 3

    # Check length uniformity across all taxa
    mismatched = []
    for t in taxa:
        seq_len = len(seq_dict[t])
        if seq_len != l_nt:
            mismatched.append((t, seq_len))
            if len(mismatched) >= 5:
                break

    if mismatched:
        details = ", ".join([f"'{t}': {l} nt" for t, l in mismatched])
        raise ValueError(
            f"Alignment sequences are not uniformly aligned to length {l_nt} nt. "
            f"Mismatched examples: {details}."
        )

    # Stop codon inspection
    stop_codons = {'TAA', 'TAG', 'TGA'}
    taxa_with_stops = []
    total_stops = 0

    for t, s in seq_dict.items():
        s_clean = s.upper().replace('U', 'T')
        for c_idx in range(num_codons - 1):
            codon = s_clean[c_idx * 3:(c_idx + 1) * 3]
            if codon in stop_codons:
                total_stops += 1
                if t not in taxa_with_stops:
                    taxa_with_stops.append(t)

    if taxa_with_stops:
        pct_affected = 100.0 * len(taxa_with_stops) / len(taxa)
        msg = (
            f"[!] Notice: Detected {total_stops} internal stop codon(s) across "
            f"{len(taxa_with_stops)}/{len(taxa)} taxa ({pct_affected:.1f}%). "
            f"HyphAeon automatically tokenizes stop codons to token 64 ('*')."
        )
        if not allow_stop_codons:
            raise ValueError(f"{msg} Set allow_stop_codons=True to proceed anyway.")
        else:
            print(msg)

    return len(taxa), num_codons


def generate_consensus_sequence(seq_dict: Dict[str, str], taxa: Optional[List[str]] = None) -> str:
    """Computes the majority-rule nucleotide consensus sequence across specified taxa."""
    if taxa is None:
        taxa = list(seq_dict.keys())
    if not taxa:
        raise ValueError("Cannot compute consensus of 0 sequences.")

    seq_len = len(seq_dict[taxa[0]])
    consensus_chars = []
    for pos in range(seq_len):
        counts: Dict[str, int] = {}
        for t in taxa:
            char = seq_dict[t][pos].upper()
            if char not in ['-', '?', 'N']:
                counts[char] = counts.get(char, 0) + 1
        if counts:
            best_char = max(counts.items(), key=lambda x: x[1])[0]
        else:
            best_char = '-'
        consensus_chars.append(best_char)
    return "".join(consensus_chars)


def generate_time_decay_consensus_sequence(
    seq_dict: Dict[str, str],
    dates_map: Dict[str, float],
    taxa: Optional[List[str]] = None,
    gamma: Optional[float] = None,
    half_life: Optional[float] = None,
) -> Tuple[str, float]:
    r"""
    Computes a time-decay weighted nucleotide consensus sequence:
        w_i \propto \exp(-\gamma * (t_i - t_min))

    Downweights modern, densely-sampled contemporary isolates and upweights ancestral/early isolates,
    producing an optimal tree-free reference root anchor resistant to temporal sampling bias.

    Returns:
        (consensus_sequence, effective_gamma)
    """
    if taxa is None:
        taxa = list(seq_dict.keys())

    valid_taxa = [t for t in taxa if t in dates_map and not np.isnan(dates_map[t])]
    if not valid_taxa:
        # Fallback to unweighted consensus if dates are unavailable
        return generate_consensus_sequence(seq_dict, taxa), 0.0

    times = np.array([dates_map[t] for t in valid_taxa], dtype=np.float64)
    t_min = float(np.min(times))
    t_max = float(np.max(times))
    delta_t = t_max - t_min

    if half_life is not None and half_life > 0:
        eff_gamma = float(np.log(2.0) / half_life)
    elif gamma is not None and gamma > 0:
        eff_gamma = float(gamma)
    else:
        # Adaptive default: if span > 0, set gamma = 0.05 or adapt
        if delta_t > 0:
            eff_gamma = 0.05 if (0.05 * delta_t >= 1.0) else float(2.0 / delta_t)
        else:
            eff_gamma = 0.0

    weights = np.exp(-eff_gamma * (times - t_min))
    w_sum = np.sum(weights)
    if w_sum > 0:
        weights /= w_sum
    else:
        weights = np.ones(len(valid_taxa)) / len(valid_taxa)

    seq_len = len(seq_dict[valid_taxa[0]])
    consensus_chars = []
    for pos in range(seq_len):
        char_weights: Dict[str, float] = {}
        for i, t in enumerate(valid_taxa):
            c = seq_dict[t][pos].upper()
            if c not in ['-', '?', 'N']:
                char_weights[c] = char_weights.get(c, 0.0) + float(weights[i])
        if char_weights:
            best_c = max(char_weights.items(), key=lambda x: x[1])[0]
        else:
            best_c = '-'
        consensus_chars.append(best_c)

    return "".join(consensus_chars), eff_gamma


def compute_time_decay_profile_divergences(
    seq_dict: Dict[str, str],
    dates_map: Dict[str, float],
    dated_taxa: List[str],
    gamma: Optional[float] = None,
    half_life: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    r"""
    Computes tree-free root-to-tip divergence using a continuous time-decay nucleotide profile
    rather than a forced discrete consensus sequence.

    At each site k = 1..L, the ancestral root state is represented as a probability simplex:
        q_k(b) = \sum_{i=1}^N w_i * \mathbb{I}(S_{ik} = b)
    where w_i \propto \exp(-\gamma * (t_i - t_min)).

    Divergences are evaluated as expected substitutions under Tamura-Nei 93 (accounting for
    purine transitions P1, pyrimidine transitions P2, and transversions Q) or Hamming expectation.
    Eliminates artificial threshold flips, preserving continuity at polymorphic sites.
    """
    valid_taxa = [t for t in dated_taxa if t in dates_map and not np.isnan(dates_map[t])]
    if not valid_taxa:
        return np.zeros(len(dated_taxa), dtype=np.float64), 0.0

    times = np.array([dates_map[t] for t in valid_taxa], dtype=np.float64)
    t_min, t_max = float(np.min(times)), float(np.max(times))
    delta_t = t_max - t_min

    char_map = {'A': 0, 'a': 0, 'C': 1, 'c': 1, 'G': 2, 'g': 2, 'T': 3, 't': 3}
    N = len(valid_taxa)
    L = len(seq_dict[valid_taxa[0]])
    M = np.full((N, L), -1, dtype=np.int8)
    for i, t in enumerate(valid_taxa):
        seq = seq_dict[t]
        for k in range(min(L, len(seq))):
            M[i, k] = char_map.get(seq[k], -1)

    is_A = (M == 0)
    is_C = (M == 1)
    is_G = (M == 2)
    is_T = (M == 3)
    valid_mask = (M >= 0)
    L_valid = np.maximum(np.sum(valid_mask, axis=1), 1)

    def _eval_gamma(g: float) -> Tuple[np.ndarray, float]:
        w = np.exp(-g * (times - t_min))
        w_sum = np.sum(w)
        w = w / w_sum if w_sum > 0 else np.ones(N) / N

        Q = np.zeros((L, 4), dtype=np.float64)
        for b in range(4):
            Q[:, b] = np.sum((M == b) * w[:, None], axis=0)

        q_sum = np.sum(Q, axis=1, keepdims=True)
        zero_pos = (q_sum.squeeze() <= 0)
        Q[~zero_pos, :] /= q_sum[~zero_pos]
        Q[zero_pos, :] = 0.25

        P1 = np.sum(is_A * Q[None, :, 2] + is_G * Q[None, :, 0], axis=1) / L_valid
        P2 = np.sum(is_C * Q[None, :, 3] + is_T * Q[None, :, 1], axis=1) / L_valid
        Q_tv = np.sum((is_A | is_G) * (Q[None, :, 1] + Q[None, :, 3]) + (is_C | is_T) * (Q[None, :, 0] + Q[None, :, 2]), axis=1) / L_valid

        mean_Q = np.mean(Q, axis=0)
        f_A = np.maximum(0.5 * (mean_Q[0] + np.sum(is_A, axis=1) / L_valid), 1e-4)
        f_C = np.maximum(0.5 * (mean_Q[1] + np.sum(is_C, axis=1) / L_valid), 1e-4)
        f_G = np.maximum(0.5 * (mean_Q[2] + np.sum(is_G, axis=1) / L_valid), 1e-4)
        f_T = np.maximum(0.5 * (mean_Q[3] + np.sum(is_T, axis=1) / L_valid), 1e-4)
        f_R = f_A + f_G
        f_Y = f_C + f_T

        arg1 = 1.0 - (f_R / (2.0 * f_A * f_G)) * P1 - (Q_tv / (2.0 * f_R))
        arg2 = 1.0 - (f_Y / (2.0 * f_C * f_T)) * P2 - (Q_tv / (2.0 * f_Y))
        arg3 = 1.0 - (Q_tv / (2.0 * f_R * f_Y))

        k1 = 2.0 * f_A * f_G / f_R
        k2 = 2.0 * f_C * f_T / f_Y
        k3 = 2.0 * (f_R * f_Y - (f_A * f_G * f_Y / f_R) - (f_C * f_T * f_R / f_Y))

        p_ham = P1 + P2 + Q_tv
        valid_log = (arg1 > 0) & (arg2 > 0) & (arg3 > 0)
        d = np.zeros(N, dtype=np.float64)
        d[~valid_log] = p_ham[~valid_log]
        d[valid_log] = (
            -k1[valid_log] * np.log(arg1[valid_log])
            - k2[valid_log] * np.log(arg2[valid_log])
            - k3[valid_log] * np.log(arg3[valid_log])
        )
        invalid_d = (~np.isfinite(d)) | (d < 0)
        d[invalid_d] = p_ham[invalid_d]
        return d, g

    if half_life is not None and half_life > 0:
        eff_gamma = float(np.log(2.0) / half_life)
        dists, _ = _eval_gamma(eff_gamma)
    elif gamma is not None and gamma >= 0:
        eff_gamma = float(gamma)
        dists, _ = _eval_gamma(eff_gamma)
    elif delta_t > 0 and N >= 4:
        # Principle: Optimize gamma to maximize temporal molecular clock correlation (argmax R^2)
        # analogous to TempEst root-to-tip heuristic optimization, but in continuous profile space.
        cand_gammas = [
            0.5 / delta_t, 1.0 / delta_t, 2.0 / delta_t,
            5.0 / delta_t, 10.0 / delta_t, 20.0 / delta_t
        ]
        best_r = -1e9
        best_d = None
        best_g = cand_gammas[2]
        v_t = np.var(times)

        for g_cand in cand_gammas:
            d_c, _ = _eval_gamma(g_cand)
            v_d = np.var(d_c)
            if v_t > 0 and v_d > 0:
                r = float(np.cov(times, d_c)[0, 1] / np.sqrt(v_t * v_d))
            else:
                r = -1.0
            if r > best_r:
                best_r = r
                best_g = g_cand
                best_d = d_c

        dists = best_d if best_d is not None else _eval_gamma(best_g)[0]
        eff_gamma = best_g
    else:
        eff_gamma = float(2.0 / delta_t) if delta_t > 0 else 0.0
        dists, _ = _eval_gamma(eff_gamma)

    taxa_idx_map = {t: i for i, t in enumerate(valid_taxa)}
    full_dists = np.zeros(len(dated_taxa), dtype=np.float64)
    for j, t in enumerate(dated_taxa):
        if t in taxa_idx_map:
            full_dists[j] = dists[taxa_idx_map[t]]

    return full_dists, eff_gamma


# =========================================================================
# 2. Timestamp Extraction & Parsing
# =========================================================================

def parse_header_timestamp(name: str) -> float:
    """
    Parses timestamps from FASTA sequence headers across common phylogenetic formats:
    - ISO full date: `2021-04-15` or `2021/04/15`
    - Decimal year: `2021.25`, `1959.5`
    - Year-month: `2021-04`
    - Delimiter prefixed: `|2021-04-15`, `_1986.5`, `/1993.5`
    - Bette Korber / HIV LANL format: e.g. `B86US.SFMHS18` -> 1986.5, `Z59ZR.ZHU` -> 1959.5
    - Longitudinal intra-host WPI: e.g. `16WPI` -> 16.0
    """
    if not name:
        return np.nan

    # 1. Check for archival 1959 ZR59 anchor
    if 'Z59' in name or 'ZR59' in name or '1959' in name:
        return 1959.5

    # 2. Try standard ISO/decimal extraction from temporal module
    d = extract_date_from_string(name)
    if not np.isnan(d):
        return d

    # 3. Korber HIV-1 isolate pattern: [Subtype][2-digit year][Country].[Strain]
    # Examples: B86US.SFMHS18, F93BE_VI850, C86ET.ETH2220, A85UG.U455, D84ZR.84ZR085
    m_korber = re.search(r'^[A-Za-z](\d{2})[A-Za-z]{2}[._]', name)
    if m_korber:
        yr_short = int(m_korber.group(1))
        full_yr = 1900 + yr_short if yr_short >= 30 else 2000 + yr_short
        return float(full_yr) + 0.5

    # 4. Nextstrain / LANL pipe format with year: Ref_B_FR_83_HXB2 or Ref_C_ET_86_ETH2220
    m_pipe = re.search(r'_(?:[A-Z]{2})_(\d{2})_', name)
    if m_pipe:
        yr_short = int(m_pipe.group(1))
        full_yr = 1900 + yr_short if yr_short >= 30 else 2000 + yr_short
        return float(full_yr) + 0.5

    # 5. Longitudinal intra-host WPI (Weeks Post Infection)
    m_wpi = re.search(r'(\d+(?:\.\d+)?)\s*(?:WPI|wpi)', name)
    if m_wpi:
        return float(m_wpi.group(1))

    # 6. Longitudinal DPI (Days Post Infection)
    m_dpi = re.search(r'(\d+(?:\.\d+)?)\s*(?:DPI|dpi)', name)
    if m_dpi:
        return float(m_dpi.group(1))

    return np.nan


def _parse_timestamp_flexible(val: Any) -> float:
    """
    Parses calendar dates (e.g. 2021.25, 1985-06-15) via parse_date_to_decimal,
    falling back to arbitrary non-calendar numeric values (e.g. 0.25 years, days, months)
    for longitudinal intra-host or experimental time coordinates.
    """
    if val is None or pd.isna(val):
        return np.nan
    d = parse_date_to_decimal(val)
    if not np.isnan(d):
        return d
    try:
        val_f = float(val)
        if not np.isnan(val_f):
            return val_f
    except (ValueError, TypeError):
        pass
    return np.nan


def parse_sample_dates(
    taxa: List[str],
    dates_source: Optional[Union[str, Path, Dict[str, float]]] = None,
    date_col: Optional[str] = None,
    strain_col: Optional[str] = None,
    date_regex: Optional[str] = None
) -> Tuple[Dict[str, float], List[str]]:
    """
    Ingests and parses sampling dates for a list of taxa.

    Supports:
    - Pre-parsed dictionary {taxon: date}
    - Nextstrain Auspice JSON v2
    - CSV or TSV metadata table
    - FASTA header auto-extraction
    """
    dates_map: Dict[str, float] = {}

    # Case 1: Direct dictionary
    if isinstance(dates_source, dict):
        for t in taxa:
            if t in dates_source:
                val = _parse_timestamp_flexible(dates_source[t])
                if not np.isnan(val):
                    dates_map[t] = val

    # Case 2: External file (JSON, CSV, TSV)
    elif dates_source is not None:
        source_path = Path(dates_source)
        if not source_path.exists():
            raise FileNotFoundError(f"Dates source file not found: {source_path}")

        if source_path.suffix.lower() == '.json':
            with open(source_path, 'r', encoding='utf-8') as f:
                raw_json = json.load(f)
            if 'tree' in raw_json:
                auspice_dates = parse_dates_from_auspice_json(source_path)
                dates_map.update(auspice_dates)
            else:
                for k, v in raw_json.items():
                    val = _parse_timestamp_flexible(v)
                    if not np.isnan(val):
                        dates_map[k] = val
                    elif isinstance(v, dict) and 'year' in v:
                        dates_map[k] = _parse_timestamp_flexible(v['year'])

        elif source_path.suffix.lower() in ['.xml', '.xml.gz']:
            beast_res = parse_beast_xml(source_path)
            for t in taxa:
                t_clean = t.strip("'\"")
                if t_clean in beast_res['dates']:
                    dates_map[t] = beast_res['dates'][t_clean]
                elif t_clean.startswith('seq_') and t_clean[4:] in beast_res['dates']:
                    dates_map[t] = beast_res['dates'][t_clean[4:]]
                elif f"seq_{t_clean}" in beast_res['dates']:
                    dates_map[t] = beast_res['dates'][f"seq_{t_clean}"]

        elif source_path.suffix.lower() in ['.csv', '.tsv', '.txt']:
            sep = '\t' if source_path.suffix.lower() in ['.tsv', '.txt'] else ','
            df = pd.read_csv(source_path, sep=sep, dtype=str)

            if not strain_col:
                cand_strains = ['strain', 'taxon', 'taxa', 'name', 'id', 'genome_id', 'seq_id', 'accession', 'sequence']
                for c in df.columns:
                    if c.lower() in cand_strains:
                        strain_col = c
                        break
                if not strain_col:
                    strain_col = df.columns[0]

            if not date_col:
                cand_dates = ['date', 'year', 'time', 'num_date', 'decimal_date', 'collection_date', 'sampling_date']
                for c in df.columns:
                    if c.lower() in cand_dates:
                        date_col = c
                        break
                if not date_col:
                    date_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

            for _, row in df.iterrows():
                strain_val = str(row[strain_col]).strip()
                date_val = _parse_timestamp_flexible(row[date_col])
                if not np.isnan(date_val):
                    dates_map[strain_val] = date_val

    # Case 3: Fallback to sequence header extraction
    for t in taxa:
        if t not in dates_map or np.isnan(dates_map[t]):
            if date_regex:
                m = re.search(date_regex, t)
                if m:
                    extracted = _parse_timestamp_flexible(m.group(1))
                    if not np.isnan(extracted):
                        dates_map[t] = extracted
            if t not in dates_map or np.isnan(dates_map[t]):
                val = parse_header_timestamp(t)
                if not np.isnan(val):
                    dates_map[t] = val

    missing_taxa = [t for t in taxa if t not in dates_map or np.isnan(dates_map[t])]
    return dates_map, missing_taxa

