"""
hyphaeon/temporal.py
--------------------
Continuous Temporal Attribution Regression, Two-Stage Statistical Filtering,
and Functional Dynamic Wave Decomposition for Pathogen Genomic Surveillance.

Key Components:
1. Flexible Ingestion of Time-Stamped Data:
   - Nextstrain Auspice JSON v2 (hierarchical tree node attributes)
   - Tabular metadata (TSV, CSV, Excel with auto-discovered strain and date columns)
   - In-line FASTA header timestamp extraction (ISO dates or decimal years)
2. Ancestral Root Anchoring:
   - Data-driven ancestral founder consensus (earliest sampled taxa)
   - Explicit user-specified reference taxon or alignment-wide consensus
3. Neural Single-Taxon Attribution Regression:
   - Directional leaf attention gating: a_{s, n} = alpha_{root->n, s} * I{x_{s, n} != x_{s, root}}
   - Continuous Nadaraya-Watson Gaussian kernel smoothing: a_s(t)
4. Two-Stage Statistical Filtering:
   - Stage 1: Energy floor (peak intensity and cumulative AUC thresholds)
   - Stage 2: Date-shuffling permutation significance (B=1,000) and functional alignment (R^2_fPCA >= 0.35)
   - Cross-classification: Confirmed sweeps, Rescued sweeps, Concordant sweeps, Filtered static noise
5. Functional Dynamic Factor Decomposition (fPCA):
   - Singular value decomposition across standardized continuous trajectories
   - 4 collective dynamic waves capturing epochal lineage replacements
   - Full site factor loadings
6. Publication Visualization:
   - Automatic 4-panel publication-grade PDF/PNG figure generation
"""

import os
import sys
import json
import time
import re
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

from .epistasis import REV_AA_MAP, AA_MAP
from .inference import (
    load_model,
    get_device,
    prepare_alignment,
    compute_adaptive_safe_batch_size,
)
from .stats import pvals_from_lrt_self_liang, benjamini_hochberg
from .io import ensure_parent_directory, write_json, write_csv

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# NumPy 1.x vs 2.x compatibility: np.trapz was removed in 2.0 in favor of np.trapezoid
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


# =========================================================================
# 1. Flexible Date Parsing and Ingestion
# =========================================================================

def parse_date_to_decimal(val: Any, time_units: str = "years") -> float:
    """
    Converts various date representations into a float time coordinate.
    Calendar (time_units='years'):
      - float or int: 2021.25 -> 2021.25
      - ISO string: "2021-04-15" -> 2021.2868
      - Partial ISO: "2021-04" or "2021-04-XX" -> 2021.2868 (mid-month)
      - Year only: "2021" or "2021-XX-XX" -> 2021.5 (mid-year)
      - Slash formatted: "2021/04/15" -> 2021.2868
    Non-calendar (time_units in {'generations','days','arbitrary'}):
      - any non-negative number, or the first number embedded in a string
        ("gen_5000" -> 5000.0). No [1800,2100] gate.
    """
    if val is None or pd.isna(val):
        return np.nan

    # Non-calendar time coordinates: accept any non-negative real (no year gate)
    if time_units in ("generations", "days", "arbitrary"):
        try:
            val_f = float(val)
            return val_f if val_f >= 0.0 else np.nan
        except (ValueError, TypeError):
            pass
        m = re.search(r'(\d+(?:\.\d+)?)', str(val))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return np.nan
        return np.nan

    if isinstance(val, (int, float)):
        val_f = float(val)
        if 1800.0 <= val_f <= 2100.0:
            return val_f
        return np.nan

    val_str = str(val).strip()
    if not val_str or val_str.lower() in ['unknown', 'nan', 'none', 'na', '?']:
        return np.nan

    # Try direct float conversion (e.g., "2021.25")
    try:
        val_f = float(val_str)
        if 1800.0 <= val_f <= 2100.0:
            return val_f
    except ValueError:
        pass

    # Normalize delimiters
    clean_str = val_str.replace('/', '-').replace('.', '-')
    parts = clean_str.split('-')

    if len(parts) >= 1 and parts[0].isdigit() and len(parts[0]) == 4:
        try:
            year = int(parts[0])
            if not (1800 <= year <= 2100):
                return np.nan

            month = 6
            if len(parts) >= 2 and parts[1].isdigit():
                m = int(parts[1])
                if 1 <= m <= 12:
                    month = m

            day = 15
            if len(parts) >= 3 and parts[2].isdigit():
                d = int(parts[2])
                if 1 <= d <= 31:
                    day = d

            dt = datetime.date(year, month, min(day, 28 if month == 2 else 30))
            start_of_year = datetime.date(year, 1, 1)
            days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
            return float(year + (dt - start_of_year).days / days_in_year)
        except Exception:
            return np.nan

    return np.nan


def parse_dates_from_auspice_json(json_path: Union[str, Path]) -> Dict[str, float]:
    """Recursively extracts tip node dates from Nextstrain Auspice JSON v2."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tree_root = data.get('tree', data)
    dates: Dict[str, float] = {}

    def recurse(node: Dict[str, Any]):
        children = node.get('children', [])
        if not children:
            name = node.get('name')
            if not name:
                return
            attrs = node.get('node_attrs', {})
            val = np.nan

            if 'num_date' in attrs and isinstance(attrs['num_date'], dict) and 'value' in attrs['num_date']:
                val = parse_date_to_decimal(attrs['num_date']['value'])
            elif 'date' in attrs and isinstance(attrs['date'], dict) and 'value' in attrs['date']:
                val = parse_date_to_decimal(attrs['date']['value'])
            elif 'year' in attrs and isinstance(attrs['year'], dict) and 'value' in attrs['year']:
                val = parse_date_to_decimal(attrs['year']['value'])
            elif 'num_date' in attrs and isinstance(attrs['num_date'], (int, float, str)):
                val = parse_date_to_decimal(attrs['num_date'])
            elif 'date' in attrs and isinstance(attrs['date'], (int, float, str)):
                val = parse_date_to_decimal(attrs['date'])

            if np.isnan(val):
                # Try parsing timestamp from the tip name itself
                val = extract_date_from_string(name)

            if not np.isnan(val):
                dates[name] = val
        else:
            for child in children:
                recurse(child)

    recurse(tree_root)
    return dates


def extract_date_from_string(name: str, time_units: str = "years") -> float:
    """Extracts a time coordinate from a string / FASTA header using standard patterns."""
    if not name:
        return np.nan

    # Non-calendar: match embedded generation/day tokens e.g. _gen2000, |gen_5000,
    # _20000gen, _g50000, or a bare trailing number after a delimiter (|5000).
    # Prioritize explicit unit prefix/suffix before matching bare delimiter-bound numbers.
    if time_units in ("generations", "days", "arbitrary"):
        m = re.search(r'(?:[\|/_\-\s]|^)(?:gen|generation|g|day|d|t)[\-_]?(\d+(?:\.\d+)?)(?:[\|/_\-\s]|$)', name, re.IGNORECASE)
        if not m:
            m = re.search(r'(?:[\|/_\-\s]|^)(\d+(?:\.\d+)?)(?:gen|g|d)(?:[\|/_\-\s]|$)', name, re.IGNORECASE)
        if not m:
            m = re.search(r'(?:[\|/_\-\s]|^)(\d+(?:\.\d+)?)(?:[\|/_\-\s]|$)', name)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return np.nan

    # Pattern 1: ISO full date e.g. |2021-05-14 or /2021-05-14 or _2021-05-14
    m = re.search(r'(?:[\|/_\s]|^)(\d{4}-\d{2}-\d{2})(?:[\|/_\s]|$)', name)
    if m:
        d = parse_date_to_decimal(m.group(1))
        if not np.isnan(d):
            return d

    # Pattern 2: Decimal year e.g. |2021.35 or /2021.35
    m = re.search(r'(?:[\|/_\s]|^)(\d{4}\.\d{2,4})(?:[\|/_\s]|$)', name)
    if m:
        d = parse_date_to_decimal(m.group(1))
        if not np.isnan(d):
            return d

    # Pattern 3: Year-month e.g. |2021-05
    m = re.search(r'(?:[\|/_\s]|^)(\d{4}-\d{2})(?:[\|/_\s]|$)', name)
    if m:
        d = parse_date_to_decimal(m.group(1))
        if not np.isnan(d):
            return d

    # Pattern 4: Trailing year e.g. /2021 or |2021
    m = re.search(r'(?:[\|/_\s])(\d{4})$', name)
    if m:
        d = parse_date_to_decimal(m.group(1))
        if not np.isnan(d):
            return d

    return np.nan


def parse_temporal_metadata(
    dates_source: Optional[Union[str, Path, Dict[str, Any]]] = None,
    taxa: Optional[List[str]] = None,
    date_col: Optional[str] = None,
    strain_col: Optional[str] = None,
    time_units: str = "years",
) -> Dict[str, float]:
    """
    Universal date ingestion dispatcher:
    1. If dict: converts values to decimal years.
    2. If JSON: parses as Nextstrain Auspice JSON.
    3. If TSV/CSV: parses metadata table with intelligent column discovery.
    4. If None: extracts dates from taxa header strings.
    """
    if isinstance(dates_source, dict):
        return {k: parse_date_to_decimal(v, time_units) for k, v in dates_source.items() if not np.isnan(parse_date_to_decimal(v, time_units))}

    dates: Dict[str, float] = {}

    if dates_source is not None and os.path.exists(str(dates_source)):
        src_str = str(dates_source)
        if src_str.endswith('.json'):
            dates = parse_dates_from_auspice_json(src_str)
        else:
            # Parse tabular metadata (CSV, TSV, tab-delimited)
            try:
                sep = '\t' if src_str.endswith('.tsv') else None
                df_meta = pd.read_csv(src_str, sep=sep, engine='python')
            except Exception:
                df_meta = pd.read_csv(src_str)

            # Auto-detect strain column
            col_strain = strain_col
            if col_strain is None or col_strain not in df_meta.columns:
                cand_strains = [
                    'strain', 'taxon', 'taxa', 'name', 'id', 'accession',
                    'sequence_id', 'isolate', 'Isolate', 'Strain', 'Sequence ID'
                ]
                for c in cand_strains:
                    if c in df_meta.columns:
                        col_strain = c
                        break
                if col_strain is None:
                    col_strain = df_meta.columns[0]

            # Auto-detect date column
            col_date = date_col
            if col_date is None or col_date not in df_meta.columns:
                cand_dates = [
                    'generation', 'generations', 'gen', 'timepoint', 'time',
                    'day', 'days', 'transfer',
                    'date', 'num_date', 'collection_date', 'Collection Date',
                    'Date', 'submission_date', 'year', 'Collection_Date'
                ]
                for c in cand_dates:
                    if c in df_meta.columns:
                        col_date = c
                        break
                if col_date is None:
                    # Look for column with 'date' in name
                    for c in df_meta.columns:
                        if 'date' in c.lower():
                            col_date = c
                            break

            if col_date is not None and col_date in df_meta.columns:
                for _, row in df_meta.iterrows():
                    s_name = str(row[col_strain]).strip()
                    d_val = parse_date_to_decimal(row[col_date], time_units)
                    if not np.isnan(d_val):
                        dates[s_name] = d_val

    # If dates dictionary is empty or partial, fallback to header inspection for taxa
    if taxa is not None:
        missing_taxa = [t for t in taxa if t not in dates]
        if len(missing_taxa) > 0 and len(dates) == 0:
            # Attempt to extract from header strings
            for t in taxa:
                d = extract_date_from_string(t, time_units)
                if not np.isnan(d):
                    dates[t] = d

    return dates


# =========================================================================
# 2. Ancestral Root Determination
# =========================================================================

def infer_root_sequence(
    a_tensor: torch.Tensor,
    taxa: List[str],
    taxa_dates: Optional[np.ndarray] = None,
    valid_taxa_mask: Optional[np.ndarray] = None,
    root_taxon: Optional[str] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Determines the ancestral root state for each codon position to ensure
    the forward arrow of evolutionary time:
    - If root_taxon specified and present in taxa: uses root_taxon.
    - If taxa_dates provided: takes consensus of earliest 5% (min 3) sampled taxa.
    - Otherwise: alignment-wide modal consensus.
    """
    L = a_tensor.shape[0]
    a_valid = a_tensor.squeeze(-1).numpy()
    if valid_taxa_mask is not None:
        a_valid = a_valid[:, valid_taxa_mask]

    N = a_valid.shape[1]

    if root_taxon is not None and root_taxon in taxa:
        taxa_arr = np.array(taxa)
        if valid_taxa_mask is not None:
            taxa_arr = taxa_arr[valid_taxa_mask]
        idx = np.where(taxa_arr == root_taxon)[0]
        if len(idx) > 0:
            target_col = a_valid[:, idx[0]]
            root_indices = np.array([int(c) if c < 20 else 0 for c in target_col], dtype=np.int64)
            root_aas = [REV_AA_MAP.get(idx, 'X') for idx in root_indices]
            return root_aas, root_indices

    # Early sample consensus
    if taxa_dates is not None and len(taxa_dates) > 0:
        sort_idx = np.argsort(taxa_dates)
        n_early = max(3, min(25, int(0.05 * len(taxa_dates))))
        early_indices = sort_idx[:n_early]
        a_early = a_valid[:, early_indices]
    else:
        a_early = a_valid

    root_aas = []
    root_indices = np.zeros(L, dtype=np.int64)
    for s in range(L):
        vals = [v for v in a_early[s] if v < 20]
        if len(vals) == 0:
            vals = [v for v in a_valid[s] if v < 20]
        if len(vals) == 0:
            root_idx = 0
        else:
            counts = np.bincount(vals, minlength=20)
            root_idx = int(np.argmax(counts))
        root_indices[s] = root_idx
        root_aas.append(REV_AA_MAP.get(root_idx, '-'))

    return root_aas, root_indices


# =========================================================================
# 3. Main Temporal Surveillance Engine
# =========================================================================

def run_temporal_surveillance(
    alignment_path: str,
    tree_path: Optional[str] = None,
    dates_source: Optional[Union[str, Path, Dict[str, Any]]] = None,
    output_prefix: Optional[str] = None,
    bandwidth: Optional[float] = None,
    num_time_points: int = 250,
    n_permutations: int = 1000,
    perm_alpha: float = 0.05,
    min_r2_fpca: float = 0.35,
    tau_peak: float = 1e-4,
    tau_auc: Optional[float] = None,
    date_col: Optional[str] = None,
    strain_col: Optional[str] = None,
    root_taxon: Optional[str] = None,
    weights_path: Optional[str] = None,
    variant: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_species: Optional[int] = None,
    cpu: bool = False,
    use_tn93: bool = False,
    plot: bool = False,
    domain_map: Optional[Dict[int, str]] = None,
    time_units: str = "years",
    sweep_mode: str = "auto",
    keep_duplicates: bool = False,
) -> Dict[str, Any]:
    """
    Executes continuous model attribution regression, two-stage statistical filtering,
    dynamic wave decomposition, and table/figure export for surveillance datasets.
    """
    device = get_device(cpu=cpu)
    t_start = time.time()

    # Resolve experimental-evolution vs calendar behavior
    non_calendar = time_units in ("generations", "days", "arbitrary")
    if sweep_mode == "auto":
        sweep_mode = "fixation" if non_calendar else "episodic"
    # Identical longitudinal clones carry the temporal signal in fixation regimes,
    # so preserve duplicates unless explicitly running the calendar/episodic path.
    prune_dups = not (keep_duplicates or non_calendar)
    unit_label = {"years": "yrs", "generations": "gen", "days": "days", "arbitrary": "units"}.get(time_units, "units")

    print("\n" + "=" * 85)
    print(f"[*] HyphAeon Temporal Surveillance Engine")
    print(f"    Hardware Acceleration: {device.type.upper()}")
    print(f"    Alignment:             {alignment_path}")
    print(f"    Phylogenetic Tree:     {tree_path if tree_path else ('TN93' if use_tn93 else 'Embedded/TN93')}")
    print("=" * 85)

    # 1. Load Model
    model = load_model(weights=weights_path, variant=variant, device=device)
    model.eval()

    # 2. Alignment and Tree Preparation
    c, a, d, z, inv, taxa, L, tree_cache = prepare_alignment(
        alignment_path, tree_path, model=model, device=device,
        max_species=max_species, prune_duplicates=prune_dups, use_tn93=use_tn93
    )
    N_taxa = len(taxa)
    n_inv = int(np.sum(inv))
    n_var = L - n_inv
    print(f"[✓] Alignment loaded: {N_taxa} taxa, {L} codon sites ({n_var} variable, {n_inv} invariable)")

    # 3. Ingest Temporal Information
    date_dict = parse_temporal_metadata(
        dates_source=dates_source, taxa=taxa, date_col=date_col, strain_col=strain_col,
        time_units=time_units
    )
    taxa_dates = np.array([date_dict.get(t, np.nan) for t in taxa], dtype=np.float32)
    valid_taxa_mask = ~np.isnan(taxa_dates)
    valid_taxa_indices = np.where(valid_taxa_mask)[0]
    taxa_dates = taxa_dates[valid_taxa_mask]
    N = len(taxa_dates)

    if N < 5:
        raise ValueError(
            f"Insufficient time-stamped taxa found (N={N}/{N_taxa}). "
            "Please provide a valid Auspice JSON, metadata TSV/CSV with --dates, or ensure FASTA headers contain dates."
        )

    t_min = float(np.min(taxa_dates))
    t_max = float(np.max(taxa_dates))
    timespan = t_max - t_min
    if timespan <= 0:
        raise ValueError(f"Sampled timespan is non-positive (t_min={t_min}, t_max={t_max}).")

    if bandwidth is None or bandwidth <= 0:
        if time_units == "years":
            bandwidth = float(np.clip(timespan * 0.05, 0.05, 2.0))
        else:
            # No ceiling: for wide generation/day spans a 2.0-unit cap underflows the
            # Gaussian weight to 0 for samples spaced by hundreds/thousands of units.
            bandwidth = float(max(timespan * 0.05, (timespan / max(num_time_points, 1)) * 2.0))

    print(f"[✓] Time-stamped taxa: {N}/{N_taxa} ({100 * N / N_taxa:.1f}%) | Timespan: {t_min:.2f} - {t_max:.2f} ({timespan:.2f} {unit_label}) | units={time_units}, mode={sweep_mode}")
    print(f"[✓] Smoothing Kernel: Gaussian Nadaraya-Watson with Bandwidth h = {bandwidth:.3f} {unit_label}")

    # 4. Infer Root / Ancestral State
    root_aas, root_indices = infer_root_sequence(
        a, taxa, taxa_dates=taxa_dates, valid_taxa_mask=valid_taxa_mask, root_taxon=root_taxon
    )

    # Convert aligned sequences to amino acid indices
    a_valid = a.squeeze(-1).numpy()[:, valid_taxa_mask]  # [L, N]

    # 5. Neural Transformer Selection & Attribution Inference
    t0 = time.time()
    bs = compute_adaptive_safe_batch_size(N_taxa, user_batch_size=batch_size, device=device)
    bs = min(bs, max(1, L))

    lrts = np.zeros(L, dtype=np.float32)
    mean_attns = np.zeros((L, N), dtype=np.float32)

    with torch.no_grad():
        for start_idx in range(0, L, bs):
            end_idx = min(start_idx + bs, L)
            c_chunk = c[start_idx:end_idx].to(device)
            a_chunk = a[start_idx:end_idx].to(device)
            y_soft, _, root_attns = model.forward_cached(c_chunk, a_chunk, tree_cache, return_attentions=True)
            lrts[start_idx:end_idx] = torch.clamp(y_soft, min=0.0).cpu().numpy().flatten()
            mean_attns[start_idx:end_idx] = root_attns.cpu().numpy()[:, valid_taxa_mask]

    t_inf = time.time() - t0
    print(f"[✓] Neural inference completed in {t_inf:.2f}s | Mean LRT = {lrts.mean():.3f}, Max LRT = {lrts.max():.3f}")

    # 6. Static Scan Baseline (Self & Liang mixture asymptotics + BH FDR)
    pvals_static = pvals_from_lrt_self_liang(lrts)
    var_indices = np.where(~inv)[0]
    qvals_static = np.ones(L, dtype=np.float32)
    if len(var_indices) > 0:
        qvals_static[var_indices] = benjamini_hochberg(pvals_static[var_indices])
    n_sig_static = int(np.sum(qvals_static <= 0.10))
    print(f"[✓] Static Baseline: {n_sig_static} sites significant at FDR q <= 0.10")

    # 7. Directional Ancestral-Anchored Attribution
    delta_root = np.zeros((L, N), dtype=np.float32)
    for s in range(L):
        ref_idx = root_indices[s]
        valid_mask = (a_valid[s] < 20)
        delta_root[s] = (valid_mask & (a_valid[s] != ref_idx)).astype(np.float32)
    leaf_attributions = mean_attns * delta_root  # [L, N]

    # 8. Nadaraya-Watson Continuous Kernel Regression
    T = num_time_points
    dense_t = np.linspace(t_min, t_max, T)
    diff = dense_t[:, None] - taxa_dates[None, :]  # [T, N]
    weights = np.exp(-0.5 * (diff / bandwidth) ** 2)  # [T, N]
    norm_weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)  # [T, N]

    curves_matrix = leaf_attributions @ norm_weights.T  # [L, T] prevalence

    # 8b. Sweep metric on a SCALE-INVARIANT normalized time axis t~ in [0,1].
    # Differentiating w.r.t. t~ multiplies da/dt by the timespan, so the metric is
    # dimensionless whether time is in years or 100,000 generations (fixes the
    # generation-scale derivative underflow). Two regimes:
    #   episodic  : v_s(t~) = max(0, d a_s / d t~)   — active turnover (viral)
    #   fixation  : shift_s(t~) = a_s(t~) - a_s(0)   — permanent step-fixation (LTEE)
    norm_dense_t = np.linspace(0.0, 1.0, T)
    # Calendar/episodic keeps the original real-time gradient axis (byte-identical
    # viral behavior); non-calendar uses normalized time so the derivative and its
    # floors are scale-invariant across arbitrary generation spans.
    grad_t = norm_dense_t if non_calendar else dense_t

    def _metric(curves):
        if sweep_mode == "fixation":
            return curves - curves[:, :1]  # cumulative amplitude shift from baseline
        return np.maximum(0.0, np.gradient(curves, grad_t, axis=1))

    velocity_matrix = _metric(curves_matrix)  # [L,T]; named for downstream reuse
    if sweep_mode == "fixation":
        # Attention is distributed across taxa, so the raw attribution trajectory scales
        # as ~1/N_taxa. Normalizing by the per-site mean attention converts the amplitude
        # and velocity into an (attention-weighted) derived-state fraction in [0,1], making
        # the energy floor independent of taxon count and matching velocity/intensity scales for t_half.
        site_scale = mean_attns.mean(axis=1) + 1e-8   # [L] mean per-taxon attention
        velocity_matrix = velocity_matrix / site_scale[:, None]
        peak_intensities = np.ptp(curves_matrix, axis=1) / site_scale                       # max - min amplitude
        aucs = _trapezoid(np.maximum(0.0, velocity_matrix), norm_dense_t, axis=1)
        peak_times = dense_t[np.argmax(np.abs(velocity_matrix), axis=1)]
    else:
        peak_intensities = np.max(velocity_matrix, axis=1)  # [L]
        aucs = _trapezoid(velocity_matrix, grad_t, axis=1)  # [L]
        peak_times = dense_t[np.argmax(velocity_matrix, axis=1)]  # [L]

    t_half_start = np.zeros(L, dtype=np.float32)
    t_half_end = np.zeros(L, dtype=np.float32)
    fwhm_arr = np.zeros(L, dtype=np.float32)

    for s in range(L):
        y = velocity_matrix[s]
        pv = peak_intensities[s]
        if pv > 1e-7:
            above_half = y >= (pv / 2.0)
            if np.any(above_half):
                idx = np.where(above_half)[0]
                t_half_start[s] = dense_t[idx[0]]
                t_half_end[s] = dense_t[idx[-1]]
                fwhm_arr[s] = dense_t[idx[-1]] - dense_t[idx[0]]

    # 9. STAGE 1 FILTER: Energy Floor. On the normalized axis the floors are
    # dimensionless and independent of timespan (fixes tau_auc inflation on long
    # generation timelines, where the old timespan-scaled floor could never be met).
    if non_calendar:
        if tau_auc is None:
            tau_auc = 0.005   # dimensionless area on normalized time
        if tau_peak is None or tau_peak == 1e-4:
            tau_peak = 0.010  # dimensionless peak amplitude shift
    else:
        if tau_auc is None:
            tau_auc = 0.01e-3 * max(1.0, timespan / 5.0)
        if tau_peak is None or tau_peak == 1e-4:
            tau_peak = 0.5e-4

    stage1_mask = (~inv) & (peak_intensities >= tau_peak) & (aucs >= tau_auc)
    cand_indices = np.where(stage1_mask)[0]
    n_stage1 = len(cand_indices)
    print(f"[✓] Stage 1 Filter: {n_stage1} candidate codons passed sweep velocity floor (tau_peak={tau_peak * 1e3:.2f}e-3, tau_auc={tau_auc * 1e3:.2f}e-3)")

    # 10. STAGE 2 FILTER: Permutation Significance & Dynamic Wave Alignment
    B = n_permutations
    p_perm = np.ones(L, dtype=np.float32)
    q_perm = np.ones(L, dtype=np.float32)
    r2_fpca = np.zeros(L, dtype=np.float32)

    def _perm_stat(curves):
        # Test statistic for the date-shuffling null, matched to the sweep regime:
        #   fixation : directional Pearson correlation of the smoothed trajectory
        #              with time. A genuine fixation rises monotonically, so corr is
        #              large & positive regardless of WHERE in the timeline the
        #              transition sits (robust to early vs late sweeps); date-shuffling
        #              scatters derived states across time, driving corr to ~0. Signed
        #              (rewards rising derived-state sweeps), and unlike variance /
        #              amplitude / third-contrasts it is insensitive to transition
        #              location and to the derived-state base rate.
        #   episodic : variance of the positive velocity pulse (unchanged; viral).
        if sweep_mode == "fixation":
            tt = norm_dense_t - norm_dense_t.mean()
            cc = curves - curves.mean(axis=1, keepdims=True)
            num = (cc * tt[None, :]).sum(axis=1)
            den = np.sqrt((cc ** 2).sum(axis=1) * (tt ** 2).sum()) + 1e-12
            return num / den
        v = np.maximum(0.0, np.gradient(curves, grad_t, axis=1))
        return np.var(v, axis=1)

    if n_stage1 > 0:
        cand_attrs = leaf_attributions[cand_indices]  # [C, N]
        cand_curves = velocity_matrix[cand_indices]  # [C, T] (metric, for fPCA/SVD)
        cand_prev = curves_matrix[cand_indices]      # [C, T] (smoothed prevalence)
        v_obs = _perm_stat(cand_prev)  # [C]

        print(f"[*] Running {B} date-shuffling permutations across {n_stage1} candidates...")
        rng = np.random.RandomState(42)
        perm_exceed = np.zeros(n_stage1, dtype=np.int32)
        b_batch = 50
        for b_start in range(0, B, b_batch):
            b_cur = min(b_batch, B - b_start)
            for _ in range(b_cur):
                p_idx = rng.permutation(N)
                W_p = norm_weights[:, p_idx]
                c_p = cand_attrs @ W_p.T
                v_p = _perm_stat(c_p)
                perm_exceed += (v_p >= v_obs).astype(np.int32)

        p_cand = (1.0 + perm_exceed) / (B + 1.0)
        p_perm[cand_indices] = p_cand
        q_perm[cand_indices] = benjamini_hochberg(p_cand)

        # Dynamic Wave Alignment via SVD
        if n_stage1 >= 2:
            means_c = cand_curves.mean(axis=1, keepdims=True)
            stds_c = cand_curves.std(axis=1, keepdims=True) + 1e-8
            Z_cand = (cand_curves - means_c) / stds_c
            k_eff = min(4, n_stage1)
            u, s_vals, vt = np.linalg.svd(Z_cand, full_matrices=False)
            Z_recon = u[:, :k_eff] @ np.diag(s_vals[:k_eff]) @ vt[:k_eff, :]
            sse = np.sum((Z_cand - Z_recon) ** 2, axis=1)
            sst = np.sum(Z_cand ** 2, axis=1) + 1e-8
            r2_fpca[cand_indices] = np.clip(1.0 - sse / sst, 0.0, 1.0)
        else:
            r2_fpca[cand_indices] = 1.0

    # Solitary / asynchronous clonal-interference sweeps (few candidate sites, or any
    # experimental-evolution run) cannot form a collective multi-wave SVD signature, so
    # the fPCA R^2 gate is inapplicable — confirm on permutation significance alone.
    # Passing --min-r2 0.0 also selects this regime.
    solitary_regime = (n_stage1 <= 3) or non_calendar or (min_r2_fpca <= 0.0)
    if solitary_regime:
        is_confirmed_sweep = stage1_mask & (p_perm <= perm_alpha)
    else:
        is_confirmed_sweep = stage1_mask & (p_perm <= perm_alpha) & (r2_fpca >= min_r2_fpca)
    # Static-LRT escape hatch only in the calendar/epidemic regime (avoids reintroducing
    # static-selection noise into experimental-evolution results).
    if np.sum(is_confirmed_sweep) == 0 and n_stage1 > 0 and not solitary_regime:
        is_confirmed_sweep = stage1_mask & ((p_perm <= 0.10) | (lrts >= 3.84))

    n_sweeps = int(np.sum(is_confirmed_sweep))
    _r2_note = "n/a (solitary regime)" if solitary_regime else f"R2 >= {min_r2_fpca}"
    print(f"[✓] Stage 2 Filter: {n_sweeps} confirmed {sweep_mode} sweeps (p_perm <= {perm_alpha}, {_r2_note})")

    # 11. Cross-Classification vs Static Scans
    class_labels = []
    cross_labels = []
    for s in range(L):
        if inv[s]:
            c_label = "INVARIABLE"
            x_label = "NEGATIVE_CONSENSUS"
        elif not stage1_mask[s]:
            c_label = "FLAT_NO_SIGNAL"
            x_label = "FILTERED_STATIC_NOISE" if qvals_static[s] <= 0.10 else "NEGATIVE_CONSENSUS"
        elif not is_confirmed_sweep[s]:
            c_label = "TEMPORAL_NOISE"
            x_label = "FILTERED_STATIC_NOISE" if qvals_static[s] <= 0.10 else "NEGATIVE_CONSENSUS"
        else:
            c_label = "CONFIRMED_SWEEP"
            x_label = "CONCORDANT_SWEEP" if qvals_static[s] <= 0.10 else "RESCUED_SWEEP"
        class_labels.append(c_label)
        cross_labels.append(x_label)

    # 12. Functional Dynamic Wave Decomposition (fPCA)
    K = 4
    sweep_indices = np.where(is_confirmed_sweep)[0]
    if len(sweep_indices) < K:
        top_indices = np.argsort(-peak_intensities)[:max(K, len(cand_indices))]
        f_indices = top_indices if len(top_indices) >= K else np.arange(min(L, K))
    else:
        f_indices = sweep_indices

    Y_fpca = velocity_matrix[f_indices]
    m_fpca = Y_fpca.mean(axis=1, keepdims=True)
    s_fpca = Y_fpca.std(axis=1, keepdims=True) + 1e-8
    Z_fpca = (Y_fpca - m_fpca) / s_fpca
    u_f, s_f, vt_f = np.linalg.svd(Z_fpca, full_matrices=False)

    var_explained = (s_f ** 2) / np.sum(s_f ** 2) if np.sum(s_f ** 2) > 0 else np.zeros(K)
    waves = vt_f[:K]  # [K, T]

    all_loadings = np.zeros((L, K), dtype=np.float32)
    means_all = velocity_matrix.mean(axis=1, keepdims=True)
    stds_all = velocity_matrix.std(axis=1, keepdims=True) + 1e-8
    Z_all = (velocity_matrix - means_all) / stds_all
    for k in range(min(K, len(waves))):
        all_loadings[:, k] = Z_all @ waves[k]

    # Mutation labels (e.g., D614G, N501Y)
    major_aas = []
    mutation_labels = []
    domains = []
    for s in range(L):
        ref_idx = root_indices[s]
        vals_mut = [v for v in a_valid[s] if v < 20 and v != ref_idx]
        if len(vals_mut) > 0:
            der_idx = int(np.argmax(np.bincount(vals_mut, minlength=20)))
            der_aa = REV_AA_MAP.get(der_idx, root_aas[s])
        else:
            vals_all = [v for v in a_valid[s] if v < 20]
            der_aa = REV_AA_MAP.get(int(np.argmax(np.bincount(vals_all, minlength=20))), root_aas[s]) if len(vals_all) > 0 else root_aas[s]
        major_aas.append(der_aa)
        mut_str = f"{root_aas[s]}{s + 1}{der_aa}"
        mutation_labels.append(mut_str)
        dom = domain_map.get(s + 1, "Core") if domain_map else "Core"
        domains.append(dom)

    # 13. Compile Results DataFrames
    df_sites = pd.DataFrame({
        'site': np.arange(1, L + 1),
        'ref_aa': root_aas,
        'derived_aa': major_aas,
        'mutation_label': mutation_labels,
        'domain': domains,
        'cross_classification': cross_labels,
        'classification': class_labels,
        'is_confirmed_sweep': is_confirmed_sweep,
        'is_concordant_sweep': np.array([x == "CONCORDANT_SWEEP" for x in cross_labels]),
        'is_rescued_sweep': np.array([x == "RESCUED_SWEEP" for x in cross_labels]),
        'lrt': lrts,
        'p_static': pvals_static,
        'q_static': qvals_static,
        'p_perm': p_perm,
        'q_perm': q_perm,
        'r2_fpca': r2_fpca,
        'peak_date': peak_times,
        'peak_intensity': peak_intensities,
        't_half_start': t_half_start,
        't_half_end': t_half_end,
        'fwhm_years': fwhm_arr,
        'mean_intensity': np.mean(velocity_matrix, axis=1),
        'auc': aucs,
        'Wave_1_loading': all_loadings[:, 0],
        'Wave_2_loading': all_loadings[:, 1],
        'Wave_3_loading': all_loadings[:, 2],
        'Wave_4_loading': all_loadings[:, 3],
    })

    # Long format curves DataFrame (for plotting compatibility)
    curve_records = []
    # Only export curves for candidate / sweep sites to maintain compact CSV footprint
    export_sites = np.where(stage1_mask)[0] if n_stage1 > 0 else np.arange(min(L, 20))
    for s_idx in export_sites:
        s_num = s_idx + 1
        m_label = mutation_labels[s_idx]
        v_vals = velocity_matrix[s_idx]
        p_vals = curves_matrix[s_idx]
        for t_idx, t_val in enumerate(dense_t):
            curve_records.append({
                'site': s_num,
                'mutation_label': m_label,
                'time': float(t_val),
                'selection_intensity': float(v_vals[t_idx]),
                'sweep_velocity': float(v_vals[t_idx]),
                'prevalence': float(p_vals[t_idx])
            })
    df_curves = pd.DataFrame(curve_records)

    # Dynamic Waves DataFrame
    df_waves = pd.DataFrame({
        'time': dense_t,
        'wave_1': waves[0] if len(waves) > 0 else np.zeros(T),
        'wave_2': waves[1] if len(waves) > 1 else np.zeros(T),
        'wave_3': waves[2] if len(waves) > 2 else np.zeros(T),
        'wave_4': waves[3] if len(waves) > 3 else np.zeros(T),
    })

    # 14. Export Files
    exported_files = {}
    if output_prefix:
        ensure_parent_directory(output_prefix + "_sites_summary.csv")

        path_sites = f"{output_prefix}_sites_summary.csv"
        df_sites.to_csv(path_sites, index=False)
        exported_files['sites_summary'] = path_sites

        path_curves = f"{output_prefix}_curves.csv"
        df_curves.to_csv(path_curves, index=False)
        exported_files['curves'] = path_curves

        path_waves = f"{output_prefix}_waves.csv"
        df_waves.to_csv(path_waves, index=False)
        exported_files['waves'] = path_waves

        path_json = f"{output_prefix}_summary.json"
        summary_meta = {
            'alignment': alignment_path,
            'tree': tree_path,
            'taxa_total': N_taxa,
            'taxa_timestamped': N,
            'codons_total': L,
            'codons_variable': n_var,
            'codons_invariable': n_inv,
            'timespan_years': round(timespan, 2),
            't_min': round(t_min, 2),
            't_max': round(t_max, 2),
            'bandwidth_years': round(bandwidth, 4),
            'sig_static_q10': n_sig_static,
            'stage1_candidates': n_stage1,
            'confirmed_sweeps': n_sweeps,
            'concordant_sweeps': int(np.sum([1 for x in cross_labels if x == "CONCORDANT_SWEEP"])),
            'rescued_sweeps': int(np.sum([1 for x in cross_labels if x == "RESCUED_SWEEP"])),
            'filtered_static_noise': int(np.sum([1 for x in cross_labels if x == "FILTERED_STATIC_NOISE"])),
            'fpca_wave_variance_pct': [round(float(v * 100), 2) for v in var_explained[:4]],
            'runtime_sec': round(time.time() - t_start, 2)
        }
        write_json(path_json, summary_meta)
        exported_files['summary_json'] = path_json

        # 15. Optional Publication Figure Generation
        if plot:
            if not HAS_MATPLOTLIB:
                print("[!] Matplotlib not installed; skipping plot generation.")
            else:
                pdf_path = f"{output_prefix}_regressions.pdf"
                png_path = f"{output_prefix}_regressions.png"
                title_str = Path(alignment_path).stem.replace('_', ' ').title()
                render_temporal_4panel_figure(
                    df_sites=df_sites,
                    df_curves=df_curves,
                    df_waves=df_waves,
                    out_pdf_path=pdf_path,
                    out_png_path=png_path,
                    title=f"{title_str} Selection Dynamics"
                )
                exported_files['plot_pdf'] = pdf_path
                exported_files['plot_png'] = png_path

    t_total = time.time() - t_start
    print("\n" + "=" * 85)
    print(f"🎉 HyphAeon Temporal Surveillance Complete in {t_total:.2f}s!")
    print(f"   Confirmed Sweeps:       {n_sweeps}")
    print(f"   Rescued Sweeps:         {int(np.sum([1 for x in cross_labels if x == 'RESCUED_SWEEP']))}")
    print(f"   Concordant Sweeps:      {int(np.sum([1 for x in cross_labels if x == 'CONCORDANT_SWEEP']))}")
    print(f"   Filtered Static Noise:  {int(np.sum([1 for x in cross_labels if x == 'FILTERED_STATIC_NOISE']))}")
    print(f"   fPCA Cumulative Var:    {np.sum(var_explained[:4]) * 100:.1f}%")
    if exported_files:
        print(f"   Exported Summary Table: {exported_files.get('sites_summary')}")
        print(f"   Exported Curves:        {exported_files.get('curves')}")
        print(f"   Exported Waves:         {exported_files.get('waves')}")
        if 'plot_pdf' in exported_files:
            print(f"   Exported 4-Panel Plot:  {exported_files.get('plot_pdf')}")
    print("=" * 85 + "\n")

    return {
        'sites_summary': df_sites,
        'curves': df_curves,
        'waves': df_waves,
        'exported_files': exported_files,
        'metadata': summary_meta if output_prefix else {}
    }


# =========================================================================
# 4. Publication-Grade 4-Panel Visualization
# =========================================================================

def render_temporal_4panel_figure(
    df_sites: pd.DataFrame,
    df_curves: pd.DataFrame,
    df_waves: pd.DataFrame,
    out_pdf_path: str,
    out_png_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Renders a publication-grade 4-panel figure:
      Panel A: Waterfall Heatmap of all confirmed sweeps ordered chronologically by peak date.
      Panel B: Macro domain / cluster trajectories (B1) and individual landmark driver traces (B2).
      Panel C: Functional Dynamic Factor Waves (fPCA) capturing collective turnover.
      Panel D: Factor loadings for leading adaptive drivers across each dynamic wave.
    """
    if not HAS_MATPLOTLIB:
        return

    # Filter to confirmed sweeps (or top sites if sweeps are sparse)
    df_sweeps = df_sites[df_sites['is_confirmed_sweep']].sort_values("peak_date", ascending=True).reset_index(drop=True)
    if len(df_sweeps) < 5:
        df_sweeps = df_sites[df_sites['peak_intensity'] > 0].sort_values("peak_intensity", ascending=False).head(40)
        df_sweeps = df_sweeps.sort_values("peak_date", ascending=True).reset_index(drop=True)

    dense_t = df_waves["time"].values
    T = len(dense_t)
    S = len(df_sweeps)

    if S == 0:
        return

    sites_order = df_sweeps["site"].values
    mut_labels = df_sweeps["mutation_label"].values

    # Build matrix of normalized intensity: [S, T]
    norm_matrix = np.zeros((S, T))
    raw_matrix = np.zeros((S, T))
    curves_dict = {}

    for i, s in enumerate(sites_order):
        sub = df_curves[df_curves["site"] == s].sort_values("time")
        if "sweep_velocity" in sub.columns:
            vals = sub["sweep_velocity"].values
        elif "selection_intensity" in sub.columns:
            vals = sub["selection_intensity"].values
        else:
            vals = sub.iloc[:, -1].values

        if len(vals) != T:
            vals = np.interp(dense_t, sub["time"].values, vals)
        raw_matrix[i, :] = vals
        v_max = np.max(vals)
        norm_matrix[i, :] = vals / v_max if v_max > 0 else vals
        curves_dict[s] = vals

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 8.5,
        'axes.labelsize': 9.2,
        'axes.titlesize': 10.2,
        'xtick.labelsize': 8.0,
        'ytick.labelsize': 7.6,
        'legend.fontsize': 7.5,
        'figure.titlesize': 12.0
    })

    fig = plt.figure(figsize=(16.5, 12.2), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.35, 1.0], hspace=0.34, wspace=0.28)

    # -----------------------------------------------------------------
    # PANEL A: Waterfall Heatmap
    # -----------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    im = ax_a.imshow(
        norm_matrix, aspect='auto', cmap='inferno', origin='upper',
        extent=[dense_t.min(), dense_t.max(), S - 0.5, -0.5]
    )

    # Subsample y-ticks for clarity
    tick_step = max(1, S // 30)
    tick_indices = list(range(0, S, tick_step))
    if (S - 1) not in tick_indices:
        tick_indices.append(S - 1)

    ax_a.set_yticks(tick_indices)
    ax_a.set_yticklabels([mut_labels[idx] for idx in tick_indices], fontsize=7.2)
    ax_a.set_xlabel("Evolutionary Date (Years)")
    ax_a.set_ylabel("Episodic Selective Sweeps (Chronological Order)")
    ax_a.set_title(f"A. Continuous Sweep Velocity Waterfall ({S} Confirmed Sweeps)", fontweight='bold', loc='left')

    cbar = plt.colorbar(im, ax=ax_a, fraction=0.035, pad=0.02)
    cbar.set_label(r"Normalized Sweep Velocity $v_s(t) / \max(v_s)$", fontsize=8.0)

    # -----------------------------------------------------------------
    # PANEL B: Nested Subplots (B1: Macro / Mean, B2: Landmark Traces)
    # -----------------------------------------------------------------
    gs_b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1], hspace=0.36)
    ax_b1 = fig.add_subplot(gs_b[0])
    ax_b2 = fig.add_subplot(gs_b[1])

    # B1: Macro domain trajectories if domain column exists and has multiple domains, else collective mean
    if "domain" in df_sweeps.columns and df_sweeps["domain"].nunique() > 1:
        domains = df_sweeps["domain"].unique()
        palette_dom = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#ff7f0e', '#e377c2']
        for d_idx, dom in enumerate(domains):
            dom_mask = (df_sweeps["domain"] == dom).values
            if np.sum(dom_mask) > 0:
                dom_curves = raw_matrix[dom_mask, :]
                mean_d = np.mean(dom_curves, axis=0)
                sem_d = stats.sem(dom_curves, axis=0) if np.sum(dom_mask) > 1 else np.zeros(T)
                col = palette_dom[d_idx % len(palette_dom)]
                ax_b1.plot(dense_t, mean_d, color=col, lw=2.0, label=f"{dom} (N={np.sum(dom_mask)})")
                ax_b1.fill_between(dense_t, np.maximum(0, mean_d - sem_d), mean_d + sem_d, color=col, alpha=0.18)
        ax_b1.set_ylabel(r"Mean Sweep Velocity $\bar{v}(t)$")
        ax_b1.set_title(r"B$_1$. Domain-Stratified Sweep Velocities ($\pm$ SEM)", fontweight='bold', loc='left')
        ax_b1.legend(loc='upper right', frameon=True, fontsize=6.8, ncol=2)
    else:
        mean_traj = np.mean(raw_matrix, axis=0)
        sem_traj = stats.sem(raw_matrix, axis=0) if S > 1 else np.zeros(T)
        ax_b1.plot(dense_t, mean_traj, color='#1f77b4', lw=2.0, label='Mean Sweep Velocity')
        ax_b1.fill_between(dense_t, np.maximum(0, mean_traj - sem_traj), mean_traj + sem_traj, color='#1f77b4', alpha=0.25)
        ax_b1.set_ylabel(r"Mean Sweep Velocity $\bar{v}(t)$")
        ax_b1.set_title(r"B$_1$. Collective Sweep Velocity ($\pm$ SEM)", fontweight='bold', loc='left')
        ax_b1.legend(loc='upper right', frameon=True)
    ax_b1.grid(True, linestyle=':', alpha=0.5)

    # B2: Top individual drivers
    top_indices = np.argsort(-df_sweeps["peak_intensity"].values)[:min(8, S)]
    palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    for idx_c, s_idx in enumerate(top_indices):
        s_num = sites_order[s_idx]
        col = palette[idx_c % len(palette)]
        ax_b2.plot(dense_t, curves_dict[s_num], color=col, lw=1.8, label=mut_labels[s_idx])

    ax_b2.set_xlabel("Evolutionary Date (Years)")
    ax_b2.set_ylabel(r"Sweep Velocity $v_s(t)$")
    ax_b2.set_title(r"B$_2$. Top Driver Residue Sweep Velocities", fontweight='bold', loc='left')
    ax_b2.grid(True, linestyle=':', alpha=0.5)
    ax_b2.legend(loc='upper right', ncol=2, frameon=True, fontsize=7.0)

    # -----------------------------------------------------------------
    # PANEL C: Functional Dynamic Waves (fPCA)
    # -----------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    wave_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for k in range(min(4, len(df_waves.columns) - 1)):
        col_name = f"wave_{k+1}"
        if col_name in df_waves.columns:
            ax_c.plot(dense_t, df_waves[col_name].values, color=wave_colors[k], lw=2.0, label=f"Wave {k+1}")

    ax_c.set_xlabel("Evolutionary Date (Years)")
    ax_c.set_ylabel(r"Wave Amplitude $w_k(t)$")
    ax_c.set_title("C. Functional Dynamic Factor Waves (fPCA on Sweep Velocity)", fontweight='bold', loc='left')
    ax_c.grid(True, linestyle=':', alpha=0.5)
    ax_c.legend(loc='upper right', frameon=True)

    # -----------------------------------------------------------------
    # PANEL D: Factor Loadings for Leading Drivers
    # -----------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    top_load_idx = np.argsort(-np.abs(df_sweeps["Wave_1_loading"].values))[:min(10, S)]
    y_pos = np.arange(len(top_load_idx))
    loadings = df_sweeps["Wave_1_loading"].values[top_load_idx]
    labels = [mut_labels[i] for i in top_load_idx]

    bar_colors = ['#1f77b4' if l >= 0 else '#d62728' for l in loadings]
    ax_d.barh(y_pos, loadings, color=bar_colors, alpha=0.85, edgecolor='black', lw=0.6)
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(labels, fontsize=7.8)
    ax_d.invert_yaxis()
    ax_d.set_xlabel(r"Factor Loading $L_{s, 1} = \int z_s(t) w_1(t) dt$")
    ax_d.set_title("D. Leading Adaptive Drivers: In-Phase ($L > 0$) vs Anti-Phase ($L < 0$)", fontweight='bold', loc='left', pad=14)
    ax_d.text(0.02, 1.02, r"$\leftarrow$ Anti-Phase ($L < 0$; active during wave troughs)",
              transform=ax_d.transAxes, fontsize=6.8, color='#d62728', fontweight='bold', va='bottom')
    ax_d.text(0.98, 1.02, r"In-Phase ($L > 0$; active during wave peaks) $\rightarrow$",
              transform=ax_d.transAxes, fontsize=6.8, color='#1f77b4', fontweight='bold', va='bottom', ha='right')
    ax_d.grid(True, linestyle=':', alpha=0.5, axis='x')

    if title:
        fig.suptitle(title, fontsize=13.5, fontweight='bold', y=0.995)

    ensure_parent_directory(out_pdf_path)
    fig.savefig(out_pdf_path, bbox_inches='tight')
    if out_png_path:
        ensure_parent_directory(out_png_path)
        fig.savefig(out_png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[✓] Publication 4-panel figure rendered: {out_pdf_path}")
