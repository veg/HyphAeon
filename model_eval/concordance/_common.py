"""
concordance/_common.py — shared helpers for HyphAeon-vs-MEME concordance
tests (test_hyphaeon_vs_meme.py: TestHyphAeonvsMEME and
TestHyphAeonvsMEMETypicalCase).

Not collected by pytest (module name doesn't match test_*.py). Provides:
  - run_hyphy_meme: run (or fetch cached) HyPhy MEME on an alignment+tree,
    returning per-site {lrt, p_value}.
  - meme_dict_to_arrays: convert that dict into arrays aligned with
    HyphAeon's per-site LRT/p-value arrays.
  - concordance_metrics: compute Spearman rho, Cohen's kappa, and F1
    between HyphAeon and MEME on the tested sites.

CACHING: MEME results are cached in model_eval/_cache/ keyed on
(fasta hash, tree hash, hyphy version). MEME is deterministic for a given
input + version, so re-running is wasteful — especially on camelid (212
taxa, which takes minutes). The cache is committed to the repo so CI
doesn't need to run MEME either. Delete the cache file to force a re-run.
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest
from scipy.stats import spearmanr

try:
    from sklearn.metrics import cohen_kappa_score, f1_score
except ImportError:
    cohen_kappa_score = None
    f1_score = None

_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


def _hyphy_version():
    """Return hyphy version string, or None if not available."""
    if not shutil.which("hyphy"):
        return None
    try:
        result = subprocess.run(["hyphy", "--version"],
                                capture_output=True, text=True, timeout=10)
        # Output: "HYPHY 2.5.101(MP) for Linux on x86_64 ..."
        for part in result.stdout.split():
            if part.startswith("2.") or part.startswith("3."):
                return part
        return result.stdout.strip().split("\n")[0]
    except Exception:
        return None


def _file_hash(path):
    """SHA256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def run_hyphy_meme(fasta_path, tree_path, timeout=600):
    """Run HyPhy MEME and return a dict: site_index -> {p_value, lrt}.

    Results are cached in model_eval/_cache/ keyed on
    (fasta hash, tree hash, hyphy version). Returns None if hyphy is
    unavailable or the run fails.
    """
    version = _hyphy_version()
    if version is None:
        return None

    os.makedirs(_CACHE_DIR, exist_ok=True)

    # Cache key: hashes of inputs + hyphy version
    fa_hash = _file_hash(fasta_path)
    nwk_hash = _file_hash(tree_path)
    cache_key = f"meme_{fa_hash}_{nwk_hash}_hyphy{version}.json"
    cache_path = os.path.join(_CACHE_DIR, cache_key)

    # Check cache first
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            sites = json.load(f)
        # Convert string keys back to int (JSON serializes dict keys as strings)
        sites = {int(k): v for k, v in sites.items()}
        print(f"  [cache hit] {cache_key}")
        return sites if sites else None

    # Run MEME
    tmp = tempfile.mkdtemp(prefix="hyphy_meme_")
    json_out = os.path.join(tmp, "meme_results.json")

    print(f"  [cache miss] running HyPhy MEME (this may take a while)...")
    result = subprocess.run(
        ["hyphy", "meme", "--alignment", fasta_path, "--tree", tree_path,
         "--output", json_out],
        capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0 or not os.path.exists(json_out):
        return None

    with open(json_out) as f:
        data = json.load(f)

    # HyPhy MEME JSON: MLE.content.<partition> is a list of per-site rows.
    # Column 5 (0-based) = LRT, column 6 = p-value.
    sites = {}
    for partition_key, rows in data.get("MLE", {}).get("content", {}).items():
        for i, row in enumerate(rows):
            if len(row) > 6:
                lrt = float(row[5])
                pval = float(row[6])
                sites[i] = {"p_value": pval, "lrt": lrt}

    if not sites:
        return None

    # Write to cache (keys as strings for JSON)
    with open(cache_path, "w") as f:
        json.dump({str(k): v for k, v in sites.items()}, f, indent=2)
    print(f"  [cache write] {cache_key}")

    return sites


def run_hyphy_busted(fasta_path, tree_path, timeout=1200):
    """Run HyPhy BUSTED and return a dict: {lrt, p_value} (gene-level, not per-site).

    Results are cached in model_eval/_cache/ keyed on
    (fasta hash, tree hash, hyphy version). Returns None if hyphy is
    unavailable or the run fails.
    """
    version = _hyphy_version()
    if version is None:
        return None

    os.makedirs(_CACHE_DIR, exist_ok=True)

    fa_hash = _file_hash(fasta_path)
    nwk_hash = _file_hash(tree_path)
    cache_key = f"busted_{fa_hash}_{nwk_hash}_hyphy{version}.json"
    cache_path = os.path.join(_CACHE_DIR, cache_key)

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            result = json.load(f)
        print(f"  [cache hit] {cache_key}")
        return result if result else None

    tmp = tempfile.mkdtemp(prefix="hyphy_busted_")
    json_out = os.path.join(tmp, "busted_results.json")

    print(f"  [cache miss] running HyPhy BUSTED...")
    result = subprocess.run(
        ["hyphy", "busted", "--alignment", fasta_path, "--tree", tree_path,
         "--output", json_out],
        capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0 or not os.path.exists(json_out):
        return None

    with open(json_out) as f:
        data = json.load(f)

    test_results = data.get("test results", {})
    lrt = float(test_results.get("LRT", 0.0))
    pval = float(test_results.get("p-value", 1.0))

    result_dict = {"lrt": lrt, "p_value": pval}

    with open(cache_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"  [cache write] {cache_key}")

    return result_dict


def meme_dict_to_arrays(meme_sites, n_sites):
    """Convert a {site_index: {lrt, p_value}} dict into arrays aligned with
    HyphAeon's per-site LRT/p-value arrays (length n_sites).

    Sites with no MEME entry default to lrt=0, p_value=1 (non-significant).

    Returns (meme_lrts, meme_pvals, meme_tested) where meme_tested is a
    boolean mask of sites where MEME actually produced a result. Use this
    mask to avoid penalizing HyphAeon for sites MEME skipped.
    """
    meme_lrts = np.zeros(n_sites)
    meme_pvals = np.ones(n_sites)
    meme_tested = np.zeros(n_sites, dtype=bool)
    for site_idx, info in meme_sites.items():
        if 0 <= site_idx < n_sites:
            meme_lrts[site_idx] = info["lrt"]
            meme_pvals[site_idx] = info["p_value"]
            meme_tested[site_idx] = True
    return meme_lrts, meme_pvals, meme_tested


def concordance_metrics(axo_lrts, axo_pvals, meme_lrts, meme_pvals, tested,
                        meme_tested=None, alpha=0.05):
    """Compute Spearman rho, Cohen's kappa, and F1 between HyphAeon and MEME
    on the tested (variable) sites.

    If meme_tested is provided, only sites where both HyphAeon has a
    variable site AND MEME produced a result are included in the metrics.
    This avoids penalizing HyphAeon for sites that MEME skipped (e.g.,
    insufficient substitutions), which would default to p=1 in the MEME
    arrays and count as discordances.

    Returns a dict suitable for direct inclusion in a JSON report.
    """
    if cohen_kappa_score is None:
        pytest.skip("scikit-learn not available")

    if meme_tested is not None:
        concordance_tested = tested & meme_tested
    else:
        concordance_tested = tested

    axo_var = axo_lrts[concordance_tested]
    meme_var = meme_lrts[concordance_tested]
    axo_p_var = axo_pvals[concordance_tested]
    meme_p_var = meme_pvals[concordance_tested]

    rho, p_rho = spearmanr(axo_var, meme_var)

    axo_sig = axo_p_var <= alpha
    meme_sig = meme_p_var <= alpha
    kappa = cohen_kappa_score(meme_sig, axo_sig)
    f1 = f1_score(meme_sig, axo_sig, zero_division=0)

    alpha_tag = f"{alpha:g}".replace(".", "")  # 0.05 -> "005", matches
                                                 # existing artifact key naming
    return {
        "n_concordance_sites": int(concordance_tested.sum()),
        "n_variable_sites": int(tested.sum()),
        "n_meme_tested_sites": int(meme_tested.sum()) if meme_tested is not None else int(tested.sum()),
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        f"cohen_kappa_{alpha_tag}": float(kappa),
        f"f1_{alpha_tag}": float(f1),
        f"hyphaeon_significant_{alpha_tag}": int(axo_sig.sum()),
        f"meme_significant_{alpha_tag}": int(meme_sig.sum()),
    }
