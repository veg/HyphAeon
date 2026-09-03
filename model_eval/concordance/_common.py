"""
concordance/_common.py — HyPhy subprocess wrappers + cache for concordance
tests (test_hyphaeon_vs_meme.py, test_busted_vs_hyphy.py).

Not collected by pytest (module name doesn't match test_*.py). Provides:
  - run_hyphy_meme: run (or fetch cached) HyPhy MEME on an alignment+tree,
    returning per-site MemeSite records (parsed via the shared backend in
    concordance_compare.load_meme_json).
  - run_hyphy_busted: run (or fetch cached) HyPhy BUSTED, returning a
    gene-level {lrt, p_value} dict.

File parsing, site alignment, and concordance metrics live in
``concordance_compare.py`` (the shared backend used by both the pytest tests
and the ``python -m model_eval`` CLI). This module only owns the HyPhy
subprocess invocation and its on-disk cache.

CACHING: MEME/BUSTED results are cached in model_eval/_cache/ keyed on
(fasta hash, tree hash, hyphy version). Both are deterministic for a given
input + version, so re-running is wasteful — especially on camelid (212
taxa, which takes minutes). The cache is committed to the repo so CI
doesn't need to run HyPhy either. Delete the cache file to force a re-run.
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    # Under `python -m model_eval`, this module is model_eval.concordance._common
    # and concordance_compare is a sibling of the concordance package.
    from ..concordance_compare import (
        MemeSite,
        load_meme_json,
        meme_sites_to_arrays,
        concordance_metrics,
    )
except ImportError:
    # Under pytest, model_eval/ is on sys.path and concordance_compare is a
    # top-level module (conftest.py inserts the model_eval dir onto sys.path).
    from concordance_compare import (
        MemeSite,
        load_meme_json,
        meme_sites_to_arrays,
        concordance_metrics,
    )

# Re-export the shared backend symbols so existing imports
# (`from concordance._common import ...`) keep working during the transition.
__all__ = [
    "run_hyphy_meme",
    "run_hyphy_busted",
    "load_meme_json",
    "meme_sites_to_arrays",
    "concordance_metrics",
    "MemeSite",
]

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


def _meme_sites_from_cache(path):
    """Deserialize a cached MEME file into Dict[int, MemeSite].

    Cache files are JSON of {str(site): {lrt, p_value[, lrt_was_clamped]}}.

    Two cache generations exist:
      - Old (pre-refactor _common.py): 0-based site indices, no
        ``lrt_was_clamped`` field. Built by ``enumerate(rows)`` directly.
      - New (post-refactor): 1-based site IDs (matching load_meme_json),
        with ``lrt_was_clamped``.

    Old caches are detected by the absence of ``lrt_was_clamped`` and
    shifted to 1-based to match load_meme_json's convention.
    """
    with open(path) as f:
        raw = json.load(f)
    if not raw:
        return None
    # Detect old 0-based caches: they lack lrt_was_clamped on every entry.
    is_old_cache = all("lrt_was_clamped" not in v for v in raw.values())
    shift = 1 if is_old_cache else 0
    return {
        (int(k) + shift): MemeSite(
            lrt=float(v["lrt"]),
            p_value=float(v["p_value"]),
            lrt_was_clamped=bool(v.get("lrt_was_clamped", False)),
        )
        for k, v in raw.items()
    }


def _meme_sites_to_cache(sites):
    """Serialize Dict[int, MemeSite] to a JSON-friendly dict for caching.

    Site IDs are 1-based (matching load_meme_json's convention).
    """
    return {
        str(k): {"lrt": v.lrt, "p_value": v.p_value, "lrt_was_clamped": v.lrt_was_clamped}
        for k, v in sites.items()
    }


def run_hyphy_meme(fasta_path, tree_path, timeout=600):
    """Run HyPhy MEME and return a dict: site_index -> MemeSite.

    Results are cached in model_eval/_cache/ keyed on
    (fasta hash, tree hash, hyphy version). Returns None if hyphy is
    unavailable or the run fails. Parsing uses the shared backend
    (concordance_compare.load_meme_json), which handles partition coverage
    and negative-LRT clamping.
    """
    version = _hyphy_version()
    if version is None:
        return None

    os.makedirs(_CACHE_DIR, exist_ok=True)

    fa_hash = _file_hash(fasta_path)
    nwk_hash = _file_hash(tree_path)
    cache_key = f"meme_{fa_hash}_{nwk_hash}_hyphy{version}.json"
    cache_path = os.path.join(_CACHE_DIR, cache_key)

    if os.path.exists(cache_path):
        sites = _meme_sites_from_cache(cache_path)
        if sites is None:
            return None
        print(f"  [cache hit] {cache_key}")
        return sites

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

    sites = load_meme_json(Path(json_out))

    with open(cache_path, "w") as f:
        json.dump(_meme_sites_to_cache(sites), f, indent=2)
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
