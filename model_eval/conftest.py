"""
conftest.py for model_eval/

Resolves real HyphAeon weights (local path or Hugging Face download) and provides
shared fixtures. If weights cannot be resolved, all tests in this directory are
skipped with a clear reason — not silently passed, not errored.

This is deliberately separate from tests/conftest.py, which uses a dummy
random-weight checkpoint and must never depend on real weights.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Make the package and this directory importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hyphaeon.model import PhyloAxialTransformer
from hyphaeon import dataset as ds
from hyphaeon.inference import load_model

EXAMPLES_DIR = REPO_ROOT / "examples"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Weight resolution: use hyphaeon.weights module (Hugging Face download by
# default, HYPHAEON_WEIGHTS env var for local .pt/.safetensors files).
# This mirrors the CLI's behavior: CI downloads from HF; local devs can
# point at a working checkpoint while iterating before pushing to HF.
from hyphaeon.weights import resolve_weights_path


# ---------------------------------------------------------------------------
# Weight resolution
# ---------------------------------------------------------------------------

def _resolve_weights():
    """Return (path, source_label) or None if weights are unavailable.

    Resolution order (matches CLI):
    1. HYPHAEON_WEIGHTS env var → local file (if it exists)
    2. Default variant from Hugging Face (downloads + caches on first use)
    """
    explicit = os.environ.get("HYPHAEON_WEIGHTS")
    if explicit and os.path.exists(explicit):
        return explicit, f"HYPHAEON_WEIGHTS={explicit}"
    try:
        path = resolve_weights_path(weights=None)
        return path, f"Hugging Face (cached at {path})"
    except Exception as e:
        return None, (
            f"weights unavailable: {e}. Set HYPHAEON_WEIGHTS to a local "
            f".pt/.safetensors checkpoint, or ensure network access for Hugging Face download."
        )


_WEIGHTS_PATH, _WEIGHTS_SOURCE = _resolve_weights()

WEIGHTS_AVAILABLE = _WEIGHTS_PATH is not None


def _load_model_from(path):
    """Load a checkpoint and return an eval-mode PhyloAxialTransformer on CPU."""
    return load_model(weights=path, device=torch.device("cpu"), strict=True)


@pytest.fixture(scope="session")
def weights_info():
    """Return (path, source_label); skip if weights unavailable."""
    if not WEIGHTS_AVAILABLE:
        pytest.skip(f"HyphAeon weights not available ({_WEIGHTS_SOURCE}). "
                    f"Set HYPHAEON_WEIGHTS or ensure network access to run model_eval tests.")
    return _WEIGHTS_PATH, _WEIGHTS_SOURCE


@pytest.fixture(scope="session")
def model(weights_info):
    """Session-scoped trained model in eval mode on CPU."""
    return _load_model_from(weights_info[0])


# ---------------------------------------------------------------------------
# Example data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def examples_dir():
    return str(EXAMPLES_DIR)


# ---------------------------------------------------------------------------
# Real dataset fixtures — three datasets spanning the parameter space
# ---------------------------------------------------------------------------

def _load_example(name, examples_dir):
    """Return (fasta_path, newick_path) for an example dataset, or skip."""
    fa = os.path.join(examples_dir, f"{name}.fasta")
    nwk = os.path.join(examples_dir, f"{name}.nwk")
    if not (os.path.exists(fa) and os.path.exists(nwk)):
        pytest.skip(f"{name} example data not found at {fa} / {nwk}")
    return fa, nwk


@pytest.fixture(scope="session")
def smc6_paths(examples_dir):
    """Smc6: 20 taxa, shallow tree (depth ~0.04), 26% variable sites."""
    return _load_example("Smc6", examples_dir)


@pytest.fixture(scope="session")
def bat_oas1_paths(examples_dir):
    """bat_oas1: 18 taxa, ultra-deep tree (depth ~62), 84% variable sites."""
    return _load_example("bat_oas1", examples_dir)


@pytest.fixture(scope="session")
def camelid_paths(examples_dir):
    """camelid: 212 taxa, topology-only tree (no branch lengths), 100% variable."""
    return _load_example("camelid", examples_dir)


def _make_base(model, paths):
    """Load tensors + baseline LRT for a dataset. Returns dict or None."""
    from _harness import load_tensors, predict, pvals_from_lrt
    fa, nwk = paths
    try:
        c, a, d, z, inv, taxa, L = load_tensors(fa, nwk)
    except Exception as e:
        print(f"  [WARNING] Dataset {os.path.basename(fa)} failed to load: "
              f"{type(e).__name__}: {e}")
        return None
    lrt = predict(model, c, a, d, z, inv)
    pval = pvals_from_lrt(lrt)
    return {
        "name": os.path.basename(fa).replace(".fasta", ""),
        "fa": fa, "nwk": nwk,
        "c": c, "a": a, "d": d, "z": z, "inv": inv,
        "taxa": taxa, "L": L, "lrt": lrt, "pval": pval,
        "tested": ~inv, "n_taxa": len(taxa),
    }


@pytest.fixture(scope="session")
def smc6_base(model, smc6_paths):
    """Loaded Smc6 tensors + baseline LRT predictions."""
    return _make_base(model, smc6_paths)


@pytest.fixture(scope="session")
def bat_oas1_base(model, bat_oas1_paths):
    """Loaded bat_oas1 tensors + baseline LRT predictions."""
    return _make_base(model, bat_oas1_paths)


@pytest.fixture(scope="session")
def camelid_base(model, camelid_paths):
    """Loaded camelid tensors + baseline LRT predictions."""
    return _make_base(model, camelid_paths)


@pytest.fixture(scope="session")
def real_datasets(model, smc6_base, bat_oas1_base, camelid_base):
    """All real datasets that loaded successfully. Returns list of base dicts.

    Each dict has: name, fa, nwk, c, a, d, z, inv, taxa, L, lrt, pval, tested, n_taxa.
    Datasets that fail to load (e.g. tree parsing issues) are skipped with
    a warning printed to stdout.
    """
    datasets = []
    for base in [smc6_base, bat_oas1_base, camelid_base]:
        if base is not None:
            datasets.append(base)
    if not datasets:
        pytest.skip("No real datasets loaded")
    return datasets


@pytest.fixture(scope="session")
def sim_datasets(model, seqgen_available):
    """Simulated datasets at varying tree depths and taxon counts.

    Fills the grid between the real datasets:
      - 50 taxa, moderate depth (0.2)
      - 100 taxa, deep (0.5)
      - 200 taxa, deep (1.0)

    Uses seq-gen with HKY under neutrality. Even though there's no real
    selection, the model produces non-trivial LRTs on variable sites —
    enough to test whether tree perturbations change the output.
    """
    from _sim import simulate_neutral_alignment
    from _harness import load_tensors, predict, pvals_from_lrt

    configs = [
        ("sim_50_mod", 50, 100, 0.2, 1.0, 100),
        ("sim_100_deep", 100, 100, 0.5, 1.0, 200),
        ("sim_200_deep", 200, 100, 1.0, 1.0, 300),
    ]
    datasets = []
    for name, n_taxa, n_codons, depth, scale, seed in configs:
        fa, nwk = simulate_neutral_alignment(
            n_taxa=n_taxa, n_codons=n_codons,
            tree_depth=depth, scale=scale, seed=seed)
        try:
            c, a, d, z, inv, taxa, L = load_tensors(fa, nwk)
        except Exception as e:
            print(f"  [WARNING] Simulated dataset {name} failed to load: "
                  f"{type(e).__name__}: {e}")
            continue
        lrt = predict(model, c, a, d, z, inv)
        pval = pvals_from_lrt(lrt)
        tested = ~inv
        if tested.sum() < 5:
            continue  # not enough variable sites
        datasets.append({
            "name": name, "fa": fa, "nwk": nwk,
            "c": c, "a": a, "d": d, "z": z, "inv": inv,
            "taxa": taxa, "L": L, "lrt": lrt, "pval": pval,
            "tested": tested, "n_taxa": len(taxa),
        })
    if not datasets:
        pytest.skip("No simulated datasets loaded")
    return datasets


@pytest.fixture(scope="session")
def all_datasets(real_datasets, sim_datasets):
    """All real + simulated datasets. Used by parametrized invariance gates."""
    total = real_datasets + sim_datasets
    if len(sim_datasets) == 0:
        print(f"  [WARNING] No simulated datasets available (seq-gen missing?). "
              f"Invariance gates will run on {len(real_datasets)} real datasets "
              f"only — majority threshold is {len(real_datasets)//2 + 1}.")
    return total


@pytest.fixture
def artifacts_dir():
    return str(ARTIFACTS_DIR)


# ---------------------------------------------------------------------------
# Tool availability checks
# ---------------------------------------------------------------------------

def _check_tool(name):
    """Return True if a command-line tool is on PATH."""
    import shutil
    return shutil.which(name) is not None


HYPHY_AVAILABLE = _check_tool("hyphy")
SEQGEN_AVAILABLE = _check_tool("seq-gen")


@pytest.fixture(scope="session")
def hyphy_available():
    """Skip test if HyPhy is not installed."""
    if not HYPHY_AVAILABLE:
        pytest.skip("HyPhy not on PATH — required for MEME concordance tests")
    return True


@pytest.fixture(scope="session")
def seqgen_available():
    """Skip test if seq-gen is not installed."""
    if not SEQGEN_AVAILABLE:
        pytest.skip("seq-gen not on PATH — required for neutral simulation tests")
    return True
