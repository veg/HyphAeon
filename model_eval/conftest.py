"""
conftest.py for model_eval/

Resolves real AxoMEME weights (local path or Hugging Face download) and provides
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

# Make the package and this directory importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hyphaeon.model import PhyloAxialTransformer
from hyphaeon import dataset as ds

EXAMPLES_DIR = REPO_ROOT / "examples"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Weight resolution: use hyphaeon.weights module (Hugging Face download by
# default, AXOMEME_WEIGHTS env var for local .pt/.safetensors files).
# This mirrors the CLI's behavior: CI downloads from HF; local devs can
# point at a working checkpoint while iterating before pushing to HF.
from hyphaeon.weights import resolve_weights_path, load_weights, load_arch_config


# ---------------------------------------------------------------------------
# Weight resolution
# ---------------------------------------------------------------------------

def _resolve_weights():
    """Return (path, source_label) or None if weights are unavailable.

    Resolution order (matches CLI):
    1. AXOMEME_WEIGHTS env var → local file (if it exists)
    2. Default variant from Hugging Face (downloads + caches on first use)
    """
    explicit = os.environ.get("AXOMEME_WEIGHTS")
    if explicit and os.path.exists(explicit):
        return explicit, f"AXOMEME_WEIGHTS={explicit}"
    try:
        path = resolve_weights_path(weights=None)
        return path, f"Hugging Face (cached at {path})"
    except Exception as e:
        return None, (
            f"weights unavailable: {e}. Set AXOMEME_WEIGHTS to a local "
            f".pt/.safetensors checkpoint, or set HF_TOKEN for Hugging Face download."
        )


_WEIGHTS_PATH, _WEIGHTS_SOURCE = _resolve_weights()

WEIGHTS_AVAILABLE = _WEIGHTS_PATH is not None


def _load_model_from(path):
    """Load a checkpoint and return an eval-mode PhyloAxialTransformer.

    Uses hyphaeon.weights.load_arch_config for architecture config (handles
    both .pt and .safetensors) and load_weights for state_dict extraction.
    """
    config = load_arch_config(weights=path)
    state_dict = load_weights(weights=path, map_location="cpu")
    m = PhyloAxialTransformer(
        embed_dim=config["embed_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        window_size=config["window_size"],
    )
    m.load_state_dict(state_dict)
    m.eval()
    return m


@pytest.fixture(scope="session")
def weights_info():
    """Return (path, source_label); skip if weights unavailable."""
    if not WEIGHTS_AVAILABLE:
        pytest.skip(f"AxoMEME weights not available ({_WEIGHTS_SOURCE}). "
                    f"Set AXOMEME_WEIGHTS or HF_TOKEN to run model_eval tests.")
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


# Shared simulation config: (name, n_taxa, n_codons, tree_depth, scale, seed)
# Used by both sim_datasets (with model loading) and sim_alignments (without).
_SIM_CONFIGS = [
    ("sim_50_mod", 50, 100, 0.2, 1.0, 100),
    ("sim_100_deep", 100, 100, 0.5, 1.0, 200),
    ("sim_200_deep", 200, 100, 1.0, 1.0, 300),
]


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
    except (ValueError, FileNotFoundError, RuntimeError) as e:
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

    datasets = []
    for name, n_taxa, n_codons, depth, scale, seed in _SIM_CONFIGS:
        fa, nwk = simulate_neutral_alignment(
            n_taxa=n_taxa, n_codons=n_codons,
            tree_depth=depth, scale=scale, seed=seed)
        try:
            c, a, d, z, inv, taxa, L = load_tensors(fa, nwk)
        except (ValueError, FileNotFoundError, RuntimeError) as e:
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
def sim_alignments(seqgen_available):
    """Neutral simulated alignments only (no model loading).

    Lightweight version of sim_datasets for tests that only need
    alignments + trees (e.g. Mode I baseline tests). Does not load
    the neural model or compute LRTs.
    """
    from _sim import simulate_neutral_alignment

    alignments = []
    for name, n_taxa, n_codons, depth, scale, seed in _SIM_CONFIGS:
        fa, nwk = simulate_neutral_alignment(
            n_taxa=n_taxa, n_codons=n_codons,
            tree_depth=depth, scale=scale, seed=seed)
        # Extract taxa names from the FASTA
        from hyphaeon.dataset import parse_alignment_sequences
        seq_dict = parse_alignment_sequences(fa)
        taxa = list(seq_dict.keys())
        alignments.append({
            "name": name, "fa": fa, "nwk": nwk,
            "taxa": taxa, "n_taxa": len(taxa),
        })
    if not alignments:
        pytest.skip("No simulated alignments generated")
    return alignments


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


# ---------------------------------------------------------------------------
# Method-level fixtures — BUSTED, PhyloWAS, ESSM
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def weights_path(weights_info):
    """Return the resolved weights file path (skips if unavailable)."""
    return weights_info[0]


@pytest.fixture(scope="session")
def busted_runner(weights_path, tmp_path_factory):
    """Return a callable that runs BUSTED omnibus inference with weights.
    Returns the result record (dict for single alignment, list for batch).
    """
    from hyphaeon.cli import cmd_busted
    from argparse import Namespace
    import json

    def _run(alignment_path, tree_path=None, **kw):
        # Prevent tests from accidentally overwriting fixed CLI parameters
        reserved = {"weights", "variant", "cpu", "dir", "pattern",
                    "tree_suffix", "tree_dir", "no_prune_duplicates",
                    "output", "csv"}
        conflicts = reserved & set(kw)
        if conflicts:
            raise ValueError(f"busted_runner reserved params cannot be overridden: {conflicts}")
        out_json = str(tmp_path_factory.mktemp("busted") / "output.json")
        args = Namespace(
            alignment=alignment_path,
            tree=tree_path,
            weights=weights_path,
            variant=None,
            cpu=True,
            dir=None,
            pattern="*.aln,*.fa,*.fasta,*.nex,*.fna",
            tree_suffix=".raxml.bestTree",
            tree_dir=None,
            no_prune_duplicates=False,
            max_species=None,
            batch_size=None,
            output=out_json,
            csv=None,
            **kw
        )
        cmd_busted(args)
        with open(out_json) as f:
            data = json.load(f)
        # Single alignment returns a dict, batch returns a list
        if isinstance(data, list):
            return data
        return [data]
    return _run


@pytest.fixture(scope="session")
def phylowas_runner(weights_path):
    """Return a callable that runs run_phenotype_association with weights."""
    from hyphaeon.phenotype import run_phenotype_association

    def _run(alignment_path, tree_path=None, **kw):
        return run_phenotype_association(
            alignment_path=alignment_path,
            tree_path=tree_path,
            weights_path=weights_path,
            cpu=True,
            **kw
        )
    return _run


@pytest.fixture(scope="session")
def essm_runner(weights_path):
    """Return a callable that runs run_epistatic_sector_mining with weights."""
    from hyphaeon.epistasis import run_epistatic_sector_mining

    def _run(alignment_path, tree_path=None, **kw):
        return run_epistatic_sector_mining(
            alignment_path=alignment_path,
            tree_path=tree_path,
            weights_path=weights_path,
            cpu=True,
            **kw
        )
    return _run


# ---------------------------------------------------------------------------
# Session-scoped method results — shared across test files to avoid
# redundant neural inference runs on the same alignment.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def essm_smc6_result(essm_runner, smc6_paths):
    """Run ESSM once on Smc6 with default params. Shared across all test
    files that need default Smc6 epistasis results.

    Without this, ~10 tests each call essm_runner separately, resulting
    in redundant neural inference runs on the same alignment (~14s each).
    """
    fa, nwk = smc6_paths
    return essm_runner(fa, tree_path=nwk)


@pytest.fixture(scope="session")
def essm_smc6_max_fdr_1(essm_runner, smc6_paths):
    """Run ESSM on Smc6 with max_fdr=1.0 and no candidate filters.

    Returns ALL candidate edges with q-values. max_fdr is applied AFTER
    BH correction, so filtering client-side by fdr_q is equivalent to
    running with a lower max_fdr. The min_sim/min_lrt/min_shared filters
    also change the BH denominator, so setting them to permissive values
    here gives the widest candidate set — stricter filters can be applied
    client-side on the similarity/lrt/shared_taxa fields.

    Shared by test_epistasis_independence.py (saves 3 ESSM runs) and
    available to any test that needs the full edge list.
    """
    fa, nwk = smc6_paths
    return essm_runner(fa, tree_path=nwk, max_fdr=1.0,
                       min_sim=0.0, min_lrt=0.0, min_shared=1)


@pytest.fixture(scope="session")
def busted_smc6_result(busted_runner, smc6_paths):
    """Run BUSTED once on Smc6 and cache for all test classes/files.

    Without this, each test method calls busted_runner separately,
    resulting in redundant inference runs on the same alignment.
    """
    fa, nwk = smc6_paths
    return busted_runner(fa, tree_path=nwk)


@pytest.fixture(scope="session")
def sister_taxa_7_result(phylowas_runner, smc6_paths):
    """Run PhyloWAS on Smc6 with 7 sister-taxa foreground, shared across
    test files to avoid redundant neural inference runs.

    Used by TestSisterTaxaConfounding (test_mode_comparison_phenotype.py)
    and TestPhylogeneticConfounding (test_phenotype_outputs.py).
    """
    from _harness import get_taxa, fg_string
    fa, nwk = smc6_paths
    taxa = get_taxa(fa, nwk)
    return phylowas_runner(fa, tree_path=nwk, foreground=fg_string(taxa, 7))


@pytest.fixture(scope="session")
def busted_cross_dataset(busted_runner, smc6_paths, bat_oas1_paths, camelid_paths):
    """Run BUSTED on all three example datasets. Shared by
    TestBustedMultipleDatasets and TestBustedConcordance to avoid
    redundant neural BUSTED runs on bat_oas1 and camelid.
    Returns dict: {name: result_list}.
    """
    results = {}
    for name, paths in [("Smc6", smc6_paths),
                        ("bat_oas1", bat_oas1_paths),
                        ("camelid", camelid_paths)]:
        fa, nwk = paths
        results[name] = busted_runner(fa, tree_path=nwk)
    return results


# ---------------------------------------------------------------------------
# Mode I baseline fixtures — no weights, no model, pure numpy/scipy
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mode_i_phylowas_runner():
    """Return a callable that runs Mode I PhyloWAS (binary substitution + Poisson).
    No weights, no tree, no neural model. Phylogeny-blind by construction.
    """
    from _mode_i_baseline import run_phenotype_association_mode_i
    return run_phenotype_association_mode_i


@pytest.fixture(scope="session")
def mode_i_essm_runner():
    """Return a callable that runs Mode I ESSM (binary cosine + Poisson + cliques).
    No weights, no tree, no neural model. Phylogeny-blind by construction.
    """
    from _mode_i_baseline import run_epistatic_sector_mining_mode_i
    return run_epistatic_sector_mining_mode_i


@pytest.fixture(scope="module")
def injected_dataset(seqgen_available):
    """Simulate a neutral alignment and inject selection on a clade.

    Shared by TestTruePositiveDetection (phenotype) and
    TestEpistasisTruePositiveDetection (epistasis) to avoid duplicating
    ~35 lines of setup code.
    """
    from _sim import simulate_neutral_alignment, inject_selection
    n_taxa, n_codons, depth = 50, 100, 0.2
    fa, nwk = simulate_neutral_alignment(
        n_taxa=n_taxa, n_codons=n_codons, tree_depth=depth,
        seed=42, scale=1.0)
    fa_sel, selected_sites, n_selected_taxa, selected_taxa = inject_selection(
        fa, nwk, n_taxa, n_codons,
        n_selected_sites=10, n_selected_branches=15, seed=43)

    all_taxa = [f"taxon_{i}" for i in range(n_taxa)]

    return {
        "fa": fa_sel, "nwk": nwk,
        "selected_sites": selected_sites,
        "selected_1idx": {s + 1 for s in selected_sites},
        "n_selected_taxa": n_selected_taxa,
        "selected_taxa": selected_taxa,
        "taxa": all_taxa,
    }
