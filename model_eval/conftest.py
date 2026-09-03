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

import pytest
import torch

# Make the package and this directory importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
            f".pt/.safetensors checkpoint, or set HF_TOKEN for Hugging Face download."
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
                    f"Set HYPHAEON_WEIGHTS or HF_TOKEN to run model_eval tests.")
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
# BUSTED fixtures
# ---------------------------------------------------------------------------

def _load_busted_head(weights_path):
    """Load the BUSTED head from a checkpoint, or None if absent."""
    from hyphaeon.weights import load_weights, load_arch_config
    from hyphaeon.model import BustedMultiTaskHead
    state_dict = load_weights(weights=weights_path)
    arch_config = load_arch_config(weights_path)
    head = BustedMultiTaskHead(embed_dim=arch_config["embed_dim"])
    busted_dict = {k.replace("head_busted.", ""): v
                   for k, v in state_dict.items() if k.startswith("head_busted.")}
    if busted_dict:
        head.load_state_dict(busted_dict, strict=False)
    head.eval()
    return head


@pytest.fixture(scope="session")
def busted_head(weights_info):
    """Session-scoped BUSTED head (loaded from checkpoint, or random init)."""
    return _load_busted_head(weights_info[0])


def _run_busted_on(model, busted_head, paths):
    """Run BUSTED on a single dataset; return [record] or skip on failure."""
    from _harness import run_busted
    fa, nwk = paths
    try:
        record = run_busted(model, fa, nwk, busted_head=busted_head)
        return [record]
    except Exception as e:
        print(f"  [WARNING] BUSTED failed on {os.path.basename(fa)}: "
              f"{type(e).__name__}: {e}")
        return None


@pytest.fixture(scope="session")
def busted_smc6_result(model, busted_head, smc6_paths):
    """BUSTED result for Smc6 (20 taxa, shallow tree)."""
    result = _run_busted_on(model, busted_head, smc6_paths)
    if result is None:
        pytest.skip("BUSTED failed on Smc6")
    return result


@pytest.fixture(scope="session")
def busted_cross_dataset(model, busted_head, smc6_paths, bat_oas1_paths,
                         camelid_paths):
    """BUSTED results for all real datasets that load successfully.

    Returns a dict: {name: [record]} (list to match cmd_busted's batch format).
    """
    results = {}
    for name, paths in [("Smc6", smc6_paths),
                        ("bat_oas1", bat_oas1_paths),
                        ("camelid", camelid_paths)]:
        record = _run_busted_on(model, busted_head, paths)
        if record is not None:
            results[name] = record
    if not results:
        pytest.skip("No datasets produced BUSTED results")
    return results


@pytest.fixture(scope="session")
def busted_runner(model, busted_head):
    """Functional fixture: run BUSTED on a given (fa, tree_path) pair.

    Returns a callable ``runner(fa_path, tree_path=...)`` that delegates to
    _harness.run_busted. Used by error-handling tests that pass bad inputs.
    """
    from _harness import run_busted

    def runner(fa_path, tree_path=None):
        return run_busted(model, fa_path, tree_path, busted_head=busted_head)

    return runner


# ---------------------------------------------------------------------------
# PhyloWAS (phenotype association) fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def phylowas_runner(weights_info):
    """Functional fixture: run Mode II PhyloWAS on a given alignment.

    Returns a callable ``runner(alignment_path, tree_path=..., **kw)`` that
    delegates to ``hyphaeon.phenotype.run_phenotype_association``. Each call
    re-loads the model from the session weights path (the function does this
    internally); the weights resolution is session-scoped.
    """
    from hyphaeon.phenotype import run_phenotype_association

    def runner(alignment_path, tree_path=None, **kwargs):
        return run_phenotype_association(
            alignment_path=alignment_path,
            tree_path=tree_path,
            weights_path=weights_info[0],
            progress=False,
            **kwargs,
        )

    return runner


@pytest.fixture(scope="session")
def sister_taxa_7_result(phylowas_runner, smc6_paths):
    """Mode II PhyloWAS on Smc6 with 7 sister-taxa foreground.

    Session-scoped so the expensive neural inference runs once and is reused
    by both TestPhylogeneticConfounding and TestSisterTaxaConfounding.
    """
    from _harness import get_taxa, fg_string

    fa, nwk = smc6_paths
    taxa = get_taxa(fa, nwk)
    return phylowas_runner(fa, tree_path=nwk, foreground=fg_string(taxa, 7))


# ---------------------------------------------------------------------------
# ESSM (epistatic sector mining) fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def essm_runner(weights_info):
    """Functional fixture: run Mode II ESSM on a given alignment.

    Returns a callable ``runner(alignment_path, tree_path=..., **kw)`` that
    delegates to ``hyphaeon.epistasis.run_epistatic_analysis``. Each call
    re-loads the model from the session weights path.
    """
    from hyphaeon.epistasis import run_epistatic_analysis

    def runner(alignment_path, tree_path=None, **kwargs):
        return run_epistatic_analysis(
            alignment_path=alignment_path,
            tree_path=tree_path,
            weights_path=weights_info[0],
            progress=False,
            **kwargs,
        )

    return runner


@pytest.fixture(scope="session")
def essm_smc6_result(essm_runner, smc6_paths):
    """Mode II ESSM on Smc6 with default parameters.

    Session-scoped so the expensive neural inference + DMS runs once and is
    reused by all TestCoSelectionNetwork / TestSectors / TestDMS tests and
    the mode-comparison epistasis tests.
    """
    fa, nwk = smc6_paths
    return essm_runner(fa, tree_path=nwk)


@pytest.fixture(scope="session")
def essm_smc6_max_fdr_1(essm_runner, smc6_paths):
    """Mode II ESSM on Smc6 with max_fdr=1.0, min_sim=0.0, min_lrt=0.0.

    Relaxed filters so all candidate edges are retained — used by the
    epistasis independence tests to verify filter behavior.
    """
    fa, nwk = smc6_paths
    return essm_runner(fa, tree_path=nwk, max_fdr=1.0, min_sim=0.0,
                       min_lrt=0.0, min_shared=1, min_cesi=0.0)


# ---------------------------------------------------------------------------
# Mode I baseline fixtures (phylogeny-blind, no weights required)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mode_i_phylowas_runner():
    """Functional fixture: run Mode I PhyloWAS on a given alignment.

    Returns a callable ``runner(alignment_path, **kw)`` that delegates to
    ``_mode_i_baseline.run_phenotype_association_mode_i``. No model or
    weights needed — Mode I is pure binary substitution counting.
    """
    from _mode_i_baseline import run_phenotype_association_mode_i

    def runner(alignment_path, **kwargs):
        return run_phenotype_association_mode_i(alignment_path, **kwargs)

    return runner


@pytest.fixture(scope="session")
def mode_i_essm_runner():
    """Functional fixture: run Mode I ESSM on a given alignment.

    Returns a callable ``runner(alignment_path, **kw)`` that delegates to
    ``_mode_i_baseline.run_epistatic_sector_mining_mode_i``. No model or
    weights needed.
    """
    from _mode_i_baseline import run_epistatic_sector_mining_mode_i

    def runner(alignment_path, **kwargs):
        return run_epistatic_sector_mining_mode_i(alignment_path, **kwargs)

    return runner


# ---------------------------------------------------------------------------
# Simulated data fixtures for mode-comparison tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sim_alignments(sim_datasets):
    """Alias for sim_datasets, used by mode-comparison tests.

    The mode-comparison tests were written against a ``sim_alignments``
    fixture name; ``sim_datasets`` provides the same data (list of dicts
    with ``fa``, ``nwk``, ``taxa`` keys).
    """
    return sim_datasets


@pytest.fixture(scope="session")
def injected_dataset(seqgen_available):
    """Simulated alignment with injected positive selection on a clade.

    Uses ``_sim.inject_selection`` to create an alignment with radical AA
    changes on a clade of taxa. Returns a dict with ``fa``, ``nwk``,
    ``taxa``, ``selected_taxa``, ``selected_sites`` (0-based), and
    ``selected_1idx`` (1-based set) keys.
    """
    from _sim import simulate_neutral_alignment, inject_selection, purge_stop_codons
    from hyphaeon.dataset import parse_alignment_sequences

    # Simulate a neutral alignment with low mutation rate so the background
    # is nearly constant at injected sites, purge stop codons (seq-gen
    # produces them under raw nucleotide substitution; real coding sequences
    # never have them and Mode I treats them as gaps, corrupting the
    # analysis), then inject radical AA changes on a clade of 7 taxa.
    # With 7/30 fg taxa and a low-mutation background, the Poisson test has
    # enough power to detect the injected shared substitutions.
    sim_fa, sim_nwk = simulate_neutral_alignment(
        n_taxa=30, n_codons=100, tree_depth=0.05, scale=0.1, seed=42)
    sim_fa, _ = purge_stop_codons(sim_fa, seed=42)
    mod_fa, selected_sites, n_sel_taxa, selected_taxa = inject_selection(
        sim_fa, sim_nwk, n_taxa=30, n_codons=100,
        n_selected_sites=10, n_selected_branches=7, seed=42)
    taxa = list(parse_alignment_sequences(mod_fa).keys())
    return {
        "fa": mod_fa,
        "nwk": sim_nwk,
        "taxa": taxa,
        "selected_taxa": selected_taxa,
        "selected_sites": selected_sites,
        "selected_1idx": {s + 1 for s in selected_sites},
    }


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
