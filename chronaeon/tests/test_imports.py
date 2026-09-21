"""Smoke test: verify all chronaeon submodules import cleanly."""
import chronaeon
from chronaeon.dating import run_mrca_dating, run_ols_dating
from chronaeon.autoclock import AutoClockDeconvolution, run_hierarchical_autoclock
from chronaeon.triage import ChronAeonSieve
from chronaeon.geo import run_phylogeography_analysis
from chronaeon.r0 import run_r0_analysis, compute_reproduction_numbers
from chronaeon.sketch import CanonicalMinHashSketcher, AlignmentFreeBinner
from chronaeon.alignment import ReferenceCodonAligner


def test_version():
    assert chronaeon.__version__ == "0.1.1"


def test_all_imports():
    assert run_mrca_dating is not None
    assert run_ols_dating is not None
    assert AutoClockDeconvolution is not None
    assert run_hierarchical_autoclock is not None
    assert ChronAeonSieve is not None
    assert run_phylogeography_analysis is not None
    assert run_r0_analysis is not None
    assert compute_reproduction_numbers is not None
    assert CanonicalMinHashSketcher is not None
    assert AlignmentFreeBinner is not None
    assert ReferenceCodonAligner is not None
