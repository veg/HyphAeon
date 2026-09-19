"""Anomaly triage tests for chronaeon.autoclock.

Net-new coverage: temporal-outlier flagging. Builds a tiny single-clock dataset,
injects one clear temporal outlier (dated recent but nearly identical to the
ancestor), then runs the flat AutoClockDeconvolution engine and inspects the
triage mechanism:
  - sus_count in the return dict is an int >= 0
  - the triage CSV exists with an `is_sus` column
  - every taxon gets a per-taxon triage row
  - if the outlier is strong enough, it is flagged (asserted loosely because
    clustering may absorb it into its own K=1 community).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from chronaeon.autoclock import AutoClockDeconvolution


# ----------------------------------------------------------------------------
# Synthetic data helpers
# ----------------------------------------------------------------------------
SEQ_LEN = 600
BASES = "ACGT"


def _ancestor(rng):
    return list(rng.choice(list(BASES), size=SEQ_LEN))


def _mutate(ancestor, n_mut, rng):
    """Return a copy of ancestor with n_mut point substitutions at distinct sites."""
    seq = list(ancestor)
    n_mut = min(n_mut, SEQ_LEN)
    if n_mut > 0:
        sites = rng.choice(SEQ_LEN, size=n_mut, replace=False)
        for s in sites:
            cur = seq[s]
            alts = [b for b in BASES if b != cur]
            seq[s] = alts[rng.integers(0, len(alts))]
    return "".join(seq)


def build_single_clock_with_outlier(seed=7, n=20, rate_muts_per_year=3.0):
    """Single molecular clock over ~20 taxa + one clear temporal outlier.

    Normal taxa: dated 2000..2009, divergence grows ~ rate * (date - 2000).
    Outlier ("OUTLIER_recent"): dated recent (2009) but nearly identical to the
    ancestor (0 mutations) -> far too little divergence for its date.
    """
    rng = np.random.default_rng(seed)
    ancestor = _ancestor(rng)

    records = []
    dates_map = {}
    t0 = 2000.0
    for i in range(n):
        date = t0 + (i % 10)
        n_mut = int(round(rate_muts_per_year * (date - t0)))
        n_mut = max(0, n_mut + int(rng.integers(-1, 2)))
        tid = f"taxon_{i:02d}"
        records.append(SeqRecord(Seq(_mutate(ancestor, n_mut, rng)), id=tid, description=""))
        dates_map[tid] = float(date)

    # Outlier: dated recent (2009) but identical to ancestor (near-zero divergence).
    out_id = "OUTLIER_recent"
    records.append(SeqRecord(Seq("".join(ancestor)), id=out_id, description=""))
    dates_map[out_id] = 2009.0

    return records, dates_map, out_id


def build_clean_single_clock(seed=11, n=20, rate_muts_per_year=3.0):
    """Single clock, NO outlier -- clean data used to check triage runs cleanly."""
    rng = np.random.default_rng(seed)
    ancestor = _ancestor(rng)
    records = []
    dates_map = {}
    t0 = 2000.0
    for i in range(n):
        date = t0 + (i % 10)
        n_mut = max(0, int(round(rate_muts_per_year * (date - t0))) + int(rng.integers(-1, 2)))
        tid = f"clean_{i:02d}"
        records.append(SeqRecord(Seq(_mutate(ancestor, n_mut, rng)), id=tid, description=""))
        dates_map[tid] = float(date)
    return records, dates_map


def _make_engine(records, dates_map, output_dir):
    return AutoClockDeconvolution(
        records=records,
        dates_map=dates_map,
        manifold="distance",
        max_k=6,
        min_cluster_size=5,
        random_state=42,
        output_dir=str(output_dir),
        quiet=True,
    )


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
def test_triage_return_dict_and_csv_wellformed(tmp_path):
    """run() returns a well-formed dict; sus_count is int>=0; triage CSV has is_sus."""
    records, dates_map, out_id = build_single_clock_with_outlier()
    engine = _make_engine(records, dates_map, tmp_path / "run1")
    res = engine.run(plot=False)

    for key in ("optimal_k", "communities", "sus_count", "triage_path"):
        assert key in res, f"missing key {key} in run() return dict"

    assert isinstance(res["optimal_k"], int) and res["optimal_k"] >= 1
    assert isinstance(res["sus_count"], int)
    assert res["sus_count"] >= 0

    triage_path = Path(res["triage_path"])
    assert triage_path.exists(), f"triage CSV not written to {triage_path}"
    tdf = pd.read_csv(triage_path)
    assert "is_sus" in tdf.columns
    # Documented column schema from triage_anomalies().
    for col in ("strain", "date", "clock_community", "divergence",
                "residual", "studentized_residual", "is_sus"):
        assert col in tdf.columns, f"triage CSV missing column {col}"

    # sus_count in dict matches the CSV's flagged count.
    assert res["sus_count"] == int(tdf["is_sus"].astype(bool).sum())


def test_every_taxon_has_a_triage_row(tmp_path):
    """Triage screens every taxon: one row per taxon, all finite numeric fields."""
    records, dates_map, out_id = build_single_clock_with_outlier()
    n_taxa = len(records)
    engine = _make_engine(records, dates_map, tmp_path / "run2")
    res = engine.run(plot=False)

    tdf = pd.read_csv(res["triage_path"])
    assert len(tdf) == n_taxa, f"expected {n_taxa} triage rows, got {len(tdf)}"
    strains = set(tdf["strain"].astype(str))
    expected = {r.id for r in records}
    assert strains == expected

    for col in ("date", "divergence", "residual", "studentized_residual"):
        vals = tdf[col].astype(float).values
        assert np.all(np.isfinite(vals)), f"non-finite values in {col}"

    # Membership: sum of per-community row counts equals n.
    per_comm = tdf.groupby("clock_community").size().sum()
    assert int(per_comm) == n_taxa


def test_outlier_is_flagged_or_triage_ran(tmp_path):
    """The injected outlier should be flagged if it lands in a multi-taxon
    community; if clustering isolates it (or absorbs it into a K=1 community
    whose residuals collapse), assert loosely that triage ran per-taxon."""
    records, dates_map, out_id = build_single_clock_with_outlier()
    engine = _make_engine(records, dates_map, tmp_path / "run3")
    res = engine.run(plot=False)

    tdf = pd.read_csv(res["triage_path"])
    out_rows = tdf[tdf["strain"].astype(str) == out_id]
    assert len(out_rows) == 1, "outlier missing from triage output"

    out_comm = int(out_rows["clock_community"].iloc[0])
    comm_size = int((tdf["clock_community"] == out_comm).sum())

    if res["sus_count"] >= 1:
        # If the outlier shares a community with >= 3 taxa, the strongest |z| is
        # expected to be the outlier itself.
        if comm_size >= 3:
            same_comm = tdf[tdf["clock_community"] == out_comm]
            max_abs_z_strain = str(
                same_comm.loc[same_comm["studentized_residual"].abs().idxmax(), "strain"]
            )
            assert max_abs_z_strain == out_id, (
                "outlier was not the most extreme residual in its community"
            )
            assert bool(out_rows["is_sus"].iloc[0]), (
                "outlier is the extreme residual but was not flagged is_sus"
            )
            # A recent sample ~identical to the ancestor sits BELOW the clock line.
            assert float(out_rows["studentized_residual"].iloc[0]) < 0, (
                "under-diverged recent outlier should have a negative residual"
            )
    else:
        # Clustering absorbed the outlier; verify triage still produced a
        # complete per-taxon screen (loose fallback).
        assert len(tdf) == len(records)
        assert "is_sus" in tdf.columns


def test_clean_dataset_triage_no_false_alarm(tmp_path):
    """A clean single-clock dataset (no injected outlier) should not raise, and
    triage should produce a per-taxon screen with a small sus_count."""
    records, dates_map = build_clean_single_clock()
    n_taxa = len(records)
    engine = _make_engine(records, dates_map, tmp_path / "run4")
    res = engine.run(plot=False)

    assert isinstance(res["sus_count"], int) and res["sus_count"] >= 0
    tdf = pd.read_csv(res["triage_path"])
    assert len(tdf) == n_taxa
    assert "is_sus" in tdf.columns
    # Clean single clock: a correct |z| > 3 screen flags essentially nothing.
    assert res["sus_count"] <= 2, (
        f"clean single clock produced {res['sus_count']} SUS flags "
        f"(false-alarm regression); expected ~0"
    )
    assert tdf["studentized_residual"].abs().max() < 6.0


def test_triage_df_attribute_matches_csv(tmp_path):
    """Engine's in-memory triage_df matches the written CSV and the summary JSON
    sus_count field."""
    import json

    records, dates_map, out_id = build_single_clock_with_outlier()
    engine = _make_engine(records, dates_map, tmp_path / "run5")
    res = engine.run(plot=False)

    assert engine.triage_df is not None
    assert len(engine.triage_df) == len(records)
    csv_count = int(pd.read_csv(res["triage_path"])["is_sus"].astype(bool).sum())
    df_count = int(engine.triage_df["is_sus"].astype(bool).sum())
    assert csv_count == df_count == res["sus_count"]

    summary_path = Path(res["summary_path"])
    assert summary_path.exists()
    with open(summary_path) as f:
        summary = json.load(f)
    assert int(summary["sus_count"]) == res["sus_count"]


def test_global_safety_net_columns_present(tmp_path):
    """Issue #58 fix: triage now also fits a GLOBAL clock so an outlier isolated into a
    tiny community can't escape via a collapsed within-community residual. The CSV must
    carry the global residual + reason columns, and community_size."""
    records, dates_map, out_id = build_single_clock_with_outlier()
    engine = _make_engine(records, dates_map, tmp_path / "gsn")
    res = engine.run(plot=False)
    tdf = pd.read_csv(res["triage_path"])
    for col in ("community_size", "global_studentized_residual", "sus_reason"):
        assert col in tdf.columns, f"triage CSV missing #58 column {col}"
    # Any SUS row must attribute a reason; global-only flags are now possible.
    flagged = tdf[tdf["is_sus"].astype(bool)]
    assert (flagged["sus_reason"].astype(str).str.len() > 0).all()


def test_outlier_flagged_even_when_isolated(tmp_path):
    """Issue #58: an outlier that clustering splits into a singleton/tiny community used to
    escape triage (its within-community residual collapses to ~0). The global safety net
    should still flag it. We assert the strong outlier is caught regardless of how it clusters."""
    records, dates_map, out_id = build_single_clock_with_outlier(seed=7, n=24)
    engine = _make_engine(records, dates_map, tmp_path / "iso")
    res = engine.run(plot=False)
    tdf = pd.read_csv(res["triage_path"])
    row = tdf[tdf["strain"] == out_id]
    assert len(row) == 1, f"{out_id} missing from triage table"
    # It is flagged, and — because its community may be tiny — via the global test at least.
    assert bool(row["is_sus"].iloc[0]), "isolated temporal outlier escaped triage (issue #58)"
    assert abs(float(row["global_studentized_residual"].iloc[0])) > 3.0
