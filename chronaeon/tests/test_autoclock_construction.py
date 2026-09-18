"""Constructor + ingestion wiring tests for chronaeon.autoclock.

Net-new coverage relative to the existing chronaeon test suite: these focus on
construction/validation and metadata ingestion wiring (no full pipeline runs):
ValueError on missing input, in-memory records path, CSV/TSV date ingestion,
date_regex header extraction, and BEAST XML auto-detection by extension and
header sniff.
"""

from pathlib import Path

import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from chronaeon.autoclock import AutoClockDeconvolution


# ---------------------------------------------------------------------------
# Tiny synthetic data helpers
# ---------------------------------------------------------------------------
_ANCESTOR = "ATG" * 60  # 180 bp, codon-aligned


def _make_records(taxa):
    """Build a list of SeqRecord objects with trivial mutations per taxon."""
    recs = []
    for i, t in enumerate(taxa):
        seq = list(_ANCESTOR)
        # sprinkle a couple of deterministic point mutations
        for j in range(i % 5):
            pos = (7 * (i + 1) + 11 * j) % len(seq)
            seq[pos] = "ACGT"[(i + j) % 4]
        recs.append(SeqRecord(Seq("".join(seq)), id=t, description=""))
    return recs


def _write_fasta(path, taxa):
    lines = []
    for r in _make_records(taxa):
        lines.append(f">{r.id}\n{str(r.seq)}\n")
    Path(path).write_text("".join(lines))
    return str(path)


# ---------------------------------------------------------------------------
# 1. Constructor validation
# ---------------------------------------------------------------------------
def test_constructor_raises_without_any_input(tmp_path):
    """No alignment_path, no beast_path, no records -> ValueError."""
    with pytest.raises(ValueError):
        AutoClockDeconvolution(output_dir=str(tmp_path), quiet=True)


def test_missing_dates_source_raises_on_load(tmp_path):
    """A dates_source path that does not exist surfaces as an error at load time.

    Construction is permitted (path not touched yet); load_and_validate resolves
    dates via parse_sample_dates, which raises FileNotFoundError for a missing file.
    """
    taxa = [f"tx{i}" for i in range(6)]
    fasta = _write_fasta(tmp_path / "aln.fasta", taxa)

    engine = AutoClockDeconvolution(
        alignment_path=fasta,
        dates_source=str(tmp_path / "does_not_exist.csv"),
        date_col="date",
        strain_col="id",
        manifold="distance",
        output_dir=str(tmp_path / "out"),
        quiet=True,
    )
    with pytest.raises(FileNotFoundError):
        engine.load_and_validate()


# ---------------------------------------------------------------------------
# 2. In-memory records + dates_map construction and load_and_validate
# ---------------------------------------------------------------------------
def test_in_memory_records_construction(tmp_path):
    taxa = [f"tx{i}" for i in range(6)]
    recs = _make_records(taxa)
    dates_map = {t: 2000.0 + i for i, t in enumerate(taxa)}

    engine = AutoClockDeconvolution(
        records=recs,
        dates_map=dates_map,
        manifold="distance",
        output_dir=str(tmp_path),
        quiet=True,
    )
    # Wiring: records/taxa/seq_dict/dates_map populated at construction time.
    assert len(engine.records) == 6
    assert set(engine.taxa) == set(taxa)
    assert engine.seq_dict["tx0"] == str(recs[0].seq)
    assert engine.dates_map["tx3"] == 2003.0
    # output_dir is created eagerly.
    assert Path(engine.output_dir).is_dir()

    # load_and_validate short-circuits to the in-memory branch (no files).
    engine.load_and_validate()
    assert engine.meta_df is not None
    assert len(engine.meta_df) == 6
    assert set(engine.meta_df["id"]) == set(taxa)
    # meta_df carries the resolved numeric dates, aligned by id.
    md = engine.meta_df.set_index("id")
    for i, t in enumerate(taxa):
        assert float(md.loc[t, "date"]) == pytest.approx(2000.0 + i)


def test_in_memory_records_alone_is_allowed(tmp_path):
    """records provided (no alignment/beast) must NOT raise at construction."""
    taxa = [f"s{i}" for i in range(5)]
    engine = AutoClockDeconvolution(
        records=_make_records(taxa),
        manifold="distance",
        output_dir=str(tmp_path),
        quiet=True,
    )
    assert len(engine.records) == 5
    # dates_map defaults to empty dict when not supplied.
    assert engine.dates_map == {}


# ---------------------------------------------------------------------------
# 3. CSV date ingestion via date_col / strain_col
# ---------------------------------------------------------------------------
def test_csv_date_ingestion(tmp_path):
    taxa = [f"tx{i}" for i in range(6)]
    fasta = _write_fasta(tmp_path / "aln.fasta", taxa)

    csv = tmp_path / "dates.csv"
    # Include a supplementary column ("region") to verify it is carried into meta_df.
    csv.write_text(
        "strain,collection_date,region\n"
        + "".join(f"{t},{2000.0 + i},reg{i % 2}\n" for i, t in enumerate(taxa))
    )

    engine = AutoClockDeconvolution(
        alignment_path=fasta,
        dates_source=str(csv),
        date_col="collection_date",
        strain_col="strain",
        manifold="distance",
        output_dir=str(tmp_path / "out"),
        quiet=True,
    )
    engine.load_and_validate()
    assert len(engine.taxa) == 6
    for i, t in enumerate(taxa):
        assert engine.dates_map[t] == pytest.approx(2000.0 + i)
    # dates_source suffix recognized as csv.
    assert engine.dates_source.suffix.lower() == ".csv"

    # Supplementary metadata merge: extra CSV columns land in meta_df keyed by id.
    assert engine.meta_df is not None and len(engine.meta_df) == 6
    assert {"id", "date", "region"}.issubset(set(engine.meta_df.columns))
    md = engine.meta_df.set_index("id")
    for i, t in enumerate(taxa):
        assert md.loc[t, "region"] == f"reg{i % 2}"
        assert float(md.loc[t, "date"]) == pytest.approx(2000.0 + i)


# ---------------------------------------------------------------------------
# 4. TSV date ingestion
# ---------------------------------------------------------------------------
def test_tsv_date_ingestion(tmp_path):
    taxa = [f"tx{i}" for i in range(6)]
    fasta = _write_fasta(tmp_path / "aln.fasta", taxa)

    tsv = tmp_path / "dates.tsv"
    tsv.write_text(
        "id\tdate\n"
        + "".join(f"{t}\t{2010.0 + i}\n" for i, t in enumerate(taxa))
    )

    engine = AutoClockDeconvolution(
        alignment_path=fasta,
        dates_source=str(tsv),
        date_col="date",
        strain_col="id",
        manifold="distance",
        output_dir=str(tmp_path / "out"),
        quiet=True,
    )
    engine.load_and_validate()
    assert len(engine.taxa) == 6
    for i, t in enumerate(taxa):
        assert engine.dates_map[t] == pytest.approx(2010.0 + i)
    assert engine.dates_source.suffix.lower() == ".tsv"


def test_csv_date_ingestion_autodetect_columns(tmp_path):
    """No date_col/strain_col given: parser must auto-detect from known header names."""
    taxa = [f"tx{i}" for i in range(6)]
    fasta = _write_fasta(tmp_path / "aln.fasta", taxa)

    # "strain"/"date" are candidate columns; a decoy leading column must NOT be chosen.
    csv = tmp_path / "dates.csv"
    csv.write_text(
        "row_index,strain,date\n"
        + "".join(f"{i},{t},{2020.0 + i}\n" for i, t in enumerate(taxa))
    )

    engine = AutoClockDeconvolution(
        alignment_path=fasta,
        dates_source=str(csv),
        manifold="distance",
        output_dir=str(tmp_path / "out"),
        quiet=True,
    )
    engine.load_and_validate()
    assert len(engine.taxa) == 6
    # If "row_index" had been chosen as strain col, no taxon would map and
    # load_and_validate would raise; correct auto-detection yields real dates.
    for i, t in enumerate(taxa):
        assert engine.dates_map[t] == pytest.approx(2020.0 + i)


# ---------------------------------------------------------------------------
# 5. date_regex extraction from FASTA headers
# ---------------------------------------------------------------------------
def test_date_regex_header_extraction(tmp_path):
    # Custom marker "_DT<year>" that a caller supplies a regex for.
    taxa = [f"sample{i}_DT{2001 + i}" for i in range(6)]
    fasta = _write_fasta(tmp_path / "aln.fasta", taxa)

    engine = AutoClockDeconvolution(
        alignment_path=fasta,
        date_regex=r"_DT(\d{4})",
        manifold="distance",
        output_dir=str(tmp_path / "out"),
        quiet=True,
    )
    engine.load_and_validate()
    assert len(engine.taxa) == 6
    for i, t in enumerate(taxa):
        assert engine.dates_map[t] == pytest.approx(float(2001 + i))
    # No external dates file was used; source stays None.
    assert engine.dates_source is None
    assert engine.date_regex == r"_DT(\d{4})"


# ---------------------------------------------------------------------------
# 6. BEAST XML auto-detection by extension
# ---------------------------------------------------------------------------
def test_beast_autodetect_by_extension(tmp_path):
    # A .xml alignment path: dates_source auto-detected from the .xml extension.
    xml = tmp_path / "config.xml"
    xml.write_text("<beast></beast>\n")

    engine = AutoClockDeconvolution(
        alignment_path=str(xml),
        manifold="distance",
        output_dir=str(tmp_path / "out"),
        quiet=True,
    )
    assert engine.dates_source is not None
    assert str(engine.dates_source) == str(xml)


def test_beast_path_sets_alignment_and_dates_source(tmp_path):
    """Passing beast_path populates both alignment_path and dates_source."""
    xml = tmp_path / "run.xml"
    xml.write_text("<beast></beast>\n")

    engine = AutoClockDeconvolution(
        beast_path=str(xml),
        manifold="distance",
        output_dir=str(tmp_path / "out"),
        quiet=True,
    )
    assert str(engine.alignment_path) == str(xml)
    assert engine.dates_source is not None
    assert str(engine.dates_source) == str(xml)


# ---------------------------------------------------------------------------
# 7. BEAST XML auto-detection by header sniff (non-.xml extension)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("root_tag", ["<beast", "<alignment", "<data"])
def test_beast_autodetect_by_header_sniff(tmp_path, root_tag):
    # File has a non-xml extension but a BEAST-style XML header -> sniffed.
    path = tmp_path / "mystery.dat"
    path.write_text(f"{root_tag}>\n  <sequence/>\n</{root_tag.strip('<')}>\n")

    engine = AutoClockDeconvolution(
        alignment_path=str(path),
        manifold="distance",
        output_dir=str(tmp_path / "out"),
        quiet=True,
    )
    assert engine.dates_source is not None
    assert str(engine.dates_source) == str(path)


def test_plain_fasta_does_not_trigger_beast_detection(tmp_path):
    """A normal FASTA (non-xml, no xml header) leaves dates_source unset."""
    taxa = [f"tx{i}" for i in range(5)]
    fasta = _write_fasta(tmp_path / "plain.fasta", taxa)

    engine = AutoClockDeconvolution(
        alignment_path=fasta,
        manifold="distance",
        output_dir=str(tmp_path / "out"),
        quiet=True,
    )
    # No dates_source specified and header is not XML -> stays None.
    assert engine.dates_source is None
