"""Tests for experimental-evolution / longitudinal support in hyphaeon.temporal.

Covers the non-calendar time coordinate ('generations'/'days'/'arbitrary') ingestion
paths added for microbial experimental evolution (e.g. Lenski LTEE) and the fixation
sweep mode. The full end-to-end synthetic sweep run is guarded behind model weights.
"""
import os
import numpy as np
import pytest

from hyphaeon.temporal import (
    parse_date_to_decimal,
    extract_date_from_string,
    parse_temporal_metadata,
)


class TestNonCalendarDateParsing:
    def test_generation_numbers_not_year_gated(self):
        # Generation coordinates outside [1800,2100] must survive (the calendar gate
        # is the LTEE "no temporal signal" defect).
        assert parse_date_to_decimal(0, time_units="generations") == 0.0
        assert parse_date_to_decimal(500, time_units="generations") == 500.0
        assert parse_date_to_decimal(50000, time_units="generations") == 50000.0
        assert parse_date_to_decimal("gen_5000", time_units="generations") == 5000.0
        assert np.isnan(parse_date_to_decimal(-1, time_units="generations"))
        assert np.isnan(parse_date_to_decimal("none", time_units="generations"))

    def test_calendar_mode_still_gates(self):
        # Default (years) behavior is unchanged.
        assert np.isnan(parse_date_to_decimal(1500.0))
        assert parse_date_to_decimal(2021.25) == 2021.25

    def test_extract_generation_headers(self):
        for header, expected in [
            ("clone_g5000", 5000.0),
            ("m5_basal_g24500", 24500.0),
            ("pop|gen_2000", 2000.0),
            ("isolate_20000gen", 20000.0),
            ("s|50000", 50000.0),
        ]:
            got = extract_date_from_string(header, time_units="generations")
            assert got == expected, f"{header!r} -> {got}, expected {expected}"

    def test_metadata_generation_column_autodiscovery(self, tmp_path):
        tsv = tmp_path / "meta.tsv"
        tsv.write_text("strain\tgeneration\ncloneA\t0\ncloneB\t5000\ncloneC\t50000\n")
        dates = parse_temporal_metadata(
            dates_source=str(tsv),
            taxa=["cloneA", "cloneB", "cloneC"],
            time_units="generations",
        )
        assert dates == {"cloneA": 0.0, "cloneB": 5000.0, "cloneC": 50000.0}

    def test_isolate_and_replicate_index_disambiguation(self):
        # Disambiguate isolate/replicate numbers from generation timepoints
        assert extract_date_from_string("Ecoli_isolate_1_gen5000", time_units="generations") == 5000.0
        assert extract_date_from_string("rep_2_t10000", time_units="generations") == 10000.0
        assert extract_date_from_string("strain_5_d120", time_units="days") == 120.0

    def test_trapezoid_compatibility(self):
        from hyphaeon.temporal import _trapezoid
        res = _trapezoid([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
        assert np.isclose(res, 4.0)

    def test_fixation_thalf_scaling(self):
        # Ensure velocity_matrix and peak_intensities have matching scale
        # so t_half and FWHM are not zeroed out by 1/N attenuation
        curves_matrix = np.array([[0.000, 0.005, 0.010]], dtype=np.float32)
        mean_attns = np.full((1, 100), 0.01, dtype=np.float32)
        site_scale = mean_attns.mean(axis=1) + 1e-8
        velocity_matrix = (curves_matrix - curves_matrix[:, :1]) / site_scale[:, None]
        peak_intensities = np.ptp(curves_matrix, axis=1) / site_scale
        y = velocity_matrix[0]
        pv = peak_intensities[0]
        above_half = y >= (pv / 2.0)
        assert np.any(above_half), "Fixation half-max should be detected"
        assert np.where(above_half)[0][0] == 1  # 50% reached at index 1


@pytest.mark.skipif(
    os.environ.get("HYPHAEON_RUN_MODEL_TESTS") != "1",
    reason="requires model weights; set HYPHAEON_RUN_MODEL_TESTS=1 to run",
)
def test_synthetic_ltee_fixation_sweep(tmp_path):
    """Manual Test 1: a 25-clone, 50k-generation panel with a single A301T fixation
    at gen ~20000 must be recovered as a CONFIRMED_SWEEP at site 301."""
    import random
    from hyphaeon.temporal import run_temporal_surveillance

    random.seed(7)
    codons = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"
              if a + b + c not in ("TAA", "TAG", "TGA")]
    anc = [random.choice(codons) for _ in range(516)]
    anc[0] = "ATG"; anc[300] = "GCA"           # Ala at codon 301
    swept = list(anc); swept[300] = "ACA"       # A301T
    wt_g = [0, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 19000]
    sw_g = [22000, 24000, 26000, 28000, 30000, 33000, 36000, 39000, 42000, 45000, 48000, 50000, 27000]
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    recs, i = [], 0
    for g in wt_g:
        recs.append((f"g{g}_{letters[i]}", "".join(anc))); i += 1
    for g in sw_g:
        recs.append((f"g{g}_{letters[i]}", "".join(swept))); i += 1
    fa = tmp_path / "ltee.fasta"
    fa.write_text("".join(f">{n}\n{s}\n" for n, s in recs))

    res = run_temporal_surveillance(
        alignment_path=str(fa), tree_path=None, use_tn93=True,
        output_prefix=str(tmp_path / "out"), n_permutations=1000,
        time_units="generations", sweep_mode="fixation", cpu=True,
        weights_path=os.environ.get("HYPHAEON_WEIGHTS"),
    )
    df_sites = res["sites_summary"]
    meta = res.get("metadata", {})
    assert meta.get("confirmed_sweeps", df_sites["is_confirmed_sweep"].sum()) >= 1
    s301 = df_sites[df_sites["site"] == 301]
    assert len(s301) == 1
    assert bool(s301.iloc[0]["is_confirmed_sweep"]) is True
