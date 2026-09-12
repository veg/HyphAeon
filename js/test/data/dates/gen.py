"""
WHY THIS FILE EXISTS

Reference values for js/src/dates.js, produced by the reference functions themselves so the port is
measured against what the Python DOES rather than against a reading of its source. ../../../fixtures/
(scripts/gen_fixtures.py) covers dataset.py and the method modules; it has nothing for the date
layer, and PLAN-TEMPORAL phase 2 ports only the pure parsers, so this file is where their cases live.

The four functions under test are:

    parse_date_to_decimal        hyphaeon/temporal.py:73-151
    extract_date_from_string     hyphaeon/temporal.py:196-245
    parse_header_timestamp       hyphaeon/dating.py:317-364
    _parse_timestamp_flexible    hyphaeon/dating.py:367-384

HOW THEY ARE LOADED, and why not by `import hyphaeon.temporal`. Both modules import torch at module
scope (temporal.py:42, dating.py imports inference), and the date parsers need none of it. Importing
either would make this generator depend on a full model environment to produce four string tables.
So the four function definitions are lifted OUT of the checked-out source by `ast` — by name, with
their exact line ranges printed into the output so a reader can check them against the file — and
executed in a namespace holding only what their bodies reference (`re`, `datetime`, `np`, `pd`).
Nothing is retyped: a drift between the reference and the extracted body is impossible, and the
line ranges in the output are the audit trail.

Regenerate from js/ with a Python that has numpy and pandas (no torch needed):

    python test/data/dates/gen.py [--engine ../]

Output: parsers.json in this directory. NaN and infinities are written as the strings "NaN",
"Infinity", "-Infinity" (the fixtures/README.md convention).
"""

from __future__ import annotations

import argparse
import ast
import datetime  # noqa: F401  (the extracted bodies reference it)
import json
import math
import re  # noqa: F401  (the extracted bodies reference it)
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

FUNCTIONS = {
    "hyphaeon/temporal.py": ["parse_date_to_decimal", "extract_date_from_string"],
    "hyphaeon/dating.py": ["parse_header_timestamp", "_parse_timestamp_flexible"],
}


def load_reference(engine_root: Path):
    """Execute the four date functions out of the checked-out Python, without importing torch."""
    ns = {"re": re, "datetime": datetime, "np": np, "pd": pd, "Any": object}
    provenance = {}
    for rel, names in FUNCTIONS.items():
        path = engine_root / rel
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        by_name = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in names:
            node = by_name[name]
            segment = ast.get_source_segment(source, node)
            provenance[name] = f"{rel}:{node.lineno}-{node.end_lineno}"
            exec(compile(segment, str(path), "exec"), ns)  # noqa: S102
    return ns, provenance


def jsonable(v):
    if isinstance(v, (np.floating, float)):
        f = float(v)
        if math.isnan(f):
            return "NaN"
        if math.isinf(f):
            return "Infinity" if f > 0 else "-Infinity"
        return f
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    return v


# ---------------------------------------------------------------------------------------------
# The cases. Each one is (label, input, ...) and the reference's answer is computed, never typed.
# ---------------------------------------------------------------------------------------------

# parse_date_to_decimal, calendar. Grouped by the quirk each block pins.
CALENDAR_VALUES = [
    # Plain numbers and the [1800, 2100] gate (temporal.py:104-108, 115-120).
    2021.25, 1959.5, 1800, 2100, 1799.999, 2100.001, 0, -1, 1e9,
    True, False,
    # Missing / null tokens (temporal.py:86, 111).
    None, "", "   ", "unknown", "UNKNOWN", "nan", "NaN", "none", "NA", "na", "?", "??",
    # Decimal-year strings (temporal.py:115-120), including CPython float() spellings.
    "2021.25", " 2021.25 ", "2021", "2_021", "2.0212e3", "+2021", "inf", "-inf",
    # ISO and partial ISO (temporal.py:126-147) -- Q2 mid-year / mid-month imputation.
    "2021-04-15", "2021-04", "2021-01-01", "2021-12-31", "2020-02-29", "2021-02-28",
    "2021-XX-XX", "2021-04-XX", "2021-4-5", "2021-004-015",
    # Slash and dot delimiters (temporal.py:123) -- Q5.
    "2021/04/15", "2021.04.15", "2021.4.15", "2021/4", "2021.5", "2021.05",
    # The 28/30 day cap (temporal.py:144) -- Q3.
    "2021-01-31", "2021-03-31", "2021-02-29", "2020-02-29", "2021-02-30", "2021-12-30",
    # Invalid components discarded silently (temporal.py:133-142) -- Q4.
    "2021-13-15", "2021-00-15", "2021-04-00", "2021-04-32", "2021-04-31",
    # Out of gate and unparseable (temporal.py:129-130, 151) -- Q6.
    "1700-04-15", "2500-04-15", "0021-04-15", "21-04-15", "202-04-15", "20210415",
    "notadate", "2021-04-15T09:30:00Z", "15-04-2021", "April 2021",
    # Leap-year arithmetic across a century (temporal.py:146).
    "1900-03-01", "2000-03-01", "2004-03-01", "1996-12-31",
]

# parse_date_to_decimal, non-calendar (temporal.py:90-102) -- Q10.
NON_CALENDAR_VALUES = [
    0, 5000, 5000.5, -1, -0.0, "5000", "5000.5", "-5000", "gen_5000", "generation 42",
    "t=17.5", "abc", "", None, "nan", "inf", "-inf", "  12  ", 1e9, True,
]

# extract_date_from_string, calendar (temporal.py:217-243) -- Q7.
HEADER_NAMES_CALENDAR = [
    "A/USA/CA-1/2021|2021-05-14",
    "seq_2021-05-14",
    "seq 2021-05-14",
    "A/USA/2021-05-14/2021",
    "2021-05-14",
    "A-2021-05-14",
    "hCoV-19/England/X|2021.35",
    "hCoV-19/England/X|2021.3512",
    "hCoV-19/England/X|2021.5",
    "strain_2021-05",
    "strain|2021",
    "strain/2021",
    "2021",
    "strain_2021",
    "strain-2021",
    "EPI_ISL_402124|2019-12-30|China",
    "no_date_here",
    "",
    "X|9999-05-14",
    "X|1700-05-14",
    "X|2021-13-14",
]

# extract_date_from_string, non-calendar (temporal.py:204-215).
HEADER_NAMES_NON_CALENDAR = [
    "pop1_gen2000",
    "pop1|gen_5000",
    "pop1_20000gen",
    "pop1_g50000",
    "lineage|5000",
    "lineage_generation_120",
    "sample_day-7",
    "sample_d30",
    "sample_t12.5",
    "sample_D30",
    "clone7",
    "7clone",
    "no_number",
    "",
    "x_2021-05-14",
]

# parse_header_timestamp (dating.py:317-364). The 1959 block is Q1: every name here that contains
# Z59, ZR59 or 1959 is answered 1959.5 by the reference no matter what else it says.
HEADER_TIMESTAMP_NAMES = [
    "Z59ZR.ZHU",
    "ZR59",
    "AZ59012",
    "A/Kinshasa/1959-03-04",
    "X|1959-03-04",
    "X|2019-03-04",
    "B86US.SFMHS18",
    "F93BE_VI850",
    "C86ET.ETH2220",
    "A85UG.U455",
    "D84ZR.84ZR085",
    "B29US.X",
    "B30US.X",
    "b86us.SFMHS18",
    "Ref_B_FR_83_HXB2",
    "Ref_C_ET_86_ETH2220",
    "Ref_C_et_86_ETH2220",
    "patient1_16WPI",
    "patient1_16 WPI",
    "patient1_16wpi",
    "patient1_16Wpi",
    "patient1_120DPI",
    "patient1_120.5dpi",
    "seq_2021-05-14",
    "",
    "nothing_here",
]

# _parse_timestamp_flexible (dating.py:367-384).
FLEXIBLE_VALUES = [
    None, "", "unknown", "nan", "2021-04-15", "2021.25", 2021.25, "5000", 5000, 0.25, "0.25",
    "-3", -3, "1700", 1700, "inf", "abc", True, "1e6",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--engine",
        default=str(HERE.parents[3]),
        help="root of the HyphAeon checkout holding hyphaeon/temporal.py (default: this repository)",
    )
    args = ap.parse_args(argv)
    engine_root = Path(args.engine).resolve()

    ns, provenance = load_reference(engine_root)
    parse_date_to_decimal = ns["parse_date_to_decimal"]
    extract_date_from_string = ns["extract_date_from_string"]
    parse_header_timestamp = ns["parse_header_timestamp"]
    parse_timestamp_flexible = ns["_parse_timestamp_flexible"]

    payload = {
        "generated_by": "js/test/data/dates/gen.py",
        "source": provenance,
        "python": ".".join(str(x) for x in sys.version_info[:3]),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "parse_date_to_decimal_years": [
            {"input": v, "output": parse_date_to_decimal(v)} for v in CALENDAR_VALUES
        ],
        "parse_date_to_decimal_generations": [
            {"input": v, "output": parse_date_to_decimal(v, "generations")} for v in NON_CALENDAR_VALUES
        ],
        "parse_date_to_decimal_days": [
            {"input": v, "output": parse_date_to_decimal(v, "days")} for v in NON_CALENDAR_VALUES
        ],
        "parse_date_to_decimal_arbitrary": [
            {"input": v, "output": parse_date_to_decimal(v, "arbitrary")} for v in NON_CALENDAR_VALUES
        ],
        # An unrecognised units string takes the calendar path, because the reference's test is
        # `if time_units in ('generations', 'days', 'arbitrary')`.
        "parse_date_to_decimal_unknown_units": [
            {"input": v, "output": parse_date_to_decimal(v, "fortnights")}
            for v in ["2021-04-15", "5000", 5000]
        ],
        "extract_date_from_string_years": [
            {"input": v, "output": extract_date_from_string(v)} for v in HEADER_NAMES_CALENDAR
        ],
        "extract_date_from_string_generations": [
            {"input": v, "output": extract_date_from_string(v, "generations")}
            for v in HEADER_NAMES_NON_CALENDAR
        ],
        "parse_header_timestamp": [
            {"input": v, "output": parse_header_timestamp(v)} for v in HEADER_TIMESTAMP_NAMES
        ],
        # The same names with the 1959 block removed from the reference body, which is what
        # dates.js does by default (Q1). Reproduced here by calling the rest of the chain in the
        # reference's own order, not by a second implementation.
        "parse_header_timestamp_without_1959": [
            {"input": v, "output": _without_1959(v, ns)} for v in HEADER_TIMESTAMP_NAMES
        ],
        "parse_timestamp_flexible": [
            {"input": v, "output": parse_timestamp_flexible(v)} for v in FLEXIBLE_VALUES
        ],
    }

    path = HERE / "parsers.json"
    path.write_text(json.dumps(jsonable(payload), indent=1) + "\n")
    print(f"wrote {path.relative_to(HERE.parents[3])}")
    return 0


def _without_1959(name, ns):
    """
    parse_header_timestamp (dating.py:334-364) with step 1 skipped.

    The body below is steps 2-6 of the reference, in the reference's order, over the reference's own
    `extract_date_from_string`. It exists because dates.js makes the 1959 anchor opt-in (Q1) and the
    default therefore needs a reference of its own; it is the only hand-written parse here, and the
    test replays BOTH tables so the opt-in path stays pinned to the unmodified reference.
    """
    if not name:
        return np.nan
    d = ns["extract_date_from_string"](name)
    if not np.isnan(d):
        return d
    m_korber = re.search(r"^[A-Za-z](\d{2})[A-Za-z]{2}[._]", name)
    if m_korber:
        yr_short = int(m_korber.group(1))
        return float(1900 + yr_short if yr_short >= 30 else 2000 + yr_short) + 0.5
    m_pipe = re.search(r"_(?:[A-Z]{2})_(\d{2})_", name)
    if m_pipe:
        yr_short = int(m_pipe.group(1))
        return float(1900 + yr_short if yr_short >= 30 else 2000 + yr_short) + 0.5
    m_wpi = re.search(r"(\d+(?:\.\d+)?)\s*(?:WPI|wpi)", name)
    if m_wpi:
        return float(m_wpi.group(1))
    m_dpi = re.search(r"(\d+(?:\.\d+)?)\s*(?:DPI|dpi)", name)
    if m_dpi:
        return float(m_dpi.group(1))
    return np.nan


if __name__ == "__main__":
    raise SystemExit(main())
