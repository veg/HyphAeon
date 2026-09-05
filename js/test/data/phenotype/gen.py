"""
WHY THIS FILE EXISTS

Reference values for js/src/phenotype.js and js/src/permulations.js that fixtures/phenotype/
does not carry, produced by the libraries hyphaeon/phenotype.py actually calls — CPython's
`fnmatch`, `re` and `repr`, pandas' `read_csv` dtype inference, and Biopython's `Bio.Phylo` —
so the JavaScript ports are measured against what those libraries DO rather than against a
reading of their source.

fixtures/phenotype/ covers the three functions on the bundled examples and on two metadata
tables: eight preset / foreground / CSV cases of `resolve_phenotype_vector`, `compute_
phylogenetic_covariance` on Smc6.nwk and bat_oas1.nwk, and `generate_permulations` at B = 200.
This file is where the EDGE cases live — the ones a port gets wrong silently:

  fnmatch          fnmatch.fnmatch(name, pat) over globs with *, ?, character classes, negated
                   classes, an unterminated '[', an empty class and regex metacharacters that
                   fnmatch treats as literals. resolve_phenotype_vector calls it on every
                   (taxon, pattern) pair of both the preset and the inline-foreground branch.
  repr_list        repr(list[str]) — what f"...{fg_list}" interpolates into
                   phenotype_meta.description. Includes apostrophes and quotes, where CPython
                   switches the quoting character.
  float_str        str(float) across the bands where Python and JavaScript disagree on when to
                   use exponent notation (Python: >= 1e16 or < 1e-4; JavaScript: >= 1e21 or
                   < 1e-6). A trait column pandas typed float64 is compared as str(val).
  read_csv         pandas' per-column dtype inference and the str(val) it implies, on tables
                   that mix integers, floats, strings and NA — the difference between a trait
                   column that matches "1" and one that renders "1.0" and matches nothing.
  resolve          resolve_phenotype_vector on synthetic taxa: the error texts, the substring
                   fallback, the echolocation preset's bare-name substring path, an all-numeric
                   trait column, a continuous column with an unparseable cell, and the proof
                   that `background` changes nothing.
  find_any         Bio.Phylo tree.find_any(name=...) where the name is a regex to Biopython:
                   a '.' that matches any character, an alternation, and a name that is a
                   prefix of another. compute_phylogenetic_covariance calls it per taxon.
  covariance       compute_phylogenetic_covariance on small trees: a taxon absent from the
                   tree (diagonal 1.0), a name with a regex metacharacter, an internal node
                   name, a zero-length branch, and a missing branch length.
  permulations     generate_permulations' DETERMINISTIC properties on a small tree: the binary
                   row sums, the continuous row multisets, the all-zero y that is not binary,
                   and the Cholesky failure on a zero-length tree.

It is NOT a fixture in the fixtures/README.md sense: it lives with the tests, and
js/test/phenotype.test.js / js/test/permulations.test.js read its output.

Regenerate from js/ with the reference environment (the venv that has hyphaeon installed):

    HYPHAEON_WEIGHTS=../model.safetensors HF_HUB_OFFLINE=1 python test/data/phenotype/gen.py

Output: python_quirks.json in this directory. No model weights are loaded (nothing here calls
run_phenotype_association, which is covered end to end by fixtures/e2e/phenotype_RHO_*.json).
"""

import fnmatch
import io
import json
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd
from Bio import Phylo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from hyphaeon.phenotype import (  # noqa: E402
    PRESETS,
    compute_phylogenetic_covariance,
    generate_permulations,
    resolve_phenotype_vector,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "python_quirks.json")


def tree_from(newick):
    return Phylo.read(io.StringIO(newick), "newick")


# --------------------------------------------------------------------------------------------
# fnmatch
# --------------------------------------------------------------------------------------------
FNMATCH_CASES = [
    ("turtru", "turtru"),
    ("turtru", "tur*"),
    ("turtru", "*tru"),
    ("turtru", "*urt*"),
    ("turtru", "t?rtru"),
    ("turtru", "t??tru"),
    ("turtru", "tur**"),
    ("turtru", "[tu]urtru"),
    ("turtru", "[!x]urtru"),
    ("turtru", "[]urtru"),
    ("turtru", "[!]urtru"),
    ("turtru", "[urtru"),
    ("tur.tru", "tur.tru"),
    ("turXtru", "tur.tru"),
    ("tur+tru", "tur+tru"),
    ("a(b)c", "a(b)c"),
    ("a|b", "a|b"),
    ("a", "a|b"),
    ("enhlutken", "enhlut*"),
    ("enhlut", "enhlut*"),
    ("xenhlutken", "enhlut*"),
    ("mus.eve", "mus?eve"),
    ("", "*"),
    ("", ""),
    ("abc", ""),
    ("a\\b", "a\\b"),
    ("ab", "a[\\]b"),
]

fnmatch_out = [{"name": n, "pattern": p, "match": bool(fnmatch.fnmatch(n, p))} for n, p in FNMATCH_CASES]

# --------------------------------------------------------------------------------------------
# repr(list[str]) and str(float)
# --------------------------------------------------------------------------------------------
REPR_LISTS = [
    [],
    ["turTru"],
    ["turTru", "balMus", "balPhys"],
    ["^M_.*", "R_sin"],
    ["it's"],
    ['say "hi"'],
    ["""both ' and \""""],
    ["back\\slash"],
    ["tab\there"],
    ["new\nline"],
]
repr_out = [{"items": xs, "repr": repr(xs)} for xs in REPR_LISTS]

FLOATS = [
    0.0, -0.0, 1.0, -1.0, 62.0, 0.35, 8.0, 1e-3, 1e-4, 9.9e-5, 1e-5, 1e-6, 1e-7,
    1234567890123456.0, 1e16, 1.5e16, 1e21, 1e22, 123.456, 1 / 3, 2.5e-8, -4.75e17,
]
float_out = [{"value": v, "str": str(v)} for v in FLOATS]

# --------------------------------------------------------------------------------------------
# pandas read_csv dtype inference and str(val)
# --------------------------------------------------------------------------------------------
CSV_CASES = [
    (
        "int_trait_column",
        ",",
        "species,trait\nhomSap,1\npanTro,0\nmacMul,1\n",
    ),
    (
        "float_trait_column",
        ",",
        "species,trait\nhomSap,1.0\npanTro,0.0\nmacMul,1.0\n",
    ),
    (
        "int_with_missing_becomes_float",
        ",",
        "species,trait\nhomSap,1\npanTro,\nmacMul,1\n",
    ),
    (
        "mixed_object_column",
        ",",
        "species,trait\nhomSap,yes\npanTro,1\nmacMul,no\n",
    ),
    (
        "na_strings",
        ",",
        "species,trait\nhomSap,NA\npanTro,nan\nmacMul,NULL\ncalJac,None\nsaiBol,N/A\n",
    ),
    (
        "quoted_fields",
        ",",
        'species,note\nhomSap,"a, b"\npanTro,"say ""hi"""\n',
    ),
    (
        "tab_separated",
        "\t",
        "taxon\ttrait\nhomSap\t1\npanTro\t0\n",
    ),
    (
        "blank_lines_skipped",
        ",",
        "species,trait\nhomSap,1\n\npanTro,0\n",
    ),
    (
        "trailing_spaces_in_values",
        ",",
        "species,trait\n homSap ,  yes  \npanTro,no\n",
    ),
]

csv_out = []
for name, sep, text in CSV_CASES:
    df = pd.read_csv(io.StringIO(text), sep=sep)
    rows = []
    for _, row in df.iterrows():
        cells = {}
        for col in df.columns:
            v = row[col]
            cells[col] = {"isna": bool(pd.isna(v)), "str": None if pd.isna(v) else str(v)}
        rows.append(cells)
    csv_out.append({
        "name": name,
        "sep": sep,
        "text": text,
        "columns": list(df.columns),
        "dtypes": [str(d) for d in df.dtypes],
        "rows": rows,
    })

# --------------------------------------------------------------------------------------------
# resolve_phenotype_vector edge cases
# --------------------------------------------------------------------------------------------
SMALL_TAXA = ["homSap_293T", "hg18", "panTro4", "panPan", "macMul", "aotTri_OMK", "turTru", "mus.eve"]

TMP = tempfile.mkdtemp(prefix="hyphaeon-phenotype-gen-")


def write_tmp(name, text):
    path = os.path.join(TMP, name)
    with open(path, "w") as fh:
        fh.write(text)
    return path


RESOLVE_CASES = [
    ("preset_echolocation_substring", SMALL_TAXA, {"preset": "echolocation"}),
    ("preset_key_normalised", SMALL_TAXA, {"preset": "  HIGH-ALTITUDE "}),
    ("preset_dim_light_star_both_ends", ["Xcavefishy", "dolphin", "nothing"], {"preset": "dim_light"}),
    ("foreground_single_name", SMALL_TAXA, {"foreground": "turTru"}),
    ("foreground_list_with_spaces", SMALL_TAXA, {"foreground": " panTro4 , panPan "}),
    ("foreground_dot_is_regex", SMALL_TAXA, {"foreground": "mus.eve"}),
    ("foreground_invalid_regex_falls_back", SMALL_TAXA, {"foreground": "hg18["}),
    ("foreground_case_insensitive", SMALL_TAXA, {"foreground": "HOMSAP_293T"}),
    ("foreground_pipe_and_comma_uses_comma", SMALL_TAXA, {"foreground": "panTro4|panPan,macMul"}),
    ("foreground_dotstar_both_ends", SMALL_TAXA, {"foreground": ".*Tro4.*"}),
    ("background_is_ignored", SMALL_TAXA, {"foreground": "turTru", "background": "panTro4"}),
    (
        "csv_int_trait_matches",
        ["homSap", "panTro", "macMul"],
        {"phenotype_file": write_tmp("int.csv", "species,trait\nhomSap,1\npanTro,0\nmacMul,1\n")},
    ),
    (
        "csv_float_trait_matches_nothing",
        ["homSap", "panTro", "macMul"],
        {"phenotype_file": write_tmp("float.csv", "species,trait\nhomSap,1.0\npanTro,0.0\nmacMul,1.0\n")},
    ),
    (
        "csv_substring_fallback",
        ["aotTri_OMK", "homSap_293T"],
        {"phenotype_file": write_tmp("sub.csv", "species,trait\naotTri,yes\nhomSap,no\n")},
    ),
    (
        "csv_species_column_fallback_first_column",
        ["homSap", "panTro"],
        {"phenotype_file": write_tmp("nocol.csv", "who,trait\nhomSap,yes\npanTro,no\n")},
    ),
    (
        "csv_continuous_unparseable_cell",
        ["homSap", "panTro", "macMul"],
        {
            "phenotype_file": write_tmp("cont.csv", "species,mass\nhomSap,62\npanTro,oops\nmacMul,8\n"),
            "continuous": True,
        },
    ),
    (
        "csv_continuous_constant_not_scaled",
        ["homSap", "panTro"],
        {
            "phenotype_file": write_tmp("const.csv", "species,mass\nhomSap,5.0\npanTro,5.0\n"),
            "continuous": True,
        },
    ),
    (
        "tsv_separator_from_extension",
        ["homSap", "panTro"],
        {"phenotype_file": write_tmp("t.tsv", "taxon\ttrait\nhomSap\t1\npanTro\t0\n")},
    ),
    (
        "csv_extra_foreground_tokens",
        ["homSap", "panTro"],
        {
            "phenotype_file": write_tmp("tok.csv", "species,trait\nhomSap,marine\npanTro,land\n"),
            "foreground": "marine",
        },
    ),
    (
        "csv_explicit_columns",
        ["homSap", "panTro"],
        {
            "phenotype_file": write_tmp("two.csv", "species,a,b\nhomSap,no,yes\npanTro,yes,no\n"),
            "trait_col": "b",
            "species_col": "species",
        },
    ),
]

resolve_out = []
for name, taxa, kwargs in RESOLVE_CASES:
    call = dict(kwargs)
    fpath = call.get("phenotype_file")
    entry = {"name": name, "taxa": taxa, "kwargs": {k: v for k, v in kwargs.items() if k != "phenotype_file"}}
    if fpath:
        with open(fpath) as fh:
            entry["phenotype_csv"] = fh.read()
        entry["phenotype_file"] = os.path.basename(fpath)
    y, meta = resolve_phenotype_vector(taxa=taxa, **call)
    entry["y"] = [float(v) for v in y]
    entry["meta"] = {k: (int(v) if isinstance(v, (int, np.integer)) and k.endswith("count") else v) for k, v in meta.items()}
    resolve_out.append(entry)

RESOLVE_ERRORS = [
    ("no_source", ["a", "b"], {}),
    ("unknown_preset", ["a", "b"], {"preset": "nope"}),
    ("empty_foreground_string", ["a", "b"], {"foreground": ""}),
    ("empty_foreground_list", ["a", "b"], {"foreground": []}),
]
resolve_errors_out = []
for name, taxa, kwargs in RESOLVE_ERRORS:
    try:
        resolve_phenotype_vector(taxa=taxa, **kwargs)
        resolve_errors_out.append({"name": name, "taxa": taxa, "kwargs": kwargs, "error": None})
    except Exception as exc:  # noqa: BLE001
        resolve_errors_out.append({
            "name": name, "taxa": taxa, "kwargs": kwargs,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        })

# --------------------------------------------------------------------------------------------
# Bio.Phylo find_any as a regex, and compute_phylogenetic_covariance edge cases
# --------------------------------------------------------------------------------------------
FIND_ANY_TREES = [
    ("plain", "((a:1,b:2)ab:3,(c:4,d:5)cd:6)root;", ["a", "b", "c", "d", "ab", "cd", "root", "e"]),
    ("dot_is_any_char", "((aXc:1,abc:2)i1:3,d:4);", ["a.c", "abc", "aXc"]),
    ("alternation", "((a:1,b:2)i1:3,c:4);", ["a|b", "b|a", "z|c"]),
    ("prefix_not_matched", "((ab:1,a:2)i1:3,c:4);", ["a", "ab", "a.*"]),
]
find_any_out = []
for name, newick, names in FIND_ANY_TREES:
    tree = tree_from(newick)
    hits = []
    for nm in names:
        try:
            found = tree.find_any(name=nm)
            hits.append({"query": nm, "found": None if found is None else found.name})
        except Exception as exc:  # noqa: BLE001
            hits.append({"query": nm, "error": type(exc).__name__})
    find_any_out.append({"name": name, "newick": newick, "hits": hits})

COV_CASES = [
    ("balanced", "((a:1,b:2)ab:3,(c:4,d:5)cd:6):0;", ["a", "b", "c", "d"]),
    ("missing_taxon_diagonal_one", "((a:1,b:2)ab:3,c:4):0;", ["a", "b", "zzz"]),
    ("regex_name_matches_other", "((aXc:1,abc:2)i1:3,d:4):0;", ["a.c", "abc", "d"]),
    ("internal_node_name", "((a:1,b:2)i1:3,c:4):0;", ["i1", "a", "c"]),
    ("zero_length_branch", "((a:0.0,b:0.0)i1:0.5,c:1.0):0;", ["a", "b", "c"]),
    ("missing_branch_length", "((a,b:2)i1:3,c:4):0;", ["a", "b", "c"]),
    ("reordered_subset", "((a:1,b:2)ab:3,(c:4,d:5)cd:6):0;", ["d", "a", "c"]),
]
cov_out = []
for name, newick, taxa in COV_CASES:
    V = compute_phylogenetic_covariance(tree_from(newick), taxa)
    cov_out.append({"name": name, "newick": newick, "taxa": taxa, "V": [[float(x) for x in r] for r in V]})

# --------------------------------------------------------------------------------------------
# generate_permulations: the deterministic properties
# --------------------------------------------------------------------------------------------
PERM_NEWICK = "(((a:0.1,b:0.1)ab:0.2,(c:0.15,d:0.15)cd:0.15)abcd:0.1,(e:0.2,f:0.2)ef:0.2):0;"
PERM_TAXA = ["a", "b", "c", "d", "e", "f"]
PERM_CASES = [
    ("binary_k2", [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    ("binary_k5", [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
    ("all_zero_not_binary", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("all_one_not_binary", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    ("three_values_rank_matched", [0.0, 0.5, 1.0, 0.0, 0.5, 1.0]),
    ("continuous", [-1.5, 0.25, 3.0, -0.5, 0.75, 1.25]),
]
perm_out = []
for name, y in PERM_CASES:
    arr = np.array(y, dtype=float)
    P = generate_permulations(arr, tree_from(PERM_NEWICK), PERM_TAXA, n_perm=64, seed=7)
    perm_out.append({
        "name": name,
        "y": y,
        "n_perm": 64,
        "seed": 7,
        "is_binary": bool(len(np.unique(arr)) == 2 and set(np.unique(arr)).issubset({0, 1, 0.0, 1.0})),
        "row_sums": sorted(set(float(r.sum()) for r in P)),
        "sorted_row_values": [float(v) for v in np.sort(P[0])],
        "all_rows_same_multiset": bool(all(np.array_equal(np.sort(r), np.sort(arr)) for r in P)),
        "column_means": [float(v) for v in P.mean(axis=0)],
    })

payload = {
    "generated_by": "js/test/data/phenotype/gen.py",
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "python": sys.version.split()[0],
    "preset_keys": list(PRESETS.keys()),
    "presets": {k: {kk: vv for kk, vv in v.items()} for k, v in PRESETS.items()},
    "fnmatch": fnmatch_out,
    "repr_list": repr_out,
    "float_str": float_out,
    "read_csv": csv_out,
    "resolve": resolve_out,
    "resolve_errors": resolve_errors_out,
    "find_any": find_any_out,
    "covariance": cov_out,
    "permulations": {"newick": PERM_NEWICK, "taxa": PERM_TAXA, "cases": perm_out},
}

with open(OUT, "w") as fh:
    json.dump(payload, fh, indent=1, sort_keys=False)
    fh.write("\n")
print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")
