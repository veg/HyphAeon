"""
WHY THIS FILE EXISTS

Reference strings for js/src/evaluate.js and js/src/writers.js, produced by the Python standard
library, pandas and networkx/lxml themselves, so the JavaScript ports of `csv.DictReader`,
`float()`, `html.unescape`, `float.__repr__`, `format(x, '.6f')`, `json.dumps(indent=2)`,
`DataFrame.to_csv(index=False)` and `nx.write_graphml` are measured against the real thing and not
against a reading of their documentation. fixtures/ (scripts/gen_fixtures.py) covers the
evaluation.py functions end to end; it does not exercise the serialisers or the parser corner
cases, which is what this file is for. It is NOT a fixture in the fixtures/README.md sense: it
lives with the tests, and js/test/evaluate.test.js and js/test/writers.test.js read its output.

Regenerate from js/ with the reference environment (the venv that runs hyphaeon; pandas, lxml and
networkx installed):

    python test/data/evaluate/gen.py

Output: python_formatting.json in this directory. Record-shaped inputs are read from
fixtures/e2e/*.json and fixtures/evaluation/*.json so the CSV/JSON/GraphML strings here are the
reference CLI's own output layouts for the reference's own numbers.
"""

from __future__ import annotations

import csv
import io
import json
import math
import random
import struct
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from hyphaeon.evaluation import (  # noqa: E402
    EvaluationError,
    _boolean,
    _normalized_header,
    _probability,
    _site_number,
    format_text_report,
)

SEED = 20260904


def load(rel):
    return json.load(open(ROOT / rel))


def to_csv(records, drop=None):
    df = pd.DataFrame(records)
    if drop and drop in df.columns:
        df = df.drop(columns=[drop])
    return df.to_csv(index=False)


def graphml(edges):
    G = nx.Graph()
    for e in edges:
        G.add_edge(
            str(e["site_u"]),
            str(e["site_v"]),
            weight=float(e["similarity"]),
            cesi=float(e["cesi"]),
            shared=int(e["shared_branches"]),
            fdr_q=float(e["fdr_q"]),
        )
    buf = io.BytesIO()
    nx.write_graphml(G, buf)
    return buf.getvalue().decode("utf-8")


def main():
    out = {}

    # --- float repr -------------------------------------------------------------------------
    random.seed(SEED)
    samples = [
        0.0, -0.0, 1.0, 100.0, 1e15, 1e16, 1e17, 1e21, 1e22, 1e-4, 1e-5, 0.001,
        123456789012345678.0, 1.5e-300, 5e-324, 1.7976931348623157e308, 2.2250738585072014e-308,
        0.1, 0.30000000000000004, 1 / 3, 2 / 3, 9007199254740993.0, 1e100, 1.5e-7, 12345.678,
        -1234567.0, 1e-10, 999999999999999.9, 9999999999999998.0, 0.000123456, 1e-6, 1e-7,
        0.6666666865348816, 5.023776531219482, 1.0150756628943599e-05,
    ]
    for _ in range(120):
        e = random.randint(-320, 308)
        samples.append(random.random() * random.choice([1, 10, 100]) * 10.0 ** e)
    for _ in range(40):
        samples.append(struct.unpack("<d", struct.pack("<Q", random.getrandbits(64)))[0])
    out["float_repr"] = [[v, repr(v)] for v in samples if math.isfinite(v)]
    out["float_repr_special"] = [[str(v), repr(v)] for v in (float("inf"), -float("inf"), float("nan"))]

    fixed = [0.0078125, 2.5e-7, 123456.9999995, -0.0000001, 1e21, 0.5, 1.0, 0.1234565, 0.1234575,
             5e-7, 1.5e-6, 0.9999995, 1e-300, 123456789.123456789, -1.5, 0.0, -0.0, 0.6769239610533377,
             0.00390625, 0.0000005, 2.0000005, 1e-7, 0.9999999]
    out["format_fixed6"] = [[v, f"{v:.6f}"] for v in fixed]
    gvals = [0.05, 0.10, 1e-5, 123456789.0, 0.0001, 1.5, 100000.0, 1000000.0, 0.00001234, 1e-4,
             12345.6789, 0.0, 1.0, 999999.4, 999999.5, 0.5, 2.0]
    out["format_g"] = [[v, f"{v:g}"] for v in gvals]

    # --- repr / str ---------------------------------------------------------------------------
    objs = ["abc", "it's", 'say "hi"', "both ' and \"", "back\\slash", "tab\tnl\n", "\x01\x7f",
            "é€", "\U0001F600", [1, 2, "x"], {"A": 0.5, "B": -1.0}, [], {}, None, True,
            False, 1e-5, [[1], ["a"]], " ", "a b", ""]
    out["repr"] = [[o, repr(o)] for o in objs]

    # --- json.dumps(indent=2) -----------------------------------------------------------------
    d = load("fixtures/e2e/meme_bat_oas1_attribute_filter.json")[0]["outputs"]
    sub = dict(d)
    sub["sites"] = d["sites"][139:142]
    sub["attributions"] = {"141": d["attributions"]["141"]}
    sub["runtime_sec"] = 1.25
    out["meme_json"] = {"sites_slice": [139, 142], "attribution_key": "141", "runtime_sec": 1.25,
                        "text": json.dumps(sub, indent=2)}
    # The JS test rebuilds this value by hand (NaN/inf cannot round-trip through JSON): keep in sync.
    edge = {"a": [], "b": {}, "c": [1, 2.5, None, True, False, "xé\n\"y\"\\/\t\U0001F600\x01\x7f"],
            "d": float("nan"), "e": float("inf"), "f": -float("inf"), "g": 1e16, "h": 1e-5, "i": -0.0,
            "j": {"k": [{"l": []}]}, "ü": 1}
    out["json_edge"] = {"text": json.dumps(edge, indent=2)}
    try:
        json.dumps({"x": float("nan")}, allow_nan=False)
    except ValueError as exc:
        out["json_allow_nan_error"] = str(exc)

    # --- pandas to_csv ------------------------------------------------------------------------
    recs = []
    for r in d["sites"][138:144]:
        row = {k: r[k] for k in ("site", "hyphaeon_lrt", "p_value", "q_value", "is_invariable")}
        if "evolutionary_epoch" in r:
            for k in ("evolutionary_epoch", "adaptation_mode", "top_driver", "top_mutation"):
                row[k] = r[k]
        recs.append(row)
    out["meme_csv_attr"] = {"sites_slice": [138, 144], "text": to_csv(recs)}
    d2 = load("fixtures/e2e/meme_Smc6.json")[0]["outputs"]
    recs = [{k: r[k] for k in ("site", "hyphaeon_lrt", "p_value", "q_value", "is_invariable")} for r in d2["sites"][:4]]
    out["meme_csv_plain"] = {"sites_slice": [0, 4], "text": to_csv(recs)}

    b = load("fixtures/e2e/busted_Smc6.json")[0]["outputs"]
    r = dict(b)
    r.update(selection_probability=0.664965033531189, predicted_gene_lrt=0.0, positive_selection_detected=True,
             elapsed_seconds=0.123456, synonymous_rate_variation=0.0123)
    r["rate_distributions"] = dict(r["rate_distributions"])
    r["rate_distributions"].update(omega_3=0.5653032064437866, proportion_1=0.3, proportion_2=0.2469, proportion_3=0.45309603214263916)
    row = {
        "Gene": r["gene"], "Taxa": r["taxa"], "Sites": r["sites"], "p_ACAT": r["p_value_acat"],
        "p_Simes": r["p_value_simes"], "Selection_Prob": r["selection_probability"],
        "Pred_Gene_LRT": r["predicted_gene_lrt"], "Omnibus_LRT": r["omnibus_lrt"],
        "Omega_3": float(r["rate_distributions"]["omega_3"]),
        "Prop_Positive": float(r["rate_distributions"]["proportion_3"]),
        "Sig_Sites_p05": r["sig_sites_p05"], "Selected": r["positive_selection_detected"],
        "Time_ms": r["elapsed_seconds"] * 1000,
    }
    out["busted"] = {"record": r, "csv": to_csv([row]), "json_single": json.dumps(r, indent=2),
                     "json_batch": json.dumps([r, r], indent=2)}

    e = load("fixtures/e2e/epistasis_Smc6_n_permutations_1000.json")[0]["outputs"]
    out["edges_csv"] = to_csv(e["edges"])
    out["plasticity_csv"] = {"slice": [0, 2], "text": to_csv(e["plasticity"][:2])}
    out["dms_csv"] = {"slice": [0, 2], "text": to_csv(e["plasticity"][:2], drop="mutant_deltas")}
    out["sectors_csv"] = to_csv(e["sectors"])
    out["edges_graphml"] = graphml(e["edges"])
    p = load("fixtures/e2e/phenotype_RHO_marine_n_permutations_0.json")[0]["outputs"]
    out["phenotype_csv"] = {"slice": [0, 3], "text": to_csv(p["sites"][:3])}

    out["csv_quirks"] = to_csv([
        {"a": 1, "b": True, "c": "x,y", "d": 'q"q', "e": None, "f": 1.5},
        {"a": 2, "f": float("nan"), "g": [1, 2], "h": {"A": 0.5, "B": -1.0}, "i": "line\nbreak"},
    ])
    out["csv_floats"] = to_csv([{"v": v} for v in [0.0, 1.0, 1e16, 1e15, 0.0001, 0.00001, 123456789012345678.0,
                                                    1.5e-300, -0.0, 2.5, 1e22, float("inf"), -float("inf")]])
    out["csv_int_missing"] = to_csv([{"a": 1, "b": 1}, {"a": 2}])
    out["csv_bool_missing"] = to_csv([{"a": True, "b": False}, {"a": False}])
    out["csv_single_column"] = to_csv([{"a": ""}, {"a": "x"}, {"a": None}])

    # --- graphml ------------------------------------------------------------------------------
    small = [
        {"site_u": 279, "site_v": 930, "similarity": 0.8029388189315796, "cesi": 2.938567876815796, "shared_branches": 5, "fdr_q": 0.00011417182395234704},
        {"site_u": 930, "site_v": 12, "similarity": 0.5, "cesi": 1e-05, "shared_branches": 3, "fdr_q": 1.0},
        {"site_u": 279, "site_v": 12, "similarity": 0.25, "cesi": 0.0, "shared_branches": 2, "fdr_q": 2.5e-07},
        {"site_u": 7, "site_v": 279, "similarity": 1.0, "cesi": 3.0, "shared_branches": 10, "fdr_q": 1e-16},
    ]
    dup = [
        {"site_u": 1, "site_v": 2, "similarity": 0.1, "cesi": 0.2, "shared_branches": 1, "fdr_q": 0.3},
        {"site_u": 2, "site_v": 1, "similarity": 0.9, "cesi": 0.8, "shared_branches": 7, "fdr_q": 0.6},
        {"site_u": 2, "site_v": 2, "similarity": 0.5, "cesi": 0.5, "shared_branches": 5, "fdr_q": 0.5},
    ]
    esc = [{"site_u": "a<b", "site_v": 'c&"d', "similarity": 1.0, "cesi": 1.0, "shared_branches": 1, "fdr_q": 1.0}]
    out["graphml"] = {"small": {"edges": small, "text": graphml(small)}, "empty": {"edges": [], "text": graphml([])},
                      "dup": {"edges": dup, "text": graphml(dup)}, "esc": {"edges": esc, "text": graphml(esc)}}

    # --- text report --------------------------------------------------------------------------
    cases = load("fixtures/evaluation/evaluate_files.json")
    r = cases[2]["outputs"]["result"]
    r0 = cases[0]["outputs"]["result"]
    r2 = json.loads(json.dumps(r0))
    r2["pearson_r"] = None
    r2["thresholds"]["0.05"]["ppv"] = None
    r2["thresholds"]["0.10"]["fpr"] = 0.0078125
    r2["warnings"] = []
    out["text_report"] = [{"result": r, "text": format_text_report(r)}, {"result": r2, "text": format_text_report(r2)}]

    # --- csv.reader / DictReader ----------------------------------------------------------------
    csv_cases = ["a,b\n1,2\n", "a,b\r\n1,2\r\n", "a,b\r1,2\r", "a,b\n\n1,2\n\n", 'a,"b,c",d\n"x""y",,"z\nw"\n',
                 "a,b,c\n1,2\n1,2,3,4\n", "a,b\n1,2", "a,b\n1,", 'a\n"unterminated', 'a,b\n"q"x,"r" s\n',
                 "\ufeffsite,x\n1,2\n", " a , b \n 1 , 2 \n", 'a,b\n"",""\n', "", "\n", "\n\na,b\n1,2\n",
                 "a,b\n1,2\n\n", '"a\nb",c\n1,2\n', "a,b\r\n\r\n1,2", 'a,b\n"x"\n', "a,b\n1,2\r\n3,4\n5,6\r7,8",
                 '"a""b",c\n', "a,b,\n,,\n"]
    out["csv_reader"] = [[text, list(csv.reader(io.StringIO(text, newline="")))] for text in csv_cases]
    dict_cases = ["\ufeffsite,hyphaeon_lrt,p_value,is_invariable\n1,0.5,0.5,True\n",
                  "\nsite,hyphaeon_lrt,p_value,is_invariable\n1,0.5,0.5,True\n",
                  "site,hyphaeon_lrt,p_value,is_invariable\n\n1,0.5,0.5,True\n\n2,0.5,0.5,False\n",
                  "site,hyphaeon_lrt,p_value,is_invariable,site\n1,0.5,0.5,True,2\n",
                  "site,hyphaeon_lrt,p_value,is_invariable\n1,0.5\n", "", "\n",
                  "site,hyphaeon_lrt,p_value,is_invariable\n1,0.5,0.5,True,extra,more\n"]
    dict_out = []
    for text in dict_cases:
        stripped = text[1:] if text.startswith("\ufeff") else text  # utf-8-sig
        reader = csv.DictReader(io.StringIO(stripped, newline=""))
        rows = [[i, [[("None" if k is None else k), v] for k, v in row.items()]] for i, row in enumerate(reader, start=2)]
        dict_out.append([text, reader.fieldnames, rows])
    out["dict_reader"] = dict_out

    # --- float() / int() / _boolean / _normalized_header / messages ---------------------------
    floats = ["1.5", " 1.5 ", "inf", "-Infinity", "nan", "1e3", "1_000.5", "", "abc", ".5", "5.", "+.5e-3", "1e",
              "0x10", "٣.5", "1__0", "  ", "InFiNiTy", "NAN", "-nan", "1.5\n", "1,5", "١٢", "1e400",
              "-.5", "1_0e1_0", "1.", ".", "+", "1e+", "  7  ", "\x1c8\x1f"]
    fl = []
    for s in floats:
        try:
            v = float(s)
            fl.append([s, "nan" if math.isnan(v) else repr(v)])
        except ValueError:
            fl.append([s, None])
    out["float_parse"] = fl
    ints = ["12", " 12 ", "+3", "-3", "1_000", "1.0", "1e2", "", "0x10", "٣", "1__0", "_1", "1_", "০৩",
            "abc", "007", "-0", "\U0001d7d9"]
    il = []
    for s in ints:
        try:
            il.append([s, int(s)])
        except ValueError:
            il.append([s, None])
    out["int_parse"] = il
    bools = []
    for v in ["True", " yes ", "0", "NO", "1", "t", None, "  false\t", "Yes", "TRUE", "", "none"]:
        try:
            bools.append([v, _boolean(v, "is_invariable", "geneX.csv")])
        except EvaluationError as exc:
            bools.append([v, str(exc)])
    out["boolean"] = bools
    headers = ["<b>LRT</b>", "p-value", "&beta;<sup>-</sup>", "LRT&nbsp;", "L&#82;T", "L&#x52;T", "&fjlig;x", "&Idot;",
               "&ltx", "&ampLRT", "&notin;", "&noti", "&#0;a", "&#xD800;b", "&#150;", "&#1;z", ["LRT", "desc"], [], 5,
               None, True, "&unknownentity;lrt", "&#xZZ;", "&#;", "&#x;", "a&b", "&LT;rt", "&CounterClockwiseContourIntegral;p",
               "&#1114112;x", "&#1114111;x", "&#128;", "&#8490;", "&#304;", "&amp;#76;", "&#x1D7D9;", "<sup>+</sup>p",
               "p<sup>-</sup>", "# branches under selection", "&#65;&#x42;C", "&aacute", "&aacuteX", "&AMPx", "&Aacute;&amp", "&&lt;;",
               "&#xd7ff;", "&#xe000;", "&#0xff;", "&#12;", "&#13;", "&#65536;", "&Longlefta", "&Longleftarrow;", "&nbs", "&nbsp"]
    out["normalized_header"] = [[h, _normalized_header(h)] for h in headers]
    msgs = []
    for v in [1.5, "2", -0.5, 2.0, "1e0", "0", "1", "abc", None, "inf", True]:
        try:
            _probability(v, "p_value", "geneX.csv")
            msgs.append([v, "ok"])
        except EvaluationError as exc:
            msgs.append([v, str(exc)])
    out["probability_messages"] = msgs
    msgs = []
    for v in ["1.5", "0", "-1", "3.0", "abc", None, "inf", "12", " 7 ", "٣", True]:
        try:
            msgs.append([v, _site_number(v, "geneX.csv")])
        except EvaluationError as exc:
            msgs.append([v, str(exc)])
    out["site_number_messages"] = msgs

    with open(HERE / "python_formatting.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {HERE / 'python_formatting.json'}")


if __name__ == "__main__":
    main()
