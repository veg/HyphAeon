# Parity: the Python reference against every other surface

`hyphaeon/*.py` is the reference implementation. `js/` (`@veg/hyphaeon-js`) is a second
implementation of the same methods, a library of pure functions with no onnxruntime, no I/O, no
workers. It is consumed by three runtime surfaces that live in the application repository
(`veg/hyphaeon-app`), not here. Two implementations drift; this document says how drift is
measured, what counts as a violation, and where the files go.

There are two layers:

| Layer | Where | Granularity | Runs on |
|---|---|---|---|
| Fixture replay | `js/test`, fixtures from `scripts/gen_fixtures.py` | per function, identical inputs by construction | every push touching `js/`, `fixtures/`, `models/` (`js.yml`) |
| End-to-end parity | `scripts/parity.py`, this document | per site / per gene / per edge on the bundled examples | pushes to `main` touching `hyphaeon/`, `js/`, `fixtures/`, `models/`; on demand (`parity.yml`) |

Fixture replay is where the 1e-9 class is exercised on identical inputs. End-to-end parity is
where the question a user would ask gets answered: same alignment in, same table out?

## Surfaces

| Surface | What runs | Produced by | Status |
|---|---|---|---|
| `python` | `python -m hyphaeon.cli <analysis> --cpu ...` on torch | `scripts/parity.py` (this repository) | reference; always run |
| `node` | `@veg/hyphaeon-js` + `runtime/` under Node with `onnxruntime-node` | runner in `veg/hyphaeon-app` | compared when its files exist |
| `web` | the same library in headless Chromium with `onnxruntime-web` | runner in `veg/hyphaeon-app` | compared when its files exist |
| `mcp` | the `@veg/hyphaeon-mcp` tool | runner in `veg/hyphaeon-app` | compared when its files exist |

`parity.py` never produces a non-Python surface. Nothing in this repository runs a model outside
torch, and `js/` has no session code by design; the app repository's runner writes the files into
the layout below and `parity.py` compares. A surface with no files is reported as *missing* and
does not fail the run unless `--strict-missing` is given. Until the app runner publishes into
`parity.yml`, that workflow verifies the Python side and the harness's own self-check only.

## Examples and analyses

Examples are every `examples/*.fasta` with its `examples/<name>.nwk` when one exists (RHO carries
its tree embedded and is run without `-t`). Analyses:

| Analysis | CLI | Compared |
|---|---|---|
| `meme` | `hyphaeon meme -a <fasta> -t <nwk> --cpu -o ...` | `sites[]` per site; `taxa_count`, `codon_count` |
| `busted` | `hyphaeon busted -a <fasta> -t <nwk> --cpu -o ...` | the gene record |
| `epistasis` | `hyphaeon epistasis ... --n-permutations B --seed S` | `edges[]`, `sectors[]`, `plasticity[]`; counts |

`epistasis` runs only once the CLI exposes `--seed` (`PLAN.md` section 7, item 7, in the app
repository). The library has `rng_seed=42` as a default, but the CLI does not surface it, so the
Python side cannot record the seed it used; `parity.py` detects the flag from `--help` and skips
with a note in the report until it lands. The comparator for epistasis is implemented and waiting.

## Tolerance classes

From `PLAN.md` 5.4 (app repository). Constants are in `scripts/parity.py` next to their reasons.

| Class | Test | Applies to |
|---|---|---|
| **exact** | equality after canonicalisation | site order; `is_invariable`; `taxa_count`, `codon_count`; busted `taxa`, `sites`, `sig_sites_p05`, `sig_sites_p10`, `positive_selection_detected`; epistasis counts, edge set and order `(site_u, site_v)`, `ref_u/v`, `shared_branches`, `branches_u/v`, sector ids and `sites`, `size`, `pars_signature`; plasticity `wt_aa`, `focal_taxon` |
| **tolerance, graph** (1e-6) | `max |Δ|` ≤ 1e-6 | `hyphaeon_lrt`; busted `predicted_gene_lrt`, `selection_probability`, `synonymous_rate_variation`, `rate_distributions.*`; edge `lrt_u/v`, `similarity`, `cesi`; sector `mean_lrt`; plasticity `baseline_lrt`, `intrinsic_plasticity`, `max_shift` |
| **tolerance, sum** (L · 1e-6) | `max |Δ|` ≤ L · 1e-6 | busted `omnibus_lrt`, `total_selection_energy` — sums over L per-site values each within 1e-6 |
| **tolerance, eigen** (1e-5) | `max |Δ|` ≤ 1e-5 | sector `spectral_coherence` (an eigenvalue ratio; the MDS class) |
| **special** (1e-9) | `max |Δ|` ≤ 1e-9 *given equal input* | `p_value`, `q_value` (see below); edge `p_hyper`, `fdr_q` (hypergeometric on integer counts, BH over those) |
| **derived** (1e-6) | `max |Δ|` ≤ 1e-6 | busted `p_value_acat`, `p_value_simes`; plasticity `p_value` — special functions applied to graph-class inputs, so 1e-9 cannot apply end to end |
| **statistical** | `|Δp| ≤ 3·√(p(1−p)/B)`, p floored at 1/B; null moments within 2% relative | sector `p_perm`, `null_coherence_mean`, `null_coherence_std`, `null_coherence_95`; each side at B = 10,000 with its own seed; skipped with a note when absent on either side |

**`p_value` and `q_value` "given equal LRT".** A surface's LRT may differ from the reference by up
to 1e-6 (the graph class). Comparing its p-value to the reference file at 1e-9 would then fail
on every site for a reason already reported under `hyphaeon_lrt`. So the special class evaluates
the reference function on the *surface's own input*: `p_value` is checked against
`hyphaeon.stats.pvals_from_lrt_meme(surface lrt)` and `q_value` against
`hyphaeon.stats.benjamini_hochberg(surface p)`. Both sides are rounded to float32 first, because
the reference CLI writes p and q as float32 (`cli.py`, `cmd_meme`: `.astype(np.float32)`); a
float64 port compared to a float32 file at 1e-9 would fail by construction. The same check runs
on the Python file itself as a self-check of the harness and reports `max_abs_diff = 0`.

A **violation** is any check in any class exceeding its bound. One violation fails the run
(exit 1). Missing surfaces and skipped classes are notes, not violations.

## File layout

```
parity/                                  gitignored; regenerated by every run
├── report.json                          the result (below)
├── python/                              written by scripts/parity.py
│   ├── <example>.<analysis>.json        the CLI's -o output, unmodified
│   └── <example>.<analysis>.log         the CLI's stdout/stderr
├── node/                                written by the app repository's runner
│   └── <example>.<analysis>.json        same schema as python/, same example names
├── web/                                 idem
└── mcp/                                 idem
```

`<example>` is the alignment's stem (`Smc6`, `bat_oas1`, `camelid`, `HIV1_RT`, `RHO`);
`<analysis>` is `meme`, `busted` or `epistasis`. A surface file carries the same JSON the Python
CLI writes for that analysis (Appendix B of `PLAN.md`): for `meme`, `sites[]` of
`{site, hyphaeon_lrt, p_value, q_value, is_invariable}` in site order with `taxa_count` and
`codon_count`; for `busted`, the single gene record; for `epistasis`, `edges[]`, `sectors[]`,
`plasticity[]` and the counts. Extra keys (provenance, timings) are ignored; missing keys are
noted as "absent" when absent on both sides and are violations when absent on one.

`report.json`:

```
generated_at, repo_commit, reference, surfaces_requested, surfaces_present, surfaces_missing,
examples, analyses, weights, n_permutations, seed, tolerances,
runs[]          {surface, example, analysis, path, status: ok|failed|missing, seconds, note, command}
self_checks[]   {surface: python, example, analysis, checks[]}
comparisons[]   {surface, example, analysis, path, status: pass|fail|missing|no-reference,
                 violations, checks[], notes[]}
   checks[]     {field, class, n, violations, tol, max_abs_diff, note, first_violations[≤20]}
notes[]
summary         {reference_runs, reference_failed, self_check_violations, comparisons, missing,
                 violations, strict_missing, passed}
```

Exit codes: `0` no violations; `1` violations (or a missing surface under `--strict-missing`);
`2` a reference run failed.

## Running locally

```bash
# from the repository root, with the package installed (pip install -e .)
export HYPHAEON_WEIGHTS=$PWD/model.safetensors   # the bundled suite weights; skip to let the CLI resolve
export HF_HUB_OFFLINE=1

python scripts/parity.py --examples Smc6,bat_oas1 --surfaces python      # reference + self-check, ~8 s
python scripts/parity.py --examples all --surfaces python,node           # what parity.yml runs
python scripts/parity.py --no-run --surfaces python,node,web,mcp         # recompare existing files
python scripts/parity.py --examples Smc6 --surfaces python,node --strict-missing   # missing = failure
```

Options: `--analyses meme,busted,epistasis`, `--out <dir>` (default `parity/`), `--weights`,
`--n-permutations` (B; default 10,000), `--seed` (default 42; D17), `--no-run`,
`--strict-missing`. The runner in `veg/hyphaeon-app` writes its outputs into `parity/<surface>/`
of a checkout of this repository (or any `--out` directory) and then calls `parity.py --no-run`.

## Reading a failure

`report.json` → `comparisons[]` → the entry with `status: fail` → `checks[]` with
`violations > 0`. `first_violations` lists up to twenty `{key, ref, got, abs_diff, tol}`; `key`
is the site number, the `(site_u, site_v)` pair, the sector id, or the field path. A failure in
`site order` or an edge/sector order check suppresses the per-item checks under it, since the
items cannot be aligned. A `hyphaeon_lrt` failure with clean `p_value`/`q_value` means the graph
(or the tensors fed to it) differs and the statistics do not; the reverse means the special
functions or BH differ; both failing usually means the LRT.
