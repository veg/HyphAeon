# Parity: the Python reference against every other surface

`hyphaeon/*.py` is the reference implementation. `js/` (`@veg/hyphaeon-js`) is a second
implementation of the same methods, a library of pure functions with no onnxruntime, no I/O, no
workers. It is consumed by runtime surfaces that live in the application repository
(`veg/hyphaeon-app`), not here. Two implementations drift; this document says how drift is
measured, what counts as a violation, and where the files go.

There are two layers:

| Layer | Where | Granularity | Runs on |
|---|---|---|---|
| Fixture replay | `js/test`, fixtures from `scripts/gen_fixtures.py` | per function, identical inputs by construction | every push touching `js/`, `fixtures/`, `models/` (`js.yml`) |
| End-to-end parity | `scripts/parity.py`, this document | per site / per gene / per edge / per mutant / per trait site on the bundled examples | pushes to `main` touching `hyphaeon/`, `js/`, `fixtures/`, `models/`; on demand (`parity.yml`) |

Fixture replay is where the 1e-9 class is exercised on identical inputs. End-to-end parity is
where the question a user would ask gets answered: same alignment in, same table out?

## Surfaces

| Surface | What runs | Compared against | Produced by |
|---|---|---|---|
| `python` | `python -m hyphaeon.cli <analysis> --cpu ...` on torch | is the reference | `scripts/parity.py` (this repository) |
| `python-tn93` | the same commands with `--use-tn93` (tree-free, PLAN.md D22) | is the reference for every `*-tn93` surface | `scripts/parity.py` |
| `node` | `@veg/hyphaeon-js` + `runtime/` under Node with `onnxruntime-node` | `python` | `runtime/scripts/parity-node.mjs` in `veg/hyphaeon-app` |
| `node-tn93` | the same runner, the examples it ran tree-free | `python-tn93` | the same runner, which writes them to the `-tn93` sibling directory |
| `web` (alias `browser`) | the same library in headless Chromium with `onnxruntime-web` | `python` | the Playwright e2e in `veg/hyphaeon-app` (`parity/browser/`; either directory name is read) |
| `web-tn93` (alias `browser-tn93`) | idem, tree-free | `python-tn93` | idem |
| `mcp`, `mcp-tn93` | the `@veg/hyphaeon-mcp` tool | `python`, `python-tn93` | no runner yet; reported as missing |

`parity.py` never produces a non-Python surface. Nothing in this repository runs a model outside
torch, and `js/` has no session code by design; the app repository's runners write the files into
the layout below and `parity.py` compares.

**Tree-free examples.** Under D22 a runtime uses a tree only when it carries branch lengths;
`examples/camelid.nwk` and `examples/HIV1_RT.nwk` are topologies without lengths, so every runtime
surface runs those two tree-free (pairwise TN93 distances straight into the MDS, exactly the
reference's `--use-tn93`). The reference's tree-based run of those examples is still made — it is
what the CLI does when handed `-t` — but comparing a tree-free surface file to it would compare two
different analyses of the same data, which is what the Phase 2 parity gap was. So the reference is
run twice for those examples, `parity/python/` with the tree and `parity/python-tn93/` without, and
each surface file is compared to the run that matches how it was produced. `--tn93-examples auto`
(the default) picks every example whose `.nwk` has no branch lengths plus any example a requested
`-tn93` surface has files for. The tree-free reference runs hide the compiled `tn93` binary from
`PATH` so the Python `tn93` package computes the distances, the path the fixtures pin and the JS
port mirrors (the two measured identical; `gen_fixtures.py`).

A tree-based surface has no file for a tree-free example by construction; that pair is reported as
**redirected** (compared under the `-tn93` sibling), not as missing.

## Examples and analyses

Examples are every `examples/*.fasta` with its `examples/<name>.nwk` when one exists (RHO carries
its tree embedded and is run without `-t`).

| Analysis | Reference command | Examples | Compared |
|---|---|---|---|
| `meme` | `hyphaeon meme -a <fasta> -t <nwk> --cpu -o ...` | all | `sites[]` per site; `taxa_count`, `codon_count` |
| `busted` | `hyphaeon busted -a <fasta> -t <nwk> --cpu -o ...` | all | the gene record's statistical half |
| `epistasis` | `hyphaeon epistasis ... --n-permutations B --seed S` | all | `edges[]`, `sectors[]`, `plasticity[]`; counts |
| `dms` | `hyphaeon dms -a <fasta> -t <nwk> --cpu -o ...` | `--dms-examples` (default `Smc6`: 19 x L forward passes) | `plasticity[]` per site and per mutant; counts, `focal_taxon` |
| `phenotype` | `hyphaeon phenotype -a <fasta> -fg <list> --n-permutations B --permulations 0 --seed S --cpu` | `--phenotype-examples` (default `RHO`, `--phenotype-fg` the README Example 3 marine list) | the 21-key record: gene fields, `sites[]`, `coselection_pairs[]`, `trait_sectors[]`, `phenotype_meta` |

B defaults to 10,000 (`--n-permutations`), the seed to 42 (`--seed`, D17); `--permulations`
defaults to 0 (parametric association p, as the fixture and the app runner). `epistasis` and
`phenotype` run only when the CLI exposes `--seed` (it does since Phase 2a; the check stays).

## Tolerance classes

From `PLAN.md` 5.4 (app repository) as refined by the phase documents. Constants are in
`scripts/parity.py` next to their reasons.

| Class | Test | Applies to |
|---|---|---|
| **exact** | equality after canonicalisation | site order; `is_invariable`; every count (`taxa_count`, `codon_count`, `evaluated_taxa`, `*_count`, busted `taxa`, `sites`, `sig_sites_p05/p10`); busted `omega_1`, `omega_2` (constants); edge set and order `(site_u, site_v)`, `ref_u/v`, `shared_taxa`, `shared_branches`; sector ids and order, `sites`, `size`, `shared_*`, `pars_signature`, `consensus_signature`; plasticity `wt_aa` and mutant set; phenotype `phenotype_meta`, `compact_pars_signature`, `permulations_count`, site `ref_aa`/`derived_aa`, site order (score-descending) |
| **graph** | `|Δ|` ≤ 1e-5 · max(1, \|ref\|) | `hyphaeon_lrt`; edge `lrt_u/v`, `cesi`; sector `mean_lrt`; plasticity `baseline_lrt`; phenotype `score_track_a`, `dual_track_composite`. PLAN 5.4's measured class: fp32 torch paths themselves differ by 6.7e-6, so the first harness's 1e-6 absolute was unreachable and Phases 1–3 re-evaluated every comparison by hand |
| **graph, delta** | `|Δ|` ≤ 1e-5 · (max(1, \|baseline\|) + max(1, \|mutant\|)) | DMS `mutant_deltas[aa]` (`mutant_lrt − baseline_lrt`, `epistasis.py:560`), `mean/max/min_delta_lrt`, `intrinsic_plasticity` (mean \|delta\|) — a difference of two graph-class values carries both operands' budgets; measured up to 4.1x the single-value bound on RHO before this rule |
| **graph, sum** | `|Δ|` ≤ L · 1e-5 | busted `omnibus_lrt`, `total_selection_energy` — sums over L per-site values |
| **eigen** (1e-5) | `|Δ|` ≤ 1e-5 | sector `spectral_coherence` (an eigenvalue ratio; the MDS class) |
| **special** (1e-9) | `|Δ|` ≤ 1e-9 *given equal input*, after float32 rounding | meme `p_value`, `q_value` (below) |
| **derived** (1e-6 absolute) | `|Δ|` ≤ 1e-6 | edge `similarity`, `p_val`, `hyper_p`, `fdr_q`; phenotype pair `similarity`, `p_value`, `q_value`; busted `p_value_acat`, `p_value_simes`; plasticity `p_value`; phenotype gene `spectral_energy`, `norm_spectral_ratio`, `max_assoc`, `p_evd_length_adjusted`, `score_track_b` and every site column but the LRT (`p_lrt`, `attribution_norm`, `fg/bg_mean_attn`, `association_rho`, `p_value`, `p_assoc*`, `score`, `q_value`, `*_freq_pct`). The cosine network is float32 arithmetic on float32 attributions whose BLAS accumulation order nobody reproduces (PHASE2A.md gap 4); everything downstream inherits that |
| **statistical** | `|Δp|` ≤ 3·√(p(1−p)·(1/B_ref + 1/B_surface)), p floored at 1/min(B) | sector `p_perm` (epistasis and trait sectors), `gene_p_value_perm`, `p_assoc_perm` (B = `permulations_count` for the last two); each side with its own seed and its **own B** (below) |
| **statistical, null moments** | relative difference ≤ 2 %, **enforced only when both B ≥ 10,000** | `null_coherence_mean`, `null_coherence_std`, `null_coherence_95`; below that they are *informational* (below) |
| **skipped** | not compared; note in the report | the BUSTED neural head (below) |

**`p_value` and `q_value` "given equal LRT".** A surface's LRT may differ from the reference within
the graph class. Comparing its p-value to the reference file at 1e-9 would then fail on every site
for a reason already reported under `hyphaeon_lrt`. So the special class evaluates the reference
function on the *surface's own input*: `p_value` is checked against
`hyphaeon.stats.pvals_from_lrt_meme(surface lrt)` and `q_value` against
`hyphaeon.stats.benjamini_hochberg(surface p)`. Both sides are rounded to float32 first, because
the reference CLI writes p and q as float32 (`cli.py`, `cmd_meme`: `.astype(np.float32)`); a
float64 port compared to a float32 file at 1e-9 would fail by construction. The same check runs
on the Python files themselves as a self-check of the harness and reports `max_abs_diff = 0`.

**The skipped BUSTED fields, and why.** `predicted_gene_lrt`, `selection_probability`,
`synonymous_rate_variation`, `rate_distributions.omega_3`, `proportion_1..3` and
`positive_selection_detected` come from `BustedMultiTaskHead`. `model.safetensors` lacks 11 of its
parameters and `cmd_busted` loads it with `strict=False` and never seeds torch (PHASE0.md; PHASE1A.md
gap 4; the fixtures null the same fields), so the reference draws them anew on every run;
`busted_head.onnx` is one seeded draw of that head. Nothing can be at parity with an unseeded
reference, so the harness records them as `skipped` with that note rather than failing on them or
pretending a tolerance. The statistical half of the record (ACAT, Simes, omnibus LRT, energy, site
counts) is compared in full. Seeding the head upstream, or shipping its weights, would move these
fields into the graph class.

**B and the statistical class.** The bound is 3 sigma of the *difference* of two independent Monte
Carlo estimates, so each side contributes its own variance: with both at B = 10,000 it is
3·√(2p(1−p)/10,000), PLAN 5.4's bound with the second estimate accounted for. B_surface is read
from the file (`provenance.options.permutations`), else from the runner's `summary.json` in the
same directory (`n_permutations`), else assumed equal to `--n-permutations` and noted. The browser
runs its default B = 1,000 against the reference's 10,000. The null moments are a different matter:
PHASE2A.md measured the standard error of the null std estimate at B = 1,000 as about 2.2 %
relative (a 3–5 % excursion at that B is the estimator, not a defect; at B = 100,000 the two nulls
agree to well inside 1 %). So the 2 % class is enforced only when both sides ran B ≥ 10,000; below
that the moments are reported as **informational** — their differences appear in `report.json`
(`excursions`) and in the run's `informational excursions` count, and never as violations. A
surface that wants the moments enforced runs at B = 10,000 (`parity-node.mjs --permutations`).

**Incomparable files.** When a surface file's taxon count differs from the reference's, the surface
analysed a different taxon set (the browser's default cap is 256 taxa; the reference and the runner
score all 655 RHO taxa) and nothing below the counts can be aligned. Such a pair is reported as
**incomparable** with both counts in the note, never silently compared and never called a pass.
`--strict-missing` makes missing and incomparable files fail the run.

**Epistasis `plasticity` absent from a surface** (the browser runs its DMS as a separate,
work-budgeted analysis) is a note, not a violation; the DMS comparator covers the same fields.

A **violation** is any check in any class exceeding its bound. One violation fails the run
(exit 1). Missing, redirected and incomparable files, informational excursions and skipped fields
are reported, not violations.

## File layout

```
parity/                                  gitignored; regenerated by every run
├── report.json                          the result (below)
├── python/                              written by scripts/parity.py, tree-based reference
│   ├── <example>.<analysis>.json        the CLI's -o output, unmodified
│   └── <example>.<analysis>.log         the CLI's stdout/stderr
├── python-tn93/                         written by scripts/parity.py with --use-tn93 (tree-free examples)
├── node/                                written by the app repository's runner (tree-based runs)
│   ├── <example>.<analysis>.json        same schema as python/, same example names
│   └── summary.json                     the runner's record: B, seed, tree_free per run
├── node-tn93/                           the runs that went tree-free (D22)
├── browser/  browser-tn93/              written by the Playwright e2e (read as `web`, `web-tn93`)
├── web/      web-tn93/                  the same files under the surface's name, when copied
└── mcp/      mcp-tn93/                  no runner yet
```

`<example>` is the alignment's stem (`Smc6`, `bat_oas1`, `camelid`, `HIV1_RT`, `RHO`);
`<analysis>` is `meme`, `busted`, `epistasis`, `dms` or `phenotype`. A surface file carries the same
JSON the Python CLI writes for that analysis (Appendix B of `PLAN.md`). Extra keys (`provenance`,
timings, the writer's duplicate collections `coselection_edges` / `epistatic_sectors` /
`selection_dms_plasticity`) are ignored; a key absent on both sides is noted as "absent"; a key
absent on one side is a violation.

`report.json`:

```
generated_at, repo_commit, reference, surfaces_requested, surfaces_present, surfaces_missing,
surface_dirs, examples, tree_free_examples, analyses, dms_examples, phenotype_examples, weights,
n_permutations, permulations, seed, classes,
runs[]          {surface: python|python-tn93, example, analysis, path, status: ok|failed|missing,
                 seconds, note, command}
self_checks[]   {surface, example, analysis: meme, checks[]}
comparisons[]   {surface, example, analysis, path,
                 status: pass|fail|missing|redirected|incomparable|no-reference,
                 violations, informational_excursions, surface_B, checks[], notes[]}
   checks[]     {field, class, n, violations, tol, max_abs_diff, max_ratio_to_tol, note,
                 first_violations[≤20], excursions, first_excursions[≤20]}
notes[]
summary         {reference_runs, reference_failed, self_check_violations, comparisons,
                 passed_comparisons, failed_comparisons, missing, redirected, incomparable,
                 violations, informational_excursions, strict_missing, passed}
```

`max_ratio_to_tol` is the worst `|Δ| / bound` of a check — how close a field came to its class;
the console line prints the worst one per comparison.

Exit codes: `0` every check within its class; `1` violations (or missing/incomparable files under
`--strict-missing`); `2` a reference run failed or the arguments were unusable.

## Running locally, end to end

Two repositories side by side: this one (`HyphAeon`, the engine) and `hyphaeon-app`, which links
the library via `file:../../HyphAeon/js`.

```bash
# 1. the surfaces (app repository): node + node-tn93 from the runner, browser + browser-tn93 from the e2e
cd hyphaeon-app
node runtime/scripts/parity-node.mjs --examples all \
     --analyses meme,busted,epistasis,dms,phenotype \
     --busted-examples all --dms-examples Smc6 --phenotype-examples RHO --threads 6
#   -> ../HyphAeon/parity/node/ and ../HyphAeon/parity/node-tn93/ (+ summary.json), ~2.5 min
(cd e2e && npx playwright test)             # -> ../HyphAeon/parity/browser/, browser-tn93/

# 2. the reference and the comparison (engine repository)
cd ../HyphAeon                              # package installed: pip install -e .
export HYPHAEON_WEIGHTS=$PWD/model.safetensors HF_HUB_OFFLINE=1
python scripts/parity.py --examples all \
       --surfaces python,node,node-tn93,web,web-tn93,mcp,mcp-tn93 --reuse
#   --reuse runs only the reference files that do not exist yet (~2 min for a cold parity/python*);
#   drop it to re-run every reference, or use --no-run to re-compare existing files only

# smaller slices
python scripts/parity.py --examples Smc6,bat_oas1 --surfaces python              # reference + self-check
python scripts/parity.py --examples all --surfaces python,node-tn93               # parity.yml step 1 shape (reference + tree-free reference; see the workflow for its three analysis slices)
python scripts/parity.py --no-run --surfaces python,node,web                      # recompare
python scripts/parity.py --examples Smc6 --surfaces python,node --strict-missing  # missing = failure
```

Options: `--analyses`, `--dms-examples` (default `Smc6`), `--phenotype-examples` (default `RHO`),
`--phenotype-fg`, `--permulations` (default 0), `--tn93-examples auto|none|<list>`, `--out <dir>`
(default `parity/`), `--weights`, `--n-permutations` (B; default 10,000), `--seed` (default 42),
`--no-run`, `--reuse`, `--strict-missing`. `browser` and `browser-tn93` are accepted as aliases of
`web` and `web-tn93`.

## Reading a failure

`report.json` → `comparisons[]` → the entry with `status: fail` → `checks[]` with
`violations > 0`. `first_violations` lists up to twenty `{key, ref, got, abs_diff, tol}`; `key`
is the site number, the `(site_u, site_v)` pair, the sector id, `(site, amino acid)` for a mutant
delta, or the field path. A failure in `site order` or an edge/sector order check suppresses the
per-item checks under it, since the items cannot be aligned. A `hyphaeon_lrt` failure with clean
`p_value`/`q_value` means the graph (or the tensors fed to it) differs and the statistics do not;
the reverse means the special functions or BH differ; both failing usually means the LRT. A
`redirected` or `incomparable` entry is not a failure: it says which run to look at instead, or
that the two files did not analyse the same taxa.

## State at 61d681a (phase-3a), 2026-09-05

`python scripts/parity.py --examples all --surfaces python,node,node-tn93,web,web-tn93,mcp,mcp-tn93 --no-run`
against the files the Phase 3 app runner and e2e produced:

```
reference runs: 23 (0 failed); self-check violations: 0; comparisons: 20 (20 pass, 0 fail);
missing: 41; redirected: 7; incomparable: 1; violations: 0; informational excursions: 1
PASS
```

Twenty comparisons pass: node on RHO, Smc6, bat_oas1 (meme, busted, epistasis), RHO phenotype and
Smc6 dms; node-tn93 on camelid and HIV1_RT (meme, busted, epistasis); web on Smc6 epistasis and
bat_oas1 meme; web-tn93 on camelid meme. The worst field in any comparison sits at 0.94 of its
bound (`null_coherence_std`, node Smc6 epistasis, both sides B = 10,000, 1.9 % relative); the worst
graph-class field is at 0.24 (`hyphaeon_lrt`, node bat_oas1 meme), the worst DMS delta at 0.27
(`max_delta_lrt`, node Smc6 dms), the worst derived field at 0.69 (`score`, node RHO phenotype). The one informational excursion is the browser's
`null_coherence_std` on Smc6 at B = 1,000 (3.2 % relative, the estimator's own error). The one
incomparable file is the browser's RHO phenotype (256 taxa against 655). The 41 missing are the
browser files the e2e does not write and the whole `mcp` / `mcp-tn93` surfaces.
