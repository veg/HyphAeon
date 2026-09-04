# Phase 0 — what exists in `veg/HyphAeon` after the scaffold

Phase 0 of the plan of record (`../hyphaeon-app/PLAN.md`, §8) put the JavaScript library, the
ONNX export, the fixture generator, the parity harness and the CI/release scaffolding into this
repository beside the Python reference. This page is the map: what is here, how each check is
run, and what is knowingly carried into Phase 1. Nothing here changes a Python method; the only
Python additions are `hyphaeon/export.py` and the `export-onnx` subcommand in `hyphaeon/cli.py`.

## What exists

| Path | What it is |
|---|---|
| `js/` | `@veg/hyphaeon-js` 1.0.0: a **library** of pure functions mirroring `hyphaeon/*.py`. Phase 0 seeds it with DataMonkey 3's preprocessing port (newick, patristic, symmetric eigen, MDS, tokenizer, tensor assembly, model contract, site variability), byte-identical bodies with provenance headers. No dependencies, no onnxruntime, no I/O. `src/README.md` states the ownership rule. |
| `js/test/` | 8 vitest files, 121 tests: DM3's 109 cases, the exports-map guard, and `fixtures.test.js` (fixture manifest validation + the tokenizer replay, pinned as a divergence — see gaps). |
| `hyphaeon/export.py`, `scripts/export_onnx.py` | `hyphaeon export-onnx --variant {general,viral,all}`: opset 17 backbone graphs with inputs `msa_codons`, `msa_aas`, `dist_matrix`, `mds_coords` and outputs `lrt`, `mean_root_attns`, `root_repr` (dynamic batch and `num_species`), `busted_head.onnx`, and `models/manifest.json` with sha256 hashes. |
| `models/` | `general.onnx`, `viral.onnx` (7.7 MB each), `busted_head.onnx` (2.0 MB), `manifest.json`. **Committed on purpose** (PLAN §5.5). `models/_hf/` (HF downloads) is ignored. |
| `scripts/verify_onnx.py` | PyTorch (`forward_cached`) vs onnxruntime on Smc6 (20 taxa) and camelid (212 taxa) for both variants, viral vs HF `model.viral.onnx`, the BUSTED head vs the module, and every manifest hash. |
| `scripts/gen_fixtures.py`, `fixtures/` | 174 cases in 41 JSON files (2.9 MB) across stats, filter, evaluation, epistasis, phenotype, dataset, attribution, dms and CLI e2e; `fixtures/manifest.json` records the engine commit, weights hash, environment, tolerance classes and the Python quirks the fixtures pin; `fixtures/README.md` documents the format. |
| `scripts/parity.py`, `PARITY.md` | End-to-end comparator: runs the Python surface on the examples and compares any `node`/`web`/`mcp` surface files the app repository produces under `parity/` (gitignored), by the tolerance classes of PLAN §5.4. |
| `.github/workflows/js.yml` | Node 22: `npm ci`, `npm test` (vitest + fixture replay), `npm run typecheck` on pushes/PRs touching `js/`, `fixtures/`, `models/`. |
| `.github/workflows/parity.yml` | Python + Node; `scripts/parity.py --examples all --surfaces python,node`; uploads `parity/`. |
| `.github/workflows/release.yml` | On tag `v*`: checks `js/package.json` and `pyproject.toml` carry the tag version and `models/manifest.json` exists; publishes to PyPI (trusted publishing) and npm (`--provenance`); attaches the manifest to the GitHub release. |
| `CODEOWNERS` | `hyphaeon/`, `js/`, `fixtures/`, `models/` share one reviewer set (placeholder team). |
| `README.md` | Appended section "JavaScript library (@veg/hyphaeon-js)". |

## How to run each check

All commands from the repository root, with the engine installed editable in a venv
(`pip install -e .[dev]`), torch on CPU, and:

```bash
export HYPHAEON_WEIGHTS=$PWD/model.safetensors HF_HUB_OFFLINE=1
```

| # | Check | Command | Expected |
|---|---|---|---|
| 1 | Python suite | `python -m pytest tests/ -q` | `252 passed, 2 skipped` |
| 2 | ONNX parity | `python scripts/verify_onnx.py` | `[✓] All checks passed`; lrt max Δ ≤ 1.2e-05 (scaled ≤ 2.4e-06), `mean_root_attns` ≤ 9e-08, `root_repr` ≤ 3.4e-06; every manifest hash matches |
| 3 | JS library | `cd js && npm ci && npm test && npm run typecheck` | `Test Files 8 passed / Tests 121 passed`; tsc silent |
| 4 | Fixtures | `python scripts/gen_fixtures.py && find fixtures -name '*.json' \| wc -l` | `total 2851 KB in ~75s`; 45 JSON files (41 fixtures + manifest + 3 evaluation inputs). Needs `hyphy` on PATH for the camelid cases. |
| 5 | Parity, Python side | `python scripts/parity.py --examples Smc6,bat_oas1 --surfaces python` | `reference runs: 4 (0 failed); ... violations: 0` / `[parity] PASS` |
| 6 | Re-export (only when the graph or weights change) | `hyphaeon export-onnx --variant all` then check 2 | rewrites `models/*.onnx` and `manifest.json`; the app's `runtime/` tests read the new hashes |

`git status --short --untracked-files=all` after all of the above should show nothing under
`models/_hf/`, `parity/`, `js/node_modules/`, `results/` or `__pycache__/`.

## Gaps carried to Phase 1

Ordered by how much they change numbers a user sees.

1. **Tokenizer vocabulary.** `js/src/preprocess/tokenizer.js` is DM3's 64-codon TCAG table with
   separate gap/unknown tokens (64/65) and amino-acid sentinels 20/21/22. `hyphaeon/dataset.py`
   numbers the 61 sense codons 0..60, sends stops/gaps/unknown to 64, and has one amino-acid
   sentinel (20). `js/test/fixtures.test.js` pins the divergence (54/64 codon tokens differ; the 20
   residues agree). The app's bat_oas1 experiment (`../hyphaeon-app/CLAUDE.md`) shows the shipped
   checkpoint wants dataset.py's tables: switching them (with item 2) lifts variable-site Spearman
   vs `hyphaeon meme` from 0.13 to 0.94 on the viral variant. Change the library, then rewrite the
   pinned test as a plain replay.
2. **`>10` patristic rescale** (`dataset.py:679-681`, divide by codon length when the matrix max
   exceeds 10) and **`prune_identical_sequences`** are not in the library. `fixtures/dataset/
   rescale_rule.json` and `prune_identical_sequences.json` are ready to replay.
3. **Site variability**: the serine two-family rule in `variability.js` is not in `dataset.py`,
   and stops count as observations in JS but not in Python. Changes which sites are scored.
4. **MDS**: padded-matrix vs real-N double-centring (measured negligible on bat_oas1), no sign
   canonicalisation in `dataset.py`, float32 vs float64 centring, and scipy's Lanczos path above
   500 taxa. `fixtures/dataset/compute_mds_coordinates.json` carries the Gram matrix for a
   sign-insensitive comparison.
5. **Max-PD seeding and the `2*max_species` pre-stride** differ (`dataset.py:406`, `:670-674`);
   `MAX_SPECIES_DEFAULT` is 512 in `modelContract.js` vs `max_species=256` in `model.py`.
6. **No fixture replay beyond the tokenizer.** Every other fixture (stats, filter, evaluation,
   epistasis, phenotype, attribution, dms, e2e) waits for its port (PLAN §5.2 order). The
   `fixtures.test.js` manifest check keeps the files honest until then.
7. **BUSTED neural head is non-deterministic upstream.** `model.safetensors` lacks 11
   `BustedMultiTaskHead` parameters (`in_proj`, `head_prop`, `head_syn_var`, `coral_*.theta_steps`)
   and `cmd_busted` loads with `strict=False`, unseeded. `busted_head.onnx` was exported after
   `torch.manual_seed(0)`; its numbers are one of the reference's possible outputs. The e2e
   fixtures null those fields. Needs an upstream checkpoint fix, then a re-export.
8. **Upstream PRs 3–8 of PLAN §7 are not opened**: `hyphaeon.api` entry points and typed
   exceptions, `--progress-json`, `schema_version`/`provenance` in results, `diagnose()`,
   `--seed` on `epistasis`/`phenotype` (parity.py skips epistasis until it exists),
   `list-models --json`.
9. **Tolerance note**: PLAN §7 item 1 asks for ONNX parity ≤ 1e-6; fp32 gives 1.2e-05 absolute
   on one camelid site at LRT 5.02 (torch `forward` vs `forward_cached` differ by 6.7e-06 on the
   same inputs), so `verify_onnx.py` asserts `1e-5 * max(1, |lrt|)`.
10. **Release plumbing** is untested end to end: PyPI trusted publishing, the `pypi` environment,
    `NPM_TOKEN`, and a real team in `CODEOWNERS` are needed before the first `v*` tag. The
    `parity.yml` `node` surface has no runner yet (it lives in the app repository).
11. **Reference bugs found and replicated, not fixed** (all recorded in `fixtures/manifest.json`
    `known_quirks`): PHYLIP multi-line records mis-parse; NEXUS quoted labels split on spaces;
    `TREE x = [&R] (...)` rejected; `cmd_meme --filter` is a second copy of the OCI logic;
    epistasis CLI/library `min_sim` defaults disagree; `extract_epistatic_sectors_tse` ignores
    `max_overlap`; `resolve_phenotype_vector` ignores `background`; `PhyloAxialTransformer.forward()`'s
    micro-batch branch mis-slices per-batch `dist_matrix`/`mds_coords` (not on the CLI path).
12. `examples/` has five alignments, not the six PLAN §3.3 mentions.
