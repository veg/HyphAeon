# Preprint Review — Actionable Items

Scope: items that could block or undermine a preprint submission, **excluding**
model_eval results (tree sensitivity, FPR, concordance — will be addressed with
explanatory text), dead model params (explicit decision), BUSTED/aBSREL/ancestral
(future targets, not in preprint), and weights gating (plan in place).

---

## 1. `model_eval/README.md` is stale on Mode I vs Mode II

**File:** `model_eval/README.md`

The README says:
> "The repo implements Mode I. The paper describes Mode II. Mode II is not implemented in this repo."

This is no longer true. `hyphaeon/phenotype.py` implements Mode II — it uses
continuous Transformer Attribution Vectors (multi-head phylogenetic attention
attributions and branch projections), not binary substitution counts. Mode I
exists only in `model_eval/` as a baseline comparison.

**Action:** Update `model_eval/README.md` to reflect that Mode II is implemented
in the main codebase, and Mode I is used only as a baseline in model_eval.

---

## 2. Speedup claims inconsistent across docs

| Location | Claim |
|----------|-------|
| `README.md:26` | ">100× faster than standard numerical MLE" |
| `README.md:274` | ">10,000× faster than MLE" |
| `ARCHITECTURE.md:146` | "100×–250× faster" (L=1000, N=20) |
| `ARCHITECTURE.md:147` | "600×–1,100× faster" (L=1000, N=256) |

The README's two claims differ by 100×. The ARCHITECTURE.md numbers are
specific and benchmarked. The README should use the same range or cite
ARCHITECTURE.md.

**Action:** Pick one consistent speedup figure for README. The ">10,000×" on
line 274 appears to be an error or refers to a different benchmark; align with
ARCHITECTURE.md's 100×–1,100× range.

---

## 3. Citation block is placeholder

**File:** `README.md:310-315`

```bibtex
@article{hyphaeon2026,
  author={Kosakovsky Pond, Sergei L. and team},
  journal={Nature Methods / Nature Biotechnology (in submission)},
}
```

- Author list is "and team" — needs real authors
- Journal should be the preprint server (e.g., bioRxiv) with DOI
- "Nature Methods / Nature Biotechnology (in submission)" is speculative

**Action:** Replace with real author list and preprint citation once submitted.

---

## 4. README lists `busted` as a key capability — not in preprint scope

**File:** `README.md:31-32, 278`

BUSTED is listed as capability #4 in the README and in the CLI reference table.
It's implemented and tested (`tests/test_busted.py`), but if the preprint won't
claim it, the README should either:
- Mark it as preliminary/experimental, or
- Note it as an upcoming feature

Otherwise reviewers may ask for BUSTED validation data that doesn't exist yet.

**Action:** Add a note to the README that BUSTED is implemented but not
validated in the preprint, or remove it from the key capabilities list.

---

## 5. `aBSRELBranchHead` in model code but no CLI command

**Files:** `hyphaeon/model.py:728-731`, `model_config.json:15`

`aBSRELBranchHead` is defined in the model and listed as "pillar5_absrel" in
`model_config.json`, but there is no `absrel` CLI subcommand. If the preprint
doesn't mention aBSREL, this is just dead code that could confuse reviewers
browsing the repo.

**Action:** Either remove `pillar5_absrel` from `model_config.json` (it's
public-facing), or add a comment that it's a future target. The model class
itself can stay (it's in the checkpoint).

---

## 6. Frameshift input runs silently without warning

**File:** `model_eval/invariance/test_input_diagnostics.py:52-71`

The pipeline trims trailing nucleotides to a multiple of 3 but does not detect
or warn about frame errors. The model produces predictions on misaligned codons
without complaint. This is documented in the test but not in user-facing docs.

**Action:** Consider adding a frame-check warning in the alignment parser, or
at minimum document the assumption in README/ARCHITECTURE.md.

---

## 7. Deprecated function in CLI

**File:** `hyphaeon/cli.py:47`

`determine_adaptive_batch_size` is marked `# DEPRECATED: retained for
test/backcompat; delegates to compute_adaptive_safe_batch_size`. Not a
blocker, but if reviewers read the code, deprecated public functions suggest
the API is in flux.

**Action:** Either remove after confirming no external callers, or document
the deprecation path.

---

## 8. Code duplication (internal REVIEW.md items 1-6)

**File:** `REVIEW.md` (gitignored, internal)

6 categories of duplicated logic: model loading (10 copies), device selection
(7 copies), alignment loading (8+ copies), inference loop (6 copies),
attribution engine (2 copies), hypergeometric scan (2 copies). Stats functions
were already consolidated (items 7-9, fixed).

Not a scientific issue, but if the repo is linked from the preprint, code
quality will be scrutinized. The proposed `inference.py`, `weights.py`,
`io.py` modules would address this.

**Action:** Consider at least consolidating model loading and inference loops
before submission. The rest can be post-preprint cleanup.

---

## 9. Rhodopsin example missing tree file

**File:** `README.md:58`

The benchmark table says RHO tree is "Embedded / Auto", but `examples/RHO.nwk`
does not exist. The other 4 datasets all have explicit `.nwk` files. The
README's Example 3 (`hyphaeon phenotype -a examples/RHO.fasta ...`) doesn't
pass `-t`, so it relies on auto-detection or TN93 fallback. This works but is
inconsistent with the other examples and may confuse users trying to reproduce.

**Action:** Either include `RHO.nwk` in `examples/`, or clarify in the README
that RHO intentionally uses TN93 distance estimation (no tree).

---

## 10. `disease` and `filter` CLI commands undocumented in README

**Files:** `hyphaeon/cli.py:1225` (`disease` subcommand), `cli.py:1228`
(`filter` subcommand), `hyphaeon/__init__.py:18` (exports
`predict_disease_pathogenicity`)

The CLI has 8+ subcommands: `meme`, `epistasis`, `dms`, `busted`, `phenotype`,
`evaluate`, `disease`, `filter`, `ancestral`, `list-models`. The README only
documents 6 (`meme`, `epistasis`, `dms`, `busted`, `phenotype`, `evaluate`).
`disease` and `filter` are fully implemented with CLI parsers but:
- Not listed in the key capabilities or CLI reference table
- Have no dedicated example section
- `disease` has no unit tests (only referenced in `test_gpu_mem_guard.py`
  comments)
- `filter` has no unit tests

If the preprint mentions disease prediction or alignment filtering, these need
documentation and tests. If not, they should be marked as extras.

**Action:** Either document `disease` and `filter` in the README with examples
and add tests, or note them as experimental/extra features not covered in the
preprint.

---

## 11. `scripts/monitor_and_review_prs.py` committed to repo

**File:** `scripts/monitor_and_review_prs.py`

This is a GitHub PR auto-review/merge bot that reads GitHub tokens from git
credentials. It's a devops utility, not science code. Its presence in a public
repo linked from a preprint is unprofessional clutter and could raise security
questions (though it doesn't hardcode secrets).

**Action:** Move to `.github/scripts/` or remove from the repo before
preprint submission.

---

## 12. No test coverage for `disease`, `filter`, or `ancestral` CLI commands

**Files:** `tests/` directory

The test suite covers `meme`, `busted`, `phenotype`, `epistasis`, `evaluation`,
`weights`, `training`, `parsing`, `tokenization`, `batch_size`, `gpu_mem_guard`,
`distance_mds`, `integration`, `stats`, `ancestral` (unit tests for internal
functions), and `training_data`. However:
- **`disease` CLI**: No end-to-end or CLI integration test
- **`filter` CLI**: No tests at all (not even unit tests for `hyphaeon/filter.py`)
- **`ancestral` CLI**: Has unit tests for internal functions but no CLI
  integration test

If any of these are mentioned in the preprint, missing tests are a gap. If not
mentioned, they should be clearly marked as unvalidated extras.

**Action:** Add at least smoke tests for `disease` and `filter` CLI commands,
or mark them as experimental in the README.

---

## 13. README says "five complementary engines" but lists six capabilities

**File:** `README.md:22-23`

> "HyphAeon integrates five complementary phylogenetic deep learning and
> geometric projection engines, plus a pooled MEME concordance workflow:"

Then lists 6 items (meme, epistasis, dms, busted, phenotype, evaluate). The
"five engines + evaluate" framing means evaluate isn't counted as an "engine",
which is fine, but `busted` is also listed as one of the five. If BUSTED is
out of preprint scope, this count needs updating.

**Action:** Reconcile the count with what's actually claimed in the preprint.
If BUSTED is excluded, say "four complementary engines" and note BUSTED as
preliminary.

---

## 14. Hardcoded user paths in `scripts/` files

**Files:** `scripts/run_orthomam_masked_selection.py`,
`scripts/plot_yokoyama_rhodopsin_benchmark.py`, `scripts/calc_masked_fractions.py`,
`scripts/recreate_masked_alignments.py`

Multiple scripts contain hardcoded paths to `/Users/sergei/Projects/TOGA_MEME/...`,
including:
- `weights_file = '/Users/sergei/Projects/TOGA_MEME/hyphaeon_repo/model.safetensors'`
- `base_dir = '/Users/sergei/Projects/TOGA_MEME/benchmark/orthomam_v12'`
- `pd.read_csv('/Users/sergei/Projects/TOGA_MEME/yoko.csv')`
- Output paths to `/Users/sergei/Projects/TOGA_MEME/hyphaeon_paper/figures/...`

These are personal machine paths that won't work for anyone else. If the repo is
public, these scripts are broken for all external users.

**Action:** Either parameterize these scripts with argparse (like
`build_training_npz.py` already does), or move them out of the public repo.

---

## 15. `model.safetensors` (9.8 MB) committed directly to git

**File:** `model.safetensors` (repo root)

The pretrained weights file (~9.8 MB) is committed directly to the git repo.
This is unusual for model weights — they're normally hosted on Hugging Face
(which this repo already does via `datamonkey/hyphaeon`) or Git LFS. Committing
binary weights to git bloats the repo history and makes cloning slow.

**Action:** Remove `model.safetensors` from git tracking and add it to
`.gitignore`. The CLI already falls back to HF download when the local file
doesn't exist (`cli.py:43-44`).

---

## 16. `scripts/` directory contains analysis/plotting scripts not meant for public use

**Files:** `scripts/` directory (8 files on main)

The `scripts/` directory contains a mix of:
- `build_training_npz.py` — legitimate utility with argparse
- `monitor_and_review_prs.py` — GitHub PR bot (item #11)
- `run_avian_genome_filter.py` — benchmark analysis with hardcoded paths
- `run_orthomam_masked_selection.py` — benchmark analysis with hardcoded paths
- `plot_yokoyama_rhodopsin_benchmark.py` — paper figure generation with hardcoded paths
- `plot_avian_artifact_context_multipanel.py` — paper figure generation with hardcoded paths
- `calc_masked_fractions.py` — analysis script with hardcoded paths
- `recreate_masked_alignments.py` — analysis script with hardcoded paths

6 of 8 scripts have hardcoded paths and are effectively private analysis tools.
Only `build_training_npz.py` is a general-purpose utility.

**Action:** Move analysis/plotting scripts to a separate `paper/` or `analysis/`
directory that's clearly marked as supplementary, or remove them from the public
repo. Keep only `build_training_npz.py` in `scripts/`.

---

## 17. Hide `busted`, `absrel`, and `evaluate` subcommands from public-facing surface

**Files:** `hyphaeon/cli.py` (subcommand parsers + dispatch), `README.md`
(key capabilities list, CLI reference table, examples), `model_config.json`
(`pillar4_busted`, `pillar5_absrel`), `model.safetensors` (contains
`head_busted.*` and `head_absrel.*` keys), `tests/test_busted.py`,
`tests/test_evaluation.py`

BUSTED, aBSREL, and the MEME concordance evaluation workflow are not in the
preprint scope. Currently they're exposed in multiple public-facing places:

- **CLI:** `busted` (aliases: `omnibus`, `gene-selection`), `evaluate`, and
  `disease`/`filter` subcommands are all registered in the argparse dispatch
- **README.md:** BUSTED is listed as capability #4 (`README.md:31-32`), in the
  CLI reference table (`README.md:278`), and in Example 4; `evaluate` is
  listed as capability #6 and in the CLI table
- **`model_config.json`:** `pillar4_busted` and `pillar5_absrel` are listed
  in the description
- **`model.safetensors`:** The committed checkpoint contains `head_busted.*`
  (20 keys) and `head_absrel.*` (10 keys) — the HF "general" variant does
  NOT have these heads, so the local file is the only source of these weights
- **Tests:** `tests/test_busted.py` and `tests/test_evaluation.py` exist and
  run in CI

Reviewers browsing the repo will find these features and may ask for
validation data that doesn't exist in the preprint.

**Action:**
- Remove BUSTED and `evaluate` from the README key capabilities list, CLI
  reference table, and examples
- Remove `pillar4_busted` and `pillar5_absrel` from `model_config.json`
- Consider removing or gating the `busted` and `evaluate` subcommand parsers
  in `cli.py` (or at minimum removing their aliases and help text from the
  public `--help` output)
- Decide whether to keep `tests/test_busted.py` and `tests/test_evaluation.py`
  in CI (they validate code that won't be claimed in the preprint)
- The committed `model.safetensors` contains busted/absrel heads — if these
  features aren't claimed, consider stripping those keys and re-uploading a
  leaner checkpoint (or just rely on the HF variant which already lacks them)

**Implementation note — dev-only subcommands without separate branches:**
Gate the `busted`, `evaluate`, and `disease`/`filter` subparser registration
behind a `.git` directory check in `cli.py`:

```python
_IS_DEV = (Path(__file__).resolve().parent.parent / ".git").exists()
```

- Git clone / `pip install -e .` → `.git/` exists → all subcommands available
- `pip install .` / PyPI / bioconda → package in `site-packages/`, no `.git/`
  → only preprint-scoped subcommands registered
- sdist build → `.git/` not included in source dist → also minimal

Zero config for devs. README and `model_config.json` changes still need to
happen regardless (those ship in the package).

---

## 18. Version should be `0.1.0`, not `1.0.0`

**Files:** `pyproject.toml:7`, `hyphaeon/__init__.py:20`, `model_config.json:17`

All three currently say `1.0.0`. The first release should be `0.1.0` to signal
pre-release / early-stage software. The bibtex citation key in `README.md` is
`hyphaeon2026` — should also be updated to match the actual preprint year/DOI
when available.

**Action:** Set version to `0.1.0` in all three files before submission.

---

## 19. Orphaned example output files with no corresponding inputs or README mention

**Files:** `examples/REDIC1_edges.csv`, `examples/REDIC1_epistasis.json`,
`examples/bglobin_edges.csv`, `examples/bglobin_epistasis.json`,
`examples/rbcL_edges.csv`, `examples/rbcL_epistasis.json`

Six output files are committed in `examples/` for three datasets (REDIC1,
bglobin, rbcL) that have:
- No corresponding `.fasta` or `.nwk` input files
- No mention anywhere in `README.md`
- No test coverage

These appear to be leftover outputs from development or internal benchmarks.
Their presence in a public repo is confusing — users can't reproduce them and
they aren't referenced anywhere.

**Action:** Remove these 6 files from the repo.

---

## 20. BUSTED example output files committed in `examples/`

**Files:** `examples/HIV1_RT_busted.csv`, `examples/HIV1_RT_busted.json`,
`examples/Smc6_busted.csv`, `examples/Smc6_busted.json`

BUSTED output files are committed alongside the other example outputs. If
BUSTED is being hidden from the public-facing surface (item #17), these output
files should also be removed — they're discoverable and signal that BUSTED is
a supported feature.

**Action:** Remove BUSTED example output files, or move them to a dev-only
location.

---

## 21. `viral` variant exists on HF but is untested via auto-download

**File:** `.env.example:8`, `hyphaeon/weights.py`

The `viral` variant exists on Hugging Face (`datamonkey/hyphaeon`), and
`.env.example` correctly references it. However, all testing so far has been
against the `general` variant. The auto-download path in `weights.py` for
non-default variants has not been exercised end-to-end.

**Action:** Run a quick smoke test with `--model-variant viral` (or
`HYPHAEON_VARIANT=viral`) to confirm the download, cache, and load path works
correctly before submission.

---

## 22. Camelid dataset listed in benchmark table but has no example section

**File:** `README.md:61`

The Camelid VHH dataset (`examples/camelid.fasta`, `examples/camelid.nwk`) is
listed in the benchmark datasets table and has expected results in
`examples/expected_results/`, but has no corresponding "Example N" section in
the README showing how to run it. The other datasets (HIV-1 RT, Rhodopsin,
Smc6, Bat OAS1) all have dedicated example sections.

**Action:** Either add an example section for Camelid, or note that it's
included for testing/benchmarking only.

---

## 23. `epistasis` and `dms` subcommands missing `--model-variant` argument

**Files:** `hyphaeon/cli.py` — `epi_parser` (~line 1058), `dms_parser`
(~line 1085)

All subcommands that load model weights accept `--weights` to specify a local
checkpoint. Most also accept `--model-variant` to select which HF variant to
download. However, `epistasis` and `dms` have `--weights` but are missing
`--model-variant`:

| Subcommand | `--weights` | `--model-variant` |
|:---|:---:|:---:|
| `meme` | ✅ | ✅ |
| `phenotype` | ✅ | ✅ |
| `epistasis` | ✅ | ❌ |
| `dms` | ✅ | ❌ |
| `busted` | ✅ | ✅ |
| `disease` | ✅ | ✅ |
| `filter` | ✅ | ✅ |

This means users running `epistasis` or `dms` without `--weights` will always
get the default variant with no way to override via CLI. Appears to be an
oversight since the surrounding code pattern is identical.

**Action:** Add `--model-variant` to both `epi_parser` and `dms_parser`,
matching the pattern used by the other subcommands.

---

## 24. CLI UX inconsistencies across subcommands

**File:** `hyphaeon/cli.py`

Several user-facing inconsistencies in argument definitions across subcommands
that could confuse users or require breaking changes later:

### a. `--no-tree` and `--use-tn93` are redundant flags
Both flags do the exact same thing — the code treats them identically
(`use_tn93 = no_tree or use_tn93`). Both are present on all 6 tree-accepting
subcommands. Having two flags for the same action is confusing; pick one and
deprecate the other before the API stabilizes.

### b. `--cpu` help text inconsistent
`meme` says `"Force CPU inference"`, all 6 others say `"Force CPU execution"`.
Minor but visible in `--help`.

### c. `--max-species` (`-s`) missing on 4 subcommands, inconsistent default
Present on `meme` (default `None`), `busted` (default `512`), `filter`
(default `None`). Missing entirely on `phenotype`, `epistasis`, `dms`,
`disease`. The hardcoded default in `load_alignment_and_tree` is 512, so the
behavior is the same when the flag is absent, but users of the 4 missing
subcommands can't override it via CLI.

### d. `--no-prune-duplicates` only on `meme`
Duplicate taxon pruning is handled in `load_alignment_and_tree` (shared by all
subcommands), but only `meme` exposes `--no-prune-duplicates` to disable it.
Users of other subcommands can't turn it off.

### e. `-o` / `--output` semantics differ for `filter`
For 6 subcommands, `-o` means "output JSON results". For `filter`, it means
"output cleaned alignment FASTA". Users who learn the pattern from one
subcommand will be surprised. Consider `--output-aln` for filter.

### f. `--batch-size` (`-b`) help text varies
4 different wordings across subcommands: "auto-selected", "adaptive",
"adaptive hardware budget", "Number of codon sites to process in parallel".
Same parameter, different descriptions.

### g. `busted` has unique batch-directory mode (`-d`/`--dir`/`--pattern`)
`busted` is the only subcommand that can process a directory of alignments
(`--dir` + `--pattern`). All others accept only single alignments. If batch
mode is useful, it should be consistent across subcommands; if it's a BUSTED-
specific feature, it's another reason BUSTED differs from the rest (item #17).

**Action:** Normalize these before the API freezes. The highest-impact ones
are (a) redundant `--no-tree`/`--use-tn93`, (c) missing `--max-species`, (d)
missing `--no-prune-duplicates`, and (e) `-o` semantics for `filter`.

---

## 25. Verify author list before submission

**Files:** `README.md:310-315` (bibtex), `pyproject.toml:8-9` (authors field)

The bibtex citation block says `author={Kosakovsky Pond, Sergei L. and team}`
and `pyproject.toml` lists only one author. The real author list needs to be
finalized and both locations updated. Also update the bibtex key (`hyphaeon2026`)
and journal/preprint server fields once the preprint is submitted.

**Action:** Finalize author list, update `README.md` citation block and
`pyproject.toml` authors field.

---

## 26. Update READMEs to reflect that weights are no longer gated

**Files:** `README.md` (installation/usage sections), `model_eval/README.md`,
`.env.example`

Weights on HF (`datamonkey/hyphaeon`) are currently gated behind `HF_TOKEN`.
Once the repo goes public for the preprint, the HF repo should be ungated and
the docs updated to remove references to `HF_TOKEN` being required. Currently:

- `.env.example:5` says `HF_TOKEN=` with a comment about gated access
- `README.md` installation section doesn't mention `HF_TOKEN` (good) but also
  doesn't say weights download automatically on first run
- `hyphaeon/weights.py:128-131` raises an error mentioning `HF_TOKEN` if
  download fails — this message will be misleading once ungated

**Action:** After making the HF repo public: remove `HF_TOKEN` references from
`.env.example`, update the error message in `weights.py`, and add a note to
`README.md` that weights download automatically on first use.
