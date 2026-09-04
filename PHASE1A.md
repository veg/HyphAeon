# Phase 1a — what landed in `js/` after the five builders and the integration pass

Phase 1a of the plan of record (`../hyphaeon-app/PLAN.md`, §5.1–5.4) turned `@veg/hyphaeon-js`
from DataMonkey 3's seed port into a library that mirrors `hyphaeon/dataset.py`, `stats.py`,
`filter.py`, `attribution.py`, `evaluation.py`, the `cmd_busted` statistics and the CLI writers, with
a numeric kernel underneath and a pre-flight diagnostics module beside them. Every function is a
pure function; anything that needs the network takes an async `predict` callback. Nothing in
`hyphaeon/*.py`, `fixtures/` or `scripts/` changed. This page is the map: what is here, how each
check is run and what it printed, and what is knowingly carried into Phase 1b.

## What exists

| Path | What it is |
|---|---|
| `js/src/preprocess/` | `dataset.py` function for function: `parse.js` (FASTA/NEXUS/PHYLIP with CPython string semantics), `tokenizer.js` (61 sense codons 0..60 in TCAG order, stop/gap/unknown → 64; residues 0..19, else 20), `variability.js` (`dataset.py:718-723` rule), `tree.js` (a Biopython-1.85-faithful Newick parser, `extract_tree_from_string_or_file`, branch-length enforcement, three-tier taxon matching), `patristic.js` (float32 `compute_fast_dist_matrix`, the `> 10` rescale), `downsample.js` (duplicate pruning, Faith's PD, stride pre-selection), `mds.js` (dense path, float32 H/B, no sign convention), `assemble.js` (`loadAlignmentAndTree(alignmentText, treeText, options)` end to end with notices; `siteBatch`/`siteBatches`/`batchSizeFor`), `modelContract.js`. DM3's `newick.js` is gone. |
| `js/src/numeric/` | The kernel: `special.js` (lgamma, incomplete gamma/beta, erfc, χ²/t/normal survival, hypergeometric pmf/cdf/sf; worst measured error vs scipy 1.16 over 2,199 points 4.6e-13 relative), `prng.js` (xoshiro256\*\* + splitmix64, BigInt state, verified against the published vectors), `ranks.js`, `bh.js`, `cauchy.js` (float32-aware, as numpy 2 promotion makes them), `linalg.js` (Cholesky, eigenvalues-only tred2/tqli), `reduce.js` (numpy's pairwise summation order; added in integration to replace three identical private copies). |
| `js/src/stats.js` | `stats.py:16-48` plus `memeSitePq`, the `cmd_meme` float32 cast order (bit-exact against all 2,228 sites of the five e2e meme fixtures). |
| `js/src/filter.js`, `attribution.js` | `filter.py` and `attribution.py` as callback-driven pure functions, including the divergent second copy of the OCI screen in `cli.py cmd_meme --filter` behind `{cliVariant: true}`. |
| `js/src/omnibus.js` | `cmd_busted`'s statistical bridge (`cli.py:424-508`): ACAT, Simes, omnibus LRT with float32 pairwise reductions (bit-equal to the e2e fixtures), head-output decoding, `runBusted` over `predictSites`/`predictHead` callbacks. |
| `js/src/evaluate.js`, `writers.js` | `evaluation.py` over `{name, text}` pairs (csv.DictReader, `float()`, `html.unescape` semantics reproduced); writers byte-equal to `json.dump(indent=2)`, pandas `to_csv` and networkx's lxml GraphML (Python 3.14 / pandas 3.0.5 / lxml 6.1.0 / networkx 3.6.1 references under `js/test/data/evaluate/`). |
| `js/src/diagnostics.js` | PLAN.md §4.3 `diagnose()`: 23 codes, thresholds with sources in the header, `WORK_PER_SECOND` calibrated on HIV1_RT. No Python counterpart yet (PLAN §7 item 6). |
| `js/src/index.js` | The one entry point; `package.json` exports `.` only. 179 public names, pinned by `js/test/index.test.js`. |
| `js/test/` | 22 vitest files, 753 tests: fixture replays for every replayable fixture, plus Python-generated local references under `js/test/data/<module>/` (each with its `gen.py`). |
| `js/scripts/fixture-coverage.mjs` | Runs the suite once with a tracing setup file and reports which test file actually reads each `fixtures/**/*.json`, separating the manifest sanity walk from real replays. Development tool; outside the published `files`. |
| `js/src/README.md` | The module map, the `predict` callback convention, and the purity rule. |

## How to run each check, and what it printed

All commands from `js/` unless noted; Node 22, the scratch venv with scipy/numpy/torch for the
Python side, `HYPHAEON_WEIGHTS=$PWD/model.safetensors HF_HUB_OFFLINE=1`.

| # | Check | Command | Result |
|---|---|---|---|
| 1 | JS suite | `npm ci && npx vitest run` | `Test Files  22 passed (22)` / `Tests  753 passed (753)` |
| 2 | Typecheck | `npm run typecheck` | `tsc --noEmit -p tsconfig.json` silent |
| 3 | Public surface | `npx vitest run test/index.test.js` | 6 passed: 179 names, identity of re-exports, shared-name bindings, pipeline through the barrel, exports map serves `.` only (`ERR_PACKAGE_PATH_NOT_EXPORTED` for a deep import) |
| 4 | Fixture coverage | `node scripts/fixture-coverage.mjs` | `34/41 fixture files read by a test`; the 7 uncovered are listed below |
| 5 | Purity | `grep -rnE "onnxruntime\|node:\|\bfs\b\|fetch(\|Worker(\|process\." src/` outside comments | only `src/README.md` prose; no cycles among the 25 source files (checked with an import-graph walk) |
| 6 | Parity probe | Python `load_alignment_and_tree(examples/bat_oas1.fasta, bat_oas1.nwk)` vs `loadAlignmentAndTree` on the same text | L 351, N 18, taxa equal; `c` identical, `a` identical, invariable mask identical; `d` max\|Δ\| **0** (rescaled from raw max 123.32); `z` per column up to sign 1.5e-8 / 2.2e-8 / 2.6e-8 / 1.5e-7 (column magnitudes ~0.18); Gram max\|Δ\| 3.0e-8 — within the 1e-5 class |
| 7 | Attribution strings | `grep -rniE "claude\|anthropic\|co-authored\|generated with" src test scripts` | none (two "generated with numpy/Biopython" version notes) |
| 8 | Tree | `git status --short --untracked-files=all` at the repository root | only `js/` paths and `PHASE1A.md` |

## Integration changes (beyond wiring)

- `src/index.js` rebuilt in dependency order over 18 modules. Three names are exported by two
  modules each as the same binding (`symmetricEigen`, `benjaminiHochberg`, `rocAuc`); the ES
  linker keeps a shared name only when every `export *` supplies the same binding, and
  `test/index.test.js` asserts the identities so a copy can never slip in silently.
- `src/numeric/reduce.js` replaces the identical private copies of numpy's `pairwise_sum` that
  `numeric/cauchy.js`, `filter.js` and `omnibus.js` each carried; `float32Sum`,
  `numpyPairwiseSum` and `numpyMeanFloat32` are now kernel exports and `filter.js`/`omnibus.js`
  import them. `filter.js`'s private `memePvals` is gone; it imports `pvalsFromLrtMeme` from
  `stats.js` (no cycle: `stats.js` depends on `numeric/` only).
- `test/modelContract.test.js`: the three DataMonkey 3 pins (gap 64 / unknown 65, AA sentinels
  20/21/22, max_species 512) now pin `dataset.py`'s vocabulary (64 and 20; 256 default with a 512
  cap); the edits are marked in the file header.
- `test/index.test.js`: `PUBLIC_SURFACE` regenerated (179 names, grouped by module); the inline
  pipeline case uses `loadAlignmentAndTree` + `siteBatch` and asserts the serine site invariable, as
  `dataset.py` has it.

## Public API

Grouped by module; signatures are in each module's JSDoc. `Loaded` is the object
`loadAlignmentAndTree` returns; `predict(c, a, meta)` is the callback described in `src/README.md`.

**preprocess** — `parseAlignmentSequences`, `pyStrip`/`pyRstrip`/`pyLstrip`/`pyStripChars`/`pySplit`/
`pySplit1`/`pySplitLines`/`pyIsAlnum`/`pyIsDigit`/`pyLen`, `PY_WS`, `PY_NOT_WS`; `CODON_LIST`,
`GENETIC_CODE`, `AA_MAP`, `CODON_TO_AA`, `codonToken`, `aaToken`, `tokenizeSequence`;
`isAaInvariable`, `invariableMask`, `isSiteVariable`, `siteVariability`; `NewickError`,
`parseNewickTrees`, `readNewick`, `findClades`, `getTerminals`, `stripQuotes`, `treeTaxa`,
`hasNonzeroBranchLengths`, `needsBranchLengths`, `enforceNonzeroBranchLengths`, `extractTree`,
`matchTaxa`; `rootDistances`, `patristicRow`, `patristicMatrix`, `computeFastDistMatrix`,
`rescaleDistances`; `pruneIdenticalSequences`, `downsampleTaxaFaithPd`, `stridePreselect`;
`symmetricEigen`, `computeMdsCoordinates`; `loadAlignmentAndTree(alignmentText, treeText, {maxSpecies,
pruneDuplicates, referenceName, useTn93})`, `siteBatch`, `siteBatches`, `batchSizeFor`; the
`modelContract` constants (`CODON_*`, `AA_*`, `NUM_*_TOKENS`, `MAX_SPECIES_DEFAULT` 256,
`MAX_SPECIES_CAP` 512, `WINDOW_SIZE_DEFAULT`, `MDS_COMPONENTS`, `INPUT_SPEC`, `INPUT_NAMES`,
`OUTPUT_SPEC`, `OUTPUT_SPEC_V1`, `OUTPUT_NAMES_V1`, `VERIFIED_MODEL_SHA256`) and `validateInputBundle`.

**numeric** — `lgamma`, `gammaincReg`, `gammaincc`, `betaincReg`, `erfc`, `chi2Sf`, `chi2Cdf`, `tSf`,
`tCdf`, `normSf`, `normCdf`, `logChoose`, `hypergeomPmf`, `hypergeomCdf`, `hypergeomSf`;
`Xoshiro256(seed)` with `next64`, `next`, `uniform`, `normal`, `integers`, `shuffle`,
`choiceWithoutReplacement`; `rankdata`, `pearson`, `spearman`, `rocAuc`; `benjaminiHochberg`,
`cauchyCombination`; `cholesky`, `symmetricEigenvalues`, `largestEigenvalue`; `numpyPairwiseSum`,
`float32Sum`, `numpyMeanFloat32`.

**stats** — `pvalsFromLrtMeme`, `pvalsFromLrtSelfLiang`, `cauchyCombinationP`, `memeSitePq`.

**filter** — `scanHypergeometricPatches`, `predictSiteLrts(loaded, predict, opts)`,
`consensusCodons`, `auditPatch`, `maskCodonSpan`, `fastaText`, `runAlignmentFilter({alignmentText,
treeText, loaded?, baseLrts?}, predict, opts)`.

**attribution** — `attributeSelection(loaded, predict, {focalSites, minLrt, baseLrts, taxa, batchSize})`,
`attributionSiteFields`, `attributionsOneIndexed`, `INV_GENETIC_CODE`.

**omnibus** — `variableSiteIndices`, `totalSelectionEnergy`, `omnibusLrt`, `simesP`,
`bustedStatistics`, `bustedHeadFields`, `bustedVerdict`, `bustedRecord`, `gatherSiteBatch`,
`runBusted(loaded, {predictSites, predictHead}, opts)`, `BUSTED_*` constants.

**evaluate** — `EvaluationError`, `loadPredictionCsv`, `loadMemeJson`, `matchGeneFiles`,
`correlations`, `confusion`, `ppv`, `fpr`, `thresholdMetrics`, `evaluatePairs`, `evaluateDirectories`,
`evaluateFiles`, `formatTextReport`, `parseCsvRows`, `dictReaderRows`, `normalizedHeader`, `rocAuc`.

**writers** — `memeSiteRecords`, `memeResult`, `memeJson`, `memeCsv`, `bustedJson`, `bustedCsv`,
`epistasisCsv`, `dmsCsv`, `phenotypeCsv`, `graphml`, `resultJson`, `evaluateJson`, `dataFrameCsv`,
`pyJsonDumps`, `pyRepr`, `pyStr`, `pyFloatRepr`, `pyFormatFixed`, `pyFormatG`, `PY_FLOAT_KEYS`,
`PY_INT_KEYS`.

**diagnostics** — `diagnose({alignmentText, treeText, parsed?, maxSpecies?, taxaLimit?})`,
`DIAGNOSTIC_CODES`, `DIAGNOSTIC_THRESHOLDS`, `WORK_PER_SECOND`, `sniffAlignmentFormat`,
`meanPairwiseDivergence`, `medianOffDiagonal`.

## Fixture coverage

`node scripts/fixture-coverage.mjs` traces `readFileSync` under `fixtures/` during one run of the
suite and attributes each read to the test file on the stack. Reads made by the manifest sanity
walk in `fixtures.test.js` (which opens every file to assert no absolute path leaked) are reported
separately and do not count.

| Fixture | Replayed by |
|---|---|
| `dataset/*.json` (10 files, 54 cases) | `fixtures.test.js` — tokens, names, distances exact/1e-6; MDS up to per-column sign + Gram |
| `stats/*.json` (4) | `stats.test.js`, `numeric-stats.test.js` — 1e-9 class, measured ≤ 2.6e-15; float32 cases replayed with `Float32Array` |
| `filter/scan_hypergeometric_patches.json` | `filter.test.js` — ints exact, `p_local` 1e-9 |
| `attribution/attribute_selection.json` | `attribution.test.js` — with a playback `predict` reconstructed from the recorded `delta_lrt` (see gaps) |
| `evaluation/*.json` (6) | `evaluate.test.js` (+ `roc_auc` in `numeric-ranks.test.js`) — exact / 1e-9, measured 1e-13 |
| `e2e/meme_*.json` (6) | `stats.test.js` (p/q bit-exact from the fixture LRTs), `filter.test.js`, `attribution.test.js`, `omnibus.test.js`, `writers.test.js` |
| `e2e/busted_*.json` (2) | `omnibus.test.js` — `total_selection_energy` and `omnibus_lrt` bit-equal, ACAT/Simes 1e-12, from the meme fixtures' per-site LRTs |
| `e2e/filter_*.json` (2) | `filter.test.js` — patches, artifact records, masked ranges and cleaned-FASTA sha256 exact; raw metrics 1e-5 |
| `e2e/epistasis_Smc6_n_permutations_1000.json`, `e2e/phenotype_RHO_marine_n_permutations_0.json` | `writers.test.js` only — the CSV/JSON/GraphML writers are checked against these documents; the methods are not ported (Phase 1b) |

**Uncovered (7 files), and why no replay was added.** All belong to modules the plan schedules for
Phase 1b (order 4–5 in PLAN.md §5.2); there is no JS function to replay them against yet, and
writing one would be porting the module, not adding a test.

| Fixture | Cases | Class | Needs |
|---|---|---|---|
| `epistasis/compute_branch_coselection_network.json` | 5 | 1e-6 / exact | `epistasis.js` (cosine network, Student-t p via `tSf`, BH, CESI, APC). Inputs are recorded attribution matrices, so it is replayable without a model once the module exists. |
| `epistasis/extract_epistatic_sectors_tse.json` | 8 | 1e-6 / exact / statistical | `sectors.js` (connected components → greedy modularity with networkx tie-breaking → spectral coherence via `largestEigenvalue`). |
| `epistasis/compute_sector_permutation_test.json` | 5 | statistical / exact | the permutation null with `Xoshiro256`; compared statistically (PLAN §5.4). |
| `phenotype/resolve_phenotype_vector.json` | 15 | exact / 1e-9 | `phenotype.js` trait resolution (presets, regex, CSV). Not model-dependent. |
| `phenotype/compute_phylogenetic_covariance.json` | 3 | 1e-9 | `phenotype.js` (root-to-MRCA distances over `tree.js`). Not model-dependent. |
| `phenotype/generate_permulations.json` | 3 | statistical | `permulations.js` (Cholesky of the covariance + Gaussian draws; numpy legacy MT19937 upstream, so statistical only). |
| `dms/run_insilico_selection_dms.json` | 2 | 1e-5 | `dms.js` and the real model: every output is a forward pass. A Phase 1b runtime test. |

**Fixture defects reported by builders, not regenerated.** None is a wrong value; each is a
generator convention the replay had to work around, so the fixtures were left as generated and
`scripts/gen_fixtures.py` untouched:

1. `dataset/compute_mds_coordinates.json` declares 1e-5 *absolute* for
   `example_bat_oas1.nwk_raw_distances`, whose coordinates are ~60 on the unrescaled matrix and
   whose reference is float32 (ULP ~4e-6 per coordinate). Measured JS-vs-Python difference is
   8e-6..9e-5 per column, 1.5e-6 relative. `fixtures.test.js` applies 1e-5 relative to the
   compared magnitude with a floor of 1 (plain 1e-5 absolute for every rescaled case). Suggest the
   generator record MDS tolerance as relative, or restrict the class to rescaled inputs.
2. `stats/benjamini_hochberg.json` and `stats/cauchy_combination_p.json` do not record the input
   dtype; the two float32 cases (`float32_meme_like`, `float32_input`) miss a float64 replay by
   4.2e-8 and 9e-8. The tests select float32 arithmetic by the case-name prefix. Suggest
   `inputs.dtype`.
3. `e2e/busted_*.json` carry no per-site LRTs or invariable mask; the replay takes them from
   `e2e/meme_*.json` (verified bit-exact with Python). Suggest `outputs.site_lrts` and
   `is_invariable` on the busted cases.
4. `attribution/attribute_selection.json` and the e2e filter/meme cases record only model outputs,
   so the replays use playback predicts (per-site LRTs from the meme fixtures; counterfactual LRTs
   as `predicted_lrt - delta_lrt`). The real-model replay at the 1e-5 class is a runtime test.

## Python quirks replicated (pinned by tests; fix upstream first)

The full lists are in the module headers; the ones a consumer will meet:

- `cli.py cmd_meme --filter` is a second copy of the OCI screen that disagrees with
  `filter.run_alignment_filter` in six ways (consensus keeps `?` and skips the length check;
  `min_patch_consec` has no flag; an embedded tree plus ≥ 1 artifact crashes with "No tree
  specified"; the cleaned re-score reuses the baseline tree cache; `--attribute` runs on the
  original tokens with the cleaned LRTs; p/q are float32). All behind `{cliVariant: true}`.
- `attribution.py` maps token 64 back to `TGA`, so a gap/unknown consensus reports codon `TGA`, AA `*`.
- `dataset.py`: PHYLIP multi-line records parse to garbage; a quoted NEXUS label with a space
  splits; `TREE x = [&R] (...)` is rejected; `enforce_nonzero_branch_lengths` raises negative lengths
  to 1e-4; `compute_fast_dist_matrix` leaves zero rows for taxa matched by tier 2/3 name matching;
  the `> 10` rescale runs after enforcement; `downsample_taxa_faith_pd(max_species=1)` returns two
  taxa; the stride pre-selection keeps the first `2·max_species` taxa.
- `evaluation.py`: `fpr_confusion_matrix` is the same computation as `ppv_confusion_matrix`;
  `_boolean` and `_finite_float` tolerate different whitespace; `max(0.0, -0.0)` leaves -0.0.
- `cmd_busted`: Simes runs over all sites (invariable at p = 1) while ACAT runs over variable sites
  only; the 3.841 threshold is rounded; dead fallbacks for absent head outputs are kept.
- `stats.py`, `numeric`: none found. scipy's own `betainc(0.5, 0.5, 1-1e-12)` is 3.5e-11 off mpmath;
  the kernel matches mpmath and `test/data/numeric/special.json` records the two deviating points.

## Known gaps for Phase 1b

1. **Branch lengths and TN93.** `estimate_tree_branch_lengths_hyphy` (`dataset.py:224-287`) and the
   TN93 path (`dataset.py:443-521, 544-580`) shell out and are not library functions.
   `loadAlignmentAndTree` takes the "HyPhy not found" branch (1e-3 / 1e-4 defaults) and reports
   `notices.branchLengthsMissing`; `needsBranchLengths(extractTree(text))` is the predicate. Both
   `examples/camelid.nwk` and `examples/HIV1_RT.nwk` have no branch lengths, so their e2e fixtures
   went through HyPhy; the runtime must supply an estimated tree to reproduce those distances.
2. **Lanczos MDS for N > 500** (`dataset.py:360-381`) is not ported; the library runs dense. At the
   app's cap of 512, N in 501..512 is where the reference returns an iterative approximation of
   what JS computes exactly.
3. **Real-model replays.** `attribution/attribute_selection.json`, `e2e/filter_camelid.json`
   `cleaned_metrics`, and `dms/run_insilico_selection_dms.json` need the ONNX `predict`; the test
   files mark where playback predicts stand in.
4. **Neural BUSTED head.** `model.safetensors` lacks 11 `BustedMultiTaskHead` parameters and
   `cmd_busted` loads unseeded, so the neural fields are null in the fixtures. `bustedHeadFields`
   reproduces the arithmetic only; `busted_head.onnx` is one seeded draw. `runBusted` passes an
   all-zero mask, as `cmd_busted` calls the head with `mask=None`.
5. **`epistasis.py`, `phenotype.py`, `dms.py`** — the seven uncovered fixtures above. `numeric/`
   already provides `tSf`, `normSf`, `cholesky`, `largestEigenvalue`, `Xoshiro256` and
   `benjaminiHochberg` for them; the networkx greedy-modularity semantics are the new work.
6. **Diagnostics calibration.** `SHALLOW_TREE` (median patristic < 0.05) flags Smc6, the general
   model's own regime; `STAR_LIKE` fires for every panel with < 5 haplotypes; `COST_ESTIMATE` uses
   `L·N²/4.7e6` from CPU torch and needs a WASM constant. Product review before the panel ships;
   PLAN §7 item 6 should give the Python `diagnose()` the same numbers.
7. **Fixture generator conventions** (the four items above): relative MDS tolerance, `inputs.dtype`
   for float32 stats cases, per-site LRTs on the busted cases, and an `evaluate_directories` /
   writer-output case so writer byte-parity is pinned by `fixtures/` rather than only by
   `js/test/data/evaluate/`.
8. **Writers and Python's int/float.** `PY_FLOAT_KEYS`/`PY_INT_KEYS` in `writers.js` decide `0`
   vs `0.0`; a new upstream float field prints as an int until it is added. `bustedCsv` writes an
   empty `Time_ms` where the Python would raise on a null `elapsed_seconds`.
9. **Runtime contract reminders.** Pass `maxSpecies` explicitly (dataset.py applies no cap unless
   asked); call `memeSitePq` before `memeSiteRecords` so p/q are the float32 casts `cmd_meme`
   writes; feed `busted_head.onnx` the all-false mask from `runBusted`; use
   `batchSizeFor(N)` for the attribution counterfactual batches.
