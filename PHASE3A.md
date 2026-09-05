# Phase 3a — tree-free TN93, the phenotype pillar, and no unported method left

Phase 3a of the plan of record (`../hyphaeon-app/PLAN.md`, §5.1–5.4 and D22) closed the last two
holes in `@veg/hyphaeon-js`. It ported **`hyphaeon/phenotype.py`** — trait vectors, the directional
PhyloWAS driver, trait sectors, Brownian-motion permulations — and it ported the reference's own
**tree-free TN93 distance path**, which under D22 is no longer an option but the default whenever a
usable tree is absent. HyPhy is now unreachable from the library: a missing tree and a tree without
branch lengths both take TN93 distances into the MDS instead of raising or asking the runtime to
estimate lengths.

Every method module PLAN.md §5.1 lists is now ported. The library exports **232 public names**, up
from 204, and reads **48 of 48** fixture files.

## What landed

### The library (`js/`, 232 public names)

| Path | What it is |
|---|---|
| `js/src/preprocess/tn93.js` (683 lines) | `dataset.py:493-571` (`compute_tn93_distance_matrix`: the sentinel, the two imputation rules, float32 assembly) **plus the distance itself**, which the reference does not write but imports — the `tn93` PyPI package 1.2.2's `get_counts("resolve")` → `get_nucleotide_frequency` → `calculate_distance`, including its 256-entry character map, the 18×4 resolution table and the six-significant-digit rounding. `tn93Distance`, `tn93DistanceMatrix`, `tn93Counts`, `tn93SaturatedPairs` and the tables. |
| `js/src/phenotype.js` (1,266 lines) | `phenotype.py:48-274, 347-646`. `PRESETS` transcribed verbatim; `resolvePhenotypeVector` (presets, inline list / regex / glob, CSV or TSV, continuous); `runPhenotypeAssociation` over an async `{lrt, attention}` callback — the 21 keys of `phenotype.py:624-646` in order. The attribution matrix comes from `epistasis.js` and trait sectors from `sectors.js`, exactly as the reference imports them, so the two attention pillars share one definition. |
| `js/src/permulations.js` (358 lines) | `phenotype.py:275-346`. `computePhylogeneticCovariance` (root-to-tip diagonal, root-to-MRCA off-diagonal, Biopython `find_any` regex semantics included) and `generatePermulations` (Cholesky of V + 1e-7 I, Gaussian draws through `Xoshiro256`, rank-matched to a binary or continuous trait). A tree-free caller gets `permMatrix: null` and `skipped.reason` rather than an exception. |
| `js/src/preprocess/assemble.js` | `loadAlignmentAndTree` gained the tree-free branch (`dataset.py:598-636`) with the reference's step order. `tree` may now be null, `matchTier` may be null, and `notices` gained `treeFree: {reason, taxaOrder}` and `tn93SaturatedPairs`. |
| `js/src/diagnostics.js` | `TREE_MISSING` (refuse) and `BRANCH_LENGTHS_MISSING` (warn) are gone; `TREE_FREE_TN93` (info, with `data.reason`) and `TN93_SATURATED_PAIRS` replace them. `diagnose` takes `useTn93`. |
| `js/test/` | 29 files, 1,067 tests (was 26 / 891). New: `tn93.test.js` (52), `phenotype.test.js` (83), `permulations.test.js` (41), plus Python-generated references under `js/test/data/phenotype/` with their `gen.py`. |

### The fixtures (`fixtures/`, `scripts/gen_fixtures.py`)

Nine new files, all generated from the Python: `dataset/tn93_distance.json` (24 pairs),
`dataset/tn93_distance_matrix.json` (8 matrices), `dataset/load_alignment_and_tree_tn93.json` (2),
and four `e2e/*_tn93.json` CLI runs. `fixtures/README.md` gained a "TN93 and tree-free mode"
section; `manifest.json` records `tn93_package` 1.2.2 and `tn93_binary_present_but_hidden` v1.0.15.

**The generator hides the compiled `tn93` binary** from `shutil.which` and from the CLI
subprocesses' PATH, because there IS one on this machine (`/usr/local/bin/tn93`) and the reference
prefers it when it is on PATH. Measured: the binary and the package agree exactly — max |Δ| = 0.0 on
`bat_oas1` and `HIV1_RT` — so the fixtures pin the package path and reproduce on a machine with or
without the binary.

## How to run each check, and what it printed

All JS commands from `js/`; Node 22; Python is the scratch venv with
`HYPHAEON_WEIGHTS=$PWD/model.safetensors HF_HUB_OFFLINE=1`.

| # | Check | Command | Result |
|---|---|---|---|
| 1 | Python suite | `python -m pytest tests/ -q` | `252 passed, 2 skipped, 1 warning in 28.84s` — unchanged by the generator's edits |
| 2 | JS suite | `npm ci && npx vitest run` | `Test Files  29 passed (29)` / `Tests  1067 passed (1067)` / `Duration 5.80s` |
| 3 | Typecheck | `npm run typecheck` | `tsc --noEmit -p tsconfig.json` silent |
| 4 | Public surface | `npx vitest run test/index.test.js` | `Test Files  1 passed (1)` / `Tests  6 passed (6)` — 232 names, re-export identities, a tree-free load through the barrel, `ERR_PACKAGE_PATH_NOT_EXPORTED` for a deep import |
| 5 | Fixture coverage | `node scripts/fixture-coverage.mjs` | **`48/48 fixture files read by a test`** — nothing uncovered, first time in the port |
| 6 | Purity | `grep -rnE "onnxruntime\|node:\|\bfs\b\|fetch(\|Worker(\|process\.\|require(" src/` | 8 hits, **all inside comments**; 0 in code. `Math.random` 0; the two `Date.now` hits are `filter.py`'s own `elapsed_seconds` |
| 7 | Tree-free parity probe | Python `load_alignment_and_tree(use_tn93=True)` vs `loadAlignmentAndTree(text, null, {useTn93:true})` | bat_oas1 and camelid: `d` **bit-identical**, `z` worst 7.8e-8 with no sign allowance, taxa and tokens identical (below) |
| 8 | Phenotype probe | `run_phenotype_association` on RHO with the README marine foreground vs `runPhenotypeAssociation` on its captured attention | `TOTAL FAILURES: 0` — 21 keys in order, 145 site rows, 128 pairs, 1 sector (below) |
| 9 | Attribution strings | `grep -rniE "claude\|anthropic\|co-authored\|generated with" js/src js/test scripts hyphaeon PHASE3A.md` | no `claude` / `anthropic` / `co-authored`; the `generated with` hits are all "the version the fixtures were generated with" notes |
| 10 | Working tree | `git status --short --untracked-files=all` | `js/`, `fixtures/`, `scripts/gen_fixtures.py`, `PHASE3A.md` only |

### 7. Tree-free parity, the D22 probe

`load_alignment_and_tree(examples/<x>.fasta, None, use_tn93=True, mds_sign="canonical")` in Python,
with the `tn93` binary hidden so the package path runs, against `loadAlignmentAndTree(text, null,
{useTn93: true})` on the same text. Each MDS column is compared **as it is**, with the negated column
alongside to prove a residual is a value and not a sign.

```
=== bat_oas1: N=18 L=351   taxa identical: true   L/N identical: true
  notices.treeFree = {"reason":"requested","taxaOrder":"alignment"}  tree = null  tn93SaturatedPairs = 0
  d  max|delta| = 0.000e+0   (bound 1e-9: PASS, bit-identical)
  z  col 0: as-is 1.118e-8   negated 2.498e-1
  z  col 1: as-is 2.235e-8   negated 2.491e-1
  z  col 2: as-is 1.490e-8   negated 2.233e-1
  z  col 3: as-is 3.725e-8   negated 2.097e-1
  z  worst as-is = 3.725e-8  (bound 1e-5, NO sign allowance: PASS)
  tokens c identical: true   a identical: true   invariable mask identical: true

=== camelid: N=212 L=96   taxa identical: true   L/N identical: true
  notices.treeFree = {"reason":"requested","taxaOrder":"alignment"}  tree = null  tn93SaturatedPairs = 0
  d  max|delta| = 0.000e+0   (bound 1e-9: PASS, bit-identical)
  z  col 0: as-is 7.823e-8   negated 2.418e-1
  z  col 1: as-is 5.960e-8   negated 2.822e-1
  z  col 2: as-is 5.867e-8   negated 2.807e-1
  z  col 3: as-is 6.147e-8   negated 1.929e-1
  z  worst as-is = 7.823e-8  (bound 1e-5, NO sign allowance: PASS)
  tokens c identical: true   a identical: true   invariable mask identical: true
```

**The distance matrices are bit-identical** — the six-significant-digit rounding the package applies
makes exactness reachable, so PLAN.md §5.1's "exact to 1e-9 on distances" is met with room to spare.
MDS lands at 3.7e-8 / 7.8e-8 against the 1e-5 class, and the negated columns miss by 0.19 to 0.28, so
the agreement is the canonical sign convention doing its work.

**What D22 buys, measured:** camelid has no branch lengths, so before this phase every camelid number
in the app came from HyPhy — WASM 2.5.98 in the browser against native 2.5.65 in Python, the one
parity gap PHASE2A could not close. Both sides now compute the same TN93 matrix, bit for bit, and the
gap is gone by construction rather than by tolerance.

### The tree-free e2e fixtures

The full pillar (`hyphaeon meme --use-tn93` on camelid against the library pipeline) needs the ONNX
graph and belongs to the app's `runtime/`. What exists here is the assembly-level record:

| Fixture | argv | taxa | codons | bytes |
|---|---|---|---|---|
| `e2e/meme_camelid_tn93.json` | `meme -a camelid.fasta --use-tn93 --cpu --mds-sign canonical` | 212 | **96** (96 site records) | 18,133 |
| `e2e/meme_HIV1_RT_tn93.json` | `meme -a HIV1_RT.fasta --use-tn93 --cpu --mds-sign canonical` | 475 | **335** (335 site records) | 56,497 |
| `e2e/busted_Smc6_tn93.json` | `busted -a Smc6.fasta --use-tn93 --cpu --mds-sign canonical` | 20 | **1,097** | 2,582 |
| `e2e/epistasis_Smc6_tn93.json` | `epistasis -a Smc6.fasta --use-tn93 --cpu --n-permutations 1000 --seed 42 --mds-sign canonical` | 20 | **1,097** | 24,426 |

`tn93.test.js` replays the taxon and codon counts and the per-site `is_invariable` of all four; the
model-dependent columns are the runtime's to close.

### 8. The phenotype pillar end to end

`run_phenotype_association` was run on `examples/RHO.fasta` with the README Example 3 marine
foreground (`turTru,balMus,balPhys,orcOrc,delDelp,phyCat,phoVit,halGryp,mirLeo,zalCali,odoRos`),
`--n-permutations 0 --seed 42 --cpu --mds-sign canonical`, with `model.forward_cached` wrapped to
capture the two things the JSON never records: `mean_root_attns` [349 × 655] and the raw `y_soft`
LRT surrogate [349]. Those went into `runPhenotypeAssociation` with the trait resolved from the same
foreground string.

One trap worth recording for the next probe: **`forward_cached` recurses into itself** for
memory-safe micro-batching whenever `batch × N²` > 4e6 (`model.py:356-376`; at RHO's N = 655 and
batch 64 that is every call, micro_b = 9). A wrapper on the instance is therefore hit twice per
codon, and the naive capture came back 698 rows long for a 349-codon gene. Only the outermost call
is recorded.

```
loaded: N=655 (py 655)  L=349 (py 349)  tree=embedded  treeFree=null

--- record key order (top level)
  identical and in order: true  (21 keys)

--- gene-level fields (tolerance 1e-6)
  spectral_energy         js 0.0351538373021   py 0.0351538373021   |Δ| 6.94e-18  ok
  norm_spectral_ratio     js 0.0634341918175   py 0.0634341918175   |Δ| 1.39e-17  ok
  max_assoc               js 0.249616287826    py 0.249616287826    |Δ| 0.00e+0   ok
  p_evd_length_adjusted   js 5.84947373129e-8  py 5.84947373129e-8  |Δ| 5.56e-22  ok
  score_track_a           js 7.23288320501     py 7.23288320501     |Δ| 3.55e-15  ok
  score_track_b           js 0.0634341918175   py 0.0634341918175   |Δ| 1.39e-17  ok
  dual_track_composite    js 0.723288320501    py 0.723288320501    |Δ| 3.33e-16  ok
  compact_pars_signature  js "[]"  py "[]"  ok
  permulations_count 0/0 ok   gene_p_value_perm null/null ok
  significant_sites_count 22/22 ok   coselection_pairs_count 128/128 ok   trait_sectors_count 1/1 ok
  phenotype_meta          identical, description included

--- site table
  rows: js 145  py 145            keys per row identical and in order: true
  site order (the score-descending sort): identical
  site / ref_aa / derived_aa mismatches: 0 / 0;  p_assoc_perm null on both sides: all rows
  hyphaeon_lrt         max|Δ| 0.000e+0            fg_mean_attn        max|Δ| 0.000e+0
  p_lrt                max|Δ| 0.000e+0            bg_mean_attn        max|Δ| 0.000e+0
  attribution_norm     max|Δ| 7.451e-9 (site 107) foreground_freq_pct max|Δ| 0.000e+0
  association_rho      max|Δ| 2.383e-8 (site 101) background_freq_pct max|Δ| 0.000e+0
  p_value              max|Δ| 1.078e-8 (site 217) score               max|Δ| 1.495e-8 (site 101)
  p_assoc              max|Δ| 2.394e-8 (site 107) q_value             max|Δ| 4.596e-9 (site 107)
  p_assoc_parametric   max|Δ| 2.394e-8 (site 107)         all 13 numeric fields bound 1e-6: ok

--- co-selection pairs and trait sectors
  pairs 128/128;  keys+ints+strings identical: true;  (site_u,site_v) order identical: true
  worst float |Δ| 4.414e-7 (cesi @ (325,39))  bound 1e-6: ok
  sectors 1/1;  keys identical
    sites [83,101,195,259,292,325]   size 6   spectral_coherence 0.635572969913  |Δ| 0.000e+0
    p_perm 1.0   null_coherence_mean/std/95 |Δ| 0.000e+0   isotropic_baseline 0.166666666667
    shared_taxa 32   shared_branches 32   mean_lrt 3.11738395691  |Δ| 0.000e+0
    pars_signature "[ D83 - G101 - K195 - I259 - A292 - K325 ]"  identical

TOTAL FAILURES: 0
```

Everything the model does not touch is **bit-identical**: `hyphaeon_lrt`, `p_lrt`, both mean-attention
columns, both frequency columns, every string, every count, the sort order, the sector's coherence
and its whole null block. The residuals cluster at 1e-8 to 4e-7 and all come from one place —
`np.linalg.norm(x)` with no axis is BLAS `sdot`, whose blocked float32 accumulation no JavaScript
summation reproduces (the port takes a float64 accumulation rounded once to float32, epistasis.js's
measured choice). That enters as `attribution_norm` at 7.5e-9 and propagates to `association_rho`,
its p-values, `score` and `q_value`. The largest single residual in the run, `cesi` at 4.4e-7, is the
same effect through the pair similarities. All are inside the 1e-6 class and none is near a
threshold: no site, pair, sector or call changed.

**The two shapes of `runPhenotypeAssociation`'s trait input are not equivalent, by design.** Passing
`phenotype: {foreground: ...}` makes the driver resolve the vector itself, as
`run_phenotype_association` does, and `phenotype_meta.description` comes out identical to the CLI's.
Passing a ready `y` instead gives a meta with an **empty description** — a raw vector carries no
provenance and the port does not invent one. The first probe run failed exactly one assertion this
way before it was corrected; a consumer that wants the description in its report should hand over the
options, not the vector.

## Integration changes

- **`src/index.js`** re-exports `preprocess/tn93.js` (a leaf, before `patristic.js`),
  `permulations.js` and `phenotype.js` (in dependency order — `phenotype.js` imports from
  `permulations.js`, `epistasis.js` and `sectors.js`, so it is last of the three). All 28 new names
  were checked against the existing 204 and against each other: **no collisions**, so no binding is
  silently dropped by the ES linker.
- **`test/index.test.js`** pins 232 names, adds identity checks for `tn93DistanceMatrix`,
  `runPhenotypeAssociation` and `generatePermulations`, and extends the pipeline test with a
  **tree-free load through the barrel** — the same three sequences with no tree now return
  `tree: null`, `notices.treeFree.reason === 'no_tree'` and a `d` matrix whose first off-diagonal
  entry is exactly `Math.fround(tn93Distance(...))`, where before Phase 3a the call threw.
- **`src/README.md`** gains rows for the three modules, the `assemble.js` row names both paths, the
  "not ported" paragraph now says HyPhy is not needed rather than not available, and the callback
  section shows the phenotype chain and a tree-free load.

## Public API added (28 names, 204 → 232)

**tn93** — `tn93Distance(seqA, seqB, options)`; `tn93DistanceMatrix(sequences, taxa, options)` →
`Float32Array [n*n]`; `tn93Counts`; `tn93NucleotideFrequency`; `tn93CalculateDistance`;
`tn93SaturatedPairs(dist, n, sentinel)`; `encodeSequence`; `canResolve`; `ambigFractionTooHigh`;
`TN93_MATCH_MODE`; `TN93_MAX_AMBIG_FRACTION`; `TN93_SATURATION_SENTINEL`;
`TN93_MIN_POSITIVE_DISTANCE`; `TN93_FALLBACK_MAX`; `TN93_TABLES`.

**phenotype** — `PRESETS`; `PHENOTYPE_THRESHOLDS`; `resolvePhenotypeVector(taxa, options)`;
`runPhenotypeAssociation(input, predict?, options?)`; `parsePhenotypeTable`; `pyFnmatch`;
`pyReprString`; `pyReprStringList`; `pyFloatStr`.

**permulations** — `computePhylogeneticCovariance(tree, taxa)`;
`generatePermulations(y, tree, taxa, options)`; `findAnyByName`; `pyRegexSource`.

**Changed, same name** — `DIAGNOSTIC_CODES` (contents), `diagnose` (new `useTn93` input),
`loadAlignmentAndTree` (new `useTn93` / `tn93Options`; `tree` and `matchTier` nullable; two new
`notices` fields).

## Fixture coverage

`node scripts/fixture-coverage.mjs` → **48/48**, no uncovered files. Newly covered:

| Fixture | Replayed by |
|---|---|
| `dataset/tn93_distance.json` (24) | `tn93.test.js` — counts, frequencies and distance, max \|Δ\| 0 |
| `dataset/tn93_distance_matrix.json` (8) | `tn93.test.js` — bat_oas1, Smc6, camelid whole; HIV1_RT reduced |
| `dataset/load_alignment_and_tree_tn93.json` (2) | `tn93.test.js` |
| `e2e/{meme_camelid,meme_HIV1_RT,busted_Smc6,epistasis_Smc6}_tn93.json` | `tn93.test.js` — assembly fields |
| `phenotype/resolve_phenotype_vector.json` (15) | `phenotype.test.js` — exact, continuous at 1e-9 |
| `phenotype/compute_phylogenetic_covariance.json` (3) | `permulations.test.js` — max \|ΔV\| 7.1e-15 |
| `phenotype/generate_permulations.json` (3) | `permulations.test.js` — statistical, per PLAN §5.4 |
| `e2e/phenotype_RHO_marine_n_permutations_0.json` | `phenotype.test.js` — 19 assertions |

The three `phenotype/` fixtures were the whole of PHASE2A's uncovered list.

## Python quirks replicated in this phase

Full lists are in the module headers. The ones a consumer will meet:

- **The identical-sequence imputation is the maximum, not the minimum.**
  `compute_tn93_distance_matrix` gives two byte-identical sequences `max(1.0, max_d)` — the largest
  distance in the matrix — while two *different* sequences at distance 0 get 1e-4. Duplicate pruning
  normally runs first, so it takes `prune_duplicates=False` to see it.
- **The `tn93` package raises where `dataset.py` expects a sentinel**, and nothing catches it: a
  saturated pair reaches `math.log` of a non-positive number (`ValueError`) and a pair with no
  overlapping non-gap position divides by zero. `tn93.js` throws with the Python exception name;
  `diagnostics.js` catches it and reports `TN93_SATURATED_PAIRS` at refuse level.
- **`dataset.py:552`'s `d == "-"` guard is dead code** — the `"-"` sentinel belongs to the package's
  `tn93_distance` wrapper, which the reference never calls, along with its 500-nt minimum overlap.
- **A user's foreground glob is silently a regex**: `resolve_phenotype_vector` runs `re.search`
  before `fnmatch` (`phenotype.py:249-256`), so `pan*` is `pa` plus zero-or-more `n` and matches
  `papAnu`. Pinned by the `foreground_glob_Smc6` fixture case.
- **`Bio.Phylo`'s `find_any(name=X)` treats X as a regex**, so a taxon literally named `a.c` can
  match the terminal `aXc` while the root-distance dict lookup on the same name misses.
- **Discrete mode compares `str(val)`**, so a trait column pandas types as float64 renders `"1.0"`,
  matches no target and puts every taxon in the background. One blank cell turns a working int64
  column into that.
- **`background` is declared by two public functions and never read**; so is `max_overlap` in the
  sector miner. Both are accepted and ignored, with tests saying so.
- **Continuous mode z-scores over all N taxa** (unmatched taxa contribute their 0) while deciding
  *whether* to z-score from the standard deviation of the matched values only, and leaves both counts
  at 0.
- **`cesi` is float64 here and float32 in `epistasis.py`**, so the `sim >= 0.25` / `cesi >= 1.0`
  gates in this pillar do *not* have the epistasis pillar's float32-threshold quirk.
- **The trait co-selection block indexes in score order, not site order**, so `coselection_pairs`'
  `(site_u, site_v)` are unordered by position — RHO's first pair is (325, 83).

## Deliberate divergences from the reference (all D22)

1. **No tree → tree-free.** The reference raises (`dataset.py:647-651`); the library takes TN93
   distances and records `notices.treeFree.reason = 'no_tree'`.
2. **No usable branch lengths → tree-free.** The reference calls HyPhy, or falls back to 1e-3/1e-4
   defaults that are not distances at all; the library records `'no_branch_lengths'`.
3. Unparseable tree **text** still raises on both sides — that is a bad input, not a missing one.
4. `cli.py cmd_meme --filter` re-loads the cleaned FASTA with `tree=None` and the reference
   **crashes** there (pinned as `cmd_meme: null` in the filter fixtures). The JS now completes that
   re-load through the tree-free path — the one place the port outruns the Python.

Only case 1's `'requested'` variant is what Python can produce, and that is what the fixtures pin.

## Known gaps for Phase 3b

1. **`nj.js` is not written.** D22 asks for a JS neighbour-joining tree on the same TN93 distances
   for display (site trees); `tn93DistanceMatrix` is the input it needs. `loadAlignmentAndTree`
   currently returns the supplied tree unmodified for display, or null.
2. **Real-model replays stay app-side.** The phenotype e2e fixture records no model outputs, so
   `association_rho`, `attribution_norm`, the two attention means, the two frequency columns,
   `spectral_energy`, `similarity`, `shared_branches` and `spectral_coherence` are pinned by no JSON
   replay. The probe above closes it once, by hand, at the 1e-6 class; the standing check needs the
   ONNX graph and belongs next to the existing runtime replays. The same is true of the four
   `*_tn93.json` e2e fixtures, which are replayed only at the assembly level.
3. **Diagnostics thresholds want a review now that depth is judged on TN93 distances.** camelid
   (median 0.208, 212 taxa) newly trips `DEEP_LARGE_TREE` and HIV1_RT (median 0.046) newly trips
   `SHALLOW_TREE`, where both previously suppressed the depth codes for lack of branch lengths.
   Neither is wrong; both are new advice on inputs that used to get none.
4. **Permulation cost in the browser.** The permulation block allocates two L × P float64 matrices;
   at RHO's L = 349 with P = 10,000 that is about 56 MB and 2.3e9 multiply-adds. A Web Worker and a
   chunked P loop are wanted before the trait UI offers 10,000 permulations.
5. **The neural BUSTED head** (PHASE1A gap 4, PHASE2A gap 2) is unchanged: `model.safetensors` lacks
   11 `BustedMultiTaskHead` parameters and `cmd_busted` loads unseeded, so those fields are null in
   the fixtures and random on every reference run.
6. **Upstream fixes worth making before 3b** (PLAN.md §5.3 rule 3 order — fix Python, regenerate
   fixtures, then the port follows): the identical-sequence imputation to `max(1.0, max_d)`; the
   uncaught `ValueError` / `ZeroDivisionError` from the tn93 package; the `[-0:]` slice at
   `phenotype.py:334` (dead, but it would produce all-ones rows if the binary test ever changed); the
   `leaf_attr @ Y_perms.T` computed twice at `phenotype.py:432` and `436`, which doubles the largest
   allocation in the pillar; and the unread `background` parameter.
7. **Documentation drift.** `PLAN.md` §4.2 and §5.1 still cite `dataset.py:443-521, 544-580`; at
   61d30e3 that code is at 493-571 and 598-636. `PLAN.md` and `PHASE2A.md` still describe the
   pre-D22 diagnostics vocabulary (`TREE_MISSING`, `BRANCH_LENGTHS_MISSING`). The module headers
   carry the current numbers; the plan should catch up.
8. **The reference prefers the tn93 binary.** The two agree exactly today (measured, both versions in
   the manifest), but a future binary release could diverge from the package the port mirrors, and
   nothing would notice. A CI check comparing them on the bundled examples would.
