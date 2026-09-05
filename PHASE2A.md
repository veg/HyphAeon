# Phase 2a — the epistasis pillar in `js/`, and one canonical MDS sign

Phase 2a of the plan of record (`../hyphaeon-app/PLAN.md`, §5.1–5.4 and §7) did two things. It
ported the whole of `hyphaeon/epistasis.py` — co-selection networks, sector mining, digital DMS —
into `@veg/hyphaeon-js`, on top of a new networkx-faithful graph kernel. And it removed the reason
the port could not match the reference in the first place: **MDS eigenvector signs are now a
convention, not a solver artefact**, on both sides, with `--mds-sign` and `--seed` on the CLI and
the fixtures regenerated underneath. `PHASE1.md` gap 1 in the app repository is closed; this page
is the map of what landed, what each check printed, and what Phase 2b inherits.

## What landed

### The library (`js/`, 204 public names, up from 179)

| Path | What it is |
|---|---|
| `js/src/epistasis.js` (677 lines) | `epistasis.py:51-222`. `consensusDelta` and `computeTransformerAttributions` as pure functions of what the model returned (attention × non-consensus indicator, Self–Liang p, float32 throughout); `runTransformerAttributions` over an async `predict` returning `{lrt, attention}` — the pillar's callback needs the graph's second output, `mean_root_attns`; `computeBranchCoselectionNetwork` (float32 cosine, Student-t p via `tSf`, global BH, CESI) and `CoselectionGraph`, an `nx.Graph` reproducing insertion order, shared edge dicts and `G.edges(data=True)` order. |
| `js/src/sectors.js` (575 lines) | `epistasis.py:224-445`. `computeSectorPermutationTest` (the seeded K-subset null with `Xoshiro256`) and `extractEpistaticSectorsTse` (degree > 0 subgraph → communities → spectral coherence λ₁/Tr → eigenvector pruning at \|v_dom\| ≥ 0.10 → permutation test → the stable (coherence, size) sort). |
| `js/src/dms.js` (441 lines) | `epistasis.py:446-628` and the `run_digital_dms_analysis` record at `724-768`. `runInsilicoSelectionDms` sweeps 19 substitutions per site over an async `predict`; `CANONICAL_AA_TO_CODON` is transcribed verbatim because it is a hand-written table, not the first codon per residue (it differs for 18 of 20 residues). |
| `js/src/numeric/graph.js` (734 lines) | networkx 3.6.1 as a specification, in the kernel where PLAN §5.1 puts it: `connected_components`, `greedy_modularity_communities` (Clauset–Newman–Moore with a line-for-line port of `MappedQueue`/`_HeapElement` and the exact float order of `dq`), `G.subgraph()` view iteration, and `cpythonIntSetOrder` — CPython's set table simulated, because the subgraph view iterates a `set` of node ids and that order decides sector ids. |
| `js/test/` | 26 files, 891 tests (was 22 / 753). New: `epistasis.test.js` (40), `sectors.test.js` (32), `dms.test.js` (36), `numeric-graph.test.js` (25), plus Python-generated references under `js/test/data/{epistasis,sectors,dms}/` with their `gen.py`. |

### The reference (`hyphaeon/`, PLAN §7 items 7 and 9)

- `--mds-sign {canonical,lapack}` on `meme`, `busted`, `epistasis`, `dms`, `phenotype`, `filter`;
  default `HYPHAEON_MDS_SIGN` if set, else `canonical`. `dataset.compute_mds_coordinates` and
  `load_alignment_and_tree` take `mds_sign=None` → env → canonical; the CLI publishes the resolved
  value into the environment so the loads inside `inference.py`, `filter.py`, `epistasis.py` and
  `phenotype.py` see the same convention.
- `--seed <int>` (default 42) on `epistasis` (→ `rng_seed`) and `phenotype` (→ `generate_permulations`
  **and** the trait-sector null). `scripts/parity.py`'s epistasis comparator was waiting on exactly
  this flag and is now live.
- `MDS_SIGN.md` — the rule, where it is set, the measured effect, a plain statement for CLI users,
  and the recommendation. `scripts/mds_sign_effect.py` reproduces the numbers.
- `fixtures/` regenerated under canonical (`manifest.json` carries `mds_sign` and `mds_sign_rule`;
  e2e `argv` records `--mds-sign canonical` and `--seed 42`), `fixtures/README.md` rewritten for the
  convention and the RNG contract.

### The rule, in one sentence

For each kept eigenvector, `np.argmax(np.abs(col))` (first index on ties); flip the column when that
entry is negative; before the `sqrt(eigenvalue)` scaling; on the dense path, the Lanczos path and
the library's tred2/tql2 alike. A flip is exact in floating point, so canonical coordinates are the
solver's coordinates up to sign and nothing else changes.

## MDS_SIGN.md, the headline numbers

`hyphaeon meme --cpu`, `--mds-sign lapack` (the old behaviour) against `--mds-sign canonical` (the
new default), bundled weights, torch 2.10.0, HyPhy 2.5.65 for the two trees without branch lengths.
"median rel." is median \|ΔLRT\| / max(1, \|LRT_lapack\|) over variable sites.

| Example | N | Solver | Columns flipped | max \|ΔLRT\| (site) | median rel. | Spearman ρ | p ≤ 0.05 calls changed |
|---|---|---|---|---|---|---|---|
| HIV1_RT | 475 | LAPACK | 1, 2 | 3.90e-02 (219) | 9.06e-04 | 0.999962 | 0 / 151 (35 → 35) |
| RHO | 655 | Lanczos | 1, 3 | 5.95e-01 (183) | 1.46e-02 | 0.998792 | **2 / 145** (23 → 25) |
| Smc6 | 20 | LAPACK | 1, 2, 3 | 1.85e-02 (628) | 3.52e-04 | 0.999908 | 0 / 97 (1 → 1) |
| bat_oas1 | 18 | LAPACK | 0, 1 | 5.40e-01 (332) | 1.71e-02 | 0.998433 | 0 / 182 (5 → 5) |
| camelid | 212 | LAPACK | 1, 2 | 2.25e-01 (18) | 1.03e-02 | 0.999170 | 0 / 86 (15 → 15) |

`hyphaeon busted` on Smc6: `p_value_acat` 0.11831563 → 0.11797637, `omnibus_lrt` 3.2854052 →
3.2988663, `total_selection_energy` 84.768 → 84.817, `sig_sites_p05` 5 → 5.

**Every variable site on every example moves by more than the 1e-5 graph class.** That is the
finding, not a defect: the model reads the sign (`mds_proj = nn.Linear(4, …)` on raw coordinates),
so the pre-change answer depended on which eigensolver ran. Rankings are essentially unchanged
(ρ ≥ 0.998); two RHO sites cross p = 0.05. Neither convention is "more correct" — what canonical
buys is that the answer is the same on every machine, in every implementation. `--mds-sign lapack`
reproduces old numbers exactly.

## How to run each check, and what it printed

All JS commands from `js/`; Node 22; Python is the scratch venv with
`HYPHAEON_WEIGHTS=$PWD/model.safetensors HF_HUB_OFFLINE=1`.

| # | Check | Command | Result |
|---|---|---|---|
| 1 | Python suite | `python -m pytest tests/ -q` | `252 passed, 2 skipped, 1 warning in 30.58s` — green with the new `--mds-sign` / `--seed` flags |
| 2 | JS suite | `npm ci && npx vitest run` | `Test Files  26 passed (26)` / `Tests  891 passed (891)` |
| 3 | Typecheck | `npm run typecheck` | `tsc --noEmit -p tsconfig.json` silent |
| 4 | Public surface | `npx vitest run test/index.test.js` | 6 passed: 204 names, re-export identities, pipeline through the barrel, `ERR_PACKAGE_PATH_NOT_EXPORTED` for a deep import |
| 5 | Fixture coverage | `node scripts/fixture-coverage.mjs` | `38/41 fixture files read by a test`; uncovered: the three `phenotype/` fixtures only |
| 6 | Purity | `grep -rnE "onnxruntime\|node:\|\bfs\b\|fetch(\|Worker(\|process\.\|require(" src/` | 10 hits, **0 outside comments and `src/README.md`** |
| 7 | MDS parity probe | Python `load_alignment_and_tree` vs `loadAlignmentAndTree`, Smc6 / bat_oas1 / RHO | z equal per column at 1e-5 **with no sign allowance** — worst 6.4e-7 (below) |
| 8 | Epistasis end-to-end probe | the model's own attention through the three ports | edges bit-identical, sectors exact, `p_perm` in class (below) |
| 9 | Attribution strings | `grep -rniE "claude\|anthropic\|co-authored\|generated with" js/src js/test scripts hyphaeon PHASE2A.md` | no `claude` / `anthropic` / `co-authored` anywhere; the three `generated with` hits are "the Biopython version the fixtures were generated with" and two "regenerated with" notes about the MDS convention |

### 7. MDS parity, the probe that closes PHASE1.md gap 1

`load_alignment_and_tree(examples/<x>)` in Python against `loadAlignmentAndTree` on the same text,
default options on both sides. Taxa are identical on all three; each column is compared **as it is**,
with the negated column shown alongside to prove a residual is a value and not a sign.

| Example | N | Python solver | col 0 | col 1 | col 2 | col 3 | worst \|Δ\| |
|---|---|---|---|---|---|---|---|
| Smc6 | 20 | dense `eigh` | 3.73e-9 | 2.79e-9 | 1.40e-8 | 8.67e-9 | **1.40e-8** |
| bat_oas1 | 18 | dense `eigh` | 1.49e-8 | 2.24e-8 | 2.61e-8 | 1.49e-7 | **1.49e-7** |
| RHO (embedded tree) | 655 | **Lanczos `eigsh`** | 6.41e-7 | 2.03e-7 | 1.79e-7 | 6.11e-7 | **6.41e-7** |

Every column passes 1e-5 absolute; the same columns negated miss by 3e-2 to 1.6e0, so the agreement
is the convention doing its job, not a coincidence of magnitudes. RHO is the interesting one: the
reference took the ARPACK path (N > 500) and the library ran dense, and they still agree to 6.4e-7
— **PHASE1A gap 2 shrinks from "an iterative approximation of what JS computes exactly" to a
measured 6.4e-7 on the one bundled example that reaches it.**

### 8. Epistasis end to end, on the model's own attention

`hyphaeon epistasis -a examples/Smc6.fasta -t examples/Smc6.nwk --cpu --seed 42
--n-permutations 1000` was run, and a probe replayed `run_epistatic_analysis`'s internals to capture
the one thing the CLI never writes: `mean_root_attns` [1097 × 20]. The capture was verified against
the reference's own outputs (`attention × delta == leaf_attributions` and `clip(y_soft) == lrts`,
bit for bit) and the probe's edges and sectors are byte-identical to the CLI's `-o` JSON. That
attention then went through the three library functions.

```
computeTransformerAttributions (epistasis.py:51-114), L=1097 N=20:
  leaf_attributions [1097x20] max|Δ| = 0.000e+0   bit-equal: true
  lrts              max|Δ| = 0.000e+0
  pvals             max|Δ| = 0.000e+0
  consensus_aas     identical (all 1097)

computeBranchCoselectionNetwork (epistasis.py:121-222), CLI defaults (min_sim 0.30):
  edge count 5 vs 5   order (site_u,site_v) identical
  every key present and in the reference's order; ints/strings exact
  worst float difference over all 5 edges x 9 float fields: 2.033e-20 (p_val, edge 279-930)  bound 1e-6: PASS
    S279-D930: sim 0.803160906/0.803160906  cesi 2.938992/2.938992  fdr_q 1.131169e-4/1.131169e-4  shared 5/5
    M685-D930: sim 0.886417806/0.886417806  cesi 2.296611/2.296611  fdr_q 1.226573e-6/1.226573e-6  shared 9/9
    H244-S279: sim 0.980537474/0.980537474  cesi 2.211562/2.211562  fdr_q 2.688538e-13/2.688538e-13  shared 5/5
    S279-R557: sim 0.999999940/0.999999940  cesi 2.121076/2.121076  fdr_q 0.000000e+0/0.000000e+0  shared 5/5
    S279-V461: sim 0.999999821/0.999999821  cesi 2.003104/2.003104  fdr_q 0.000000e+0/0.000000e+0  shared 5/5
  graph: nodes 1097/1097, edges 5/5, edge order identical

extractEpistaticSectorsTse (epistasis.py:306-445), B=1000 seed=42:
  sector count 2 vs 2
  sector 1 sites [244,279,461,557] : membership EXACT, strings/counts EXACT
    spectral_coherence 0.991334975 vs 0.991334975  |Δ| 0.000e+0 (1e-6 PASS);  mean_lrt |Δ| 0.000e+0
    p_perm 0.001 vs 0.001  |Δp| 0.000000 <= 3·√(p(1-p)/B) = 0.002998  PASS
    null mean 0.510603/0.504484 (rel 1.20%), std 0.119760/0.113704 (rel 5.06%), p95 0.740589/0.721538 (rel 2.57%)
  sector 2 sites [685,930] : membership EXACT, strings/counts EXACT
    spectral_coherence 0.943870902 vs 0.943870902  |Δ| 0.000e+0 (1e-6 PASS);  mean_lrt |Δ| 0.000e+0
    p_perm 0.093 vs 0.08  |Δp| 0.013000 <= 3·√(p(1-p)/B) = 0.027553  PASS
    null mean 0.694773/0.694118 (rel 0.09%), std 0.139364/0.134670 (rel 3.37%), p95 1.000000/0.999911 (rel 0.01%)
```

The whole deterministic half is **bit-identical**, including the network's key order and the graph's
edge order; `p_perm` passes its bound on both sectors. Two null-moment figures at B = 1,000 sit
above the 2 % class (std 5.06 % and 3.37 %), which is the *estimator's* Monte Carlo error, not a
difference in the null: the standard error of a std estimate at B = 1,000 is ≈ 2.2 % relative, so
5 % is about two of them. Repeating the same two sectors at **B = 100,000 with three seeds a side**
settles it — the nulls are the same to well inside the class, and `p_perm` passes on all six pairs:

| Sector | seed | `p_perm` js / py | \|Δp\| ≤ bound | null mean | null std | null p95 |
|---|---|---|---|---|---|---|
| 1 (K=4) | 42 | 0.00110 / 0.00140 | 3.0e-4 ≤ 3.6e-4 | 0.08 % | 0.41 % | 0.06 % |
| 1 | 7 | 0.00109 / 0.00119 | 1.0e-4 ≤ 3.3e-4 | 0.18 % | 0.00 % | 0.06 % |
| 1 | 20260905 | 0.00144 / 0.00129 | 1.5e-4 ≤ 3.6e-4 | 0.03 % | 0.01 % | 0.06 % |
| 2 (K=2) | 42 | 0.08004 / 0.08166 | 1.6e-3 ≤ 2.6e-3 | 0.02 % | 0.20 % | 0.00 % |
| 2 | 7 | 0.08152 / 0.08001 | 1.5e-3 ≤ 2.6e-3 | 0.02 % | 0.60 % | 0.00 % |
| 2 | 20260905 | 0.08033 / 0.08052 | 1.9e-4 ≤ 2.6e-3 | 0.02 % | 0.09 % | 0.00 % |

**Consequence for consumers: `p_perm` and the null moments at the CLI's own B = 1,000 carry roughly
±0.03 absolute and ±5 % of Monte Carlo error on each side independently.** A UI that prints
`p_perm = 0.093` to three decimals is printing noise; B = 10,000 (the function default) is the
setting the parity class was written for.

## Integration changes (beyond wiring)

- **`src/index.js`** re-exports `epistasis.js`, `sectors.js` and `dms.js` after `omnibus.js`;
  `numeric/graph.js` joins the kernel through `numeric/index.js`, where PLAN §5.1 puts graph
  primitives. All 25 new names were checked against the existing 179 and against each other: no
  collisions. `test/index.test.js` pins 204.
- **`src/sectors.js` — the seam between the two halves of the port.**
  `computeTransformerAttributions` returns `leafAttributions` as a flat `Float32Array [L*N]` and
  `computeBranchCoselectionNetwork` reads that shape, but the sector miner took only rows or
  `{data, rows, cols}`. Chaining the three the way `run_epistatic_analysis` chains them therefore
  threw `attributions: data has 0 entries, expected NaN*undefined` — a flat buffer fell into the
  rows branch and `rows[0].length` was `undefined`. A bare flat buffer plus `options.N` is now the
  third accepted shape for both `attributions` and `aNp` (a `LoadedAlignment`'s `a` is exactly
  that), and an unaccompanied one is refused by name. Three tests in `sectors.test.js` pin the
  equivalence of all three shapes and the two error messages.
- **`test/writers.test.js:220`** — the hard-coded `bustedCsv` line for `busted_Smc6` was the
  pre-convention run's numbers (`p_ACAT 0.11831563373812454`, `Omnibus_LRT 3.285405158996582`).
  Refreshed from the regenerated fixture (`0.1179763653810807`, `3.2988662719726562`) with a comment
  saying what the literal pins — the formatting, not the science — so the next regeneration is not a
  puzzle. `js/test/data/evaluate/python_formatting.json`, which its own `gen.py` derives from the
  e2e fixtures, was regenerated by the same pass.
- **`src/README.md`** gains rows for the three modules and `numeric/graph.js`, the `mds.js` row now
  names the sign convention, and the callback section shows the epistasis chain end to end.

## Public API added

**epistasis** — `consensusDelta(aaTokens, L, N)`;
`computeTransformerAttributions({attention, lrts, aaTokens, L?, N?})` →
`{leafAttributions, lrts, pvals, consensusAas, delta, L, N}`;
`runTransformerAttributions(loaded, predict, {batchSize?, onProgress?})` where `predict` returns
`{lrt, attention}`; `computeBranchCoselectionNetwork(attributions, lrts, branchNames, consensusAas,
{minSim=0.35, minShared=2, maxFdr=0.05, minLrt=1.0, minCesi=2.0, N?})` →
`{sigPairs, graph}`; `class CoselectionGraph` (`nodes`, `adj`, `addNode`, `addEdge`, `edges()`,
`degree`, `numberOfNodes/Edges`, `toJson`, `fromJson`).

**sectors** — `computeSectorPermutationTest(attributions, siteIndices, observedCoherence,
{nPermutations=10000, activeOnly=true, seed=42, N?})`; `extractEpistaticSectorsTse(graph,
attributions, lrts, consensusAas, {minCliqueSize=3, maxOverlap=0.5, minCoherence=0.5, focalTaxon,
aNp, taxa, nPermutations=10000, maxPermP, seed=42, N?})`; `REV_AA_MAP`; `float32Percentile`.

**dms** — `runInsilicoSelectionDms(loaded, predict, {focalTaxon, batchSize=64, siteSubset, taxa,
progress})`; `runDigitalDmsAnalysis`; `digitalDmsRecord`; `resolveFocalTaxon`; `dmsTargetSites`;
`dmsWildTypeAa`; `CANONICAL_AA_TO_CODON`; `DMS_MUTANTS_PER_SITE`.

**numeric/graph** — `adjacencyFromGraph`, `inducedSubgraph`, `connectedComponents`,
`greedyModularityCommunities`, `numberOfEdges`, `degree`, `graphSize`, `cpythonIntSetOrder`.

**Python** — `dataset.compute_mds_coordinates(D, n_components=4, mds_sign=None)`,
`dataset.load_alignment_and_tree(..., mds_sign=None)`, `resolve_mds_sign`,
`canonicalize_eigenvector_signs`, `MDS_SIGN_MODES`, `MDS_SIGN_ENV`,
`phenotype.run_phenotype_association(..., seed=42)`; CLI `--mds-sign`, `--seed`.

## Fixture coverage

`node scripts/fixture-coverage.mjs` → **38/41**. Newly covered in this phase:

| Fixture | Replayed by |
|---|---|
| `epistasis/compute_branch_coselection_network.json` (5 cases) | `epistasis.test.js` — edge set, cesi order, graph node/edge order exact; floats ≤ 1 float32 ulp |
| `epistasis/extract_epistatic_sectors_tse.json` (8) | `sectors.test.js` — membership, ids, strings, focal fields exact; max \|dC\| 0.0 |
| `epistasis/compute_sector_permutation_test.json` (5) | `sectors.test.js` — degenerate cases exact, Monte Carlo cases in class |
| `dms/run_insilico_selection_dms.json` (2) | `dms.test.js` — every delta, baseline, reduction and key order bit-equal via a playback `predict`; `p_value` at 1e-9 (worst measured 1.9e-15) |

Still uncovered, all `phenotype.py`, all Phase 2b: `resolve_phenotype_vector.json` (15 cases),
`compute_phylogenetic_covariance.json` (3), `generate_permulations.json` (3).

`e2e/epistasis_Smc6_n_permutations_1000.json` counts as covered by `writers.test.js`, but only as a
document the writers are checked against — the *methods* behind it are replayed by the probe above,
not by a test. An e2e replay of that fixture needs the ONNX `predict` and belongs to the app
runtime (see gaps).

## Python quirks replicated in this phase

Full lists are in the module headers; the ones a consumer will meet:

- **`max_overlap` is dead.** `extract_epistatic_sectors_tse` declares it and never reads it — there
  is no Jaccard suppression in the reference at `cf838ab`. `extractEpistaticSectorsTse` accepts
  `maxOverlap` and ignores it, and a test says so.
- **`branch_names` is dead too**, in `compute_branch_coselection_network`; `shared_branches`
  duplicates `shared_taxa` and `hyper_p` duplicates `p_val` (names left from a hypergeometric
  formulation). There is no branch projection and no APC correction anywhere in lines 51-222 —
  "branches" means taxa.
- **numpy 2 weak-scalar promotion decides five thresholds.** `sim_arr >= min_sim` and friends
  promote the Python float to the float32 array's dtype, and `float32(0.35) = 0.34999999404`, so a
  pair whose cosine is exactly that passes `min_sim=0.35`. The port compares against
  `Math.fround(threshold)` — same for `min_cesi`, `max_fdr`, `min_lrt`, `v_dom >= 0.10` and
  `coherences >= observed - 1e-7`.
- **The float32 cosine can exceed 1** (identical rows give 1.0000001192), so `1 - sim²` goes
  negative, the `max(1e-9, ·)` guard fires and collinear pairs do *not* all get the same p.
- **`shared_taxa` is computed on the unpruned community rows** while `sites` and `mean_lrt` use the
  pruned set (`sub_A` is not reassigned after pruning).
- **The CLI and the function disagree on `min_sim`**: 0.30 from `cmd_epistasis`, 0.35 as the
  function default; `min_cesi` is never passed, so 2.0 always applies.
- **DMS**: `run_digital_dms_analysis` reports the *caller's* `focal_taxon` string rather than the
  taxon actually swept; `total_mutations` is `19 × codon_count` regardless of what was swept;
  `plasticity` and `selection_dms_plasticity` are the same list object; a `focal_taxon` matching
  nothing silently keeps index 0.
- **Sector ordering depends on CPython's set iteration order**, because networkx's subgraph view
  iterates a `set` of node ids when `2k < |G|`. Reproduced exactly for non-negative integer ids
  < 2³¹ (which is all `epistasis.py` produces).

One latent Python bug is **not** emulated: if `wt_aa` were ever outside the 20 standard residues,
the DMS sweep would write past the site's row block. It is unreachable (the fallback always yields a
residue), so `dms.js` throws a documented guard instead of reproducing an out-of-bounds write.

## Known gaps for Phase 2b

1. **`phenotype.py` and `permulations`** — the three uncovered fixtures. The kernel already has
   `cholesky`, `normSf`, `tSf` and `Xoshiro256`; `sectors.js` is reusable for trait sectors.
   Permulations use numpy's *legacy* MT19937 global state upstream, so that parity is statistical
   from the start.
2. **The neural BUSTED head** (PHASE1A gap 4) is unchanged: `model.safetensors` lacks 11
   `BustedMultiTaskHead` parameters and `cmd_busted` loads unseeded, so those fields are null in the
   fixtures and random on every reference run.
3. **Real-model replays stay app-side.** `dms/run_insilico_selection_dms.json`,
   `attribution/attribute_selection.json` and `e2e/filter_camelid.json` `cleaned_metrics` are
   replayed here with playback `predict`s reconstructed from the recorded outputs; the 1e-5-class
   replay through ONNX belongs to `veg/hyphaeon-app`'s `runtime/`, next to the existing ones. The
   epistasis e2e fixture wants the same treatment (`runTransformerAttributions` → the ONNX `lrt` and
   `mean_root_attns`).
4. **BLAS `sgemm` is not reproducible, by anyone.** The cosine network's `A @ Aᵀ` is a blocked
   float32 accumulation whose order no fixed summation reproduces (measured: sequential and pairwise
   float32 match 25–95 % of entries depending on N). The port uses a float64 dot rounded once to
   float32, within ~5e-7 relative at N = 511. On Smc6 that came out bit-identical, but a pair whose
   cosine sits within ~5e-7 of a threshold could be classified differently. The class is 1e-6, not
   bit-equality, and it should stay that way.
5. **`p_perm` at small B is noisy on both sides** (the measurement above). PLAN §5.4's "null moments
   within 2 %" is only meaningful from about B = 10,000; a parity run at the CLI's B = 1,000 will
   produce excursions that are not defects. `scripts/parity.py` defaults to B = 10,000 — keep it.
6. **Upstream fixes worth making before 2b** (PLAN §5.3 rule 3 order: fix Python, regenerate
   fixtures, then the port follows): implement or delete `max_overlap`; clamp the float32 cosine to
   [-1, 1] and compare thresholds in float64; reconcile the CLI's `min_sim` 0.30 with the function's
   0.35.
7. **Provenance and the examples.** `mds_sign` should be recorded in result provenance (PLAN §3.5),
   and `examples/*_results.csv`, `examples/*.json`, `examples/expected_results/` and `model_eval/`
   outputs predate the canonical default and no longer reproduce byte for byte (PLAN §7 item 5).
   The ML team may also want to evaluate — or fine-tune — on canonical coordinates, since
   `training_data.py` now produces them automatically.
8. **App-side follow-ups now unblocked** (`../hyphaeon-app/PHASE1.md` gap 1): drop the `test.fail()`
   on the strict e2e parity test, the `ctxt.skip` in `mcp/test/engine.test.js` and the "unaligned run
   differs" assertion in `runtime/test/parity-fixtures.test.js`, whose sign-alignment logic is now
   wrong by construction; then rerun `parity-node.mjs` and `parity.py`. `parity.py`'s `TOL_GRAPH` is
   still 1e-6 absolute where PLAN §5.4 says 1e-5·max(1, |lrt|), and its epistasis comparator is now
   live at B = 10,000 (a long run).
9. **`loadAlignmentAndTree` has no `mdsSign` option.** It calls `computeMdsCoordinates` with the
   default, so it is canonical; a caller reproducing an `--mds-sign lapack` run cannot ask for that
   through the loader yet.
