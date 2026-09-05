<!--
WHY THIS FILE EXISTS

The split between this package and `veg/hyphaeon-app` is the plan's load-bearing architectural
decision (PLAN.md §3.2, §5.5) and it is invisible from inside a source file: nothing about
`assemble.js` announces that adding one `fetch` to it would be a design failure. This is where the
rule is written down, next to the code it constrains, so a contributor meets it before opening a
pull request rather than in review.
-->

# `@veg/hyphaeon-js` — the methods, as pure functions

This directory is a **library**. It mirrors `hyphaeon/*.py` in the repository above it, function for
function, and it does nothing else.

The Python is the reference implementation. The JavaScript is a port of it, living in the same
repository so that a change to a method and a change to its port are **one pull request, reviewed
side by side, released under one tag** to PyPI and npm. Drift between the two is then a CI failure in
the same change that caused it, instead of a bug report from a consumer three weeks later.

## What is in here

Every module carries a "WHY THIS FILE EXISTS" header naming the Python file and line range it mirrors
at the commit the port was taken from, what it deliberately does not do, and the measurement behind
any constant. Where a body replaced DataMonkey 3's seed port, a "Divergence from the datamonkey3
port" note lists what changed and why (PLAN.md §5.3 rules 1 and 3). Read the header before changing
anything.

| Module | Mirrors |
|---|---|
| `preprocess/modelContract.js` | the vocabulary sentinels of `dataset.py:25-57` (61 sense codons 0..60 in TCAG order; stop/gap/unknown → 64; residues 0..19, everything else → 20), the input/output contract of `PhyloAxialTransformer.forward` (`model.py`), `MAX_SPECIES_DEFAULT` 256 / `MAX_SPECIES_CAP` 512, `validateInputBundle` |
| `preprocess/parse.js` | `parse_alignment_sequences` (`dataset.py:59-159`: FASTA / NEXUS / PHYLIP with the Python's format detection) plus the CPython `str` helpers (`pyStrip`, `pySplit`, `pySplitLines`, `pyIsAlnum`, …) it and the other ports need to reproduce whitespace semantics |
| `preprocess/tokenizer.js` | `get_codon_token`, `get_aa_token` and their tables (`dataset.py:25-57`) |
| `preprocess/variability.js` | the invariable-site rule (`dataset.py:718-723`: amino-acid tokens < 20 only, no serine two-family case) |
| `preprocess/tree.js` | `Bio.Phylo.NewickIO.Parser` (Biopython 1.85) as a Newick parser, `extract_tree_from_string_or_file` (`dataset.py:161-212`), `has_nonzero_branch_lengths` / `enforce_nonzero_branch_lengths` (`dataset.py:214-222, 289-300`), three-tier taxon matching (`dataset.py:616-642`), and `needsBranchLengths`, the predicate under which the reference calls HyPhy |
| `preprocess/patristic.js` | `compute_fast_dist_matrix` (`dataset.py:302-352`, float32, zero row for an unmatched taxon) and the `> 10` rescale (`dataset.py:678-681`) |
| `preprocess/downsample.js` | `prune_identical_sequences` (`dataset.py:420-441`), `downsample_taxa_faith_pd` (`dataset.py:396-418`), the stride pre-selection (`dataset.py:672-673`) |
| `preprocess/symmetricEigen.js` | the `np.linalg.eigh` call inside `compute_mds_coordinates` (`dataset.py:384`), tred2/tql2 |
| `preprocess/mds.js` | `compute_mds_coordinates`, dense path (`dataset.py:354-394`; H and B in float32, no Lanczos). `{mdsSign: 'canonical'}` (the default) applies the reference's sign convention — each kept eigenvector flipped so its largest-magnitude entry is positive, before the `sqrt(eigenvalue)` scaling; `'lapack'` keeps tql2's own signs. `../../MDS_SIGN.md` has the rule and what it changes |
| `preprocess/assemble.js` | `load_alignment_and_tree` end to end (`dataset.py:523-730`) as `loadAlignmentAndTree(alignmentText, treeText, options)`, plus `siteBatch` / `siteBatches` / `batchSizeFor` for the runtime's tensor bundles |
| `numeric/` | the kernel PLAN.md §5.1 names: `special.js` (lgamma, regularized incomplete gamma and beta, erfc, χ²/t/normal survival, hypergeometric pmf/cdf/sf, all within 1e-12 of scipy 1.16), `prng.js` (xoshiro256\*\* with splitmix64 seeding), `ranks.js` (rankdata, Pearson, Spearman, Mann–Whitney ROC-AUC), `bh.js` and `cauchy.js` (`stats.py:50-84`, float32-aware), `linalg.js` (Cholesky, eigenvalues-only tred2/tqli), `reduce.js` (numpy's pairwise summation order, which float32 reductions need to match bit for bit), `graph.js` (networkx 3.6.1 as a spec: `connected_components` and `greedy_modularity_communities` with Clauset–Newman–Moore's exact float order and `MappedQueue` tie-breaking, `G.subgraph()` view iteration, and `cpythonIntSetOrder`, CPython's set table simulated because the subgraph view iterates it and it decides sector ids) |
| `stats.js` | `stats.py:16-48` (`pvalsFromLrtMeme`, `pvalsFromLrtSelfLiang`), re-exports of BH and CCT under the Python names, and `memeSitePq`, the exact float32 cast order `cmd_meme` writes (`cli.py:99-100`) |
| `filter.js` | `filter.py` (`scanHypergeometricPatches`, the OCI audit, NNN masking, `runAlignmentFilter` with both forward-pass phases) and, behind `{cliVariant: true}`, the second copy of the screen in `cli.py cmd_meme --filter` |
| `attribution.js` | `attribution.py:19-175` (`attributeSelection`: per-taxon counterfactual ΔLRT, driver ranking, epoch and adaptation mode) and the `cmd_meme` site fields |
| `omnibus.js` | the statistical bridge of `cmd_busted` (`cli.py:424-508`: Self–Liang p, ACAT, Simes, omnibus LRT, verdict, head-output decoding) and `runBusted` over callbacks |
| `epistasis.js` | `epistasis.py:51-222`: `consensusDelta` and `computeTransformerAttributions` (attention × non-consensus indicator) as pure functions of what the model returned, `runTransformerAttributions` over an async `{lrt, attention}` callback, `computeBranchCoselectionNetwork` (float32 cosine, Student-t p via `tSf`, global BH, CESI) and `CoselectionGraph`, an `nx.Graph` with networkx's insertion-order semantics |
| `sectors.js` | `epistasis.py:224-445`: `computeSectorPermutationTest` (the seeded K-subset null) and `extractEpistaticSectorsTse` (communities → spectral coherence λ₁/Tr → eigenvector pruning → permutation test), over `numeric/graph.js` |
| `dms.js` | `epistasis.py:446-628, 724-768`: `runInsilicoSelectionDms`, the 19-substitution sweep over an async `predict`, and the `run_digital_dms_analysis` record. `CANONICAL_AA_TO_CODON` is transcribed verbatim — it is a hand-written table, not the first codon per residue |
| `evaluate.js` | `evaluation.py:28-553` over `{name, text}` pairs instead of paths, including `csv.DictReader`, `float()` and `html.unescape` semantics |
| `writers.js` | the JSON / CSV / GraphML writers of `cli.py` and `io.py`, byte-equal to `json.dump(indent=2)`, pandas `to_csv` and networkx's lxml GraphML layout |
| `diagnostics.js` | PLAN.md §4.3 pre-flight `diagnose()` — the 23 warning codes, thresholds with their sources, and a cost estimate; has no Python counterpart yet (§7 item 6) |
| `index.js` | the single public entry point; `package.json`'s `exports` map exposes nothing else; `test/index.test.js` pins the 204 public names |

Not ported, because the library cannot do them and the runtime must: HyPhy branch-length estimation
(`dataset.py:224-287`) and TN93 distances (`dataset.py:443-521`) — `needsBranchLengths(tree)` and
`notices.branchLengthsMissing` tell the runtime when the reference would have called out; the
Lanczos MDS path for N > 500 (`dataset.py:360-381`; the library always runs dense). Not ported yet
(Phase 1b): `epistasis.py`, `phenotype.py`, `dms.py`.

## Pure functions plus a `predict` callback

Nothing in here loads a model. Every function that needs the network takes an async callback and
applies the reference's post-processing to whatever it returns:

```js
// predict(c, a, meta) -> Promise<ArrayLike<number>>   (the graph's raw `lrt`, one per batch element)
//   c, a : Int32Array token tensors in [batch, N, 1] layout (values 0..64 / 0..20)
//   meta : { batch, N, d: Float32Array [N*N], z: Float32Array [N*4], siteIndices, phase, loaded, taxonIndices? }
const lrts = await predictSiteLrts(loaded, predict);            // clamp(min=0), float32, zeros at invariable sites
const result = await runAlignmentFilter({ alignmentText, treeText }, predict, options);
const attributions = await attributeSelection(loaded, predict, { minLrt: 3.84 });
const busted = await runBusted(loaded, { predictSites, predictHead });

// The epistasis pillar needs a SECOND graph output, so its callback returns a pair:
//   predict(c, a, meta) -> Promise<{lrt: ArrayLike, attention: ArrayLike [batch * N]}>   (mean_root_attns)
const attr = await runTransformerAttributions(loaded, predictWithAttention);
const { sigPairs, graph } = computeBranchCoselectionNetwork(
  attr.leafAttributions, attr.lrts, loaded.taxa, attr.consensusAas, { minSim: 0.30, N: loaded.N }
);
const sectors = extractEpistaticSectorsTse(graph, attr.leafAttributions, attr.lrts, attr.consensusAas, {
  aNp: loaded.a, taxa: loaded.taxa, N: loaded.N, nPermutations: 10000, seed: 42
});
const plasticity = await runInsilicoSelectionDms(loaded, predict, { siteSubset: sitesOf(sectors) });
```

The three epistasis functions chain as `run_epistatic_analysis` chains them: the flat
`Float32Array [L*N]` one returns is a shape the next two accept, with `N` naming the axis a flat
buffer cannot. `graph` is a `CoselectionGraph` and goes straight into the sector miner.

The clamp (`torch.clamp(y, min=0)`), the float32 storage and the batching bookkeeping happen in the
library, exactly where `filter.py` / `inference.py` do them, so the callback returns what the graph
returns and nothing else. The app's `runtime/` wraps an onnxruntime session (or anything else) in
that signature; the tests wrap lookup tables and fake models in it.

Precision follows the reference (PLAN.md §5.3 rule 4): tokens are ints, distances and MDS
coordinates are `Float32Array` as `dataset.py` forms them, LRTs are float32, p-values are float64
unless the Python casts (`memeSitePq`), and every float32 reduction uses numpy's pairwise order
(`numeric/reduce.js`) because a float64 or sequential sum misses the fixtures.

## What is deliberately NOT in here

- **No `onnxruntime`, of any kind.** Not `onnxruntime-web`, not `onnxruntime-node`, not as an
  optional or peer dependency. This package has no runtime dependencies at all; `package.json` has
  no `dependencies` key, and that is a thing to preserve, not an omission.
- **No file, network or model I/O.** Parsing a string that someone else read is in scope. Reading
  the file, fetching the model, verifying its hash and loading the manifest are not. (`js/scripts/`
  holds development tools that do use `node:fs`; it is outside the published `files`.)
- **No workers, no server, no MCP, no UI.** Nothing that schedules, listens, renders or holds a
  session.
- **No result semantics.** Tier gates, z-scores, percentile ranks and call modes are product
  decisions that change without the methods changing; `diagnostics.js` reports and never decides.
- **No improvements.** A Python quirk is replicated and listed in the module header and in
  `PHASE1A.md`; it is fixed upstream first, the fixtures regenerated, then the port follows.

All of that lives in `veg/hyphaeon-app`: `runtime/` (ORT sessions, manifest and sha256 verification,
pipeline orchestration, result post-processing), `web/`, `mcp/`, `server/`. The app depends on this
package at a pinned tag; this package depends on nothing.

The practical test, when it is unclear which side something belongs on: **could two different
products want it to behave differently?** If yes it is the app's. If it is the same arithmetic no
matter who is asking, it is the library's.

## Running the tests

```sh
cd js
npm ci
npm test                            # vitest run: fixture replays + Python-generated local references
npm run typecheck                   # tsc --noEmit; allowJs with checkJs off
node scripts/fixture-coverage.mjs   # which fixtures/**/*.json each test file replays; exit 1 if any is not
```

Two kinds of test data. `../fixtures/` is generated by `scripts/gen_fixtures.py` from the Python
and replayed here at the tolerance class each case declares (`../fixtures/README.md`).
`test/data/<module>/` holds references the builders computed with the same Python environment for
cases the fixtures do not cover (scipy grids, Python string formatting, fake-model runs of
`run_alignment_filter`); each directory keeps the `gen.py` that produced it.
