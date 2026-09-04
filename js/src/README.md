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

| Module | Mirrors |
|---|---|
| `preprocess/newick.js` | Biopython tree handling in `dataset.py:161-212, 302-311` — parse, terminal order, missing branch length as `0.0`, quote-stripped names |
| `preprocess/patristic.js` | `compute_fast_dist_matrix` (`dataset.py:302-350`), `downsample_taxa_faith_pd` (`dataset.py:396-418`) |
| `preprocess/symmetricEigen.js` | the `np.linalg.eigh` call inside `compute_mds_coordinates` (`dataset.py:384`) |
| `preprocess/mds.js` | `compute_mds_coordinates` (`dataset.py:354-394`) |
| `preprocess/tokenizer.js` | `get_codon_token`, `get_aa_token` and their tables (`dataset.py:25-57`) |
| `preprocess/variability.js` | the invariable-site rule (`dataset.py:718-723`) |
| `preprocess/assemble.js` | the tensor-assembly half of `load_alignment_and_tree` (`dataset.py:523-730`) |
| `preprocess/modelContract.js` | the input/output contract of `PhyloAxialTransformer.forward` (`model.py:481-583`) and `model_config.json` |
| `index.js` | the single public entry point; `package.json`'s `exports` map exposes nothing else |

Everything under `preprocess/` was **ported verbatim from DataMonkey 3** (`main@fac1330`,
`src/lib/services/axomeme/`), with function bodies unchanged. Each file's header says so, names the
Python it mirrors, and lists the places where the port and `hyphaeon/*.py` at v1.0.0 **do not agree
yet** — most importantly the codon vocabulary (`tokenizer.js`) and the serine rule
(`variability.js`). Those are open Phase 0 questions for the fixture harness to settle, not defects
to fix by inspection. Read the header before changing anything.

## What is deliberately NOT in here

- **No `onnxruntime`, of any kind.** Not `onnxruntime-web`, not `onnxruntime-node`, not as an
  optional or peer dependency. This package has no runtime dependencies at all; `package.json` has
  no `dependencies` key, and that is a thing to preserve, not an omission.
- **No file, network or model I/O.** Parsing a string that someone else read is in scope. Reading
  the file, fetching the model, verifying its hash and loading the manifest are not.
- **No workers, no server, no MCP, no UI.** Nothing that schedules, listens, renders or holds a
  session.
- **No result semantics.** Tier gates, z-scores, percentile ranks and call modes are product
  decisions that change without the methods changing. `variability.js` took only the two alignment
  functions out of DataMonkey 3's `postprocess.js` for exactly this reason; `buildPredictions` and
  `callModes.js` stayed behind.

All of that lives in `veg/hyphaeon-app`: `runtime/` (ORT sessions, manifest and sha256 verification,
pipeline orchestration, result post-processing), `web/`, `mcp/`, `server/`. The app depends on this
package at a pinned tag; this package depends on nothing.

The practical test, when it is unclear which side something belongs on: **could two different
products want it to behave differently?** If yes it is the app's. If it is the same arithmetic no
matter who is asking, it is the library's.

## Running the tests

```sh
cd js
npm install
npm test          # vitest run
npm run typecheck # tsc --noEmit; allowJs with checkJs off, so .js is parsed and not type-checked
```

The tests came across from DataMonkey 3 with the modules they exercise and carry their own
provenance headers. They check **properties and hand-derived values** — distances you can read off
the newick, eigenvector orthonormality, translations you can check against a codon table — rather
than numbers recorded from this implementation, because an expectation captured from the code under
test only proves the code is deterministic.

Cross-implementation parity against Python is a different harness and is not these tests:
`scripts/gen_fixtures.py` writes `fixtures/`, `js/test` replays them in the same CI run, and that is
what will decide the open questions listed in the module headers.
