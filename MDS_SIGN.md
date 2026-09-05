# MDS eigenvector signs: the `canonical` convention, what it changes, and why it is the default

## The problem in one paragraph

`compute_mds_coordinates` (`hyphaeon/dataset.py`) embeds the patristic distance matrix into four
classical-MDS coordinates and feeds them to the model as `mds_coords`. An eigenvector is only
defined up to sign, and every eigensolver picks its own: LAPACK `ssyevd` (dense, N ≤ 500), ARPACK
`eigsh` (Lanczos, N > 500) and the JavaScript library's tred2/tql2 disagree with each other, and
none of them is stable under a change of BLAS, thread count or library version. The model is **not**
sign-invariant — `mds_proj = nn.Linear(4, …)` acts on the raw coordinates and Tree-RoPE consumes
them — so the sign the solver happens to return changes per-site LRTs. Until now neither the Python
reference nor the library canonicalised signs; the AxoMEME 2.0 driver did (largest-magnitude entry
positive — the handoff's trap 9), which is why DataMonkey 3's port and `hyphaeon meme` disagreed.
Measured consequence before this change (`../hyphaeon-app/PHASE1.md`, gap 1): Smc6, where the two
solvers happened to agree on all four columns, reproduced `hyphaeon meme` at max |ΔLRT| 5.7e-6;
bat_oas1 (columns 1, 2 flipped) and RHO (columns 2, 3) missed by 0.12 and 0.26.

## The rule

`mds_sign="canonical"`: for each **kept** eigenvector (the top `n_components`), find the index of
its largest-magnitude entry with `np.argmax(np.abs(col))` — the **first** index on an exact tie —
and multiply the whole column by −1 when that entry is negative. An all-zero column is left alone.
The flip is applied **before** the `sqrt(eigenvalue)` scaling, on the float32 eigenvectors, and on
both the dense and the Lanczos path (`canonicalize_eigenvector_signs`). A sign flip is exact in
floating point, so canonical coordinates are bitwise the solver's coordinates up to sign; nothing
else about the embedding changes. `mds_sign="lapack"` leaves the solver's signs alone — that is the
pre-change behaviour, kept so an old run can be reproduced.

`js/src/preprocess/mds.js` implements the identical rule (`{ mdsSign: 'canonical' }`, the default;
`'lapack'` disables it — there it means "tql2's signs", which are not LAPACK's). With both sides
canonical, `js/test/fixtures.test.js` compares `z` **exactly** per column at the 1e-5 class, with no
sign-flip allowance; a sign disagreement is now a defect.

## Where it is set

| Surface | How | Default |
|---|---|---|
| `hyphaeon meme / busted / epistasis / dms / phenotype / filter` | `--mds-sign {canonical,lapack}` | `HYPHAEON_MDS_SIGN` if set, else `canonical` |
| any process | env `HYPHAEON_MDS_SIGN=canonical\|lapack` | `canonical` |
| `dataset.load_alignment_and_tree(..., mds_sign=None)` | keyword; `None` reads the env var | `canonical` |
| `dataset.compute_mds_coordinates(D, n_components=4, mds_sign=None)` | keyword; `None` reads the env var | `canonical` |
| `@veg/hyphaeon-js` `computeMdsCoordinates(dist, n, k, { mdsSign })` | option | `'canonical'` |

The CLI publishes the resolved value into `HYPHAEON_MDS_SIGN` before the pipeline runs, so the
`load_alignment_and_tree` calls inside `inference.prepare_alignment`, `filter.run_alignment_filter`,
`epistasis.run_epistatic_analysis`, `run_digital_dms_analysis` and `phenotype.run_phenotype_association`
all see the same convention as the direct calls in `cmd_meme --filter` and `cmd_busted`.

Also added in the same change, because the parity harness was waiting on it (PLAN.md §7 item 7,
`PARITY.md`): `hyphaeon epistasis --seed <int>` (default 42 → `rng_seed`, the PCG64 sector
permutation null) and `hyphaeon phenotype --seed <int>` (default 42 → `generate_permulations(seed=)`,
numpy's legacy global state, **and** `rng_seed` of the trait-sector permutation null). Defaults equal
the hard-wired values the code had, so outputs are unchanged unless the flag is passed.

## Measured effect on the reference's own outputs

`scripts/mds_sign_effect.py` ran `hyphaeon meme --cpu` twice per bundled example, `--mds-sign lapack`
(the old default) against `--mds-sign canonical` (the new one), with the bundled weights
(`model.safetensors`, sha256 `0278dff0…`), torch 2.10.0, numpy 2.3.3, Python 3.14.0, macOS arm64,
HyPhy 2.5.65 for the two trees without branch lengths (camelid, HIV1_RT), engine `cf838ab` plus this
change. "Columns flipped" is the set of MDS columns whose LAPACK/ARPACK pivot entry was negative, i.e.
exactly the columns canonicalisation changes. "median rel." is median |ΔLRT| / max(1, |LRT_lapack|)
over variable sites (the PLAN.md §5.4 scaling). "calls changed" counts sites whose `p_value ≤ 0.05`
verdict differs between the two runs.

| Example | N (after pruning) | Solver | Columns flipped by canonical | max \|ΔLRT\| (site) | median rel. \|ΔLRT\| (variable) | Spearman ρ (variable) | p ≤ 0.05 calls changed (lapack → canonical) |
|---|---|---|---|---|---|---|---|
| HIV1_RT | 475 | LAPACK eigh | 1, 2 | 3.90e-02 (219) | 9.06e-04 | 0.999962 | 0 / 151 variable (35 → 35 significant) |
| RHO | 655 | Lanczos | 1, 3 | 5.95e-01 (183) | 1.46e-02 | 0.998792 | 2 / 145 variable (23 → 25 significant) |
| Smc6 | 20 | LAPACK eigh | 1, 2, 3 | 1.85e-02 (628) | 3.52e-04 | 0.999908 | 0 / 97 variable (1 → 1 significant) |
| bat_oas1 | 18 | LAPACK eigh | 0, 1 | 5.40e-01 (332) | 1.71e-02 | 0.998433 | 0 / 182 variable (5 → 5 significant) |
| camelid | 212 | LAPACK eigh | 1, 2 | 2.25e-01 (18) | 1.03e-02 | 0.999170 | 0 / 86 variable (15 → 15 significant) |

Every variable site on every example moves by more than the 1e-5 graph class (that is the point:
the model reads the sign). The largest single shifts: RHO site 183, LRT 11.39 → 10.79; bat_oas1 site
332, 5.34 → 5.88; camelid site 18, 4.18 → 4.41; Smc6 site 628, 3.72 → 3.74; HIV1_RT site 219,
17.25 → 17.29. Rankings are essentially unchanged (ρ ≥ 0.998). The two RHO call changes are sites
crossing p = 0.05 from just above to just below.

`hyphaeon busted` on Smc6 (statistical bridge only; the neural head loads unseeded and is random on
every run, so its fields are not comparable):

| field | lapack | canonical | Δ |
|---|---|---|---|
| `p_value_acat` | 0.11831563373812454 | 0.1179763653810807 | -3.393e-04 |
| `p_value_simes` | 1.0 | 1.0 | +0.000e+00 |
| `omnibus_lrt` | 3.285405158996582 | 3.2988662719726562 | +1.346e-02 |
| `total_selection_energy` | 84.76806640625 | 84.81687927246094 | +4.881e-02 |
| `sig_sites_p05` | 5 | 5 | +0.000e+00 |
| `sig_sites_p10` | 14 | 14 | +0.000e+00 |

Wall time is unchanged (the flip is O(N·k)): 17.4 vs 17.8 s HIV1_RT, 4.4 vs 4.5 s RHO, 1.6–1.8 s for
the small examples, on the CPU.

## What the default change means for existing CLI users, plainly

* **Outputs change.** Anyone who re-runs `hyphaeon meme` (or busted, epistasis, dms, phenotype,
  filter) on an alignment they analysed before this change will see different per-site LRTs — on
  the bundled examples up to 0.59 absolute, typically 1e-3 to 2e-2 relative — and consequently
  different p, q, omnibus sums and ACAT values. Site rankings stay essentially the same (ρ ≥ 0.998);
  a handful of sites near a threshold can change their call (2 of 145 variable RHO sites at
  p ≤ 0.05; none on the other four examples). `examples/*_results.csv`, `examples/*.json`,
  `examples/expected_results/` and `model_eval/` outputs made before this change no longer reproduce
  byte for byte under the default.
* **Neither convention is "more correct".** The model was trained on coordinates whose signs were
  whatever LAPACK returned per training alignment, i.e. effectively random per column; the shift
  measured here is the model's sensitivity to an arbitrary input convention, not a change in method.
  What canonical buys is that the answer no longer depends on which eigensolver, BLAS, or
  implementation (Python or JavaScript) produced the coordinates, and is the same on every machine.
* **The old behaviour is one flag away:** `--mds-sign lapack`, or `HYPHAEON_MDS_SIGN=lapack` for a
  whole session, reproduces pre-change outputs exactly (bit-for-bit on the same machine and BLAS).
* **Downstream in this repository:** `fixtures/` were regenerated under canonical (`manifest.json`
  carries `mds_sign`); `js/test/fixtures.test.js` now requires exact per-column agreement;
  `fixtures/README.md` documents the convention. Not regenerated here: `examples/*_results.csv`,
  `examples/*.json`, `examples/expected_results/`, `model_eval/` (PLAN.md §7 item 5 covers those).

## Recommendation

Adopt **canonical everywhere**: the Python default (done), the JavaScript library default (done),
every runtime surface in `veg/hyphaeon-app` (they call `loadAlignmentAndTree`, which now
canonicalises by default), and the training-data builder (`hyphaeon/training_data.py` goes through
`load_alignment_and_tree`, so the next training run will see canonical coordinates automatically;
a model fine-tuned on canonical inputs would remove the arbitrary-sign sensitivity this document
measures rather than merely pin it). Keep `lapack` as an explicit, documented escape hatch for
reproducing pre-change numbers and nothing else. Record `mds_sign` in result provenance
(`PLAN.md` §3.5) so a results file says which convention produced it.

## Reproduce

From the repository root, with the engine installed (`pip install -e .`), `hyphy` on PATH for
camelid and HIV1_RT (they are skipped with a note otherwise):

```bash
HYPHAEON_WEIGHTS=model.safetensors HF_HUB_OFFLINE=1 python scripts/mds_sign_effect.py --out /tmp/mds_sign
```

The script writes the CLI's `-o` JSON for every run to `--out`, a `report.json` with the numbers
above plus per-column pivots (index, taxon, value) and run times, and prints the Markdown tables.
`--examples Smc6,bat_oas1` restricts the set; `--busted-examples` picks which examples also run
`busted` (default Smc6). About 70 s on an Apple M4 Pro CPU for all five examples.

A single example by hand:

```bash
hyphaeon meme -a examples/bat_oas1.fasta -t examples/bat_oas1.nwk --cpu --mds-sign lapack    -o lapack.json
hyphaeon meme -a examples/bat_oas1.fasta -t examples/bat_oas1.nwk --cpu --mds-sign canonical -o canonical.json
```
