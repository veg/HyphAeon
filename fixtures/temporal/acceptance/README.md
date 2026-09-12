# `hyphaeon temporal` on H1N1 — the acceptance run

These six files are **the CLI's own output**, not a `scripts/gen_fixtures.py` product, which is why
they are not in `fixtures/manifest.json`: nothing here was lifted, stubbed or re-serialised, and the
three CSVs below are byte-for-byte what `run_temporal_surveillance` wrote. They exist so that the
JavaScript port can be held against the reference end to end rather than function by function —
`hyphaeon-app`'s `runtime/test/temporal-port.test.js` reads them, and skips with a warning when this
checkout does not carry them.

## What produced them

```
cd <this checkout>
HYPHAEON_WEIGHTS=$PWD/model.safetensors HF_HUB_OFFLINE=1 python -m hyphaeon.cli temporal \
  -a examples/H1N1_2009_pandemic.fasta \
  -t examples/H1N1_2009_pandemic.nwk \
  -B 100 --time-points 60 --cpu \
  -o fixtures/temporal/acceptance/h1n1_cpu
```

`h1n1_cpu_sites_summary.csv` (4,384 rows, 27 columns), `h1n1_cpu_curves.csv` (14,760 rows: 246
candidates x 60 grid points), `h1n1_cpu_waves.csv` (60 rows) and `h1n1_cpu_summary.json`.

`h1n1_mps_sites_summary.csv` and `h1n1_mps_summary.json` are the same command **without `--cpu`**, on
Apple MPS. They are here for one purpose and should be used for no other: see *the floor on every
claim*, below.

## What the run does, in numbers

100 taxa of which 95 carry a date; 4,384 codons of which 273 are variable and 4,111 invariable; the
span is 2009.2490234375 - 2009.9150390625, so 0.666015625 years; the bandwidth clamp binds and
h = 0.050; the static scan calls **nothing** at q <= 0.10; **246 codons** pass the stage-one energy
floor at `tau_peak = 5e-5`, `tau_auc = 1e-5`; **18** are confirmed sweeps, all 18 rescued and none
concordant (because the static scan called nothing); the four wave modes carry 39.67 / 32.37 / 13.92
/ 9.31 % of the variance, 95.27 % together.

Two figures circulated in the phase-5 surveys are **wrong** and these files are the arbiter:
`t_max` is 2009.9150390625, not 2009.9200439453125 (read the last row of `h1n1_cpu_waves.csv`), so
the span is 0.666, not 0.671.

## The floor on every claim

`hyphaeon temporal` does not reproduce **itself** across devices. Comparing the MPS and CPU runs of
the identical command, field by field over all 4,384 rows:

| column | worst abs | worst rel |
|---|---|---|
| `lrt` | 7.0e-6 | 6.3e-6 |
| `p_static` / `q_static` | 1.8e-7 / 1.2e-7 | 3.7e-6 / 5.7e-7 |
| `peak_intensity` / `auc` / `mean_intensity` | 3.5e-8 / 3.3e-9 / 5.2e-9 | 5.5e-7 / 3.5e-7 / 3.5e-7 |
| `peak_date`, `t_half_start`, `t_half_end`, `fwhm_years` | 0 | 0 |
| `p_perm`, `q_perm` | 0 | 0 |
| `r2_fpca` | 1.5e-7 | 2.6e-7 |
| `Wave_1_loading` | 2.4e-6 | **1.7e-4** |
| `classification`, `cross_classification` | 0 rows differ | |

**No port may be held tighter than that.** `Wave_1_loading` moving by 1.7e-4 relative between two
runs of the same code is the number that sets the wave tolerance an order looser than everything
else; it is a standardised projection with small denominators. The `--cpu` run is the fixture for
exactly this reason, and the MPS pair is kept so that the tolerance table is itself testable rather
than asserted.

## What is and is not comparable

- **Strict class**, element-wise: the time axis, `lrt`, `p_static`, `q_static`, the trajectories and
  velocities, `peak_intensity`, `mean_intensity`, `auc`, `r2_fpca`, and the stage-one candidate set.
- **Exact**, as text: `site`, `ref_aa`, `derived_aa`, `mutation_label`, `domain`, and the four
  *selected* quantities `peak_date`, `t_half_start`, `t_half_end`, `fwhm_years` — each is a grid
  point or an argmax, so a bound on them would hide precisely the off-by-one a bad gradient makes.
- **Statistical class only**: `p_perm`, `q_perm`, `is_confirmed_sweep`, `is_rescued_sweep`, both
  classification columns, and the three sweep counts. The reference draws from
  `np.random.RandomState(42)` (MT19937, temporal.py:651) and the JavaScript library uses xoshiro256**
  per-draw substreams by design (D17). MEASURED: at this B = 100, eleven of the 246 candidates sit
  within ONE draw of the p <= 0.05 cut — five of the eighteen sweeps at exactly 0.0495 and six
  non-sweeps at 0.0594 — so the sweep set is fragile by +/-11 and a test must state that rather than
  assert 18 == 18.
- **Statistical class, less obviously**: `fpca_wave_variance_pct` and the four `Wave_k_loading`
  columns. The singular values themselves are deterministic *given a matrix*, but the matrix here is
  the velocity rows of the CONFIRMED SWEEPS (temporal.py:720), which is thresholded on `p_perm`. A
  run that confirms a different set decomposes a different matrix. To compare the decomposition at
  the strict class, condition on this file's own `is_confirmed_sweep` column.
- **Sign-dependent**: `wave_1..4` and `Wave_k_loading`. `temporal.py` has no sign convention and
  writes its solver's raw right singular vectors; the library applies D28's canonical rule. The flip
  vector against this file is **(-1, +1, -1, -1)**, measured, and is recorded rather than absorbed.
