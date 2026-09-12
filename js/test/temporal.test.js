/**
 * WHY THIS FILE EXISTS
 *
 * `src/temporal.js` is a port of `hyphaeon/temporal.py`'s arithmetic, and the only thing worth
 * proving about it is that it lands on THE REFERENCE'S OWN NUMBERS. A suite that checked the port
 * against the port would prove nothing: a wrong root residue, a gradient edge taken at the wrong
 * order or a candidate set off by one site does not look wrong, it looks like a different dataset.
 *
 * Three layers, which fail for different reasons, so a failure says WHERE the fault is:
 *
 *   1. ROOT REPLAY. `fixtures/temporal/root.json` is `infer_root_sequence` itself, run by the
 *      generator on inputs it writes into the file, so the consensus window, the bincount tie-break
 *      and both unknown-residue paths are replays rather than opinions.
 *
 *   2. CHAIN REPLAY, which is the acceptance layer for this file. `fixtures/temporal/chain.json`
 *      holds eleven runs of the REAL `run_temporal_surveillance` — stubbed only at the two seams
 *      this port does not own, the forward pass and the alignment loader — with the four output
 *      FILES recorded verbatim. The test drives the library's own chain over the same inputs, writes
 *      the same four files, and compares them. Ten cases run at `n_permutations = 0`, where
 *      `p_cand = (1 + 0) / (0 + 1) = 1` for every candidate and the whole chain is deterministic, so
 *      those are compared COLUMN BY COLUMN at the classes below — including the two columns that are
 *      byte-identical strings. The eleventh runs a real B = 50 null on identical inputs and is
 *      compared only where the shuffle cannot reach.
 *
 *   3. THE SEAMS AND THE QUIRKS. The regime switches, the bandwidth clip, the energy floors, the
 *      classification ladder and the writers, each asserted directly, plus one test per replicated
 *      upstream bug so that a later well-meaning "fix" fails loudly and is settled upstream first.
 *
 * WHAT IS COMPARED AT WHICH CLASS, and why — all measured at this commit, not assumed:
 *
 *   - `site`, `ref_aa`, `derived_aa`, `mutation_label`, `domain`, the invariable set, the stage-one
 *     candidate set, `peak_date`, `t_half_start`, `t_half_end`, `fwhm_years`: EXACT, as strings.
 *     These are selected, not computed — a grid point, an argmax, a residue index — and a bound on
 *     them would hide precisely the off-by-one a bad gradient produces. MEASURED on the real
 *     acceptance run (`hyphaeon temporal -a examples/H1N1_2009_pandemic.fasta -B 100
 *     --time-points 60 --cpu`, 4384 codons): `lrt`, `q_static`, `r2_fpca`, `peak_date` and
 *     `fwhm_years` are BIT-IDENTICAL to the reference's CSV across all 4384 rows.
 *   - `lrt`, `q_static`, `p_perm`, `q_perm`, `r2_fpca`, `peak_intensity`, `mean_intensity`, `auc`,
 *     the curves and the waves: 1e-9 absolute, the fixtures' own class. MEASURED on the acceptance
 *     run: `peak_intensity`, `mean_intensity` and `auc` differ from the reference by 2.5e-13
 *     RELATIVE — which is BLAS's summation order against ours in the one matmul, not a port
 *     difference — and `r2_fpca` agrees to the last bit over all 246 candidates.
 *   - `p_static`: 1e-12. It is `0.5·chi2.sf(lrt, 1)` and the library's own special-function class is
 *     what bounds it; measured worst on the acceptance run, 2.4e-15 absolute / 3.4e-14 relative.
 *   - the four `Wave_k_loading` columns: 1e-6, and ONLY for modes that are neither rank-deficient
 *     nor part of a near-degenerate pair. These are float32 stores of a float64 dot product, so
 *     1e-7 is their own quantisation; measured on the acceptance run against the reference's own
 *     18-sweep decomposition, worst |Δ|loading|| = 2.3e-7 over all 4384 sites.
 *
 * WHAT CANNOT BE COMPARED ELEMENT BY ELEMENT, stated here rather than hidden in a widened bound:
 *
 *   - THE NULL. `temporal.py:651` is `np.random.RandomState(42)`, MT19937, and the library uses
 *     xoshiro256** per-draw substreams by design (D17). So `p_perm`, `q_perm`, `is_confirmed_sweep`,
 *     `classification`, `cross_classification` and the three sweep counts are STATISTICAL class.
 *     MEASURED on the acceptance run at B = 100 with C = 246: the reference confirms 18 sweeps and
 *     this library confirms 17, 25, 30, 32 and 33 at five different seeds. That spread is not a
 *     defect — 11 of the 246 candidates sit within ONE draw of the p ≤ 0.05 cut — and the test
 *     asserts the spread rather than a count.
 *   - THE WAVE SIGNS. `temporal.py` has no convention and writes the solver's raw signs; the library
 *     applies D28's canonical rule by default. MEASURED against the reference's own `_waves.csv` on
 *     its own 18 confirmed sweeps: the flip vector is (−1, +1, −1, −1) and, once applied, the four
 *     wave curves agree to 1.3e-15 and the singular values to σ = 20.6989 / 18.6961 / 12.2629 /
 *     10.0253 against the reference's recovered 20.699 / 18.696 / 12.263 / 10.025. The flip vector
 *     is RECORDED in `test/numeric-svd.test.js` rather than absorbed, for the reason `MDS_SIGN.md`
 *     gives. Until `--wave-sign` lands upstream the two sides differ by that vector by construction,
 *     so this file compares wave curves and loadings by MAGNITUDE and leaves the signed comparison
 *     to the convention test next door.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
	resolveTemporalRegime,
	resolveTemporalBandwidth,
	temporalTimeGrid,
	rootConsensusWindow,
	inferRootSequence,
	stableArgsortAscending,
	stableArgsortDescending,
	directionalAttribution,
	nadarayaWatsonWeights,
	taxonMajorWeights,
	smoothTrajectories,
	sweepMetric,
	temporalTrajectoryStatistics,
	resolveEnergyFloors,
	stageOneMask,
	temporalPermStat,
	temporalNullDraws,
	temporalPermPValues,
	temporalDrawPermutation,
	fpcaShapeGate,
	confirmSweeps,
	classifyTemporalSites,
	temporalWaveDecomposition,
	temporalMutationLabels,
	temporalSitesCsv,
	temporalCurvesCsv,
	temporalWavesCsv,
	temporalSummaryJson,
	pyFloat32Repr,
	pyRound,
	pvalsFromLrtSelfLiang,
	benjaminiHochberg,
	TEMPORAL_CLASSES,
	TEMPORAL_CROSS_CLASSES,
	TEMPORAL_SITES_COLUMNS,
	TEMPORAL_SUMMARY_KEYS,
	WAVE_GAP_THRESHOLD,
	WAVE_RANK_EPS
} from '../src/index.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, '..', '..', 'fixtures');
const load = (fn) => JSON.parse(readFileSync(join(FIXTURES, 'temporal', `${fn}.json`), 'utf8'));
const ROOT_CASES = load('root');
const CHAIN_CASES = load('chain');
const MANIFEST = JSON.parse(readFileSync(join(FIXTURES, 'manifest.json'), 'utf8'));

// ---------------------------------------------------------------------------------------------
// The chain, driven the way an application drives it (runtime/, mcp/ and server/ all do this).
// ---------------------------------------------------------------------------------------------

/**
 * Everything from the model's outputs to the four files, so the replay exercises the composition
 * and not only the pieces. It is deliberately NOT in src/: the orchestration, the option surface and
 * the progress reporting belong to the application (PLAN.md §5.5), and this is the smallest possible
 * stand-in for them.
 */
function runChain(c) {
	const o = c.inputs.options ?? {};
	const taxa = c.inputs.taxa;
	const nTaxa = taxa.length;
	const L = c.inputs.a.length;
	const aAll = Int16Array.from(c.inputs.a.flat());
	const inv = Uint8Array.from(c.inputs.inv, (v) => (v ? 1 : 0));
	const lrts = Float32Array.from(c.inputs.lrts);
	const attnsAll = Float32Array.from(c.inputs.mean_attns.flat());

	const regime = resolveTemporalRegime({
		timeUnits: o.time_units ?? 'years',
		sweepMode: o.sweep_mode ?? 'auto',
		keepDuplicates: o.keep_duplicates ?? false
	});

	// The date filter (temporal.py:468-472): float32, NaN where unmatched.
	const allDates = Float32Array.from(taxa, (t) => (t in c.inputs.dates ? c.inputs.dates[t] : NaN));
	const validIdx = [];
	for (let i = 0; i < nTaxa; i++) if (!Number.isNaN(allDates[i])) validIdx.push(i);
	const N = validIdx.length;
	const taxaDates = Float32Array.from(validIdx, (i) => allDates[i]);
	const aValid = new Int16Array(L * N);
	const meanAttns = new Float32Array(L * N);
	for (let s = 0; s < L; s++) {
		for (let j = 0; j < N; j++) {
			aValid[s * N + j] = aAll[s * nTaxa + validIdx[j]];
			meanAttns[s * N + j] = attnsAll[s * nTaxa + validIdx[j]];
		}
	}

	let tMin = Infinity;
	let tMax = -Infinity;
	for (const v of taxaDates) {
		if (v < tMin) tMin = v;
		if (v > tMax) tMax = v;
	}
	const timespan = tMax - tMin;
	const T = o.num_time_points ?? 250;
	const bandwidth = resolveTemporalBandwidth(timespan, {
		timeUnits: regime.timeUnits,
		bandwidth: o.bandwidth ?? null,
		numTimePoints: T
	});
	const { denseT, normDenseT, gradT } = temporalTimeGrid(tMin, tMax, T, regime.nonCalendar);

	const root = inferRootSequence({
		aValid, L, N, taxa, validTaxaIndices: validIdx, taxaDates, rootTaxon: o.root_taxon ?? null
	});

	// The static baseline (temporal.py:525-532): BH over the VARIABLE subset only.
	const pStatic = pvalsFromLrtSelfLiang(lrts);
	const qStatic = new Float32Array(L).fill(1);
	const varIdx = [];
	for (let s = 0; s < L; s++) if (!inv[s]) varIdx.push(s);
	if (varIdx.length > 0) {
		const q = benjaminiHochberg(Float64Array.from(varIdx, (s) => pStatic[s]));
		varIdx.forEach((s, i) => {
			qStatic[s] = q[i];
		});
	}
	let nSigStatic = 0;
	for (let s = 0; s < L; s++) if (qStatic[s] <= 0.1) nSigStatic++;

	const attr = directionalAttribution(meanAttns, aValid, root.rootIndices, L, N);
	const W = nadarayaWatsonWeights(denseT, taxaDates, bandwidth);
	const WT = taxonMajorWeights(W, T, N);
	const curves = smoothTrajectories(attr, L, N, WT, T);
	const metric = sweepMetric(curves, L, T, { sweepMode: regime.sweepMode, gradT });
	const st = temporalTrajectoryStatistics({
		curves, metric, L, T, denseT, normDenseT, gradT, sweepMode: regime.sweepMode, meanAttns, N
	});
	const floors = resolveEnergyFloors({
		timespan, nonCalendar: regime.nonCalendar,
		tauPeak: o.tau_peak === undefined ? 1e-4 : o.tau_peak,
		tauAuc: o.tau_auc === undefined ? null : o.tau_auc
	});
	const { mask, candIndices } = stageOneMask(inv, st.peakIntensities, st.aucs, floors.tauPeak, floors.tauAuc);
	const C = candIndices.length;

	const pPerm = new Float32Array(L).fill(1);
	const qPerm = new Float32Array(L).fill(1);
	const r2 = new Float32Array(L);
	const B = o.n_permutations === undefined ? 1000 : o.n_permutations;
	if (C > 0) {
		const candAttrs = new Float32Array(C * N);
		const candCurves = new Float64Array(C * T);
		const candMetric = new Float64Array(C * T);
		for (let i = 0; i < C; i++) {
			candAttrs.set(attr.subarray(candIndices[i] * N, candIndices[i] * N + N), i * N);
			candCurves.set(curves.subarray(candIndices[i] * T, candIndices[i] * T + T), i * T);
			candMetric.set(st.velocity.subarray(candIndices[i] * T, candIndices[i] * T + T), i * T);
		}
		const vObs = temporalPermStat(candCurves, C, T, { sweepMode: regime.sweepMode, gradT, normDenseT });
		const exceed = temporalNullDraws({
			candAttrs, C, N, T, WT, vObs, sweepMode: regime.sweepMode, gradT, normDenseT, seed: 42, toDraw: B
		});
		const { p, q } = temporalPermPValues(exceed, B);
		candIndices.forEach((s, i) => {
			pPerm[s] = p[i];
			qPerm[s] = q[i];
		});
		if (C >= 2) {
			const gate = fpcaShapeGate(candMetric, C, T);
			candIndices.forEach((s, i) => {
				r2[s] = gate.r2[i];
			});
		} else {
			candIndices.forEach((s) => {
				r2[s] = 1;
			});
		}
	}

	const conf = confirmSweeps({
		stage1Mask: mask, pPerm, r2Fpca: r2, lrts, L, nStage1: C, nonCalendar: regime.nonCalendar,
		permAlpha: o.perm_alpha ?? 0.05, minR2Fpca: o.min_r2_fpca ?? 0.35
	});
	const cls = classifyTemporalSites({ inv, stage1Mask: mask, isConfirmedSweep: conf.isConfirmedSweep, qStatic, L });
	const wd = temporalWaveDecomposition({
		velocity: st.velocity, L, T, isConfirmedSweep: conf.isConfirmedSweep,
		peakIntensities: st.peakIntensities, candIndices
	});
	const labels = temporalMutationLabels({ aValid, L, N, rootIndices: root.rootIndices, rootAas: root.rootAas });

	const exportSites = C > 0 ? candIndices : Int32Array.from({ length: Math.min(L, 20) }, (_, i) => i);
	return {
		L, N, T, C, regime, bandwidth, timespan, tMin, tMax, nSigStatic, floors, root, labels,
		curves, velocity: st.velocity, stats: st, mask, candIndices, pPerm, qPerm, r2, conf, cls, wd,
		nTaxa, nVar: varIdx.length,
		files: {
			sites_csv: temporalSitesCsv({
				L, refAas: root.rootAas, derivedAas: labels.derivedAas, mutationLabels: labels.mutationLabels,
				domains: labels.domains, crossClassification: cls.crossClassification, classification: cls.classification,
				isConfirmedSweep: conf.isConfirmedSweep, isConcordant: cls.isConcordant, isRescued: cls.isRescued,
				lrts, pStatic, qStatic, pPerm, qPerm, r2Fpca: r2, peakTimes: st.peakTimes,
				peakIntensities: st.peakIntensities, tHalfStart: st.tHalfStart, tHalfEnd: st.tHalfEnd,
				fwhm: st.fwhm, meanIntensity: st.meanIntensity, aucs: st.aucs, loadings: wd.loadings, K: 4
			}),
			curves_csv: temporalCurvesCsv({
				exportSites, mutationLabels: labels.mutationLabels, denseT, velocity: st.velocity, curves, T
			}),
			waves_csv: temporalWavesCsv({ denseT, waves: wd.waves, nWaves: wd.nWaves, T }),
			summary_json: temporalSummaryJson({
				alignment: 'synthetic.fasta', tree: null, taxaTotal: nTaxa, taxaTimestamped: N,
				codonsTotal: L, codonsVariable: varIdx.length, codonsInvariable: L - varIdx.length,
				timespan, tMin, tMax, bandwidth, sigStaticQ10: nSigStatic, stage1Candidates: C,
				confirmedSweeps: conf.nSweeps, concordantSweeps: cls.counts.concordant,
				rescuedSweeps: cls.counts.rescued, filteredStaticNoise: cls.counts.filteredStaticNoise,
				varExplained: wd.varExplained, runtimeSec: null
			})
		}
	};
}

/** A CSV as {header: [...], rows: [[cell, ...], ...]}. */
function parseCsv(text) {
	const lines = text.trim().split('\n');
	return { header: lines[0].split(','), rows: lines.slice(1).map((l) => l.split(',')) };
}

const num = (s) => (s === '' ? NaN : Number(s));

/** Compare one CSV column; `EXACT` compares the text, everything else the value. */
function compareColumn(ref, got, name, bound) {
	const i = ref.header.indexOf(name);
	expect(i, `${name} is a column`).toBeGreaterThanOrEqual(0);
	let worstDiff = 0;
	for (let r = 0; r < ref.rows.length; r++) {
		if (bound === 'EXACT') {
			expect(got.rows[r][i], `${name} row ${r + 1}`).toBe(ref.rows[r][i]);
			continue;
		}
		const a = num(ref.rows[r][i]);
		const b = num(got.rows[r][i]);
		if (Number.isNaN(a) && Number.isNaN(b)) continue;
		worstDiff = Math.max(worstDiff, Math.abs(a - b));
	}
	if (bound !== 'EXACT') expect(worstDiff, `${name} worst |Δ|`).toBeLessThan(bound);
	return worstDiff;
}

/** Columns the date shuffle can reach, and therefore the ones a B > 0 case may not compare. */
const RNG_COLUMNS = ['p_perm', 'q_perm', 'is_confirmed_sweep', 'is_concordant_sweep', 'is_rescued_sweep',
	'classification', 'cross_classification'];
/** Columns whose value is SELECTED (a grid point, an argmax, a residue) and so compared as text. */
const EXACT_COLUMNS = ['site', 'ref_aa', 'derived_aa', 'mutation_label', 'domain',
	'peak_date', 't_half_start', 't_half_end', 'fwhm_years'];
/** Columns compared as numbers, with the class each is held at and why (see the file header). */
const NUMERIC_COLUMNS = {
	lrt: 1e-9, p_static: 1e-12, q_static: 1e-9, r2_fpca: 1e-9,
	peak_intensity: 1e-9, mean_intensity: 1e-9, auc: 1e-9
};

describe('the chain, against the reference running its own code', () => {
	it('has every case the generator wrote', () => {
		expect(CHAIN_CASES.length).toBe(11);
		expect(MANIFEST.counts.temporal.chain).toBe(CHAIN_CASES.length);
	});

	const deterministic = CHAIN_CASES.filter((c) => c.tolerance !== 'statistical');

	for (const c of deterministic) {
		describe(c.name, () => {
			const got = runChain(c);
			const ref = parseCsv(c.outputs.sites_csv);
			const mine = parseCsv(got.files.sites_csv);

			it('writes the reference\'s 27 columns, in its order, with one row per codon', () => {
				expect(mine.header).toEqual(TEMPORAL_SITES_COLUMNS.slice());
				expect(mine.header).toEqual(ref.header);
				expect(mine.rows.length).toBe(ref.rows.length);
			});

			it('the selected columns are EXACT, as text', () => {
				for (const col of EXACT_COLUMNS) compareColumn(ref, mine, col, 'EXACT');
			});

			it('the computed columns are within their class', () => {
				for (const [col, bound] of Object.entries(NUMERIC_COLUMNS)) compareColumn(ref, mine, col, bound);
			});

			it('the classification, the sweep flags and the null are reproduced exactly at B = 0', () => {
				// At n_permutations = 0 the estimator is (1 + 0)/(0 + 1) = 1 for every candidate, so
				// there is no Monte Carlo left and every label is a deterministic function of the
				// inputs -- which is what makes this the case worth comparing string for string.
				for (const col of RNG_COLUMNS) compareColumn(ref, mine, col, col.startsWith('p_') || col.startsWith('q_') ? 1e-9 : 'EXACT');
			});

			it('the wave loadings agree in magnitude, for modes that are neither degenerate nor empty', () => {
				// Signed comparison waits for `--wave-sign` upstream (see the file header); the
				// magnitude is invariant to the sign and is compared now.
				const gaps = got.wd.gaps;
				for (let k = 0; k < 4; k++) {
					const comparable = k < got.wd.nWaves && !got.wd.rankDeficient[k] &&
						(k === 0 || gaps[k - 1] >= WAVE_GAP_THRESHOLD) && gaps[k] >= WAVE_GAP_THRESHOLD;
					if (!comparable) continue;
					const i = ref.header.indexOf(`Wave_${k + 1}_loading`);
					let w = 0;
					for (let r = 0; r < ref.rows.length; r++) {
						w = Math.max(w, Math.abs(Math.abs(num(ref.rows[r][i])) - Math.abs(num(mine.rows[r][i]))));
					}
					expect(w, `Wave_${k + 1}_loading magnitude`).toBeLessThan(1e-6);
				}
			});

			it('_curves.csv has the reference\'s rows, its columns and its duplicated column', () => {
				const rc = parseCsv(c.outputs.curves_csv);
				const mc = parseCsv(got.files.curves_csv);
				expect(mc.header).toEqual(rc.header);
				expect(mc.rows.length).toBe(rc.rows.length);
				const iSite = rc.header.indexOf('site');
				const iTime = rc.header.indexOf('time');
				const iSel = rc.header.indexOf('selection_intensity');
				const iVel = rc.header.indexOf('sweep_velocity');
				const iPrev = rc.header.indexOf('prevalence');
				let wv = 0;
				let wp = 0;
				for (let r = 0; r < rc.rows.length; r++) {
					expect(mc.rows[r][iSite]).toBe(rc.rows[r][iSite]);
					expect(mc.rows[r][iTime]).toBe(rc.rows[r][iTime]); // the grid, exactly
					// UPSTREAM BUG (temporal.py:807-808) replicated: the two columns are ONE array.
					expect(mc.rows[r][iSel]).toBe(mc.rows[r][iVel]);
					expect(rc.rows[r][iSel]).toBe(rc.rows[r][iVel]);
					wv = Math.max(wv, Math.abs(num(rc.rows[r][iVel]) - num(mc.rows[r][iVel])));
					wp = Math.max(wp, Math.abs(num(rc.rows[r][iPrev]) - num(mc.rows[r][iPrev])));
				}
				expect(wv).toBeLessThan(1e-9);
				expect(wp).toBeLessThan(1e-9);
			});

			it('_waves.csv has the reference\'s time axis exactly and its modes up to sign', () => {
				const rw = parseCsv(c.outputs.waves_csv);
				const mw = parseCsv(got.files.waves_csv);
				expect(mw.header).toEqual(['time', 'wave_1', 'wave_2', 'wave_3', 'wave_4']);
				expect(mw.rows.length).toBe(rw.rows.length);
				for (let r = 0; r < rw.rows.length; r++) expect(mw.rows[r][0]).toBe(rw.rows[r][0]);
				for (let k = 0; k < 4; k++) {
					const comparable = k < got.wd.nWaves && !got.wd.rankDeficient[k] &&
						got.wd.gaps[k] >= WAVE_GAP_THRESHOLD && (k === 0 || got.wd.gaps[k - 1] >= WAVE_GAP_THRESHOLD);
					if (!comparable) continue;
					let same = 0;
					let flipped = 0;
					for (let r = 0; r < rw.rows.length; r++) {
						const a = num(rw.rows[r][k + 1]);
						const b = num(mw.rows[r][k + 1]);
						same = Math.max(same, Math.abs(a - b));
						flipped = Math.max(flipped, Math.abs(a + b));
					}
					expect(Math.min(same, flipped), `wave_${k + 1} up to sign`).toBeLessThan(1e-9);
				}
			});

			it('_summary.json is byte-equal but for the wave shares, which round the same way', () => {
				const refMeta = JSON.parse(c.outputs.summary_json);
				const mineMeta = JSON.parse(got.files.summary_json);
				expect(Object.keys(mineMeta)).toEqual(TEMPORAL_SUMMARY_KEYS.slice());
				expect(Object.keys(mineMeta)).toEqual(Object.keys(refMeta));
				for (const key of ['taxa_total', 'taxa_timestamped', 'codons_total', 'codons_variable',
					'codons_invariable', 'timespan_years', 't_min', 't_max', 'bandwidth_years',
					'sig_static_q10', 'stage1_candidates', 'confirmed_sweeps', 'concordant_sweeps',
					'rescued_sweeps', 'filtered_static_noise']) {
					expect(mineMeta[key], key).toBe(refMeta[key]);
				}
				for (let k = 0; k < 4; k++) {
					// The shares are sign- and rotation-invariant, and rounded to 2 dp by the reference.
					expect(Math.abs(mineMeta.fpca_wave_variance_pct[k] - refMeta.fpca_wave_variance_pct[k])).toBeLessThanOrEqual(0.01);
				}
			});
		});
	}

	describe('episodic_calendar_B50 (a real null: statistical class only)', () => {
		const c = CHAIN_CASES.find((x) => x.name === 'episodic_calendar_B50');
		const base = CHAIN_CASES.find((x) => x.name === 'episodic_calendar_B0');
		const got = runChain(c);
		const ref = parseCsv(c.outputs.sites_csv);
		const mine = parseCsv(got.files.sites_csv);

		it('is marked statistical in the fixture, and says why in its notes', () => {
			expect(c.tolerance).toBe('statistical');
			expect(c.notes).toContain('MT19937');
		});

		it('everything UPSTREAM of the shuffle is identical to the B = 0 run', () => {
			// The reference creates its RandomState inside the candidate branch and the SVD runs
			// afterwards without touching the stream, so B must not reach the deterministic half.
			// Running the same inputs at two values of B and asserting the columns agree is the
			// cheapest possible detector of an RNG leak into the wave stage.
			const refB0 = parseCsv(base.outputs.sites_csv);
			for (const col of [...EXACT_COLUMNS, ...Object.keys(NUMERIC_COLUMNS)]) {
				if (col === 'r2_fpca') continue; // identical too, but compared numerically below
				const i = ref.header.indexOf(col);
				for (let r = 0; r < ref.rows.length; r++) expect(ref.rows[r][i]).toBe(refB0.rows[r][i]);
			}
			for (const col of EXACT_COLUMNS) compareColumn(ref, mine, col, 'EXACT');
			for (const [col, bound] of Object.entries(NUMERIC_COLUMNS)) {
				if (RNG_COLUMNS.includes(col)) continue;
				compareColumn(ref, mine, col, bound);
			}
		});

		it('the candidate set is identical; only the labels below it may differ', () => {
			const iClass = ref.header.indexOf('classification');
			let refCand = 0;
			let mineCand = 0;
			for (let r = 0; r < ref.rows.length; r++) {
				if (ref.rows[r][iClass] !== 'INVARIABLE' && ref.rows[r][iClass] !== 'FLAT_NO_SIGNAL') refCand++;
				if (mine.rows[r][iClass] !== 'INVARIABLE' && mine.rows[r][iClass] !== 'FLAT_NO_SIGNAL') mineCand++;
			}
			expect(mineCand).toBe(refCand);
		});

		it('every site the two sides disagree about has a p within the Monte-Carlo band of the cut', () => {
			// The class test, concretely: two estimates of the same p agree within 3·sqrt(p(1−p)/B).
			// A disagreement is acceptable only for a site whose reference p lies inside that band of
			// α = 0.05; anything further out would be a real difference, not sampling noise.
			const B = 50;
			const band = 3 * Math.sqrt((0.05 * 0.95) / B);
			const iP = ref.header.indexOf('p_perm');
			const iSweep = ref.header.indexOf('is_confirmed_sweep');
			for (let r = 0; r < ref.rows.length; r++) {
				if (ref.rows[r][iSweep] === mine.rows[r][iSweep]) continue;
				expect(Math.abs(num(ref.rows[r][iP]) - 0.05), `site ${r + 1} is inside the band`).toBeLessThanOrEqual(band);
			}
		});
	});
});

// ---------------------------------------------------------------------------------------------
// The pieces, and the upstream quirks each one replicates.
// ---------------------------------------------------------------------------------------------

describe('the regime switches (temporal.py:430-441)', () => {
	it('auto resolves on the units, and the two switches stay independent', () => {
		expect(resolveTemporalRegime({}).sweepMode).toBe('episodic');
		expect(resolveTemporalRegime({ timeUnits: 'generations' }).sweepMode).toBe('fixation');
		// A calendar run may be FORCED to fixation and a generations run to episodic. Every branch
		// downstream tests one or the other; conflating them is the mistake this pins.
		expect(resolveTemporalRegime({ sweepMode: 'fixation' }).nonCalendar).toBe(false);
		expect(resolveTemporalRegime({ timeUnits: 'days', sweepMode: 'episodic' }).nonCalendar).toBe(true);
	});

	it('prunes duplicates on the calendar path and nowhere else', () => {
		// The hazard the application must surface: identical haplotypes sampled on DIFFERENT DAYS
		// collapse to one taxon, and the other days are silently deleted.
		expect(resolveTemporalRegime({}).pruneDuplicates).toBe(true);
		expect(resolveTemporalRegime({ keepDuplicates: true }).pruneDuplicates).toBe(false);
		expect(resolveTemporalRegime({ timeUnits: 'generations' }).pruneDuplicates).toBe(false);
	});

	it('an unrecognised unit is calendar, because the reference tests membership of a tuple', () => {
		const r = resolveTemporalRegime({ timeUnits: 'weeks' });
		expect(r.nonCalendar).toBe(false);
		expect(r.sweepMode).toBe('episodic');
		expect(r.unitLabel).toBe('units'); // the .get(..., "units") default
	});
});

describe('the bandwidth (temporal.py:486-495)', () => {
	it('clips to [0.05, 2.0] years, and the acceptance run is where the FLOOR binds', () => {
		// MEASURED: the acceptance alignment spans 0.666 years, so 5 % of it is 0.0333 and the floor
		// is what sets h = 0.05. A port that dropped the clip would smooth over a third as much time.
		expect(resolveTemporalBandwidth(0.666015625, { timeUnits: 'years' })).toBe(0.05);
		expect(resolveTemporalBandwidth(9.0, { timeUnits: 'years' })).toBeCloseTo(0.45, 12);
		expect(resolveTemporalBandwidth(100.0, { timeUnits: 'years' })).toBe(2.0);
	});

	it('has no ceiling off the calendar path, so the Gaussian does not underflow', () => {
		expect(resolveTemporalBandwidth(100000, { timeUnits: 'generations', numTimePoints: 250 })).toBe(5000);
	});

	it('branches on time_units == "years", not on the non-calendar tuple', () => {
		// An unrecognised unit is calendar everywhere else and takes the non-calendar bandwidth
		// formula here. Replicated, not tidied.
		expect(resolveTemporalBandwidth(100.0, { timeUnits: 'weeks', numTimePoints: 250 })).toBe(5.0);
	});

	it('an explicit positive bandwidth wins; zero and negative do not', () => {
		expect(resolveTemporalBandwidth(1, { bandwidth: 0.3 })).toBe(0.3);
		expect(resolveTemporalBandwidth(1, { bandwidth: 0 })).toBe(0.05);
		expect(resolveTemporalBandwidth(1, { bandwidth: -1 })).toBe(0.05);
	});
});

describe('the root consensus (temporal.py:337-392)', () => {
	it('has every case the generator wrote', () => {
		expect(ROOT_CASES.length).toBe(8);
		expect(MANIFEST.counts.temporal.root).toBe(ROOT_CASES.length);
	});

	for (const c of ROOT_CASES) {
		it(`${c.name}: reproduces the reference exactly`, () => {
			const L = c.inputs.a.length;
			const N = c.inputs.a[0].length;
			const aValid = Int16Array.from(c.inputs.a.flat());
			const validIdx = c.inputs.valid_taxa_mask
				? c.inputs.valid_taxa_mask.map((v, i) => (v ? i : -1)).filter((i) => i >= 0)
				: null;
			// The fixture's `a` is the FULL matrix; when a mask is present the filtered view is what
			// the reference works on, exactly as the application hands it over.
			let aUse = aValid;
			let nUse = N;
			if (validIdx) {
				nUse = validIdx.length;
				aUse = new Int16Array(L * nUse);
				for (let s = 0; s < L; s++) for (let j = 0; j < nUse; j++) aUse[s * nUse + j] = aValid[s * N + validIdx[j]];
			}
			const r = inferRootSequence({
				aValid: aUse, L, N: nUse, taxa: c.inputs.taxa,
				validTaxaIndices: validIdx ?? c.inputs.taxa.map((_, i) => i),
				taxaDates: Float32Array.from(c.inputs.taxa_dates),
				rootTaxon: c.inputs.root_taxon ?? null
			});
			expect(Array.from(r.rootIndices)).toEqual(c.outputs.root_indices);
			expect(r.rootAas).toEqual(c.outputs.root_aas);
			if (c.outputs.n_early !== undefined) expect(r.nEarly).toBe(c.outputs.n_early);
		});
	}

	it('the window is 3 below 60 dated taxa and 25 above 500, not "the earliest 5 %"', () => {
		expect(rootConsensusWindow(10)).toBe(3);
		expect(rootConsensusWindow(59)).toBe(3);
		expect(rootConsensusWindow(60)).toBe(3);
		expect(rootConsensusWindow(95)).toBe(4); // the acceptance run's own window
		expect(rootConsensusWindow(500)).toBe(25);
		expect(rootConsensusWindow(5000)).toBe(25);
	});

	it('UPSTREAM BUG: an unknown residue in an explicit root taxon becomes ALANINE', () => {
		const c = ROOT_CASES.find((x) => x.name === 'explicit_root_taxon_unknown_becomes_alanine');
		// temporal.py:355 substitutes index 0 rather than a sentinel, so a gapped root position reads
		// as A and every other taxon then "differs from the root" there. Pinned so a later fix in the
		// port alone would fail here and be settled upstream first.
		expect(c.outputs.root_aas[1]).toBe('A');
		expect(c.outputs.root_indices[1]).toBe(0);
	});

	it('sorts dates STABLY, where numpy uses an unstable quicksort', () => {
		// temporal.py:365 is np.argsort's default. On a tie-heavy surveillance set the window
		// membership -- hence the root, hence every attribution -- becomes implementation-defined.
		// This port pins one choice and says so; the acceptance run's own window is tie-free.
		const order = stableArgsortAscending([3, 1, 3, 1, 2]);
		expect(Array.from(order)).toEqual([1, 3, 4, 0, 2]);
		const desc = stableArgsortDescending([3, 1, 3, 1, 2]);
		expect(Array.from(desc)).toEqual([0, 2, 4, 1, 3]);
	});
});

describe('the attribution, the kernel and the metric (temporal.py:534-582)', () => {
	const denseT = temporalTimeGrid(2009.25, 2009.75, 8, false).denseT;
	const dates = Float32Array.from([2009.3, 2009.4, 2009.5, 2009.6]);

	it('an unknown residue contributes 0, not a difference', () => {
		const attns = Float32Array.from([0.25, 0.25, 0.25, 0.25]);
		const a = Int16Array.from([5, 7, 20, 5]);
		const d = directionalAttribution(attns, a, [5], 1, 4);
		expect(Array.from(d)).toEqual([0, 0.25, 0, 0]);
	});

	it('the kernel rows sum to 1 up to the guard, which is on the ROW SUM and not per element', () => {
		// `weights / (weights.sum(axis=1, keepdims=True) + 1e-8)`: the row sums to
		// S/(S + 1e-8), so the deviation from 1 is 1e-8/S and grows as the row's raw weight shrinks.
		// That the rows normalise AT ALL is what makes a permutation of the taxon columns still a
		// valid weight matrix, which is why nothing is renormalised per draw.
		const W = nadarayaWatsonWeights(denseT, dates, 0.05);
		for (let t = 0; t < 8; t++) {
			let s = 0;
			for (let n = 0; n < 4; n++) s += W[t * 4 + n];
			expect(Math.abs(s - 1)).toBeLessThan(1e-5);
		}
	});

	it('a permutation of the taxon columns leaves the rows normalised, so nothing is renormalised per draw', () => {
		const W = nadarayaWatsonWeights(denseT, dates, 0.05);
		const WT = taxonMajorWeights(W, 8, 4);
		const perm = temporalDrawPermutation(4, 42, 0);
		for (let t = 0; t < 8; t++) {
			let s = 0;
			let unpermuted = 0;
			for (let n = 0; n < 4; n++) {
				s += WT[perm[n] * 8 + t];
				unpermuted += WT[n * 8 + t];
			}
			expect(Math.abs(s - unpermuted)).toBeLessThan(1e-15);
		}
	});

	it('the weights must be float64: the acceptance run exports values three decades below float32', () => {
		// MEASURED on the acceptance run: the smallest exported prevalence is 2.99e-41, and float32's
		// smallest normal is 1.18e-38. A Float32Array kernel flushes it to zero.
		const far = Float32Array.from([2009.25, 2009.26, 2009.27, 2009.9]);
		const W = nadarayaWatsonWeights(temporalTimeGrid(2009.25, 2009.9, 4, false).denseT, far, 0.01);
		let smallest = Infinity;
		for (const v of W) if (v > 0) smallest = Math.min(smallest, v);
		expect(smallest).toBeLessThan(1.18e-38);
		expect(Math.fround(smallest)).toBe(0);
	});

	it('skipping zero attributions is bit-identical to summing them', () => {
		const N = 6;
		const T = 8;
		const W = nadarayaWatsonWeights(denseT, Float32Array.from([2009.3, 2009.35, 2009.4, 2009.5, 2009.6, 2009.7]), 0.05);
		const WT = taxonMajorWeights(W, T, N);
		const sparse = Float32Array.from([0, 0.3, 0, 0, 0.2, 0]);
		const dense = Float64Array.from(sparse);
		const a = smoothTrajectories(sparse, 1, N, WT, T);
		const b = new Float64Array(T);
		for (let n = 0; n < N; n++) for (let t = 0; t < T; t++) b[t] += dense[n] * WT[n * T + t];
		// x + 0.0 = x exactly for finite x, and the accumulator never sees −0.0 + 0.0.
		for (let t = 0; t < T; t++) expect(a[t]).toBe(b[t]);
	});

	it('the fixation metric is the shift from the first grid point; the episodic one is a clamped gradient', () => {
		const curves = Float64Array.from([1, 2, 4, 4, 3, 3, 3, 3]);
		const fx = sweepMetric(curves, 1, 8, { sweepMode: 'fixation' });
		expect(fx[0]).toBe(0);
		expect(fx[2]).toBe(3);
		const ep = sweepMetric(curves, 1, 8, { sweepMode: 'episodic', gradT: denseT });
		for (const v of ep) expect(v).toBeGreaterThanOrEqual(0);
	});

	it('an all-zero row reports peak_date = denseT[0], because argmax ties to the first index', () => {
		// Numerically right, and a result semantic the application must render as an em dash rather
		// than as "peaked at the start of the epidemic".
		const zeros = new Float64Array(8);
		const { normDenseT, gradT } = temporalTimeGrid(2009.25, 2009.75, 8, false);
		const st = temporalTrajectoryStatistics({
			curves: zeros, metric: zeros, L: 1, T: 8, denseT, normDenseT, gradT, sweepMode: 'episodic'
		});
		expect(st.peakTimes[0]).toBe(denseT[0]);
		expect(st.peakIntensities[0]).toBe(0);
		expect(st.fwhm[0]).toBe(0);
	});

	it('t_half and fwhm are FLOAT32 fields, so the grid dates are downcast on assignment', () => {
		const curves = Float64Array.from([0, 0, 1, 2, 1, 0, 0, 0]);
		const { normDenseT, gradT } = temporalTimeGrid(2009.25, 2009.75, 8, false);
		const metric = sweepMetric(curves, 1, 8, { sweepMode: 'fixation' });
		const st = temporalTrajectoryStatistics({
			curves, metric, L: 1, T: 8, denseT, normDenseT, gradT, sweepMode: 'fixation',
			meanAttns: Float32Array.from([0.25, 0.25, 0.25, 0.25]), N: 4
		});
		expect(st.tHalfStart[0]).toBe(Math.fround(st.tHalfStart[0]));
		expect(st.tHalfEnd[0]).toBe(Math.fround(st.tHalfEnd[0]));
	});
});

describe('the energy floors (temporal.py:599-616)', () => {
	it('UPSTREAM BUG: tau_peak is tested by VALUE, so the documented default is silently overridden', () => {
		const supplied = resolveEnergyFloors({ timespan: 1, nonCalendar: false, tauPeak: 1e-4 });
		const omitted = resolveEnergyFloors({ timespan: 1, nonCalendar: false, tauPeak: null });
		expect(supplied.tauPeak).toBe(0.5e-4);
		expect(supplied.tauPeak).toBe(omitted.tauPeak);
		expect(supplied.tauPeakOverridden).toBe(true);
		// A value the caller could plausibly supply and that is NOT the magic one survives.
		expect(resolveEnergyFloors({ timespan: 1, nonCalendar: false, tauPeak: 2e-4 }).tauPeak).toBe(2e-4);
	});

	it('the calendar tau_auc scales with the timespan and the non-calendar floors do not', () => {
		// The acceptance run's own numbers: span 0.666 yr gives max(1, 0.133) = 1, so tau_auc = 1e-5.
		expect(resolveEnergyFloors({ timespan: 0.666015625, nonCalendar: false }).tauAuc).toBe(1e-5);
		expect(resolveEnergyFloors({ timespan: 50, nonCalendar: false }).tauAuc).toBeCloseTo(1e-4, 15);
		const nc = resolveEnergyFloors({ timespan: 100000, nonCalendar: true });
		expect(nc.tauAuc).toBe(0.005);
		expect(nc.tauPeak).toBe(0.01);
	});

	it('the mask is (~inv) & peak >= tau_peak & auc >= tau_auc, with >= on both', () => {
		const inv = [0, 0, 1, 0];
		const peak = [1, 0.5, 9, 9];
		const auc = [1, 9, 9, 0.5];
		const { mask, candIndices } = stageOneMask(inv, peak, auc, 1, 1);
		expect(Array.from(mask)).toEqual([1, 0, 0, 0]);
		expect(Array.from(candIndices)).toEqual([0]);
	});
});

describe('confirmation, the escape hatch and the four-way ladder (temporal.py:681-716)', () => {
	const L = 4;
	const stage1 = [1, 1, 1, 1];

	it('the solitary regime drops the shape gate, and three or fewer candidates is solitary', () => {
		const r = confirmSweeps({
			stage1Mask: stage1, pPerm: [0.01, 0.01, 0.01, 0.01], r2Fpca: [0, 0, 0, 0],
			lrts: [0, 0, 0, 0], L, nStage1: 3, nonCalendar: false
		});
		expect(r.solitaryRegime).toBe(true);
		expect(r.nSweeps).toBe(4); // the R² of 0 would have failed the gate had it been applied
	});

	it('any non-calendar run is solitary, whatever the candidate count', () => {
		const r = confirmSweeps({
			stage1Mask: stage1, pPerm: [0.01, 0.01, 0.01, 0.01], r2Fpca: [0, 0, 0, 0],
			lrts: [0, 0, 0, 0], L, nStage1: 40, nonCalendar: true
		});
		expect(r.solitaryRegime).toBe(true);
	});

	it('UPSTREAM BUG: the escape hatch is unrecorded in every output file, so this port records it', () => {
		// temporal.py:692-693 fires on the COUNT being zero, uses a hard-coded 0.10 (not perm_alpha)
		// and the hard-coded χ²₁ cut 3.84 against the FLOAT32 LRT. A run that confirmed nothing and
		// one that confirmed thirty through the hatch look identical in the CSV.
		const r = confirmSweeps({
			stage1Mask: stage1, pPerm: [1, 1, 1, 1], r2Fpca: [1, 1, 1, 1],
			lrts: [3.84, 3.83, 0, 10], L, nStage1: 4, nonCalendar: false
		});
		expect(r.escapeHatchUsed).toBe(true);
		expect(Array.from(r.isConfirmedSweep)).toEqual([1, 0, 0, 1]);
	});

	it('the hatch is calendar-only and never fires in the solitary regime', () => {
		const r = confirmSweeps({
			stage1Mask: [1, 1, 1, 1], pPerm: [1, 1, 1, 1], r2Fpca: [1, 1, 1, 1],
			lrts: [10, 10, 10, 10], L, nStage1: 3, nonCalendar: false
		});
		expect(r.escapeHatchUsed).toBe(false);
		expect(r.nSweeps).toBe(0);
	});

	it('an INVARIABLE site is never FILTERED_STATIC_NOISE, even when its static q would qualify', () => {
		// Its q was forced to exactly 1 when BH ran over the variable subset only, and the ladder
		// tests `inv` first anyway. Both halves matter and both are here.
		const r = classifyTemporalSites({
			inv: [1, 0, 0, 0], stage1Mask: [0, 0, 1, 1], isConfirmedSweep: [0, 0, 0, 1],
			qStatic: [0.01, 0.01, 0.5, 0.5], L: 4
		});
		expect(r.classification).toEqual([
			TEMPORAL_CLASSES.INVARIABLE, TEMPORAL_CLASSES.FLAT_NO_SIGNAL,
			TEMPORAL_CLASSES.TEMPORAL_NOISE, TEMPORAL_CLASSES.CONFIRMED_SWEEP
		]);
		expect(r.crossClassification).toEqual([
			TEMPORAL_CROSS_CLASSES.NEGATIVE_CONSENSUS, TEMPORAL_CROSS_CLASSES.FILTERED_STATIC_NOISE,
			TEMPORAL_CROSS_CLASSES.NEGATIVE_CONSENSUS, TEMPORAL_CROSS_CLASSES.RESCUED_SWEEP
		]);
		expect(Array.from(r.isRescued)).toEqual([0, 0, 0, 1]);
		expect(r.counts).toEqual({ confirmed: 1, concordant: 0, rescued: 1, filteredStaticNoise: 1 });
	});

	it('reaches CONCORDANT_SWEEP, which neither bundled example does', () => {
		const r = classifyTemporalSites({
			inv: [0], stage1Mask: [1], isConfirmedSweep: [1], qStatic: [0.1], L: 1
		});
		expect(r.crossClassification[0]).toBe(TEMPORAL_CROSS_CLASSES.CONCORDANT_SWEEP);
	});
});

describe('the fPCA shape gate (temporal.py:667-679)', () => {
	it('UPSTREAM BUG: the gate is VACUOUS at exactly four candidates', () => {
		// k_eff = min(4, C) and mean-centred rows have rank at most min(C, T−1), so at C = 4 the
		// top-4 subspace contains the whole row space and every candidate scores 1. C ≤ 3 is already
		// the solitary regime, so C = 4 is the only live case; the application must say "the shape
		// gate could not discriminate (4 candidates)" rather than print R² = 1.00.
		const T = 16;
		const rows = new Float64Array(4 * T);
		for (let i = 0; i < 4; i++) for (let t = 0; t < T; t++) rows[i * T + t] = Math.sin((i + 1) * t);
		const g = fpcaShapeGate(rows, 4, T);
		expect(g.vacuous).toBe(true);
		for (const v of g.r2) expect(v).toBeCloseTo(1, 9);
	});

	it('UPSTREAM BUG: a flat candidate scores r2 = 1 on no signal at all', () => {
		// stds_c = std + 1e-8 turns a constant row into exact zeros, so sse = 0, sst = 1e-8 and the
		// row passes min_r2_fpca. Replicated; the row is reported in `flatRows` so a surface can mark it.
		const T = 16;
		const rows = new Float64Array(6 * T);
		for (let i = 0; i < 5; i++) for (let t = 0; t < T; t++) rows[i * T + t] = Math.sin((i + 1) * t * 0.7);
		// row 5 stays constant
		const g = fpcaShapeGate(rows, 6, T);
		expect(g.r2[5]).toBe(1);
		expect(g.flatRows).toContain(5);
	});
});

describe('the date-shuffling null (temporal.py:618-666)', () => {
	const T = 12;
	const N = 9;
	const C = 5;
	const denseT = temporalTimeGrid(2009.0, 2010.0, T, false).denseT;
	const dates = Float32Array.from({ length: N }, (_, i) => 2009.0 + i / (N - 1));
	const W = nadarayaWatsonWeights(denseT, dates, 0.1);
	const WT = taxonMajorWeights(W, T, N);
	const attrs = new Float32Array(C * N);
	for (let c = 0; c < C; c++) for (let n = 0; n < N; n++) attrs[c * N + n] = n >= c + 2 ? 0.1 * (c + 1) : 0;
	const curves = smoothTrajectories(attrs, C, N, WT, T);
	const vObs = temporalPermStat(curves, C, T, { sweepMode: 'episodic', gradT: denseT });
	const draw = (from, to, exceed) => temporalNullDraws({
		candAttrs: attrs, C, N, T, WT, vObs, sweepMode: 'episodic', gradT: denseT, seed: 42,
		fromDraw: from, toDraw: to, exceed
	});

	it('v_obs comes from the SAME kernel at the identity permutation as the reported curves', () => {
		// The reference computes v_obs from one dgemm and v_p from another; at the `>=` tie boundary
		// -- and a degenerate constant curve puts every draw exactly there -- a different summation
		// order on the two sides biases the count systematically. One path, both sides.
		const identity = smoothTrajectories(attrs, C, N, WT, T, { baseRow: Int32Array.from({ length: N }, (_, n) => n * T) });
		for (let i = 0; i < curves.length; i++) expect(identity[i]).toBe(curves[i]);
	});

	it('is resumable: draws split across calls give the same counts as one long call', () => {
		// The property that lets the application chunk, report progress, cancel and resume without
		// any of that living in the library -- and the one a later refactor is most likely to break.
		const whole = draw(0, 40, null);
		const split = new Int32Array(C);
		draw(0, 7, split);
		draw(7, 23, split);
		draw(23, 40, split);
		expect(Array.from(split)).toEqual(Array.from(whole));
	});

	it('a run stopped at B is bit-identical to a run configured with B, because draws are addressed by index', () => {
		const stopped = draw(0, 17, null);
		const configured = temporalNullDraws({
			candAttrs: attrs, C, N, T, WT, vObs, sweepMode: 'episodic', gradT: denseT, seed: 42, toDraw: 17
		});
		expect(Array.from(stopped)).toEqual(Array.from(configured));
	});

	it('each draw is a genuine permutation, and a different draw index is a different one', () => {
		const a = temporalDrawPermutation(N, 42, 0);
		const b = temporalDrawPermutation(N, 42, 1);
		expect(Array.from(a).slice().sort((x, y) => x - y)).toEqual([...Array(N).keys()]);
		expect(Array.from(a)).not.toEqual(Array.from(b));
		expect(Array.from(temporalDrawPermutation(N, 42, 0))).toEqual(Array.from(a));
	});

	it('p is (1 + exceed)/(B + 1), so its grid step is 1/(B + 1)', () => {
		const { p, gridStep } = temporalPermPValues(Int32Array.from([0, 1, 4, 100]), 100);
		expect(gridStep).toBeCloseTo(1 / 101, 15);
		expect(p[0]).toBeCloseTo(1 / 101, 15);
		expect(p[2]).toBeCloseTo(5 / 101, 15);
		expect(p[3]).toBeCloseTo(101 / 101, 15);
	});

	it('q_perm cannot fall below C/(B + 1), which is why the call is made on p', () => {
		// UPSTREAM OBSERVATION, not a fix: on the acceptance run that floor is 246/101 = 2.4, clipped
		// to 1, and every confirmed sweep there reports the same q_perm = 0.4298. Reaching q ≤ 0.10
		// would need B + 1 ≥ 10·C -- about 2,500 draws for that run, where the reference's own
		// default is 1,000. A surface must not print q_perm as a decision threshold.
		const C = 246;
		const exceed = new Int32Array(C).fill(100); // every candidate at p = 1 ...
		exceed[0] = 0; // ... except one, at the smallest p the grid can express
		const { p, q } = temporalPermPValues(exceed, 100);
		expect(p[0]).toBeCloseTo(1 / 101, 15);
		expect(C * p[0]).toBeGreaterThan(1); // the BH value at rank 1 is 2.44, before the clip
		expect(q[0]).toBe(1);
	});

	it('reports progress once per draw and can be driven without a callback', () => {
		const seen = [];
		temporalNullDraws({
			candAttrs: attrs, C, N, T, WT, vObs, sweepMode: 'episodic', gradT: denseT, seed: 42,
			toDraw: 3, onProgress: (p) => seen.push(p)
		});
		expect(seen).toEqual([
			{ phase: 'temporal-null', done: 1, total: 3 },
			{ phase: 'temporal-null', done: 2, total: 3 },
			{ phase: 'temporal-null', done: 3, total: 3 }
		]);
	});

	it('the fixation statistic is a signed correlation with normalised time', () => {
		const { normDenseT } = temporalTimeGrid(0, 1, T, true);
		const rising = Float64Array.from(normDenseT);
		const falling = Float64Array.from(normDenseT, (v) => -v);
		const both = new Float64Array(2 * T);
		both.set(rising, 0);
		both.set(falling, T);
		const v = temporalPermStat(both, 2, T, { sweepMode: 'fixation', normDenseT });
		expect(v[0]).toBeCloseTo(1, 9);
		expect(v[1]).toBeCloseTo(-1, 9);
	});
});

describe('the wave decomposition (temporal.py:718-741)', () => {
	it('uses the confirmed sweeps when there are at least four, and the top peaks otherwise', () => {
		const L = 8;
		const T = 12;
		const velocity = new Float64Array(L * T);
		for (let s = 0; s < L; s++) for (let t = 0; t < T; t++) velocity[s * T + t] = Math.sin((s + 1) * (t + 1) * 0.3);
		const peaks = Float64Array.from({ length: L }, (_, s) => L - s);
		const four = temporalWaveDecomposition({
			velocity, L, T, isConfirmedSweep: [0, 1, 1, 1, 1, 0, 0, 0], peakIntensities: peaks, candIndices: [1, 2, 3, 4]
		});
		expect(Array.from(four.fIndices)).toEqual([1, 2, 3, 4]);
		const fallback = temporalWaveDecomposition({
			velocity, L, T, isConfirmedSweep: [0, 0, 1, 0, 0, 0, 0, 0], peakIntensities: peaks, candIndices: [2, 5, 6]
		});
		// max(4, C) = 4 of the highest peaks, over ALL L sites -- not over the candidates.
		expect(Array.from(fallback.fIndices)).toEqual([0, 1, 2, 3]);
	});

	it('the fallback sort is STABLE, because most peak intensities are tied at zero', () => {
		// MEASURED on the acceptance run: 4111 of the 4384 peak intensities are exactly 0, so numpy's
		// quicksort picks an implementation-defined set. This port pins one and says so.
		const L = 6;
		const T = 8;
		const velocity = new Float64Array(L * T);
		const flat = temporalWaveDecomposition({
			velocity, L, T, isConfirmedSweep: new Uint8Array(L), peakIntensities: new Float64Array(L), candIndices: []
		});
		expect(Array.from(flat.fIndices)).toEqual([0, 1, 2, 3]);
	});

	it('zero-pads the waves to four rows, the way _waves.csv writes them', () => {
		const T = 10;
		const velocity = new Float64Array(2 * T);
		for (let t = 0; t < T; t++) {
			velocity[t] = Math.sin(t);
			velocity[T + t] = Math.cos(t);
		}
		const wd = temporalWaveDecomposition({
			velocity, L: 2, T, isConfirmedSweep: [1, 1], peakIntensities: [1, 1], candIndices: [0, 1]
		});
		expect(wd.nWaves).toBe(2);
		expect(wd.waves.length).toBe(4 * T);
		for (let t = 0; t < T; t++) expect(wd.waves[2 * T + t]).toBe(0);
		for (let k = 2; k < 4; k++) for (let s = 0; s < 2; s++) expect(wd.loadings[s * 4 + k]).toBe(0);
	});

	it('reports the spectrum, the gaps and both degeneracy flags for the record', () => {
		const T = 12;
		const L = 5;
		const velocity = new Float64Array(L * T);
		for (let s = 0; s < L; s++) for (let t = 0; t < T; t++) velocity[s * T + t] = Math.sin((s + 1) * t * 0.4);
		const wd = temporalWaveDecomposition({
			velocity, L, T, isConfirmedSweep: [1, 1, 1, 1, 1], peakIntensities: [1, 1, 1, 1, 1], candIndices: [0, 1, 2, 3, 4]
		});
		expect(wd.sigma.length).toBeGreaterThan(0);
		expect(wd.gaps.length).toBe(wd.nWaves);
		expect(wd.nearDegenerate.length).toBe(wd.nWaves);
		expect(wd.rankDeficient.length).toBe(wd.nWaves);
		expect(wd.waveSign).toBe('canonical');
		expect(WAVE_RANK_EPS).toBeCloseTo(Math.sqrt(Number.EPSILON), 20);
	});
});

describe('the mutation labels (temporal.py:743-760)', () => {
	it('takes the modal NON-root residue, falling back twice', () => {
		const N = 6;
		// site 0: root A(0), derived mostly C(1); site 1: nothing but the root; site 2: all unknown.
		const aValid = Int16Array.from([0, 1, 1, 2, 0, 1, 5, 5, 5, 5, 5, 5, 20, 20, 20, 20, 20, 20]);
		const r = temporalMutationLabels({ aValid, L: 3, N, rootIndices: [0, 5, 7], rootAas: ['A', 'G', 'I'] });
		expect(r.derivedAas[0]).toBe('C');
		expect(r.derivedAas[1]).toBe('G'); // no non-root residue: the modal VALID residue is the root
		expect(r.derivedAas[2]).toBe('I'); // nothing valid at all: the root residue itself
		expect(r.mutationLabels).toEqual(['A1C', 'G2G', 'I3I']);
		expect(r.domains).toEqual(['Core', 'Core', 'Core']);
	});

	it('takes a domain map keyed on the ONE-based site number', () => {
		const aValid = Int16Array.from([0, 1]);
		const r = temporalMutationLabels({
			aValid, L: 1, N: 2, rootIndices: [0], rootAas: ['A'], domainMap: new Map([[1, 'RBD']])
		});
		expect(r.domains).toEqual(['RBD']);
	});
});

describe('the writers', () => {
	it('pyFloat32Repr matches numpy, including the two cases a digit-string rule gets wrong', () => {
		// np.float32(1e-4) is 9.99999975e-05, whose exponent is −5, so it prints in scientific form
		// with the CARRIED digit exponent; np.float32(2e-4) is 2.00000009e-04 and prints positionally.
		expect(pyFloat32Repr(1e-4)).toBe('1e-04');
		expect(pyFloat32Repr(2e-4)).toBe('0.0002');
		// And the half-to-EVEN tie, which toExponential gets wrong and which twelve of the acceptance
		// run's t_half_end cells land on.
		expect(pyFloat32Repr(2009.53125)).toBe('2009.5312');
		expect(pyFloat32Repr(3.062294)).toBe('3.062294');
		expect(pyFloat32Repr(999999.94)).toBe('999999.94');
		expect(pyFloat32Repr(1234567.8)).toBe('1.2345678e+06');
		expect(pyFloat32Repr(0)).toBe('0.0');
		expect(pyFloat32Repr(-0)).toBe('-0.0');
		expect(pyFloat32Repr(1.4e-45)).toBe('1e-45');
		expect(pyFloat32Repr(3.4028235e38)).toBe('3.4028235e+38');
		expect(pyFloat32Repr(NaN)).toBe('nan');
		expect(pyFloat32Repr(Infinity)).toBe('inf');
	});

	it('pyRound is half-to-even on the binary value, where toFixed is not', () => {
		expect(pyRound(0.0078125, 6)).toBe(0.007812);
		expect(Number((0.0078125).toFixed(6))).toBe(0.007813);
		expect(pyRound(0.666015625, 2)).toBe(0.67);
		expect(pyRound(0.05, 4)).toBe(0.05);
	});

	it('the summary names two fields "years" whatever the units are', () => {
		// UPSTREAM QUIRK replicated: `timespan_years` and `bandwidth_years` keep the word in a
		// generations run, as does the CSV's `fwhm_years`. A surface must not print the unit from the
		// field name.
		const json = temporalSummaryJson({
			alignment: 'a.fasta', tree: null, taxaTotal: 3, taxaTimestamped: 3, codonsTotal: 1,
			codonsVariable: 1, codonsInvariable: 0, timespan: 5000, tMin: 0, tMax: 5000,
			bandwidth: 250, sigStaticQ10: 0, stage1Candidates: 0, confirmedSweeps: 0,
			concordantSweeps: 0, rescuedSweeps: 0, filteredStaticNoise: 0, varExplained: [1, 0, 0, 0],
			runtimeSec: null
		});
		expect(JSON.parse(json).timespan_years).toBe(5000);
		expect(JSON.parse(json).bandwidth_years).toBe(250);
		expect(json).toContain('"t_min": 0.0'); // a float field stays a float, as Python writes it
		expect(json.split('\n')[1]).toBe('  "alignment": "a.fasta",'); // indent 2, key order preserved
	});
});
