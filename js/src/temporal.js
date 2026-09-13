/**
 * WHY THIS FILE EXISTS
 *
 * The pure arithmetic of `hyphaeon/temporal.py`'s temporal-selection pillar — the second of the two
 * pillars PLAN-TEMPORAL.md describes, and the one that asks *which codons were under selection
 * when* rather than *when the ancestor lived*. The reference is one 500-line function,
 * `run_temporal_surveillance` (temporal.py:399-905), which interleaves model loading, alignment
 * preparation, printing, CSV writing and figure rendering with about a dozen numeric steps. This
 * file is those numeric steps and nothing else, split at the seams the reference's own comments mark
 * and named after them, so the application can drive them in a worker, chunk the expensive one,
 * cancel it, and put sentences in front of a reader — none of which belongs here (PLAN.md §5.5).
 *
 * WHAT IS DELIBERATELY ABSENT. No model, no session, no tokenisation, no tree, no file or metadata
 * ingestion, no AbortSignal, no worker, no `B` default, no refusal copy, no progress message. The
 * date *ingestion* half of temporal.py:73-330 is already split the same way: `dates.js` holds the
 * string→decimal arithmetic and the application owns the delimiter sniffing, the Auspice walk and
 * the column discovery, because the application's ingestion is deliberately WIDER than the
 * reference's (D31) and that difference is a sentence a reader must be shown, not a number.
 *
 * WHAT THE APPLICATION MUST STILL DECIDE, because the arithmetic here does not: the reference scores
 * EVERY site through the model, invariable ones included (temporal.py:513-520), which is 4384 forward
 * rows against our meme pass's 273 on the acceptance alignment. Nothing in this file needs the
 * invariable rows — `delta_root` is identically zero there under a consensus root, so their curves,
 * velocities and loadings are exactly zero whatever the attention was — but `lrt` and `p_static`
 * ARE reported at those sites, so a surface that skips them must report those two as ABSENT rather
 * than zero.
 *
 * THE FLOAT32 BOUNDARIES, all three load-bearing and all three easy to miss:
 *   1. dates are cast to float32 at temporal.py:468, so `t_min` is 2009.2490234375, not 2009.249,
 *      and the whole time axis inherits that rounding. Pass a Float32Array.
 *   2. `curves_matrix` is float32 `[L,N]` @ float64 `[N,T]`, which numpy PROMOTES: every trajectory,
 *      velocity, peak, area and width below is float64.
 *   3. `t_half_start`, `t_half_end` and `fwhm_arr` are declared float32 (temporal.py:584-586), so
 *      the float64 grid dates are DOWNCAST on assignment — `Math.fround` at exactly those stores.
 *
 * WHAT THE NULL COSTS, measured on this machine (node 22, x64 under Rosetta on an Apple-silicon Mac,
 * so this is a floor and a native arm64 browser should be 1.5-2x faster). The work is
 * `W = B · C · T · (nnz/C + 7)` — the matmul plus a fixed ~7T for the statistic, which DOMINATES at
 * realistic sparsity. On the acceptance alignment (C = 246, N = 95, T = 60, nnz = 1000, so the mean
 * derived fraction is 0.043 and 96 % of the attribution matrix is zero): 28 ms at B = 100 and 253 ms
 * at B = 1000, i.e. **R = 0.65e9 units/s**. The BigInt PRNG is 23 ms of that 253 — 9 %, and not worth
 * replacing. Two decisions are measured rather than assumed: the CSR layout over the candidate rows
 * is 2.1x faster than a dense scan with a zero test (0.30e9 before it), and taking the gradient over
 * the whole matrix in one call instead of row by row is most of the rest. An application budgeting
 * the section should use R = 0.65e9 and re-measure; the plan's own estimate of 1.0e9 was taken on a
 * kernel with no `Xoshiro256` in it.
 *
 * UPSTREAM QUIRKS REPLICATED AND FLAGGED HERE (PLAN.md §5.3 rule 3 — port faithfully, fix upstream):
 *   Q1 `tau_peak == 1e-4` is tested by VALUE (temporal.py:605, 610), so a caller who passes the
 *      documented default explicitly is silently overridden to 0.5e-4. {@link resolveEnergyFloors}
 *      returns `tauPeakOverridden` so a surface can say so.
 *   Q2 an unknown residue in an EXPLICIT root taxon becomes index 0 = Alanine (temporal.py:365), not
 *      a sentinel, so every gapped position of that taxon reads as `A` and invariable sites acquire
 *      nonzero trajectories and wrong mutation labels.
 *   Q3 `n_early = max(3, min(25, int(0.05·N)))` is 3 below 60 dated taxa and 25 above 500 — not
 *      "the earliest 5 %" in either tail.
 *   Q4 `np.argsort` on dates (temporal.py:371) and on `−peak_intensities` (temporal.py:722) is an
 *      UNSTABLE quicksort. Both are ported as explicit STABLE sorts keyed on (value, index); the
 *      acceptance run's root window is tie-free (measured: argsort and argsort(kind='stable') agree
 *      on the four selected indices [93,12,38,4] and first diverge at rank 7), but 4111 of that
 *      run's 4384 peak intensities are tied at zero, so the fallback wave set is not reproducible
 *      across sort implementations and this port pins one choice.
 *   Q5 the escape hatch (temporal.py:692-693) is recorded in no output file, so a run that confirmed
 *      nothing and one that confirmed thirty through the hatch look identical in the CSV.
 *      {@link confirmSweeps} returns `escapeHatchUsed`.
 *   Q6 the fixation branch divides by `site_scale` and the episodic branch does not, so calendar
 *      candidate counts move with taxon count while the stage-one floors stay fixed.
 *   Q7 a flat candidate scores `r2_fpca = 1`: `stds_c = std + 1e-8` turns a constant row into zeros,
 *      giving `sse = 0`, `sst = 1e-8` and a pass on no signal at all.
 *   Q8 `k_eff = min(4, C)` with mean-centred rows makes the shape gate VACUOUS at C = 4 (rank ≤ 3),
 *      so every candidate scores 1. C ≤ 3 is already the solitary regime, so C = 4 is the live case.
 *   Q9 `mean_intensity` is the mean of the VELOCITY, not of the intensity (temporal.py:785).
 *
 * Ported from `hyphaeon/temporal.py` at veg/HyphAeon `phase-5c` (c7f246b): regimes 433-440, root
 * 337-392, attribution 534-540, smoothing 542-549, metric 551-582, widths 584-597, floors 599-616,
 * null 618-666, gate 667-679, confirmation 681-697, classification 699-716, waves 718-741, labels
 * 743-760. Line numbers are that file.
 */

import { numpyLinspace, numpyGradientRows, numpyTrapezoidRows } from './numeric/calculus.js';
import { numpyPairwiseSum, numpyMeanFloat32, numpyMeanFloat64, numpyVarFloat64, numpyStdFloat64 } from './numeric/reduce.js';
import { dominantTimeModes, projectionResidual } from './numeric/svd.js';
import { benjaminiHochberg } from './numeric/bh.js';
import { Xoshiro256 } from './numeric/prng.js';
import { REV_AA_MAP } from './sectors.js';

/** Float64 accumulation: no rounding between additions. */
const f64 = (/** @type {number} */ v) => v;

/** `AA_MAP` has 20 residues; every token at or above this is gap / stop / ambiguity (dataset.py:59). */
export const TEMPORAL_UNKNOWN_AA = 20;

/** The time-unit labels temporal.py:440 prints, and the `.get(..., "units")` default it falls back to. */
export const TEMPORAL_UNIT_LABELS = Object.freeze({ years: 'yrs', generations: 'gen', days: 'days', arbitrary: 'units' });

/** The three non-calendar unit names (temporal.py:434). Anything else, including a typo, is calendar. */
export const TEMPORAL_NON_CALENDAR_UNITS = Object.freeze(['generations', 'days', 'arbitrary']);

/** `classification` (temporal.py:701-715): what the dates alone say about a codon. */
export const TEMPORAL_CLASSES = Object.freeze({
	INVARIABLE: 'INVARIABLE',
	FLAT_NO_SIGNAL: 'FLAT_NO_SIGNAL',
	TEMPORAL_NOISE: 'TEMPORAL_NOISE',
	CONFIRMED_SWEEP: 'CONFIRMED_SWEEP'
});

/** `cross_classification` (temporal.py:701-715): what the dates add to the ordinary static scan. */
export const TEMPORAL_CROSS_CLASSES = Object.freeze({
	NEGATIVE_CONSENSUS: 'NEGATIVE_CONSENSUS',
	FILTERED_STATIC_NOISE: 'FILTERED_STATIC_NOISE',
	CONCORDANT_SWEEP: 'CONCORDANT_SWEEP',
	RESCUED_SWEEP: 'RESCUED_SWEEP'
});

/** The static-scan FDR cut the cross-classification tests (temporal.py:708, 711, 714). */
export const TEMPORAL_Q_STATIC_CUT = 0.10;

/** χ²₁ at 0.05, the hard-coded LRT the escape hatch ORs in (temporal.py:693). */
export const TEMPORAL_ESCAPE_LRT = 3.84;

/** The hard-coded p the escape hatch ORs in (temporal.py:693) — not `perm_alpha`. */
export const TEMPORAL_ESCAPE_P = 0.10;

/** How many collective wave modes the reference reports (temporal.py:719). */
export const TEMPORAL_WAVE_K = 4;

// ---------------------------------------------------------------------------------------------
// S0 — regime switches (temporal.py:433-440)
// ---------------------------------------------------------------------------------------------

/**
 * The four switches temporal.py:433-440 derives from `time_units`, `sweep_mode` and
 * `keep_duplicates`, and every branch below keys off one of them.
 *
 * `sweepMode` and `nonCalendar` are INDEPENDENT once resolved: a calendar run may be forced to
 * `fixation` and a generations run to `episodic`, and the reference tests one or the other at every
 * branch — never conflate them. In particular the gradient axis (temporal.py:561) and the energy
 * floors (599-616) key off UNITS, while the metric (563-566) and the null statistic (624-642) key
 * off MODE.
 *
 * `pruneDuplicates` is what the application must surface: in a surveillance set, identical
 * haplotypes sampled on different days collapse to one taxon and the other days are silently
 * deleted. It is true on the calendar path by default.
 *
 * @param {{ timeUnits?: string, sweepMode?: string, keepDuplicates?: boolean }} [options]
 * @returns {{ timeUnits: string, nonCalendar: boolean, sweepMode: 'episodic'|'fixation',
 *   pruneDuplicates: boolean, unitLabel: string }}
 */
export function resolveTemporalRegime({ timeUnits = 'years', sweepMode = 'auto', keepDuplicates = false } = {}) {
	const nonCalendar = TEMPORAL_NON_CALENDAR_UNITS.includes(timeUnits);
	let mode = sweepMode;
	if (mode === 'auto') mode = nonCalendar ? 'fixation' : 'episodic';
	if (mode !== 'episodic' && mode !== 'fixation') {
		throw new RangeError(`sweepMode must be 'auto', 'episodic' or 'fixation', got ${JSON.stringify(sweepMode)}`);
	}
	return {
		timeUnits,
		nonCalendar,
		sweepMode: mode,
		pruneDuplicates: !(keepDuplicates || nonCalendar),
		unitLabel: TEMPORAL_UNIT_LABELS[/** @type {keyof TEMPORAL_UNIT_LABELS} */ (timeUnits)] ?? 'units'
	};
}

// ---------------------------------------------------------------------------------------------
// S4 — bandwidth (temporal.py:486-495)
// ---------------------------------------------------------------------------------------------

/**
 * The Gaussian kernel bandwidth, resolved exactly as temporal.py:486-495 resolves it.
 *
 * The branch tests `time_units == "years"` — NOT `not non_calendar` — so an unrecognised unit string
 * takes the non-calendar formula while still being calendar everywhere else. Replicated.
 *
 * Both the 0.05 floor and the 2.0 ceiling of the calendar branch are quantities IN DECIMAL YEARS and
 * are meaningless on any other time coordinate; the non-calendar branch has no ceiling on purpose
 * (a 2.0-unit cap underflows the Gaussian at generation spacing). On the acceptance run the span is
 * 0.666015625 yr (t_min 2009.2490234375, t_max 2009.9150390625 — the CLI's own
 * `h1n1_cpu_waves.csv`), so 0.05·span = 0.0333 is clipped UP to the 0.05 floor and the floor
 * is what binds.
 *
 * @param {number} timespan `t_max − t_min`, in the run's own units
 * @param {{ timeUnits?: string, bandwidth?: number|null, numTimePoints?: number }} [options]
 *   `bandwidth` null, undefined or ≤ 0 means "resolve it"
 * @returns {number}
 */
export function resolveTemporalBandwidth(timespan, { timeUnits = 'years', bandwidth = null, numTimePoints = 250 } = {}) {
	if (bandwidth !== null && bandwidth !== undefined && bandwidth > 0) return bandwidth;
	if (timeUnits === 'years') return Math.min(Math.max(timespan * 0.05, 0.05), 2.0);
	return Math.max(timespan * 0.05, (timespan / Math.max(numTimePoints, 1)) * 2.0);
}

// ---------------------------------------------------------------------------------------------
// S5 — root inference (temporal.py:337-392)
// ---------------------------------------------------------------------------------------------

/**
 * How many earliest-sampled taxa the root consensus is taken over: `max(3, min(25, int(0.05·N)))`
 * (temporal.py:372). Exposed because it is the reference's own rule and not the "earliest 5 %" its
 * docstring claims — Q3.
 *
 * @param {number} n dated taxon count
 * @returns {number}
 */
export function rootConsensusWindow(n) {
	return Math.max(3, Math.min(25, Math.trunc(0.05 * n)));
}

/**
 * The ancestral residue at every codon, over the DATE-FILTERED alignment (temporal.py:337-392).
 *
 * Two paths, and they differ in more than their inputs:
 *   - an explicit `rootTaxon` that is present in the FULL taxon list AND survives the date filter
 *     supplies its own column verbatim, with Q2's Alanine substitution for any unknown residue;
 *   - otherwise the modal residue of the `rootConsensusWindow(N)` earliest-dated columns, falling
 *     back to the whole date-filtered matrix when the window has no valid residue at a site, and to
 *     index 0 when nothing does.
 *
 * TIE RULES, both deterministic and both worth pinning: the modal residue is `bincount` +
 * `argmax`, so the LOWEST amino-acid index wins a tie (`AA_MAP` order, A < C < D < …) — never a
 * max-by-count scan that keeps the last; and the date ordering is a STABLE ascending sort on
 * (date, original index) rather than numpy's unstable quicksort (Q4). Where the window boundary
 * falls inside a run of tied dates the two disagree, the root changes, and every attribution with
 * it — measured tie-free on the acceptance run, not tie-free in general.
 *
 * @param {{
 *   aValid: ArrayLike<number>, L: number, N: number,
 *   taxa?: ArrayLike<string>|null, validTaxaIndices?: ArrayLike<number>|null,
 *   taxaDates?: ArrayLike<number>|null, rootTaxon?: string|null
 * }} args `aValid` is the DATE-FILTERED `[L, N]` amino-acid token matrix, row-major; `taxa` is the
 *   FULL taxon list (the reference tests membership against it before filtering) and
 *   `validTaxaIndices` maps the N filtered columns back into it.
 * @returns {{ rootIndices: Int32Array, rootAas: string[], source: 'root-taxon'|'early-consensus'|'all-consensus',
 *   earlyIndices: Int32Array|null, nEarly: number }}
 */
export function inferRootSequence({ aValid, L, N, taxa = null, validTaxaIndices = null, taxaDates = null, rootTaxon = null }) {
	const rootIndices = new Int32Array(L);
	/** @type {string[]} */
	const rootAas = [];

	if (rootTaxon !== null && rootTaxon !== undefined && taxa) {
		let inFull = false;
		for (let i = 0; i < taxa.length; i++) if (taxa[i] === rootTaxon) { inFull = true; break; }
		if (inFull) {
			// The filtered name list the reference searches: taxa[valid_taxa_mask].
			let col = -1;
			if (validTaxaIndices) {
				for (let j = 0; j < validTaxaIndices.length; j++) {
					if (taxa[validTaxaIndices[j]] === rootTaxon) { col = j; break; }
				}
			} else {
				for (let j = 0; j < N && j < taxa.length; j++) if (taxa[j] === rootTaxon) { col = j; break; }
			}
			if (col >= 0) {
				for (let s = 0; s < L; s++) {
					const tok = aValid[s * N + col];
					// Q2: an unknown residue becomes 0 = Alanine, not a sentinel.
					const idx = tok < TEMPORAL_UNKNOWN_AA ? tok : 0;
					rootIndices[s] = idx;
					rootAas.push(REV_AA_MAP.get(idx) ?? 'X');
				}
				return { rootIndices, rootAas, source: 'root-taxon', earlyIndices: null, nEarly: 0 };
			}
		}
	}

	/** @type {Int32Array|null} */
	let earlyIndices = null;
	let nEarly = 0;
	if (taxaDates && taxaDates.length > 0) {
		const order = stableArgsortAscending(taxaDates);
		nEarly = rootConsensusWindow(taxaDates.length);
		earlyIndices = order.slice(0, nEarly);
	}

	const counts = new Int32Array(TEMPORAL_UNKNOWN_AA);
	for (let s = 0; s < L; s++) {
		counts.fill(0);
		let seen = 0;
		if (earlyIndices) {
			for (let j = 0; j < earlyIndices.length; j++) {
				const tok = aValid[s * N + earlyIndices[j]];
				if (tok < TEMPORAL_UNKNOWN_AA) { counts[tok]++; seen++; }
			}
		} else {
			for (let j = 0; j < N; j++) {
				const tok = aValid[s * N + j];
				if (tok < TEMPORAL_UNKNOWN_AA) { counts[tok]++; seen++; }
			}
		}
		if (seen === 0 && earlyIndices) {
			for (let j = 0; j < N; j++) {
				const tok = aValid[s * N + j];
				if (tok < TEMPORAL_UNKNOWN_AA) { counts[tok]++; seen++; }
			}
		}
		let idx = 0;
		if (seen > 0) {
			let best = -1;
			for (let k = 0; k < TEMPORAL_UNKNOWN_AA; k++) {
				if (counts[k] > best) { best = counts[k]; idx = k; }
			}
		}
		rootIndices[s] = idx;
		rootAas.push(REV_AA_MAP.get(idx) ?? '-');
	}
	return { rootIndices, rootAas, source: earlyIndices ? 'early-consensus' : 'all-consensus', earlyIndices, nEarly };
}

/**
 * Ascending order of `values` as a STABLE sort keyed on (value, original index) — the deterministic
 * stand-in for `np.argsort`'s unstable quicksort (Q4). NaN sorts last, as numpy puts it.
 *
 * @param {ArrayLike<number>} values
 * @returns {Int32Array}
 */
export function stableArgsortAscending(values) {
	const n = values.length;
	const idx = new Int32Array(n);
	for (let i = 0; i < n; i++) idx[i] = i;
	const arr = Array.from(idx);
	arr.sort((a, b) => {
		const va = values[a];
		const vb = values[b];
		if (Number.isNaN(va)) return Number.isNaN(vb) ? a - b : 1;
		if (Number.isNaN(vb)) return -1;
		return va - vb || a - b;
	});
	return Int32Array.from(arr);
}

/**
 * Descending order of `values` as a STABLE sort keyed on (−value, original index) — the stand-in for
 * `np.argsort(-x)` at temporal.py:722, where 4137 of the acceptance run's 4384 peak intensities are
 * tied at exactly zero and numpy's quicksort therefore picks an implementation-defined set (Q4).
 * COUNTED from the reference's own `h1n1_cpu_sites_summary.csv`: 4111 of those ties are the
 * invariable sites, whose trajectories are identically zero, and 26 are VARIABLE sites whose
 * positive velocity never left the floor — so the tie block is wider than the invariable mask, and
 * a port that assumed the two coincided would still be picking a different top-K.
 *
 * @param {ArrayLike<number>} values
 * @returns {Int32Array}
 */
export function stableArgsortDescending(values) {
	const n = values.length;
	const arr = new Array(n);
	for (let i = 0; i < n; i++) arr[i] = i;
	arr.sort((a, b) => {
		const va = values[a];
		const vb = values[b];
		if (Number.isNaN(va)) return Number.isNaN(vb) ? a - b : 1;
		if (Number.isNaN(vb)) return -1;
		return vb - va || a - b;
	});
	return Int32Array.from(arr);
}

// ---------------------------------------------------------------------------------------------
// S8 — directional attribution (temporal.py:534-540)
// ---------------------------------------------------------------------------------------------

/**
 * `a_{s,n} = α_{root→n,s} · 1{x_{s,n} ≠ x_{s,root}}`, `[L, N]` float32 (temporal.py:534-540).
 *
 * The indicator is 1 only where the taxon's residue is UNAMBIGUOUS (`< 20`) and differs from the
 * root: an unknown residue contributes 0 rather than counting as a difference. Under a consensus
 * root that makes every invariable site identically zero, which is why nothing downstream of here
 * needs the model's output at an invariable site.
 *
 * The product is float32 × {0, 1}, so it is exactly the attention or exactly zero; the result is
 * held in a Float32Array to keep the dtype the reference's `leaf_attributions` has.
 *
 * @param {ArrayLike<number>} meanAttns row-major L*N, float32 values, rows summing to 1 over taxa
 * @param {ArrayLike<number>} aValid row-major L*N amino-acid tokens, date-filtered
 * @param {ArrayLike<number>} rootIndices length L
 * @param {number} L
 * @param {number} N
 * @returns {Float32Array} row-major L*N
 */
export function directionalAttribution(meanAttns, aValid, rootIndices, L, N) {
	const out = new Float32Array(L * N);
	for (let s = 0; s < L; s++) {
		const ref = rootIndices[s];
		const o = s * N;
		for (let n = 0; n < N; n++) {
			const tok = aValid[o + n];
			out[o + n] = tok < TEMPORAL_UNKNOWN_AA && tok !== ref ? meanAttns[o + n] : 0;
		}
	}
	return out;
}

// ---------------------------------------------------------------------------------------------
// S9 — Nadaraya–Watson smoothing (temporal.py:542-549)
// ---------------------------------------------------------------------------------------------

/**
 * The normalised Gaussian kernel weights `[T, N]`, float64 (temporal.py:543-547).
 *
 * `dense_t` is float64 and `taxa_dates` float32, so numpy promotes the difference to float64 —
 * the float32 dates are read as the exact float64 values they represent, which is why `t_min` is
 * 2009.2490234375 rather than 2009.249 and why a port that keeps dates in float64 throughout misses
 * the time axis in the seventh digit.
 *
 * THE GUARD IS ON THE ROW SUM, not per element: `weights / (weights.sum(axis=1, keepdims=True) +
 * 1e-8)`. Rows sum to 1 afterwards (to within the guard), which is what makes a permutation of the
 * taxon columns still a valid normalised weight matrix — nothing is renormalised per draw.
 *
 * The row sums are accumulated with numpy's pairwise summation because that is what
 * `weights.sum(axis=1)` does on a C-contiguous last axis.
 *
 * MEASURED, and the reason every buffer downstream is float64: the smallest normalised weight the
 * acceptance run exports is 2.99e-41, three decades below float32's smallest normal (1.18e-38). A
 * Float32Array kernel matrix flushes it to zero and changes both the curves and every permuted
 * statistic.
 *
 * @param {ArrayLike<number>} denseT length T, float64
 * @param {ArrayLike<number>} taxaDates length N, FLOAT32 values
 * @param {number} bandwidth h > 0
 * @returns {Float64Array} row-major T*N
 */
export function nadarayaWatsonWeights(denseT, taxaDates, bandwidth) {
	const T = denseT.length;
	const N = taxaDates.length;
	const W = new Float64Array(T * N);
	const row = new Float64Array(N);
	for (let t = 0; t < T; t++) {
		const o = t * N;
		for (let n = 0; n < N; n++) {
			const d = (denseT[t] - taxaDates[n]) / bandwidth;
			row[n] = Math.exp(-0.5 * (d * d));
		}
		const s = numpyPairwiseSum(row, 0, N, f64) + 1e-8;
		for (let n = 0; n < N; n++) W[o + n] = row[n] / s;
	}
	return W;
}

/**
 * `norm_weights` transposed to taxon-major `[N, T]`.
 *
 * WHY THIS EXISTS AND MUST NOT BE "OPTIMISED AWAY". In the permutation null a drawn permutation
 * selects `W[t, p[n]]` for every t, which in taxon-major layout is a CONTIGUOUS row base
 * `p[n]·T` computed once per (candidate, taxon) pair and amortised over T fused multiply-adds. The
 * reference materialises a fresh `[T, N]` permuted matrix per draw (temporal.py:658, `N·T` float64
 * writes — 512 KB at the taxon cap); with this layout the permutation costs one `Int32Array(N)` of
 * row bases. Any future change that materialises a permuted weight matrix per draw is a regression.
 *
 * @param {ArrayLike<number>} W row-major T*N
 * @param {number} T
 * @param {number} N
 * @returns {Float64Array} row-major N*T
 */
export function taxonMajorWeights(W, T, N) {
	const WT = new Float64Array(N * T);
	for (let t = 0; t < T; t++) {
		const o = t * N;
		for (let n = 0; n < N; n++) WT[n * T + t] = W[o + n];
	}
	return WT;
}

/**
 * `attr @ norm_weights.T` — the smoothed trajectories `[rows, T]`, float64 (temporal.py:549).
 *
 * The attributions are float32 and the weights float64, so numpy promotes and the product is
 * float64; this routine accumulates in float64 throughout.
 *
 * Zero attributions are SKIPPED. That is bit-safe, not an approximation: for finite `x`,
 * `x + 0.0 = x` exactly, the accumulator starts at `+0` and every product is a non-negative weight
 * times a non-negative attribution, so `−0.0 + 0.0` — IEEE's one exception — cannot arise. It is
 * what makes the null affordable: MEASURED on the acceptance alignment, the mean derived fraction
 * is ρ = 0.022 (median 0.010, p90 0.040), so the attribution matrix is ~98 % zeros.
 *
 * Passing `baseRow` drives the same routine at a permuted taxon order (`baseRow[n] = p[n]·T`); the
 * default is the identity, which is why the reported curves and the null's `v_obs` come out of ONE
 * code path. Computing `v_obs` from a different summation order than `v_p` would bias the `>=`
 * count at the tie boundary, and degenerate constant curves put every draw exactly on that boundary.
 *
 * @param {ArrayLike<number>} attr row-major rows*N, float32 values
 * @param {number} rows
 * @param {number} N
 * @param {ArrayLike<number>} WT taxon-major weights, row-major N*T
 * @param {number} T
 * @param {{ baseRow?: ArrayLike<number>|null, out?: Float64Array }} [options]
 * @returns {Float64Array} row-major rows*T
 */
export function smoothTrajectories(attr, rows, N, WT, T, { baseRow = null, out = new Float64Array(rows * T) } = {}) {
	for (let r = 0; r < rows; r++) {
		const ao = r * N;
		const oo = r * T;
		out.fill(0, oo, oo + T);
		for (let n = 0; n < N; n++) {
			const v = attr[ao + n];
			if (v === 0) continue;
			const p0 = baseRow ? baseRow[n] : n * T;
			for (let t = 0; t < T; t++) out[oo + t] += v * WT[p0 + t];
		}
	}
	return out;
}

// ---------------------------------------------------------------------------------------------
// S10/S11 — the sweep metric, its peaks, areas and widths (temporal.py:551-597, 785)
// ---------------------------------------------------------------------------------------------

/**
 * `_metric` (temporal.py:563-566): the sweep metric on the smoothed prevalence curves.
 *
 *   fixation : `curves − curves[:, :1]` — the cumulative amplitude shift from the first grid point.
 *   episodic : `max(0, np.gradient(curves, grad_t, axis=1))` — the positive part of the derivative.
 *
 * `gradT` is the reference's `grad_t` and keys off UNITS, not mode (temporal.py:561): real calendar
 * time for a calendar run, the normalised [0, 1] axis for a non-calendar one. See
 * `numeric/calculus.js` for why the gradient's uniformity test and its one-sided edges are not
 * optional details.
 *
 * @param {ArrayLike<number>} curves row-major rows*T, float64
 * @param {number} rows
 * @param {number} T
 * @param {{ sweepMode: 'episodic'|'fixation', gradT?: ArrayLike<number>|null, out?: Float64Array }} options
 * @returns {Float64Array} row-major rows*T
 */
export function sweepMetric(curves, rows, T, { sweepMode, gradT = null, out = new Float64Array(rows * T) }) {
	if (sweepMode === 'fixation') {
		for (let r = 0; r < rows; r++) {
			const o = r * T;
			const base = curves[o];
			for (let t = 0; t < T; t++) out[o + t] = curves[o + t] - base;
		}
		return out;
	}
	if (!gradT) throw new TypeError('sweepMetric: the episodic branch needs gradT');
	numpyGradientRows(curves, rows, T, gradT, out);
	// np.maximum(0.0, ...) (temporal.py:566). `!(x > 0)` also sends NaN to 0, where numpy's maximum
	// PROPAGATES it. Unreachable on any finite model output — `smoothTrajectories` cannot produce a
	// NaN curve from finite attributions and finite Nadaraya-Watson weights, and `numpyGradientRows`
	// is finite differences on a strictly increasing axis — but it IS a divergence, recorded here
	// rather than absorbed, and it is the only one in this function. A NaN reaching this point would
	// be clipped to 0 and then carried silently into `peak_intensity`, `auc` and the fPCA gate
	// instead of announcing itself.
	for (let i = 0; i < rows * T; i++) if (!(out[i] > 0)) out[i] = 0;
	return out;
}

/**
 * Everything temporal.py:568-597 and :785 derive from the curves and the metric: the (possibly
 * rescaled) velocity matrix, the peak height, date, area, half-maximum window and width, and the
 * mean intensity.
 *
 * THE TWO BRANCHES DIFFER IN MORE THAN A SCALE FACTOR:
 *   fixation — `site_scale = mean_attns.mean(axis=1) + 1e-8` is a FLOAT32 pairwise mean plus a
 *     float32 add; the velocity is divided by it; `peak_intensity` is `ptp` of the PREVALENCE
 *     (not of the scaled velocity) over the same scale; `auc` integrates `max(0, velocity)` over
 *     the NORMALISED axis; and `peak_date` is the argmax of `|velocity|`.
 *   episodic — no rescaling at all (Q6), `peak_intensity` is the plain row max, `auc` integrates
 *     the velocity over `gradT`, and `peak_date` is the argmax of the velocity itself.
 *
 * `argmax` resolves ties to the FIRST index, so an all-zero row — every invariable site, and any
 * site with no derived residue — reports `peak_date = denseT[0]`. That is numerically correct and
 * is a result semantic the application must render as an em dash rather than as "peaked at the
 * start of the epidemic".
 *
 * `t_half_start`, `t_half_end` and `fwhm` are FLOAT32 fields (temporal.py:584-586), so the float64
 * grid dates are rounded on assignment — hence `Math.fround` at exactly those three stores and
 * nowhere else. The half-maximum window is the FIRST and LAST crossing over the whole axis, so a
 * bimodal trajectory yields a width spanning both humps rather than the peak's own width; and in
 * the fixation branch `pv` comes from `ptp(curves)/site_scale` while `y` is the scaled shift, so
 * the half-max test compares two differently constructed quantities. Both replicated.
 *
 * @param {{
 *   curves: ArrayLike<number>, metric: ArrayLike<number>, L: number, T: number,
 *   denseT: ArrayLike<number>, normDenseT: ArrayLike<number>, gradT: ArrayLike<number>,
 *   sweepMode: 'episodic'|'fixation', meanAttns?: ArrayLike<number>|null, N?: number
 * }} args `metric` is {@link sweepMetric}'s output; it is NOT modified (the fixation rescale is
 *   written into a fresh buffer).
 * @returns {{ velocity: Float64Array, peakIntensities: Float64Array, aucs: Float64Array,
 *   peakTimes: Float64Array, tHalfStart: Float32Array, tHalfEnd: Float32Array, fwhm: Float32Array,
 *   meanIntensity: Float64Array, siteScale: Float32Array|null }}
 */
export function temporalTrajectoryStatistics({ curves, metric, L, T, denseT, normDenseT, gradT, sweepMode, meanAttns = null, N = 0 }) {
	let velocity = Float64Array.from(metric);
	/** @type {Float32Array|null} */
	let siteScale = null;
	const peakIntensities = new Float64Array(L);
	const peakTimes = new Float64Array(L);
	let aucs;

	if (sweepMode === 'fixation') {
		if (!meanAttns) throw new TypeError('temporalTrajectoryStatistics: the fixation branch needs meanAttns');
		siteScale = new Float32Array(L);
		for (let s = 0; s < L; s++) siteScale[s] = Math.fround(numpyMeanFloat32(meanAttns, s * N, N) + 1e-8);
		for (let s = 0; s < L; s++) {
			const o = s * T;
			const sc = siteScale[s];
			for (let t = 0; t < T; t++) velocity[o + t] = velocity[o + t] / sc;
			let lo = curves[o];
			let hi = curves[o];
			for (let t = 1; t < T; t++) {
				const v = curves[o + t];
				if (v < lo) lo = v;
				if (v > hi) hi = v;
			}
			peakIntensities[s] = (hi - lo) / sc;
			let best = -1;
			let bestIdx = 0;
			for (let t = 0; t < T; t++) {
				const m = Math.abs(velocity[o + t]);
				if (m > best) { best = m; bestIdx = t; }
			}
			peakTimes[s] = denseT[bestIdx];
		}
		const positive = new Float64Array(L * T);
		for (let i = 0; i < L * T; i++) positive[i] = velocity[i] > 0 ? velocity[i] : 0;
		aucs = numpyTrapezoidRows(positive, L, T, normDenseT);
	} else {
		for (let s = 0; s < L; s++) {
			const o = s * T;
			let best = velocity[o];
			let bestIdx = 0;
			for (let t = 1; t < T; t++) {
				if (velocity[o + t] > best) { best = velocity[o + t]; bestIdx = t; }
			}
			peakIntensities[s] = best;
			peakTimes[s] = denseT[bestIdx];
		}
		aucs = numpyTrapezoidRows(velocity, L, T, gradT);
	}

	const tHalfStart = new Float32Array(L);
	const tHalfEnd = new Float32Array(L);
	const fwhm = new Float32Array(L);
	for (let s = 0; s < L; s++) {
		const pv = peakIntensities[s];
		if (!(pv > 1e-7)) continue;
		const half = pv / 2.0;
		const o = s * T;
		let first = -1;
		let last = -1;
		for (let t = 0; t < T; t++) {
			if (velocity[o + t] >= half) {
				if (first < 0) first = t;
				last = t;
			}
		}
		if (first < 0) continue;
		tHalfStart[s] = Math.fround(denseT[first]);
		tHalfEnd[s] = Math.fround(denseT[last]);
		fwhm[s] = Math.fround(denseT[last] - denseT[first]);
	}

	const meanIntensity = new Float64Array(L);
	for (let s = 0; s < L; s++) meanIntensity[s] = numpyMeanFloat64(velocity, s * T, T);

	return { velocity, peakIntensities, aucs, peakTimes, tHalfStart, tHalfEnd, fwhm, meanIntensity, siteScale };
}

// ---------------------------------------------------------------------------------------------
// S12 — the stage-one energy floors (temporal.py:599-616)
// ---------------------------------------------------------------------------------------------

/**
 * The two stage-one floors (temporal.py:599-613).
 *
 * Q1, replicated exactly: the `tau_peak` override tests the VALUE (`tau_peak is None or tau_peak ==
 * 1e-4`), not whether the caller supplied one, so passing the documented default explicitly is
 * silently overridden. `tauPeakOverridden` says when that happened so an option layer can warn; the
 * reference has no such signal.
 *
 * UNITS. The calendar floors and `tau_auc`'s `timespan/5.0` are quantities PER DECIMAL YEAR; the
 * non-calendar floors are dimensionless only because `grad_t` is the normalised axis there. Letting
 * a reader label a `days` column as `years` therefore changes the candidate set silently, which is
 * why a surface must echo the resolved units, timespan, bandwidth and both floors beside the count.
 *
 * @param {{ timespan: number, nonCalendar: boolean, tauPeak?: number|null, tauAuc?: number|null }} args
 * @returns {{ tauPeak: number, tauAuc: number, tauPeakOverridden: boolean, tauAucDefaulted: boolean }}
 */
export function resolveEnergyFloors({ timespan, nonCalendar, tauPeak = 1e-4, tauAuc = null }) {
	const tauAucDefaulted = tauAuc === null || tauAuc === undefined;
	const tauPeakOverridden = tauPeak === null || tauPeak === undefined || tauPeak === 1e-4;
	let peak = tauPeak;
	let auc = tauAuc;
	if (nonCalendar) {
		if (tauAucDefaulted) auc = 0.005;
		if (tauPeakOverridden) peak = 0.010;
	} else {
		if (tauAucDefaulted) auc = 0.01e-3 * Math.max(1.0, timespan / 5.0);
		if (tauPeakOverridden) peak = 0.5e-4;
	}
	return { tauPeak: /** @type {number} */ (peak), tauAuc: /** @type {number} */ (auc), tauPeakOverridden, tauAucDefaulted };
}

/**
 * `stage1_mask = (~inv) & (peak_intensities >= tau_peak) & (aucs >= tau_auc)` (temporal.py:613-614).
 *
 * @param {ArrayLike<number|boolean>} inv length L, the AA-level invariability mask over ALL taxa
 * @param {ArrayLike<number>} peakIntensities length L
 * @param {ArrayLike<number>} aucs length L
 * @param {number} tauPeak
 * @param {number} tauAuc
 * @returns {{ mask: Uint8Array, candIndices: Int32Array }}
 */
export function stageOneMask(inv, peakIntensities, aucs, tauPeak, tauAuc) {
	const L = peakIntensities.length;
	const mask = new Uint8Array(L);
	/** @type {number[]} */
	const cand = [];
	for (let s = 0; s < L; s++) {
		if (!inv[s] && peakIntensities[s] >= tauPeak && aucs[s] >= tauAuc) {
			mask[s] = 1;
			cand.push(s);
		}
	}
	return { mask, candIndices: Int32Array.from(cand) };
}

// ---------------------------------------------------------------------------------------------
// S13 — the date-shuffling null (temporal.py:618-666)
// ---------------------------------------------------------------------------------------------

/**
 * `_perm_stat` (temporal.py:624-642), on the SMOOTHED PREVALENCE curves — not on the metric.
 *
 *   fixation : signed Pearson correlation with normalised time,
 *              `Σ(c − c̄)·t̃ / (sqrt(Σ(c − c̄)²·Σt̃²) + 1e-12)`.
 *   episodic : `np.var(max(0, gradient(curves, grad_t, axis=1)), axis=1)` with `ddof = 0` —
 *              the full gradient, edge rules and all, re-run on every draw.
 *
 * The centring in the fixation numerator is algebraically redundant (`Σt̃ = 0`) but is NOT redundant
 * in floating point, and it costs nothing; it is evaluated as the reference writes it.
 *
 * @param {ArrayLike<number>} curves row-major rows*T, float64
 * @param {number} rows
 * @param {number} T
 * @param {{ sweepMode: 'episodic'|'fixation', gradT?: ArrayLike<number>|null,
 *   normDenseT?: ArrayLike<number>|null, scratch?: Float64Array|null, devScratch?: Float64Array|null,
 *   out?: Float64Array }} options
 * @returns {Float64Array} length rows
 */
export function temporalPermStat(curves, rows, T, { sweepMode, gradT = null, normDenseT = null, scratch = null, devScratch = null, out = new Float64Array(rows) }) {
	if (sweepMode === 'fixation') {
		if (!normDenseT) throw new TypeError('temporalPermStat: the fixation branch needs normDenseT');
		// `scratch` and `devScratch` are the EPISODIC branch's buffers; this branch takes neither and
		// forms four `Float64Array(T)` of its own per CALL — so per draw when the null drives it, not
		// per row. MEASURED (node 22.22.0, darwin/x64, C = 246, T = 60, 1,000 calls, best of five,
		// three processes): 453.4 / 463.0 / 503.6 ms as written against 439.0 / 435.6 / 456.8 ms with
		// all four hoisted into caller-owned buffers, bit-identical output — 3 %, 6 % and 9 % of this
		// branch. Left as it is on purpose: four T-length arrays per draw is a different order of
		// allocation from the episodic branch's one per row per draw, and a few per cent of one
		// regime's statistic does not pay for two more scratch options on a public pure function.
		// Re-measure before adding them.
		const tt = new Float64Array(T);
		const mt = numpyMeanFloat64(normDenseT, 0, T);
		for (let t = 0; t < T; t++) tt[t] = normDenseT[t] - mt;
		const sq = new Float64Array(T);
		for (let t = 0; t < T; t++) sq[t] = tt[t] * tt[t];
		const sumTT2 = numpyPairwiseSum(sq, 0, T, f64);
		const num = new Float64Array(T);
		const den = new Float64Array(T);
		for (let r = 0; r < rows; r++) {
			const o = r * T;
			const m = numpyMeanFloat64(curves, o, T);
			for (let t = 0; t < T; t++) {
				const cc = curves[o + t] - m;
				num[t] = cc * tt[t];
				den[t] = cc * cc;
			}
			const n = numpyPairwiseSum(num, 0, T, f64);
			const d = Math.sqrt(numpyPairwiseSum(den, 0, T, f64) * sumTT2) + 1e-12;
			out[r] = n / d;
		}
		return out;
	}
	if (!gradT) throw new TypeError('temporalPermStat: the episodic branch needs gradT');
	// The gradient is taken over the WHOLE matrix in one call rather than row by row, so no row is
	// ever copied into a scratch: the null calls this on every draw, and a per-row copy plus a
	// subarray there measured as a third of the kernel's time. The arithmetic is identical —
	// numpyGradientRows builds its coefficients once from gradT and applies them per row.
	// `scratch` must be a Float64Array of at least rows*T if it is supplied; `devScratch` is the same
	// bargain one level down: `np.var`'s second pass needs the squared deviations AS AN ARRAY (numpy
	// reduces them pairwise, so the forming loop cannot be fused into a running sum), and without a
	// caller-owned buffer that is one Float64Array(T) per ROW — 246,000 of them at the acceptance
	// run's shape (C = 246, N = 95, T = 60) over the reference's default B = 1000 draws
	// (temporal.py:406). Both guards test `instanceof Float64Array` and not merely a length:
	// a Float32Array of the right length is an ArrayLike that would silently carry the gradient, or
	// every squared deviation, in float32. Anything else is IGNORED and the buffer allocated — the
	// number is then the same number, one allocation slower — rather than throwing on a path that
	// runs once per row per draw.
	// WHAT THE DEVIATION BUFFER BUYS, re-measured at round two (node 22.22.0, darwin/x64, C = 246,
	// N = 95, T = 60, B = 1000, best of five inside a process, three processes on a machine that was
	// not idle): on the whole null, NOTHING outside the noise — 351.4 / 363.0 / 369.6 ms with it
	// against 354.5 / 357.9 / 366.0 ms without, ahead in one of the three. Timing `numpyVarFloat64`
	// alone does show 2.6x-4.1x (see that function's header), but V8 escape-analyses the allocation
	// away once this kernel is around it. The buffers stay because they remove 246,000 allocations
	// per null and cannot change an output bit — verified: every float64 bit of `out` and every one
	// of the 246 exceedance counts is identical with and without them — not because they are faster.
	const grad = scratch instanceof Float64Array && scratch.length >= rows * T ? scratch : new Float64Array(rows * T);
	const dev = devScratch instanceof Float64Array && devScratch.length >= T ? devScratch : new Float64Array(T);
	numpyGradientRows(curves, rows, T, gradT, /** @type {Float64Array} */ (grad));
	for (let r = 0; r < rows; r++) {
		const o = r * T;
		// np.maximum(0.0, ...) (temporal.py:641). `!(x > 0)` also sends NaN to 0, where numpy's
		// maximum PROPAGATES it; the branch is unreachable here (the gradient of a finite smoothed
		// curve over a strictly increasing axis is finite, and `nadarayaWatsonWeights` cannot emit a
		// NaN weight from finite dates), and a NaN that did arrive would otherwise poison `np.var`
		// and hence the `>=` exceedance test for the whole row. Deliberate, and the only divergence
		// in this function.
		for (let t = o; t < o + T; t++) if (!(grad[t] > 0)) grad[t] = 0;
		out[r] = numpyVarFloat64(grad, o, T, dev);
	}
	return out;
}

/**
 * One date-shuffling permutation of `0 … N−1`, from a per-draw substream.
 *
 * PER-DRAW SUBSTREAMS, not one advancing stream: draw `b` is seeded by `seed0 + b·2^64/φ` through
 * `Xoshiro256`'s own splitmix64 expansion, so a run stopped at 500 draws is bit-identical to a run
 * configured with B = 500, a resumed run continues exactly where it left off, and a future split of
 * draws across workers changes nothing. A single stream would be deterministic only while the draws
 * stay in order in one process.
 *
 * This CANNOT reproduce `np.random.RandomState(42).permutation(N)` (temporal.py:651, MT19937) and is
 * not meant to: the library standardises on xoshiro256** by design (D17), so `p_perm`, `q_perm` and
 * every label thresholded on them are comparable to the reference in STATISTICAL class only, while
 * everything upstream of the shuffle stays at the strict graph class.
 *
 * @param {number} N
 * @param {number} seed0
 * @param {number} drawIndex
 * @returns {Int32Array} a permutation of 0 … N−1
 */
export function temporalDrawPermutation(N, seed0, drawIndex) {
	const rng = new Xoshiro256(BigInt.asUintN(64, BigInt(seed0) + BigInt(drawIndex) * 0x9e3779b97f4a7c15n));
	const p = new Int32Array(N);
	for (let i = 0; i < N; i++) p[i] = i;
	rng.shuffle(p);
	return p;
}

/**
 * Draws `[fromDraw, toDraw)` of the date-shuffling null, accumulating exceedances into `exceed`
 * (temporal.py:648-661).
 *
 * PURE AND RESUMABLE, which is what lets the application own the worker, the chunking, the progress
 * cadence and the cancel without any of that leaking in here: call it again with a later draw range
 * and the same `exceed` array and the result is identical to one long call. A PARTIALLY completed
 * draw must never be counted — the range boundary is the only place a caller may stop — which is
 * what makes the answer independent of where a cancel landed.
 *
 * The kernel is the one described in {@link smoothTrajectories}: a draw's permutation is turned into
 * an `Int32Array(N)` of contiguous row bases into the taxon-major weight matrix, rebuilt per draw at
 * cost N and amortised over T fused multiply-adds per (candidate, taxon) pair.
 *
 * WHAT IS ALLOCATED PER DRAW, and what is not — stated exactly, because an earlier revision of this
 * block claimed "nothing is allocated inside the draw loop" and that was never true. Every buffer
 * THIS function writes into is allocated ONCE above the loop: the curves, the statistic, the row
 * bases, the gradient scratch, the deviation scratch {@link temporalPermStat} needs for `np.var`,
 * and the CSR over the candidate attributions. THREE things are still allocated on every draw:
 *   - the `Int32Array(N)` {@link temporalDrawPermutation} returns;
 *   - the `Xoshiro256` its per-draw substream seeding requires (whose splitmix64 expansion also
 *     makes four BigInts);
 *   - the coefficient arrays the statistic's own kernel builds per call — three `Float64Array(T−2)`
 *     inside `numpyGradientRows`'s non-uniform branch, which is the branch a linspace time axis
 *     takes, in the episodic regime; or the four `Float64Array(T)` the fixation branch of
 *     {@link temporalPermStat} forms, measured in that branch's own comment.
 *
 * MEASURED before leaving them there (node 22.22.0, darwin/x64, C = 246, N = 95, T = 60, B = 1000,
 * episodic; best of five inside a process, four processes on a machine that was NOT idle, so the
 * absolutes spread by up to 2.6x and it is the pairing WITHIN each process that carries the signal):
 *   - the 1,000 permutations are 19.8-51.5 ms against a whole null of 193.6-390.6 ms, 9-14 % of it.
 *     Writing them into a hoisted `Int32Array` instead is 19.5-51.1 ms — ahead in all four
 *     processes, but by 0.2-1.1 ms across all 1,000 draws, which is 0.1-0.4 % of the null and well
 *     inside its own run-to-run spread. Re-seeding ONE hoisted `Xoshiro256` in place on top of that
 *     is SLOWER in all four (21.3-54.7 ms), because splitmix64 on BigInts costs more than the object
 *     allocation it avoids. All three variants give identical permutations and identical exceedance
 *     counts over the 246 candidates.
 *   - hoisting `numpyGradientRows`'s three coefficient arrays out of the call is SLOWER too, and not
 *     marginally: 1,000 calls at this shape are 48.9 / 54.6 / 54.8 ms as the function stands against
 *     77.0 / 85.6 / 86.1 ms reading hoisted module-scope copies, bit-identical output. Rebuilding
 *     58-element locals beats loading them from outside the function.
 * So nothing here is hoisted: the measurement says there is nothing to win, and an out-parameter on
 * {@link temporalDrawPermutation} would make a pure function stateful to buy it.
 *
 * @param {{
 *   candAttrs: ArrayLike<number>, C: number, N: number, T: number,
 *   WT: ArrayLike<number>, vObs: ArrayLike<number>,
 *   sweepMode: 'episodic'|'fixation', gradT?: ArrayLike<number>|null, normDenseT?: ArrayLike<number>|null,
 *   seed?: number, fromDraw?: number, toDraw: number, exceed?: Int32Array|null,
 *   onProgress?: ((p: {phase: string, done: number, total: number}) => void)|null
 * }} args
 * @returns {Int32Array} `exceed`, one count per candidate
 */
export function temporalNullDraws({ candAttrs, C, N, T, WT, vObs, sweepMode, gradT = null, normDenseT = null, seed = 42, fromDraw = 0, toDraw, exceed = null, onProgress = null }) {
	const counts = exceed ?? new Int32Array(C);
	const curves = new Float64Array(C * T);
	const stat = new Float64Array(C);
	const baseRow = new Int32Array(N);
	const scratch = new Float64Array(C * T);
	const devScratch = new Float64Array(T);
	// CSR over the candidate attributions, built ONCE. MEASURED on the acceptance alignment: only
	// 4.3 % of the [C, N] matrix is nonzero, because `delta_root` is an indicator of carrying a
	// derived residue, so a dense scan spends 95 loads and branches per candidate to do four
	// multiply-accumulate chains. The row order is preserved exactly, so the sum is the same sum in
	// the same order as {@link smoothTrajectories}'s and `v_obs` stays comparable to `v_p` bit for
	// bit at the `>=` tie boundary.
	const rowPtr = new Int32Array(C + 1);
	for (let c = 0; c < C; c++) {
		let k = 0;
		for (let n = 0; n < N; n++) if (candAttrs[c * N + n] !== 0) k++;
		rowPtr[c + 1] = rowPtr[c] + k;
	}
	const nnz = rowPtr[C];
	const colIdx = new Int32Array(nnz);
	const val = new Float64Array(nnz);
	{
		let k = 0;
		for (let c = 0; c < C; c++) {
			for (let n = 0; n < N; n++) {
				const v = candAttrs[c * N + n];
				if (v !== 0) {
					colIdx[k] = n;
					val[k] = v;
					k++;
				}
			}
		}
	}
	for (let b = fromDraw; b < toDraw; b++) {
		const perm = temporalDrawPermutation(N, seed, b);
		for (let n = 0; n < N; n++) baseRow[n] = perm[n] * T;
		curves.fill(0);
		for (let c = 0; c < C; c++) {
			const oo = c * T;
			for (let k = rowPtr[c]; k < rowPtr[c + 1]; k++) {
				const v = val[k];
				const p0 = baseRow[colIdx[k]];
				for (let t = 0; t < T; t++) curves[oo + t] += v * WT[p0 + t];
			}
		}
		temporalPermStat(curves, C, T, { sweepMode, gradT, normDenseT, scratch, devScratch, out: stat });
		for (let c = 0; c < C; c++) if (stat[c] >= vObs[c]) counts[c]++;
		if (onProgress) onProgress({ phase: 'temporal-null', done: b + 1 - fromDraw, total: toDraw - fromDraw });
	}
	return counts;
}

/**
 * `p_cand = (1 + exceed) / (B + 1)` and its BH q-values over the CANDIDATES ONLY (temporal.py:663-665).
 *
 * BH's denominator here is C, not L — and that is why `q_perm` is uninformative at every feasible B:
 * the smallest reachable q is `C/(B+1)`, which on the acceptance run is 246/101 = 2.4, clipped to 1,
 * so every confirmed sweep there reports the same `q_perm = 0.4298`. Reaching `q ≤ 0.10` at all
 * would need `B + 1 ≥ 10·C` — about 2,500 draws for that run, and the reference's own default of
 * 1,000 does not get there either. The sweep call is made on p (temporal.py:687-689); a surface must
 * not print `q_perm` as though it were a decision threshold. Upstream observation, not a fix.
 *
 * `drawsCompleted` is the reference's `B` on a complete run and the achieved count on a stopped one;
 * the estimator is the same either way, just on a coarser grid of `1/(b+1)`.
 *
 * @param {ArrayLike<number>} exceed length C
 * @param {number} drawsCompleted
 * @returns {{ p: Float64Array, q: Float64Array, gridStep: number }}
 */
export function temporalPermPValues(exceed, drawsCompleted) {
	const C = exceed.length;
	const p = new Float64Array(C);
	for (let c = 0; c < C; c++) p[c] = (1.0 + exceed[c]) / (drawsCompleted + 1.0);
	return { p, q: benjaminiHochberg(p), gridStep: 1.0 / (drawsCompleted + 1.0) };
}

// ---------------------------------------------------------------------------------------------
// S14 — the fPCA shape gate (temporal.py:667-679)
// ---------------------------------------------------------------------------------------------

/**
 * Row standardisation `(x − rowmean) / (rowstd + 1e-8)` in float64 — the transform both
 * decompositions start from (temporal.py:669-671, 728-730, 737-739).
 *
 * The `+1e-8` is Q7's bug in miniature: a constant row has std 0, so it becomes a row of exact
 * zeros rather than an error, and every quantity derived from it is 0 — which the gate then reads
 * as a perfect fit.
 *
 * @param {ArrayLike<number>} M row-major rows*T
 * @param {number} rows
 * @param {number} T
 * @returns {Float64Array} row-major rows*T
 */
export function rowStandardise(M, rows, T) {
	const Z = new Float64Array(rows * T);
	for (let r = 0; r < rows; r++) {
		const o = r * T;
		const m = numpyMeanFloat64(M, o, T);
		const sd = numpyStdFloat64(M, o, T) + 1e-8;
		for (let t = 0; t < T; t++) Z[o + t] = (M[o + t] - m) / sd;
	}
	return Z;
}

/**
 * `r2_fpca` over the candidates (temporal.py:667-679): how much of each candidate's standardised
 * VELOCITY trajectory the top `min(4, C)` collective shapes reconstruct.
 *
 * Note the deliberate asymmetry in the reference, preserved here: the null's `v_obs` is computed on
 * the PREVALENCE curves while this gate runs on the METRIC (velocity) rows.
 *
 * The residual is taken from the projector rather than from a reconstructed matrix (see
 * `numeric/svd.js`), so the gate is invariant to both the sign and any rotation of the basis and
 * therefore sits at the strict graph class with no convention involved.
 *
 * Two upstream degeneracies come out for free and are reported rather than hidden: Q8, the gate is
 * vacuous at C = 4 because mean-centred rows have rank ≤ 3 and the top-4 subspace contains the whole
 * row space; and Q7, a flat candidate standardises to zeros and scores 1.
 *
 * @param {ArrayLike<number>} candMetric row-major C*T, the velocity rows at the candidate sites
 * @param {number} C
 * @param {number} T
 * @param {{ kMax?: number, waveSign?: 'canonical'|'lapack' }} [options]
 * @returns {{ r2: Float64Array, kEff: number, modes: ReturnType<typeof dominantTimeModes>,
 *   vacuous: boolean, flatRows: number[] }}
 */
export function fpcaShapeGate(candMetric, C, T, { kMax = TEMPORAL_WAVE_K, waveSign = 'canonical' } = {}) {
	const Z = rowStandardise(candMetric, C, T);
	const kEff = Math.min(kMax, C);
	const modes = dominantTimeModes(Z, C, T, kEff, { waveSign });
	const { sse, sst } = projectionResidual(Z, C, T, modes.vectors, modes.kept);
	const r2 = new Float64Array(C);
	/** @type {number[]} */
	const flatRows = [];
	for (let c = 0; c < C; c++) {
		const v = 1.0 - sse[c] / (sst[c] + 1e-8);
		r2[c] = v < 0 ? 0 : v > 1 ? 1 : v;
		if (sst[c] === 0) flatRows.push(c);
	}
	// Q8: mean-centred rows span at most min(C, T − 1) dimensions, so a top-k subspace with
	// k >= that rank reconstructs everything and the gate discriminates nothing.
	const vacuous = C > 0 && kEff >= Math.min(C, T - 1);
	return { r2, kEff, modes, vacuous, flatRows };
}

// ---------------------------------------------------------------------------------------------
// S15/S16 — confirmation, the escape hatch and the four-way classification (temporal.py:681-716)
// ---------------------------------------------------------------------------------------------

/**
 * `is_confirmed_sweep` and the escape hatch (temporal.py:681-697).
 *
 * The solitary regime — three or fewer candidates, ANY non-calendar run, or `min_r2_fpca <= 0` —
 * drops the shape gate entirely, because a handful of asynchronous sweeps cannot form a collective
 * multi-wave signature. The escape hatch is calendar-only, fires on the COUNT being zero, uses a
 * hard-coded 0.10 and the hard-coded χ²₁ cut 3.84 against the FLOAT32 LRT, and is recorded in no
 * reference output file (Q5) — `escapeHatchUsed` is this port's addition so a surface can say "no
 * site cleared the threshold; the sites below are the fallback selection".
 *
 * @param {{
 *   stage1Mask: ArrayLike<number|boolean>, pPerm: ArrayLike<number>, r2Fpca: ArrayLike<number>,
 *   lrts: ArrayLike<number>, L: number, nStage1: number, nonCalendar: boolean,
 *   permAlpha?: number, minR2Fpca?: number
 * }} args `pPerm` and `r2Fpca` are the full length-L arrays (1.0 and 0.0 outside the candidates).
 * @returns {{ isConfirmedSweep: Uint8Array, solitaryRegime: boolean, escapeHatchUsed: boolean, nSweeps: number }}
 */
export function confirmSweeps({ stage1Mask, pPerm, r2Fpca, lrts, L, nStage1, nonCalendar, permAlpha = 0.05, minR2Fpca = 0.35 }) {
	const solitaryRegime = nStage1 <= 3 || nonCalendar || minR2Fpca <= 0.0;
	const out = new Uint8Array(L);
	let n = 0;
	for (let s = 0; s < L; s++) {
		let ok = Boolean(stage1Mask[s]) && pPerm[s] <= permAlpha;
		if (ok && !solitaryRegime) ok = r2Fpca[s] >= minR2Fpca;
		if (ok) { out[s] = 1; n++; }
	}
	let escapeHatchUsed = false;
	if (n === 0 && nStage1 > 0 && !solitaryRegime) {
		escapeHatchUsed = true;
		n = 0;
		for (let s = 0; s < L; s++) {
			const ok = Boolean(stage1Mask[s]) && (pPerm[s] <= TEMPORAL_ESCAPE_P || lrts[s] >= TEMPORAL_ESCAPE_LRT);
			out[s] = ok ? 1 : 0;
			if (ok) n++;
		}
	}
	return { isConfirmedSweep: out, solitaryRegime, escapeHatchUsed, nSweeps: n };
}

/**
 * The four-way `classification` and `cross_classification` ladder (temporal.py:699-716), in the
 * reference's own order — the order is the semantics, because the branches overlap.
 *
 * An INVARIABLE site is never `FILTERED_STATIC_NOISE` even when its static evidence would qualify,
 * because its `q_static` was forced to exactly 1 when BH ran over the variable subset only.
 *
 * `is_concordant_sweep` and `is_rescued_sweep` are DERIVED from the cross strings (temporal.py:772-773),
 * not recomputed, so they are returned from here rather than from a second test.
 *
 * @param {{ inv: ArrayLike<number|boolean>, stage1Mask: ArrayLike<number|boolean>,
 *   isConfirmedSweep: ArrayLike<number|boolean>, qStatic: ArrayLike<number>, L: number }} args
 * @returns {{ classification: string[], crossClassification: string[], isConcordant: Uint8Array,
 *   isRescued: Uint8Array, counts: {confirmed: number, concordant: number, rescued: number, filteredStaticNoise: number} }}
 */
export function classifyTemporalSites({ inv, stage1Mask, isConfirmedSweep, qStatic, L }) {
	/** @type {string[]} */
	const classification = [];
	/** @type {string[]} */
	const crossClassification = [];
	const isConcordant = new Uint8Array(L);
	const isRescued = new Uint8Array(L);
	let confirmed = 0;
	let concordant = 0;
	let rescued = 0;
	let filteredStaticNoise = 0;
	for (let s = 0; s < L; s++) {
		let c;
		let x;
		if (inv[s]) {
			c = TEMPORAL_CLASSES.INVARIABLE;
			x = TEMPORAL_CROSS_CLASSES.NEGATIVE_CONSENSUS;
		} else if (!stage1Mask[s]) {
			c = TEMPORAL_CLASSES.FLAT_NO_SIGNAL;
			x = qStatic[s] <= TEMPORAL_Q_STATIC_CUT ? TEMPORAL_CROSS_CLASSES.FILTERED_STATIC_NOISE : TEMPORAL_CROSS_CLASSES.NEGATIVE_CONSENSUS;
		} else if (!isConfirmedSweep[s]) {
			c = TEMPORAL_CLASSES.TEMPORAL_NOISE;
			x = qStatic[s] <= TEMPORAL_Q_STATIC_CUT ? TEMPORAL_CROSS_CLASSES.FILTERED_STATIC_NOISE : TEMPORAL_CROSS_CLASSES.NEGATIVE_CONSENSUS;
		} else {
			c = TEMPORAL_CLASSES.CONFIRMED_SWEEP;
			x = qStatic[s] <= TEMPORAL_Q_STATIC_CUT ? TEMPORAL_CROSS_CLASSES.CONCORDANT_SWEEP : TEMPORAL_CROSS_CLASSES.RESCUED_SWEEP;
			confirmed++;
		}
		classification.push(c);
		crossClassification.push(x);
		if (x === TEMPORAL_CROSS_CLASSES.CONCORDANT_SWEEP) { isConcordant[s] = 1; concordant++; }
		if (x === TEMPORAL_CROSS_CLASSES.RESCUED_SWEEP) { isRescued[s] = 1; rescued++; }
		if (x === TEMPORAL_CROSS_CLASSES.FILTERED_STATIC_NOISE) filteredStaticNoise++;
	}
	return { classification, crossClassification, isConcordant, isRescued, counts: { confirmed, concordant, rescued, filteredStaticNoise } };
}

// ---------------------------------------------------------------------------------------------
// S17 — the collective wave decomposition (temporal.py:718-741)
// ---------------------------------------------------------------------------------------------

/**
 * The four collective wave modes, their variance shares and every site's loading on them
 * (temporal.py:718-741).
 *
 * The row set is the confirmed sweeps when there are at least four of them, and otherwise the
 * `max(4, C)` sites with the largest peak intensity over ALL L sites — a stable descending sort here
 * (Q4), because 4111 of the acceptance run's peak intensities are tied at exactly zero. The
 * reference's third branch (`arange(min(L, 4))`) is unreachable, since `max(4, C) ≥ 4` always;
 * it is kept for the `L < 4` shape.
 *
 * `var_explained`'s denominator is the FULL spectrum, so the four reported shares need not sum to
 * 100 (the acceptance run's four sum to 95.27 %).
 *
 * `waves` is returned zero-padded to four rows the way `_waves.csv` writes it, with `nWaves` saying
 * how many are real. `loadings` is float32 (temporal.py:736), so each loading is the float64 dot
 * product downcast on store.
 *
 * SIGNS. Under the default `waveSign: 'canonical'` (D28) each retained `v_j` is flipped so its
 * largest-magnitude entry is positive BEFORE the loadings are computed, so a wave and its loading
 * column flip together and "this site moves with the curve you are looking at" stays true. The
 * reference has no convention and writes `gesdd`'s signs; on the acceptance run three of its four
 * waves come out inverted relative to this rule. Where two singular values are close no sign rule
 * helps at all — the pair can rotate — which is what `gaps` and `nearDegenerate` are for.
 *
 * @param {{
 *   velocity: ArrayLike<number>, L: number, T: number,
 *   isConfirmedSweep: ArrayLike<number|boolean>, peakIntensities: ArrayLike<number>,
 *   candIndices: ArrayLike<number>, K?: number, waveSign?: 'canonical'|'lapack'
 * }} args
 * @returns {{ fIndices: Int32Array, waves: Float64Array, nWaves: number, varExplained: Float64Array,
 *   sigma: Float64Array, gaps: Float64Array, nearDegenerate: boolean[], rankDeficient: boolean[],
 *   loadings: Float32Array, modes: ReturnType<typeof dominantTimeModes>, waveSign: 'canonical'|'lapack' }}
 *   `waves` is row-major K*T, zero-padded; `varExplained` is the first K shares (zeros when the
 *   spectrum has no energy, as temporal.py:733 writes them); `loadings` is row-major L*K.
 */
export function temporalWaveDecomposition({ velocity, L, T, isConfirmedSweep, peakIntensities, candIndices, K = TEMPORAL_WAVE_K, waveSign = 'canonical' }) {
	/** @type {number[]} */
	const sweeps = [];
	for (let s = 0; s < L; s++) if (isConfirmedSweep[s]) sweeps.push(s);
	/** @type {Int32Array} */
	let fIndices;
	if (sweeps.length < K) {
		const order = stableArgsortDescending(peakIntensities);
		const take = Math.max(K, candIndices.length);
		const top = order.slice(0, take);
		if (top.length >= K) {
			fIndices = top;
		} else {
			const m = Math.min(L, K);
			fIndices = new Int32Array(m);
			for (let i = 0; i < m; i++) fIndices[i] = i;
		}
	} else {
		fIndices = Int32Array.from(sweeps);
	}

	const F = fIndices.length;
	const Y = new Float64Array(F * T);
	for (let i = 0; i < F; i++) {
		const src = fIndices[i] * T;
		for (let t = 0; t < T; t++) Y[i * T + t] = velocity[src + t];
	}
	const Z = rowStandardise(Y, F, T);
	const modes = dominantTimeModes(Z, F, T, K, { waveSign });

	const nWaves = modes.kept;
	const waves = new Float64Array(K * T);
	waves.set(modes.vectors.subarray(0, nWaves * T), 0);

	const varExplained = new Float64Array(K);
	for (let j = 0; j < K && j < modes.varExplained.length; j++) varExplained[j] = modes.varExplained[j];

	const Zall = rowStandardise(velocity, L, T);
	const loadings = new Float32Array(L * K);
	for (let j = 0; j < Math.min(K, nWaves); j++) {
		const vo = j * T;
		for (let s = 0; s < L; s++) {
			const o = s * T;
			let dot = 0;
			for (let t = 0; t < T; t++) dot += Zall[o + t] * waves[vo + t];
			loadings[s * K + j] = Math.fround(dot);
		}
	}

	return {
		fIndices,
		waves,
		nWaves,
		varExplained,
		sigma: modes.sigma,
		gaps: modes.gaps,
		nearDegenerate: modes.nearDegenerate,
		rankDeficient: modes.rankDeficient,
		loadings,
		modes,
		waveSign: modes.waveSign
	};
}

// ---------------------------------------------------------------------------------------------
// S18 — mutation labels (temporal.py:743-760)
// ---------------------------------------------------------------------------------------------

/**
 * The per-site `derived_aa`, `mutation_label` and `domain` columns (temporal.py:743-760).
 *
 * The derived residue is the modal token among the date-filtered residues that are valid AND differ
 * from the root; failing that, the modal valid residue overall; failing that, the root residue
 * itself. Same `bincount` + `argmax` lowest-index tie-break as the root consensus. At an invariable
 * site the derived residue equals the reference residue, giving labels like `M1M` — replicated,
 * because the reference writes them.
 *
 * Site numbering in the label is 1-based over the ALIGNMENT's codons, which on a concatenated genome
 * is not the gene's own numbering; that is a sentence for the application, not a change here.
 *
 * @param {{ aValid: ArrayLike<number>, L: number, N: number, rootIndices: ArrayLike<number>,
 *   rootAas: ArrayLike<string>, domainMap?: Map<number,string>|Record<number,string>|null }} args
 * @returns {{ derivedAas: string[], mutationLabels: string[], domains: string[] }}
 */
export function temporalMutationLabels({ aValid, L, N, rootIndices, rootAas, domainMap = null }) {
	/** @type {string[]} */
	const derivedAas = [];
	/** @type {string[]} */
	const mutationLabels = [];
	/** @type {string[]} */
	const domains = [];
	const counts = new Int32Array(TEMPORAL_UNKNOWN_AA);
	const get = domainMap instanceof Map ? (/** @type {number} */ k) => domainMap.get(k) : (/** @type {number} */ k) => (domainMap ? /** @type {any} */ (domainMap)[k] : undefined);
	for (let s = 0; s < L; s++) {
		const ref = rootIndices[s];
		const o = s * N;
		counts.fill(0);
		let seen = 0;
		for (let n = 0; n < N; n++) {
			const tok = aValid[o + n];
			if (tok < TEMPORAL_UNKNOWN_AA && tok !== ref) { counts[tok]++; seen++; }
		}
		if (seen === 0) {
			counts.fill(0);
			for (let n = 0; n < N; n++) {
				const tok = aValid[o + n];
				if (tok < TEMPORAL_UNKNOWN_AA) { counts[tok]++; seen++; }
			}
		}
		let der = rootAas[s];
		if (seen > 0) {
			let best = -1;
			let idx = 0;
			for (let k = 0; k < TEMPORAL_UNKNOWN_AA; k++) if (counts[k] > best) { best = counts[k]; idx = k; }
			der = REV_AA_MAP.get(idx) ?? rootAas[s];
		}
		derivedAas.push(der);
		mutationLabels.push(`${rootAas[s]}${s + 1}${der}`);
		const d = domainMap ? get(s + 1) : undefined;
		domains.push(d === undefined || d === null ? 'Core' : d);
	}
	return { derivedAas, mutationLabels, domains };
}

// ---------------------------------------------------------------------------------------------
// The grid (temporal.py:544, 557)
// ---------------------------------------------------------------------------------------------

/**
 * The two time axes and the gradient axis the whole pillar runs on (temporal.py:544, 557, 561).
 *
 * `gradT` keys off UNITS, not the sweep mode: a calendar run differentiates against real calendar
 * time (which keeps viral behaviour byte-identical to the pre-`--time-units` reference), a
 * non-calendar one against the normalised [0, 1] axis so the derivative and its floors are
 * scale-invariant across arbitrary generation spans.
 *
 * @param {number} tMin float64 of a FLOAT32 date
 * @param {number} tMax float64 of a FLOAT32 date
 * @param {number} T
 * @param {boolean} nonCalendar
 * @returns {{ denseT: Float64Array, normDenseT: Float64Array, gradT: Float64Array }}
 */
export function temporalTimeGrid(tMin, tMax, T, nonCalendar) {
	const denseT = numpyLinspace(tMin, tMax, T);
	const normDenseT = numpyLinspace(0.0, 1.0, T);
	return { denseT, normDenseT, gradT: nonCalendar ? normDenseT : denseT };
}
