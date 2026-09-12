/**
 * WHY THIS FILE EXISTS
 *
 * numpy's reduction order, as one kernel routine. `np.sum` / `np.add.reduce` / `np.mean` over a
 * contiguous float array do not add left to right: they run `pairwise_sum`
 * (numpy/_core/src/umath/loops_utils.h.src, numpy 2.3.3) — a plain loop below 8 elements, eight
 * independent accumulators over blocks of up to 128, then recursive halving on a multiple of 8.
 * In float64 the difference from a sequential sum is ~1e-16 relative and invisible at every
 * fixture class; in FLOAT32 it is not. The reference holds site LRTs as float32 arrays and reduces
 * them in float32 (cli.py:485-486 `np.sum(lrts)`, filter.py:269/385/391 `np.mean(...)`,
 * attribution.py `d.mean(axis=1)`, stats.py:78 `np.mean(np.tan(...))` on a float32 input), and a
 * float64 or sequentially-ordered sum misses the fixtures by 6e-6 (busted_Smc6
 * total_selection_energy) to 9e-8 (cauchy_combination_p float32_input) — measured with the
 * reference Python by the omnibus and numeric builders, both above the classes those fields are
 * pinned at. With the rounding function injected (`Math.fround` for float32, identity for
 * float64) this one routine reproduces every such reduction bit for bit.
 *
 * Three ports (numeric/cauchy.js, filter.js, omnibus.js) each carried a private copy of the same
 * function during Phase 1a; this file is the single copy they now import. No I/O, no method
 * semantics; the Python it mirrors is a C loop, cited above.
 */

/**
 * numpy's `pairwise_sum` over `a[lo, lo + n)` with `round` applied after every addition.
 *
 * @param {ArrayLike<number>} a
 * @param {number} lo first index
 * @param {number} n element count
 * @param {(v: number) => number} round `Math.fround` for a float32 accumulator, identity for float64
 * @returns {number}
 */
export function numpyPairwiseSum(a, lo, n, round) {
	if (n < 8) {
		let res = 0;
		for (let i = 0; i < n; i++) res = round(res + a[lo + i]);
		return res;
	}
	if (n <= 128) {
		const r = new Array(8);
		for (let j = 0; j < 8; j++) r[j] = a[lo + j];
		const limit = n - (n % 8);
		for (let i = 8; i < limit; i += 8) {
			for (let j = 0; j < 8; j++) r[j] = round(r[j] + a[lo + i + j]);
		}
		let res = round(
			round(round(r[0] + r[1]) + round(r[2] + r[3])) + round(round(r[4] + r[5]) + round(r[6] + r[7]))
		);
		for (let i = limit; i < n; i++) res = round(res + a[lo + i]);
		return res;
	}
	let n2 = Math.floor(n / 2);
	n2 -= n2 % 8;
	return round(numpyPairwiseSum(a, lo, n2, round) + numpyPairwiseSum(a, lo + n2, n - n2, round));
}

/**
 * `float(np.sum(x))` for a float32 array: numpy's pairwise summation with a float32 accumulator.
 * Values are rounded to float32 on read, so any ArrayLike is treated as the float32 array the
 * reference holds. An empty input gives 0, as `np.sum([])` does.
 *
 * @param {ArrayLike<number>} values
 * @returns {number}
 */
export function float32Sum(values) {
	const f = values instanceof Float32Array ? values : Float32Array.from(values);
	return numpyPairwiseSum(f, 0, f.length, Math.fround);
}

/**
 * `float(np.mean(x[lo:lo+n]))` for a float32 array `x`: pairwise float32 sum, float32 divide.
 * An empty slice gives NaN, as numpy does (with a warning).
 *
 * @param {ArrayLike<number>} x
 * @param {number} [lo]
 * @param {number} [n]
 * @returns {number}
 */
export function numpyMeanFloat32(x, lo = 0, n = x.length - lo) {
	if (n <= 0) return NaN;
	return Math.fround(numpyPairwiseSum(x, lo, n, Math.fround) / n);
}

/**
 * `float(np.percentile(x, q))` in float64, numpy 2.3's default method `'linear'`.
 *
 * dating.py:1844 picks the restricted cubic spline's second and third knots as
 * `np.median(times)` and `np.percentile(times, 90)`, and a knot that moves changes every spline
 * number downstream of it — F, p, ΔAIC and therefore the model selection — so this is on the
 * deterministic critical path and is held to EXACT agreement with numpy, not to a tolerance.
 * MEASURED on the acceptance case: reproduces `np.percentile(times, 90) = 1995.5` and
 * `np.median(times) = 1992.5` bit-exactly, and `np.median` bit-exactly for even n too, which is
 * why no separate `median` exists here — `percentile(x, 50)` is it.
 *
 * The virtual index is numpy's `_QuantileMethods['linear']` spelt EXACTLY as numpy spells it:
 * `v = (n − 1)·(q/100)`, floored to the lower index and clamped to the last element above the top.
 * The algebraically identical `n·p + (1 − p) − 1` is NOT the same in floating point — measured, it
 * disagrees with numpy on 9 of the 63 grid points in test/data/numeric/optimize.json — and the
 * interpolation is numpy's own `_lerp`, which switches to `b − (b − a)(1 − t)` once t ≥ 0.5 so the
 * result is monotone and lands exactly on `b` at t = 1.
 *
 * NaN propagates (np.percentile's own behaviour, minus the RuntimeWarning); an empty input is NaN.
 * The input is not mutated.
 *
 * @param {ArrayLike<number>} x
 * @param {number} q percentile in [0, 100]
 * @returns {number}
 */
export function percentile(x, q) {
	const n = x.length;
	if (n === 0) return NaN;
	const a = Float64Array.from(x);
	for (let i = 0; i < n; i++) if (Number.isNaN(a[i])) return NaN;
	a.sort();
	const p = q / 100;
	const v = (n - 1) * p;
	if (Number.isNaN(v)) return NaN;
	if (v >= n - 1) return a[n - 1];
	if (v < 0) return a[0];
	const i = Math.floor(v);
	const t = v - i;
	const lo = a[i];
	const hi = a[i + 1];
	const diff = hi - lo;
	// numpy/lib/_function_base_impl.py `_lerp`
	return t >= 0.5 ? hi - diff * (1 - t) : lo + diff * t;
}

/**
 * `float(np.mean(x[lo:lo+n]))` in float64: pairwise sum, then one divide.
 *
 * Added for `hyphaeon/temporal.py`, which reduces float64 `[L, T]` trajectory matrices along the
 * contiguous axis in four places (temporal.py:709-710, 728-730, 738-739, 785) — numpy reduces a
 * C-contiguous last axis pairwise, per row.
 *
 * @param {ArrayLike<number>} x
 * @param {number} [lo]
 * @param {number} [n]
 * @returns {number}
 */
export function numpyMeanFloat64(x, lo = 0, n = x.length - lo) {
	if (n <= 0) return NaN;
	return numpyPairwiseSum(x, lo, n, identity) / n;
}

/**
 * `float(np.var(x[lo:lo+n]))` in float64 with `ddof = 0` — the statistic
 * `hyphaeon/temporal.py:642` uses for the episodic date-shuffling null.
 *
 * numpy's `_var` is TWO passes and both of them are pairwise: `arrmean = umr_sum(arr)/div`, then
 * `umr_sum((arr − arrmean)**2)/div`. A one-pass `E[x²] − E[x]²` is a different (and much worse)
 * float, and a running accumulation is a different float again — which matters because the null's
 * decision is a `>=` comparison against `v_obs` computed by the same routine, so any asymmetry
 * between the two sides biases the exceedance count.
 *
 * @param {ArrayLike<number>} x
 * @param {number} [lo]
 * @param {number} [n]
 * @returns {number}
 */
export function numpyVarFloat64(x, lo = 0, n = x.length - lo) {
	if (n <= 0) return NaN;
	const mean = numpyPairwiseSum(x, lo, n, identity) / n;
	const dev = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		const d = x[lo + i] - mean;
		dev[i] = d * d;
	}
	return numpyPairwiseSum(dev, 0, n, identity) / n;
}

/**
 * `float(np.std(x[lo:lo+n]))` in float64, `ddof = 0` — `sqrt` of {@link numpyVarFloat64}, which is
 * how numpy computes it (`_std` calls `_var` and takes the square root of the result).
 *
 * @param {ArrayLike<number>} x
 * @param {number} [lo]
 * @param {number} [n]
 * @returns {number}
 */
export function numpyStdFloat64(x, lo = 0, n = x.length - lo) {
	return Math.sqrt(numpyVarFloat64(x, lo, n));
}

/** Float64 accumulation: no rounding between additions. */
function identity(/** @type {number} */ v) {
	return v;
}
