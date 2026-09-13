/**
 * WHY THIS FILE EXISTS
 *
 * The three numpy array routines `hyphaeon/temporal.py` builds its time axis and its sweep metric
 * out of — `np.linspace` (temporal.py:544, 557), `np.gradient` (temporal.py:566, 641) and
 * `np.trapezoid` (temporal.py:577, 581, through the `_trapezoid` alias at temporal.py:66) — spelt
 * the way numpy spells them rather than the way the formulae are usually written. Every one of the
 * three has a detail that a "mathematically equivalent" transcription gets wrong, and all three sit
 * on the deterministic half of the temporal pillar, which PLAN-TEMPORAL.md §5.1 holds at the strict
 * graph class:
 *
 *   - `linspace` computes `y[i] = i·step + start` with ONE step, then ASSIGNS `y[num-1] = stop`.
 *     `start + i·(stop − start)/(num − 1)` evaluated per element is a different float.
 *   - `gradient` reduces to its uniform branch only when `(np.diff(x) == np.diff(x)[0]).all()` —
 *     EXACT equality. MEASURED on the acceptance run (`np.linspace(2009.2490234375,
 *     2009.9150390625, 60)`, numpy 2.3.3): the 59 spacings take two distinct values,
 *     1.1288400423609346e-2 and 1.128840042383672e-2, so the NON-uniform branch is what actually
 *     runs for a linspace axis. A port that assumes `h = (t_max − t_min)/(T − 1)` misses every
 *     interior velocity.
 *   - `trapezoid` forms the whole array of half-trapezoids and reduces it with numpy's PAIRWISE
 *     summation, not a running left-to-right accumulation.
 *
 * The edge rule matters as much as the interior one: `edge_order` defaults to 1, so `gradient`'s
 * first and last columns are plain one-sided first-order differences while the interior is the
 * three-point non-uniform formula. Those two columns are exactly where a bad port shows up, and
 * they are the two the FWHM search (temporal.py:588-597) reads first and last.
 *
 * Ported from numpy 2.3.3: `numpy/_core/function_base.py` (`linspace`),
 * `numpy/lib/_function_base_impl.py` (`gradient`, `trapezoid`). No method semantics, no I/O.
 */

import { numpyPairwiseSum } from './reduce.js';

/** Identity rounding: a float64 accumulator, the dtype every routine here works in. */
const f64 = (/** @type {number} */ v) => v;

/**
 * `np.linspace(start, stop, num)` with `endpoint=True`, in float64.
 *
 * numpy computes `y = arange(num) * step` and then `y += start`, so each element is ONE multiply
 * and ONE add away from the integer index; the last element is then OVERWRITTEN with `stop` rather
 * than computed. `num == 1` gives `[start]` (numpy's `div = 0` branch: no step, and the endpoint
 * assignment is skipped because `num > 1` fails).
 *
 * @param {number} start
 * @param {number} stop
 * @param {number} num how many points, >= 0
 * @returns {Float64Array}
 */
export function numpyLinspace(start, stop, num) {
	if (!Number.isInteger(num) || num < 0) throw new RangeError(`numpyLinspace: num must be a non-negative integer, got ${num}`);
	const y = new Float64Array(num);
	if (num === 0) return y;
	const div = num - 1;
	if (div > 0) {
		const step = (stop - start) / div;
		if (step === 0) {
			// numpy's `y /= div; y *= delta` branch, taken when the step underflows to zero.
			for (let i = 0; i < num; i++) y[i] = (i / div) * (stop - start);
		} else {
			for (let i = 0; i < num; i++) y[i] = i * step;
		}
	}
	for (let i = 0; i < num; i++) y[i] += start;
	if (div > 0) y[num - 1] = stop;
	return y;
}

/**
 * Whether `np.gradient` would take its uniform-spacing shortcut for the coordinate array `x`:
 * `diffx = np.diff(x)`, then `(diffx == diffx[0]).all()` — EXACT float equality, no tolerance.
 *
 * Exposed because the answer is surprising on a linspace axis (see the header) and because a test
 * that asserts which branch ran is worth more than one that only checks the numbers.
 *
 * @param {ArrayLike<number>} x
 * @returns {boolean} true when fewer than two spacings exist or all spacings are bit-identical
 */
export function isUniformSpacing(x) {
	const n = x.length;
	if (n < 3) return true; // one spacing (or none): trivially "all equal to the first"
	const d0 = x[1] - x[0];
	for (let i = 2; i < n; i++) if (x[i] - x[i - 1] !== d0) return false;
	return true;
}

/**
 * `np.gradient(row, x, edge_order=1)` for one 1-D float64 row against a coordinate array.
 *
 * Interior, non-uniform branch (numpy's own coefficient form and operand order):
 *
 *     hd = x[i] − x[i−1];  hs = x[i+1] − x[i]
 *     a = −hs / (hd·(hd + hs));  b = (hs − hd)/(hs·hd);  c = hd / (hs·(hd + hs))
 *     out[i] = a·f[i−1] + b·f[i] + c·f[i+1]
 *
 * Interior, uniform branch: `(f[i+1] − f[i−1]) / (2·h)`.
 * Edges at `edge_order = 1`: `(f[1] − f[0]) / (x[1] − x[0])` and `(f[n−1] − f[n−2]) / (x[n−1] − x[n−2])`.
 *
 * @param {ArrayLike<number>} f
 * @param {ArrayLike<number>} x same length as f, length >= 2
 * @param {Float64Array} [out] optional destination (length f.length)
 * @returns {Float64Array}
 */
export function numpyGradient(f, x, out = new Float64Array(f.length)) {
	const n = f.length;
	if (x.length !== n) throw new RangeError(`numpyGradient: f and x must have the same length (${n} vs ${x.length})`);
	if (n < 2) throw new RangeError('numpyGradient: at least 2 points are required for edge_order=1');
	if (n === 2) {
		const d = (f[1] - f[0]) / (x[1] - x[0]);
		out[0] = d;
		out[1] = d;
		return out;
	}
	if (isUniformSpacing(x)) {
		const h = x[1] - x[0];
		const twoH = 2 * h;
		for (let i = 1; i < n - 1; i++) out[i] = (f[i + 1] - f[i - 1]) / twoH;
	} else {
		for (let i = 1; i < n - 1; i++) {
			const hd = x[i] - x[i - 1];
			const hs = x[i + 1] - x[i];
			const a = -hs / (hd * (hd + hs));
			const b = (hs - hd) / (hs * hd);
			const c = hd / (hs * (hd + hs));
			out[i] = a * f[i - 1] + b * f[i] + c * f[i + 1];
		}
	}
	out[0] = (f[1] - f[0]) / (x[1] - x[0]);
	out[n - 1] = (f[n - 1] - f[n - 2]) / (x[n - 1] - x[n - 2]);
	return out;
}

/**
 * `np.gradient(M, x, axis=1)` over a row-major `[rows, n]` float64 matrix, edge_order = 1.
 *
 * numpy evaluates the whole array at once, but every element of a row depends only on that row, so
 * a per-row loop is the same arithmetic in the same order. The uniformity test is done ONCE on `x`
 * (numpy does the same), so both branches are shared by every row.
 *
 * @param {ArrayLike<number>} M row-major rows*n
 * @param {number} rows
 * @param {number} n
 * @param {ArrayLike<number>} x coordinate array, length n
 * @param {Float64Array} [out]
 * @returns {Float64Array} row-major rows*n
 */
export function numpyGradientRows(M, rows, n, x, out = new Float64Array(rows * n)) {
	if (n < 2) throw new RangeError('numpyGradientRows: at least 2 columns are required for edge_order=1');
	const uniform = isUniformSpacing(x);
	// Coefficients depend only on x, so they are built once per CALL and reused across every row of
	// it. Per call, not once ever: the non-uniform branch allocates three `Float64Array(n − 2)` every
	// time, and the temporal null calls this once per draw. MEASURED (node 22.22.0, darwin/x64,
	// 246 x 60 rows, 1,000 calls, best of five, three processes) before leaving it that way: 48.9 /
	// 54.6 / 54.8 ms as written against 77.0 / 85.6 / 86.1 ms reading hoisted module-scope copies,
	// bit-identical output — rebuilding 58-element locals beats loading them from outside.
	let a = null;
	let b = null;
	let c = null;
	let twoH = 0;
	if (n > 2) {
		if (uniform) {
			twoH = 2 * (x[1] - x[0]);
		} else {
			a = new Float64Array(n - 2);
			b = new Float64Array(n - 2);
			c = new Float64Array(n - 2);
			for (let i = 1; i < n - 1; i++) {
				const hd = x[i] - x[i - 1];
				const hs = x[i + 1] - x[i];
				a[i - 1] = -hs / (hd * (hd + hs));
				b[i - 1] = (hs - hd) / (hs * hd);
				c[i - 1] = hd / (hs * (hd + hs));
			}
		}
	}
	const dx0 = x[1] - x[0];
	const dxn = x[n - 1] - x[n - 2];
	for (let r = 0; r < rows; r++) {
		const o = r * n;
		if (n === 2) {
			const d = (M[o + 1] - M[o]) / dx0;
			out[o] = d;
			out[o + 1] = d;
			continue;
		}
		if (uniform) {
			for (let i = 1; i < n - 1; i++) out[o + i] = (M[o + i + 1] - M[o + i - 1]) / twoH;
		} else {
			for (let i = 1; i < n - 1; i++) {
				out[o + i] = a[i - 1] * M[o + i - 1] + b[i - 1] * M[o + i] + c[i - 1] * M[o + i + 1];
			}
		}
		out[o] = (M[o + 1] - M[o]) / dx0;
		out[o + n - 1] = (M[o + n - 1] - M[o + n - 2]) / dxn;
	}
	return out;
}

/**
 * `np.trapezoid(y, x)` for one 1-D row: `d = np.diff(x)`, then
 * `(d * (y[1:] + y[:-1]) / 2.0).sum()` — the half-trapezoids formed as an array and reduced by
 * numpy's PAIRWISE summation, which is why the array is materialised here rather than accumulated.
 *
 * A single point integrates to 0 (numpy's `d` is empty and `sum` of an empty array is 0).
 *
 * @param {ArrayLike<number>} y
 * @param {ArrayLike<number>} x same length as y
 * @returns {number}
 */
export function numpyTrapezoid(y, x) {
	const n = y.length;
	if (x.length !== n) throw new RangeError(`numpyTrapezoid: y and x must have the same length (${n} vs ${x.length})`);
	if (n < 2) return 0;
	const terms = new Float64Array(n - 1);
	for (let i = 0; i < n - 1; i++) terms[i] = ((x[i + 1] - x[i]) * (y[i + 1] + y[i])) / 2.0;
	return numpyPairwiseSum(terms, 0, n - 1, f64);
}

/**
 * `np.trapezoid(M, x, axis=1)` over a row-major `[rows, n]` matrix.
 *
 * @param {ArrayLike<number>} M row-major rows*n
 * @param {number} rows
 * @param {number} n
 * @param {ArrayLike<number>} x length n
 * @param {Float64Array} [out]
 * @returns {Float64Array} length rows
 */
export function numpyTrapezoidRows(M, rows, n, x, out = new Float64Array(rows)) {
	if (n < 2) return out;
	const d = new Float64Array(n - 1);
	for (let i = 0; i < n - 1; i++) d[i] = x[i + 1] - x[i];
	const terms = new Float64Array(n - 1);
	for (let r = 0; r < rows; r++) {
		const o = r * n;
		for (let i = 0; i < n - 1; i++) terms[i] = (d[i] * (M[o + i + 1] + M[o + i])) / 2.0;
		out[r] = numpyPairwiseSum(terms, 0, n - 1, f64);
	}
	return out;
}
