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
