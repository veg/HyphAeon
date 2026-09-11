/**
 * WHY THIS FILE EXISTS
 *
 * Mirrors `benjamini_hochberg` in hyphaeon/stats.py:50-70 (at reconcile/phase-5a), line for line:
 *
 *     n = len(pvals); if n == 0: return []
 *     sorted_idx = np.argsort(pvals); sorted_p = pvals[sorted_idx]
 *     min_q = 1.0
 *     for i in range(n - 1, -1, -1):
 *         q = sorted_p[i] * n / (i + 1)
 *         if q < min_q: min_q = q
 *         qvals[i] = min_q
 *     res[sorted_idx] = qvals
 *     return np.clip(res, 0.0, 1.0)
 *
 * It lives in the numeric kernel rather than the stats port because filter.py, epistasis.py and
 * phenotype.py all call it, and the fixture class is 1e-9 with an "exact" expectation on ties
 * (PLAN.md §5.4: BH is in the exact class).
 *
 * TWO THINGS THE PYTHON DOES THAT ARE EASY TO GET WRONG:
 *   1. The running minimum starts at 1.0, not at the top q, so a top-ranked q above 1 is already
 *      capped before the final clip — and the clip to [0, 1] is applied on top of that.
 *   2. Arithmetic happens in the input dtype. With float32 p-values (epistasis.py casts q to
 *      float32; `cmd_meme` stores p as float32) `sorted_p[i] * n / (i + 1)` is float32 × int and
 *      numpy 2's promotion keeps it float32, so the q-values themselves are float32. The fixture
 *      case `float32_meme_like` is off by 4.2e-8 from a float64 replay (measured with the reference
 *      Python) and matches a float32 replay exactly. This port therefore rounds each operation with
 *      Math.fround when it is handed a Float32Array, and works in float64 otherwise. Callers must
 *      pass the array type that matches the Python dtype at the call site.
 *
 * numpy's argsort is an unstable quicksort; since tied p-values yield identical q it does not
 * matter, and this uses a stable sort.
 */

/**
 * Benjamini–Hochberg q-values, clipped to [0, 1] — hyphaeon/stats.py:50-70.
 *
 * @param {ArrayLike<number>} pvals a Float32Array selects float32 arithmetic (see header)
 * @returns {Float64Array} same length as `pvals`
 */
export function benjaminiHochberg(pvals) {
	const n = pvals.length;
	const res = new Float64Array(n);
	if (n === 0) return res;
	const f32 = pvals instanceof Float32Array;
	const round = f32 ? Math.fround : (/** @type {number} */ v) => v;
	const idx = new Array(n);
	for (let i = 0; i < n; i++) idx[i] = i;
	idx.sort((a, b) => pvals[a] - pvals[b] || a - b);
	const qvals = new Float64Array(n);
	let minQ = 1.0;
	for (let i = n - 1; i >= 0; i--) {
		const q = round(round(pvals[idx[i]] * n) / (i + 1));
		if (q < minQ) minQ = q;
		qvals[i] = minQ;
	}
	for (let i = 0; i < n; i++) res[idx[i]] = qvals[i];
	for (let i = 0; i < n; i++) {
		if (res[i] < 0) res[i] = 0;
		else if (res[i] > 1) res[i] = 1;
	}
	return res;
}
