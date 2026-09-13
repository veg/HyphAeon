/**
 * WHY THIS FILE EXISTS
 *
 * Since 2026-09-13 `src/preprocess/tn93.js` computes no TN93 distance: the compiled veg/tn93 engine
 * is the only implementation and it is injected as `options.pairwiseDistances` (see that file's
 * header for the decision and why there is no fallback). Every library path that builds a tree-free
 * distance matrix therefore needs an engine, and so does every test that drives one. This module is
 * where the test suite gets one, in exactly two flavours, kept apart on purpose because they carry
 * very different weight:
 *
 *   fixtureSquareEngine / fixtureCrossEngine  REPLAY. They hand back the PYTHON reference's own
 *       numbers, taken from the fixture the test is replaying and looked up BY TAXON NAME so that
 *       duplicate pruning, stride pre-selection and Faith's PD sub-sampling — all of which reorder
 *       and shorten the taxon list before the matrix is built — cannot silently misalign them. A
 *       taxon the fixture does not carry throws rather than defaulting, so a misalignment is loud.
 *       Everything downstream of the numbers is then the library's own work and is still proved
 *       against Python: the float32 rounding, the sentinel, dataset.py's imputation, the MDS, the
 *       tokens, the invariable mask, the notices.
 *
 *   stubEngine                                A TEST DOUBLE, AND NOT A TN93 IMPLEMENTATION. It
 *       returns a deterministic, well-conditioned number per pair (a scaled Hamming-ish disagreement
 *       count) purely so that a path which needs SOME distances can run. Use it only where the test
 *       asserts nothing about distances — taxon matching, notices, refusal codes, filter mechanics,
 *       export lists. Never use it in a block that claims parity with the reference; if a test's
 *       subject is a number, it must come from a fixture.
 *
 * Deliberately NOT here: anything resembling the deleted port. Reconstructing TN93 arithmetic in the
 * test tree would recreate exactly the second implementation the deletion removed, one directory
 * further away from the code that used it.
 */

/** @typedef {(...args: any[]) => ArrayLike<number>} PairwiseDistances */

/**
 * A `pairwiseDistances` hook for the SQUARE `tn93DistanceMatrix`, replaying a fixture's matrix.
 *
 * @param {string[]} fixtureTaxa the fixture's taxon order
 * @param {number[][]|number[]} matrix the fixture's [n, n] distances (rows, or already row-major)
 * @returns {PairwiseDistances} called as (seqs, taxa, threshold), read at [i * n + j]
 */
export function fixtureSquareEngine(fixtureTaxa, matrix) {
	const n = fixtureTaxa.length;
	const flat = Array.isArray(matrix[0]) ? /** @type {number[][]} */ (matrix).flat() : /** @type {number[]} */ (matrix);
	if (flat.length !== n * n) throw new Error(`fixtureSquareEngine: ${flat.length} values for ${n} taxa`);
	const index = new Map(fixtureTaxa.map((t, i) => [t, i]));
	return (/** @type {string[]} */ _seqs, /** @type {string[]} */ taxa) => {
		const m = taxa.length;
		const out = new Float64Array(m * m);
		const rows = taxa.map((t) => {
			const i = index.get(t);
			// Loud on purpose: a taxon the fixture never scored means the library reordered or pruned
			// in a way the replay does not model, and a quiet 0 there would look like a measurement.
			if (i === undefined) throw new Error(`fixtureSquareEngine: the fixture has no row for taxon '${t}'`);
			return i;
		});
		for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) out[i * m + j] = flat[rows[i] * n + rows[j]];
		return out;
	};
}

/**
 * A `pairwiseDistances` hook for the RECTANGULAR `tn93CrossDistanceMatrix`, replaying a fixture's
 * matrix. The library calls this shape with FIVE arguments (allSeqs, lmSeqs, taxaAll, taxaLandmarks,
 * threshold), which is how a dual-shape engine tells the two apart.
 *
 * @param {string[]} fixtureRows the fixture's row taxa, in order
 * @param {string[]} fixtureCols the fixture's landmark taxa, in order
 * @param {number[][]|number[]} matrix the fixture's [n, m] distances
 * @returns {PairwiseDistances} read at [i * m + j]
 */
export function fixtureCrossEngine(fixtureRows, fixtureCols, matrix) {
	const n = fixtureRows.length;
	const m = fixtureCols.length;
	const flat = Array.isArray(matrix[0]) ? /** @type {number[][]} */ (matrix).flat() : /** @type {number[]} */ (matrix);
	if (flat.length !== n * m) throw new Error(`fixtureCrossEngine: ${flat.length} values for ${n} x ${m}`);
	const rowOf = new Map(fixtureRows.map((t, i) => [t, i]));
	const colOf = new Map(fixtureCols.map((t, j) => [t, j]));
	return (/** @type {string[]} */ _all, /** @type {string[]} */ _lm, /** @type {string[]} */ taxaAll, /** @type {string[]} */ taxaLandmarks) => {
		const out = new Float64Array(taxaAll.length * taxaLandmarks.length);
		const M = taxaLandmarks.length;
		for (let i = 0; i < taxaAll.length; i++) {
			const ri = rowOf.get(taxaAll[i]);
			if (ri === undefined) throw new Error(`fixtureCrossEngine: the fixture has no row for taxon '${taxaAll[i]}'`);
			for (let j = 0; j < M; j++) {
				const cj = colOf.get(taxaLandmarks[j]);
				if (cj === undefined) throw new Error(`fixtureCrossEngine: the fixture has no column for landmark '${taxaLandmarks[j]}'`);
				out[i * M + j] = flat[ri * m + cj];
			}
		}
		return out;
	};
}

/**
 * A TEST DOUBLE. Not TN93, not parity evidence, and never to be read as a distance: the fraction of
 * positions at which two sequences carry different characters, scaled to a plausible range so that
 * the MDS behind it is well conditioned and the imputation rule is not triggered. Identical
 * sequences give 0, as a distance would, so duplicate-pruning tests still make sense.
 *
 * It answers BOTH library call shapes, told apart by argument count exactly as the application's own
 * dual provider does (hyphaeon-app runtime/src/tn93-wasm.js): three arguments is the square matrix,
 * five is the rectangular one.
 *
 * @returns {PairwiseDistances}
 */
export function stubEngine() {
	const disagreement = (/** @type {string} */ a, /** @type {string} */ b) => {
		const len = Math.min(a.length, b.length);
		if (len === 0) return 0;
		let diff = 0;
		for (let i = 0; i < len; i++) if (a[i] !== b[i]) diff++;
		// 0.5 keeps every value inside the (0, 1) band where nothing is imputed and nothing counts as
		// saturated, so a test using this double never trips a rule it did not mean to exercise.
		return (0.5 * diff) / len;
	};
	return (/** @type {any[]} */ ...args) => {
		if (args.length === 3) {
			const [seqs] = /** @type {[string[], string[], number]} */ (args);
			const n = seqs.length;
			const out = new Float64Array(n * n);
			for (let i = 0; i < n; i++) {
				for (let j = i + 1; j < n; j++) {
					const d = disagreement(seqs[i], seqs[j]);
					out[i * n + j] = d;
					out[j * n + i] = d;
				}
			}
			return out;
		}
		if (args.length === 5) {
			const [allSeqs, lmSeqs] = /** @type {[string[], string[], string[], string[], number]} */ (args);
			const out = new Float64Array(allSeqs.length * lmSeqs.length);
			for (let i = 0; i < allSeqs.length; i++) {
				for (let j = 0; j < lmSeqs.length; j++) out[i * lmSeqs.length + j] = disagreement(allSeqs[i], lmSeqs[j]);
			}
			return out;
		}
		throw new Error(`stubEngine: unrecognised pairwiseDistances signature (${args.length} arguments)`);
	};
}

/** `stubEngine()` wrapped as the options bag `loadAlignmentAndTree` and `diagnose` take. */
export function stubTn93Options() {
	return { pairwiseDistances: stubEngine() };
}
