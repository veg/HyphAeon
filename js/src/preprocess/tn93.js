/**
 * WHY THIS FILE EXISTS
 *
 * The tree-free distance path (PLAN.md D22): when no usable tree is available, pairwise
 * Tamura-Nei 1993 distances replace the patristic matrix and feed the MDS directly.
 *
 * THIS FILE DOES NOT COMPUTE A TN93 DISTANCE. It never will again. It mirrors the WRAPPING that
 * `hyphaeon/dataset.py:715-821` (`compute_tn93_distance_matrix`) and `:824-928`
 * (`compute_tn93_cross_distance_matrix`) put around whatever numbers an engine hands back — the
 * `-1.0` "not written" prefill, the dead `d < 0 or isnan -> 1.0` guard, the float32 rounding, the
 * `max(1.0, max_d)` imputation, the zeroed diagonal and the saturation sentinel. The numbers
 * themselves come from `options.pairwiseDistances`, which is now REQUIRED: without it both matrix
 * functions throw `Tn93EngineRequiredError`.
 *
 * ============================ WHY THE PORT IS GONE (2026-09-13) ============================
 *
 * Until this change the file also carried a line-by-line port of the `tn93` PyPI package 1.2.2
 * (`tn93/tn93.py`), the pure-Python fallback `dataset.py:785-810` takes when no `tn93` binary is on
 * PATH: `encodeSequence`, `canResolve`, `ambigFractionTooHigh`, `tn93Counts`,
 * `tn93NucleotideFrequency`, `tn93CalculateDistance`, `tn93Distance` and the `TN93_TABLES` they
 * read. It matched the reference bit for bit on every fixture. It is deleted anyway, by the
 * repository owner's decision, and the reason is about edge cases rather than speed:
 *
 *   veg/tn93 is a tool THIS TEAM MAINTAINS. The compiled WebAssembly build the application vendors
 *   (hyphaeon-app `runtime/vendor/tn93/`, verified by sha256 before it is instantiated) is how that
 *   repository's fixes reach a run: a bug is fixed upstream, the release is re-vendored, the hash
 *   changes. A JavaScript port is a SECOND IMPLEMENTATION of the same arithmetic, kept in step by
 *   hand, and every path still able to reach it is a path where the two can silently disagree the
 *   day the tool changes — on an ambiguity convention, a gap rule, a saturated pair. Timings are
 *   not the argument; the compiled engine is the slower of the two on a small matrix and that price
 *   was accepted. An earlier change that picked between them by matrix size was rejected for
 *   exactly this reason, and a size-based rule is why the throw below is UNCONDITIONAL: a one-taxon
 *   matrix refuses as loudly as a thousand-taxon one, so no input size quietly takes a different
 *   code path.
 *
 * BREAKING CHANGE to `@veg/hyphaeon-js`. Eight exports are removed, listed in the release notes of
 * this change; nothing inside the library called any of them except the two matrix functions here.
 *
 * WHAT THE ENGINE IS ASKED TO DO, and what the caller must therefore run. dataset.py:738
 * `tn93_bin = shutil.which("tn93")`; when the compiled binary is on PATH the reference runs
 * (dataset.py:740-784):
 *
 *     tn93 -t <threshold> -l 1 -q -o distances.csv subset.fa
 *
 * i.e. the reporting threshold below, minimum overlap 1 nucleotide, quiet, CSV out (columns
 * ID1,ID2,Distance) and the binary's DEFAULT ambiguity strategy, `-a resolve` with `-g 1.0` —
 * TN93_MATCH_MODE and TN93_MAX_AMBIG_FRACTION here. Each row's Distance is `float()`d into both
 * `[i][j]` and `[j][i]`; any pair the binary omitted (above the threshold, or below the overlap)
 * stays -1.0 and is imputed below. That contract is the caller's to honour: this file only says
 * what shape it reads the answers in.
 *
 * MEASURED, 2026-09-05, before the port was deleted (tn93 binary v1.0.15 vs tn93 package 1.2.2,
 * both through `compute_tn93_distance_matrix`): the two paths agree EXACTLY — max |Δ| = 0.0 — on
 * examples/bat_oas1.fasta and examples/HIV1_RT.fasta, because the binary writes 6 significant
 * digits (the same rounding the package applies) and every pair it drops at a 1.0 threshold is
 * imputed back to exactly the 1.0 the package returns for it. MEASURED, 2026-09-07, by the
 * application (runtime/vendor/tn93/MANIFEST.json): the vendored v1.0.17 WebAssembly build gives
 * identical rows and values to native tn93 1.0.15 on bat_oas1 (153 pairs), Smc6 (190), camelid
 * (22,366) and HIV1_RT (113,050). Those two measurements are the parity chain that survives the
 * deletion, and both of them live outside this repository's JavaScript.
 *
 * ============================ WHAT THE WRAPPING STILL IS ============================
 *
 * PRECISION. An engine's distance is float64 here. The MATRIX is
 * `np.full((n, n), -1.0, dtype=np.float32)` (dataset.py:732), so each stored value is rounded to
 * float32 with Math.fround — matching the dtype `compute_mds_coordinates` then squares and
 * double-centres in float32.
 *
 * THE MATRIX (dataset.py:715-821):
 *   - the matrix starts at -1.0 with a zeroed diagonal: -1.0 marks an entry NOT WRITTEN, which is
 *     outside the range of any distance, so a measured 0.0 is distinguishable from a missing one.
 *   - n <= 1: that matrix is returned as it is, no distances computed (dataset.py:734-735).
 *   - off-diagonal, i < j only, mirrored into [j][i].
 *   - `if d is None or d == "-" or d < 0 or np.isnan(d): d = 1.0` (dataset.py:805) — replicated,
 *     and no longer dead: an engine may report a negative or a NaN where the Python package raised.
 *   - IMPUTATION (dataset.py:816-821), after every pair is scored:
 *         np.fill_diagonal(dist_mat, 0.0)
 *         missing = (dist_mat < 0.0)
 *         max_d = dist_mat.max() if dist_mat.max() > 0 else 1.0
 *         dist_mat[missing] = max(1.0, max_d)
 *     Only UNWRITTEN entries are imputed — a pair the engine omitted from its CSV. `max_d` is read
 *     ONCE, from the filled matrix.
 *   - the diagonal is zeroed last (dataset.py:821).
 *
 *   UPSTREAM FIX FOLLOWED, NOT A LOCAL IMPROVEMENT. The rule this replaced could not tell a genuine
 *   zero from a missing entry, because the matrix started at 0.0: it raised a zero distance between
 *   DISTINCT sequences to 1e-4, and gave BYTE-IDENTICAL sequences `max(1.0, max_d)` — the largest
 *   distance in the matrix, which this file previously flagged as a bug and replicated. The
 *   reference repaired it by moving the sentinel out of the value range. Both quirks are therefore
 *   gone, TN93_MIN_POSITIVE_DISTANCE is unreachable, and TN93_FALLBACK_MAX is 1.0 rather than 0.1.
 *   On the bundled examples this changes up to 35 entries of HIV1_RT's matrix (34 of them by 1e-4,
 *   one identical pair by ~1.0, and that one is pruned before the matrix is built in a default run)
 *   and none of camelid's, bat_oas1's or Smc6's.
 *
 * WHAT IT DELIBERATELY DOES NOT DO:
 *   - Compute a distance. See above. There is no subprocess and no WebAssembly in the library
 *     either: the engine is the caller's, injected as a plain function.
 *   - Decide the `*` convention, the match mode or the threshold. Those are the engine's argv and
 *     the application's per-pillar choice; see the note on `tn93CrossDistanceMatrix`.
 *   - Neighbour joining. PLAN.md D22's display topology (`nj.js`) is the application's module.
 *
 * A PORT GAP THAT IS NOW THE ENGINE'S. The Python package RAISES on two degenerate pairs —
 * `ZeroDivisionError` when two sequences share no position where both are non-gap, and
 * `math.log`'s `ValueError` on a saturated pair — and `dataset.py` catches the second in the
 * rectangular path only (`:903-906`), letting the first kill the run. The deleted port raised in
 * the same two places, which is what `diagnostics.js` turned into its TN93_SATURATED_PAIRS refusal.
 * The library can no longer raise either: whether such a pair throws, omits a row, or comes back as
 * a number is now the engine's behaviour, and what reaches this file is either a value (guarded
 * below) or a hole (imputed below). The refusal path in `diagnostics.js` still works — it catches
 * whatever the injected engine throws — but the library no longer guarantees it fires.
 */

/** The ambiguity strategy `dataset.py:791` asks for, and `-a resolve` on the compiled engine. */
export const TN93_MATCH_MODE = 'resolve';

/** `TN93(max_ambig_fraction=1.0)`, the constructor default the reference takes (tn93.py:136), and
 * `-g 1.0` on the compiled engine. */
export const TN93_MAX_AMBIG_FRACTION = 1.0;

/**
 * The value `calculate_distance` returns for a pair it cannot resolve (tn93.py:205, the `0 in
 * nucleotide_frequency` branch), the value the reference's `d < 0 or isnan` guard substitutes, and
 * the floor `compute_tn93_distance_matrix` imputes an unwritten entry with (dataset.py:819-820,
 * `max(1.0, max_d)`). A distance EQUAL to it is a saturation sentinel, not a measurement.
 */
export const TN93_SATURATION_SENTINEL = 1.0;

/**
 * DEPRECATED, and no longer reachable. It was the floor for a zero distance between two DIFFERENT
 * sequence strings, under the imputation rule the reference replaced (see the matrix section of the
 * header). A measured 0.0 is now kept as 0.0, so nothing is imputed to this value. Still exported
 * because the export list is a stated commitment and callers may pin it.
 */
export const TN93_MIN_POSITIVE_DISTANCE = 1e-4;

/** dataset.py:818 — `max_d` when the whole matrix is still zero. */
export const TN93_FALLBACK_MAX = 1.0;

/**
 * dataset.py:732 — the value `np.full((n, n), -1.0)` leaves in an entry that was never written,
 * which dataset.py:817's `missing = (dist_mat < 0.0)` then imputes. Module-private on purpose: it is
 * an internal marker, never a distance a caller should see or compare against.
 */
const TN93_MISSING_SENTINEL = -1.0;

/**
 * dataset.py:719 `threshold: float = 100.0` — the reporting cutoff the reference hands the compiled
 * binary (`-t 100.0`, retried at 1.0 when a stock build refuses it), so that divergent pairs are not
 * omitted from a deep alignment. It reaches the `pairwiseDistances` engine and nothing else.
 * Module-private because it is not a distance and not part of the export contract.
 */
const TN93_REPORTING_THRESHOLD = 100.0;

/**
 * Thrown when a matrix is asked for and no compiled TN93 engine was supplied.
 *
 * It is a PROGRAMMING error, not an input one: the caller reached a tree-free path without wiring
 * an engine into it. Callers that translate library throws into reader-facing findings — the
 * TN93_SATURATED_PAIRS refusal in `diagnostics.js` is the one in this repository — must let this
 * one through rather than describe it as a property of the data.
 */
export class Tn93EngineRequiredError extends Error {
	/** @param {string} who the function that refused */
	constructor(who) {
		super(
			`${who}: no compiled TN93 engine was supplied. Pass options.pairwiseDistances — a function ` +
				'returning the RAW pairwise distances from veg/tn93. @veg/hyphaeon-js contains no TN93 ' +
				'implementation of its own and will not compute distances itself (a second implementation ' +
				'of the same arithmetic is a second set of edge cases); the sentinel, the float32 rounding ' +
				"and dataset.py's imputation are all this library does with the numbers."
		);
		this.name = 'Tn93EngineRequiredError';
		/** A stable code for callers that classify errors without string matching. */
		this.code = 'TN93_ENGINE_REQUIRED';
		/** The option that was missing. */
		this.option = 'pairwiseDistances';
		/** The function that refused. */
		this.caller = who;
	}
}

/**
 * `compute_tn93_distance_matrix(seq_dict, taxa)`, dataset.py:715-821: the float32 [n, n] matrix,
 * row-major, with the reference's guard, imputation rule and zeroed diagonal, around an engine's
 * raw numbers.
 *
 * @param {Map<string, string>|Record<string, string>} sequences taxon -> aligned sequence
 * @param {string[]} taxa the rows/columns, in order
 * @param {{threshold?: number,
 *   pairwiseDistances: (seqs: string[], taxa: string[], threshold: number) => ArrayLike<number>}} options
 *   `pairwiseDistances` is REQUIRED and supplies the RAW pairwise distances, read at [i * n + j];
 *   the rounding, the sentinel and dataset.py's imputation happen here. `null` or `undefined` for a
 *   pair means "not written", exactly as a pair the binary omitted from its CSV leaves dataset.py's
 *   matrix at -1.0, and is imputed below; it is NOT the package's saturation answer, and at
 *   threshold 100.0 the two are different numbers. `threshold` is dataset.py:719's reporting cutoff,
 *   handed to the engine because only the engine has a cutoff.
 * @returns {Float32Array} length taxa.length ** 2
 * @throws {Tn93EngineRequiredError} when `pairwiseDistances` is absent
 */
export function tn93DistanceMatrix(sequences, taxa, options = /** @type {any} */ ({})) {
	// UNCONDITIONAL, and before the n <= 1 shortcut on purpose: a caller with no engine is broken at
	// every size, and a rule that let small inputs through would be the size-based selector this
	// library is not allowed to have.
	if (typeof options.pairwiseDistances !== 'function') throw new Tn93EngineRequiredError('tn93DistanceMatrix');

	const get = sequences instanceof Map ? (/** @type {string} */ t) => sequences.get(t) : (/** @type {string} */ t) => sequences[t];
	const n = taxa.length;
	const dist = new Float32Array(n * n);
	// dataset.py:732-733 `np.full((n, n), -1.0)` then a zeroed diagonal: an entry still negative after
	// scoring was never written, which is what dataset.py:817 imputes. A MEASURED 0.0 is a measurement
	// and survives.
	dist.fill(TN93_MISSING_SENTINEL);
	for (let i = 0; i < n; i++) dist[i * n + i] = 0.0;
	if (n <= 1) return dist;

	const seqs = taxa.map((t) => {
		const s = get(t);
		if (typeof s !== 'string') throw new Error(`tn93DistanceMatrix: no sequence for taxon '${t}'`);
		return s;
	});
	// The engine (the compiled tn93 through WebAssembly, dataset.py:740-784's branch when the binary
	// is on PATH) supplies the RAW pairwise distances; everything downstream of them stays here, so
	// two engines can differ only in the numbers they compute, never in what is done with them.
	const provider = options.pairwiseDistances(seqs, taxa, options.threshold ?? TN93_REPORTING_THRESHOLD);

	let maxD = 0;
	for (let i = 0; i < n; i++) {
		for (let j = i + 1; j < n; j++) {
			let d = provider[i * n + j];
			if (d === null || d === undefined) continue; // never written; left at the sentinel
			// dataset.py:805 `if d is None or d == "-" or d < 0 or np.isnan(d): d = 1.0`, and
			// dataset.py:795-799's `except (ValueError, OverflowError): d = 1.0`. An engine that reports
			// a degenerate pair as a negative or a NaN lands here rather than raising.
			if (d < 0 || Number.isNaN(d)) d = TN93_SATURATION_SENTINEL;
			const f = Math.fround(d);
			dist[i * n + j] = f;
			dist[j * n + i] = f;
			if (f > maxD) maxD = f;
		}
	}

	// dataset.py:816-821: the diagonal is zeroed first (it already is), `missing` is every entry still
	// below zero, and `max_d` is read ONCE from the filled matrix.
	const fill = Math.fround(Math.max(TN93_SATURATION_SENTINEL, maxD > 0 ? maxD : TN93_FALLBACK_MAX));
	for (let i = 0; i < n; i++) {
		for (let j = i + 1; j < n; j++) {
			if (dist[i * n + j] < 0.0) {
				dist[i * n + j] = fill;
				dist[j * n + i] = fill;
			}
		}
	}
	for (let i = 0; i < n; i++) dist[i * n + i] = 0.0;
	return dist;
}

/**
 * `compute_tn93_cross_distance_matrix(seq_dict, taxa_all, taxa_landmarks)`, dataset.py:824-928:
 * the RECTANGULAR float32 [n, m] matrix of every taxon against a handful of landmarks, row-major.
 * The tree-free dating path (dating.py:624-698) computes every root-to-tip divergence with it —
 * one landmark, the root sequence — so this is the shape that pillar actually uses, not the square
 * sibling above.
 *
 * Everything downstream of the numbers is the reference's and is done here: the `-1.0` prefill so a
 * pair that was never written is distinguishable from a measured zero, the float32 rounding,
 * `d < 0 or isnan -> 1.0`, the landmark self-zeros (set BEFORE `max_d` is read, as dataset.py:917
 * does, and again after the imputation), and the `max(1.0, max_d)` fill. An EMPTY axis returns the
 * all-sentinel matrix with no imputation at all, exactly as dataset.py:836 does.
 *
 * THE `*` QUESTION IS THE CALLER'S, AND IT IS NOT COSMETIC. dataset.py rewrites `*` to `-` only in
 * the branch that writes FASTA for the compiled `tn93` binary (dataset.py:840, 844); the Python
 * package branch passes `*` through, where the package's character map sends it to the catch-all
 * unknown. The two engines therefore disagree on any alignment containing `*`. MEASURED on
 * examples/korber_env_gp160.fasta (2389 asterisks across 18 of 143 sequences): with `*` rewritten to
 * `-` before this call, all 142 root divergences reproduce the reference's compiled-binary run
 * EXACTLY (max |Δ| = 0.0); without it, 18 of them differ and `Z59ZR.ZHU` moves from 0.06076440 to
 * 0.11056680. The library rewrites nothing; the application decides, per pillar, which convention
 * its reference-of-record used, and records the choice in the run's provenance.
 *
 * @param {Map<string, string>|Record<string, string>} sequences taxon -> aligned sequence
 * @param {string[]} taxaAll the rows, in order
 * @param {string[]} taxaLandmarks the columns, in order
 * @param {{threshold?: number,
 *   pairwiseDistances: (allSeqs: string[], landmarkSeqs: string[], taxaAll: string[], taxaLandmarks: string[], threshold: number) => ArrayLike<number>}} options
 *   `pairwiseDistances` is REQUIRED and supplies the RAW distances, read at [i * m + j];
 *   `null`/`undefined` for a pair means "not written" and is imputed below, exactly as a pair the
 *   binary omitted from its CSV leaves dataset.py's matrix at -1.0.
 * @returns {Float32Array} length taxaAll.length * taxaLandmarks.length, row-major
 * @throws {Tn93EngineRequiredError} when `pairwiseDistances` is absent
 */
export function tn93CrossDistanceMatrix(sequences, taxaAll, taxaLandmarks, options = /** @type {any} */ ({})) {
	// Unconditional, and before the empty-axis shortcut, for the reason given on the square sibling.
	if (typeof options.pairwiseDistances !== 'function') throw new Tn93EngineRequiredError('tn93CrossDistanceMatrix');

	const get = sequences instanceof Map ? (/** @type {string} */ t) => sequences.get(t) : (/** @type {string} */ t) => sequences[t];
	const n = taxaAll.length;
	const m = taxaLandmarks.length;
	const dist = new Float32Array(n * m);
	// dataset.py:834 `np.full((n, m), -1.0, dtype=np.float32)`.
	dist.fill(TN93_MISSING_SENTINEL);
	// dataset.py:835-836: an empty axis returns the prefilled matrix, imputation not reached.
	if (n === 0 || m === 0) return dist;

	const read = (/** @type {string} */ t) => {
		const s = get(t);
		if (typeof s !== 'string') throw new Error(`tn93CrossDistanceMatrix: no sequence for taxon '${t}'`);
		return s;
	};
	const allSeqs = taxaAll.map(read);
	const lmSeqs = taxaLandmarks.map(read);
	const provider = options.pairwiseDistances(allSeqs, lmSeqs, taxaAll, taxaLandmarks, options.threshold ?? TN93_REPORTING_THRESHOLD);

	// dataset.py:893-914, landmark-major: `for j, lm: for i, t:`.
	for (let j = 0; j < m; j++) {
		for (let i = 0; i < n; i++) {
			if (taxaAll[i] === taxaLandmarks[j]) {
				dist[i * m + j] = 0.0;
				continue;
			}
			let d = provider[i * m + j];
			if (d === null || d === undefined) continue; // never written; left at the sentinel
			// dataset.py:903-906's `except (ValueError, OverflowError): d = 1.0` covered a raising
			// engine; an engine that returns a negative or a NaN for the same pair lands here.
			if (d < 0 || Number.isNaN(d)) d = TN93_SATURATION_SENTINEL;
			dist[i * m + j] = Math.fround(d);
		}
	}

	// dataset.py:917-919, then :921-924, then :925-927. `taxa_idx` is built from taxa_all, so a name
	// repeated there resolves to its LAST row — replicated by scanning forwards and keeping the last.
	const rowOf = new Map();
	for (let i = 0; i < n; i++) rowOf.set(taxaAll[i], i);
	for (let j = 0; j < m; j++) {
		const i = rowOf.get(taxaLandmarks[j]);
		if (i !== undefined) dist[i * m + j] = 0.0;
	}
	let maxD = -Infinity;
	for (let k = 0; k < dist.length; k++) if (dist[k] > maxD) maxD = dist[k];
	const fill = Math.fround(Math.max(TN93_SATURATION_SENTINEL, maxD > 0 ? maxD : TN93_FALLBACK_MAX));
	for (let k = 0; k < dist.length; k++) if (dist[k] < 0.0) dist[k] = fill;
	for (let j = 0; j < m; j++) {
		const i = rowOf.get(taxaLandmarks[j]);
		if (i !== undefined) dist[i * m + j] = 0.0;
	}
	return dist;
}

/**
 * How many unordered pairs sit exactly at the saturation sentinel (PLAN.md D22's report line, and
 * diagnostics.js's TN93_SATURATED_PAIRS). Equality is exact: such a pair was not measured — it comes
 * from the engine's own saturation answer or from the `d < 0 or isnan` guard above.
 *
 * CAVEAT since the reference moved to the -1.0 missing sentinel: an entry that was never written is
 * imputed to `max(1.0, max_d)`, which equals this sentinel only while nothing in the matrix exceeds
 * 1.0. An engine with a reporting cutoff — and the reference's cutoff is now 100.0, so measured
 * distances above 1.0 are reported rather than dropped — can leave imputed entries this does not
 * distinguish from saturated ones. Pass an explicit `sentinel` in that case.
 *
 * @param {ArrayLike<number>} dist row-major n x n
 * @param {number} n
 * @param {number} [sentinel]
 * @returns {number}
 */
export function tn93SaturatedPairs(dist, n, sentinel = TN93_SATURATION_SENTINEL) {
	let count = 0;
	for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (dist[i * n + j] === sentinel) count++;
	return count;
}
