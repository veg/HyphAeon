/**
 * tn93.test.js — the tree-free TN93 path (PLAN.md D22): src/preprocess/tn93.js and the tree-free
 * branch of src/preprocess/assemble.js.
 *
 * WHY THIS FILE EXISTS, AND WHAT CHANGED ON 2026-09-13
 *
 * It used to prove that the library's JavaScript port of the `tn93` 1.2.2 package reproduced Python
 * bit for bit: the per-pair distance over every IUPAC code, the 4x4 counts, the nucleotide
 * frequencies, the four match modes, the two places Python raises. THAT PORT IS DELETED. The
 * compiled veg/tn93 engine is now the only implementation and the library never computes a distance
 * (src/preprocess/tn93.js's header carries the decision). So those assertions have no subject left
 * in this repository, and this file no longer makes them. Read the blocks below for what each one
 * now proves; read `lost` in the change that deleted the port for what nothing proves any more.
 *
 * WHAT IS STILL THE LIBRARY'S, AND IS STILL PROVED HERE against the same Python fixtures, by
 * injecting the fixture's OWN distances as the engine's answer (test/helpers/tn93-engine.js):
 *
 *   dataset/tn93_distance.json          23 pairs Python scored. The ARITHMETIC is no longer
 *                                       checked. The distances are replayed through the wrapping
 *                                       instead — float32 rounding, symmetry, the zeroed diagonal —
 *                                       and the four cases where Python RAISED now pin what the
 *                                       library does when an engine reports such a pair as a
 *                                       negative, a NaN or a hole.
 *   dataset/tn93_distance_matrix.json   whole matrices for bat_oas1 (18 taxa), Smc6 (20) and
 *                                       camelid (212), plus the three imputation rules: the
 *                                       fixture's matrix goes in as raw engine output and must come
 *                                       back unchanged, with `max` and the saturated-pair count as
 *                                       the fixture records them (1e-9)
 *   dataset/load_alignment_and_tree_tn93.json   the whole tree-free assembly for bat_oas1 and
 *                                       camelid on the fixture's own distances: taxa, L, tokens,
 *                                       MDS, invariable mask, notices — every one of which is still
 *                                       the library's own computation
 *   e2e/*_tn93.json                     what the CLI's own `--use-tn93` runs report about the
 *                                       assembly (taxon and codon counts, per-site invariability).
 *                                       None of those depend on the distances, which is why a test
 *                                       double is honest there and is labelled as one.
 *
 * MDS coordinates are compared per column EXACTLY under the canonical sign convention at the 1e-5
 * class, the same rule fixtures.test.js applies (measured: 3.7e-8 on bat_oas1, 7.8e-8 on camelid,
 * from the float32 eigensolver).
 *
 * WHAT IT DOES NOT DO: no model, and now no arithmetic. The LRTs, attributions and sectors in the
 * e2e fixtures come from the neural graph, which the library never runs; the app's runtime tests own
 * those, and the app's parity gate — which runs the compiled engine against the Python reference on
 * whole alignments — is where TN93 numbers are now checked at all.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { dirname, join, relative } from 'node:path';

import {
	tn93DistanceMatrix,
	tn93CrossDistanceMatrix,
	tn93SaturatedPairs,
	Tn93EngineRequiredError,
	TN93_MATCH_MODE,
	TN93_MAX_AMBIG_FRACTION,
	TN93_SATURATION_SENTINEL,
	TN93_MIN_POSITIVE_DISTANCE,
	TN93_FALLBACK_MAX
} from '../src/preprocess/tn93.js';
import * as tn93Module from '../src/preprocess/tn93.js';
import { parseAlignmentSequences } from '../src/preprocess/parse.js';
import { loadAlignmentAndTree } from '../src/preprocess/assemble.js';
import { fixtureSquareEngine, stubEngine, stubTn93Options } from './helpers/tn93-engine.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const FIXTURES = join(ROOT, 'fixtures');
const EXAMPLES = join(ROOT, 'examples');
const loadJson = (rel) => JSON.parse(readFileSync(join(FIXTURES, rel), 'utf8'));
const readExample = (name) => readFileSync(join(EXAMPLES, name), 'utf8');
const tol = (c) => (c.tolerance === 'exact' ? 0 : Number(c.tolerance));

/** An engine that answers one pair, for the two-taxon matrices the per-pair block builds. */
const onePairEngine = (d) => () => Float64Array.from([0, d, d, 0]);

/** float32 machine epsilon; the reference's eigendecomposition runs in float32 (LAPACK ssyevd). */
const EPS32 = 2 ** -23;

function maxAbsDiff(a, b) {
	expect(a.length).toBe(b.length);
	let worst = 0;
	for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
	return worst;
}

/**
 * Per-column comparison of MDS coordinates, canonical signs on both sides and no sign-flip
 * allowance, with the float32 noise floor of fixtures.test.js: a component whose magnitude is below
 * sqrt(eps32 * max Gram diagonal) is rounding noise and only has to stay there.
 */
function checkMdsColumns(name, jsFlat, pyRows, n, k, tolerance) {
	let gramMax = 0;
	for (const row of pyRows) gramMax = Math.max(gramMax, row.reduce((s, v) => s + v * v, 0));
	const noiseFloor = Math.sqrt(EPS32 * gramMax);
	for (let col = 0; col < k; col++) {
		let same = 0;
		let pyMag = 0;
		let jsMag = 0;
		for (let i = 0; i < n; i++) {
			same = Math.max(same, Math.abs(jsFlat[i * k + col] - pyRows[i][col]));
			pyMag = Math.max(pyMag, Math.abs(pyRows[i][col]));
			jsMag = Math.max(jsMag, Math.abs(jsFlat[i * k + col]));
		}
		if (pyMag < noiseFloor) {
			expect(jsMag, `${name} column ${col} below the float32 noise floor`).toBeLessThan(noiseFloor);
		} else {
			expect(same, `${name} column ${col} exact (canonical sign)`).toBeLessThanOrEqual(tolerance * Math.max(1, pyMag));
		}
	}
}

/** The fixtures pin token arrays by the sha256 of their row-major values joined with ','. */
const tokenSha = (arr) => createHash('sha256').update(Array.from(arr).join(',')).digest('hex');

describe('the port is gone, and the engine is mandatory', () => {
	// The list in the deletion's release notes. A name reappearing here means a second TN93
	// implementation has come back into @veg/hyphaeon-js, which is the thing that may not happen.
	const REMOVED = [
		'encodeSequence',
		'canResolve',
		'ambigFractionTooHigh',
		'tn93Counts',
		'tn93NucleotideFrequency',
		'tn93CalculateDistance',
		'tn93Distance',
		'TN93_TABLES'
	];

	it('exports no per-pair arithmetic any more', () => {
		for (const name of REMOVED) expect(tn93Module[name], `${name} must stay deleted`).toBeUndefined();
		// What remains is the wrapping, the reference's constants and the refusal.
		expect(Object.keys(tn93Module).sort()).toEqual(
			[
				'TN93_FALLBACK_MAX',
				'TN93_MATCH_MODE',
				'TN93_MAX_AMBIG_FRACTION',
				'TN93_MIN_POSITIVE_DISTANCE',
				'TN93_SATURATION_SENTINEL',
				'Tn93EngineRequiredError',
				'tn93CrossDistanceMatrix',
				'tn93DistanceMatrix',
				'tn93SaturatedPairs'
			].sort()
		);
	});

	it('has no TN93 arithmetic left in the source, exported or not', () => {
		// The export list above catches a port that comes BACK as an export; this catches one that
		// comes back module-private. Two primitives no TN93 implementation can do without: reading a
		// sequence character (the 256-entry map_character) and taking a logarithm (tn93.py:203/219).
		// Neither has any other business in a file that only rounds and imputes numbers.
		const src = readFileSync(join(ROOT, 'js', 'src', 'preprocess', 'tn93.js'), 'utf8');
		for (const primitive of ['charCodeAt', 'Math.log']) {
			expect(src, `${primitive} means a distance is being computed here again`).not.toContain(primitive);
		}
	});

	it('has no TN93 arithmetic anywhere else in the library either', () => {
		// THE GUARD ABOVE IS FILE-SCOPED, and a port that reappeared in a NEW module — a
		// `preprocess/tn93-fallback.js`, say — would pass it while putting a second implementation
		// back in the tree. That is the exact thing the deletion exists to prevent, so the guard is
		// tree-scoped here: every source file is read, and any that mentions TN93 must not also
		// carry the arithmetic's own primitives.
		//
		// THE PREDICATE IS BOTH PRIMITIVES TOGETHER, and the first draft of this test got it wrong:
		// "names tn93 AND takes a logarithm" flagged dating.js and datingModel.js, which call the
		// matrix functions and take logarithms for the clock, not for a distance. Measured across
		// js/src: `Math.log` appears in 2 files that are pure clock maths and `charCodeAt` in
		// exactly one (writers.js, escaping output), and NO file has both. A TN93 implementation
		// needs both — it reads sequence characters through the 256-entry map_character and takes a
		// logarithm (tn93.py:203/219) — so requiring the pair is what separates a port from the
		// arithmetic this library legitimately does. The allow-list is empty on purpose, so adding
		// to it is a decision someone has to make in a diff.
		const ALLOWED = new Set();
		const offenders = [];
		/** @param {string} dir */
		const walk = (dir) => {
			for (const entry of readdirSync(dir, { withFileTypes: true })) {
				const full = join(dir, entry.name);
				if (entry.isDirectory()) {
					walk(full);
				} else if (entry.name.endsWith('.js')) {
					const rel = relative(join(ROOT, 'js', 'src'), full);
					if (ALLOWED.has(rel)) continue;
					const text = readFileSync(full, 'utf8');
					if (text.includes('charCodeAt') && text.includes('Math.log')) offenders.push(rel);
				}
			}
		};
		walk(join(ROOT, 'js', 'src'));
		expect(
			offenders,
			`these files both read sequence characters and take a logarithm, which is what a TN93 implementation looks like: ${offenders.join(', ')}`
		).toEqual([]);
	});

	it('refuses both matrix shapes without an engine, at EVERY size', () => {
		const seqs = { a: 'ATGC', b: 'ATGG', c: 'ATTG' };
		// One taxon would have short-circuited before any distance was needed; it refuses anyway,
		// because a rule that let small inputs through is the size-based selector this library may
		// not have.
		for (const taxa of [['a'], ['a', 'b'], ['a', 'b', 'c']]) {
			expect(() => tn93DistanceMatrix(seqs, taxa)).toThrow(Tn93EngineRequiredError);
		}
		// And the rectangular sibling, including the empty axis that returns early.
		expect(() => tn93CrossDistanceMatrix(seqs, [], [])).toThrow(Tn93EngineRequiredError);
		expect(() => tn93CrossDistanceMatrix(seqs, ['a', 'b'], ['a'])).toThrow(Tn93EngineRequiredError);
	});

	it('names the option and the engine rather than failing as a TypeError', () => {
		let err;
		try {
			tn93DistanceMatrix({ a: 'ATGC', b: 'ATGG' }, ['a', 'b']);
		} catch (e) {
			err = e;
		}
		expect(err).toBeInstanceOf(Tn93EngineRequiredError);
		expect(err).toBeInstanceOf(Error);
		expect(err.name).toBe('Tn93EngineRequiredError');
		expect(err.code).toBe('TN93_ENGINE_REQUIRED');
		expect(err.option).toBe('pairwiseDistances');
		expect(err.caller).toBe('tn93DistanceMatrix');
		expect(err.message).toContain('pairwiseDistances');
		expect(err.message).toContain('veg/tn93');
		// A non-function under the key is the same refusal, not a crash inside the loop.
		expect(() => tn93DistanceMatrix({ a: 'A' }, ['a'], /** @type {any} */ ({ pairwiseDistances: [] }))).toThrow(Tn93EngineRequiredError);
	});

	it('still pins the engine settings the reference asks for', () => {
		// These are not an implementation; they are the argv the caller must run the engine with.
		expect(TN93_MATCH_MODE).toBe('resolve');
		expect(TN93_MAX_AMBIG_FRACTION).toBe(1.0);
		expect(TN93_SATURATION_SENTINEL).toBe(1.0);
		expect(TN93_FALLBACK_MAX).toBe(1.0);
		expect(TN93_MIN_POSITIVE_DISTANCE).toBe(1e-4);
	});
});

describe('dataset/tn93_distance.json: Python\'s distances through the wrapping', () => {
	// THE ARITHMETIC IS NOT CHECKED HERE ANY MORE. What each case still gives is a real number the
	// Python reference produced for a real pair of sequences, and the assertion is that the library
	// carries it into a matrix unchanged: float32 rounding, both triangles, a zeroed diagonal.
	const cases = loadJson('dataset/tn93_distance.json');

	it('replays every case', () => {
		expect(cases.length).toBeGreaterThan(20);
	});

	for (const c of cases.filter((x) => !x.outputs.error)) {
		it(`${c.name}: ${c.outputs.distance} survives the matrix wrapping`, () => {
			const taxa = ['seq1', 'seq2'];
			const seqs = { seq1: c.inputs.seq1, seq2: c.inputs.seq2 };
			const D = tn93DistanceMatrix(seqs, taxa, { pairwiseDistances: onePairEngine(c.outputs.distance) });
			const expected = Math.fround(c.outputs.distance);
			expect(D[1]).toBe(expected);
			expect(D[2]).toBe(expected);
			expect(D[0]).toBe(0);
			expect(D[3]).toBe(0);
			// float64 -> float32 is the ONLY transformation on a measured value (dataset.py:732's dtype).
			expect(Math.abs(D[1] - c.outputs.distance)).toBeLessThanOrEqual(Math.max(tol(c), Math.abs(expected - c.outputs.distance)));
		});
	}

	// The four cases where the Python package RAISED (ZeroDivisionError on a pair with no overlap,
	// math.log's ValueError on a saturated one). The library cannot raise them any more — it has no
	// arithmetic to raise from — so what is pinned is what it does with an engine's answer for such a
	// pair, which is where dataset.py's own dead guard becomes live code.
	const raising = cases.filter((x) => x.outputs.error);
	it('has the degenerate pairs Python raised on', () => {
		expect(raising.map((c) => c.name).sort()).toEqual(['all_N_raises', 'all_gap_raises', 'no_overlap_raises', 'saturated_raises']);
	});

	for (const c of raising) {
		it(`${c.name}: an engine's negative, NaN or omission all become the sentinel`, () => {
			const seqs = { seq1: c.inputs.seq1, seq2: c.inputs.seq2 };
			const taxa = ['seq1', 'seq2'];
			// dataset.py:805 `if d is None or d == "-" or d < 0 or np.isnan(d): d = 1.0`.
			for (const reported of [-1e-9, -5, NaN]) {
				const D = tn93DistanceMatrix(seqs, taxa, { pairwiseDistances: onePairEngine(reported) });
				expect(D[1], `${reported} -> sentinel`).toBe(TN93_SATURATION_SENTINEL);
			}
			// A pair the engine simply did not write (above its reporting threshold, or below its
			// minimum overlap) is a HOLE, not a saturation answer, and dataset.py:816-821 imputes it.
			// With nothing else in the matrix the fill is max(1.0, max_d) = 1.0, which lands on the
			// same number by coincidence of the fill rule, not by the same route.
			const holed = tn93DistanceMatrix(seqs, taxa, { pairwiseDistances: () => [0, undefined, undefined, 0] });
			expect(holed[1]).toBe(Math.fround(TN93_FALLBACK_MAX));
			expect(tn93SaturatedPairs(holed, 2)).toBe(1);
		});
	}
});

describe('dataset/tn93_distance_matrix.json', () => {
	const cases = loadJson('dataset/tn93_distance_matrix.json');

	// The `_reduced` case records only two rows of a 476 x 476 matrix, so its interior cannot be fed
	// back to an engine and its `max`, `min_off_diagonal` and saturated-pair count cannot be
	// reproduced. Those three assertions were about the deleted arithmetic; camelid below is a 212 x
	// 212 matrix replayed in full, so the plumbing at scale is still covered. See the header.
	for (const c of cases.filter((x) => !x.name.endsWith('_reduced'))) {
		it(c.name, () => {
			let seqs;
			let taxa;
			if (c.inputs.alignment) {
				seqs = parseAlignmentSequences(readExample(c.inputs.alignment));
				taxa = c.inputs.taxa;
				// The tree-free path takes list(seq_dict.keys()); the fixture records that order.
				expect(taxa).toEqual(Array.from(seqs.keys()));
			} else {
				seqs = new Map(Object.entries(c.inputs.seq_dict));
				taxa = c.inputs.taxa;
			}
			const n = taxa.length;
			// Python's own matrix goes in as the engine's raw answer and must come back untouched.
			const D = tn93DistanceMatrix(seqs, taxa, { pairwiseDistances: fixtureSquareEngine(taxa, c.outputs.dist_matrix) });
			let max = 0;
			for (const v of D) if (v > max) max = v;
			expect(maxAbsDiff(D, c.outputs.dist_matrix.flat())).toBeLessThanOrEqual(tol(c));
			expect(Math.abs(max - c.outputs.max)).toBeLessThanOrEqual(tol(c));
			if (c.outputs['saturated_pairs_at_1.0'] !== undefined) {
				expect(tn93SaturatedPairs(D, n)).toBe(c.outputs['saturated_pairs_at_1.0']);
			}
			// The matrix is symmetric with a zero diagonal, on every case.
			for (let i = 0; i < n; i++) {
				expect(D[i * n + i]).toBe(0);
				for (let j = 0; j < n; j++) expect(D[i * n + j]).toBe(D[j * n + i]);
			}
		}, 60000);
	}

	it('reads the reduced HIV1_RT case and states what can no longer be replayed from it', () => {
		const c = cases.find((x) => x.name.endsWith('_reduced'));
		expect(c.outputs.n).toBe(476);
		// Only two of the 476 rows were recorded, which was enough while the library computed the
		// interior itself. It no longer does, so the two recorded rows are all there is to replay:
		// feed them, mark every other pair unwritten, and check the wrapping does both jobs.
		const n = c.outputs.n;
		const taxa = c.inputs.taxa;
		const seqs = parseAlignmentSequences(readExample(c.inputs.alignment));
		const first = c.outputs.first_row;
		const last = c.outputs.last_row;
		const D = tn93DistanceMatrix(seqs, taxa, {
			pairwiseDistances: () => {
				// The wrapper reads the UPPER triangle only, [i * n + j] with i < j, so the last row's
				// pairs have to be written at [j * n + (n - 1)]; it mirrors them back itself.
				const out = new Array(n * n).fill(undefined);
				for (let j = 1; j < n; j++) out[j] = first[j];
				for (let j = 1; j < n - 1; j++) out[j * n + (n - 1)] = last[j];
				return out;
			}
		});
		// The recorded rows survive exactly...
		for (let j = 1; j < n; j++) expect(D[j], `first row ${j}`).toBe(Math.fround(first[j]));
		for (let j = 0; j < n - 1; j++) expect(D[(n - 1) * n + j], `last row ${j}`).toBe(Math.fround(last[j]));
		// ...and every pair the engine did not write is imputed to max(1.0, max_d), which here is 1.0
		// because the recorded rows top out at 0.169 (dataset.py:816-821).
		expect(c.outputs.max).toBeLessThan(1.0);
		expect(D[1 * n + 2]).toBe(Math.fround(TN93_FALLBACK_MAX));
		expect(tn93SaturatedPairs(D, n)).toBe((n * (n - 1)) / 2 - (2 * (n - 1) - 1));
	}, 60000);

	it('reproduces the reference imputation rule the synthetic cases pin', () => {
		const byName = Object.fromEntries(cases.map((c) => [c.name, c]));
		// A MEASURED zero survives, whichever kind of pair produced it. The rule these cases pinned
		// before raised the first to 1e-4 and the second to max(1.0, max_d); the reference moved the
		// "not written" marker to -1.0, so a measured zero is no longer mistaken for a missing entry.
		expect(byName.distinct_sequences_at_zero_stay_zero.outputs.dist_matrix[0][1]).toBe(0);
		const identical = byName.identical_strings_stay_zero.outputs;
		expect(identical.dist_matrix[0][1]).toBe(0);
		expect(identical.max).toBe(identical.dist_matrix[0][2]);
		// All identical -> every pair is written and measures 0, so nothing is imputed.
		expect(byName.all_identical_stays_zero.outputs.max).toBe(0);
		// The retired floor is still exported, and nothing reaches it any more.
		expect(TN93_MIN_POSITIVE_DISTANCE).toBe(1e-4);
		for (const c of cases) {
			const m = c.outputs.dist_matrix;
			if (!m) continue;
			for (const row of m) for (const v of row) expect(v).not.toBe(Math.fround(TN93_MIN_POSITIVE_DISTANCE));
		}
	});

	it('imputes only the holes, and reads max_d once from the filled matrix', () => {
		// The rule in isolation, which no fixture case can reach any more now that the engine decides
		// what is written: two measured pairs and one hole, the hole filled with max(1.0, max_d).
		const seqs = { a: 'ATGC', b: 'ATGG', c: 'ATTG' };
		const taxa = ['a', 'b', 'c'];
		const D = tn93DistanceMatrix(seqs, taxa, {
			// [a,b] = 2.5, [a,c] unwritten, [b,c] = 0.25
			pairwiseDistances: () => [0, 2.5, undefined, 2.5, 0, 0.25, undefined, 0.25, 0]
		});
		expect(D[1]).toBe(2.5);
		expect(D[5]).toBe(0.25);
		expect(D[2]).toBe(2.5); // max(1.0, max_d) with max_d = 2.5
		expect(D[6]).toBe(2.5);
		expect(D[0]).toBe(0);
		// A matrix whose only measurements are zero falls back to 1.0 (dataset.py:818).
		const allZero = tn93DistanceMatrix(seqs, taxa, { pairwiseDistances: () => [0, 0, undefined, 0, 0, 0, undefined, 0, 0] });
		expect(allZero[2]).toBe(Math.fround(TN93_FALLBACK_MAX));
	});
});

describe('dataset/load_alignment_and_tree_tn93.json', () => {
	const cases = loadJson('dataset/load_alignment_and_tree_tn93.json');

	for (const c of cases) {
		it(`${c.name}: the whole tree-free assembly`, () => {
			// Python's distances are the engine's answer; everything the assembly then does with them
			// — MDS, tokens, the invariable mask, the notices — is the library's own and is compared
			// against Python exactly as before.
			const r = loadAlignmentAndTree(readExample(c.inputs.alignment), null, {
				useTn93: c.inputs.use_tn93,
				maxSpecies: c.inputs.max_species,
				pruneDuplicates: c.inputs.prune_duplicates,
				tn93Options: { pairwiseDistances: fixtureSquareEngine(c.outputs.taxa, c.outputs.dist_matrix) }
			});
			// Exact: taxa (alignment order), L, N, the invariable mask, the token arrays.
			expect(r.taxa).toEqual(c.outputs.taxa);
			expect(r.L).toBe(c.outputs.L);
			expect(r.N).toBe(c.outputs.N);
			expect(Array.from(r.invariable)).toEqual(c.outputs.is_aa_invariable.map((v) => (v ? 1 : 0)));
			expect(Array.from(r.invariable).reduce((a, b) => a + b, 0)).toBe(c.outputs.n_invariable);
			expect(tokenSha(r.c)).toBe(c.outputs.codon_tokens_sha256);
			expect(tokenSha(r.a)).toBe(c.outputs.aa_tokens_sha256);
			if (c.outputs.codon_tokens) {
				expect(maxAbsDiff(r.c, c.outputs.codon_tokens.flat())).toBe(0);
				expect(maxAbsDiff(r.a, c.outputs.aa_tokens.flat())).toBe(0);
			}
			// Distances: 1e-9 (float32 on both sides).
			expect(maxAbsDiff(r.d, c.outputs.dist_matrix.flat())).toBeLessThanOrEqual(1e-9);
			expect(tn93SaturatedPairs(r.d, r.N)).toBe(c.outputs['saturated_pairs_at_1.0']);
			expect(r.notices.tn93SaturatedPairs).toBe(c.outputs['saturated_pairs_at_1.0']);
			// MDS at the 1e-5 class, per column, canonical signs.
			checkMdsColumns(c.name, r.z, c.outputs.mds_coords, r.N, 4, tol(c));
			if (c.outputs.mds_gram) {
				const gram = new Float64Array(r.N * r.N);
				for (let i = 0; i < r.N; i++)
					for (let j = 0; j < r.N; j++) {
						let sum = 0;
						for (let k = 0; k < 4; k++) sum += r.z[i * 4 + k] * r.z[j * 4 + k];
						gram[i * r.N + j] = sum;
					}
				expect(maxAbsDiff(gram, c.outputs.mds_gram.flat())).toBeLessThanOrEqual(tol(c) * 100);
			}
			// The path is the tree-free one, with no rescale and no taxon matching.
			expect(r.notices.treeFree).toEqual({ reason: 'requested', taxaOrder: 'alignment' });
			expect(r.notices.matchTier).toBeNull();
			expect(r.notices.distanceRescaled).toBe(false);
			expect(r.tree).toBeNull();
		}, 60000);
	}
});

describe('e2e/*_tn93.json: what the CLI reports about the tree-free assembly', () => {
	/** @type {Array<[string, string, {taxa: string, sites: string}]>} */
	const cases = [
		['meme_camelid_tn93.json', 'camelid.fasta', { taxa: 'taxa_count', sites: 'codon_count' }],
		['meme_HIV1_RT_tn93.json', 'HIV1_RT.fasta', { taxa: 'taxa_count', sites: 'codon_count' }],
		['busted_Smc6_tn93.json', 'Smc6.fasta', { taxa: 'taxa', sites: 'sites' }],
		['epistasis_Smc6_tn93.json', 'Smc6.fasta', { taxa: 'taxa_count', sites: 'codon_count' }]
	];

	for (const [file, alignment, keys] of cases) {
		it(`${file}: taxon and codon counts`, () => {
			const fx = loadJson(join('e2e', file))[0];
			expect(fx.inputs.argv).toContain('--use-tn93');
			expect(fx.inputs.argv).toContain('canonical');
			// EVERY assertion in this block is distance-independent: N comes from parsing, duplicate
			// pruning and the (absent) max_species cap, L from the sequence length, and invariability
			// from the amino-acid tokens. No downsampling runs, so no distance reaches any of them.
			// A test double is therefore honest here; it is not a TN93 and claims nothing.
			const r = loadAlignmentAndTree(readExample(alignment), null, { useTn93: true, tn93Options: stubTn93Options() });
			expect(r.N).toBe(fx.outputs[keys.taxa]);
			expect(r.L).toBe(fx.outputs[keys.sites]);
			expect(r.notices.treeFree.reason).toBe('requested');
			if (fx.outputs.evaluated_taxa !== undefined) expect(fx.outputs.evaluated_taxa).toBe(r.N);
			// The MEME writers carry per-site invariability, which is assembly output, not model
			// output: it must match site for site.
			if (Array.isArray(fx.outputs.sites)) {
				expect(fx.outputs.sites.map((s) => (s.is_invariable ? 1 : 0))).toEqual(Array.from(r.invariable));
				expect(fx.outputs.sites.map((s) => s.site)).toEqual(Array.from({ length: r.L }, (_, i) => i + 1));
			}
		}, 120000);
	}
});

describe('tn93SaturatedPairs', () => {
	it('counts only pairs sitting exactly at the sentinel', () => {
		const d = Float32Array.from([0, 1, 0.5, 1, 0, 1.5, 0.5, 1.5, 0]);
		expect(tn93SaturatedPairs(d, 3)).toBe(1);
		expect(tn93SaturatedPairs(d, 3, 1.5)).toBe(1);
	});
});

describe('the square and rectangular wrappings around an engine', () => {
	it('returns the zero matrix for a single taxon, as dataset.py:735-736 does', () => {
		const engine = stubEngine();
		expect(Array.from(tn93DistanceMatrix(new Map([['a', 'ATGC']]), ['a'], { pairwiseDistances: engine }))).toEqual([0]);
		expect(Array.from(tn93DistanceMatrix({ a: 'ATGC' }, ['a'], { pairwiseDistances: engine }))).toEqual([0]);
	});

	it('hands the engine the reference threshold, and the caller can override it', () => {
		/** @type {number[]} */
		const seen = [];
		const spy = (/** @type {any[]} */ ...args) => {
			seen.push(args[args.length - 1]);
			return new Float64Array(4);
		};
		tn93DistanceMatrix({ a: 'AT', b: 'AG' }, ['a', 'b'], { pairwiseDistances: spy });
		tn93DistanceMatrix({ a: 'AT', b: 'AG' }, ['a', 'b'], { pairwiseDistances: spy, threshold: 1.0 });
		// dataset.py:719 `threshold: float = 100.0`.
		expect(seen).toEqual([100.0, 1.0]);
	});

	it('refuses a taxon it has no sequence for, before calling the engine', () => {
		let called = false;
		const engine = () => {
			called = true;
			return new Float64Array(4);
		};
		expect(() => tn93DistanceMatrix({ a: 'AT' }, ['a', 'b'], { pairwiseDistances: engine })).toThrow(/no sequence for taxon 'b'/);
		expect(called).toBe(false);
	});

	it('zeroes the landmark self-pairs and imputes the rest, as dataset.py:917-927 does', () => {
		const seqs = { a: 'ATGC', b: 'ATGG', c: 'ATTG' };
		const cross = tn93CrossDistanceMatrix(seqs, ['a', 'b', 'c'], ['a'], {
			pairwiseDistances: () => [undefined, 0.4, undefined]
		});
		expect(cross[0]).toBe(0); // the landmark against itself
		expect(cross[1]).toBe(Math.fround(0.4));
		expect(cross[2]).toBe(Math.fround(TN93_FALLBACK_MAX)); // unwritten -> max(1.0, max_d)
		// An empty axis returns the prefilled matrix with no imputation (dataset.py:835-836).
		expect(tn93CrossDistanceMatrix(seqs, ['a'], [], { pairwiseDistances: () => [] }).length).toBe(0);
	});
});

describe('the tree-free decision in loadAlignmentAndTree (D22)', () => {
	const fasta = readExample('bat_oas1.fasta');
	const tree = readExample('bat_oas1.nwk');
	// The decision is about TREES, not distances: which branch is taken, what the notices say, which
	// taxa survive. The engine only has to answer so the branch can finish.
	const tn93Options = stubTn93Options();

	it("reports 'requested' for useTn93 and for the tn93/none/skip tree modes", () => {
		for (const r of [
			loadAlignmentAndTree(fasta, tree, { useTn93: true, tn93Options }),
			loadAlignmentAndTree(fasta, 'tn93', { tn93Options }),
			loadAlignmentAndTree(fasta, ' NONE ', { tn93Options }),
			loadAlignmentAndTree(fasta, 'skip', { tn93Options })
		]) {
			expect(r.notices.treeFree.reason).toBe('requested');
			expect(r.notices.distanceRescaled).toBe(false);
		}
		// bat_oas1's tree is in Mya and would be rescaled; the TN93 matrix is not.
		expect(loadAlignmentAndTree(fasta, tree).notices.distanceRescaled).toBe(true);
	}, 30000);

	it("reports 'no_tree' when neither a tree text nor an embedded tree is there", () => {
		const noTree = loadAlignmentAndTree(fasta, null, { tn93Options });
		expect(noTree.notices.treeFree).toEqual({ reason: 'no_tree', taxaOrder: 'alignment' });
		expect(noTree.tree).toBeNull();
		expect(noTree.notices.branchLengthsMissing).toBe(false);
		expect(noTree.taxa).toEqual(Array.from(parseAlignmentSequences(fasta).keys()));
	}, 30000);

	it('keeps a topology-only tree for display but never lets it reach the model', () => {
		const names = Array.from(parseAlignmentSequences(fasta).keys());
		// Nested: extract_tree_from_string_or_file rejects a newick with a single '(' (dataset.py quirk).
		const topology = `((${names.slice(0, 2).join(',')}),${names.slice(2).join(',')});`;
		const r = loadAlignmentAndTree(fasta, topology, { tn93Options });
		expect(r.notices.treeFree).toEqual({ reason: 'no_branch_lengths', taxaOrder: 'alignment' });
		expect(r.notices.branchLengthsMissing).toBe(true);
		expect(r.tree).not.toBeNull();
		// Untouched by enforceNonzeroBranchLengths: the reference's 1e-3 / 1e-4 defaults never ran.
		expect(r.tree.branchLength.every((b) => b === null)).toBe(true);
		expect(r.taxa).toEqual(names);
	}, 30000);

	it('still raises on tree text that will not parse', () => {
		expect(() => loadAlignmentAndTree(fasta, 'not a tree at all', { tn93Options })).toThrow(/Could not parse phylogenetic tree/);
	});

	it('a tree-free load with no engine refuses, and says so as a wiring bug', () => {
		// The trap this change exists to close: before it, a tree-free input reached a matrix builder
		// that quietly ran a second implementation. Now it stops, naming the option.
		expect(() => loadAlignmentAndTree(fasta, null)).toThrow(Tn93EngineRequiredError);
		// A usable tree needs no engine at all: the patristic path never asks for one.
		expect(() => loadAlignmentAndTree(fasta, tree)).not.toThrow();
	}, 30000);
});
