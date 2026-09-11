/**
 * patristic.test.js — compute_fast_dist_matrix and the > 10 rescale, with hand-computed values.
 *
 * WHY THIS FILE EXISTS
 *
 * Every distance below is read off the Newick string in the test, not recorded from the code: an
 * expectation captured from the implementation proves only that it is deterministic. The
 * cross-implementation check is the fixture replay (Smc6, bat_oas1, the four rescale trees).
 *
 * Replaces DataMonkey 3's axomeme-patristic.test.js. What was dropped and why: the `parseNewick` /
 * `leafIndex` / `normalizeTaxonName` cases (DM3's newick.js is replaced by tree.js, which mirrors
 * Biopython; its parser cases live in preprocess-parse.test.js against Python-computed expectations);
 * the `maxPdSelect` cases (index-0 seeding is the AxoMEME 2.0 driver's rule, dataset.py seeds with
 * the most distant pair — see preprocess-downsample.test.js); "negative distances are preserved"
 * (dataset.py raises every branch to >= 1e-4 before the matrix exists, so the case cannot arise).
 */
import { describe, it, expect } from 'vitest';
import { readNewick, getTerminals, enforceNonzeroBranchLengths } from '../src/preprocess/tree.js';
import {
	rootDistances,
	patristicRow,
	patristicMatrix,
	computeFastDistMatrix,
	rescaleDistances
} from '../src/preprocess/patristic.js';

/** ((A:0.1,B:0.2):0.05,C:0.3); — the worked example used throughout. */
const SIMPLE = '((A:0.1,B:0.2):0.05,C:0.3);';

const leafOf = (tree, n) => getTerminals(tree).find((t) => tree.name[t] === n);

describe('rootDistances', () => {
	it('accumulates root distances down the tree', () => {
		const t = readNewick(SIMPLE);
		const d = rootDistances(t);
		expect(d[leafOf(t, 'A')]).toBeCloseTo(0.15, 12); // 0.05 + 0.1
		expect(d[leafOf(t, 'B')]).toBeCloseTo(0.25, 12); // 0.05 + 0.2
		expect(d[leafOf(t, 'C')]).toBeCloseTo(0.3, 12);
		expect(d[t.root]).toBe(0);
	});

	it('treats a missing branch length (null) as 0.0, matching `if ... is not None else 0.0`', () => {
		const t = readNewick('((A,B),C);');
		expect(t.branchLength.every((v) => v === null)).toBe(true);
		expect(Array.from(rootDistances(t)).every((v) => v === 0)).toBe(true);
	});

	it('ignores the root clade"s own branch length', () => {
		const t = readNewick('((A:0.1,B:0.2):0.05,C:0.3):9.0;');
		expect(t.branchLength[t.root]).toBe(9.0);
		expect(rootDistances(t)[leafOf(t, 'C')]).toBeCloseTo(0.3, 12);
	});
});

describe('patristicRow / patristicMatrix (float64)', () => {
	it('computes hand-checkable pairwise distances', () => {
		const t = readNewick(SIMPLE);
		const nodes = ['A', 'B', 'C'].map((n) => leafOf(t, n));
		const m = patristicMatrix(t, nodes);
		expect(m[0 * 3 + 1]).toBeCloseTo(0.3, 12); // A-B: 0.1 + 0.2
		expect(m[0 * 3 + 2]).toBeCloseTo(0.45, 12); // A-C: 0.1 + 0.05 + 0.3
		expect(m[1 * 3 + 2]).toBeCloseTo(0.55, 12); // B-C: 0.2 + 0.05 + 0.3
		expect(m[0]).toBe(0);
	});

	it('reuses its ancestor marker across rows without leaking marks', () => {
		const t = readNewick('(((A:0.1,B:0.2):0.05,C:0.3):0.01,(D:0.4,E:0.05):0.2);');
		const nodes = ['A', 'B', 'C', 'D', 'E'].map((n) => leafOf(t, n));
		const shared = patristicMatrix(t, nodes);
		const rd = rootDistances(t);
		for (let i = 0; i < nodes.length; i++) {
			const fresh = patristicRow(t, rd, nodes[i], nodes);
			for (let j = 0; j < nodes.length; j++) expect(shared[i * nodes.length + j]).toBeCloseTo(fresh[j], 12);
		}
	});

	it('handles a deep ladder tree without recursing', () => {
		const N = 3000;
		let s = 'L0:0.001';
		for (let i = 1; i < N; i++) s = `(${s},L${i}:0.001)`;
		const t = readNewick(s + ';');
		const terms = getTerminals(t);
		expect(terms).toHaveLength(N);
		const d = rootDistances(t);
		expect(d[leafOf(t, `L${N - 1}`)]).toBeCloseTo(0.001, 12);
		expect(Number.isFinite(d[leafOf(t, 'L0')])).toBe(true);
	});
});

describe('computeFastDistMatrix (dataset.py:524-574)', () => {
	it('returns the float32 matrix in the order of the taxa given', () => {
		const t = readNewick(SIMPLE);
		const d = computeFastDistMatrix(t, ['C', 'A', 'B']);
		expect(d).toBeInstanceOf(Float32Array);
		expect(d[0 * 3 + 1]).toBe(Math.fround(0.45)); // C-A
		expect(d[0 * 3 + 2]).toBe(Math.fround(0.55)); // C-B
		expect(d[1 * 3 + 2]).toBe(Math.fround(0.3)); // A-B
		for (let i = 0; i < 3; i++) {
			expect(d[i * 3 + i]).toBe(0);
			for (let j = 0; j < 3; j++) expect(d[i * 3 + j]).toBe(d[j * 3 + i]);
		}
	});

	it('leaves a ZERO row and column for a taxon with no terminal of that name (replicated, not repaired)', () => {
		const t = readNewick(SIMPLE);
		const d = computeFastDistMatrix(t, ['A', 'Z', 'C']);
		expect(Array.from(d)).toEqual([0, 0, Math.fround(0.45), 0, 0, 0, Math.fround(0.45), 0, 0]);
	});

	it('resolves a duplicate tip name to the LAST terminal in preorder (dict comprehension)', () => {
		const t = readNewick('((A:0.1,A:0.2):0.05,C:0.3);');
		const d = computeFastDistMatrix(t, ['A', 'C']);
		expect(d[1]).toBe(Math.fround(0.55)); // the second A: 0.2 + 0.05 + 0.3
	});

	it('matches quote-stripped terminal names', () => {
		const t = readNewick("(('A x':0.1,B:0.2):0.05,C:0.3);");
		const d = computeFastDistMatrix(t, ['A x', 'B']);
		expect(d[1]).toBe(Math.fround(0.3));
	});

	it('is float32 of the float64 (a + b) - 2c value', () => {
		const t = readNewick('((A:0.123456789,B:0.987654321):0.05,C:0.3);');
		const nodes = ['A', 'B', 'C'].map((n) => leafOf(t, n));
		const f64 = patristicMatrix(t, nodes);
		const f32 = computeFastDistMatrix(t, ['A', 'B', 'C']);
		for (let i = 0; i < 9; i++) expect(f32[i]).toBe(Math.fround(f64[i]));
	});

	it('sees the enforced branch lengths when run after enforceNonzeroBranchLengths', () => {
		const t = readNewick('((A,B),C);');
		enforceNonzeroBranchLengths(t, 1e-4);
		const d = computeFastDistMatrix(t, ['A', 'B', 'C']);
		expect(d[1]).toBe(Math.fround(0.002)); // A-B: 1e-3 + 1e-3
		expect(d[2]).toBe(Math.fround(0.003)); // A-C: 1e-3 + 1e-3 + 1e-3
	});
});

describe('rescaleDistances (dataset.py:1037-1040)', () => {
	it('divides by L when the maximum is strictly greater than 10', () => {
		const r = rescaleDistances(Float32Array.from([0, 12, 12, 0]), 5);
		expect(r.rescaled).toBe(true);
		expect(r.rawMax).toBe(12);
		expect(r.dist[1]).toBe(Math.fround(12 / 5));
	});

	it('does nothing at exactly 10.0', () => {
		const r = rescaleDistances(Float32Array.from([0, 10, 10, 0]), 5);
		expect(r.rescaled).toBe(false);
		expect(Array.from(r.dist)).toEqual([0, 10, 10, 0]);
	});

	it('fires when a zero branch raised to 1e-4 tips a 10.0 path over (the fixture"s quirk, by hand)', () => {
		const t = readNewick('((s1:5.0,s2:1.0):0.0,(s3:5.0,s4:1.0):0.0,s5:1.0);');
		enforceNonzeroBranchLengths(t, 1e-4);
		const r = rescaleDistances(computeFastDistMatrix(t, ['s1', 's2', 's3', 's4', 's5']), 5);
		expect(r.rawMax).toBe(Math.fround(10.0002));
		expect(r.rescaled).toBe(true);
	});

	it('produces float32 quotients and leaves the input untouched', () => {
		const input = Float32Array.from([0, 11, 11, 0]);
		const r = rescaleDistances(input, 3);
		expect(r.dist).toBeInstanceOf(Float32Array);
		expect(r.dist[1]).toBe(Math.fround(11 / 3));
		expect(input[1]).toBe(11);
	});

	it('raises on an empty matrix, as np.max does', () => {
		expect(() => rescaleDistances(new Float32Array(0), 1)).toThrow(/zero-size/);
	});
});
