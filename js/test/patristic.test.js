/**
 * WHY THIS FILE EXISTS
 *
 * Ported verbatim from datamonkey3 (main@fac1330) src/test/axomeme-patristic.test.js. Cases
 * unchanged. It pins `src/preprocess/newick.js` and `src/preprocess/patristic.js` — the half of the
 * preprocessing port that CAN be made bit-identical to Python, since it is plain float arithmetic
 * over a tree walk with no eigendecomposition anywhere near it.
 *
 * Every distance asserted below is hand-computed from the newick string in the test, not recorded
 * from the implementation. That distinction is the whole point: an expectation captured from the
 * code under test asserts only that the code is deterministic, which it would be with the
 * arithmetic wrong.
 *
 * WHAT WAS CHANGED, AND ONLY THIS: the import paths, repointed from
 * `../lib/services/axomeme/…` at DataMonkey 3's layout to `../src/preprocess/…` at this package's.
 * Not one case, expectation or comment was edited, so a failure here is a failure of the port and
 * never of a local adjustment to the test. Comments that cite DataMonkey 3 paths
 * (`src/lib/utils/treeSanitation.js`, `scripts/axomeme/verify_preprocessing.py`) are left as
 * written: they are the record of where a measurement was taken, and rewriting them would break the
 * trail without moving the evidence.
 *
 * THE FILE WAS RENAMED from `axomeme-<x>.test.js` to `<x>.test.js`. "AxoMEME" was DataMonkey 3's
 * name for the pillar this model serves; in this repository the package IS HyphAeon, so the prefix
 * distinguished nothing and the module under test is what the name should say.
 */

/**
 * Tests for the newick parser and patristic distances — the first half of the AxoMEME preprocessing
 * port.
 *
 * The distances here are hand-computed from the newick, not recorded from this code. That
 * distinction is the whole point: an expectation captured from the implementation asserts only that
 * the implementation is deterministic, which it would be even if the arithmetic were wrong. Every
 * number below can be checked by reading the tree string.
 *
 * The cross-implementation check against Python lives in scripts/axomeme/verify_preprocessing.py and
 * is what actually proves parity; these tests are what make a failure there interpretable.
 */
import { describe, it, expect } from 'vitest';
import { parseNewick, leafIndex, normalizeTaxonName } from '../src/preprocess/newick.js';
import {
	rootDistances,
	patristicRow,
	patristicMatrix,
	maxPdSelect
} from '../src/preprocess/patristic.js';

/** ((A:0.1,B:0.2):0.05,C:0.3); — the worked example used throughout. */
const SIMPLE = '((A:0.1,B:0.2):0.05,C:0.3);';

/** Node index of the leaf named `n`. */
const leafOf = (tree, n) => leafIndex(tree).index.get(n);

describe('parseNewick', () => {
	it('builds the topology of a nested tree', () => {
		const t = parseNewick(SIMPLE);
		const { index } = leafIndex(t);
		expect([...index.keys()].sort()).toEqual(['A', 'B', 'C']);
		// A and B share a parent; C hangs off the root.
		expect(t.parent[index.get('A')]).toBe(t.parent[index.get('B')]);
		expect(t.parent[index.get('C')]).toBe(t.root);
		expect(t.parent[t.parent[index.get('A')]]).toBe(t.root);
	});

	it('handles arbitrary nesting on both sides', () => {
		const t = parseNewick('((A,B),(C,D));');
		const { index } = leafIndex(t);
		expect([...index.keys()].sort()).toEqual(['A', 'B', 'C', 'D']);
		expect(t.parent[index.get('A')]).toBe(t.parent[index.get('B')]);
		expect(t.parent[index.get('C')]).toBe(t.parent[index.get('D')]);
		expect(t.parent[index.get('A')]).not.toBe(t.parent[index.get('C')]);
	});

	it('nests a clade that appears after a leaf', () => {
		// (A,(B,C)) exercises the branch where ',' and '(' arrive back to back.
		const t = parseNewick('(A,(B,C));');
		const { index } = leafIndex(t);
		expect(t.parent[index.get('A')]).toBe(t.root);
		expect(t.parent[index.get('B')]).toBe(t.parent[index.get('C')]);
		expect(t.parent[index.get('B')]).not.toBe(t.root);
	});

	it('reads plain, scientific and NEGATIVE branch lengths', () => {
		const t = parseNewick('(a:1.5e-2,b:3.0E-3,c:-0.5);');
		const { index } = leafIndex(t);
		expect(t.branchLength[index.get('a')]).toBeCloseTo(0.015, 12);
		expect(t.branchLength[index.get('b')]).toBeCloseTo(0.003, 12);
		// Preserved, NOT clamped — DM3's own NJ emits these and hiding them here would turn a loud
		// downstream failure into a quiet wrong answer.
		expect(t.branchLength[index.get('c')]).toBe(-0.5);
	});

	it('treats a missing branch length as 0, matching `branch_length or 0.0`', () => {
		const t = parseNewick('((A,B),C);');
		expect(Array.from(t.branchLength).every((v) => v === 0)).toBe(true);
		expect(Array.from(rootDistances(t)).every((v) => v === 0)).toBe(true);
	});

	it('does not mistake a bootstrap value for a taxon', () => {
		// )95: is a label on an internal node. Reading it as a name invents a species.
		const t = parseNewick('((A:0.1,B:0.2)95:0.05,C:0.3);');
		const { index } = leafIndex(t);
		expect([...index.keys()].sort()).toEqual(['A', 'B', 'C']);
		expect(index.has('95')).toBe(false);
		// The label is kept on the internal node, just not treated as a leaf.
		expect(t.name[t.parent[index.get('A')]]).toBe('95');
	});

	it('keeps a quoted label containing a colon intact', () => {
		// The entire reason newick quoting exists, and the case a naive split on ':' corrupts.
		const t = parseNewick("(('Homo:sapiens':0.1,b:0.2):0.05);");
		const { index } = leafIndex(t);
		expect(index.has('Homo:sapiens')).toBe(true);
		expect(t.branchLength[index.get('Homo:sapiens')]).toBeCloseTo(0.1, 12);
	});

	it('strips newick comments rather than reading them as labels', () => {
		const t = parseNewick('((A[&&NHX:x=1]:0.1,B:0.2):0.05,C:0.3);');
		expect([...leafIndex(t).index.keys()].sort()).toEqual(['A', 'B', 'C']);
	});

	it('parses without a trailing semicolon, and a bare single taxon', () => {
		expect([...leafIndex(parseNewick('(A:0.1,B:0.2)')).index.keys()].sort()).toEqual(['A', 'B']);
		const solo = parseNewick('A:0.1;');
		expect([...leafIndex(solo).index.keys()]).toEqual(['A']);
	});

	it('rejects empty input rather than returning an empty tree', () => {
		expect(() => parseNewick('')).toThrow(/empty/);
		expect(() => parseNewick('   ')).toThrow(/empty/);
	});

	it('lists leaves in PREORDER, matching Biopython get_terminals()', () => {
		// Not cosmetic. The reference resolves duplicate tip names by dict overwrite, so the winner is
		// the last leaf in THIS order; a breadth-first walk yields the same set and a different winner.
		const t = parseNewick('((A:0.1,B:0.2):0.05,C:0.3);');
		expect(t.leaves.map((n) => t.name[n])).toEqual(['A', 'B', 'C']);
		const t2 = parseNewick('(A,((B,C),D));');
		expect(t2.leaves.map((n) => t2.name[n])).toEqual(['A', 'B', 'C', 'D']);
	});

	it('resolves a duplicate tip name to the LAST leaf, and reports it', () => {
		// Matches `{leaf.name: leaf for leaf in leaves}` — later entries overwrite earlier. Measured:
		// 3 of 270 real DM3 trees have duplicate tips, and first-wins disagreed with the reference on
		// every one of them. `duplicates` is what lets a caller refuse; the index itself stays
		// faithful.
		const t = parseNewick('((A:0.1,A:0.2):0.05,C:0.3);');
		const { index, duplicates } = leafIndex(t);
		expect(duplicates).toEqual(['A']);
		expect(index.size).toBe(2);
		// The SECOND 'A' — the one with branch length 0.2.
		expect(t.branchLength[index.get('A')]).toBeCloseTo(0.2, 12);
	});
});

describe('normalizeTaxonName', () => {
	it('removes every quote anywhere, matching the reference', () => {
		// Python's str.replace removes ALL occurrences; the reference does
		// s.replace("'", "").replace('"', '').strip().
		expect(normalizeTaxonName("'Homo sapiens'")).toBe('Homo sapiens');
		expect(normalizeTaxonName("Homo_'sapiens'")).toBe('Homo_sapiens');
		expect(normalizeTaxonName('  "x"  ')).toBe('x');
	});
});

describe('rootDistances and patristic distances', () => {
	it('accumulates root distances down the tree', () => {
		const t = parseNewick(SIMPLE);
		const d = rootDistances(t);
		expect(d[leafOf(t, 'A')]).toBeCloseTo(0.15, 12); // 0.05 + 0.1
		expect(d[leafOf(t, 'B')]).toBeCloseTo(0.25, 12); // 0.05 + 0.2
		expect(d[leafOf(t, 'C')]).toBeCloseTo(0.3, 12);
		expect(d[t.root]).toBe(0);
	});

	it('computes hand-checkable pairwise distances', () => {
		const t = parseNewick(SIMPLE);
		const nodes = ['A', 'B', 'C'].map((n) => leafOf(t, n));
		const m = patristicMatrix(t, nodes);
		const at = (i, j) => m[i * 3 + j];
		expect(at(0, 1)).toBeCloseTo(0.3, 12); // A-B: 0.1 + 0.2
		expect(at(0, 2)).toBeCloseTo(0.45, 12); // A-C: 0.1 + 0.05 + 0.3
		expect(at(1, 2)).toBeCloseTo(0.55, 12); // B-C: 0.2 + 0.05 + 0.3
	});

	it('is symmetric with a zero diagonal', () => {
		const t = parseNewick('((A:0.1,B:0.2):0.05,(C:0.3,D:0.15):0.02);');
		const nodes = ['A', 'B', 'C', 'D'].map((n) => leafOf(t, n));
		const m = patristicMatrix(t, nodes);
		for (let i = 0; i < 4; i++) {
			expect(m[i * 4 + i]).toBe(0);
			for (let j = 0; j < 4; j++) expect(m[i * 4 + j]).toBeCloseTo(m[j * 4 + i], 12);
		}
	});

	it('reuses its ancestor marker across rows without leaking marks', () => {
		// The stamped marker is the one piece of state shared between rows. If a stale generation
		// leaked, an LCA would resolve to a node on the PREVIOUS row's path and distances would come
		// out too small — so compute a matrix (shared marker) and compare against fresh single rows.
		const t = parseNewick('(((A:0.1,B:0.2):0.05,C:0.3):0.01,(D:0.4,E:0.05):0.2);');
		const names = ['A', 'B', 'C', 'D', 'E'];
		const nodes = names.map((n) => leafOf(t, n));
		const shared = patristicMatrix(t, nodes);
		const rd = rootDistances(t);
		for (let i = 0; i < nodes.length; i++) {
			const fresh = patristicRow(t, rd, nodes[i], nodes); // no marker -> its own
			for (let j = 0; j < nodes.length; j++) {
				expect(shared[i * nodes.length + j]).toBeCloseTo(fresh[j], 12);
			}
		}
	});

	it('propagates a negative branch length into the distance instead of hiding it', () => {
		const t = parseNewick('((A:-0.5,B:0.2):0.05,C:0.3);');
		const nodes = ['A', 'B', 'C'].map((n) => leafOf(t, n));
		const m = patristicMatrix(t, nodes);
		expect(m[0 * 3 + 1]).toBeCloseTo(-0.3, 12); // A-B: -0.5 + 0.2
		expect(m[0 * 3 + 2]).toBeCloseTo(-0.15, 12); // A-C: -0.5 + 0.05 + 0.3
	});

	it('gives every pair distance 0 on a topology-only tree', () => {
		const t = parseNewick('((A,B),C);');
		const nodes = ['A', 'B', 'C'].map((n) => leafOf(t, n));
		expect(Array.from(patristicMatrix(t, nodes)).every((v) => v === 0)).toBe(true);
	});

	it('handles a deep ladder tree without recursing', () => {
		// 3,000 nested clades. The Python reference recurses here and dies at its frame limit; this
		// port must not, because DM3 accepts uploads far larger than 1,000 taxa.
		const N = 3000;
		let s = 'L0:0.001';
		for (let i = 1; i < N; i++) s = `(${s},L${i}:0.001)`;
		const t = parseNewick(s + ';');
		const idx = leafIndex(t).index;
		expect(idx.size).toBe(N);
		const d = rootDistances(t);
		// L0 is the deepest tip: N-1 internal branches (all 0, no lengths given) plus its own 0.001.
		expect(Number.isFinite(d[idx.get('L0')])).toBe(true);
		expect(d[idx.get(`L${N - 1}`)]).toBeCloseTo(0.001, 12);
	});
});

describe('maxPdSelect', () => {
	it('returns everything, in order, when under the cap', () => {
		const t = parseNewick(SIMPLE);
		const nodes = ['A', 'B', 'C'].map((n) => leafOf(t, n));
		expect(maxPdSelect(t, nodes, 512).selected).toEqual([0, 1, 2]);
	});

	it('seeds at index 0 and then takes the farthest point', () => {
		// A-B are close, C and D are far. Seeded at A (index 0), the next pick is whichever is
		// farthest from A, then the one farthest from {A, that}.
		const t = parseNewick('((A:0.01,B:0.01):0.05,(C:1.0,D:2.0):0.5);');
		const nodes = ['A', 'B', 'C', 'D'].map((n) => leafOf(t, n));
		const { selected } = maxPdSelect(t, nodes, 3);
		expect(selected[0]).toBe(0); // always the seed
		expect(selected[1]).toBe(3); // D, farthest from A
		expect(selected[2]).toBe(2); // C, farthest from {A, D}
	});

	it('is sensitive to input ORDER, because the seed is index 0 — not a defect this layer fixes', () => {
		// The reference always seeds at the first taxon in alignment order, so reordering the
		// sequences in an upload changes which taxa the model sees. Pinned so nobody "fixes" it into
		// divergence from the model's training-time behaviour.
		const t = parseNewick('((A:0.01,B:0.01):0.05,(C:1.0,D:2.0):0.5);');
		const asIs = ['A', 'B', 'C', 'D'].map((n) => leafOf(t, n));
		const rotated = ['C', 'D', 'A', 'B'].map((n) => leafOf(t, n));
		const a = maxPdSelect(t, asIs, 2).selected.map((i) => t.name[asIs[i]]);
		const b = maxPdSelect(t, rotated, 2).selected.map((i) => t.name[rotated[i]]);
		expect(a).toEqual(['A', 'D']);
		expect(b).toEqual(['C', 'D']);
		expect(a).not.toEqual(b);
	});

	it('reports duplicates on an all-zero distance matrix instead of hiding them', () => {
		// Every selected index has minDist 0, so once all candidates are 0 argmax returns index 0
		// forever and one taxon fills every slot. Reproduced (the model was trained with it) but
		// counted, so a caller can refuse.
		const t = parseNewick('((A,B),(C,D));'); // no branch lengths -> all distances 0
		const nodes = ['A', 'B', 'C', 'D'].map((n) => leafOf(t, n));
		const { selected, duplicates } = maxPdSelect(t, nodes, 3);
		expect(selected).toEqual([0, 0, 0]);
		expect(duplicates).toBe(2);
	});

	it('selects exactly maxSpecies when over the cap', () => {
		const t = parseNewick('((A:0.1,B:0.2):0.05,(C:0.3,D:0.15):0.02);');
		const nodes = ['A', 'B', 'C', 'D'].map((n) => leafOf(t, n));
		expect(maxPdSelect(t, nodes, 2).selected).toHaveLength(2);
	});
});
