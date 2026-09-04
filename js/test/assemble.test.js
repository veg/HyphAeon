/**
 * WHY THIS FILE EXISTS
 *
 * Ported verbatim from datamonkey3 (main@fac1330) src/test/axomeme-assemble.test.js. Cases
 * unchanged. It pins `src/preprocess/assemble.js`, the layer where the separately-verified stages
 * are joined — so the failures available here are joining failures: right values in the wrong order,
 * right shapes with the wrong species, MDS computed on the wrong matrix. None of them crash, which
 * is why every bundle built below is also run through `validateInputBundle`.
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
 * Tests for AxoMEME tensor assembly.
 *
 * Assembly is where the separately-verified stages get joined, so the failures available here are
 * joining failures: right values in the wrong order, right shapes with the wrong species, MDS
 * computed on the wrong matrix. None of them crash. All of them produce a well-formed bundle that
 * means something the model was not trained on, which is why every bundle below is also run through
 * validateInputBundle.
 */
import { describe, it, expect } from 'vitest';
import {
	prepareAlignment,
	chooseReference,
	orderSpecies,
	batchSizeFor
} from '../src/preprocess/assemble.js';
import { parseNewick } from '../src/preprocess/newick.js';
import { computeMdsCoordinates } from '../src/preprocess/mds.js';
import { codonToken, aaToken } from '../src/preprocess/tokenizer.js';
import {
	validateInputBundle,
	CODON_UNKNOWN,
	AA_UNKNOWN,
	MDS_COMPONENTS
} from '../src/preprocess/modelContract.js';

/** Four taxa, distinct codons, 3 sites each. Tree order is deliberately NOT alignment order. */
const NAMES = ['alpha', 'beta', 'gamma', 'delta'];
const SEQS = ['ATGTTATCA', 'ATGCTATCA', 'ATGTTAAGC', 'ATGGGGTCA'];
const TREE = '((gamma:0.1,delta:0.2):0.05,(beta:0.3,alpha:0.15):0.02);';

const prep = (over = {}) =>
	prepareAlignment({ names: NAMES, sequences: SEQS, treeText: TREE, maxSpecies: 8, ...over });

describe('chooseReference', () => {
	it('honours an explicit choice', () => {
		expect(chooseReference(NAMES, 'gamma')).toBe('gamma');
	});

	it('falls back to the first sequence, which is what fires on viral data', () => {
		// The heuristic looks for 'hg' / 'hg38' / 'human' — a TOGA-mammal artifact. DataMonkey traffic
		// is viral, so the fallback is the real behaviour.
		expect(chooseReference(NAMES)).toBe('alpha');
		expect(chooseReference(['x', 'human', 'y'])).toBe('human');
		expect(chooseReference(['x', 'hg38'])).toBe('hg38');
	});

	it('ignores an explicit name that is not in the alignment', () => {
		expect(chooseReference(NAMES, 'nope')).toBe('alpha');
	});
});

describe('orderSpecies', () => {
	it('takes TREE order, not alignment order', () => {
		// Order fixes the distance matrix rows and therefore MDS, and index 0 seeds Max-PD.
		const tree = parseNewick(TREE);
		const { order, matchedFromTree } = orderSpecies(NAMES, tree, 'gamma');
		expect(matchedFromTree).toBe(true);
		expect(order.map((i) => NAMES[i])).toEqual(['gamma', 'delta', 'beta', 'alpha']);
	});

	it('moves the reference sequence to the front', () => {
		const tree = parseNewick(TREE);
		const { order } = orderSpecies(NAMES, tree, 'alpha');
		expect(order.map((i) => NAMES[i])[0]).toBe('alpha');
		// and everything else keeps tree order behind it
		expect(order.map((i) => NAMES[i])).toEqual(['alpha', 'gamma', 'delta', 'beta']);
	});

	it('falls back to alignment order when nothing matches the tree', () => {
		const tree = parseNewick('((zzz:0.1,yyy:0.2):0.05);');
		const { order, matchedFromTree } = orderSpecies(NAMES, tree, 'alpha');
		expect(matchedFromTree).toBe(false);
		expect(order.map((i) => NAMES[i])).toEqual(NAMES);
	});

	it('drops alignment sequences that are absent from the tree', () => {
		const tree = parseNewick('((gamma:0.1,delta:0.2):0.05);');
		const { order } = orderSpecies(NAMES, tree, 'gamma');
		expect(order.map((i) => NAMES[i])).toEqual(['gamma', 'delta']);
	});
});

describe('prepareAlignment', () => {
	it('produces a bundle that satisfies the contract', () => {
		const p = prep();
		const bundle = p.batch(0);
		const v = validateInputBundle(bundle, {
			batch: p.totalCodons,
			numSpecies: p.speciesCount,
			windowSize: p.windowSize
		});
		expect(v.errors).toEqual([]);
	});

	it('derives the site count from the reference sequence', () => {
		expect(prep().totalCodons).toBe(3); // 9 nt / 3
	});

	it('feeds the graph only the real species, and emits no mask tensor', () => {
		// DM3 never padded: `n` is the number of SELECTED species, so every row was always real and
		// the mask was always all-zero. v1-viral drops the input entirely, so what is worth pinning
		// now is that the bundle does not carry one — a stale mask would be a bundle built for the
		// retired 2.0 graph.
		const prepared = prep();
		const bundle = prepared.batch(0);
		expect(bundle).not.toHaveProperty('padding_mask');
		expect(bundle.dist_matrix.dims[1]).toBe(prepared.speciesCount);
		expect(bundle.mds_coords.dims[1]).toBe(prepared.speciesCount);
	});

	it('computes MDS on the PADDED matrix and slices, not on the real N', () => {
		// The single most silently-wrong thing available in this file. Coordinates depend on
		// max_species because the padded zeros take part in the double-centring.
		const p = prepareAlignment({
			names: NAMES,
			sequences: SEQS,
			treeText: TREE,
			maxSpecies: 16
		});
		const cap = 16;
		const padded = new Float64Array(cap * cap);
		const n = p.speciesCount;
		for (let i = 0; i < n; i++) {
			for (let j = 0; j < n; j++) padded[i * cap + j] = p.dist[i * n + j];
		}
		const expected = computeMdsCoordinates(padded, cap, MDS_COMPONENTS);
		for (let i = 0; i < n * MDS_COMPONENTS; i++) {
			expect(p.mds[i]).toBeCloseTo(expected[i], 6);
		}
		// And it is genuinely different from the unpadded answer, so the test above has teeth.
		const unpadded = computeMdsCoordinates(Float64Array.from(p.dist), n, MDS_COMPONENTS);
		expect(Array.from(p.mds)).not.toEqual(Array.from(unpadded));
	});

	it('orders the distance matrix rows to match the selected species', () => {
		const p = prep({ referenceName: 'gamma' });
		expect(p.selectedNames).toEqual(['gamma', 'delta', 'beta', 'alpha']);
		const n = p.speciesCount;
		// gamma-delta share a parent: 0.1 + 0.2 = 0.3. gamma-beta crosses the root: 0.1+0.05+0.02+0.3.
		expect(p.dist[0 * n + 1]).toBeCloseTo(0.3, 5);
		expect(p.dist[0 * n + 2]).toBeCloseTo(0.47, 5);
		expect(p.dist[0 * n + 0]).toBe(0);
	});

	it('tokenises each species at each site, in selected order', () => {
		const p = prep({ referenceName: 'alpha' });
		// selected: alpha, gamma, delta, beta -> site 1 (2nd codon) is TTA, TTA, GGG, CTA
		const n = p.speciesCount;
		const at = (site, s) => Number(p.codonTokens[(site * n + s) * p.windowSize]);
		expect(at(1, 0)).toBe(codonToken('TTA'));
		expect(at(1, 1)).toBe(codonToken('TTA'));
		expect(at(1, 2)).toBe(codonToken('GGG'));
		expect(at(1, 3)).toBe(codonToken('CTA'));
		const aaAt = (site, s) => Number(p.aaTokens[(site * n + s) * p.windowSize]);
		expect(aaAt(1, 2)).toBe(aaToken('GGG'));
	});

	it('leaves pad values where a sequence is shorter than the reference', () => {
		// `torch.ones(...) * 65` is never overwritten past a short sequence's end.
		const p = prepareAlignment({
			names: ['a', 'b'],
			sequences: ['ATGTTATCA', 'ATG'],
			treeText: '(a:0.1,b:0.2);',
			maxSpecies: 8
		});
		const n = p.speciesCount;
		const idx = (site, s) => (site * n + s) * p.windowSize;
		expect(Number(p.codonTokens[idx(0, 1)])).toBe(codonToken('ATG'));
		expect(Number(p.codonTokens[idx(1, 1)])).toBe(CODON_UNKNOWN);
		expect(Number(p.aaTokens[idx(1, 1)])).toBe(AA_UNKNOWN);
	});

	it('applies Max-PD when over the cap, seeded at the reference', () => {
		const names = ['r', 'near', 'far', 'mid'];
		const seqs = ['ATG', 'ATG', 'ATG', 'ATG'];
		const p = prepareAlignment({
			names,
			sequences: seqs,
			treeText: '((r:0.01,near:0.01):0.05,(far:2.0,mid:0.5):0.5);',
			maxSpecies: 2,
			referenceName: 'r'
		});
		expect(p.speciesCount).toBe(2);
		expect(p.selectedNames[0]).toBe('r'); // the seed
		expect(p.selectedNames[1]).toBe('far'); // farthest from it
	});

	it('clamps a negative distance to zero and REPORTS the magnitude', () => {
		// DM3's own NJ emits negative branch lengths; the large.nex demo produces a patristic sum of
		// -1.04e-5, which is zero with rounding error on it. The model was trained on clamped
		// distances (the handoff README says so), so clamping matches training — but doing it silently
		// would hide a genuinely broken tree, which is why the magnitude comes back out.
		const p = prepareAlignment({
			names: ['a', 'b', 'c'],
			sequences: ['ATG', 'ATG', 'ATG'],
			treeText: '((a:-0.5,b:0.2):0.05,c:0.3);',
			maxSpecies: 8
		});
		expect(Array.from(p.dist).every((v) => v >= 0)).toBe(true);
		expect(p.clampedDistances).toBeGreaterThan(0);
		expect(p.mostNegativeDistance).toBeCloseTo(-0.3, 5); // a-b: -0.5 + 0.2
	});

	it('reports nothing clamped for a clean tree', () => {
		const p = prep();
		expect(p.clampedDistances).toBe(0);
		expect(p.mostNegativeDistance).toBe(0);
	});

	it('produces a contract-valid bundle from a tree with negative branch lengths', () => {
		// The large.nex regression: the bundle used to be REJECTED by validateInputBundle because a
		// patristic sum came out at -1e-5. Clamping is what makes an ordinary NJ tree usable.
		const p = prepareAlignment({
			names: ['a', 'b', 'c', 'd'],
			sequences: ['ATGTTA', 'ATGCTA', 'ATGGGG', 'ATGAAA'],
			treeText: '((a:-0.00001,b:0.2):0.05,(c:0.3,d:0.1):0.02);',
			maxSpecies: 8
		});
		const v = validateInputBundle(p.batch(0), {
			batch: p.totalCodons,
			numSpecies: p.speciesCount,
			windowSize: p.windowSize
		});
		expect(v.errors).toEqual([]);
	});

	it('falls back to an all-zero distance matrix without a tree', () => {
		const p = prepareAlignment({ names: NAMES, sequences: SEQS, maxSpecies: 8 });
		expect(Array.from(p.dist).every((v) => v === 0)).toBe(true);
		expect(p.matchedFromTree).toBe(false);
		// MDS of an all-zero matrix is all zeros, not NaN.
		expect(Array.from(p.mds).every((v) => v === 0)).toBe(true);
	});

	it('rejects mismatched or empty input rather than producing a bundle', () => {
		expect(() => prepareAlignment({ names: ['a'], sequences: [] })).toThrow(/parallel/);
		expect(() => prepareAlignment({ names: [], sequences: [] })).toThrow(/no sequences/);
	});
});

describe('batching', () => {
	it('slices sites without disturbing the per-alignment tensors', () => {
		const p = prep();
		const all = p.batch(0);
		const tail = p.batch(1, 2);
		expect(tail.msa_codons.dims).toEqual([2, 4, 1]);
		// site 1 of the full batch is site 0 of this one
		const n = p.speciesCount;
		for (let s = 0; s < n; s++) {
			expect(tail.msa_codons.data[s]).toBe(all.msa_codons.data[n + s]);
		}
		// and the invariant tensors are repeated per site, identically
		for (let k = 0; k < n * n; k++) {
			expect(tail.dist_matrix.data[n * n + k]).toBe(tail.dist_matrix.data[k]);
		}
	});

	it('clamps a range that runs past the end', () => {
		const p = prep();
		expect(p.batch(2, 99).msa_codons.dims[0]).toBe(1);
		expect(p.batch(3, 5).msa_codons.dims[0]).toBe(0);
	});

	it('sizes batches against the dist_matrix budget', () => {
		// 4 * N^2 bytes per site is the dominant term.
		expect(batchSizeFor(512, 64 * 1024 * 1024)).toBe(64);
		expect(batchSizeFor(36, 64 * 1024 * 1024)).toBeGreaterThan(1000);
		// Never zero: one site of a huge alignment still has to go through.
		expect(batchSizeFor(4096, 1024)).toBe(1);
	});
});
