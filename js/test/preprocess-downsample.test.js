/**
 * preprocess-downsample.test.js — prune_identical_sequences, downsample_taxa_faith_pd and the
 * stride pre-selection, on hand-checkable inputs plus Python-computed edge cases.
 *
 * WHY THIS FILE EXISTS
 *
 * The fixture replay covers the documented cases (planted duplicates, ties, camelid 128 -> 64). The
 * quirks worth pinning separately are the ones a reader would "fix": `max_species == 1` returning
 * two taxa, an all-zero matrix re-selecting index 0, and the stride expression's Python slice
 * semantics. Their expected values come from js/test/data/preprocess/edge_cases.json (computed by
 * hyphaeon.dataset), as does the Smc6 `max_species=8` end-to-end run, which exercises stride,
 * Faith's PD and MDS on a downsampled matrix together (no fixture does).
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { pruneIdenticalSequences, downsampleTaxaFaithPd, stridePreselect } from '../src/preprocess/downsample.js';
import { loadAlignmentAndTree } from '../src/preprocess/assemble.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA = JSON.parse(readFileSync(join(HERE, 'data', 'preprocess', 'edge_cases.json'), 'utf8'));
const EXAMPLES = join(HERE, '..', '..', 'examples');

describe('pruneIdenticalSequences', () => {
	it('keeps the first taxon per sequence, in the given order, and maps the rest to it', () => {
		const seqs = new Map([
			['a', 'ATG'],
			['b', 'CCC'],
			['c', 'ATG'],
			['d', 'ATG']
		]);
		const r = pruneIdenticalSequences(seqs, ['c', 'a', 'b', 'd']);
		expect(r.uniqueTaxa).toEqual(['c', 'b']);
		expect(Object.fromEntries(r.dupMap)).toEqual({ c: ['a', 'd'], b: [] });
		expect(r.numPruned).toBe(2);
	});

	it('is byte-identical: case and gaps make a sequence distinct', () => {
		const r = pruneIdenticalSequences({ a: 'ATG', b: 'atg', c: 'AT-' }, ['a', 'b', 'c']);
		expect(r.numPruned).toBe(0);
	});

	it('raises on a taxon that is not in the sequence map (KeyError)', () => {
		expect(() => pruneIdenticalSequences(new Map([['a', 'ATG']]), ['a', 'zzz'])).toThrow(/KeyError/);
	});
});

describe('downsampleTaxaFaithPd', () => {
	it('seeds with the most distant pair (first maximum in row-major order) then farthest points', () => {
		// Distances: a-b 1, a-c 5, a-d 4, b-c 2, b-d 3, c-d 6. Max is c-d at (2, 3): seed [c, d].
		// min_dists = min(row c, row d) = [4, 2, 0, 0]; argmax = 0 -> a. Then [1, 2, 0, 0] -> b.
		const D = Float32Array.from([0, 1, 5, 4, 1, 0, 2, 3, 5, 2, 0, 6, 4, 3, 6, 0]);
		const r = downsampleTaxaFaithPd(D, ['a', 'b', 'c', 'd'], 3);
		expect(r.taxa).toEqual(['c', 'd', 'a']);
		expect(r.selectedIndices).toEqual([2, 3, 0]);
		expect(Array.from(r.distMat)).toEqual([0, 6, 5, 6, 0, 4, 5, 4, 0]);
	});

	it('returns the inputs unchanged when max_species >= n or <= 0', () => {
		const D = Float32Array.from([0, 1, 1, 0]);
		expect(downsampleTaxaFaithPd(D, ['a', 'b'], 2).distMat).toBe(D);
		expect(downsampleTaxaFaithPd(D, ['a', 'b'], 0).taxa).toEqual(['a', 'b']);
		expect(downsampleTaxaFaithPd(D, ['a', 'b'], -3).taxa).toEqual(['a', 'b']);
	});

	it('max_species == 1 returns TWO taxa (the seed pair precedes the loop) — Python-computed', () => {
		const D = Float32Array.from([0, 1, 2, 1, 0, 3, 2, 3, 0]);
		const r = downsampleTaxaFaithPd(D, ['a', 'b', 'c'], 1);
		expect(r.taxa).toEqual(DATA.downsample.max_species_1.selected);
		expect(Array.from(r.distMat)).toEqual(DATA.downsample.max_species_1.sub.flat());
	});

	it('re-selects index 0 on an all-zero matrix, duplicates and all — Python-computed', () => {
		const r = downsampleTaxaFaithPd(new Float32Array(16), ['a', 'b', 'c', 'd'], 3);
		expect(r.taxa).toEqual(DATA.downsample.all_zero_repeats.selected);
		expect(Array.from(r.distMat)).toEqual(DATA.downsample.all_zero_repeats.sub.flat());
	});
});

describe('stridePreselect (dataset.py:672-673)', () => {
	for (const c of DATA.stride) {
		it(`n=${c.n}, max_species=${c.max_species}`, () => {
			const taxa = Array.from({ length: c.n }, (_, i) => `t${i}`);
			expect(stridePreselect(taxa, c.max_species)).toEqual(c.result);
		});
	}

	it('divides by zero for max_species == 0, as the Python does', () => {
		expect(() => stridePreselect(['a', 'b'], 0)).toThrow(/ZeroDivisionError/);
	});
});

describe('loadAlignmentAndTree with max_species on Smc6 (Python-computed)', () => {
	it('strides to 16, Faith"s-PD selects 8, MDS runs on the 8 x 8 matrix', () => {
		const e = DATA.load_alignment_and_tree_max_species_8_Smc6;
		const r = loadAlignmentAndTree(
			readFileSync(join(EXAMPLES, 'Smc6.fasta'), 'utf8'),
			readFileSync(join(EXAMPLES, 'Smc6.nwk'), 'utf8'),
			{ maxSpecies: 8 }
		);
		expect(r.taxa).toEqual(e.taxa);
		expect(r.L).toBe(e.L);
		expect(r.notices.pdSubsampled).toBe(true);
		const flatD = e.dist_matrix.flat();
		let worst = 0;
		for (let i = 0; i < flatD.length; i++) worst = Math.max(worst, Math.abs(r.d[i] - flatD[i]));
		expect(worst).toBeLessThanOrEqual(1e-6);
		const g = e.mds_gram.flat();
		let worstG = 0;
		for (let i = 0; i < 8; i++)
			for (let j = 0; j < 8; j++) {
				let s = 0;
				for (let c = 0; c < 4; c++) s += r.z[i * 4 + c] * r.z[j * 4 + c];
				worstG = Math.max(worstG, Math.abs(s - g[i * 8 + j]));
			}
		expect(worstG).toBeLessThanOrEqual(1e-5);
		expect(r.invariable.reduce((s, v) => s + v, 0)).toBe(e.n_invariable);
		expect(Array.from(r.c.slice(0, 8))).toEqual(e.codon_tokens_site0);
	});
});
