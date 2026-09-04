/**
 * WHY THIS FILE EXISTS
 *
 * Replays test/data/numeric/ranks.json (scipy `rankdata`, `pearsonr`, `spearmanr` and a transcript
 * of evaluation.py's `_roc_auc`) against src/numeric/ranks.js. Ranks are in PLAN.md §5.4's exact
 * class; correlations and ROC-AUC are in the 1e-9 class and are held to 1e-13 here.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { rankdata, pearson, spearman, rocAuc } from '../src/numeric/ranks.js';

const DATA = join(dirname(fileURLToPath(import.meta.url)), 'data', 'numeric');
const ref = JSON.parse(readFileSync(join(DATA, 'ranks.json'), 'utf8'));

describe('ranks.json replay: rankdata / pearson / spearman vs scipy', () => {
	for (const [name, c] of Object.entries(ref.correlation)) {
		it(`${name} (n=${c.x.length})`, () => {
			expect(Array.from(rankdata(c.x))).toEqual(c.rank_x);
			expect(Array.from(rankdata(c.y))).toEqual(c.rank_y);
			expect(Math.abs(pearson(c.x, c.y) - c.pearson)).toBeLessThanOrEqual(1e-13);
			expect(Math.abs(spearman(c.x, c.y) - c.spearman)).toBeLessThanOrEqual(1e-13);
		});
	}

	it('ties take the average rank, and typed arrays are accepted', () => {
		expect(Array.from(rankdata([10, 20, 10, 30, 20, 20]))).toEqual([1.5, 4, 1.5, 6, 4, 4]);
		expect(Array.from(rankdata(Float32Array.from([3, 1, 2])))).toEqual([3, 1, 2]);
		expect(rankdata([]).length).toBe(0);
	});

	it('degenerate inputs: n < 2 and constant vectors give NaN (scipy raises / warns)', () => {
		expect(pearson([1], [2])).toBeNaN();
		expect(spearman([1], [2])).toBeNaN();
		expect(pearson([1, 1, 1], [1, 2, 3])).toBeNaN();
		expect(spearman([1, 2, 3], [5, 5, 5])).toBeNaN();
		expect(() => pearson([1, 2], [1])).toThrow(RangeError);
	});

	it('is clipped to [-1, 1] on exact linear inputs', () => {
		expect(pearson([1, 2, 3, 4], [2, 4, 6, 8])).toBe(1);
		expect(spearman([1, 2, 3, 4], [8, 6, 4, 2])).toBe(-1);
	});
});

describe('ranks.json replay: rocAuc mirrors evaluation.py:275-283', () => {
	for (const [name, c] of Object.entries(ref.roc)) {
		it(`${name}`, () => {
			const got = rocAuc(c.labels, c.scores);
			if (c.auc === null) {
				expect(got).toBeNaN();
				expect(JSON.stringify({ roc_auc: got })).toBe('{"roc_auc":null}');
			} else {
				expect(Math.abs(got - c.auc)).toBeLessThanOrEqual(1e-13);
			}
		});
	}

	it('accepts boolean labels', () => {
		expect(rocAuc([false, false, true, true], [0.1, 0.2, 0.3, 0.4])).toBe(1);
		expect(rocAuc([true, false], [0.1, 0.2])).toBe(0);
	});
});
