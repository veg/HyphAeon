/**
 * WHY THIS FILE EXISTS
 *
 * Replays test/data/numeric/linalg.json (numpy `cholesky` and `eigvalsh` on SPD and rank-deficient
 * Gram matrices) against src/numeric/linalg.js, and cross-checks the eigenvalues-only path against
 * preprocess/symmetricEigen.js. Eigenvalues are a property of the matrix, so both routes must agree
 * with numpy to near machine precision (1e-12 relative to the spectral radius here); the Cholesky
 * factor is unique for an SPD matrix and is compared entrywise.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { cholesky, symmetricEigenvalues, largestEigenvalue, symmetricEigen } from '../src/numeric/linalg.js';
import { symmetricEigen as preprocessEigen } from '../src/preprocess/symmetricEigen.js';
import * as kernel from '../src/numeric/index.js';

const DATA = join(dirname(fileURLToPath(import.meta.url)), 'data', 'numeric');
const ref = JSON.parse(readFileSync(join(DATA, 'linalg.json'), 'utf8'));

function flat(rows) {
	return Float64Array.from(rows.flat());
}

describe('linalg.json replay vs numpy', () => {
	for (const [name, m] of Object.entries(ref.matrices)) {
		const n = m.n;
		const A = flat(m.A);
		const scale = Math.max(...m.eigvalsh.map(Math.abs));

		it(`${name}: eigenvalues ascending within 1e-12·ρ(A), largest matches eigvalsh[-1]`, () => {
			const got = symmetricEigenvalues(A, n);
			expect(got.length).toBe(n);
			m.eigvalsh.forEach((want, i) => {
				expect(Math.abs(got[i] - want), `λ[${i}]`).toBeLessThanOrEqual(1e-12 * scale);
			});
			expect(Math.abs(largestEigenvalue(A, n) - m.largest)).toBeLessThanOrEqual(1e-12 * scale);
			// The eigenvalues-only path agrees with the full decomposition.
			const full = symmetricEigen(A, n).values;
			for (let i = 0; i < n; i++) expect(Math.abs(got[i] - full[i])).toBeLessThanOrEqual(1e-12 * scale);
		});

		it(`${name}: cholesky ${m.cholesky ? 'matches numpy entrywise' : 'throws (not positive definite)'}`, () => {
			if (m.cholesky === null) {
				expect(() => cholesky(A, n)).toThrow(RangeError);
				return;
			}
			const L = cholesky(A, n);
			const want = flat(m.cholesky);
			for (let i = 0; i < n * n; i++) expect(Math.abs(L[i] - want[i]), `L[${i}]`).toBeLessThanOrEqual(1e-12 * Math.sqrt(scale));
			// Upper triangle is exactly zero, and L·Lᵀ reconstructs A.
			for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) expect(L[i * n + j]).toBe(0);
			for (let i = 0; i < n; i++) {
				for (let j = 0; j < n; j++) {
					let s = 0;
					for (let k = 0; k < n; k++) s += L[i * n + k] * L[j * n + k];
					expect(Math.abs(s - A[i * n + j])).toBeLessThanOrEqual(1e-12 * scale);
				}
			}
		});
	}

	it('the Gram case is rank-deficient: trailing eigenvalues are ~0 and the coherence ratio is well defined', () => {
		const m = ref.matrices.gram_20;
		const ev = symmetricEigenvalues(flat(m.A), m.n);
		const trace = ev.reduce((a, b) => a + Math.max(b, 0), 0);
		let tr = 0;
		for (let i = 0; i < m.n; i++) tr += m.A[i][i];
		expect(Math.abs(trace - tr)).toBeLessThanOrEqual(1e-10 * tr);
		// 20 × 12 factor: at most 12 non-zero eigenvalues
		expect(ev.slice(0, 8).every((v) => Math.abs(v) < 1e-9 * tr)).toBe(true);
	});
});

describe('edge cases and re-exports', () => {
	it('handles n = 0, 1, 2', () => {
		expect(symmetricEigenvalues([], 0).length).toBe(0);
		expect(Array.from(symmetricEigenvalues([3], 1))).toEqual([3]);
		expect(largestEigenvalue([2, 1, 1, 2], 2)).toBeCloseTo(3, 14);
		expect(Array.from(symmetricEigenvalues([2, 1, 1, 2], 2))).toEqual([expect.closeTo(1, 14), expect.closeTo(3, 14)]);
		expect(Array.from(cholesky([4], 1))).toEqual([2]);
	});

	it('reads only the lower triangle, as numpy eigvalsh(UPLO="L") and cholesky do', () => {
		const A = [2, 999, 1, 2]; // upper-right garbage
		expect(largestEigenvalue(A, 2)).toBeCloseTo(3, 14);
		expect(Array.from(cholesky([4, 999, 2, 3], 2))).toEqual([2, 0, 1, expect.closeTo(Math.SQRT2, 15)]);
	});

	it('numeric/index.js re-exports the preprocess symmetricEigen binding itself', () => {
		expect(kernel.symmetricEigen).toBe(preprocessEigen);
		expect(symmetricEigen).toBe(preprocessEigen);
	});

	it('exposes the kernel API other ports code against', () => {
		const names = [
			'lgamma', 'gammaincReg', 'gammaincc', 'betaincReg', 'erfc', 'chi2Sf', 'chi2Cdf', 'tSf', 'tCdf', 'normSf', 'normCdf',
			'logChoose', 'hypergeomPmf', 'hypergeomCdf', 'hypergeomSf', 'Xoshiro256', 'rankdata', 'pearson', 'spearman',
			'rocAuc', 'benjaminiHochberg', 'cholesky', 'largestEigenvalue', 'symmetricEigenvalues', 'symmetricEigen',
			'cauchyCombination',
		];
		for (const n of names) expect(typeof kernel[n], n).toBe('function');
	});
});
