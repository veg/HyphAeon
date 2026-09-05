/**
 * mds.test.js — the symmetric eigendecomposition and classical MDS.
 *
 * WHY THIS FILE EXISTS
 *
 * The `symmetricEigen` cases are DataMonkey 3's (axomeme-mds.test.js), unchanged: they check
 * properties (A = VΛVᵀ, orthonormality, ascending order) that hold for any correct implementation.
 * The `computeMdsCoordinates` cases were rewritten for dataset.py:354-394: the DM3 cases that
 * asserted dependence on the PADDED matrix and all-zero output for n <= 4 pinned behaviour
 * dataset.py does not have and were deleted. DM3's sign-convention case was deleted with them
 * because dataset.py then had no convention; dataset.py has since adopted the same rule
 * (`canonicalize_eigenvector_signs`: largest-|entry| of each kept eigenvector positive, first index
 * on ties, before the sqrt(eigenvalue) scaling; `mds_sign="canonical"`, the default), so the
 * `canonical sign convention` block below pins it — the rule itself, the tie-break, the ordering
 * relative to scaling, the `{ mdsSign: 'lapack' }` escape hatch, and that the flip is exact. The
 * numeric oracle is the fixture replay (Smc6, bat_oas1 raw and rescaled, synthetic 2-D points, the
 * equilateral triangle), now compared EXACTLY per column with no sign allowance.
 */
import { describe, it, expect } from 'vitest';
import { symmetricEigen } from '../src/preprocess/symmetricEigen.js';
import { computeMdsCoordinates } from '../src/preprocess/mds.js';

/** Deterministic symmetric matrix, fixed seed so a failure is reproducible. */
function symMatrix(n, seed) {
	let s = seed;
	const rnd = () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff) * 2 - 1;
	const A = new Float64Array(n * n);
	for (let i = 0; i < n; i++) {
		for (let j = i; j < n; j++) {
			const v = rnd();
			A[i * n + j] = v;
			A[j * n + i] = v;
		}
	}
	return A;
}

describe('symmetricEigen', () => {
	it('solves a 2x2 with known eigenvalues', () => {
		const { values } = symmetricEigen([2, 1, 1, 2], 2);
		expect(values[0]).toBeCloseTo(1, 12);
		expect(values[1]).toBeCloseTo(3, 12);
	});

	it('returns eigenvalues ASCENDING, matching numpy.linalg.eigh', () => {
		const { values } = symmetricEigen(symMatrix(20, 3), 20);
		for (let i = 1; i < 20; i++) expect(values[i]).toBeGreaterThanOrEqual(values[i - 1]);
	});

	it('reconstructs the matrix: A = V Λ Vᵀ', () => {
		for (const n of [3, 10, 40]) {
			const A = symMatrix(n, n * 7 + 1);
			const { values, vectors } = symmetricEigen(A, n);
			let worst = 0;
			for (let i = 0; i < n; i++) {
				for (let j = 0; j < n; j++) {
					let acc = 0;
					for (let k = 0; k < n; k++) acc += vectors[i * n + k] * values[k] * vectors[j * n + k];
					worst = Math.max(worst, Math.abs(acc - A[i * n + j]));
				}
			}
			expect(worst, `n=${n}`).toBeLessThan(1e-12);
		}
	});

	it('produces orthonormal eigenvectors', () => {
		const n = 30;
		const { vectors } = symmetricEigen(symMatrix(n, 11), n);
		let worst = 0;
		for (let i = 0; i < n; i++) {
			for (let j = 0; j < n; j++) {
				let dot = 0;
				for (let k = 0; k < n; k++) dot += vectors[k * n + i] * vectors[k * n + j];
				worst = Math.max(worst, Math.abs(dot - (i === j ? 1 : 0)));
			}
		}
		expect(worst).toBeLessThan(1e-12);
	});

	it('handles a diagonal matrix, the identity, and n=1', () => {
		const { values } = symmetricEigen([3, 0, 0, 0, 1, 0, 0, 0, 2], 3);
		expect(Array.from(values)).toEqual([1, 2, 3]);
		const id = symmetricEigen([1, 0, 0, 1], 2);
		expect(Array.from(id.values)).toEqual([1, 1]);
		const one = symmetricEigen([7], 1);
		expect(one.values[0]).toBe(7);
	});

	it('does not mutate the caller"s matrix', () => {
		const A = Float64Array.from([2, 1, 1, 2]);
		symmetricEigen(A, 2);
		expect(Array.from(A)).toEqual([2, 1, 1, 2]);
	});

	it('survives a matrix with repeated eigenvalues', () => {
		const { values, vectors } = symmetricEigen([2, 0, 0, 0, 2, 0, 0, 0, 2], 3);
		expect(Array.from(values)).toEqual([2, 2, 2]);
		for (let i = 0; i < 3; i++) {
			let norm = 0;
			for (let k = 0; k < 3; k++) norm += vectors[k * 3 + i] ** 2;
			expect(norm).toBeCloseTo(1, 12);
		}
	});
});

describe('computeMdsCoordinates (dataset.py:383-394, dense path)', () => {
	it('recovers collinear points from their distances; canonical sign makes the first tied pivot positive', () => {
		// Points at 0, 1, 2: B double-centres to [[1,0,-1],[0,0,0],[-1,0,1]], eigenvalue 2 with
		// eigenvector [1,0,-1]/sqrt(2), so component 0 is ±[1, 0, -1]. |entries| 0 and 2 tie exactly in
		// float32 (both 1/sqrt(2) up to rounding); np.argmax takes the FIRST, index 0, so it is +1.
		const c = computeMdsCoordinates([0, 1, 2, 1, 0, 1, 2, 1, 0], 3, 2);
		expect(c[0 * 2]).toBeCloseTo(1, 5);
		expect(c[1 * 2]).toBeCloseTo(0, 5);
		expect(c[2 * 2]).toBeCloseTo(-1, 5);
		expect(Math.sign(c[0 * 2])).toBe(-Math.sign(c[2 * 2]));
		// The second eigenvalue is zero up to float dust; sqrt of dust is emitted, not a clean 0.
		for (const i of [0, 1, 2]) expect(Math.abs(c[i * 2 + 1])).toBeLessThan(1e-3);
	});

	it('preserves pairwise distances for points that embed exactly', () => {
		const s2 = Math.SQRT2;
		const D = [0, 1, s2, 1, 1, 0, 1, s2, s2, 1, 0, 1, 1, s2, 1, 0];
		const c = computeMdsCoordinates(D, 4, 2);
		const dist = (i, j) => Math.hypot(c[i * 2] - c[j * 2], c[i * 2 + 1] - c[j * 2 + 1]);
		expect(dist(0, 1)).toBeCloseTo(1, 4);
		expect(dist(1, 2)).toBeCloseTo(1, 4);
		expect(dist(0, 2)).toBeCloseTo(s2, 4);
		expect(dist(1, 3)).toBeCloseTo(s2, 4);
	});

	it('has no n <= n_components early return: 4 points in 4 components are real coordinates', () => {
		const s2 = Math.SQRT2;
		const D = [0, 1, s2, 1, 1, 0, 1, s2, s2, 1, 0, 1, 1, s2, 1, 0];
		const c = computeMdsCoordinates(D, 4, 4);
		expect(Array.from(c).some((v) => Math.abs(v) > 0.1)).toBe(true);
	});

	it('zero-pads the columns beyond n when n < n_components (np.hstack with zeros)', () => {
		const c = computeMdsCoordinates([0, 1, 2, 1, 0, 1, 2, 1, 0], 3, 4);
		for (let i = 0; i < 3; i++) expect(c[i * 4 + 3]).toBe(0);
	});

	it('returns float32 values on the real n x n matrix, and an empty array for n = 0', () => {
		const c = computeMdsCoordinates([0, 1, 2, 1, 0, 1, 2, 1, 0], 3, 2);
		expect(c).toBeInstanceOf(Float32Array);
		expect(c).toHaveLength(6);
		for (const v of c) expect(Math.fround(v)).toBe(v);
		expect(computeMdsCoordinates([], 0, 4)).toHaveLength(0);
	});

	it('rounds the input to float32 first, as the reference receives a float32 tensor', () => {
		const n = 8;
		const raw = new Float64Array(n * n);
		let s = 3;
		const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
		for (let i = 0; i < n; i++)
			for (let j = i + 1; j < n; j++) {
				const v = rnd() * 1000;
				raw[i * n + j] = v;
				raw[j * n + i] = v;
			}
		const rounded = Float64Array.from(raw, (v) => Math.fround(v));
		expect(Array.from(computeMdsCoordinates(raw, n, 4))).toEqual(Array.from(computeMdsCoordinates(rounded, n, 4)));
	});

	it('treats a negative distance as its magnitude, because squaring loses the sign', () => {
		const pos = computeMdsCoordinates([0, 1, 2, 1, 0, 1, 2, 1, 0], 3, 2);
		const neg = computeMdsCoordinates([0, -1, 2, -1, 0, 1, 2, 1, 0], 3, 2);
		expect(Array.from(neg)).toEqual(Array.from(pos));
	});

	describe('canonical sign convention (dataset.py canonicalize_eigenvector_signs)', () => {
		/** Deterministic Euclidean distance matrix over n points in 3-D (float32), row-major. */
		function pointCloud(n, seed) {
			let st = seed;
			const rnd = () => ((st = (st * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff) * 2 - 1;
			const P = [];
			for (let i = 0; i < n; i++) P.push([rnd(), rnd() * 0.7, rnd() * 0.3]);
			const D = new Float32Array(n * n);
			for (let i = 0; i < n; i++)
				for (let j = 0; j < n; j++) D[i * n + j] = Math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1], P[i][2] - P[j][2]);
			return D;
		}
		const n = 14;
		const k = 4;
		const D = pointCloud(n, 7);

		it('is the default: every column has its largest-magnitude entry positive (first index on ties)', () => {
			for (const z of [computeMdsCoordinates(D, n, k), computeMdsCoordinates(D, n, k, { mdsSign: 'canonical' })]) {
				for (let col = 0; col < 3; col++) {
					let best = -1;
					let pivot = -1;
					for (let i = 0; i < n; i++) {
						const v = Math.abs(z[i * k + col]);
						if (v > best) {
							best = v;
							pivot = i;
						}
					}
					expect(best, `column ${col} is not noise`).toBeGreaterThan(1e-3);
					expect(z[pivot * k + col], `column ${col} pivot ${pivot}`).toBeGreaterThan(0);
				}
			}
		});

		it("{ mdsSign: 'lapack' } keeps the solver's signs, and canonical differs from it by exactly a per-column sign", () => {
			const raw = computeMdsCoordinates(D, n, k, { mdsSign: 'lapack' });
			const can = computeMdsCoordinates(D, n, k);
			let flips = 0;
			for (let col = 0; col < k; col++) {
				let same = true;
				let flipped = true;
				for (let i = 0; i < n; i++) {
					if (can[i * k + col] !== raw[i * k + col]) same = false;
					if (can[i * k + col] !== -raw[i * k + col]) flipped = false;
				}
				expect(same || flipped, `column ${col} is bitwise the raw column up to sign`).toBe(true);
				if (!same) flips++;
			}
			// tql2 happens to return a negative pivot on at least one column of this cloud; if a future
			// eigensolver change makes all four agree the assertion below is the one to revisit.
			expect(flips).toBeGreaterThan(0);
		});

		it('rejects an unknown mdsSign', () => {
			expect(() => computeMdsCoordinates(D, n, k, { mdsSign: 'numpy' })).toThrow(RangeError);
		});

		it('is applied before the sqrt(eigenvalue) scaling: a zero-padded or clipped column stays all zero, never -0', () => {
			// n = 3 < 4 components: column 3 is np.hstack zero padding; there is nothing to flip.
			const c = computeMdsCoordinates([0, 1, 2, 1, 0, 1, 2, 1, 0], 3, 4);
			for (let i = 0; i < 3; i++) expect(Object.is(c[i * 4 + 3], 0)).toBe(true);
			// An all-zero distance matrix: every eigenvalue is 0, every coordinate 0 * eigvec = +0 (the
			// reference's np.maximum(., 0) then sqrt gives 0.0 and 0.0 * negative = -0.0 in numpy too,
			// so only the magnitude is pinned here, as in the fixture class).
			const z = computeMdsCoordinates(new Float32Array(9), 3, 4);
			for (const v of z) expect(Math.abs(v)).toBe(0);
		});

		it('picks the pivot on the float32-rounded eigenvector, as the reference does on float32 eigvecs', () => {
			// Two entries whose float64 magnitudes differ by less than a float32 ULP tie after rounding;
			// the rule must then take the first index. Build the case through the public function by
			// checking that the pivot chosen agrees with a float32 argmax over the returned column.
			const z = computeMdsCoordinates(D, n, k);
			for (let col = 0; col < 3; col++) {
				let best = -1;
				let pivot = -1;
				for (let i = 0; i < n; i++) {
					const v = Math.abs(Math.fround(z[i * k + col]));
					if (v > best) {
						best = v;
						pivot = i;
					}
				}
				expect(z[pivot * k + col]).toBeGreaterThan(0);
			}
		});
	});
});
