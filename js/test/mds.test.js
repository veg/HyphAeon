/**
 * WHY THIS FILE EXISTS
 *
 * Ported verbatim from datamonkey3 (main@fac1330) src/test/axomeme-mds.test.js. Cases unchanged. It
 * pins `src/preprocess/symmetricEigen.js` and `src/preprocess/mds.js`.
 *
 * These check PROPERTIES — A = VΛVᵀ, orthonormality, distances recovered from an embedding that
 * exists exactly — rather than recorded outputs, because eigenvectors are unique only up to sign and
 * inside a degenerate eigenspace not even that, so "the numbers we got last time" is not a contract
 * anyone can hold two implementations to. The padding case is the one to read first: it pins that
 * MDS depends on `max_species`, which is the single most silently-wrong thing available in this
 * pipeline and a live divergence against `hyphaeon/dataset.py` (see mds.js's header).
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
 * Tests for the symmetric eigendecomposition and classical MDS.
 *
 * These check PROPERTIES, not recorded outputs. An eigendecomposition test that asserts "these are
 * the numbers we got last time" proves determinism and nothing else; the real contract is
 * A·V = V·Λ with V orthonormal, and that is checkable from first principles on any input. The MDS
 * cases are hand-derived from small distance matrices whose answer can be worked out on paper.
 *
 * The cross-implementation check against numpy lives in scripts/axomeme/verify_preprocessing.py +
 * .mjs, which compares against the ML team's own `compute_mds_coordinates` over 270 real DataMonkey
 * trees. That run is what proves parity; these tests are what make a failure there interpretable.
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
		// [[2,1],[1,2]] has eigenvalues 1 and 3.
		const { values } = symmetricEigen([2, 1, 1, 2], 2);
		expect(values[0]).toBeCloseTo(1, 12);
		expect(values[1]).toBeCloseTo(3, 12);
	});

	it('returns eigenvalues ASCENDING, matching numpy.linalg.eigh', () => {
		// mds.js reads components from the END of this array; a descending convention would silently
		// return the four SMALLEST components.
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
		expect(Array.from(values)).toEqual([1, 2, 3]); // sorted ascending
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
		// 2I has a fully degenerate spectrum — any orthonormal basis is correct. The routine must
		// still return orthonormal vectors and the right values rather than dividing by a zero gap.
		const { values, vectors } = symmetricEigen([2, 0, 0, 0, 2, 0, 0, 0, 2], 3);
		expect(Array.from(values)).toEqual([2, 2, 2]);
		for (let i = 0; i < 3; i++) {
			let norm = 0;
			for (let k = 0; k < 3; k++) norm += vectors[k * 3 + i] ** 2;
			expect(norm).toBeCloseTo(1, 12);
		}
	});
});

describe('computeMdsCoordinates', () => {
	it('returns zeros when n <= nComponents', () => {
		// `if N <= n_components: return zeros` — note <=, so n === nComponents is also all zeros.
		expect(Array.from(computeMdsCoordinates(new Float64Array(9), 3, 4))).toEqual(
			new Array(12).fill(0)
		);
		expect(Array.from(computeMdsCoordinates(new Float64Array(16), 4, 4))).toEqual(
			new Array(16).fill(0)
		);
	});

	it('recovers collinear points from their distances', () => {
		// Three points at 0, 1, 2 on a line. Worked by hand: D2 double-centres to
		// [[1,0,-1],[0,0,0],[-1,0,1]], whose only positive eigenvalue is 2 with eigenvector
		// [1,0,-1]/sqrt(2), so component 0 is [1, 0, -1] and component 1 is all zeros.
		const D = [0, 1, 2, 1, 0, 1, 2, 1, 0];
		const c = computeMdsCoordinates(D, 3, 2);
		expect(c[0 * 2 + 0]).toBeCloseTo(1, 5);
		expect(c[1 * 2 + 0]).toBeCloseTo(0, 5);
		expect(c[2 * 2 + 0]).toBeCloseTo(-1, 5);
		// The second component's eigenvalue is mathematically ZERO — three collinear points need one
		// dimension. It does not come out as exactly zero, though: it lands on float dust around
		// 1e-17, and the reference's guard is `if val > 0`, which does not distinguish a true zero
		// from dust. So a coordinate of ~1e-8 (sqrt of the dust) is emitted rather than a clean zero.
		// That is the reference's behaviour and this port matches it; asserting `toBe(0)` here would
		// be asserting something neither implementation does.
		for (const i of [0, 1, 2]) expect(Math.abs(c[i * 2 + 1])).toBeLessThan(1e-6);
	});

	it('preserves pairwise distances for points that embed exactly', () => {
		// A square of side 1: MDS in 2 dimensions must reproduce the input distances.
		const s2 = Math.SQRT2;
		const D = [0, 1, s2, 1, 1, 0, 1, s2, s2, 1, 0, 1, 1, s2, 1, 0];
		const c = computeMdsCoordinates(D, 4, 2);
		const dist = (i, j) => Math.hypot(c[i * 2] - c[j * 2], c[i * 2 + 1] - c[j * 2 + 1]);
		expect(dist(0, 1)).toBeCloseTo(1, 4);
		expect(dist(1, 2)).toBeCloseTo(1, 4);
		expect(dist(0, 2)).toBeCloseTo(s2, 4);
		expect(dist(1, 3)).toBeCloseTo(s2, 4);
	});

	it('applies the sign convention: the largest-magnitude entry is positive', () => {
		// The reference's rule, and the thing that removes half the eigenvector ambiguity for free.
		const D = [0, 1, 2, 1, 0, 1, 2, 1, 0];
		const c = computeMdsCoordinates(D, 3, 2);
		let maxAbs = 0;
		let atIdx = 0;
		for (let i = 0; i < 3; i++) {
			if (Math.abs(c[i * 2]) > maxAbs) {
				maxAbs = Math.abs(c[i * 2]);
				atIdx = i;
			}
		}
		expect(c[atIdx * 2]).toBeGreaterThan(0);
	});

	it('DEPENDS ON THE PADDING, because the reference runs MDS on the padded matrix', () => {
		// Not a quirk to be optimised away. The padded zeros take part in the double-centring, so the
		// same real taxa padded to different max_species produce different coordinates. A port that
		// runs MDS on the real N and pads afterwards silently feeds the model different inputs.
		const real = [0, 1, 2, 1, 0, 1, 2, 1, 0];
		const at = (cap) => {
			const p = new Float64Array(cap * cap);
			for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) p[i * cap + j] = real[i * 3 + j];
			return computeMdsCoordinates(p, cap, 2);
		};
		const a = at(8);
		const b = at(16);
		// Component 1 is where it shows. Measured across caps for these three collinear points:
		//   cap= 3 -> 9.50e-9   cap= 4 -> -9.833e-2   cap= 8 -> -1.0097e-1
		//   cap=16 -> -1.0133e-1                      cap=64 -> -1.0150e-1
		// It converges as the padding grows but never stops depending on it.
		expect(a[0 * 2 + 1]).not.toBeCloseTo(b[0 * 2 + 1], 6);
		expect(a[1 * 2 + 1]).not.toBeCloseTo(b[1 * 2 + 1], 6);
		// Component 0's MAGNITUDE happens to be padding-invariant for this symmetric example (the
		// collinear geometry dominates), which is why the check above is on component 1 — picking
		// component 0 would have made this test pass for the wrong reason and then fail later.
		expect(Math.abs(a[0])).toBeCloseTo(Math.abs(b[0]), 6);
	});

	it('rounds distances to float32 before squaring, as the reference does', () => {
		// The reference is handed a float32 torch tensor, so it squares float32 values. Feeding
		// float64 changes components 2-3 by up to 99% on real trees (measured), because squared
		// distances reach ~1e6 while the fourth eigenvalue can be ~1e-1. Passing an already-rounded
		// matrix and a full-precision one must therefore give the SAME answer.
		const n = 8;
		const raw = new Float64Array(n * n);
		let s = 3;
		const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
		for (let i = 0; i < n; i++) {
			for (let j = i + 1; j < n; j++) {
				const v = rnd() * 1000;
				raw[i * n + j] = v;
				raw[j * n + i] = v;
			}
		}
		const rounded = Float64Array.from(raw, (v) => Math.fround(v));
		const a = computeMdsCoordinates(raw, n, 4);
		const b = computeMdsCoordinates(rounded, n, 4);
		expect(Array.from(a)).toEqual(Array.from(b));
	});

	it('returns float32 values, matching the reference cast', () => {
		const D = [0, 1, 2, 1, 0, 1, 2, 1, 0];
		const c = computeMdsCoordinates(D, 3, 2);
		expect(c).toBeInstanceOf(Float32Array);
		for (const v of c) expect(Math.fround(v)).toBe(v);
	});

	it('treats a negative distance as its magnitude, because squaring loses the sign', () => {
		// Documented consequence rather than desired behaviour: DM3's NJ emits negative branch
		// lengths, and MDS silently absorbs them where the reference's density term throws.
		const pos = computeMdsCoordinates([0, 1, 2, 1, 0, 1, 2, 1, 0], 3, 2);
		const neg = computeMdsCoordinates([0, -1, 2, -1, 0, 1, 2, 1, 0], 3, 2);
		expect(Array.from(neg)).toEqual(Array.from(pos));
	});
});
