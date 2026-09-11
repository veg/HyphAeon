/**
 * WHY THIS FILE EXISTS
 *
 * The two dense linear-algebra calls the ports need beyond the MDS eigendecomposition that
 * preprocess/symmetricEigen.js already provides:
 *
 *   - `np.linalg.cholesky(V + 1e-7 * np.eye(M))`   phenotype.py:351 (Brownian-motion permulations)
 *   - `np.linalg.eigvalsh(cov)[:, -1]`               epistasis.py:283 (largest eigenvalue of each
 *     K × K Gram matrix in the vectorised permutation null, batches of ≤ 25,000) and
 *     `np.linalg.eigvalsh(cov2)` at epistasis.py:364 (spectral coherence λ₁ / Tr)
 *
 * `symmetricEigen` (tred2 + tql2, eigenvectors included) is re-exported here so the numeric kernel
 * is the single import for linear algebra; `symmetricEigenvalues` is the eigenvalues-only variant
 * (Householder without accumulating the transformation, then the same implicit-shift QL on the
 * tridiagonal) and `largestEigenvalue` is its last entry. Measured on 2,000 random 20 × 20 Gram
 * matrices (K = 20 sites, N = 12 taxa, the epistasis.py:282 shape): 45 ms for the eigenvalues-only
 * path against 79 ms for the full decomposition (1.7×), ~22 µs per matrix, so a 25,000-draw batch
 * costs ~0.6 s; the two paths agree to 2.1e-15 relative to the spectral radius, and against
 * numpy's `eigvalsh` on the 3 × 3 … 30 × 30 test matrices both are within 1e-12·ρ(A) (the test
 * bound; the observed differences are at the 1e-15 level).
 *
 * WHAT IT DELIBERATELY DOES NOT DO. No batching or einsum: the epistasis port forms each Gram
 * matrix and calls `largestEigenvalue` per draw. No `np.maximum(·, 0)` — the clamps at
 * epistasis.py:284,361 belong to the caller. Cholesky throws on a non-positive-definite input the
 * way `np.linalg.cholesky` raises LinAlgError; the ridge is the caller's.
 *
 * The routines are Numerical Recipes 3rd ed. `tred2`/`tqli` with `yesvecs = false`, transcribed
 * rather than derived from symmetricEigen.js's JAMA tred2/tql2, so the two eigenvalue paths are
 * independent implementations that the tests check against each other and against numpy.
 */

export { symmetricEigen } from '../preprocess/symmetricEigen.js';

/**
 * Cholesky factor L (lower triangular, row-major n × n, zeros above the diagonal) with A = L·Lᵀ —
 * np.linalg.cholesky. Only the lower triangle of A is read, as LAPACK's `potrf('L')` does.
 *
 * @param {Float64Array|number[]} A row-major, n*n, symmetric positive definite
 * @param {number} n
 * @returns {Float64Array}
 * @throws {RangeError} when A is not positive definite (numpy: LinAlgError)
 */
export function cholesky(A, n) {
	const L = new Float64Array(n * n);
	for (let j = 0; j < n; j++) {
		let d = A[j * n + j];
		for (let k = 0; k < j; k++) d -= L[j * n + k] * L[j * n + k];
		if (!(d > 0)) throw new RangeError(`cholesky: matrix is not positive definite (pivot ${j} = ${d})`);
		const ljj = Math.sqrt(d);
		L[j * n + j] = ljj;
		for (let i = j + 1; i < n; i++) {
			let s = A[i * n + j];
			for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
			L[i * n + j] = s / ljj;
		}
	}
	return L;
}

/**
 * Eigenvalues of a real symmetric matrix, ascending — np.linalg.eigvalsh. Reads the lower
 * triangle (numpy's default `UPLO='L'`).
 *
 * @param {Float64Array|number[]} A row-major, n*n, symmetric
 * @param {number} n
 * @returns {Float64Array} ascending
 */
export function symmetricEigenvalues(A, n) {
	if (n === 0) return new Float64Array(0);
	if (n === 1) return Float64Array.from([A[0]]);
	const V = new Float64Array(n * n);
	for (let i = 0; i < n; i++) {
		for (let j = 0; j <= i; j++) {
			V[i * n + j] = A[i * n + j];
			V[j * n + i] = A[i * n + j];
		}
	}
	const d = new Float64Array(n);
	const e = new Float64Array(n);
	tred1(V, d, e, n);
	tql1(d, e, n);
	return d;
}

/**
 * Largest eigenvalue of a real symmetric matrix — `np.linalg.eigvalsh(A)[-1]`.
 *
 * @param {Float64Array|number[]} A row-major, n*n, symmetric
 * @param {number} n
 * @returns {number}
 */
export function largestEigenvalue(A, n) {
	const d = symmetricEigenvalues(A, n);
	return d[n - 1];
}

/**
 * Householder reduction to tridiagonal form without accumulating the transformation — Numerical
 * Recipes 3rd ed. `tred2` with `yesvecs = false`. On exit d holds the diagonal and e the
 * sub-diagonal (e[0] = 0); the lower triangle of V is destroyed.
 */
function tred1(V, d, e, n) {
	for (let i = n - 1; i > 0; i--) {
		const l = i - 1;
		let h = 0;
		let scale = 0;
		if (l > 0) {
			for (let k = 0; k < i; k++) scale += Math.abs(V[i * n + k]);
			if (scale === 0) {
				e[i] = V[i * n + l];
			} else {
				for (let k = 0; k < i; k++) {
					V[i * n + k] /= scale;
					h += V[i * n + k] * V[i * n + k];
				}
				let f = V[i * n + l];
				let g = f >= 0 ? -Math.sqrt(h) : Math.sqrt(h);
				e[i] = scale * g;
				h -= f * g;
				V[i * n + l] = f - g;
				f = 0;
				for (let j = 0; j < i; j++) {
					g = 0;
					for (let k = 0; k < j + 1; k++) g += V[j * n + k] * V[i * n + k];
					for (let k = j + 1; k < i; k++) g += V[k * n + j] * V[i * n + k];
					e[j] = g / h;
					f += e[j] * V[i * n + j];
				}
				const hh = f / (h + h);
				for (let j = 0; j < i; j++) {
					f = V[i * n + j];
					g = e[j] - hh * f;
					e[j] = g;
					for (let k = 0; k < j + 1; k++) V[j * n + k] -= f * e[k] + g * V[i * n + k];
				}
			}
		} else {
			e[i] = V[i * n + l];
		}
		d[i] = h;
	}
	e[0] = 0;
	for (let i = 0; i < n; i++) d[i] = V[i * n + i];
}

/**
 * Implicit-shift QL on a symmetric tridiagonal matrix, eigenvalues only, sorted ascending —
 * Numerical Recipes 3rd ed. `tqli` with `yesvecs = false` (d diagonal, e sub-diagonal in e[1..]).
 */
function tql1(d, e, n) {
	const eps = Math.pow(2, -52);
	for (let i = 1; i < n; i++) e[i - 1] = e[i];
	e[n - 1] = 0;
	for (let l = 0; l < n; l++) {
		let iter = 0;
		let m;
		do {
			for (m = l; m < n - 1; m++) {
				const dd = Math.abs(d[m]) + Math.abs(d[m + 1]);
				if (Math.abs(e[m]) <= eps * dd) break;
			}
			if (m !== l) {
				if (iter++ === 60) throw new RangeError('tql1: QL iteration did not converge');
				let g = (d[l + 1] - d[l]) / (2 * e[l]);
				let r = Math.hypot(g, 1);
				g = d[m] - d[l] + e[l] / (g + (g >= 0 ? Math.abs(r) : -Math.abs(r)));
				let s = 1;
				let c = 1;
				let p = 0;
				let i;
				for (i = m - 1; i >= l; i--) {
					const f = s * e[i];
					const b = c * e[i];
					r = Math.hypot(f, g);
					e[i + 1] = r;
					if (r === 0) {
						d[i + 1] -= p;
						e[m] = 0;
						break;
					}
					s = f / r;
					c = g / r;
					g = d[i + 1] - p;
					r = (d[i] - g) * s + 2 * c * b;
					p = s * r;
					d[i + 1] = g + p;
					g = c * r - b;
				}
				if (r === 0 && i >= l) continue;
				d[l] -= p;
				e[l] = g;
				e[m] = 0;
			}
		} while (m !== l);
	}

	// Sort ascending.
	for (let i = 0; i < n - 1; i++) {
		let k = i;
		let p = d[i];
		for (let j = i + 1; j < n; j++) {
			if (d[j] < p) {
				k = j;
				p = d[j];
			}
		}
		if (k !== i) {
			d[k] = d[i];
			d[i] = p;
		}
	}
}
