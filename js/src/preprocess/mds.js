/**
 * WHY THIS FILE EXISTS
 *
 * Mirrors the DENSE path of `compute_mds_coordinates` in `hyphaeon/dataset.py:354-394` at
 * veg/HyphAeon 267f5cf, which produces `mds_coords` — a model INPUT (`torch.linalg.eigh` has no
 * ONNX lowering, so the graph cannot compute it):
 *
 *     H = np.eye(n, dtype=np.float32) - (1.0 / n)      # float32
 *     B = -0.5 * H.dot(dist_matrix ** 2).dot(H)        # float32 (dist_matrix is float32)
 *     eigvals, eigvecs = np.linalg.eigh(B)             # LAPACK ssyevd, float32
 *     idx = np.argsort(eigvals)[::-1]                  # descending
 *     pos_eigvals = np.maximum(eigvals[:4], 0)
 *     coords = eigvecs[:, :4] * np.sqrt(pos_eigvals)   # float32
 *     (pad with zero columns when n < 4)
 *
 * On the REAL n x n matrix — dataset.py:688 calls it after downsampling, with no padding.
 *
 * PRECISION, DELIBERATELY: the reference forms H, D^2 and B in float32. This does the same — every
 * intermediate is rounded with Math.fround, and the two matrix products accumulate in float64 and
 * round each entry once (the closest a scalar loop can come to BLAS sgemm; bit parity with a blocked,
 * FMA-using sgemm is not available). The eigendecomposition then runs on those float32 values in
 * float64 (tred2/tql2 in symmetricEigen.js) where LAPACK's ssyevd runs in float32: eigenvalues agree
 * to float32 precision, eigenvectors to float32 precision divided by the eigenvalue gap. The fixture
 * class for MDS is 1e-5 absolute (fixtures/README.md); js/test/fixtures.test.js measures it.
 *
 * SIGNS ARE NOT CANONICALISED. dataset.py has no sign convention: each eigenvector's sign is
 * whatever the eigensolver returned, and ssyevd and tql2 do not agree on it. Parity is therefore UP
 * TO A GLOBAL SIGN PER COMPONENT (fixtures/README.md "MDS sign convention"), and the fixture replay
 * compares each column as min(max|js - py|, max|js + py|), or the sign-invariant Gram matrix
 * coords @ coords.T. Within a DEGENERATE eigenspace (equal eigenvalues — an equilateral triangle,
 * say) any orthonormal basis is correct and only the Gram matrix is comparable.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: the Lanczos path (dataset.py:360-381, `scipy.sparse.linalg.eigsh`
 * with `k=4, which='LA', maxiter=300` for n > 500) is not implemented; this always runs dense. At the
 * app's taxon cap of 512 that leaves 501..512 taxa where the reference's answer is an iterative
 * approximation of what this computes exactly — recorded for the integrator, not a defect here.
 *
 * DIVERGENCE FROM THE DATAMONKEY3 PORT (main@fac1330 src/lib/services/axomeme/mds.js), which this
 * file replaces: DM3 (1) canonicalised each column so its largest-magnitude entry was positive (the
 * AxoMEME 2.0 driver's rule) — removed; (2) ran on the PADDED max_species x max_species matrix —
 * now the real n x n; (3) double-centred in float64 via the row/column-mean identity after rounding
 * distances to float32 — now float32 throughout, by the reference's two products; (4) returned
 * all-zero coordinates for n <= n_components — dataset.py has no such early return (n < 4 gives n
 * columns plus zero padding; n == 4 is an ordinary case). Its `if val > 0` guard and the reference's
 * `np.maximum(val, 0)` then `sqrt` agree in value and are kept in the reference's form.
 */

import { symmetricEigen } from './symmetricEigen.js';
import { MDS_COMPONENTS } from './modelContract.js';

/**
 * `compute_mds_coordinates(dist_matrix, n_components=4)`, dense path.
 *
 * @param {ArrayLike<number>} dist row-major n x n distance matrix; values are rounded to float32
 *   on entry because that is the dtype the reference receives
 * @param {number} n
 * @param {number} [nComponents]
 * @returns {Float32Array} n * nComponents, row-major
 */
export function computeMdsCoordinates(dist, n, nComponents = MDS_COMPONENTS) {
	const coords = new Float32Array(n * nComponents);
	if (n === 0) return coords;

	// D2 = dist_matrix ** 2, float32.
	const D2 = new Float32Array(n * n);
	for (let i = 0; i < n * n; i++) {
		const v = Math.fround(dist[i]);
		D2[i] = Math.fround(v * v);
	}

	// H = eye(n, float32) - (1.0 / n): the Python float is cast to float32 and subtracted in float32.
	const invN = Math.fround(1.0 / n);
	const hDiag = Math.fround(1 - invN);
	const hOff = Math.fround(0 - invN);

	// M = H.dot(D2), float32 result; then B = -0.5 * M.dot(H). Each product entry is accumulated in
	// float64 and rounded once to float32.
	const M = new Float32Array(n * n);
	for (let i = 0; i < n; i++) {
		for (let j = 0; j < n; j++) {
			let acc = 0;
			for (let k = 0; k < n; k++) acc += (i === k ? hDiag : hOff) * D2[k * n + j];
			M[i * n + j] = Math.fround(acc);
		}
	}
	const B = new Float64Array(n * n);
	for (let i = 0; i < n; i++) {
		for (let j = 0; j < n; j++) {
			let acc = 0;
			for (let k = 0; k < n; k++) acc += M[i * n + k] * (k === j ? hDiag : hOff);
			B[i * n + j] = Math.fround(-0.5 * Math.fround(acc));
		}
	}

	const { values, vectors } = symmetricEigen(B, n);

	// Descending order; the top nComponents (or all n when n < nComponents, the rest zero-padded).
	const take = Math.min(nComponents, n);
	for (let c = 0; c < take; c++) {
		const src = n - 1 - c;
		const val = Math.fround(values[src]);
		const scale = Math.fround(Math.sqrt(Math.max(val, 0)));
		for (let i = 0; i < n; i++) {
			coords[i * nComponents + c] = Math.fround(Math.fround(vectors[i * n + src]) * scale);
		}
	}
	return coords;
}
