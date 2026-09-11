/**
 * WHY THIS FILE EXISTS
 *
 * Mirrors the DENSE path of `compute_mds_coordinates` in `hyphaeon/dataset.py` (lines 354-394 at
 * veg/HyphAeon cf838ab; the `mds_sign` parameter and `canonicalize_eigenvector_signs` were added
 * to it in the commit that regenerated the fixtures with the canonical convention), which produces
 * `mds_coords` — a model INPUT (`torch.linalg.eigh` has no ONNX lowering, so the graph cannot
 * compute it):
 *
 *     H = np.eye(n, dtype=np.float32) - (1.0 / n)      # float32
 *     B = -0.5 * H.dot(dist_matrix ** 2).dot(H)        # float32 (dist_matrix is float32)
 *     eigvals, eigvecs = np.linalg.eigh(B)             # LAPACK ssyevd, float32
 *     idx = np.argsort(eigvals)[::-1]                  # descending
 *     kept = eigvecs[:, :4]
 *     if mds_sign == "canonical":                      # default (env HYPHAEON_MDS_SIGN, CLI --mds-sign)
 *         kept = canonicalize_eigenvector_signs(kept)  # largest-|entry| of each column positive
 *     pos_eigvals = np.maximum(eigvals[:4], 0)
 *     coords = kept * np.sqrt(pos_eigvals)             # float32
 *     (pad with zero columns when n < 4)
 *
 * On the REAL n x n matrix — dataset.py:1047 calls it after downsampling, with no padding.
 *
 * PRECISION, DELIBERATELY: the reference forms H, D^2 and B in float32. This does the same — every
 * intermediate is rounded with Math.fround, and the two matrix products accumulate in float64 and
 * round each entry once (the closest a scalar loop can come to BLAS sgemm; bit parity with a blocked,
 * FMA-using sgemm is not available). The eigendecomposition then runs on those float32 values in
 * float64 (tred2/tql2 in symmetricEigen.js) where LAPACK's ssyevd runs in float32: eigenvalues agree
 * to float32 precision, eigenvectors to float32 precision divided by the eigenvalue gap. The fixture
 * class for MDS is 1e-5 absolute (fixtures/README.md); js/test/fixtures.test.js measures it.
 *
 * SIGNS ARE CANONICALISED BY DEFAULT, exactly as the reference does it. An eigenvector is only
 * defined up to sign; ssyevd (Python) and tql2 (here) returned opposite signs on bat_oas1 columns
 * 1, 2 and RHO columns 2, 3, and the model is NOT sign-invariant (`mds_proj = nn.Linear(4, ...)` on
 * the raw coordinates, plus Tree-RoPE), so those examples missed LRT parity by up to 0.12 / 0.26
 * absolute (hyphaeon-app PHASE1.md, gap 1) while Smc6, where the signs happened to agree, matched
 * at 1.5e-6. The rule (`canonicalize_eigenvector_signs`, dataset.py): for each KEPT column, find
 * the index of the largest |entry| — `np.argmax(np.abs(col))`, i.e. the FIRST index on an exact
 * tie — and flip the whole column when that entry is negative; an all-zero column is left alone.
 * It is applied BEFORE the sqrt(eigenvalue) scaling, on the float32 eigenvectors (here: the
 * float64 tql2 vectors rounded to float32 first, so the tie rule sees the same dtype). A sign flip
 * is exact in floating point, so canonical coordinates are bitwise the solver's up to sign.
 * With both sides canonical, the fixture replay compares each column EXACTLY (max|js - py|, 1e-5
 * class) — no sign-flip allowance. Within a DEGENERATE eigenspace (equal eigenvalues — an
 * equilateral triangle, say) any orthonormal basis is correct and only the Gram matrix is
 * comparable; and a column whose eigenvalue is at float32 noise (the synthetic 2-D case's 3rd and
 * 4th components) has a solver-dependent pivot, so the replay only requires it below the noise floor.
 *
 * `{ mdsSign: 'lapack' }` disables the rule and mirrors `mds_sign="lapack"` (the reference's
 * pre-canonical behaviour). The name is the reference's: in Python it means "whatever LAPACK /
 * ARPACK returned"; here it means "whatever tql2 returned", which is NOT LAPACK's choice — in that
 * mode the two implementations agree only up to a per-column sign. It exists so a consumer
 * reproducing an old `--mds-sign lapack` Python run can at least get the same magnitudes.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: the Lanczos path (dataset.py:582-603, `scipy.sparse.linalg.eigsh`
 * with `k=4, which='LA', maxiter=300` for n > 500) is not implemented; this always runs dense. At the
 * app's taxon cap of 512 that leaves 501..512 taxa where the reference's answer is an iterative
 * approximation of what this computes exactly — recorded for the integrator, not a defect here.
 *
 * DIVERGENCE FROM THE DATAMONKEY3 PORT (main@fac1330 src/lib/services/axomeme/mds.js), which this
 * file replaces: DM3 (1) canonicalised each column so its largest-magnitude entry was positive (the
 * AxoMEME 2.0 driver's rule) — removed in the first port because dataset.py had no convention, and
 * now back because dataset.py adopted the same rule (the pivot is chosen on the eigenvector, before
 * scaling, which for a non-negative scale is the same index DM3 chose on the scaled column);
 * (2) ran on the PADDED max_species x max_species matrix —
 * now the real n x n; (3) double-centred in float64 via the row/column-mean identity after rounding
 * distances to float32 — now float32 throughout, by the reference's two products; (4) returned
 * all-zero coordinates for n <= n_components — dataset.py has no such early return (n < 4 gives n
 * columns plus zero padding; n == 4 is an ordinary case). Its `if val > 0` guard and the reference's
 * `np.maximum(val, 0)` then `sqrt` agree in value and are kept in the reference's form.
 */

import { symmetricEigen } from './symmetricEigen.js';
import { MDS_COMPONENTS } from './modelContract.js';

/**
 * `compute_mds_coordinates(dist_matrix, n_components=4, mds_sign="canonical")`, dense path.
 *
 * @param {ArrayLike<number>} dist row-major n x n distance matrix; values are rounded to float32
 *   on entry because that is the dtype the reference receives
 * @param {number} n
 * @param {number} [nComponents]
 * @param {{ mdsSign?: 'canonical' | 'lapack' }} [options] `mdsSign` mirrors the reference's
 *   `mds_sign`: 'canonical' (default) flips each kept eigenvector so its largest-magnitude entry is
 *   positive; 'lapack' keeps the eigensolver's signs (see the header — not LAPACK's signs here)
 * @returns {Float32Array} n * nComponents, row-major
 */
export function computeMdsCoordinates(dist, n, nComponents = MDS_COMPONENTS, { mdsSign = 'canonical' } = {}) {
	if (mdsSign !== 'canonical' && mdsSign !== 'lapack') {
		throw new RangeError(`mdsSign must be 'canonical' or 'lapack', got ${String(mdsSign)}`);
	}
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
		// canonicalize_eigenvector_signs, on the float32 eigenvector and BEFORE scaling: pivot is
		// the first index of the largest |entry| (strict > keeps the first on ties, as np.argmax);
		// flip when that entry is negative. An all-zero column has a non-negative pivot: no flip.
		const sign = mdsSign === 'canonical' ? canonicalSign(vectors, n, src) : 1;
		for (let i = 0; i < n; i++) {
			coords[i * nComponents + c] = Math.fround(Math.fround(sign * vectors[i * n + src]) * scale);
		}
	}
	return coords;
}

/**
 * The sign (+1 or -1) that makes the largest-magnitude entry of eigenvector `col` positive, judged
 * on the float32-rounded entries as the reference does (its eigvecs are float32). Not exported:
 * the reference's `canonicalize_eigenvector_signs` is an implementation detail of
 * `compute_mds_coordinates`, and `index.js` pins the public surface.
 *
 * @param {Float64Array} vectors n x n, eigenvector k in column k (vectors[i * n + k])
 * @param {number} n
 * @param {number} col
 * @returns {1 | -1}
 */
function canonicalSign(vectors, n, col) {
	let best = -1;
	let pivot = 0;
	for (let i = 0; i < n; i++) {
		const v = Math.abs(Math.fround(vectors[i * n + col]));
		if (v > best) {
			best = v;
			pivot = i;
		}
	}
	return Math.fround(vectors[pivot * n + col]) < 0 ? -1 : 1;
}
