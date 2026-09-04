/**
 * WHY THIS FILE EXISTS
 *
 * Ported verbatim from datamonkey3 (main@fac1330) src/lib/services/axomeme/mds.js. Function bodies
 * unchanged; its `./symmetricEigen.js` and `./modelContract.js` imports resolve to siblings here as
 * they did there.
 *
 * It mirrors `compute_mds_coordinates` in `hyphaeon/dataset.py` (dataset.py:354-394), which produces
 * `mds_coords` — a model INPUT, not a graph computation.
 *
 * THREE DIVERGENCES AGAINST v1.0.0 dataset.py, ALL OF WHICH CHANGE NUMBERS AND NONE OF WHICH ARE
 * RESOLVED HERE. They are stated so a parity failure is read correctly rather than debugged blind:
 *
 *   1. NO SIGN CONVENTION IN dataset.py. The 2.0 driver canonicalised each eigenvector so its
 *      largest-magnitude entry is positive, and step 6 below reproduces that. dataset.py:385-391 has
 *      no such step, so its eigenvector signs are whatever LAPACK returned. Sign is the ambiguity
 *      that canonicalisation exists to remove; without it, agreement between any two
 *      eigendecompositions is a coincidence per component.
 *   2. NEGATIVE EIGENVALUES ARE CLAMPED, NOT SKIPPED. dataset.py uses
 *      `np.maximum(eigvals[:4], 0)` then multiplies — so a negative component contributes a zero
 *      coordinate by way of sqrt(0). The port's `if val > 0` leaves that component at zero too, so
 *      these agree in value; they differ only in that dataset.py's `coords` keeps the eigenvector
 *      shape while this returns a zero column. Recorded because it looks like a divergence and is
 *      not one.
 *   3. PADDING. This file's caller (assemble.js) hands it the [max_species, max_species] PADDED
 *      matrix because the 2.0 driver did; dataset.py:691 calls it with the real
 *      [n_taxa, n_taxa] matrix. The padded zeros take part in the double-centring, so this is not a
 *      rounding difference — it is a different embedding. See assemble.js.
 *
 * Also note dataset.py's dense path forms `H` and `B` in FLOAT32 (its `dist_matrix` is float32 and
 * `np.eye(n, dtype=np.float32)`), where this double-centres in float64 after rounding the distances
 * to float32. The measurement behind that rounding is in the body below and is the reason the port
 * agrees with the reference on real trees at all.
 */

/**
 * mds.js — classical multidimensional scaling, matching predict_regression_nexus.py:161-187.
 *
 * This produces `mds_coords`, one of the five ONNX graph inputs, and it is the LAST piece of the
 * preprocessing port and the only one whose agreement with Python cannot be guaranteed by
 * construction. Everything else — newick parse, patristic distances, tokenisation — is exact
 * arithmetic over discrete choices. This one runs an eigendecomposition, and eigenvectors are unique
 * only up to sign, and inside a degenerate eigenspace not even that.
 *
 * THE REFERENCE, step for step:
 *   1. N <= n_components -> all-zero coordinates, early return.
 *   2. cast to float64; D2 = D ** 2
 *   3. H = I - ones(N,N)/N ; B = -0.5 * (H @ D2 @ H)
 *   4. evals, evecs = np.linalg.eigh(B)                       (ascending)
 *   5. reorder DESCENDING by eigenvalue
 *   6. sign convention: per column, if the largest-magnitude entry is negative, negate the column
 *   7. coords[:, i] = evecs[:, i] * sqrt(evals[i]) when evals[i] > 0, else left at zero
 *   8. cast to float32
 *
 * STEP 6 IS THE GOOD NEWS. The reference already canonicalises eigenvector signs — "largest absolute
 * value element is positive". That was going to be an ask to the ML team and turned out to be
 * already there, which removes the sign half of the ambiguity for free. What remains is degeneracy:
 * when two eigenvalues are equal, any orthonormal basis of their shared eigenspace is a correct
 * answer, the sign rule does not disambiguate a rotation, and numpy's divide-and-conquer and our QL
 * iteration may land on different bases. That is measured by the parity harness, not argued about
 * here.
 *
 * ONE PLACE THIS DELIBERATELY DOES NOT COPY THE REFERENCE'S ARITHMETIC. The reference forms B with
 * two dense matrix multiplications by H. Since H = I - J/N, that product is exactly the
 * double-centring identity
 *     B[i][j] = -0.5 * (D2[i][j] - rowMean[i] - colMean[j] + grandMean)
 * so this computes it directly: same value, two O(n^2) passes instead of two O(n^3) matmuls. Bit
 * parity with numpy was never available anyway — its matmul is blocked BLAS with FMA, which no
 * hand-written loop reproduces — and the output is cast to float32, which is ~1e-7 relative and
 * swallows the difference. The harness confirms this rather than assuming it.
 *
 * MDS RUNS ON THE PADDED MATRIX. The caller must hand this the full [max_species, max_species]
 * matrix, zeros included, because the reference does: the padded region participates in the
 * double-centring, so coordinates depend on max_species and not only on the real taxa. Running MDS
 * on the real N and padding afterwards produces different numbers. See modelContract.js.
 */

import { symmetricEigen } from './symmetricEigen.js';
import { MDS_COMPONENTS } from './modelContract.js';

/**
 * Classical MDS coordinates for a distance matrix.
 *
 * @param {Float64Array|number[]} dist row-major n*n distance matrix (the PADDED one)
 * @param {number} n
 * @param {number} [nComponents]
 * @returns {Float32Array} n * nComponents, row-major — float32 to match the reference's final cast
 */
export function computeMdsCoordinates(dist, n, nComponents = MDS_COMPONENTS) {
	const coords = new Float32Array(n * nComponents);
	// `if N <= n_components: return zeros`. Note <=, not <: a 4-taxon alignment with 4 components
	// gets all-zero coordinates, which is the reference's behaviour and not an edge case to improve.
	if (n <= nComponents) return coords;

	// D2 = D ** 2, WITH THE DISTANCES FIRST ROUNDED TO FLOAT32.
	//
	// That rounding is not a detail. The reference receives `dist_tensor.numpy()`, and dist_tensor is
	// `torch.zeros(max_species, max_species, dtype=torch.float32)` — so the values it squares are
	// float32, widened back to float64 by its own `.astype(np.float64)` on the very next line. Passing
	// full float64 distances here produces a DIFFERENT MDS, and not subtly:
	//
	//   Real distance matrices are wildly ill-conditioned for this purpose. On a measured 135-taxon
	//   DM3 tree the eigenvalues run 1.26e7, 1.52e5, 1.30e2, 2.99e-1 — seven orders of magnitude from
	//   first to fourth. Squared distances reach ~1e6, so a float32 rounding of ~1e-7 relative is an
	//   ABSOLUTE perturbation of ~0.1 in D2, which is comparable to the fourth eigenvalue itself.
	//   Components 2 and 3 then differ by 40-99%. Feeding float64 here disagreed with the reference on
	//   5 of 270 real trees; rounding to float32 first is what makes them agree.
	//
	// Squaring also makes negative distances positive, so a negative branch length does not break MDS
	// the way it breaks the reference's density term — it silently becomes a positive distance of the
	// same magnitude. Worth knowing when reading coordinates from a bad tree.
	const D2 = new Float64Array(n * n);
	for (let i = 0; i < n * n; i++) {
		const v = Math.fround(dist[i]);
		D2[i] = v * v;
	}

	// Double-centre: B = -0.5 * (H @ D2 @ H), computed via row/column means. D2 is symmetric so the
	// column means equal the row means, but they are computed separately anyway — the input is only
	// assumed symmetric, and an asymmetric one should degrade predictably rather than silently.
	const rowMean = new Float64Array(n);
	const colMean = new Float64Array(n);
	let grand = 0;
	for (let i = 0; i < n; i++) {
		let s = 0;
		for (let j = 0; j < n; j++) s += D2[i * n + j];
		rowMean[i] = s / n;
		grand += s;
	}
	grand /= n * n;
	for (let j = 0; j < n; j++) {
		let s = 0;
		for (let i = 0; i < n; i++) s += D2[i * n + j];
		colMean[j] = s / n;
	}

	const B = new Float64Array(n * n);
	for (let i = 0; i < n; i++) {
		for (let j = 0; j < n; j++) {
			B[i * n + j] = -0.5 * (D2[i * n + j] - rowMean[i] - colMean[j] + grand);
		}
	}

	const { values, vectors } = symmetricEigen(B, n);

	// eigh returns ascending; the reference reorders descending. Only the top nComponents are used,
	// but the SIGN CONVENTION in the reference runs over every column, so it is applied per component
	// as they are read rather than to the whole matrix — same result, nComponents passes instead of n.
	for (let c = 0; c < nComponents; c++) {
		const src = n - 1 - c; // descending order
		const value = values[src];
		if (!(value > 0)) continue; // `if val > 0` — non-positive components stay zero

		// Sign convention: the largest-magnitude entry of the column must be positive. np.argmax
		// takes the FIRST maximum on a tie, so `>` and not `>=` here.
		let maxAbs = -1;
		let maxIdx = 0;
		for (let i = 0; i < n; i++) {
			const a = Math.abs(vectors[i * n + src]);
			if (a > maxAbs) {
				maxAbs = a;
				maxIdx = i;
			}
		}
		// np.sign(0) is 0, and `if sign < 0` is then false — an all-zero column is left alone rather
		// than negated. Matching that matters only for degenerate input, but it is free to match.
		const flip = vectors[maxIdx * n + src] < 0 ? -1 : 1;

		const scale = Math.sqrt(value);
		for (let i = 0; i < n; i++) {
			coords[i * nComponents + c] = Math.fround(flip * vectors[i * n + src] * scale);
		}
	}

	return coords;
}
