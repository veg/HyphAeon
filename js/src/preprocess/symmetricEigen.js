/**
 * WHY THIS FILE EXISTS
 *
 * Ported verbatim from datamonkey3 (main@fac1330) src/lib/services/axomeme/symmetricEigen.js.
 * Function bodies unchanged; it imports nothing.
 *
 * It is the JavaScript stand-in for the one library call `hyphaeon/dataset.py` makes that has no
 * equivalent in a browser: `np.linalg.eigh(B)` inside `compute_mds_coordinates` (dataset.py:606).
 * `torch.linalg.eigh` has no ONNX lowering either, which is why `mds_coords` is a graph INPUT and
 * why this routine has to exist on the JS side at all rather than being folded into the model.
 *
 * ONE DIVERGENCE WORTH KNOWING BEFORE READING A PARITY REPORT: dataset.py takes a different code
 * path entirely above 500 taxa — `scipy.sparse.linalg.eigsh` on a matrix-free double-centring
 * operator, `k=4`, `which='LA'`, `maxiter=300` (dataset.py:583-600), falling back to dense eigh only
 * on an exception. Lanczos with a capped iteration count is an APPROXIMATION; this file is always
 * the dense QL. So above 500 taxa the two sides differ by more than eigenvector convention, and the
 * gap is a property of scipy's iteration rather than of anything here. The header below explains
 * what parity can and cannot be promised even in the dense case.
 */

/**
 * symmetricEigen.js — eigendecomposition of a real symmetric matrix.
 *
 * WHY THIS IS HERE AT ALL. AxoMEME's `mds_coords` is a model INPUT, not something the ONNX graph
 * computes, because `torch.linalg.eigh` has no ONNX lowering — verified directly: the identical
 * module exports fine with eigh removed and fails with it present. So the browser has to do its own
 * eigendecomposition of the [max_species, max_species] double-centred matrix, which at the default
 * max_species is 512 x 512.
 *
 * THE ALGORITHM is Householder tridiagonalisation followed by implicit-shift QL — `tred2`/`tql2`,
 * the EISPACK routines that JAMA and Numerical Recipes both carry, and mathematically what LAPACK's
 * symmetric drivers reduce to. It is chosen over Jacobi for cost: Jacobi on 512 x 512 needs roughly
 * 10 sweeps of ~131k rotations and lands in the seconds; tred2/tql2 is ~(4/3)n^3 once plus cheap
 * iteration, and runs in a few hundred milliseconds.
 *
 * WHAT PARITY CAN AND CANNOT BE PROMISED HERE, stated plainly because it is the crux of the whole
 * browser port:
 *
 *   - EIGENVALUES agree with LAPACK to near machine precision. They are a property of the matrix.
 *   - EIGENVECTORS are only unique up to sign, and within a DEGENERATE eigenspace not even up to
 *     sign — any orthonormal basis of that subspace is equally correct. numpy uses divide-and-conquer
 *     (dsyevd); this is QL. Two correct implementations can therefore disagree, and no amount of care
 *     in this file changes that.
 *
 * The sign half is fixed by convention downstream (mds.js applies the reference's own rule). The
 * degeneracy half is not fixable in principle, only measurable — which is why the parity harness runs
 * over real trees rather than a proof being attempted here.
 *
 * The routine is transcribed rather than invented, and its tests check the properties that actually
 * matter (A·V = V·Λ, orthonormality, reconstruction) rather than comparing against numbers this file
 * produced.
 */

/** Machine epsilon for float64, as the reference routines define it. */
const EPS = Math.pow(2, -52);

/**
 * Eigendecomposition of a real symmetric matrix.
 *
 * @param {Float64Array|number[]} matrix row-major, n*n, assumed symmetric (upper triangle unused)
 * @param {number} n
 * @returns {{values: Float64Array, vectors: Float64Array}} eigenvalues ASCENDING (matching
 *   numpy.linalg.eigh), eigenvectors row-major with column j being the vector for values[j]
 */
export function symmetricEigen(matrix, n) {
	const V = Float64Array.from(matrix);
	const d = new Float64Array(n);
	const e = new Float64Array(n);

	tred2(V, d, e, n);
	tql2(V, d, e, n);

	return { values: d, vectors: V };
}

/**
 * Householder reduction to tridiagonal form.
 *
 * On entry V holds the symmetric matrix; on exit V holds the accumulated orthogonal transformation,
 * d the diagonal and e the sub-diagonal of the tridiagonal form.
 */
function tred2(V, d, e, n) {
	for (let j = 0; j < n; j++) d[j] = V[(n - 1) * n + j];

	for (let i = n - 1; i > 0; i--) {
		let scale = 0;
		let h = 0;
		for (let k = 0; k < i; k++) scale += Math.abs(d[k]);

		if (scale === 0) {
			e[i] = d[i - 1];
			for (let j = 0; j < i; j++) {
				d[j] = V[(i - 1) * n + j];
				V[i * n + j] = 0;
				V[j * n + i] = 0;
			}
		} else {
			for (let k = 0; k < i; k++) {
				d[k] /= scale;
				h += d[k] * d[k];
			}
			let f = d[i - 1];
			let g = Math.sqrt(h);
			if (f > 0) g = -g;
			e[i] = scale * g;
			h -= f * g;
			d[i - 1] = f - g;
			for (let j = 0; j < i; j++) e[j] = 0;

			for (let j = 0; j < i; j++) {
				f = d[j];
				V[j * n + i] = f;
				g = e[j] + V[j * n + j] * f;
				for (let k = j + 1; k <= i - 1; k++) {
					g += V[k * n + j] * d[k];
					e[k] += V[k * n + j] * f;
				}
				e[j] = g;
			}

			f = 0;
			for (let j = 0; j < i; j++) {
				e[j] /= h;
				f += e[j] * d[j];
			}
			const hh = f / (h + h);
			for (let j = 0; j < i; j++) e[j] -= hh * d[j];

			for (let j = 0; j < i; j++) {
				f = d[j];
				g = e[j];
				for (let k = j; k <= i - 1; k++) V[k * n + j] -= f * e[k] + g * d[k];
				d[j] = V[(i - 1) * n + j];
				V[i * n + j] = 0;
			}
		}
		d[i] = h;
	}

	// Accumulate the transformations.
	for (let i = 0; i < n - 1; i++) {
		V[(n - 1) * n + i] = V[i * n + i];
		V[i * n + i] = 1;
		const h = d[i + 1];
		if (h !== 0) {
			for (let k = 0; k <= i; k++) d[k] = V[k * n + (i + 1)] / h;
			for (let j = 0; j <= i; j++) {
				let g = 0;
				for (let k = 0; k <= i; k++) g += V[k * n + (i + 1)] * V[k * n + j];
				for (let k = 0; k <= i; k++) V[k * n + j] -= g * d[k];
			}
		}
		for (let k = 0; k <= i; k++) V[k * n + (i + 1)] = 0;
	}
	for (let j = 0; j < n; j++) {
		d[j] = V[(n - 1) * n + j];
		V[(n - 1) * n + j] = 0;
	}
	V[(n - 1) * n + (n - 1)] = 1;
	e[0] = 0;
}

/** Symmetric tridiagonal QL with implicit shifts. Leaves eigenvalues in d, ASCENDING. */
function tql2(V, d, e, n) {
	for (let i = 1; i < n; i++) e[i - 1] = e[i];
	e[n - 1] = 0;

	let f = 0;
	let tst1 = 0;

	for (let l = 0; l < n; l++) {
		tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
		let m = l;
		while (m < n) {
			if (Math.abs(e[m]) <= EPS * tst1) break;
			m++;
		}

		if (m > l) {
			do {
				let g = d[l];
				let p = (d[l + 1] - g) / (2 * e[l]);
				let r = Math.hypot(p, 1);
				if (p < 0) r = -r;
				d[l] = e[l] / (p + r);
				d[l + 1] = e[l] * (p + r);
				const dl1 = d[l + 1];
				let h = g - d[l];
				for (let i = l + 2; i < n; i++) d[i] -= h;
				f += h;

				p = d[m];
				let c = 1;
				let c2 = c;
				let c3 = c;
				const el1 = e[l + 1];
				let s = 0;
				let s2 = 0;
				for (let i = m - 1; i >= l; i--) {
					c3 = c2;
					c2 = c;
					s2 = s;
					g = c * e[i];
					h = c * p;
					r = Math.hypot(p, e[i]);
					e[i + 1] = s * r;
					s = e[i] / r;
					c = p / r;
					p = c * d[i] - s * g;
					d[i + 1] = h + s * (c * g + s * d[i]);
					for (let k = 0; k < n; k++) {
						h = V[k * n + (i + 1)];
						V[k * n + (i + 1)] = s * V[k * n + i] + c * h;
						V[k * n + i] = c * V[k * n + i] - s * h;
					}
				}
				p = (-s * s2 * c3 * el1 * e[l]) / dl1;
				e[l] = s * p;
				d[l] = c * p;
			} while (Math.abs(e[l]) > EPS * tst1);
		}
		d[l] += f;
		e[l] = 0;
	}

	// Sort ascending, carrying the eigenvectors with their values. Selection sort: n is at most a few
	// hundred here and the swap has to move a whole column, so the simple algorithm is the right one.
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
			for (let j = 0; j < n; j++) {
				const t = V[j * n + i];
				V[j * n + i] = V[j * n + k];
				V[j * n + k] = t;
			}
		}
	}
}
