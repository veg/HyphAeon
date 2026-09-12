/**
 * WHY THIS FILE EXISTS
 *
 * The MODEL-BASED half of the ChronAeon dating pillar — the two estimators src/dating.js's header
 * lists under "everything the transformer feeds" and declines to port, plus the covariance kernel
 * and the REML profile they are built on. Ported function by function from `hyphaeon/dating.py`
 * (branch feat/dating-model, 93645e3) and `hyphaeon/splits.py`. Every line reference below is
 * dating.py unless it says otherwise.
 *
 *   compute_neural_covariance_kernel      78-118   -> computeNeuralCovarianceKernel
 *   optimize_latent_convex_hull_root     701-832   -> optimizeLatentConvexHullRoot
 *   run_pgls_dating                     1299-1473  -> runPglsDating
 *   estimate_reml_pagel_lambda          1476-1547  -> estimateRemlPagelLambda
 *   the isometric-calibration distance block of run_mrca_dating
 *                                       2586-2596  -> pairwiseAcgtHammingMatrix
 *
 * It is still arithmetic and nothing else: the two [N, N] and [N, D] matrices the model produces
 * arrive as ARGUMENTS. What computes them (a session, a fetch list, a batch size, a divisor of
 * L·num_layers) is the runtime's, and what a reader is told about them is the application's.
 * `splits.py::extract_cross_taxa_attentions_and_embeddings` has no counterpart here on purpose.
 *
 * RECORD KEYS ARE THE REFERENCE'S, snake_case, in the reference's own insertion order, as
 * src/dating.js already argues: dating.py:3168 exports `run_pgls_dating`'s dict minus
 * `residuals`/`fitted`/`times`, so that key set in that order IS the download contract.
 *
 * ============================== WHERE THE PRECISION ACTUALLY GOES ==============================
 *
 * THE KERNEL IS THE ONE PLACE THIS LIBRARY CANNOT BE BIT-EXACT, and it is worth being blunt about
 * why. `cross_attn` and `taxa_repr` are float32 (splits.py:83-84), so numpy's `a_centered @
 * a_centered.T` at dating.py:96 and `z_centered @ z_centered.T` at 107 are float32 BLAS matmuls —
 * `sgemm`, whose accumulation order is the BLAS implementation's, blocked and vectorised, and not
 * a thing any JavaScript can reproduce. This port accumulates in float64.
 *
 * MEASURED on the acceptance case (korber, 143 taxa, CPU reference): recomputing the reference's own
 * kernel in float64 from the same float32 inputs moves it by max |ΔK| = 1.954e-6, against entries of
 * magnitude 1 (mean |K_ij| off the diagonal = 0.532). Downstream that is worth about 1.5e-4 years of
 * t_MRCA — from the phase's measured chain sensitivity, |ΔK| 1.04e-2 moves t_MRCA by 0.819 years,
 * and the relationship is linear well below that. So the kernel is held at the fixtures' 1e-5 model
 * class and everything DOWNSTREAM of it is held at 1e-9 by feeding the reference's own kernel in,
 * which is exactly how fixtures/dating is layered. A test that fed this port's kernel into its own
 * PGLS and compared the two would be comparing the port with itself.
 *
 * Two float64 choices with the same justification and the same measurement:
 *   - `alpha` (752-753) divides a float64 numerator by a FLOAT32 denominator, because
 *     `np.linalg.norm` on a float32 matrix returns float32. Float64 throughout costs 7.96e-8
 *     RELATIVE on the acceptance case (0.05406844325137635 against the reference's
 *     0.05406843894704035).
 *   - the 250 Adam steps run in float32 under torch (769-770). JavaScript has no float32 arithmetic,
 *     only `Math.fround` per operation, and torch's reductions inside `torch.norm` / `torch.sum` are
 *     vectorised and unreproducible anyway. MEASURED: the whole float64 trajectory lands at max
 *     |Δw| = 3.823e-7, |Δz_root| = 2.611e-7, |Δdists| = 1.190e-7 and Δtemporal_r = 1.500e-8 against
 *     the reference's, with the SAME six anchor taxa in the SAME order — which is the part a reader
 *     sees. Chasing float32 would add `fround` noise to every line and still not match.
 *
 * ================== THE LATENT ROOT IS A TRAJECTORY, NOT AN OPTIMUM ==================
 *
 * `optimize_latent_convex_hull_root` runs EXACTLY 250 Adam steps at lr 0.05 with no stopping
 * criterion (778, 781-795), from a closed-form initialisation, on a non-convex surface, and it is
 * NOT converged: MEASURED upstream, one extra step moves t_MRCA by 1.2e-3 years and 250 more move it
 * by 0.26 years while the correlation is still climbing. The reference's answer is therefore defined
 * by its trajectory. That is why this port replays the trajectory step for step — a hand-derived
 * gradient and a hand-rolled Adam — rather than calling a derivative-free minimiser, which would
 * find a better optimum and reproduce nothing. It is also why the fixture carries the weights at ten
 * intermediate steps and not only at 250: a port that lands on step 250 by luck with the wrong Adam
 * breaks on the next alignment.
 *
 * THE GRADIENT IS HAND-DERIVED, and it is four lines of chain rule (no autograd, no Jacobian ever
 * formed). With w = softmax(v), z_r = Σ w_i z_i, d_k = ‖z_k − z_r‖ and E the eligible set of size m:
 *
 *      ∂cov/∂d_k   = t_c,k / m                     (cov = mean(t_c · d_c), and Σ t_c = 0)
 *      ∂s_d/∂d_k   = d_c,k / ((m − 1)·s_d)         (torch.std is UNBIASED, ddof = 1)
 *      ∂corr/∂d_k  = [∂cov/∂d_k·den − cov·std_t·∂s_d/∂d_k] / den²,   den = std_t·s_d + 1e-8
 *      ∂L/∂v_j     = w_j (z_j − z_r)·G,  G = −Σ_{k∈E} (∂L/∂d_k)·(z_k − z_r)/d_k
 *
 * VERIFIED against torch autograd upstream at max |Δ| 2.98e-15, and the whole 250-step replay
 * against `torch.optim.Adam` at the figures quoted above.
 *
 * ============================ UPSTREAM QUIRKS REPLICATED AND FLAGGED ============================
 *
 *   B11 (786-787) `cov = torch.mean(t_c · d_c)` divides by m while BOTH standard deviations use
 *      torch's default `correction = 1`. The maximised quantity is therefore Pearson's r scaled by
 *      (m−1)/m, not r. It is monotone in r so the argmax is unmoved, and `temporal_r` at 803 is
 *      recomputed with `np.corrcoef`, so the REPORTED correlation is a true r — but the loss value
 *      is not, and a port that "fixed" it would take different Adam steps. Replicated.
 *   B12 (744-745) fewer than three ELIGIBLE taxa silently resets the mask to all-true, so a caller
 *      who masked almost everything gets an unmasked answer with no signal that it happened.
 *   B13 (773) ineligible taxa get logit −1e4 rather than −Infinity, stay in the parameter vector and
 *      stay in the optimiser. In float32 (and in float64) their softmax weight underflows to exactly
 *      0, so their gradient is exactly 0 and Adam leaves them alone — but they are still there, and
 *      removing them would change the softmax normalisation. Kept.
 *   B14 (752) `alpha` is fitted over ALL pairs, eligible or not, while the correlation it calibrates
 *      is measured over the eligible ones only.
 *   B15 (809-818) the anchor table breaks when `w < 0.01` AND at least three rows are already out,
 *      so it emits everything down to the first sub-1 % entry AFTER the third. On the acceptance case
 *      that is six rows, the last at 0.0255 — not "the rows above 1 %", which would be five.
 *   B16 (1330-1336, 1466) `run_pgls_dating` takes BOTH `ridge` and `pagel_lambda`. When
 *      `pagel_lambda` is not None — which is the only way the CLI calls it (2799-2802) — the `ridge`
 *      argument NEVER reaches the covariance; it is discarded and the returned `'ridge'` is
 *      recomputed as `1 − eff_lam`. MEASURED on the acceptance case: the CLI PRINTS
 *      `nugget ridge = 0.2000` (its own clip of 1 − λ* into [0.01, 0.20]) and the JSON ships
 *      `pgls.ridge = 0.301574527827`. Two different numbers for one name; a page must not show both.
 *   B17 (1505, 1347) the same matrix is eigendecomposed TWICE per fit. `estimate_reml_pagel_lambda`
 *      returns `w_K` and `V` so the caller can reuse them, but the reuse branch at 1331-1336 fires
 *      only when `ridge == 'auto'`, and the CLI passes a float. Replicated — `runPglsDating` accepts
 *      the decomposition through `options.eigen` so an application CAN share it, and takes its own
 *      when it is not given, which is the reference's own code path.
 *   B18 (1375-1376 vs 1526) PGLS computes the residual sum as `Σ(u − Zβ)²·inv_w` and REML computes
 *      the same quantity as `Σu²·inv_w − βᵀ·XᵀC⁻¹d`. Algebraically equal, differently rounded. Both
 *      are written as the reference writes them; do not unify them.
 *   B19 (81, 1276 of splits.py's alias) `compute_neural_covariance_kernel` accepts `mds_coords` and
 *      never reads it, and `compute_attention_covariance_kernel` (121-127) is a one-line alias for
 *      it. Neither is ported: a parameter that is never read is not a contract.
 *
 * ============================== WHAT IS DELIBERATELY NOT HERE ==============================
 *
 *   - `run_pgls_dating`'s TRUNCATED-LANCZOS branch (1340-1344), taken at n > 2500:
 *     `scipy.sparse.linalg.eigsh(K.astype(float32), k=min(n-2, 250), which='LM', tol=1e-4)`. That is
 *     ARPACK with an implicitly restarted Lanczos iteration on a float32 cast — an APPROXIMATION
 *     whose answer depends on its own restart schedule, and no JavaScript reproduces it.
 *     `runPglsDating` REFUSES above `PGLS_DENSE_EIGEN_MAX` rather than silently taking the dense
 *     path and returning a different number under the same name.
 *   - `estimate_reml_pagel_lambda`'s stratified subsample (1493-1501), taken at n > 2000, and the
 *     `OPTIMAL_REML_SUBSAMPLED` status it returns. Same refusal, same reason: it is reachable only
 *     above a cohort size at which the branch above has already refused.
 *   - the `poisson`, `residual-boot`/`wild` and `jackknife`/`loocv` intervals (1420-1436). Each
 *     needs numpy's PCG64 stream, exactly as src/dating.js records for the OLS fit, and each first
 *     materialises a dense N×N `C_inv` the spectral path exists to avoid. `runPglsDating` refuses
 *     them through the same `CI_METHODS_UNPORTED` list `runOlsDating` uses.
 *   - `tune_ridge_for_pgls` (1548-1560), a wrapper that calls `estimate_reml_pagel_lambda` and drops
 *     `w_K`/`V`.
 *   - the orchestration around all of this (2540-2600, 2725-2810): which distance mode `auto`
 *     resolves to, the < 50 % coverage holdout and anchor mask, the taxon subsetting, the promotion
 *     of latent over a weak tree. Those know about a reader's data, not about arithmetic.
 *
 * A SUBSETTING ORDER THAT IS EASY TO GET WRONG AND IMPOSSIBLE TO SEE. The reference builds the
 * kernel on ALL alignment taxa (2568) and only THEN slices it to the dated ones (2576) and again to
 * the training ones (2781). The centring at 95 and 106 therefore uses rows a later step drops.
 * Centre-then-subset is not subset-then-centre, and a caller who slices first gets a plausible wrong
 * number. This file takes the matrices it is given; the ORDER is the caller's to get right, and the
 * application's tests are where that is pinned.
 */

import { tPpf } from './numeric/special.js';
import { numpyPairwiseSum } from './numeric/reduce.js';
import { fminbound } from './numeric/optimize.js';
import { symmetricEigen } from './preprocess/symmetricEigen.js';
import { CI_METHODS_UNPORTED, computeDeltaMrcaInterval, computeFiellerMrcaInterval, DATING_STATUS } from './dating.js';

const identity = (/** @type {number} */ v) => v;

/** `float(np.sum(a))` in float64: numpy's pairwise order, not a left fold. */
function sum64(a, n = a.length) {
	return numpyPairwiseSum(a, 0, n, identity);
}

/** `float(np.mean(a))`. */
function mean64(a, n = a.length) {
	return sum64(a, n) / n;
}

/** A Float64Array view of any ArrayLike, copying only when it is not already one. */
function f64(a) {
	return a instanceof Float64Array ? a : Float64Array.from(a);
}

/**
 * Above this many taxa `run_pgls_dating` switches to `scipy.sparse.linalg.eigsh` (1340) and
 * `estimate_reml_pagel_lambda` to a 1500-row stratified subsample (1493). Both are refused here
 * rather than approximated; see the header. The reference itself declines the neural route entirely
 * above 1500 sequences without a tree (2732), so a cohort that reaches this bound has already been
 * refused upstream by the application.
 */
export const PGLS_DENSE_EIGEN_MAX = 2500;
/** `estimate_reml_pagel_lambda`'s own subsampling threshold (1493). */
export const REML_DENSE_MAX = 2000;

/** The two `status` values `estimate_reml_pagel_lambda` can return (1543, 1500). */
export const REML_STATUS = Object.freeze({
	OPTIMAL_REML: 'OPTIMAL_REML',
	OPTIMAL_REML_SUBSAMPLED: 'OPTIMAL_REML_SUBSAMPLED',
});

/** Pagel's lambda is clipped into this box before it ever reaches the covariance (1352). */
export const PAGEL_LAMBDA_BOUNDS = Object.freeze([0.001, 0.999]);

// =================================================================================================
// the covariance kernel
// =================================================================================================

/**
 * One half of `compute_neural_covariance_kernel`: column-centre a row-major [n, d] block, form the
 * row Gram matrix, and normalise it to a correlation matrix.
 *
 * `np.divide(cov, denom, where=(denom > 1e-12), out=np.eye(n))` (99) leaves the IDENTITY's value
 * where the guard fails — 1 on the diagonal and 0 off it, NOT a zero fill and NOT the quotient. That
 * `out=` is the single easiest line in this file to mis-port, and a matrix that silently loses its
 * unit diagonal on a constant row would still look like a correlation matrix.
 *
 * @param {ArrayLike<number>} block row-major, n*d
 * @param {number} n rows (taxa)
 * @param {number} d columns
 * @returns {Float64Array} row-major n*n correlation matrix with a unit diagonal
 */
function centredRowCorrelation(block, n, d) {
	// column means across taxa (95, 106): axis 0, keepdims
	const colMean = new Float64Array(d);
	const col = new Float64Array(n);
	for (let j = 0; j < d; j++) {
		for (let i = 0; i < n; i++) col[i] = block[i * d + j];
		colMean[j] = mean64(col);
	}
	const centred = new Float64Array(n * d);
	for (let i = 0; i < n; i++) {
		for (let j = 0; j < d; j++) centred[i * d + j] = block[i * d + j] - colMean[j];
	}

	// cov = centred @ centred.T (96, 107)
	const cov = new Float64Array(n * n);
	const term = new Float64Array(d);
	for (let i = 0; i < n; i++) {
		for (let k = i; k < n; k++) {
			for (let j = 0; j < d; j++) term[j] = centred[i * d + j] * centred[k * d + j];
			const v = sum64(term);
			cov[i * n + k] = v;
			cov[k * n + i] = v;
		}
	}

	// std = sqrt(maximum(diag(cov), 1e-12)) (98, 109)
	const std = new Float64Array(n);
	for (let i = 0; i < n; i++) std[i] = Math.sqrt(Math.max(cov[i * n + i], 1e-12));

	const K = new Float64Array(n * n);
	for (let i = 0; i < n; i++) {
		for (let k = 0; k < n; k++) {
			const denom = std[i] * std[k];
			// where=(denom > 1e-12), out=np.eye(n): the identity's entry survives, not a zero
			K[i * n + k] = denom > 1e-12 ? cov[i * n + k] / denom : i === k ? 1.0 : 0.0;
		}
	}
	for (let i = 0; i < n; i++) K[i * n + i] = 1.0; // fill_diagonal (100, 111)
	return K;
}

/**
 * `compute_neural_covariance_kernel(cross_attn, taxa_repr=None, mds_coords=None)`, dating.py:78-118.
 *
 * A Pearson correlation matrix over taxa of their column-centred cross-taxa attention profiles,
 * averaged 50/50 with the same construction over their column-centred embeddings. Column centring
 * removes each TARGET taxon's background baseline (and, for the embeddings, the dominant common
 * activation direction) rather than each source's — the mean at 95 and 106 is over `axis=0`, which
 * subtracts a column mean from every row.
 *
 * `taxa_repr` is dropped SILENTLY when its row count does not match `n` (104), which is a branch
 * rather than an error upstream and is replicated as one. When it is absent the kernel is the
 * attention half alone (115); the application always supplies it, so that arm is dead on our path
 * and is ported anyway because it is three lines.
 *
 * The docstring's "zero negative eigenvalues" is not ENFORCED here and does not need to be: an
 * average of two correlation matrices is PSD by construction. MEASURED on the acceptance case, the
 * eigenvalues of K run 0.0031309890572192036 to 82.19852224418779, strictly positive. The clamps
 * that do exist are downstream, at 1347 and 1490.
 *
 * @param {ArrayLike<number>} crossAttn row-major n*n, the site-averaged taxon-by-taxon attention
 * @param {number} n
 * @param {ArrayLike<number>|null} [taxaRepr] row-major rows*embedDim
 * @param {number} [embedDim]
 * @returns {Float64Array} row-major n*n, unit diagonal
 */
export function computeNeuralCovarianceKernel(crossAttn, n, taxaRepr = null, embedDim = 0) {
	if (crossAttn.length !== n * n) {
		throw new RangeError(`cross_attn must be ${n}x${n} (got ${crossAttn.length} entries).`);
	}
	const kAttn = centredRowCorrelation(crossAttn, n, n);

	// `taxa_repr is not None and len(taxa_repr) == n` (104): a row count, not a byte count
	if (taxaRepr && embedDim > 0 && taxaRepr.length / embedDim === n) {
		const kEmb = centredRowCorrelation(taxaRepr, n, embedDim);
		const K = new Float64Array(n * n);
		for (let i = 0; i < n * n; i++) K[i] = 0.5 * kAttn[i] + 0.5 * kEmb[i]; // 113
		for (let i = 0; i < n; i++) K[i * n + i] = 1.0; // 117, redundant and harmless
		return K;
	}
	for (let i = 0; i < n; i++) kAttn[i * n + i] = 1.0;
	return kAttn;
}

// =================================================================================================
// the shared spectral machinery
// =================================================================================================

/**
 * `w_raw, v = la.eigh(cov); w_pos = np.maximum(w_raw, 0.0)` (1346-1347, 1505-1506).
 *
 * The eigenVECTORS are only unique up to sign, and within a degenerate eigenspace not even up to
 * that — but unlike the MDS path (see preprocess/symmetricEigen.js's header, and MDS_SIGN.md), NOTHING
 * here is sensitive to the choice. Every quantity PGLS and REML extract has the form
 * `aᵀ·V·diag(f(w))·Vᵀ·b` or `Σ g(w)`, and `V diag(f(w)) Vᵀ` is a function of the MATRIX, not of the
 * basis: column sign flips and rotations within a degenerate subspace cancel identically. VERIFIED
 * upstream by flipping a random half of the eigenvector columns on the acceptance case — `XᵀC⁻¹X`
 * moved by exactly 0.000e+00. The model is never shown the eigenvectors, which is the whole
 * difference from MDS.
 *
 * @param {ArrayLike<number>} cov row-major n*n
 * @param {number} n
 * @returns {{values: Float64Array, vectors: Float64Array}} values ASCENDING and clamped at 0
 */
function eighClamped(cov, n) {
	const { values, vectors } = symmetricEigen(f64(cov), n);
	for (let i = 0; i < n; i++) values[i] = Math.max(values[i], 0.0);
	return { values, vectors };
}

/**
 * `Z = V.T @ X`, `u = V.T @ d`, `v_one = np.sum(V, axis=0)` (1361-1363, 1509-1510).
 *
 * `X = [t − t_ref, 1]`, so `Z` is n*2 row-major and `Z[:, 1]` is `v_one` — the reference forms them
 * separately and so does this, because REML never needs `v_one` and the extra column costs nothing.
 *
 * @param {Float64Array} vectors row-major n*n, column j the eigenvector for values[j]
 * @param {Float64Array} x centred times
 * @param {Float64Array} dists
 * @param {number} n
 * @returns {{Z: Float64Array, u: Float64Array, vOne: Float64Array}}
 */
function projectOntoEigenbasis(vectors, x, dists, n) {
	const Z = new Float64Array(n * 2);
	const u = new Float64Array(n);
	const vOne = new Float64Array(n);
	const colX = new Float64Array(n);
	const colD = new Float64Array(n);
	const colV = new Float64Array(n);
	for (let j = 0; j < n; j++) {
		for (let i = 0; i < n; i++) {
			const vij = vectors[i * n + j];
			colV[i] = vij;
			colX[i] = vij * x[i];
			colD[i] = vij * dists[i];
		}
		Z[j * 2] = sum64(colX);
		Z[j * 2 + 1] = sum64(colV); // the ones column of X projects to the column sum of V
		u[j] = sum64(colD);
		vOne[j] = Z[j * 2 + 1];
	}
	return { Z, u, vOne };
}

/**
 * `Xt_Cinv_X = Z.T @ (Z * inv_w[:, None])` and `Xt_Cinv_d = (Z * inv_w[:, None]).T @ u`
 * (1366-1367, 1517-1519). Both are EXACT: the projection is orthonormal, so this 2x2 and this
 * 2-vector are `XᵀC⁻¹X` and `XᵀC⁻¹d` without a dense inverse ever existing.
 *
 * @returns {{A: number[], b: number[]}} A row-major 2x2 [a00, a01, a10, a11]
 */
function normalEquations(Z, u, invW, n) {
	const t00 = new Float64Array(n);
	const t01 = new Float64Array(n);
	const t11 = new Float64Array(n);
	const b0 = new Float64Array(n);
	const b1 = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		const z0 = Z[i * 2];
		const z1 = Z[i * 2 + 1];
		const s0 = z0 * invW[i];
		const s1 = z1 * invW[i];
		t00[i] = z0 * s0;
		t01[i] = z0 * s1;
		t11[i] = z1 * s1;
		b0[i] = s0 * u[i];
		b1[i] = s1 * u[i];
	}
	const a00 = sum64(t00);
	const a01 = sum64(t01);
	const a11 = sum64(t11);
	return { A: [a00, a01, a01, a11], b: [sum64(b0), sum64(b1)] };
}

/** `la.solve(A, b)` on a 2x2, by the adjugate. Throws on a singular A, as scipy raises. */
function solve2x2(A, b) {
	const det = A[0] * A[3] - A[1] * A[2];
	if (!Number.isFinite(det) || det === 0) throw new RangeError('singular 2x2 normal equations');
	return [(A[3] * b[0] - A[1] * b[1]) / det, (A[0] * b[1] - A[2] * b[0]) / det];
}

/** `la.inv(A)` on a 2x2, row-major. */
function inv2x2(A) {
	const det = A[0] * A[3] - A[1] * A[2];
	if (!Number.isFinite(det) || det === 0) throw new RangeError('singular 2x2 normal equations');
	return [A[3] / det, -A[1] / det, -A[2] / det, A[0] / det];
}

// =================================================================================================
// Pagel's lambda by profile REML
// =================================================================================================

/**
 * `estimate_reml_pagel_lambda(times, dists, cov_matrix)`, dating.py:1476-1547.
 *
 * Maximise the profile restricted likelihood of `C(λ) = λ·K + (1 − λ)·I` over λ ∈ [0.001, 0.999].
 * The eigendecomposition is paid ONCE, outside the objective (1505); each trial λ then costs O(n),
 * because the eigenvalues of C(λ) are `λ·w_K + (1 − λ)` and every inner product is already projected.
 *
 *      −2·log L_REML(λ) = (n − 2)·log σ² + Σ log w_c + log|det XᵀC⁻¹X|            (1531-1535)
 *
 * The residual sum at 1526 is the IDENTITY `uᵀC⁻¹u − βᵀ·XᵀC⁻¹d`, a different rounding path from the
 * `Σ(u − Zβ)²·inv_w` `run_pgls_dating` uses at 1376 (B18). Written as the reference writes it.
 *
 * Two sentinels, both replicated: a singular 2x2 and a non-positive residual sum each return 1e9
 * (1524, 1528), which is a finite penalty the minimiser walks away from rather than an exception.
 *
 * @param {ArrayLike<number>} times
 * @param {ArrayLike<number>} dists
 * @param {ArrayLike<number>} covMatrix row-major n*n
 * @param {{eigen?: {values: Float64Array, vectors: Float64Array}}} [options] a decomposition of
 *   `covMatrix` to reuse, since the reference throws one away per fit (B17)
 * @returns {{best_lambda: number, status: string, w_K: Float64Array, V: Float64Array,
 *   objective: (lambda: number) => number, nfev: number}}
 */
export function estimateRemlPagelLambda(times, dists, covMatrix, options = {}) {
	const t = f64(times);
	const d = f64(dists);
	const n = t.length;
	if (d.length !== n) throw new RangeError(`times and dists differ in length (${n} vs ${d.length}).`);
	if (covMatrix.length !== n * n) {
		throw new RangeError(`cov_matrix must be ${n}x${n} (got ${covMatrix.length} entries).`);
	}
	if (n > REML_DENSE_MAX) {
		throw new RangeError(
			`estimate_reml_pagel_lambda subsamples above ${REML_DENSE_MAX} taxa (dating.py:1493-1501), which returns a ` +
				`DIFFERENT status ('OPTIMAL_REML_SUBSAMPLED') from a 1500-row stratified slice; that branch is not ported ` +
				`(got N=${n}).`
		);
	}

	const tRef = mean64(t); // 1502
	const x = new Float64Array(n);
	for (let i = 0; i < n; i++) x[i] = t[i] - tRef;

	const { values: wK, vectors: V } = options.eigen ?? eighClamped(covMatrix, n);
	const { Z, u } = projectOntoEigenbasis(V, x, d, n);

	const u2 = new Float64Array(n);
	for (let i = 0; i < n; i++) u2[i] = u[i] * u[i];

	const wc = new Float64Array(n);
	const invW = new Float64Array(n);
	const logs = new Float64Array(n);
	const weighted = new Float64Array(n);

	const negReml = (lam) => {
		for (let i = 0; i < n; i++) {
			wc[i] = lam * wK[i] + (1.0 - lam); // 1515
			invW[i] = 1.0 / Math.max(wc[i], 1e-12); // 1516
		}
		const { A, b } = normalEquations(Z, u, invW, n);
		let beta;
		try {
			beta = solve2x2(A, b);
		} catch {
			return 1e9; // 1523-1524
		}
		for (let i = 0; i < n; i++) weighted[i] = u2[i] * invW[i];
		const resSs = sum64(weighted) - (beta[0] * b[0] + beta[1] * b[1]); // 1526
		if (resSs <= 0) return 1e9; // 1527-1528

		const sigma2 = resSs / Math.max(1, n - 2); // 1530
		for (let i = 0; i < n; i++) logs[i] = Math.log(Math.max(wc[i], 1e-12));
		const logDetC = sum64(logs); // 1531
		// np.linalg.slogdet of the 2x2: only the magnitude is used (1532)
		const logDetA = Math.log(Math.abs(A[0] * A[3] - A[1] * A[2]));
		return (n - 2) * Math.log(sigma2) + logDetC + logDetA; // 1535
	};

	const res = fminbound(negReml, PAGEL_LAMBDA_BOUNDS[0], PAGEL_LAMBDA_BOUNDS[1]); // 1539
	return {
		best_lambda: res.success ? res.x : 0.95, // 1540: the 0.95 fallback is the reference's
		status: REML_STATUS.OPTIMAL_REML,
		w_K: wK,
		V,
		objective: negReml,
		nfev: res.nfev,
	};
}

// =================================================================================================
// the generalised fit
// =================================================================================================

/**
 * `run_pgls_dating(times, dists, cov_matrix, ridge, pagel_lambda, t_ref, ci_method, ...)`,
 * dating.py:1299-1473.
 *
 * The same model as `runOlsDating` — `d = Xβ + ε` with `X = [t − t_ref, 1]` centred on
 * `t_ref = mean(times)` — with `Cov(ε) = σ²·C` in place of `σ²·I`. Everything downstream of
 * `cov_beta` IS the OLS path: the delta gradient, Fieller's inversion and the three-value status
 * ladder are the same algebra on a different 2x2, so `computeDeltaMrcaInterval` and
 * `computeFiellerMrcaInterval` are imported from src/dating.js rather than restated.
 *
 * What is new is the covariance and how it is applied. `C = λ·K + (1 − λ)·I` under Pagel (1352-1353)
 * or `C = K + ridge·I` additively (1356) — and only the FIRST is reachable from the CLI (B16). The
 * fit is then spectral (1361-1378): `XᵀC⁻¹X` and `XᵀC⁻¹d` are formed by projecting `X`, `d` and `1`
 * onto the eigenbasis and weighting by `1/w_c`, so no dense `C_inv`, `C_half` or `C_inv_half` is
 * ever allocated. `r2` is Buse's generalised R² (1442-1447), which reuses the same projection for a
 * C⁻¹-weighted mean.
 *
 * WHY THE CLIP AT 1352 DOES THE CONDITIONING WORK. `eff_lam ∈ [0.001, 0.999]` floors `w_c` at 0.001,
 * so `inv_w ≤ 1000` always and the `np.maximum(w_c, 1e-12)` at 1358 can never bind. MEASURED on the
 * acceptance case at λ* = 0.8591318190927182: `w_c ∈ [0.1409, 68.7]`, condition number ≈ 488.
 *
 * @param {ArrayLike<number>} times
 * @param {ArrayLike<number>} dists
 * @param {ArrayLike<number>} covMatrix row-major n*n
 * @param {{ridge?: number, pagelLambda?: number|null, tRef?: number, ciMethod?: string,
 *   eigen?: {values: Float64Array, vectors: Float64Array}}} [options]
 * @returns {Record<string, any>} the reference's key set, in the reference's order
 */
export function runPglsDating(times, dists, covMatrix, options = {}) {
	const t = f64(times);
	const d = f64(dists);
	const n = t.length;
	if (n < 3) throw new RangeError(`At least 3 observations are required for PGLS dating (got N=${n}).`); // 1320
	if (d.length !== n) throw new RangeError(`times and dists differ in length (${n} vs ${d.length}).`);
	if (covMatrix.length !== n * n) {
		throw new RangeError(`cov_matrix must be ${n}x${n} (got ${covMatrix.length} entries).`);
	}
	if (n > PGLS_DENSE_EIGEN_MAX) {
		throw new RangeError(
			`run_pgls_dating switches to scipy.sparse.linalg.eigsh above ${PGLS_DENSE_EIGEN_MAX} taxa ` +
				`(dating.py:1340-1344) — a truncated Lanczos approximation on a float32 cast, which is not ported ` +
				`(got N=${n}). Taking the dense path here would answer a different question under the same name.`
		);
	}

	const ridge = options.ridge ?? 0.05;
	const pagelLambda = options.pagelLambda ?? null;
	const ciMethod = options.ciMethod ?? 'fieller';
	const ciLower = String(ciMethod).toLowerCase();
	if (CI_METHODS_UNPORTED.includes(ciLower)) {
		throw new RangeError(
			`ci_method '${ciMethod}' is not ported: the Poisson, wild-residual and jackknife intervals each need ` +
				`numpy's PCG64 stream to reproduce and each first materialises a dense N x N C_inv (dating.py:1420-1436). ` +
				`The reference's own fall-through would silently return the Fieller interval instead.`
		);
	}

	const tRef = options.tRef ?? mean64(t); // 1325
	const x = new Float64Array(n);
	for (let i = 0; i < n; i++) x[i] = t[i] - tRef;

	const { values: wPos, vectors: V } = options.eigen ?? eighClamped(covMatrix, n); // 1346-1347

	// C = λK + (1 − λ)I, or C = K + ridge·I when no λ is given (1351-1357)
	let effLam;
	const wc = new Float64Array(n);
	if (pagelLambda !== null) {
		effLam = Math.min(Math.max(pagelLambda, PAGEL_LAMBDA_BOUNDS[0]), PAGEL_LAMBDA_BOUNDS[1]);
		for (let i = 0; i < n; i++) wc[i] = effLam * wPos[i] + (1.0 - effLam);
	} else {
		effLam = 1.0 - ridge;
		for (let i = 0; i < n; i++) wc[i] = wPos[i] + ridge;
	}
	const invW = new Float64Array(n);
	for (let i = 0; i < n; i++) invW[i] = 1.0 / Math.max(wc[i], 1e-12); // 1359

	const { Z, u, vOne } = projectOntoEigenbasis(V, x, d, n);
	const { A, b } = normalEquations(Z, u, invW, n);
	const beta = solve2x2(A, b); // 1371
	const mu = beta[0];
	const d0 = beta[1];

	const residuals = new Float64Array(n);
	const res2 = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		residuals[i] = d[i] - (x[i] * mu + d0); // 1373: d − Xβ, in the ORIGINAL basis
		res2[i] = residuals[i] * residuals[i];
	}
	const fitted = new Float64Array(n);
	for (let i = 0; i < n; i++) fitted[i] = x[i] * mu + d0;

	const resProj = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		const r = u[i] - (Z[i * 2] * mu + Z[i * 2 + 1] * d0); // 1374
		resProj[i] = r * r * invW[i];
	}
	const ssRes = sum64(resProj); // 1375-1376
	const sigma2 = ssRes / Math.max(1, n - 2); // 1377
	const inv = inv2x2(A);
	const covBeta = [sigma2 * inv[0], sigma2 * inv[1], sigma2 * inv[2], sigma2 * inv[3]]; // 1378

	const seMu = Math.sqrt(Math.max(1e-15, covBeta[0])); // 1380
	const seD0 = Math.sqrt(Math.max(1e-15, covBeta[3])); // 1381

	let minTime = Infinity;
	for (let i = 0; i < n; i++) minTime = Math.min(minTime, t[i]); // 1383

	let ciFieller = [NaN, NaN];
	let fiellerG = NaN;
	let ciDelta = [NaN, NaN];
	let ciMrca = [NaN, NaN];
	let tMrca;
	let seMrca;
	let status;

	if (mu <= 1e-12) {
		tMrca = NaN;
		seMrca = NaN;
		status = DATING_STATUS.NON_POSITIVE_RATE; // 1392
	} else {
		tMrca = tRef - d0 / mu; // 1394
		if (tMrca >= minTime) {
			tMrca = NaN;
			seMrca = NaN;
			status = DATING_STATUS.MRCA_AFTER_EARLIEST_SAMPLE; // 1398
		} else {
			status = DATING_STATUS.OK;
			const df = Math.max(1, n - 2);
			const delta = computeDeltaMrcaInterval(mu, d0, covBeta, tRef, df, { minTime }); // 1402-1409
			seMrca = delta.seMrca;
			ciDelta = delta.ci;
			const fie = computeFiellerMrcaInterval(mu, d0, covBeta, tRef, df, { minTime }); // 1412-1414
			ciFieller = fie.ci;
			fiellerG = fie.g;
			ciMrca = ciLower === 'delta' || ciLower === 'linear' ? ciDelta : ciFieller; // 1417-1438
		}
	}

	// Buse (1973) generalised R² (1441-1447)
	const oneT = new Float64Array(n);
	const oneD = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		oneT[i] = vOne[i] * vOne[i] * invW[i];
		oneD[i] = vOne[i] * u[i] * invW[i];
	}
	const oneCinvOne = sum64(oneT);
	const oneCinvD = sum64(oneD);
	const weightedMean = oneCinvD / Math.max(1e-12, oneCinvOne);
	const totRes = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		const r = u[i] - weightedMean * vOne[i];
		totRes[i] = r * r * invW[i];
	}
	const ssTot = sum64(totRes);
	const r2 = Math.max(0.0, 1.0 - ssRes / Math.max(1e-12, ssTot));

	return {
		method: 'PGLS',
		status,
		mu,
		d0,
		t_ref: tRef,
		t_mrca: tMrca,
		se_mu: seMu,
		se_d0: seD0,
		se_mrca: seMrca,
		ci_fieller: ciFieller,
		fieller_g: fiellerG,
		ci_delta: ciDelta,
		ci_mrca: ciMrca,
		ci_method: ciMethod,
		r2,
		// B16: the `ridge` ARGUMENT is discarded whenever a lambda was given, and what ships under the
		// name is 1 − eff_lam. Both spellings are the reference's own line 1466.
		ridge: pagelLambda === null ? ridge : 1.0 - effLam,
		pagel_lambda: pagelLambda,
		sigma2,
		rmse: Math.sqrt(mean64(res2)), // 1470: mean over n, not over n − 2
		residuals,
		fitted,
		times: t,
		n,
		cov_beta: covBeta, // not a reference key; the intervals need it and the app must not refit
	};
}

// =================================================================================================
// the latent convex-hull root
// =================================================================================================

/** torch.optim.Adam's defaults (778): the reference passes only `lr`. */
export const ADAM_DEFAULTS = Object.freeze({ beta1: 0.9, beta2: 0.999, eps: 1e-8 });
/** `learning_rate` and `max_iter` as `optimize_latent_convex_hull_root` declares them (707-708). */
export const LATENT_ROOT_DEFAULTS = Object.freeze({ learningRate: 0.05, maxIter: 250 });
/** The steps the fixture records the softmax weights at; see the header on why the endpoint is not enough. */
export const LATENT_TRAJECTORY_STEPS = Object.freeze([1, 2, 5, 10, 25, 50, 100, 200, 249, 250]);

/** `np.std(a, ddof=1)`, which is torch's default `correction=1` — NOT src/dating.js's `std64`. */
function stdUnbiased(a, m = a.length) {
	const mu = mean64(a, m);
	const sq = new Float64Array(m);
	for (let i = 0; i < m; i++) sq[i] = (a[i] - mu) * (a[i] - mu);
	return Math.sqrt(sum64(sq) / (m - 1));
}

/** `torch.softmax(v, dim=0)`, max-shifted as torch's kernel is. */
function softmax(v, out) {
	const n = v.length;
	let max = -Infinity;
	for (let i = 0; i < n; i++) if (v[i] > max) max = v[i];
	let total = 0;
	for (let i = 0; i < n; i++) {
		out[i] = Math.exp(v[i] - max);
		total += out[i];
	}
	for (let i = 0; i < n; i++) out[i] /= total;
	return out;
}

/**
 * `optimize_latent_convex_hull_root(taxon_repr, times, taxa_names, pairwise_phys_dists, anchor_mask,
 * learning_rate=0.05, max_iter=250)`, dating.py:701-832.
 *
 * Place a root INSIDE the convex hull of the taxon embeddings — `z_root = Σ softmax(v)_i · z_i` —
 * by 250 Adam steps that maximise the correlation between `‖z_i − z_root‖` and sampling date, then
 * calibrate latent units to substitutions per site with a no-intercept least-squares slope `alpha`
 * against the pairwise physical distances. See the header for why this is a trajectory replay and
 * where the gradient comes from.
 *
 * `alpha`'s fallback when no physical distances are given is `0.05 / mean(d_latent)` (756-757) — a
 * hard-coded 0.05 substitutions per site, which makes the resulting rate a scale guess and not a
 * measurement. The application should say so wherever that arm is taken.
 *
 * `t_mrca_ols` (804-806) fits `np.polyfit(times, dists, 1)` on the UNCENTRED calendar axis and is a
 * DIAGNOSTIC: the reference runs `run_ols_dating` separately on the same distances, and that is the
 * number a report shows. Ported because it is in the record the reference writes.
 *
 * @param {ArrayLike<number>} taxonRepr row-major n*embedDim
 * @param {ArrayLike<number>} times
 * @param {{embedDim: number, taxaNames?: string[]|null, pairwisePhysDists?: ArrayLike<number>|null,
 *   anchorMask?: ArrayLike<boolean|number>|null, learningRate?: number, maxIter?: number,
 *   trajectorySteps?: number[]|null}} options `embedDim` is required: a flat array cannot say its own shape
 * @returns {Record<string, any>} the reference's key set, in the reference's order
 */
export function optimizeLatentConvexHullRoot(taxonRepr, times, options) {
	const D = options.embedDim;
	const t = f64(times);
	const n = t.length;
	if (!Number.isInteger(D) || D <= 0) throw new RangeError(`embedDim must be a positive integer (got ${D}).`);
	if (taxonRepr.length !== n * D) {
		throw new RangeError(`taxon_repr must be ${n}x${D} (got ${taxonRepr.length} entries).`);
	}
	const z = f64(taxonRepr);
	const lr = options.learningRate ?? LATENT_ROOT_DEFAULTS.learningRate;
	const maxIter = options.maxIter ?? LATENT_ROOT_DEFAULTS.maxIter;
	const names = options.taxaNames ?? null;
	const phys = options.pairwisePhysDists ?? null;
	const trajectorySteps = options.trajectorySteps ?? null;

	// eligible anchors (738-745). B12: fewer than three silently resets the mask to all-true.
	let eligible = new Uint8Array(n).fill(1);
	if (options.anchorMask) {
		for (let i = 0; i < n; i++) eligible[i] = options.anchorMask[i] ? 1 : 0;
	}
	let nEligible = 0;
	for (let i = 0; i < n; i++) nEligible += eligible[i];
	if (nEligible < 3) {
		eligible = new Uint8Array(n).fill(1);
		nEligible = n;
	}
	const elig = [];
	for (let i = 0; i < n; i++) if (eligible[i]) elig.push(i);

	// 1. isometric calibration over ALL pairs (747-757). B14: eligibility is not consulted here.
	const nPairs = (n * (n - 1)) / 2;
	const dLatPairs = new Float64Array(nPairs);
	{
		const term = new Float64Array(D);
		let p = 0;
		for (let i = 0; i < n; i++) {
			for (let j = i + 1; j < n; j++) {
				for (let k = 0; k < D; k++) {
					const dv = z[i * D + k] - z[j * D + k];
					term[k] = dv * dv;
				}
				dLatPairs[p++] = Math.sqrt(sum64(term));
			}
		}
	}
	let alpha;
	if (phys && nPairs > 0) {
		const num = new Float64Array(nPairs);
		const den = new Float64Array(nPairs);
		let p = 0;
		for (let i = 0; i < n; i++) {
			for (let j = i + 1; j < n; j++, p++) {
				num[p] = phys[i * n + j] * dLatPairs[p];
				den[p] = dLatPairs[p] * dLatPairs[p];
			}
		}
		const denom = sum64(den);
		alpha = denom > 1e-12 ? sum64(num) / denom : 1.0; // 752-753
	} else {
		const meanLat = nPairs > 0 ? mean64(dLatPairs) : 1.0;
		alpha = 0.05 / Math.max(1e-6, meanLat); // 756-757
	}

	// 2. the 250-step convex-hull search (759-795)
	const tEl = new Float64Array(nEligible);
	for (let k = 0; k < nEligible; k++) tEl[k] = t[elig[k]];
	let minT = Infinity;
	let maxT = -Infinity;
	for (let k = 0; k < nEligible; k++) {
		minT = Math.min(minT, tEl[k]);
		maxT = Math.max(maxT, tEl[k]);
	}
	const span = Math.max(1e-6, maxT - minT); // 765
	const tMean = mean64(tEl);
	const tCentred = new Float64Array(nEligible);
	for (let k = 0; k < nEligible; k++) tCentred[k] = tEl[k] - tMean; // 775
	const stdT = stdUnbiased(tEl) + 1e-8; // 776

	const v = new Float64Array(n);
	for (let i = 0; i < n; i++) v[i] = eligible[i] ? -0.5 * (t[i] - minT) / span : -1e4; // 768-771 (B13)

	const w = new Float64Array(n);
	const zRoot = new Float64Array(D);
	const diff = new Float64Array(n * D);
	const dLat = new Float64Array(n);
	const dEl = new Float64Array(nEligible);
	const dCentred = new Float64Array(nEligible);
	const gradD = new Float64Array(n);
	const G = new Float64Array(D);
	const grad = new Float64Array(n);
	const mAdam = new Float64Array(n);
	const vAdam = new Float64Array(n);
	const scratchN = new Float64Array(n);
	const scratchM = new Float64Array(nEligible);
	const scratchD = new Float64Array(D);
	const trajectory = trajectorySteps ? new Map() : null;

	for (let step = 1; step <= maxIter; step++) {
		softmax(v, w); // 781

		for (let k = 0; k < D; k++) {
			for (let i = 0; i < n; i++) scratchN[i] = w[i] * z[i * D + k];
			zRoot[k] = sum64(scratchN); // 782
		}
		for (let i = 0; i < n; i++) {
			for (let k = 0; k < D; k++) {
				const dv = z[i * D + k] - zRoot[k];
				diff[i * D + k] = dv;
				scratchD[k] = dv * dv;
			}
			dLat[i] = Math.sqrt(sum64(scratchD)); // 783
		}
		for (let k = 0; k < nEligible; k++) dEl[k] = dLat[elig[k]];
		const dMean = mean64(dEl);
		for (let k = 0; k < nEligible; k++) dCentred[k] = dEl[k] - dMean; // 785
		const sD = stdUnbiased(dEl);
		const den = stdT * sD + 1e-8; // 787
		for (let k = 0; k < nEligible; k++) scratchM[k] = tCentred[k] * dCentred[k];
		const cov = mean64(scratchM); // 786 (B11: mean over m, while both stds use m − 1)

		// d(loss)/d(d_k) for the eligible rows; the rest are exactly zero
		gradD.fill(0);
		for (let k = 0; k < nEligible; k++) {
			const dCov = tCentred[k] / nEligible;
			const dSd = dCentred[k] / ((nEligible - 1) * sD);
			gradD[elig[k]] = -((dCov * den - cov * stdT * dSd) / (den * den)); // loss = −corr
		}
		// G = d(loss)/d(z_root) = −Σ_k gradD_k · (z_k − z_root)/d_k
		for (let k = 0; k < D; k++) {
			for (let i = 0; i < n; i++) scratchN[i] = gradD[i] === 0 ? 0 : (gradD[i] * diff[i * D + k]) / dLat[i];
			G[k] = -sum64(scratchN);
		}
		// d(loss)/d(v_j) = w_j · (z_j − z_root)·G
		for (let i = 0; i < n; i++) {
			for (let k = 0; k < D; k++) scratchD[k] = diff[i * D + k] * G[k];
			grad[i] = w[i] * sum64(scratchD);
		}

		// torch.optim.Adam: eps is added AFTER the second moment's bias correction
		const bc1 = 1 - Math.pow(ADAM_DEFAULTS.beta1, step);
		const bc2 = 1 - Math.pow(ADAM_DEFAULTS.beta2, step);
		const stepSize = lr / bc1;
		const sqrtBc2 = Math.sqrt(bc2);
		for (let i = 0; i < n; i++) {
			mAdam[i] = ADAM_DEFAULTS.beta1 * mAdam[i] + (1 - ADAM_DEFAULTS.beta1) * grad[i];
			vAdam[i] = ADAM_DEFAULTS.beta2 * vAdam[i] + (1 - ADAM_DEFAULTS.beta2) * grad[i] * grad[i];
			v[i] -= (stepSize * mAdam[i]) / (Math.sqrt(vAdam[i]) / sqrtBc2 + ADAM_DEFAULTS.eps);
		}
		if (trajectory && trajectorySteps.includes(step)) {
			trajectory.set(step, Array.from(softmax(v, new Float64Array(n))));
		}
	}

	// 3. the record (797-822)
	const wOpt = softmax(v, new Float64Array(n));
	for (let k = 0; k < D; k++) {
		for (let i = 0; i < n; i++) scratchN[i] = wOpt[i] * z[i * D + k];
		zRoot[k] = sum64(scratchN); // 798
	}
	const distsLatent = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		for (let k = 0; k < D; k++) {
			const dv = z[i * D + k] - zRoot[k];
			scratchD[k] = dv * dv;
		}
		distsLatent[i] = Math.sqrt(sum64(scratchD)); // 799
	}
	const distsPhys = new Float64Array(n);
	for (let i = 0; i < n; i++) distsPhys[i] = alpha * distsLatent[i]; // 800

	// np.corrcoef(times_el, dists_el)[0, 1] (803): Pearson's r, ddof-free, so a true r unlike the loss
	const dPhysEl = new Float64Array(nEligible);
	for (let k = 0; k < nEligible; k++) dPhysEl[k] = distsPhys[elig[k]];
	const dPhysMean = mean64(dPhysEl);
	const co = new Float64Array(nEligible);
	const vt = new Float64Array(nEligible);
	const vd = new Float64Array(nEligible);
	for (let k = 0; k < nEligible; k++) {
		const a = tEl[k] - tMean;
		const bq = dPhysEl[k] - dPhysMean;
		co[k] = a * bq;
		vt[k] = a * a;
		vd[k] = bq * bq;
	}
	const rVal = sum64(co) / Math.sqrt(sum64(vt) * sum64(vd));

	// np.polyfit(times_el, dists_el, 1) on the UNCENTRED calendar axis (804-806).
	// Written the way numpy writes it — the Vandermonde [t, 1] with its columns scaled to unit norm,
	// then the 2x2 normal equations, then the scaling undone — and NOT as the centred `Sxy/Sxx`. The
	// centred form is algebraically identical and numerically worse here: recovering the intercept as
	// `d̄ − slope·t̄` cancels 1.105 against 0.036 at t̄ ≈ 1991. MEASURED against numpy's own lstsq on the
	// acceptance case, both forms fed the REFERENCE's latent distances so only the fit is being
	// compared: the centred form costs 7.0e-6 years of `t_mrca_ols`, this one 5.9e-10 in numpy and
	// 1.0e-8 here. (numpy uses gelsd's SVD where this uses the adjugate; the column scaling is what
	// the two share, and it is where the precision comes from.) Through the port's OWN distances the
	// figure is 6.6e-5 years, and that is the Adam trajectory's float64 residual, not this fit's.
	const nEl = nEligible;
	const colNorm = [Math.sqrt(sum64(Float64Array.from(tEl, (v) => v * v))), Math.sqrt(nEl)];
	const s00 = new Float64Array(nEl);
	const s01 = new Float64Array(nEl);
	const sb0 = new Float64Array(nEl);
	const sb1 = new Float64Array(nEl);
	for (let k = 0; k < nEl; k++) {
		const c0 = tEl[k] / colNorm[0];
		const c1 = 1 / colNorm[1];
		s00[k] = c0 * c0;
		s01[k] = c0 * c1;
		sb0[k] = c0 * dPhysEl[k];
		sb1[k] = c1 * dPhysEl[k];
	}
	const pA = [sum64(s00), sum64(s01), sum64(s01), nEl / (colNorm[1] * colNorm[1])];
	const pB = [sum64(sb0), sum64(sb1)];
	const pDet = pA[0] * pA[3] - pA[1] * pA[2];
	const slopeOls = (pA[3] * pB[0] - pA[1] * pB[1]) / pDet / colNorm[0];
	const interOls = (pA[0] * pB[1] - pA[2] * pB[0]) / pDet / colNorm[1];
	const tMrcaOls = slopeOls > 1e-6 ? -interOls / slopeOls : NaN; // 806

	// the anchor table (809-822). B15: the break needs BOTH w < 0.01 AND three rows already emitted.
	const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => (wOpt[b] - wOpt[a]) || a - b);
	const anchorTaxa = [];
	for (const idx of order) {
		if (wOpt[idx] < 0.01 && anchorTaxa.length >= 3) break;
		anchorTaxa.push({
			taxon: names && idx < names.length ? names[idx] : `taxon_${idx}`,
			weight: wOpt[idx],
			date: t[idx],
		});
	}

	return {
		z_root: zRoot,
		weights: wOpt,
		dists: distsPhys,
		dists_latent: distsLatent,
		alpha,
		anchor_taxa: anchorTaxa,
		anchor_mask: eligible,
		temporal_r: rVal,
		temporal_r2: rVal * rVal,
		mu_ols: slopeOls,
		t_mrca_ols: tMrcaOls,
		...(trajectory ? { weights_trajectory: trajectory } : {}),
	};
}

// =================================================================================================
// the isometric calibration's physical distances
// =================================================================================================

/**
 * The pairwise nucleotide distance `run_mrca_dating` builds inline at dating.py:2586-2596 to
 * calibrate `alpha`. It is here, and not in preprocess/tn93.js, because it is NOT TN93 despite
 * `optimize_latent_convex_hull_root`'s docstring saying "Hamming / TN93" (718): it is a raw
 * p-distance over the columns where BOTH sequences carry an unambiguous A, C, G or T, with no
 * multiple-hit correction at all. Feeding a TN93 matrix here would change `alpha`, and therefore
 * every rate and every date the latent mode produces.
 *
 * Two details that look like bugs and are not quite: the denominator is `max(1, tot)`, so a pair
 * with no shared resolved column scores 0 rather than raising; and the comparison `char_mat[i] !=
 * char_mat[j]` is made over the WHOLE row and then masked, so case matters — the reference's own
 * `parse_alignment_sequences` upper-cases, which is why it never bites upstream.
 *
 * The diagonal is left at 0 (the loop runs j > i only) and the matrix is symmetric by construction.
 *
 * @param {string[]} sequences aligned, equal length, upper case
 * @returns {Float64Array} row-major n*n
 */
export function pairwiseAcgtHammingMatrix(sequences) {
	const n = sequences.length;
	const out = new Float64Array(n * n);
	if (n === 0) return out;
	const width = sequences[0].length;
	// resolved[i] is a 0/1 mask of the columns where sequence i is an unambiguous ACGT
	const resolved = [];
	for (let i = 0; i < n; i++) {
		if (sequences[i].length !== width) {
			throw new RangeError(`sequence ${i} is ${sequences[i].length} columns, expected ${width}.`);
		}
		const mask = new Uint8Array(width);
		for (let c = 0; c < width; c++) {
			const ch = sequences[i][c];
			mask[c] = ch === 'A' || ch === 'C' || ch === 'G' || ch === 'T' ? 1 : 0;
		}
		resolved.push(mask);
	}
	for (let i = 0; i < n; i++) {
		for (let j = i + 1; j < n; j++) {
			let diffs = 0;
			let tot = 0;
			const mi = resolved[i];
			const mj = resolved[j];
			const si = sequences[i];
			const sj = sequences[j];
			for (let c = 0; c < width; c++) {
				if (mi[c] && mj[c]) {
					tot++;
					if (si[c] !== sj[c]) diffs++;
				}
			}
			const v = diffs / Math.max(1, tot);
			out[i * n + j] = v;
			out[j * n + i] = v;
		}
	}
	return out;
}
