/**
 * WHY THIS FILE EXISTS
 *
 * `hyphaeon/temporal.py` calls `np.linalg.svd(..., full_matrices=False)` twice — the fPCA shape gate
 * (temporal.py:673) and the collective wave decomposition (temporal.py:731) — and NEITHER call reads
 * the left singular vectors. `u` at 673 survives only inside the product `u Σ vᵀ`, whose per-row
 * residual is all that reaches `r2_fpca`; `u_f` at 731 is never referenced again. Everything the two
 * call sites actually consume — the singular values, the right singular vectors `vt_f[:4]`, and a
 * rank-k projector — lives on the TIME side, whose dimension `T` is the only bounded one
 * (`--time-points`, default 250, measured runs at 60 and 250) while the row count is the codon
 * count (246 … 4384 on the acceptance alignment).
 *
 * So this file builds the one decomposition those call sites need out of the symmetric eigensolver
 * the library already has (`preprocess/symmetricEigen.js`, tred2 + tql2), with one rule:
 *
 *     ALWAYS form the T × T time-side Gram G = Zᵀ Z and eigendecompose that.
 *     Never form the row-side Gram, and never form U.
 *
 * WHY THE GRAM IS NOT THE ACCURACY DISASTER IT IS USUALLY SAID TO BE. Squaring the condition number
 * hurts the TRAILING factors; only the leading four are read here. Measured on the acceptance run's
 * wave stage (σ = 20.699, 18.696, 12.263, 10.025, recovered from the reference's own CSVs through
 * Σ_{s ∈ f_indices} loading_j[s]² = σ_j²): (σ₁/σ₄)² = 4.26, so the relative error the squaring adds
 * to σ₄ is ~5e-16. It would matter only for a mode with σ_j/σ₁ < √ε ≈ 1.5e-8 — one explaining under
 * 1e-16 of the variance, which the caller must discard anyway. For the leading EIGENVECTORS the Gram
 * route is in fact BETTER conditioned than the SVD it replaces: the eigenvalue gap is
 * (σ_j − σ_{j+1})(σ_j + σ_{j+1}), relatively wider than the singular-value gap — measured on waves
 * 1/2, λ₁/(λ₁ − λ₂) = 5.4 against σ₁/(σ₁ − σ₂) = 10.3.
 *
 * The accuracy loss that WOULD have been real is the row-side route (eigendecompose Z Zᵀ, then
 * v_j = Zᵀ u_j / σ_j): that amplifies error by σ₁/σ_j through the division and loses orthogonality
 * among the recovered v_j. It is not implemented, deliberately, and must not be added "for when
 * rows < T" — in that case the T × T Gram is merely singular in its trailing T − rows directions,
 * which the rank cap discards.
 *
 * The largest error in the routine is forming G itself, which is why each entry is accumulated with
 * `numpyPairwiseSum` (~ε·√rows·‖Z‖² instead of ~ε·rows·‖Z‖²: at rows = 4384 the difference is
 * ~1.5e-14 against ~1e-12 relative).
 *
 * THE SIGN CONVENTION (D28, WAVE_SIGN.md) lives here because this is where the vectors are produced.
 * `temporal.py` has no convention and writes LAPACK `gesdd`'s raw signs; on the acceptance run three
 * of its four waves have their largest-magnitude entry negative, so wave 1 is drawn as a trough and
 * the figure's red/blue loading bars are decided by a coin flip inside the solver. `waveSign:
 * 'canonical'` (the default) is the same rule `MDS_SIGN.md` settled on for the MDS eigenvectors —
 * largest-magnitude entry positive, first index on a tie — applied to each retained v_j BEFORE
 * anything is derived from it, so the loadings inherit the flip and the wave-to-loading relative
 * sign stays correct by construction. `waveSign: 'lapack'` keeps the solver's signs (here that means
 * tql2's, which are not LAPACK's) and reproduces pre-convention output.
 *
 * WHAT NEEDS NO CONVENTION, and where no cycles should be spent canonicalising: the gate's `r2`.
 * Only the projector P = V_k V_kᵀ enters it, and a projector does not care which orthonormal basis
 * spans its range or which way each basis vector points. That is why {@link projectionResidual}
 * takes the vectors and never a sign.
 *
 * Mirrors: `hyphaeon/temporal.py:667-679` (gate) and `:718-741` (waves); `numpy.linalg.svd` is the
 * routine replaced. No I/O, no method semantics — the classification, the thresholds and the
 * degeneracy copy are `temporal.js`'s and the application's.
 */

import { symmetricEigen } from '../preprocess/symmetricEigen.js';
import { numpyPairwiseSum } from './reduce.js';

/** The two sign conventions, mirroring `MDS_SIGN_MODES` in `hyphaeon/dataset.py:576`. */
export const WAVE_SIGN_MODES = /** @type {const} */ (['canonical', 'lapack']);

/**
 * Relative singular-value floor below which a mode is arbitrary rather than small: √ε for float64.
 * A mode under it explains less than ε of the variance, so its vector is a null-space direction the
 * solver picked at random. numpy writes such vectors out; the caller is told instead.
 */
export const WAVE_RANK_EPS = Math.sqrt(Number.EPSILON);

/**
 * Relative adjacent-gap threshold below which two modes cannot be separated by any sign rule,
 * because the pair spans a 2-plane the solver may return rotated.
 *
 * THE CONSTANT IS SET BY MEASUREMENT, not taste: on the acceptance run the four retained modes have
 * gaps g = (σ_j − σ_{j+1})/σ₁ of 0.097, 0.311 and 0.108, so the reference's own headline example
 * sits at twice the threshold and the flag is not decorative. Below 0.05 the individual axes rotate
 * under perturbations the data cannot distinguish (one dropped taxon, a different `--time-points`).
 */
export const WAVE_GAP_THRESHOLD = 0.05;

/**
 * Resolve a wave-sign convention, mirroring `resolve_mds_sign` (dataset.py:580).
 *
 * @param {string|null|undefined} waveSign
 * @returns {'canonical'|'lapack'}
 */
export function resolveWaveSign(waveSign) {
	const value = String(waveSign ?? 'canonical').trim().toLowerCase();
	if (value !== 'canonical' && value !== 'lapack') {
		throw new RangeError(`waveSign must be one of ${JSON.stringify(WAVE_SIGN_MODES)}, got ${JSON.stringify(waveSign)}`);
	}
	return value;
}

/**
 * The time-side Gram `G = Zᵀ Z`, `[T, T]` row-major, exactly symmetric by construction (the upper
 * triangle is mirrored from the lower rather than accumulated a second time, so the matrix meets
 * `symmetricEigen`'s "upper triangle unused" contract with no roundoff asymmetry at all).
 *
 * Each entry is a reduction of length `rows`, which reaches the thousands, so it is accumulated
 * pairwise — see the header for what naive summation would cost.
 *
 * @param {ArrayLike<number>} Z row-major rows*T
 * @param {number} rows
 * @param {number} T
 * @returns {Float64Array} row-major T*T
 */
export function timeGram(Z, rows, T) {
	const G = new Float64Array(T * T);
	const scratch = new Float64Array(rows);
	for (let a = 0; a < T; a++) {
		for (let b = 0; b <= a; b++) {
			for (let i = 0; i < rows; i++) scratch[i] = Z[i * T + a] * Z[i * T + b];
			const g = numpyPairwiseSum(scratch, 0, rows, identity);
			G[a * T + b] = g;
			G[b * T + a] = g;
		}
	}
	return G;
}

/**
 * The dominant right singular vectors of `Z` `[rows, T]` and the whole singular spectrum, through
 * the time-side Gram.
 *
 * Rank is capped at `min(rows, T)` — what `np.linalg.svd(..., full_matrices=False)` returns — so a
 * caller reproducing `waves = vt_f[:4]` gets fewer than four rows when `rows < 4`, and pads with
 * zeros itself the way `temporal.py:816-819` does.
 *
 * `varExplained` is normalised by `‖Z‖_F² = trace(G)` accumulated from the eigenvalues' own source
 * rather than from the clamped singular values, matching `np.sum(s_f ** 2)` to roundoff while being
 * immune to the clamp. When the total is not positive it is all zeros, as `temporal.py:733` does.
 *
 * @param {ArrayLike<number>} Z row-major rows*T
 * @param {number} rows
 * @param {number} T
 * @param {number} k how many leading modes to return vectors for
 * @param {{ waveSign?: 'canonical'|'lapack' }} [options]
 * @returns {{
 *   nModes: number, kept: number, sigma: Float64Array, varExplained: Float64Array,
 *   vectors: Float64Array, totalVariance: number, gaps: Float64Array,
 *   rankDeficient: boolean[], nearDegenerate: boolean[], waveSign: 'canonical'|'lapack'
 * }} `vectors` is row-major `kept*T`, row j being v_j; `gaps[j] = (σ_j − σ_{j+1})/σ_0` for
 *   j = 0 … kept−1, INCLUDING the gap between the last retained mode and the first discarded one
 *   (that one is what moves the projector, hence `r2_fpca`, hence the confirmed-sweep set);
 *   `nearDegenerate[j]` is `gaps[j] < WAVE_GAP_THRESHOLD`.
 */
export function dominantTimeModes(Z, rows, T, k, { waveSign = 'canonical' } = {}) {
	const sign = resolveWaveSign(waveSign);
	const nModes = Math.min(rows, T);
	const kept = Math.max(0, Math.min(k, nModes));
	const G = timeGram(Z, rows, T);
	let total = 0;
	{
		// trace(G) = ‖Z‖_F², accumulated pairwise over the diagonal.
		const diag = new Float64Array(T);
		for (let t = 0; t < T; t++) diag[t] = G[t * T + t];
		total = numpyPairwiseSum(diag, 0, T, identity);
	}
	const { values, vectors } = symmetricEigen(G, T);

	// symmetricEigen returns ASCENDING eigenvalues with column j holding the vector for values[j],
	// so the leading mode is the LAST column. Both facts are easy to get wrong; reversing here keeps
	// every caller in descending order.
	const sigma = new Float64Array(nModes);
	for (let j = 0; j < nModes; j++) {
		const lam = values[T - 1 - j];
		sigma[j] = Math.sqrt(lam > 0 ? lam : 0);
	}
	const out = new Float64Array(kept * T);
	for (let j = 0; j < kept; j++) {
		const col = T - 1 - j;
		for (let t = 0; t < T; t++) out[j * T + t] = vectors[t * T + col];
	}
	if (sign === 'canonical') canonicalizeWaveSigns(out, kept, T);

	const varExplained = new Float64Array(nModes);
	if (total > 0) for (let j = 0; j < nModes; j++) varExplained[j] = (sigma[j] * sigma[j]) / total;

	const s0 = sigma.length > 0 ? sigma[0] : 0;
	const gaps = new Float64Array(kept);
	const nearDegenerate = new Array(kept).fill(false);
	const rankDeficient = new Array(kept).fill(false);
	for (let j = 0; j < kept; j++) {
		const next = j + 1 < nModes ? sigma[j + 1] : 0;
		gaps[j] = s0 > 0 ? (sigma[j] - next) / s0 : 0;
		nearDegenerate[j] = gaps[j] < WAVE_GAP_THRESHOLD;
		rankDeficient[j] = !(s0 > 0) || sigma[j] / s0 < WAVE_RANK_EPS;
	}

	return { nModes, kept, sigma, varExplained, vectors: out, totalVariance: total, gaps, rankDeficient, nearDegenerate, waveSign: sign };
}

/**
 * The canonical sign rule (D28), in place: each row of `V` `[k, T]` is multiplied by −1 when its
 * largest-magnitude entry is negative. Ties resolve to the FIRST index (`np.argmax(np.abs(v))`
 * semantics, strict `>` on the scan). An all-zero row is left alone.
 *
 * A sign flip is exact in floating point, so canonical vectors are bitwise the solver's vectors up
 * to sign and nothing else about the decomposition changes. MEASURED consequence of adopting it on
 * the acceptance run: three of the four waves flip (1, 3 and 4), the corresponding loading columns
 * flip with them, and every other number in every output file is bit-identical — `var_explained`,
 * `r2_fpca`, the permutation p-values and both classification columns read no sign.
 *
 * @param {Float64Array} V row-major k*T, modified in place
 * @param {number} k
 * @param {number} T
 * @returns {Float64Array} V
 */
export function canonicalizeWaveSigns(V, k, T) {
	for (let j = 0; j < k; j++) {
		const o = j * T;
		let pivot = 0;
		let best = -1;
		for (let t = 0; t < T; t++) {
			const m = Math.abs(V[o + t]);
			if (m > best) {
				best = m;
				pivot = t;
			}
		}
		if (V[o + pivot] < 0) {
			for (let t = 0; t < T; t++) V[o + t] = -V[o + t];
		}
	}
	return V;
}

/**
 * Per-row residual energy of `Z` after projecting onto the span of the first `k` rows of `V`:
 *
 *     sse_i = ‖z_i‖² − Σ_{j<k} (z_i · v_j)²
 *
 * This is `np.sum((Z_cand − Z_recon) ** 2, axis=1)` at temporal.py:675 without ever forming
 * `Z_recon` (or `U`, or a `[rows, T]` intermediate): since `Z_recon = Z V_k V_kᵀ` and `V_k` is
 * orthonormal, `Z − Z_recon = Z(I − P)` with `P` an orthogonal projector, and Pythagoras does the
 * rest. Cost is `rows·T·k` with no allocation per row.
 *
 * The result is a difference of two positive quantities and can go slightly negative when a row
 * lies almost entirely in the subspace, so it is clamped at 0 — the same guard `np.sum(...)` would
 * not need but the shortcut does.
 *
 * Sign- AND rotation-invariant: any orthonormal basis of the same k-dimensional subspace gives the
 * same numbers.
 *
 * @param {ArrayLike<number>} Z row-major rows*T
 * @param {number} rows
 * @param {number} T
 * @param {ArrayLike<number>} V row-major >= k*T
 * @param {number} k
 * @returns {{ sse: Float64Array, sst: Float64Array }} `sst` is the raw `‖z_i‖²`, WITHOUT the
 *   reference's `+1e-8` — that guard is a method decision and belongs to the caller.
 */
export function projectionResidual(Z, rows, T, V, k) {
	const sse = new Float64Array(rows);
	const sst = new Float64Array(rows);
	const sq = new Float64Array(T);
	for (let i = 0; i < rows; i++) {
		const o = i * T;
		for (let t = 0; t < T; t++) sq[t] = Z[o + t] * Z[o + t];
		const norm2 = numpyPairwiseSum(sq, 0, T, identity);
		sst[i] = norm2;
		let inSubspace = 0;
		for (let j = 0; j < k; j++) {
			const vo = j * T;
			let dot = 0;
			for (let t = 0; t < T; t++) dot += Z[o + t] * V[vo + t];
			inSubspace += dot * dot;
		}
		const r = norm2 - inSubspace;
		sse[i] = r > 0 ? r : 0;
	}
	return { sse, sst };
}

/** Float64 accumulation: no rounding between additions. */
function identity(/** @type {number} */ v) {
	return v;
}
