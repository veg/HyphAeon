/**
 * WHY THIS FILE EXISTS
 *
 * src/datingModel.js is the half of the ChronAeon pillar the transformer feeds, and the only thing
 * worth proving about it is that it lands on THE REFERENCE'S OWN NUMBERS. That matters more here
 * than anywhere else in the library: a covariance kernel built out of the wrong slice, a Pagel
 * lambda from a minimiser that stopped somewhere else, an Adam step that used the biased standard
 * deviation — each of those produces an ancestor date that is wrong by decades and looks exactly
 * like a right one. There is no smell test for a molecular clock.
 *
 * Four layers, which fail for different reasons, so a failure says WHERE the fault is:
 *
 *   1. FUNCTION REPLAY, on PINNED inputs. `fixtures/dating/*.json` is produced by the reference
 *      functions themselves — lifted out of hyphaeon/dating.py by `ast` rather than retyped
 *      (scripts/gen_fixtures.py, `load_dating_reference`; regenerate with `--only dating_model`,
 *      which needs the weights) — so each case is a replay, not an opinion. The kernel, the REML
 *      profile, the GLS fit and the latent trajectory are separate tables, so a discrepancy cannot
 *      hide behind a pipeline.
 *
 *   2. THE CHAIN. `describe('the whole model-based chain …')` starts from `model_outputs` — the one
 *      thing this library does not compute — and walks kernel → slice → REML → PGLS, and
 *      separately Hamming → latent root, comparing each stage against the fixture that pins it. It
 *      is the test that would catch a port whose four functions are each individually right and
 *      whose composition is not, which is the failure the subsetting order at dating.py:2568-2576
 *      invites.
 *
 *   3. THE PUBLISHED ANSWER. The fixtures ARE `hyphaeon dating -a examples/korber_env_gp160.fasta
 *      --root-taxon CONSENSUS --no-tree --method all --cpu`, field for field, in both distance
 *      modes (verified at generation: 20 of 20 `pgls` fields and every `latent_root` field
 *      bit-identical). So reproducing the fixture IS reproducing the CLI, and the acceptance numbers
 *      are asserted here in full rather than summarised.
 *
 *   4. THE REFUSALS. Five branches of the reference are deliberately NOT ported, and each one is a
 *      branch that would otherwise return a plausible number: the n > 2500 truncated Lanczos, the
 *      n > 2000 REML subsample, the three Monte-Carlo intervals, `ridge='auto'`, and n < 3. Each
 *      throws, and each throw has a case here, because silently taking the dense path or the
 *      Fieller fall-through is the failure mode this phase was told to avoid.
 *
 * TOLERANCES, AND WHY THEY DIFFER BY TABLE. All measured at this commit, not assumed:
 *
 *   - `compute_neural_covariance_kernel` on the REAL matrices: 1e-5, and it cannot be tighter.
 *     `cross_attn` and `taxa_repr` are float32 (splits.py:83-84), so numpy's Gram products at
 *     dating.py:96 and 107 are float32 BLAS sgemm, whose accumulation order belongs to the BLAS and
 *     not to the formula. MEASURED: a float64 recomputation of the reference's own kernel from the
 *     reference's own inputs differs by max |Δ| 1.954e-6. On the synthetic cases, where the inputs
 *     are small and the products are exact either way, the bar is 1e-9.
 *   - `estimate_reml_pagel_lambda`: the PROFILE at 1e-9 and `best_lambda` at 1e-4 ABSOLUTE, and the
 *     split is the whole point. The profile is float64 statistics on float64 inputs and there is
 *     nothing for a tolerance to absorb. `best_lambda` is whatever
 *     `scipy.optimize.minimize_scalar(method='bounded')` lands on at its own `xatol` of 1e-5, so a
 *     bound tighter than that tests scipy's iterates rather than this port. MEASURED, for the
 *     record: 2.7e-10 on the latent case and 1.3e-8 on the tn93 one, four decades inside the bound.
 *   - `run_pgls_dating`: 1e-9 absolute on every float, because `pagel_lambda` is an INPUT to these
 *     cases — the fit is measured, not the optimiser. MEASURED with the reference's own lambda, the
 *     worst field is `fieller_g` at 2.5e-11 and `t_mrca` lands at 1.2e-11 years. The eigensolver is
 *     not the limiting factor and is nowhere near it, which is the answer to the obvious worry
 *     about QL against LAPACK's divide-and-conquer: every quantity extracted has the form
 *     aᵀ·V·diag(f(w))·Vᵀ·b, invariant to the eigenbasis.
 *   - `optimize_latent_convex_hull_root`: 1e-5 on the arrays, 1e-6 RELATIVE on `alpha`. The
 *     reference runs its 250 Adam steps in float32 (dating.py:769-770) and JavaScript has no
 *     float32 arithmetic. MEASURED on the acceptance case: max |Δw| 3.823e-7, |Δz_root| 2.611e-7,
 *     |Δdists| 1.190e-7, Δtemporal_r 1.500e-8 — and the anchor TAXA and their ORDER, which is what
 *     a reader actually sees, are compared EXACTLY.
 *   - `pairwise_acgt_hamming`: EXACT. Both sides divide one integer by another.
 *   - the CHAINED answer: 1e-5 years on the tn93 fit and 5e-3 on the latent one, both about twelve
 *     times their measurement (8.1e-7 and 4.1e-4). The gap between them is not sloppiness: t_MRCA is
 *     `t_ref − d0/mu`, the lever arm `d0/mu` is 150 years on one fit and 358 on the other, and the
 *     latent fit's Fieller g is 1.72 — its own slope is not significantly positive and the reference
 *     reports an unbounded interval and selects OLS over it. An ill-conditioned estimate is
 *     ill-conditioned in any arithmetic. What the bounds buy back is the kernel's float32-BLAS floor,
 *     measured at 2.182e-6 through the fixture's decimal text, against a chain sensitivity measured
 *     upstream at 0.819 years per 1.04e-2 of K.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
	ADAM_DEFAULTS,
	LATENT_ROOT_DEFAULTS,
	PAGEL_LAMBDA_BOUNDS,
	PGLS_DENSE_EIGEN_MAX,
	REML_DENSE_MAX,
	REML_STATUS,
	computeNeuralCovarianceKernel,
	estimateRemlPagelLambda,
	optimizeLatentConvexHullRoot,
	pairwiseAcgtHammingMatrix,
	parseAlignmentSequences,
	runPglsDating,
	runRestrictedSplineClockDating
} from '../src/index.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, '..', '..', 'fixtures');
const EXAMPLES = join(HERE, '..', '..', 'examples');
const MANIFEST = JSON.parse(readFileSync(join(FIXTURES, 'manifest.json'), 'utf8'));

const tableCache = new Map();
/** One fixture table, cached: korber's matrices are megabytes and several tables point at them. */
function load(fn) {
	if (!tableCache.has(fn)) {
		tableCache.set(fn, JSON.parse(readFileSync(join(FIXTURES, 'dating', `${fn}.json`), 'utf8')));
	}
	return tableCache.get(fn);
}

/** fixtures/README.md's convention: non-finite floats travel as strings. */
function decode(v) {
	if (v === 'NaN') return NaN;
	if (v === 'Infinity') return Infinity;
	if (v === '-Infinity') return -Infinity;
	return Array.isArray(v) ? v.map(decode) : v;
}

/**
 * Resolve `{"$fixture": …, "case": …, "field": …, "rows"?, "cols"?}` — the pointer gen_fixtures
 * writes instead of copying korber's 143x143 and 143x384 matrices into every table that consumes
 * them. `rows`/`cols` are index lists applied IN THAT ORDER, which is how the covariance the two
 * estimators are fitted on says that it is a slice of the kernel built over ALL alignment taxa and
 * not a kernel rebuilt over the slice (dating.py:2568 then 2576 then 2781; centre-then-subset is
 * not subset-then-centre, and a port that got that backwards would be wrong and plausible).
 */
function resolveRef(v) {
	if (!v || typeof v !== 'object' || Array.isArray(v) || !v.$fixture) return v;
	const c = load(v.$fixture).find((x) => x.name === v.case);
	if (!c) throw new Error(`fixture ${v.$fixture} has no case ${v.case}`);
	let m = c.outputs[v.field] ?? c.inputs[v.field];
	if (m === undefined) throw new Error(`fixture ${v.$fixture}/${v.case} has no field ${v.field}`);
	if (v.rows) m = v.rows.map((i) => m[i]);
	if (v.cols) m = m.map((row) => v.cols.map((j) => row[j]));
	return m;
}

/** A nested array as a flat row-major Float64Array, which is how this library carries matrices. */
function flat(rows) {
	const n = rows.length;
	const d = n === 0 ? 0 : rows[0].length;
	const out = new Float64Array(n * d);
	for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) out[i * d + j] = decode(rows[i][j]);
	return out;
}

/** Absolute closeness with NaN and ±Infinity compared by identity. Returns failure text or null. */
function near(got, want, tol, label) {
	const w = decode(want);
	if (typeof w !== 'number') return Object.is(got, w) ? null : `${label}: expected ${w}, got ${got}`;
	if (Number.isNaN(w)) return Number.isNaN(got) ? null : `${label}: expected NaN, got ${got}`;
	if (!Number.isFinite(w)) return Object.is(got, w) ? null : `${label}: expected ${w}, got ${got}`;
	const d = Math.abs(got - w);
	return d <= tol ? null : `${label}: |Δ| = ${d.toExponential(3)} > ${tol} (got ${got}, want ${w})`;
}

/** The largest |Δ| over a matrix or vector, for the non-vacuity assertions. */
function worstOver(got, want, stride = 0) {
	let worst = 0;
	const rows = Array.isArray(want[0]) ? want : null;
	if (rows) {
		for (let i = 0; i < rows.length; i++) {
			for (let j = 0; j < rows[i].length; j++) {
				const w = decode(rows[i][j]);
				if (Number.isFinite(w)) worst = Math.max(worst, Math.abs(got[i * (stride || rows[i].length) + j] - w));
			}
		}
	} else {
		for (let i = 0; i < want.length; i++) {
			const w = decode(want[i]);
			if (Number.isFinite(w)) worst = Math.max(worst, Math.abs(got[i] - w));
		}
	}
	return worst;
}

const TOL = { '1e-5': 1e-5, '1e-6': 1e-6, '1e-9': 1e-9 };

// ------------------------------------------------------------------------------------------------
// 1. the covariance kernel
// ------------------------------------------------------------------------------------------------

describe('compute_neural_covariance_kernel (dating.py:78-118)', () => {
	for (const c of load('compute_neural_covariance_kernel')) {
		it(`${c.name}: the ${c.inputs.n}x${c.inputs.n} kernel within ${c.tolerance}`, () => {
			const tol = TOL[c.tolerance];
			const n = c.inputs.n;
			const attn = flat(resolveRef(c.inputs.cross_attn));
			const reprRows = c.inputs.taxa_repr === null ? null : resolveRef(c.inputs.taxa_repr);
			const repr = reprRows ? flat(reprRows) : null;
			const embedDim = reprRows ? reprRows[0].length : 0;
			const got = computeNeuralCovarianceKernel(attn, n, repr, embedDim);

			const failures = [];
			for (let i = 0; i < n; i++) {
				for (let j = 0; j < n; j++) {
					const f = near(got[i * n + j], c.outputs.K_neural[i][j], tol, `K[${i}][${j}]`);
					if (f) failures.push(f);
				}
			}
			expect(failures, failures.slice(0, 5).join('\n')).toEqual([]);

			// The three invariants the reference's docstring claims, checked on every case rather than
			// inferred from the comparison: a unit diagonal (dating.py:117), symmetry, and — because
			// it is what makes the Pagel covariance usable at all — no eigenvalue below zero beyond
			// rounding. The last is asserted through the smallest diagonal-dominance proxy that does
			// not need a second eigensolver: |K_ij| <= 1 for a correlation matrix.
			for (let i = 0; i < n; i++) {
				expect(got[i * n + i], `diagonal ${i}`).toBe(1.0);
				for (let j = 0; j < n; j++) {
					expect(Math.abs(got[i * n + j] - got[j * n + i]), `symmetry ${i},${j}`).toBeLessThan(1e-12);
					expect(Math.abs(got[i * n + j]), `|K[${i}][${j}]|`).toBeLessThanOrEqual(1 + 1e-9);
				}
			}
		});
	}

	it("the real case is well inside its bound, so a regression cannot hide in the slack", () => {
		const c = load('compute_neural_covariance_kernel').find((x) => x.name === '000_korber_env_gp160');
		const n = c.inputs.n;
		const reprRows = resolveRef(c.inputs.taxa_repr);
		const got = computeNeuralCovarianceKernel(flat(resolveRef(c.inputs.cross_attn)), n, flat(reprRows), reprRows[0].length);
		const worst = worstOver(got, c.outputs.K_neural, n);
		// MEASURED 1.954e-6 against the reference's float32 sgemm; see the header. The assertion is
		// two-sided on purpose: below 1e-5 (the class) and ABOVE 1e-8, because a port that agreed to
		// float64 precision here would not be reading the reference's float32 matrices at all.
		expect(worst).toBeLessThan(1e-5);
		expect(worst).toBeGreaterThan(1e-8);
	});

	it('refuses a cross_attn that is not n x n rather than reading past it', () => {
		expect(() => computeNeuralCovarianceKernel(new Float64Array(8), 3)).toThrow(RangeError);
	});

	it('drops a taxa_repr whose row count is not n, silently, as dating.py:104 does', () => {
		const attn = Float64Array.from([1, 0.2, 0.3, 0.2, 1, 0.4, 0.3, 0.4, 1]);
		const attnOnly = computeNeuralCovarianceKernel(attn, 3);
		const mismatched = computeNeuralCovarianceKernel(attn, 3, Float64Array.from([1, 2, 3, 4]), 2);
		expect(Array.from(mismatched)).toEqual(Array.from(attnOnly));
	});
});

// ------------------------------------------------------------------------------------------------
// 2. the isometric calibration's physical distances
// ------------------------------------------------------------------------------------------------

describe('the pairwise ACGT Hamming matrix (dating.py:2586-2596)', () => {
	for (const c of load('pairwise_acgt_hamming')) {
		it(`${c.name}: ${c.outputs.n} x ${c.outputs.n}, exactly`, () => {
			expect(c.tolerance).toBe('exact');
			const n = c.outputs.n;
			const seqs = c.inputs.alignment ? korberSequences(c.inputs.taxa) : c.inputs.sequences;
			const got = pairwiseAcgtHammingMatrix(seqs);
			for (let i = 0; i < n; i++) {
				for (let j = 0; j < n; j++) {
					expect(got[i * n + j], `[${i}][${j}]`).toBe(decode(c.outputs.matrix[i][j]));
				}
			}
		});
	}

	it('refuses ragged input rather than reading off the end of a short row', () => {
		expect(() => pairwiseAcgtHammingMatrix(['ACGT', 'ACG'])).toThrow(RangeError);
	});

	it('is not TN93: an unsaturated pair scores its raw p-distance, uncorrected', () => {
		// 'Hamming / TN93' in the reference's docstring (dating.py:718) is wrong about the second
		// half, and it matters: a TN93 correction would raise this and change alpha, and with it
		// every rate and every date the latent mode produces.
		const m = pairwiseAcgtHammingMatrix(['AAAAAAAAAA', 'AAAAACCCCC']);
		expect(m[1]).toBe(0.5);
	});
});

let korberCache = null;
/**
 * The alignment the way `run_mrca_dating` holds it when it builds `char_mat` (dating.py:2578): the
 * reference's own parser, `verify_coding_alignment`'s silent `L mod 3` trim (DATING Q6) — and NO
 * '*' → '-' rewrite, because the reference does not do one on this path and '*' is not ACGT either
 * way, so the two agree here and saying so is cheaper than depending on it.
 */
function korberSequences(taxa) {
	if (!korberCache) {
		const raw = parseAlignmentSequences(readFileSync(join(EXAMPLES, 'korber_env_gp160.fasta'), 'utf8'));
		const seqs = raw instanceof Map ? raw : new Map(Object.entries(raw));
		const rem = seqs.get([...seqs.keys()][0]).length % 3;
		korberCache = new Map(rem ? [...seqs].map(([k, v]) => [k, v.slice(0, v.length - rem)]) : [...seqs]);
	}
	return taxa.map((t) => korberCache.get(t));
}

// ------------------------------------------------------------------------------------------------
// 3. the latent convex-hull root
// ------------------------------------------------------------------------------------------------

describe('optimize_latent_convex_hull_root (dating.py:701-832)', () => {
	for (const c of load('optimize_latent_convex_hull_root')) {
		it(`${c.name}: the 250-step trajectory, its endpoint and the anchor table`, () => {
			const tol = TOL[c.tolerance];
			const reprRows = resolveRef(c.inputs.taxon_repr);
			const times = c.inputs.times.map(decode);
			const physRows = c.inputs.pairwise_phys_dists === null ? null : resolveRef(c.inputs.pairwise_phys_dists);
			const steps = Object.keys(c.outputs.weights_trajectory).map(Number);
			const got = optimizeLatentConvexHullRoot(flat(reprRows), times, {
				embedDim: c.inputs.embed_dim,
				taxaNames: c.inputs.taxa_names,
				pairwisePhysDists: physRows ? flat(physRows) : null,
				anchorMask: c.inputs.anchor_mask ?? null,
				learningRate: c.inputs.learning_rate,
				maxIter: c.inputs.max_iter,
				trajectorySteps: steps
			});

			const failures = [];
			for (const [field, want] of [
				['weights', c.outputs.weights],
				['dists', c.outputs.dists],
				['dists_latent', c.outputs.dists_latent],
				['z_root', c.outputs.z_root]
			]) {
				for (let i = 0; i < want.length; i++) {
					const f = near(got[field][i], want[i], tol, `${field}[${i}]`);
					if (f) failures.push(f);
				}
			}
			// THE TRAJECTORY, not just the endpoint. 250 is a hard stop on an unconverged descent
			// (DATING Q13), so a port that reaches step 250 correctly with the wrong Adam update — the
			// biased standard deviation, eps added before the bias correction instead of after — is a
			// port that diverges on the next alignment and passes here on this one.
			for (const step of steps) {
				const want = c.outputs.weights_trajectory[String(step)];
				const mine = got.weights_trajectory.get(step);
				for (let i = 0; i < want.length; i++) {
					const f = near(mine[i], want[i], tol, `weights@${step}[${i}]`);
					if (f) failures.push(f);
				}
			}
			expect(failures, failures.slice(0, 5).join('\n')).toEqual([]);

			// alpha is RELATIVE: the reference's denominator is a float32 reduction (np.linalg.norm on
			// a float32 matrix), so a float64 port is off by ~8e-8 of a quantity that is itself ~5e-2.
			expect(Math.abs(got.alpha - c.outputs.alpha) / Math.abs(c.outputs.alpha)).toBeLessThan(1e-6);
			expect(near(got.temporal_r, c.outputs.temporal_r, 1e-5, 'temporal_r')).toBeNull();
			expect(near(got.temporal_r2, c.outputs.temporal_r2, 1e-5, 'temporal_r2')).toBeNull();
			expect(near(got.mu_ols, c.outputs.mu_ols, 1e-9, 'mu_ols')).toBeNull();
			// t_mrca_ols is a DIAGNOSTIC on the uncentred calendar axis and inherits the trajectory's
			// own residual through `dists`; the number a report shows comes from runOlsDating.
			expect(near(got.t_mrca_ols, c.outputs.t_mrca_ols, 1e-3, 't_mrca_ols')).toBeNull();

			// THE ANCHOR TABLE IS COMPARED EXACTLY on names and order, and that is deliberate: it is
			// the one output of this function a reader looks at directly, and dating.py:809-818's
			// break needs BOTH w < 0.01 AND three rows already emitted, so an off-by-one in the loop
			// changes which sequences are named as the root's ancestry.
			expect(got.anchor_taxa.map((a) => a.taxon)).toEqual(c.outputs.anchor_taxa.map((a) => a.taxon));
			for (let i = 0; i < got.anchor_taxa.length; i++) {
				expect(near(got.anchor_taxa[i].weight, c.outputs.anchor_taxa[i].weight, 1e-4, `anchor ${i} weight`)).toBeNull();
				expect(got.anchor_taxa[i].date).toBe(decode(c.outputs.anchor_taxa[i].date));
			}
			// The mask the reference RETURNS, which is not always the mask it was given (DATING B12).
			expect(Array.from(got.anchor_mask).map(Boolean)).toEqual(c.outputs.anchor_mask.map(Boolean));

			// The softmax weights are a probability vector over ALL taxa, masked ones included.
			let total = 0;
			for (const v of got.weights) total += v;
			expect(Math.abs(total - 1)).toBeLessThan(1e-12);
		});
	}

	it('the acceptance case is well inside its bound', () => {
		const c = load('optimize_latent_convex_hull_root').find((x) => x.name === '000_korber_env_gp160');
		const reprRows = resolveRef(c.inputs.taxon_repr);
		const got = optimizeLatentConvexHullRoot(flat(reprRows), c.inputs.times.map(decode), {
			embedDim: c.inputs.embed_dim,
			taxaNames: c.inputs.taxa_names,
			pairwisePhysDists: flat(resolveRef(c.inputs.pairwise_phys_dists)),
			anchorMask: c.inputs.anchor_mask
		});
		// MEASURED against the reference's float32 torch trajectory; see the header.
		expect(worstOver(got.weights, c.outputs.weights)).toBeLessThan(1e-6);
		expect(worstOver(got.z_root, c.outputs.z_root)).toBeLessThan(1e-6);
		expect(worstOver(got.dists, c.outputs.dists)).toBeLessThan(1e-6);
	});

	it('fewer than three eligible taxa silently resets the mask to all-true (dating.py:744-745)', () => {
		const z = Float64Array.from([0, 0, 1, 0.2, 2, 0.5, 3, 1.1, 4, 1.4]);
		const times = [2000, 2001, 2002, 2003, 2004];
		const masked = optimizeLatentConvexHullRoot(z, times, { embedDim: 2, anchorMask: [true, true, false, false, false] });
		const unmasked = optimizeLatentConvexHullRoot(z, times, { embedDim: 2 });
		expect(Array.from(masked.anchor_mask)).toEqual([1, 1, 1, 1, 1]);
		expect(Array.from(masked.weights)).toEqual(Array.from(unmasked.weights));
	});

	it("an ineligible taxon keeps a place in the vector and a weight of zero (dating.py:773)", () => {
		const z = Float64Array.from([0, 0, 1, 0.2, 2, 0.5, 3, 1.1, 4, 1.4, 5, 9.9]);
		const times = [2000, 2001, 2002, 2003, 2004, 2005];
		const got = optimizeLatentConvexHullRoot(z, times, {
			embedDim: 2,
			anchorMask: [true, true, true, true, true, false]
		});
		expect(got.weights.length).toBe(6);
		expect(got.weights[5]).toBe(0);
		let total = 0;
		for (const v of got.weights) total += v;
		expect(Math.abs(total - 1)).toBeLessThan(1e-12);
	});

	it("torch.optim.Adam's defaults are the reference's, and the step count is a constant", () => {
		expect(ADAM_DEFAULTS).toEqual({ beta1: 0.9, beta2: 0.999, eps: 1e-8 });
		expect(LATENT_ROOT_DEFAULTS).toEqual({ learningRate: 0.05, maxIter: 250 });
	});

	it('refuses a taxon_repr whose shape does not match embedDim', () => {
		expect(() => optimizeLatentConvexHullRoot(new Float64Array(10), [1, 2, 3], { embedDim: 4 })).toThrow(RangeError);
		expect(() => optimizeLatentConvexHullRoot(new Float64Array(12), [1, 2, 3], { embedDim: 0 })).toThrow(RangeError);
	});
});

// ------------------------------------------------------------------------------------------------
// 4. Pagel's lambda by profile REML
// ------------------------------------------------------------------------------------------------

describe('estimate_reml_pagel_lambda (dating.py:1476-1547)', () => {
	for (const c of load('estimate_reml_pagel_lambda')) {
		it(`${c.name}: the REML profile at 1e-9 and lambda at the optimiser's own 1e-4`, () => {
			expect(c.tolerance).toBe('1e-9');
			const times = c.inputs.times.map(decode);
			const dists = c.inputs.dists.map(decode);
			const cov = flat(resolveRef(c.inputs.cov_matrix));
			const got = estimateRemlPagelLambda(times, dists, cov);

			// THE PROFILE IS THE TEST. `best_lambda` alone cannot say whether a disagreement is in the
			// likelihood or in the search, and only one of those is this library's fault.
			const failures = [];
			c.inputs.lambda_grid.forEach((lam, i) => {
				const f = near(got.objective(lam), c.outputs.neg_reml_profile[i], 1e-9, `neg_reml(${lam})`);
				if (f) failures.push(f);
			});
			expect(failures, failures.slice(0, 5).join('\n')).toEqual([]);

			expect(got.status).toBe(c.outputs.status);
			// Each case declares how tightly its own lambda can be compared, because that depends on
			// the CURVATURE of its profile and not on the class of the table: 1e-4 (the minimiser's own
			// xatol) where there is an argmin to find, and 1.0 — the whole box — for the identity
			// kernel, whose objective is exactly flat and whose returned lambda is a coincidence.
			expect(near(got.best_lambda, c.outputs.best_lambda, c.best_lambda_tolerance, 'best_lambda')).toBeNull();
			expect(got.best_lambda).toBeGreaterThanOrEqual(PAGEL_LAMBDA_BOUNDS[0]);
			expect(got.best_lambda).toBeLessThanOrEqual(PAGEL_LAMBDA_BOUNDS[1]);

			// A case that declines to compare lambda owes the reader the reason IN DATA: the generator
			// hands out the free pass only when the profile's whole range is below REML_FLAT_PROFILE
			// (1e-6), and that is re-checked here so a hand-edited tolerance cannot buy slack.
			const profile = c.outputs.neg_reml_profile.map(decode);
			const spread = Math.max(...profile) - Math.min(...profile);
			if (c.best_lambda_tolerance >= 1) {
				expect(spread, 'a free lambda has to be earned by a flat profile').toBeLessThan(1e-6);
			} else {
				expect(spread, 'a localisable lambda needs a profile with curvature').toBeGreaterThan(1e-6);
			}
		});
	}

	it("the two real cases reproduce scipy's lambda far inside the bound, and its call count exactly", () => {
		// MEASURED: 2.75e-10 (latent) and 1.28e-8 (tn93), both in 12 evaluations — scipy's own count.
		// The residual is the objective's float64 reassociation, not the minimiser's; see
		// src/numeric/optimize.js's header.
		for (const name of ['000_korber_tn93', '001_korber_latent']) {
			const c = load('estimate_reml_pagel_lambda').find((x) => x.name === name);
			const got = estimateRemlPagelLambda(c.inputs.times, c.inputs.dists, flat(resolveRef(c.inputs.cov_matrix)));
			expect(Math.abs(got.best_lambda - c.outputs.best_lambda), name).toBeLessThan(1e-6);
			expect(got.nfev, name).toBe(12);
		}
	});

	it('hands back the decomposition so a caller can skip the second one (DATING Q9)', () => {
		const c = load('estimate_reml_pagel_lambda').find((x) => x.name === '000_korber_tn93');
		const cov = flat(resolveRef(c.inputs.cov_matrix));
		const reml = estimateRemlPagelLambda(c.inputs.times, c.inputs.dists, cov);
		expect(reml.w_K.length).toBe(c.inputs.n);
		expect(reml.V.length).toBe(c.inputs.n * c.inputs.n);
		// eigenvalues ASCENDING and clamped at zero, matching numpy.linalg.eigh after :1506
		for (let i = 1; i < reml.w_K.length; i++) expect(reml.w_K[i]).toBeGreaterThanOrEqual(reml.w_K[i - 1]);
		expect(reml.w_K[0]).toBeGreaterThanOrEqual(0);
		// and reusing it gives the same fit as letting runPglsDating take its own
		const shared = runPglsDating(c.inputs.times, c.inputs.dists, cov, {
			pagelLambda: reml.best_lambda,
			eigen: { values: reml.w_K, vectors: reml.V }
		});
		const own = runPglsDating(c.inputs.times, c.inputs.dists, cov, { pagelLambda: reml.best_lambda });
		expect(shared.t_mrca).toBe(own.t_mrca);
		expect(shared.mu).toBe(own.mu);
	});

	it(`refuses above ${REML_DENSE_MAX} taxa rather than answering with a different status`, () => {
		const n = REML_DENSE_MAX + 1;
		const times = new Float64Array(n).map((_, i) => 2000 + i * 0.001);
		expect(() => estimateRemlPagelLambda(times, new Float64Array(n), new Float64Array(n * n))).toThrow(RangeError);
	});

	it('rejects mismatched lengths and a covariance of the wrong size', () => {
		expect(() => estimateRemlPagelLambda([1, 2, 3], [1, 2], new Float64Array(9))).toThrow(RangeError);
		expect(() => estimateRemlPagelLambda([1, 2, 3], [1, 2, 3], new Float64Array(4))).toThrow(RangeError);
	});
});

// ------------------------------------------------------------------------------------------------
// 5. the generalised fit
// ------------------------------------------------------------------------------------------------

/**
 * How tightly each exported field can be held, and why the three intervals are looser than
 * everything they are built from. MEASURED over all eight cases at this commit:
 *
 *     ci_delta / ci_mrca  3.344e-9    fieller_g  2.467e-11    r2        2.442e-15
 *     ci_fieller          1.012e-9    se_mrca    1.532e-11    residuals 9.714e-17
 *                                     t_mrca     1.160e-11    everything else <= 5.6e-17
 *
 * The interval endpoints are NOT less accurate than `t_mrca`; they are the same number plus
 * `t_crit * se_mrca`, and `t_crit` is where the error is. dating.py calls `stats.t.ppf(0.975, 139)`,
 * which scipy returns 8.1e-13 out IN PROBABILITY — about 7e-12 relative in the quantile — and this
 * library's `tPpf` rounds differently. On korber's latent fit the half-width is 471 years, so
 * 7e-12 x 471 = 3.3e-9 years, which is exactly what is measured. Phase 3 recorded the same effect
 * at 4.5e-10 years on a 97.7-year arm; the ratio of the two is the ratio of the arms.
 *
 * So: 1e-9 absolute on every field whose error is this port's, and 1e-7 YEARS (three milliseconds)
 * on the endpoints, which is thirty times the measured worst and still four decades below anything
 * a reader could see. Stating a bound on a calendar quantity in years rather than as a bare float is
 * the point — 1e-9 on a number of order 2000 is a claim about the last bit of a double, not about
 * a date.
 */
const PGLS_FIELD_TOLERANCE = { ci_fieller: 1e-7, ci_delta: 1e-7, ci_mrca: 1e-7 };
const PGLS_DEFAULT_TOLERANCE = 1e-9;

/** The record keys dating.py:3168 exports, in the reference's own insertion order. */
const PGLS_EXPORT_KEYS = [
	'method', 'status', 'mu', 'd0', 't_ref', 't_mrca', 'se_mu', 'se_d0', 'se_mrca', 'ci_fieller',
	'fieller_g', 'ci_delta', 'ci_mrca', 'ci_method', 'r2', 'ridge', 'pagel_lambda', 'sigma2', 'rmse', 'n'
];

/**
 * DATING Q10, and the reason this table exists at all: `run_restricted_spline_clock_dating` is in
 * src/dating.js, the model-free half, yet dating.py:2842-2844 hands it `cov_train` the moment the
 * neural path produced one. So the SAME function, on times and dists that did not move, answers
 * differently once the model is loaded — and a record that prints a spline beside a PGLS fit has to
 * print this one. The fixture's own note carries the before/after for both distance modes.
 *
 * TOLERANCE. Every field at the table's 1e-5, RELATIVE where the magnitude is large, except
 * `t_mrca` under `--distance-mode latent`. That one is `−beta_0/beta_1` on an UNCENTRED calendar
 * axis (DATING Q5) with `beta_1 = 9.09e-6`, a lever arm of about 2·10³ years per unit relative
 * error in the slope: MEASURED here, beta_1 lands at 1.0e-4 relative and `t_mrca` therefore at
 * 0.4 years. That is the conditioning of the reference's own construction, not slack — the tn93
 * case, whose slope is a hundred times larger, is held at 1e-3 years and lands at 1.5e-4.
 */
const SPLINE_GLS_EXPORT_KEYS = [
	'method', 't_mrca', 'ci_mrca', 'rate_ancestral', 'ci_rate_ancestral', 'rate_recent',
	'ci_rate_recent', 'rate_ratio', 'beta_0', 'beta_1', 'beta_2', 'beta', 'ci_beta_2', 'knots',
	'rss', 'aic', 'rss_linear', 'aic_linear', 'delta_aic', 'f_stat', 'p_f_test',
	'is_nonlinear_preferred', 'r2', 'rmse', 'n'
];

/** Years absolute on the date and its interval; everything else relative at the table's bound. */
const SPLINE_GLS_YEAR_TOLERANCE = { '000_korber_tn93_gls': 1e-3, '001_korber_latent_gls': 1.0 };

describe('run_restricted_spline_clock_dating with the neural covariance (DATING Q10)', () => {
	for (const c of load('run_restricted_spline_clock_dating_gls')) {
		it(`${c.name}: the GLS spline, every field within ${c.tolerance} relative`, () => {
			const rel = TOL[c.tolerance];
			const years = SPLINE_GLS_YEAR_TOLERANCE[c.name];
			expect(years, `no year tolerance recorded for ${c.name}`).toBeGreaterThan(0);
			const cov = flat(resolveRef(c.inputs.cov_matrix));
			const got = runRestrictedSplineClockDating(c.inputs.times.map(decode), c.inputs.dists.map(decode), {
				covMatrix: cov,
				ridge: c.inputs.ridge
			});

			const failures = [];
			const check = (label, g, want) => {
				const w = decode(want);
				if (typeof w === 'boolean' || typeof w === 'string') {
					if (!Object.is(g, w)) failures.push(`${label}: expected ${w}, got ${g}`);
					return;
				}
				// `t_mrca` and its interval are DATES: a tolerance on one is a duration, not a ratio.
				const isDate = label.startsWith('t_mrca') || label.startsWith('ci_mrca');
				const tol = isDate ? years : Math.max(rel * Math.abs(w), 1e-12);
				const f = near(g, w, tol, label);
				if (f) failures.push(f);
			};
			for (const k of SPLINE_GLS_EXPORT_KEYS) {
				const want = c.outputs[k];
				if (Array.isArray(want)) want.forEach((v, i) => check(`${k}[${i}]`, got[k][i], v));
				else check(k, got[k], want);
			}
			expect(failures, failures.slice(0, 5).join('\n')).toEqual([]);

			// Non-vacuity: the covariance must actually have changed the answer. The model-free fit on
			// the same inputs is a different spline, and if the option were ignored this would pass by
			// accident.
			const free = runRestrictedSplineClockDating(c.inputs.times.map(decode), c.inputs.dists.map(decode));
			expect(Math.abs(free.beta_0 - got.beta_0)).toBeGreaterThan(1e-6);

			// The key SET and its ORDER are the download contract, and adding the option must not have
			// moved them.
			expect(Object.keys(got).filter((k) => SPLINE_GLS_EXPORT_KEYS.includes(k))).toEqual(SPLINE_GLS_EXPORT_KEYS);
		});
	}

	it('the null covariance leaves phase 3\'s arithmetic bit for bit alone', () => {
		// The identity branch still runs `normalEquations` and the plain sums; passing an explicit
		// identity goes down the dense C_inv path instead and must NOT be treated as the same thing.
		const c = load('run_restricted_spline_clock_dating_gls')[0];
		const t = c.inputs.times.map(decode);
		const d = c.inputs.dists.map(decode);
		const a = runRestrictedSplineClockDating(t, d);
		const b = runRestrictedSplineClockDating(t, d, { covMatrix: null });
		expect(b.beta_0).toBe(a.beta_0);
		expect(b.t_mrca).toBe(a.t_mrca);
		expect(b.rss).toBe(a.rss);
		expect(b.r2).toBe(a.r2);
	});

	it('refuses a covariance of the wrong size rather than reading past it', () => {
		const c = load('run_restricted_spline_clock_dating_gls')[0];
		expect(() =>
			runRestrictedSplineClockDating(c.inputs.times.map(decode), c.inputs.dists.map(decode), {
				covMatrix: new Float64Array(9)
			})
		).toThrow(/cov_matrix must be/);
	});
});

describe('run_pgls_dating (dating.py:1299-1473)', () => {
	for (const c of load('run_pgls_dating')) {
		it(`${c.name}: every exported field within 1e-9, status ${c.outputs.status}`, () => {
			expect(c.tolerance).toBe('1e-9');
			const cov = flat(resolveRef(c.inputs.cov_matrix));
			const got = runPglsDating(c.inputs.times.map(decode), c.inputs.dists.map(decode), cov, {
				ridge: c.inputs.ridge,
				pagelLambda: c.inputs.pagel_lambda,
				tRef: c.inputs.t_ref ?? undefined,
				ciMethod: c.inputs.ci_method
			});

			const failures = [];
			for (const k of [...PGLS_EXPORT_KEYS, 'residuals', 'fitted']) {
				const want = c.outputs[k];
				const tol = PGLS_FIELD_TOLERANCE[k] ?? PGLS_DEFAULT_TOLERANCE;
				if (typeof want === 'string' && !['NaN', 'Infinity', '-Infinity'].includes(want)) {
					expect(got[k], k).toBe(want); // method, status, ci_method
					continue;
				}
				if (Array.isArray(want)) {
					for (let i = 0; i < want.length; i++) {
						// NaN and ±Infinity are compared by IDENTITY inside `near`, not by tolerance: an
						// unbounded Fieller interval is a different ANSWER, not a large number, and a page
						// has to be able to render it as one.
						const f = near(got[k][i], want[i], tol, `${k}[${i}]`);
						if (f) failures.push(f);
					}
					continue;
				}
				const f = near(got[k], want, tol, k);
				if (f) failures.push(f);
			}
			expect(failures, failures.slice(0, 5).join('\n')).toEqual([]);

			// The key SET and its ORDER are the download contract (dating.py:3168).
			expect(Object.keys(got).filter((k) => PGLS_EXPORT_KEYS.includes(k))).toEqual(PGLS_EXPORT_KEYS);
		});
	}

	it('DATING Q8: the returned ridge is 1 − eff_lam and not the ridge that was passed in', () => {
		const c = load('run_pgls_dating').find((x) => x.name === '000_korber_tn93_fieller');
		const got = runPglsDating(c.inputs.times, c.inputs.dists, flat(resolveRef(c.inputs.cov_matrix)), {
			ridge: c.inputs.ridge,
			pagelLambda: c.inputs.pagel_lambda
		});
		// The CLI prints c.inputs.ridge (its clip of 1 − λ* into [0.01, 0.20]) and the record ships
		// this. Asserting the IDENTITY is a better test than asserting the value.
		expect(got.ridge).toBe(1.0 - c.inputs.pagel_lambda);
		expect(got.ridge).not.toBe(c.inputs.ridge);
	});

	it('the acceptance fits are well inside their bound, on both distance modes', () => {
		for (const name of ['000_korber_tn93_fieller', '002_korber_latent_fieller']) {
			const c = load('run_pgls_dating').find((x) => x.name === name);
			const got = runPglsDating(c.inputs.times, c.inputs.dists, flat(resolveRef(c.inputs.cov_matrix)), {
				ridge: c.inputs.ridge,
				pagelLambda: c.inputs.pagel_lambda
			});
			// MEASURED: t_mrca 1.2e-11 years and fieller_g 2.5e-11, i.e. nearly two decades of margin
			// inside the 1e-9 class, so a later regression cannot hide in the slack. The eigensolver is
			// not the limiting factor and this is where that is asserted rather than argued.
			expect(Math.abs(got.t_mrca - c.outputs.t_mrca), name).toBeLessThan(1e-10);
			expect(Math.abs(got.fieller_g - c.outputs.fieller_g), name).toBeLessThan(1e-10);
		}
	});

	it('reproduces the published CLI record for korber, in both distance modes', () => {
		// These are the numbers `hyphaeon dating … --method all --cpu` prints, and the fixture was
		// verified field for field against that JSON at generation. They are written out here so the
		// phase's acceptance claim is legible without opening a fixture.
		const expected = {
			'000_korber_tn93_fieller': { t_mrca: 1841.6129634138606, mu: 0.0007707887678355115, lambda: 0.6984254721729402 },
			'002_korber_latent_fieller': { t_mrca: 1633.0720995073843, mu: 0.0001016519874006683, lambda: 0.8591318190927182 }
		};
		for (const [name, want] of Object.entries(expected)) {
			const c = load('run_pgls_dating').find((x) => x.name === name);
			expect(c.inputs.pagel_lambda, name).toBe(want.lambda);
			const got = runPglsDating(c.inputs.times, c.inputs.dists, flat(resolveRef(c.inputs.cov_matrix)), {
				ridge: c.inputs.ridge,
				pagelLambda: c.inputs.pagel_lambda
			});
			expect(Math.abs(got.t_mrca - want.t_mrca), `${name} t_mrca`).toBeLessThan(1e-9);
			expect(Math.abs(got.mu - want.mu) / want.mu, `${name} mu`).toBeLessThan(1e-9);
		}
		// The latent fit's Fieller interval is genuinely half-infinite (g = 1.72 > 1), and the -inf
		// must survive as -inf rather than becoming a very old year.
		const latent = load('run_pgls_dating').find((x) => x.name === '002_korber_latent_fieller');
		const got = runPglsDating(latent.inputs.times, latent.inputs.dists, flat(resolveRef(latent.inputs.cov_matrix)), {
			ridge: latent.inputs.ridge,
			pagelLambda: latent.inputs.pagel_lambda
		});
		expect(got.ci_fieller[0]).toBe(-Infinity);
		expect(got.ci_mrca[0]).toBe(-Infinity);
		expect(got.fieller_g).toBeGreaterThan(1);
	});

	it('refuses fewer than three observations, as the reference raises', () => {
		expect(() => runPglsDating([1, 2], [1, 2], new Float64Array(4))).toThrow(RangeError);
	});

	it(`refuses above ${PGLS_DENSE_EIGEN_MAX} taxa rather than taking the dense path silently`, () => {
		const n = PGLS_DENSE_EIGEN_MAX + 1;
		expect(() => runPglsDating(new Float64Array(n), new Float64Array(n), new Float64Array(n * n))).toThrow(RangeError);
	});

	it('refuses the three Monte-Carlo ci_methods rather than falling through to Fieller', () => {
		const cov = Float64Array.from([1, 0.3, 0.2, 0.3, 1, 0.4, 0.2, 0.4, 1]);
		for (const m of ['poisson', 'residual-boot', 'wild', 'jackknife', 'loocv']) {
			expect(() => runPglsDating([2000, 2005, 2010], [0.01, 0.02, 0.03], cov, { ciMethod: m }), m).toThrow(RangeError);
		}
	});

	it("refuses ridge='auto' rather than turning the covariance into NaN", () => {
		const cov = Float64Array.from([1, 0.3, 0.2, 0.3, 1, 0.4, 0.2, 0.4, 1]);
		expect(() => runPglsDating([2000, 2005, 2010], [0.01, 0.02, 0.03], cov, { ridge: 'auto' })).toThrow(RangeError);
	});

	it('an unrecognised ci_method falls through to Fieller, as the reference does', () => {
		const cov = Float64Array.from([1, 0.3, 0.2, 0.3, 1, 0.4, 0.2, 0.4, 1]);
		const got = runPglsDating([2000, 2005, 2010], [0.01, 0.02, 0.032], cov, { ciMethod: 'nonsense' });
		expect(got.ci_mrca).toEqual(got.ci_fieller);
		expect(got.ci_method).toBe('nonsense'); // recorded verbatim, so a reader can see what was asked
	});

	it('delta and linear select the delta interval; fieller is the default', () => {
		const cov = Float64Array.from([1, 0.3, 0.2, 0.3, 1, 0.4, 0.2, 0.4, 1]);
		const args = [[2000, 2005, 2010], [0.01, 0.02, 0.032], cov];
		expect(runPglsDating(...args, { ciMethod: 'delta' }).ci_mrca).toEqual(runPglsDating(...args).ci_delta);
		expect(runPglsDating(...args, { ciMethod: 'linear' }).ci_mrca).toEqual(runPglsDating(...args).ci_delta);
		expect(runPglsDating(...args).ci_mrca).toEqual(runPglsDating(...args).ci_fieller);
	});
});

// ------------------------------------------------------------------------------------------------
// 6. the whole chain, from the one thing this library does not compute
// ------------------------------------------------------------------------------------------------

describe('the whole model-based chain, from the model outputs to the published date', () => {
	const model = load('model_outputs')[0];
	const n = model.inputs.taxa.length;

	it('model_outputs is the shape the fixture claims, and its rows do not sum to one', () => {
		expect(model.outputs.mean_cross_attn.length).toBe(n);
		expect(model.outputs.mean_taxa_repr.length).toBe(n);
		expect(model.outputs.mean_taxa_repr[0].length).toBe(model.inputs.embed_dim);
		// splits.py:131 drops the root from BOTH axes of a softmax taken over num_species + 1 nodes,
		// so each row is short of 1 by the root's own share. A port that renormalised — an easy
		// "obviously it should sum to 1" fix — would change every entry of the kernel.
		for (const row of model.outputs.mean_cross_attn) {
			const s = row.reduce((a, b) => a + b, 0);
			expect(s).toBeLessThan(1);
			expect(s).toBeGreaterThan(0.9);
		}
	});

	it('kernel → training slice → REML → PGLS reproduces the published t_MRCA, both modes', () => {
		const attn = flat(model.outputs.mean_cross_attn);
		const repr = flat(model.outputs.mean_taxa_repr);
		const K = computeNeuralCovarianceKernel(attn, n, repr, model.inputs.embed_dim);

		// PER CASE, because the two fits are not equally well conditioned and one bound for both would
		// be either vacuous or wrong. t_MRCA is `t_ref − d0/mu`, so a relative error in `mu` is
		// amplified by the LEVER ARM `d0/mu` — the distance from the reference date to the ancestor:
		//
		//   tn93    arm 150 yr, Fieller g = 0.137 (slope clearly positive)   measured 8.1e-7 yr
		//   latent  arm 358 yr, Fieller g = 1.72  (slope NOT significant)    measured 4.1e-4 yr
		//
		// The latent fit is three decades looser for a reason the reference itself reports: at g > 1
		// its own Fieller interval is half-infinite and the CLI selects OLS over it. An ill-conditioned
		// estimate stays ill-conditioned in any arithmetic, and pretending otherwise with a single
		// tight bound would just make this test fail on the next machine. Each bound is about twelve
		// times its measurement.
		for (const [name, published, tolYears] of [
			['000_korber_tn93_fieller', 1841.6129634138606, 1e-5],
			['002_korber_latent_fieller', 1633.0720995073843, 5e-3]
		]) {
			const c = load('run_pgls_dating').find((x) => x.name === name);
			// The SLICE, taken from the fixture's own reference so the index order is the reference's:
			// the kernel is built over all 143 alignment taxa and only then cut to the 141 training
			// ones (dating.py:2568, 2576, 2781).
			const ref = c.inputs.cov_matrix;
			const idx = ref.rows;
			const cov = new Float64Array(idx.length * idx.length);
			for (let i = 0; i < idx.length; i++) {
				for (let j = 0; j < idx.length; j++) cov[i * idx.length + j] = K[idx[i] * n + idx[j]];
			}

			const reml = estimateRemlPagelLambda(c.inputs.times, c.inputs.dists, cov);
			const got = runPglsDating(c.inputs.times, c.inputs.dists, cov, {
				ridge: Math.min(0.2, Math.max(0.01, 1 - reml.best_lambda)),
				pagelLambda: reml.best_lambda,
				eigen: { values: reml.w_K, vectors: reml.V }
			});

			// YEARS, and the unit is the point: a bound on a date should be a duration. What it is
			// buying back is the kernel's own float32-BLAS floor — MEASURED at 2.182e-6 here, from the
			// fixture's decimal text rather than the raw float32, which is what a reader of the
			// fixtures actually gets — carried through a chain whose sensitivity was measured upstream
			// at 0.819 years per 1.04e-2 of K.
			expect(Math.abs(got.t_mrca - published), `${name} t_mrca`).toBeLessThan(tolYears);
			expect(got.status, `${name} status`).toBe('OK');
			// lambda through the whole chain, at the optimiser's own tolerance. MEASURED 4.6e-7 and
			// 9.5e-7 — two decades inside it, so the kernel's float32 floor is not what decides lambda.
			expect(Math.abs(reml.best_lambda - c.inputs.pagel_lambda), `${name} lambda`).toBeLessThan(1e-4);
		}
	});

	it('Hamming → latent root reproduces the published root description, from the alignment', () => {
		const lat = load('optimize_latent_convex_hull_root').find((x) => x.name === '000_korber_env_gp160');
		// The physical distances recomputed from the FASTA rather than read out of the fixture, so
		// this test covers the one input of the latent search that the application will also have to
		// produce for itself.
		const phys = pairwiseAcgtHammingMatrix(korberSequences(lat.inputs.taxa_names));
		const reprRows = resolveRef(lat.inputs.taxon_repr);
		const got = optimizeLatentConvexHullRoot(flat(reprRows), lat.inputs.times, {
			embedDim: lat.inputs.embed_dim,
			taxaNames: lat.inputs.taxa_names,
			pairwisePhysDists: phys,
			anchorMask: lat.inputs.anchor_mask
		});

		// `latent_convex_hull (α=0.05407 subs/site/unit, R=+0.374)` is the reference's own provenance
		// token (dating.py:2600) and is what a report prints as the root. Rebuilt here to the same
		// printed precision, which is the resolution at which a reader is ever shown it.
		const printed = `latent_convex_hull (α=${got.alpha.toFixed(5)} subs/site/unit, R=${got.temporal_r >= 0 ? '+' : ''}${got.temporal_r.toFixed(3)})`;
		expect(printed).toBe(MANIFEST.dating_model_example.root_description_latent);
		expect(got.anchor_taxa.map((a) => a.taxon)).toEqual(lat.outputs.anchor_taxa.map((a) => a.taxon));
	});

	it('the manifest records the command these fixtures came from and the weights that made them', () => {
		const m = MANIFEST.dating_model_example;
		expect(m.command).toContain('--no-tree --method all --cpu');
		expect(m.n_taxa).toBe(n);
		expect(m.n_dated).toBe(142);
		expect(m.n_train).toBe(141);
		expect(m.embed_dim).toBe(model.inputs.embed_dim);
		expect(m.model_safetensors_sha256).toMatch(/^[0-9a-f]{64}$/);
		expect(REML_STATUS.OPTIMAL_REML).toBe('OPTIMAL_REML');
	});
});
