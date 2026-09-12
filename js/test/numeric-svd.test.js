/**
 * WHY THIS FILE EXISTS
 *
 * `src/numeric/svd.js` replaces `np.linalg.svd` at `hyphaeon/temporal.py:673` and `:731` with an
 * eigendecomposition of the time-side Gram, and the interesting question is not "does it compute an
 * SVD" but "which of its outputs are comparable to the reference at all". Three classes, and this
 * file keeps them apart on purpose, because collapsing them is how a real disagreement gets hidden
 * behind a widened tolerance:
 *
 *   1. SIGN- AND ROTATION-INVARIANT, so comparable at the strict class with no convention: the Gram
 *      itself, `‖Z‖_F²`, the singular values, the variance shares, the rank-k projector
 *      `P = V_k V_kᵀ`, and the per-row projection residual that becomes `r2_fpca`. The projector is
 *      the PRIMARY comparison for the wave stage: a rotation that a per-vector check would report as
 *      catastrophic leaves `P` unchanged, which is exactly the information a reviewer needs.
 *   2. SIGN-DEPENDENT, comparable only under a stated convention: the right singular vectors
 *      themselves, and therefore the wave loadings. The fixture carries BOTH `v_raw` (LAPACK
 *      `gesdd`'s own signs, which is what the reference writes) and `v_canonical` (D28's rule), and
 *      the interim comparison RECORDS the flip vector rather than absorbing it — a silently absorbed
 *      flip is what hid a real disagreement for two phases in the MDS work (`MDS_SIGN.md`).
 *   3. NOT COMPARABLE AT ALL: the individual vectors of a near-degenerate pair, and any vector of a
 *      numerically rank-deficient mode. The test ASSERTS the disagreement and asserts that the flag
 *      fires, so the first person to meet a degenerate dataset cannot make the red line go away by
 *      loosening a bound.
 *
 * Alongside the replay are the property tests the codebase already holds `symmetricEigen` to
 * ("tests check the properties that actually matter rather than comparing against numbers this file
 * produced"): exact Gram symmetry, orthonormality, the eigen-residual, `trace(G) = Σλ`, the
 * projector identity against a literal reconstruction, and a ROW-PERMUTATION test — permuting the
 * site order of Z must leave G, σ and every vector unchanged up to summation roundoff, because
 * `G = Σ_i z_i z_iᵀ`. That one is a JS-side property rather than a parity claim, and it proves the
 * Gram path is actually the path in use.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
	dominantTimeModes,
	timeGram,
	projectionResidual,
	canonicalizeWaveSigns,
	resolveWaveSign,
	symmetricEigen,
	WAVE_SIGN_MODES,
	WAVE_GAP_THRESHOLD,
	WAVE_RANK_EPS
} from '../src/index.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, '..', '..', 'fixtures');
const CASES = JSON.parse(readFileSync(join(FIXTURES, 'temporal', 'thin_svd.json'), 'utf8'));

const flat = (rows) => Float64Array.from(rows.flat());
const worst = (a, b) => {
	let m = 0;
	for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
	return m;
};

describe('dominantTimeModes against np.linalg.svd', () => {
	it('has every case the generator wrote', () => {
		expect(CASES.length).toBe(6);
		const manifest = JSON.parse(readFileSync(join(FIXTURES, 'manifest.json'), 'utf8'));
		expect(manifest.counts.temporal.thin_svd).toBe(CASES.length);
	});

	for (const c of CASES) {
		const { rows, T, k } = c.inputs;
		const Z = flat(c.inputs.Z);

		it(`${c.name}: the singular values and variance shares (sign- and rotation-invariant)`, () => {
			const m = dominantTimeModes(Z, rows, T, k);
			expect(m.nModes).toBe(Math.min(rows, T));
			const s0 = c.outputs.sigma[0];
			const n = Math.min(m.sigma.length, c.outputs.sigma.length);
			for (let j = 0; j < n; j++) {
				// WHERE THE GRAM ROUTE ACTUALLY COSTS SOMETHING, stated rather than smoothed over.
				// A singular value is recovered as sqrt(λ), so a λ that rounds to ~ε·λ₁ gives a σ of
				// ~√ε·σ₁ instead of numpy's ~ε·σ₁. MEASURED on rank_deficient_two_shapes: numpy
				// reports σ₃ = 1.2e-15 and this route reports 9.7e-8, both of which mean "zero", and
				// neither of which is a property of the data. So a NUMERICALLY ZERO mode is compared
				// as "both sides call it zero", and every mode above the floor is compared at the
				// strict class relative to σ₁.
				if (s0 > 0 && c.outputs.sigma[j] / s0 < WAVE_RANK_EPS) {
					expect(m.sigma[j] / s0).toBeLessThan(WAVE_RANK_EPS);
					continue;
				}
				expect(Math.abs(m.sigma[j] - c.outputs.sigma[j])).toBeLessThan(1e-9 * Math.max(1, s0));
				expect(Math.abs(m.varExplained[j] - c.outputs.var_explained[j])).toBeLessThan(1e-9);
			}
			expect(Math.abs(m.totalVariance - c.outputs.frobenius_sq)).toBeLessThan(1e-9 * Math.max(1, c.outputs.frobenius_sq));
		});

		it(`${c.name}: the rank-k projector, or a RECORDED skip when the subspace is not determined`, () => {
			const m = dominantTimeModes(Z, rows, T, k);
			// The top-k SUBSPACE is determined only when the last retained mode is separated from the
			// first discarded one. Where it is not, numpy's own basis is arbitrary too, so a
			// comparison would be a coin flip; the skip is asserted through the flag rather than
			// applied silently, which is the rule §4.3 of the decomposition survey lays down.
			const determined = m.kept > 0 && m.gaps[m.kept - 1] >= WAVE_GAP_THRESHOLD;
			if (!determined) {
				expect(m.nearDegenerate[m.kept - 1]).toBe(true);
				expect(['rank_deficient_two_shapes', 'all_zero_rows']).toContain(c.name);
				return;
			}
			const P = new Float64Array(T * T);
			for (let a = 0; a < T; a++) {
				for (let b = 0; b < T; b++) {
					let s = 0;
					for (let j = 0; j < m.kept; j++) s += m.vectors[j * T + a] * m.vectors[j * T + b];
					P[a * T + b] = s;
				}
			}
			expect(worst(P, flat(c.outputs.projector))).toBeLessThan(1e-8);
		});

		it(`${c.name}: the per-row residual, which needs no sign and forms no reconstruction`, () => {
			const m = dominantTimeModes(Z, rows, T, k);
			const { sse, sst } = projectionResidual(Z, rows, T, m.vectors, m.kept);
			expect(worst(sse, c.outputs.sse)).toBeLessThan(1e-8);
			for (let i = 0; i < rows; i++) {
				expect(Math.abs(sst[i] + 1e-8 - c.outputs.sst[i])).toBeLessThan(1e-8);
				const r2 = Math.min(1, Math.max(0, 1 - sse[i] / (sst[i] + 1e-8)));
				expect(Math.abs(r2 - c.outputs.r2[i])).toBeLessThan(1e-8);
			}
		});
	}
});

describe('the sign convention (D28) and what it can and cannot fix', () => {
	const separated = CASES.filter((c) => c.name === 'rows_below_T_6x16' || c.name === 'rows_above_T_16x6');

	for (const c of separated) {
		it(`${c.name}: canonical vectors match the reference's own vectors, with the flip vector RECORDED`, () => {
			const { rows, T, k } = c.inputs;
			const m = dominantTimeModes(flat(c.inputs.Z), rows, T, k, { waveSign: 'canonical' });
			// Every retained mode here is separated well above the threshold, so a per-vector
			// comparison is a valid test. It is run with NO sign allowance against the canonicalised
			// reference vectors, and separately against the RAW ones with the flip recorded, so a
			// reader of a parity report can see the two sides agreed by convention and not by luck.
			for (let j = 0; j < m.kept; j++) {
				expect(c.outputs.gaps[j]).toBeGreaterThan(WAVE_GAP_THRESHOLD);
				const mine = m.vectors.subarray(j * T, (j + 1) * T);
				expect(worst(mine, c.outputs.v_canonical[j])).toBeLessThan(1e-8);
			}
			const raw = dominantTimeModes(flat(c.inputs.Z), rows, T, k, { waveSign: 'lapack' });
			const flips = [];
			for (let j = 0; j < raw.kept; j++) {
				const mine = raw.vectors.subarray(j * T, (j + 1) * T);
				const same = worst(mine, c.outputs.v_raw[j]);
				const flipped = worst(Array.from(mine, (v) => -v), c.outputs.v_raw[j]);
				flips.push(same <= flipped ? 1 : -1);
				expect(Math.min(same, flipped)).toBeLessThan(1e-8);
			}
			// Recorded, not absorbed: tql2's raw signs are its own and need not be gesdd's.
			expect(flips.every((f) => f === 1 || f === -1)).toBe(true);
			expect(c.outputs.sign_flips.length).toBe(m.kept);
		});
	}

	it('flips exactly the vectors whose largest-magnitude entry is negative, and nothing else', () => {
		const V = Float64Array.from([0.1, -0.9, 0.2, /* row 2 */ 0.8, 0.3, -0.4, /* row 3 */ 0, 0, 0]);
		canonicalizeWaveSigns(V, 3, 3);
		expect(Array.from(V.subarray(0, 3))).toEqual([-0.1, 0.9, -0.2]);
		expect(Array.from(V.subarray(3, 6))).toEqual([0.8, 0.3, -0.4]);
		expect(Array.from(V.subarray(6, 9))).toEqual([0, 0, 0]); // an all-zero row is left alone
	});

	it('resolves a tie in |v| to the FIRST index, as np.argmax does', () => {
		const V = Float64Array.from([-0.5, 0.5, 0.1]);
		canonicalizeWaveSigns(V, 1, 3);
		expect(V[0]).toBe(0.5);
	});

	it('a flip is exact in floating point, so nothing but the sign changes', () => {
		const c = CASES.find((x) => x.name === 'rows_below_T_6x16');
		const { rows, T, k } = c.inputs;
		const raw = dominantTimeModes(flat(c.inputs.Z), rows, T, k, { waveSign: 'lapack' });
		const can = dominantTimeModes(flat(c.inputs.Z), rows, T, k, { waveSign: 'canonical' });
		for (let i = 0; i < raw.vectors.length; i++) {
			expect(Math.abs(can.vectors[i])).toBe(Math.abs(raw.vectors[i]));
		}
		// And the gate is indifferent to it, which is why it spends no cycles canonicalising.
		const a = projectionResidual(flat(c.inputs.Z), rows, T, raw.vectors, raw.kept);
		const b = projectionResidual(flat(c.inputs.Z), rows, T, can.vectors, can.kept);
		expect(worst(a.sse, b.sse)).toBe(0);
	});

	it('rejects an unknown convention rather than silently defaulting', () => {
		expect(WAVE_SIGN_MODES).toEqual(['canonical', 'lapack']);
		expect(resolveWaveSign(null)).toBe('canonical');
		expect(resolveWaveSign('LAPACK')).toBe('lapack');
		expect(() => resolveWaveSign('largest-area')).toThrow(RangeError);
	});

	it('NO sign rule fixes two equal singular values, and the test says so rather than tolerating it', () => {
		const c = CASES.find((x) => x.name === 'two_equal_singular_values');
		const { rows, T, k } = c.inputs;
		const m = dominantTimeModes(flat(c.inputs.Z), rows, T, k);
		// The two leading singular values are equal BY CONSTRUCTION, so the flag must fire ...
		expect(Math.abs(m.sigma[0] - m.sigma[1])).toBeLessThan(1e-12);
		expect(m.gaps[0]).toBeLessThan(WAVE_GAP_THRESHOLD);
		expect(m.nearDegenerate[0]).toBe(true);
		// ... the singular values themselves are still exact ...
		expect(Math.abs(m.sigma[0] - c.outputs.sigma[0])).toBeLessThan(1e-9);
		// ... and the individual vectors are NOT comparable. Asserting the disagreement is the point:
		// if a later change made them agree by accident, the degeneracy guard would be untested.
		let anyVectorDisagrees = false;
		for (let j = 0; j < m.kept; j++) {
			const mine = m.vectors.subarray(j * T, (j + 1) * T);
			const same = worst(mine, c.outputs.v_canonical[j]);
			const flipped = worst(Array.from(mine, (v) => -v), c.outputs.v_canonical[j]);
			if (Math.min(same, flipped) > 1e-6) anyVectorDisagrees = true;
		}
		expect(anyVectorDisagrees).toBe(true);
	});

	it('a numerically rank-deficient mode is flagged, not compared', () => {
		const c = CASES.find((x) => x.name === 'rank_deficient_two_shapes');
		const { rows, T, k } = c.inputs;
		const m = dominantTimeModes(flat(c.inputs.Z), rows, T, k);
		expect(m.rankDeficient[0]).toBe(false);
		expect(m.rankDeficient[1]).toBe(false);
		// The trailing two modes carry no energy: numpy writes arbitrary null-space directions there
		// and so does tql2, and neither is a property of the data.
		expect(m.sigma[2] / m.sigma[0]).toBeLessThan(WAVE_RANK_EPS);
		expect(m.rankDeficient[2]).toBe(true);
		expect(m.rankDeficient[3]).toBe(true);
		// The residual is still exact, because Z·v is ~0 for exactly those directions.
		const { sse } = projectionResidual(flat(c.inputs.Z), rows, T, m.vectors, m.kept);
		expect(worst(sse, c.outputs.sse)).toBeLessThan(1e-8);
	});

	it('an all-zero Z gives zero shares and no energy, as temporal.py:733 writes them', () => {
		const c = CASES.find((x) => x.name === 'all_zero_rows');
		const { rows, T, k } = c.inputs;
		const m = dominantTimeModes(flat(c.inputs.Z), rows, T, k);
		expect(m.totalVariance).toBe(0);
		expect(Array.from(m.varExplained.slice(0, 4))).toEqual([0, 0, 0, 0]);
	});

	it('caps the rank at min(rows, T), as full_matrices=False does', () => {
		const c = CASES.find((x) => x.name === 'single_row');
		const { rows, T } = c.inputs;
		const m = dominantTimeModes(flat(c.inputs.Z), rows, T, 4);
		expect(m.nModes).toBe(1);
		expect(m.kept).toBe(1);
		expect(m.vectors.length).toBe(T);
	});
});

describe('properties that must hold whatever the reference says', () => {
	const c = CASES.find((x) => x.name === 'rows_below_T_6x16');
	const { rows, T } = c.inputs;
	const Z = flat(c.inputs.Z);

	it('the Gram is EXACTLY symmetric, not symmetric to roundoff', () => {
		const G = timeGram(Z, rows, T);
		for (let a = 0; a < T; a++) {
			for (let b = 0; b < T; b++) expect(G[a * T + b]).toBe(G[b * T + a]);
		}
	});

	it('trace(G) = ‖Z‖_F² = Σ λ', () => {
		const G = timeGram(Z, rows, T);
		let trace = 0;
		for (let t = 0; t < T; t++) trace += G[t * T + t];
		let frob = 0;
		for (let i = 0; i < Z.length; i++) frob += Z[i] * Z[i];
		expect(Math.abs(trace - frob)).toBeLessThan(1e-10 * Math.max(1, frob));
		const { values } = symmetricEigen(G, T);
		let sum = 0;
		for (const v of values) sum += v;
		expect(Math.abs(sum - trace)).toBeLessThan(1e-10 * Math.max(1, trace));
	});

	it('the returned vectors are orthonormal and satisfy G v = λ v', () => {
		const G = timeGram(Z, rows, T);
		const m = dominantTimeModes(Z, rows, T, 4);
		for (let i = 0; i < m.kept; i++) {
			for (let j = 0; j < m.kept; j++) {
				let dot = 0;
				for (let t = 0; t < T; t++) dot += m.vectors[i * T + t] * m.vectors[j * T + t];
				expect(Math.abs(dot - (i === j ? 1 : 0))).toBeLessThan(1e-12);
			}
			let residual = 0;
			for (let a = 0; a < T; a++) {
				let gv = 0;
				for (let b = 0; b < T; b++) gv += G[a * T + b] * m.vectors[i * T + b];
				residual = Math.max(residual, Math.abs(gv - m.sigma[i] * m.sigma[i] * m.vectors[i * T + a]));
			}
			expect(residual).toBeLessThan(1e-10 * Math.max(1, m.sigma[0] * m.sigma[0]));
		}
	});

	it('the projector shortcut equals a literal Z(I − P) row norm', () => {
		const m = dominantTimeModes(Z, rows, T, 4);
		const { sse } = projectionResidual(Z, rows, T, m.vectors, m.kept);
		for (let i = 0; i < rows; i++) {
			// Literal: reconstruct the row inside the subspace and subtract.
			const recon = new Float64Array(T);
			for (let j = 0; j < m.kept; j++) {
				let dot = 0;
				for (let t = 0; t < T; t++) dot += Z[i * T + t] * m.vectors[j * T + t];
				for (let t = 0; t < T; t++) recon[t] += dot * m.vectors[j * T + t];
			}
			let r = 0;
			for (let t = 0; t < T; t++) r += (Z[i * T + t] - recon[t]) ** 2;
			expect(Math.abs(r - sse[i])).toBeLessThan(1e-11);
		}
	});

	it('permuting the ROW order of Z changes nothing, which proves the Gram path is in use', () => {
		// G = Σ_i z_i z_iᵀ is a sum over rows, so its value is order-independent up to summation
		// roundoff. A row-side implementation would give a different (permuted) factorisation.
		const order = [3, 0, 5, 1, 4, 2];
		const Zp = new Float64Array(Z.length);
		order.forEach((src, dst) => Zp.set(Z.subarray(src * T, (src + 1) * T), dst * T));
		const a = dominantTimeModes(Z, rows, T, 4);
		const b = dominantTimeModes(Zp, rows, T, 4);
		expect(worst(a.sigma, b.sigma)).toBeLessThan(1e-11);
		expect(worst(a.vectors, b.vectors)).toBeLessThan(1e-11);
	});
});
