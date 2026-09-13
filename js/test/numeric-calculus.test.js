/**
 * WHY THIS FILE EXISTS
 *
 * `src/numeric/calculus.js` is three numpy routines, and the only thing worth proving about them is
 * that they land on NUMPY'S OWN NUMBERS, not on the ones a correct-looking formula produces. Every
 * one of the three has a branch or a tie that a "mathematically equivalent" transcription gets
 * wrong, and all three sit on the deterministic half of the temporal pillar, which
 * PLAN-TEMPORAL.md §5.4 holds at the strict graph class:
 *
 *   - `np.linspace` assigns its last element instead of computing it, so the endpoint is exact and
 *     the second-to-last is not what `start + i·(stop−start)/(num−1)` gives.
 *   - `np.gradient` takes its uniform shortcut only on EXACT equality of the spacings, and the
 *     fixture pins the surprise: a linspace axis is NOT uniformly spaced in floating point, so the
 *     non-uniform three-point branch is what runs for every real time axis. `uniform_for_gradient`
 *     is recorded in the fixture as a boolean and asserted against `isUniformSpacing`, so the test
 *     says WHICH branch ran and not merely that the numbers matched.
 *   - `np.trapezoid` reduces its half-trapezoids pairwise, and a running accumulation is a
 *     different float.
 *
 * The file also owns the fixture replay for `numpyVarFloat64` / `numpyStdFloat64` (`src/numeric/
 * reduce.js`), which the same `fixtures/temporal/numeric.json` covers, and — since round two of the
 * temporal review — the direct proof that their optional `scratch` buffer is BIT-invisible: the
 * last describe block compares the two call shapes as float64 bit patterns rather than as values,
 * including the buffers the guard must refuse.
 *
 * The cases come from `fixtures/temporal/numeric.json`, written by `scripts/gen_fixtures.py --only
 * temporal` — which calls numpy directly and needs neither weights nor a model — so each is a
 * replay rather than an opinion.
 *
 * TOLERANCES. `linspace` is EXACT: it is two float operations per element and there is nothing for a
 * correct port to round differently, so any tolerance at all would only hide a missing endpoint
 * assignment. `gradient`, `trapezoid` and `var` carry the fixtures' 1e-9 class; the measured worst
 * deviations are asserted to sit far below it (see the per-case assertion), so a later regression
 * cannot hide inside the slack.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
	numpyLinspace,
	numpyGradient,
	numpyGradientRows,
	numpyTrapezoid,
	numpyTrapezoidRows,
	isUniformSpacing,
	numpyVarFloat64,
	numpyStdFloat64,
	numpyMeanFloat64
} from '../src/index.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, '..', '..', 'fixtures');
const CASES = JSON.parse(readFileSync(join(FIXTURES, 'temporal', 'numeric.json'), 'utf8'));
const byPrefix = (p) => CASES.filter((c) => c.name.startsWith(p));

/** Worst absolute difference between two sequences, for the "below the bound" assertions. */
function worst(a, b) {
	let m = 0;
	for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
	return m;
}

describe('numpyLinspace (np.linspace, endpoint=True)', () => {
	const cases = byPrefix('linspace_');
	it('has every case the generator wrote', () => {
		expect(cases.length).toBe(7);
	});

	for (const c of cases) {
		it(`${c.name}: reproduces numpy EXACTLY`, () => {
			const y = numpyLinspace(c.inputs.start, c.inputs.stop, c.inputs.num);
			expect(Array.from(y)).toEqual(c.outputs.y);
		});
	}

	it('assigns the endpoint rather than computing it', () => {
		// With num > 1 the last element is bit-exactly `stop`. On some spans `start + i·step` lands
		// there too; on others it does not, and numpy's assignment is what makes the outcome
		// independent of which. Both halves are asserted: every case ends exactly on `stop`, and the
		// computed form is shown to disagree on at least one span.
		for (const c of cases) {
			if (c.inputs.num < 2) continue;
			const y = numpyLinspace(c.inputs.start, c.inputs.stop, c.inputs.num);
			expect(y[y.length - 1]).toBe(c.inputs.stop);
		}
		// MEASURED: on [0.1, 0.7] with 38 points the computed form misses the endpoint by one ulp.
		const start = 0.1;
		const stop = 0.7;
		const num = 38;
		const step = (stop - start) / (num - 1);
		expect(start + (num - 1) * step).not.toBe(stop);
		expect(numpyLinspace(start, stop, num)[num - 1]).toBe(stop);
	});

	it('a linspace time axis is NOT uniformly spaced in floating point', () => {
		// This is the measurement the whole gradient port hangs on: if it were uniform, the
		// two-point shortcut would be correct and the three-point coefficients would be dead code.
		const c = cases.find((x) => x.name === 'linspace_acceptance_axis_T60');
		expect(c.outputs.uniform_for_gradient).toBe(false);
		expect(c.outputs.distinct_spacings).toBeGreaterThan(1);
		expect(isUniformSpacing(numpyLinspace(c.inputs.start, c.inputs.stop, c.inputs.num))).toBe(false);
	});
});

describe('numpyGradient (np.gradient, edge_order=1)', () => {
	const cases = byPrefix('gradient_');
	it('has every case the generator wrote', () => {
		expect(cases.length).toBe(6);
	});

	for (const c of cases) {
		it(`${c.name}: reproduces numpy within 1e-9, and far below it`, () => {
			const g = numpyGradient(c.inputs.f, c.inputs.x);
			const w = worst(g, c.outputs.gradient);
			expect(w).toBeLessThan(1e-9);
			// Measured at this commit: every case is at or below 1e-15 in absolute terms. The
			// assertion is 1e-11 so a regression of four orders still fails rather than hiding.
			expect(w).toBeLessThan(1e-11);
		});
	}

	it('takes the uniform branch only on an exactly uniform axis, and the numbers differ', () => {
		const nonUniform = cases.find((x) => x.name === 'gradient_pulse_on_acceptance_axis');
		const uniform = cases.find((x) => x.name === 'gradient_pulse_on_uniform_axis');
		expect(isUniformSpacing(nonUniform.inputs.x)).toBe(false);
		expect(isUniformSpacing(uniform.inputs.x)).toBe(true);
		// On the non-uniform axis the two-point shortcut is a DIFFERENT answer, which is the whole
		// reason the branch exists. Compare the shortcut against numpy's own output.
		const x = nonUniform.inputs.x;
		const f = nonUniform.inputs.f;
		const h2 = 2 * (x[1] - x[0]);
		let differs = 0;
		for (let i = 1; i < f.length - 1; i++) {
			if ((f[i + 1] - f[i - 1]) / h2 !== nonUniform.outputs.gradient[i]) differs++;
		}
		expect(differs).toBeGreaterThan(0);
	});

	it('the edges are one-sided first differences, not second-order', () => {
		const c = cases.find((x) => x.name === 'gradient_random_on_acceptance_axis');
		const { f, x } = c.inputs;
		const n = f.length;
		expect(c.outputs.gradient[0]).toBeCloseTo((f[1] - f[0]) / (x[1] - x[0]), 14);
		expect(c.outputs.gradient[n - 1]).toBeCloseTo((f[n - 1] - f[n - 2]) / (x[n - 1] - x[n - 2]), 14);
		const g = numpyGradient(f, x);
		expect(g[0]).toBe((f[1] - f[0]) / (x[1] - x[0]));
	});

	it('T = 2 gives the same one-sided difference in both entries', () => {
		const c = cases.find((x) => x.name === 'gradient_two_points');
		const g = numpyGradient(c.inputs.f, c.inputs.x);
		expect(Array.from(g)).toEqual(c.outputs.gradient);
		expect(g[0]).toBe(g[1]);
	});

	it('numpyGradientRows is the per-row loop of the same routine', () => {
		const a = cases.find((x) => x.name === 'gradient_pulse_on_acceptance_axis');
		const b = cases.find((x) => x.name === 'gradient_random_on_acceptance_axis');
		const T = a.inputs.f.length;
		const M = Float64Array.from([...a.inputs.f, ...b.inputs.f]);
		const out = numpyGradientRows(M, 2, T, a.inputs.x);
		expect(worst(out.subarray(0, T), a.outputs.gradient)).toBeLessThan(1e-11);
		expect(worst(out.subarray(T, 2 * T), b.outputs.gradient)).toBeLessThan(1e-11);
	});

	it('refuses a single point, as numpy does', () => {
		expect(() => numpyGradient([1], [0])).toThrow(RangeError);
	});
});

describe('numpyTrapezoid (np.trapezoid)', () => {
	const cases = byPrefix('trapezoid_');
	it('has every case the generator wrote', () => {
		expect(cases.length).toBe(5);
	});

	for (const c of cases) {
		it(`${c.name}: reproduces numpy within 1e-9, and far below it`, () => {
			const v = numpyTrapezoid(c.inputs.y, c.inputs.x);
			const e = Math.abs(v - c.outputs.integral);
			expect(e).toBeLessThan(1e-9);
			expect(e).toBeLessThan(1e-13);
		});
	}

	it('integrates a single point to 0, as summing an empty array does', () => {
		expect(numpyTrapezoid([7], [1])).toBe(0);
	});

	it('numpyTrapezoidRows is the per-row loop of the same routine', () => {
		const a = cases.find((x) => x.name === 'trapezoid_positive_pulse_acceptance_axis');
		const b = cases.find((x) => x.name === 'trapezoid_negative_values');
		const T = a.inputs.y.length;
		const M = Float64Array.from([...a.inputs.y, ...b.inputs.y]);
		const out = numpyTrapezoidRows(M, 2, T, a.inputs.x);
		expect(Math.abs(out[0] - a.outputs.integral)).toBeLessThan(1e-13);
		expect(Math.abs(out[1] - b.outputs.integral)).toBeLessThan(1e-13);
	});
});

describe('numpyVarFloat64 (np.var, ddof = 0)', () => {
	const cases = byPrefix('var_');
	it('has every case the generator wrote', () => {
		expect(cases.length).toBe(4);
	});

	for (const c of cases) {
		it(`${c.name}: reproduces numpy's two-pass variance`, () => {
			expect(Math.abs(numpyVarFloat64(c.inputs.x) - c.outputs.var)).toBeLessThan(1e-9);
			expect(Math.abs(numpyStdFloat64(c.inputs.x) - c.outputs.std)).toBeLessThan(1e-9);
			expect(Math.abs(numpyMeanFloat64(c.inputs.x) - c.outputs.mean)).toBeLessThan(1e-9);
		});
	}

	it('is two passes, not the one-pass identity that is algebraically the same', () => {
		// E[x²] − E[x]² cancels catastrophically on an offset sample: on [1e8, 1e8+1, 1e8+2] the
		// true variance is 2/3 and the one-pass form loses every digit of it. numpy does not use it
		// and neither may this port, which is why the routine allocates a deviation buffer.
		const x = [1e8, 1e8 + 1, 1e8 + 2];
		let s = 0;
		let s2 = 0;
		for (const v of x) {
			s += v;
			s2 += v * v;
		}
		const onePass = s2 / x.length - (s / x.length) ** 2;
		expect(numpyVarFloat64(x)).toBeCloseTo(2 / 3, 12);
		expect(Math.abs(onePass - 2 / 3)).toBeGreaterThan(1e-3);
	});

	it('n = 1 with ddof = 0 is 0, not NaN', () => {
		expect(numpyVarFloat64([4])).toBe(0);
	});
});

describe('the fixture table itself', () => {
	it('is the one the manifest describes', () => {
		const manifest = JSON.parse(readFileSync(join(FIXTURES, 'manifest.json'), 'utf8'));
		expect(manifest.counts.temporal.numeric).toBe(CASES.length);
		expect(CASES.length).toBe(22);
	});
});

/**
 * The `scratch` parameter is a PERFORMANCE affordance on a routine whose whole reason to exist is
 * that its arithmetic is exactly numpy's. It must therefore be provably invisible: the two call
 * shapes are asserted BIT-identical, not close, because the null this feeds compares `v_p >= v_obs`
 * and a single ulp on one side of that test moves an exceedance count.
 *
 * The shapes cross both of `numpyPairwiseSum`'s branch boundaries (n < 8, n <= 128, the recursive
 * halving above it) and both degenerate cases (n = 0 -> NaN, n = 1 -> 0), and the buffers cover the
 * ones the guard must REJECT — a `Float32Array`, a too-short `Float64Array`, a plain `Array` — since
 * a rejected buffer takes the allocating path and must land on the same bits as no buffer at all.
 */
describe('numpyVarFloat64 / numpyStdFloat64: the scratch buffer is bit-invisible', () => {
	/** The float64 bit pattern, so NaN compares equal to NaN and −0 does not compare equal to 0. */
	const bits = (/** @type {number} */ v) => {
		const b = new Float64Array(1);
		b[0] = v;
		return new BigUint64Array(b.buffer)[0];
	};

	/** Data with a different float character per shape: a plain ramp, an offset ramp that makes the
	 * one-pass identity fail, a Gaussian bump like a smoothed velocity row, and alternating signs. */
	const generators = {
		ramp: (/** @type {number} */ i) => i * 0.25,
		offset: (/** @type {number} */ i) => 1e8 + i,
		bump: (/** @type {number} */ i) => 3.7e-4 * Math.exp(-(((i - 30) / 8) ** 2) / 2),
		alternating: (/** @type {number} */ i) => (i % 2 ? -1 : 1) * (1 + i) ** 1.5
	};
	const SHAPES = [0, 1, 2, 7, 8, 9, 15, 16, 128, 129, 137, 246, 512];

	for (const [gname, gen] of Object.entries(generators)) {
		it(`${gname}: every shape and every lo agrees to the bit, with and without a buffer`, () => {
			for (const n of SHAPES) {
				for (const lo of [0, 3]) {
					const x = Float64Array.from({ length: lo + n + 5 }, (_, i) => gen(i));
					const exact = new Float64Array(Math.max(n, 1));
					const roomy = new Float64Array(n + 64).fill(-7); // dirty on purpose
					for (const [label, scratch] of /** @type {[string, any][]} */ ([
						['exact-length Float64Array', exact],
						['over-length dirty Float64Array', roomy],
						['Float32Array (must be refused)', new Float32Array(n + 8)],
						['short Float64Array (must be refused)', new Float64Array(Math.max(n - 1, 0))],
						['plain Array (must be refused)', new Array(n + 8).fill(0)],
						['null', null],
						['undefined', undefined]
					])) {
						const why = `${gname} n=${n} lo=${lo} scratch=${label}`;
						expect(bits(numpyVarFloat64(x, lo, n, scratch)), `var ${why}`).toBe(
							bits(numpyVarFloat64(x, lo, n))
						);
						expect(bits(numpyStdFloat64(x, lo, n, scratch)), `std ${why}`).toBe(
							bits(numpyStdFloat64(x, lo, n))
						);
					}
				}
			}
		});
	}

	it('the degenerate shapes are the documented ones, and the buffer does not change them', () => {
		const s = new Float64Array(8);
		expect(Number.isNaN(numpyVarFloat64([1, 2, 3], 0, 0, s))).toBe(true);
		expect(bits(numpyVarFloat64([1, 2, 3], 0, 0, s))).toBe(bits(numpyVarFloat64([1, 2, 3], 0, 0)));
		expect(numpyVarFloat64([4], 0, 1, s)).toBe(0);
		expect(numpyStdFloat64([4], 0, 1, s)).toBe(0);
	});

	it('one buffer reused across calls gives the same bits as a fresh one each time', () => {
		// The hot-loop usage: {@link temporalNullDraws} hands the SAME Float64Array to every row of
		// every draw, so stale deviations from the previous row must not be able to leak in.
		const T = 60;
		const shared = new Float64Array(T);
		const rows = Array.from({ length: 12 }, (_, r) =>
			Float64Array.from({ length: T }, (_, t) => (r + 1) * Math.sin(t / 3) * 1e-4)
		);
		const reused = rows.map((row) => bits(numpyVarFloat64(row, 0, T, shared)));
		const fresh = rows.map((row) => bits(numpyVarFloat64(row, 0, T, new Float64Array(T))));
		const none = rows.map((row) => bits(numpyVarFloat64(row, 0, T)));
		expect(reused).toEqual(none);
		expect(fresh).toEqual(none);
	});

	it('a Float32Array buffer would not be harmless, which is why the guard is a type test', () => {
		// The same two passes with the guard made length-only, to show what is being refused: the
		// squared deviations round to float32 and the variance moves. MEASURED on the acceptance
		// run's own velocity rows in the round-two review; this is the smallest case that shows it.
		const x = Float64Array.from({ length: 60 }, (_, i) => 1e8 + Math.sin(i) * 3);
		const mean = (() => {
			let s = 0;
			for (const v of x) s += v;
			return s / x.length;
		})();
		const dev32 = new Float32Array(60);
		const dev64 = new Float64Array(60);
		for (let i = 0; i < 60; i++) {
			const d = x[i] - mean;
			dev32[i] = d * d;
			dev64[i] = d * d;
		}
		let s32 = 0;
		let s64 = 0;
		for (let i = 0; i < 60; i++) {
			s32 += dev32[i];
			s64 += dev64[i];
		}
		expect(s32).not.toBe(s64);
		// and the guarded routine is on the float64 side of that gap
		expect(bits(numpyVarFloat64(x, 0, 60, dev32))).toBe(bits(numpyVarFloat64(x, 0, 60)));
	});
});
