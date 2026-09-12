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
