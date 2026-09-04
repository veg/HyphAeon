/**
 * WHY THIS FILE EXISTS
 *
 * Pins src/numeric/prng.js: the reference xoshiro256** / splitmix64 outputs (so the stream a seed
 * produces is a contract, as PLAN.md §5.3 rule 5 requires — a recorded seed must reproduce a run),
 * determinism, the [0, 1) / N(0, 1) moments over 1e5 draws, and the draw-without-replacement
 * semantics the epistasis permutation null relies on (distinct indices, uniform coverage).
 *
 * The first reference outputs for seed 1 are computed independently here in BigInt from the
 * published C (Blackman & Vigna) so that a transcription slip in prng.js cannot hide behind its own
 * output; the splitmix64 vector for seed 0 (0xE220A8397B1DCDAF) is the widely published one.
 */
import { describe, it, expect } from 'vitest';

import { Xoshiro256 } from '../src/numeric/prng.js';

const M64 = (1n << 64n) - 1n;

function splitmix64Stream(seed, count) {
	let s = BigInt.asUintN(64, BigInt(seed));
	const out = [];
	for (let i = 0; i < count; i++) {
		s = (s + 0x9e3779b97f4a7c15n) & M64;
		let z = s;
		z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & M64;
		z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & M64;
		out.push(z ^ (z >> 31n));
	}
	return out;
}

function xoshiroReference(state, count) {
	const s = state.slice();
	const rotl = (x, k) => ((x << k) | (x >> (64n - k))) & M64;
	const out = [];
	for (let i = 0; i < count; i++) {
		out.push((rotl((s[1] * 5n) & M64, 7n) * 9n) & M64);
		const t = (s[1] << 17n) & M64;
		s[2] ^= s[0];
		s[3] ^= s[1];
		s[1] ^= s[2];
		s[0] ^= s[3];
		s[2] ^= t;
		s[3] = rotl(s[3], 45n);
	}
	return out;
}

describe('Xoshiro256: seeding and reference stream', () => {
	it('splitmix64 seeding matches the published vector for seed 0', () => {
		expect(splitmix64Stream(0, 1)[0]).toBe(0xe220a8397b1dcdafn);
		const r = new Xoshiro256(0);
		expect(r.s[0]).toBe(0xe220a8397b1dcdafn);
		expect(r.s).toEqual(splitmix64Stream(0, 4));
	});

	it('the 64-bit stream is xoshiro256** on the splitmix64 state', () => {
		for (const seed of [1, 42, 20260904, 2n ** 63n, -1]) {
			const r = new Xoshiro256(seed);
			const want = xoshiroReference(splitmix64Stream(seed, 4), 8);
			const got = Array.from({ length: 8 }, () => r.next64());
			expect(got).toEqual(want);
		}
	});

	it('is deterministic for a seed and different across seeds', () => {
		const a = new Xoshiro256(42);
		const b = new Xoshiro256(42);
		const c = new Xoshiro256(43);
		const ua = Array.from({ length: 50 }, () => a.uniform());
		const ub = Array.from({ length: 50 }, () => b.uniform());
		const uc = Array.from({ length: 50 }, () => c.uniform());
		expect(ua).toEqual(ub);
		expect(ua).not.toEqual(uc);
		expect(Array.from({ length: 5 }, () => a.normal())).toEqual(Array.from({ length: 5 }, () => b.normal()));
	});

	it('rejects non-integer, unsafe and non-numeric seeds', () => {
		expect(() => new Xoshiro256(0.5)).toThrow(RangeError);
		expect(() => new Xoshiro256(2 ** 53)).toThrow(RangeError);
		// @ts-expect-error deliberate misuse
		expect(() => new Xoshiro256('42')).toThrow(TypeError);
	});

	it('next() is the high 32 bits of next64() and uniform() the top 53', () => {
		const a = new Xoshiro256(7);
		const b = new Xoshiro256(7);
		const c = new Xoshiro256(7);
		for (let i = 0; i < 20; i++) {
			const w = a.next64();
			expect(b.next()).toBe(Number(w >> 32n));
			expect(c.uniform()).toBe(Number(w >> 11n) * 2 ** -53);
		}
	});
});

describe('Xoshiro256: distribution moments over 1e5 draws', () => {
	const N = 100000;

	it('uniform() lies in [0, 1) with mean 1/2 and variance 1/12', () => {
		const r = new Xoshiro256(20260904);
		let sum = 0;
		let sumSq = 0;
		let min = 1;
		let max = 0;
		for (let i = 0; i < N; i++) {
			const u = r.uniform();
			expect(u >= 0 && u < 1).toBe(true);
			sum += u;
			sumSq += u * u;
			if (u < min) min = u;
			if (u > max) max = u;
		}
		const mean = sum / N;
		const variance = sumSq / N - mean * mean;
		// standard error of the mean is sqrt(1/12/N) ≈ 9.1e-4; 4 s.e.
		expect(Math.abs(mean - 0.5)).toBeLessThan(4 * Math.sqrt(1 / 12 / N));
		expect(Math.abs(variance - 1 / 12)).toBeLessThan(0.002);
		expect(min).toBeLessThan(1e-3);
		expect(max).toBeGreaterThan(1 - 1e-3);
	});

	it('normal() has mean 0, variance 1, and near-zero skew and excess kurtosis', () => {
		const r = new Xoshiro256(11);
		const xs = new Float64Array(N);
		let sum = 0;
		for (let i = 0; i < N; i++) {
			xs[i] = r.normal();
			sum += xs[i];
		}
		const mean = sum / N;
		let m2 = 0;
		let m3 = 0;
		let m4 = 0;
		for (let i = 0; i < N; i++) {
			const d = xs[i] - mean;
			m2 += d * d;
			m3 += d * d * d;
			m4 += d * d * d * d;
		}
		m2 /= N;
		m3 /= N;
		m4 /= N;
		expect(Math.abs(mean)).toBeLessThan(4 / Math.sqrt(N));
		expect(Math.abs(m2 - 1)).toBeLessThan(0.02);
		expect(Math.abs(m3 / Math.pow(m2, 1.5))).toBeLessThan(0.05);
		expect(Math.abs(m4 / (m2 * m2) - 3)).toBeLessThan(0.1);
		// Tail mass: P(|Z| > 3) = 0.0027
		let tail = 0;
		for (let i = 0; i < N; i++) if (Math.abs(xs[i]) > 3) tail++;
		expect(Math.abs(tail / N - 0.0027)).toBeLessThan(0.001);
	});

	it('normal() never produces NaN or infinity (log(0) avoided)', () => {
		const r = new Xoshiro256(3);
		for (let i = 0; i < 20000; i++) expect(Number.isFinite(r.normal())).toBe(true);
	});
});

describe('Xoshiro256: integers, shuffle, choiceWithoutReplacement', () => {
	it('integers with replacement cover 0..n-1 uniformly', () => {
		const r = new Xoshiro256(5);
		const n = 10;
		const draws = r.integers(n, 100000);
		expect(draws).toBeInstanceOf(Int32Array);
		const counts = new Array(n).fill(0);
		for (const d of draws) {
			expect(d >= 0 && d < n).toBe(true);
			counts[d]++;
		}
		for (const c of counts) expect(Math.abs(c / 100000 - 0.1)).toBeLessThan(0.006);
	});

	it('choiceWithoutReplacement draws k distinct indices and every index equally often', () => {
		const r = new Xoshiro256(99);
		const n = 40;
		const k = 8;
		const counts = new Array(n).fill(0);
		const B = 20000;
		for (let b = 0; b < B; b++) {
			const draw = r.choiceWithoutReplacement(n, k);
			expect(draw).toBeInstanceOf(Int32Array);
			expect(draw.length).toBe(k);
			expect(new Set(draw).size).toBe(k);
			for (const d of draw) counts[d]++;
		}
		const expected = (B * k) / n; // 4000
		for (const c of counts) expect(Math.abs(c - expected) / expected).toBeLessThan(0.05);
		expect(r.choiceWithoutReplacement(5, 5).slice().sort()).toEqual(Int32Array.from([0, 1, 2, 3, 4]));
		expect(r.choiceWithoutReplacement(5, 0).length).toBe(0);
		expect(() => r.choiceWithoutReplacement(5, 6)).toThrow(RangeError);
		expect(Array.from(r.integers(5, 5, { replace: false })).sort()).toEqual([0, 1, 2, 3, 4]);
	});

	it('shuffle is an in-place permutation with every position equally likely', () => {
		const r = new Xoshiro256(2024);
		const n = 6;
		const pos = Array.from({ length: n }, () => new Array(n).fill(0));
		const B = 30000;
		for (let b = 0; b < B; b++) {
			const arr = [0, 1, 2, 3, 4, 5];
			const same = r.shuffle(arr);
			expect(same).toBe(arr);
			expect(arr.slice().sort()).toEqual([0, 1, 2, 3, 4, 5]);
			arr.forEach((v, i) => pos[v][i]++);
		}
		for (const row of pos) for (const c of row) expect(Math.abs(c / B - 1 / n)).toBeLessThan(0.02);
	});
});
