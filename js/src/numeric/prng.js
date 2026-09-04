/**
 * WHY THIS FILE EXISTS
 *
 * The one seeded random number generator the library uses, so that every Monte Carlo output the
 * ports produce (epistasis.py:263 `np.random.default_rng(rng_seed)` for the sector permutation
 * null; phenotype.py:318,326 `np.random.seed(seed)` / `np.random.randn(M, n_perm)` for the
 * Brownian-motion permulations) is reproducible from a recorded seed. PLAN.md §5.3 rule 5 and
 * fixtures/README.md make randomness a documented contract: the Python streams (PCG64, MT19937)
 * cannot be matched bit-for-bit, so those outputs are compared statistically (§5.4) and the
 * manifest names the PRNG and seed. This is that PRNG: xoshiro256** (Blackman & Vigna 2018)
 * seeded through splitmix64, as PLAN.md decision D17 specifies.
 *
 * WHAT IT DELIBERATELY DOES NOT DO. It never calls Math.random and takes no entropy from the
 * environment: a caller without a seed has no generator. It does not try to reproduce numpy's
 * `choice` or `randn` sequences. The state is four 64-bit words held as BigInt, which is slower
 * than a split-32-bit implementation but leaves the reference algorithm readable and verifiable
 * against the published C; measured throughput is 4.9 M uniforms/s on an M-series laptop and
 * 11 µs per `choiceWithoutReplacement(3000, 40)`, so a B = 10,000 permutation null spends ~0.1 s
 * drawing indices (PLAN.md §5.4).
 *
 * CONVENTIONS:
 *   - `uniform()` takes the top 53 bits of the 64-bit output, giving a float64 in [0, 1) with
 *     every representable multiple of 2⁻⁵³ equally likely (the generator authors' recommendation).
 *   - `normal()` is Box–Muller with the second variate cached, so consecutive calls consume one
 *     pair of uniforms per two normals. log(0) is avoided by using 1 − u ∈ (0, 1].
 *   - Draws without replacement are a partial Fisher–Yates over 0..n−1, so `integers(n, k, {replace:
 *     false})` and `choiceWithoutReplacement(n, k)` both cost O(n) memory and O(k) swaps.
 */

const M64 = (1n << 64n) - 1n;
const TWO_POW_NEG53 = 2 ** -53;

/** rotl on a 64-bit BigInt. */
function rotl(x, k) {
	return ((x << k) | (x >> (64n - k))) & M64;
}

/**
 * xoshiro256** seeded with splitmix64.
 */
export class Xoshiro256 {
	/**
	 * @param {number|bigint} seed any integer (numbers must be safe integers); reduced mod 2⁶⁴
	 */
	constructor(seed) {
		if (typeof seed === 'number') {
			if (!Number.isSafeInteger(seed)) throw new RangeError(`Xoshiro256 seed must be a safe integer, got ${seed}`);
			seed = BigInt(seed);
		} else if (typeof seed !== 'bigint') {
			throw new TypeError('Xoshiro256 seed must be a number or bigint');
		}
		this.seed = BigInt.asUintN(64, seed);
		let sm = this.seed;
		/** @type {bigint[]} */
		this.s = [0n, 0n, 0n, 0n];
		for (let i = 0; i < 4; i++) {
			// splitmix64
			sm = (sm + 0x9e3779b97f4a7c15n) & M64;
			let z = sm;
			z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & M64;
			z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & M64;
			z = z ^ (z >> 31n);
			this.s[i] = z;
		}
		this._hasSpare = false;
		this._spare = 0;
	}

	/**
	 * Next 64-bit output as a BigInt in [0, 2⁶⁴).
	 * @returns {bigint}
	 */
	next64() {
		const s = this.s;
		const result = (rotl((s[1] * 5n) & M64, 7n) * 9n) & M64;
		const t = (s[1] << 17n) & M64;
		s[2] ^= s[0];
		s[3] ^= s[1];
		s[1] ^= s[2];
		s[0] ^= s[3];
		s[2] ^= t;
		s[3] = rotl(s[3], 45n);
		return result;
	}

	/**
	 * Next 32-bit unsigned integer (the high 32 bits of the 64-bit output).
	 * @returns {number}
	 */
	next() {
		return Number(this.next64() >> 32n);
	}

	/**
	 * Uniform float64 in [0, 1) from the top 53 bits.
	 * @returns {number}
	 */
	uniform() {
		return Number(this.next64() >> 11n) * TWO_POW_NEG53;
	}

	/**
	 * Standard normal variate, Box–Muller with the pair cached.
	 * @returns {number}
	 */
	normal() {
		if (this._hasSpare) {
			this._hasSpare = false;
			return this._spare;
		}
		const u1 = 1 - this.uniform(); // (0, 1]
		const u2 = this.uniform();
		const r = Math.sqrt(-2 * Math.log(u1));
		const theta = 2 * Math.PI * u2;
		this._spare = r * Math.sin(theta);
		this._hasSpare = true;
		return r * Math.cos(theta);
	}

	/**
	 * `count` integers from 0..n−1.
	 *
	 * @param {number} n exclusive upper bound, ≥ 1
	 * @param {number} count number of draws
	 * @param {{replace?: boolean}} [options] replace defaults to true; without replacement needs
	 *   count ≤ n and is a partial Fisher–Yates
	 * @returns {Int32Array}
	 */
	integers(n, count, options = {}) {
		const replace = options.replace !== false;
		if (!Number.isInteger(n) || n < 1) throw new RangeError(`integers: n must be a positive integer, got ${n}`);
		if (!Number.isInteger(count) || count < 0) throw new RangeError(`integers: count must be a non-negative integer, got ${count}`);
		if (replace) {
			const out = new Int32Array(count);
			for (let i = 0; i < count; i++) out[i] = Math.floor(this.uniform() * n);
			return out;
		}
		return this.choiceWithoutReplacement(n, count);
	}

	/**
	 * In-place Fisher–Yates (Durstenfeld) shuffle; returns the same array.
	 *
	 * @template {{length: number, [i: number]: any}} T
	 * @param {T} array
	 * @returns {T}
	 */
	shuffle(array) {
		for (let i = array.length - 1; i > 0; i--) {
			const j = Math.floor(this.uniform() * (i + 1));
			const tmp = array[i];
			array[i] = array[j];
			array[j] = tmp;
		}
		return array;
	}

	/**
	 * k distinct integers from 0..n−1 in draw order (partial Fisher–Yates), the index analogue of
	 * numpy's `rng.choice(pool, size=k, replace=False)`.
	 *
	 * @param {number} n population size
	 * @param {number} k number to draw, 0 ≤ k ≤ n
	 * @returns {Int32Array}
	 */
	choiceWithoutReplacement(n, k) {
		if (!Number.isInteger(n) || n < 0) throw new RangeError(`choiceWithoutReplacement: n must be a non-negative integer, got ${n}`);
		if (!Number.isInteger(k) || k < 0 || k > n) throw new RangeError(`choiceWithoutReplacement: k must satisfy 0 <= k <= n, got k=${k}, n=${n}`);
		const pool = new Int32Array(n);
		for (let i = 0; i < n; i++) pool[i] = i;
		const out = new Int32Array(k);
		for (let i = 0; i < k; i++) {
			const j = i + Math.floor(this.uniform() * (n - i));
			const v = pool[j];
			pool[j] = pool[i];
			pool[i] = v;
			out[i] = v;
		}
		return out;
	}
}
