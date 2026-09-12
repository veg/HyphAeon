/**
 * WHY THIS FILE EXISTS
 *
 * src/dating.js is the model-free half of the ChronAeon pillar, and the only thing worth proving
 * about it is that it lands on the REFERENCE'S OWN NUMBERS. A suite that checked the port against
 * the port would prove nothing here: every formula in it is short enough to transcribe wrongly and
 * plausibly, and a wrong ancestor date does not look wrong.
 *
 * Three layers, which fail for different reasons, so a failure says WHERE the fault is:
 *
 *   1. FUNCTION REPLAY. `fixtures/dating/*.json` is produced by the reference functions themselves —
 *      lifted out of hyphaeon/dating.py and hyphaeon/dataset.py by `ast` rather than retyped
 *      (scripts/gen_fixtures.py, `load_dating_reference`; regenerate with `--only dating`, which
 *      needs no weights) — so each case is a replay, not an opinion. Distances, estimators and
 *      intervals are separate tables, so a discrepancy cannot hide behind a pipeline.
 *
 *   2. THE ACCEPTANCE RUN. `it('reproduces hyphaeon dating …')` takes two REAL alignments as bytes,
 *      walks the whole chain — parse → the L mod 3 trim → '*' → '-' → parse_header_timestamp →
 *      tree-free divergences → the coverage holdout → runOlsDating and the spline → the per-taxon
 *      table — and compares every field against `fixtures/dating/run_mrca_dating.json`, which IS the
 *      reference CLI's own output. That is the phase's acceptance test, and its measured worst
 *      deviations are asserted to sit at least 100x below their bounds so a later regression cannot
 *      hide inside the slack.
 *
 *   3. THE PROVENANCE. `dating_reference_source` is asserted against a hard-coded map of file and
 *      line ranges, as dates.test.js does, so an upstream edit surfaces as a data diff rather than a
 *      mystery failure; and the per-file case counts are hard-coded so a silently shortened table
 *      fails.
 *
 * TOLERANCES, AND WHY THEY DIFFER BY FIELD. All measured at this commit, not assumed:
 *
 *   - divergences, sampling dates, statuses, counts, booleans: EXACT. The distances are float32
 *     values widened to float64; a port that stayed in float64 would be wrong by ~6e-8 relative for
 *     free, and exact equality is the cheapest possible detector of a missing rounding step.
 *   - the OLS block: 1e-9 ABSOLUTE, the fixtures' own special-function class. Measured worst on
 *     korber: mu 8.7e-19, t_mrca 0, se_mrca 1.8e-15, the Fieller endpoints 4.5e-10 years. The
 *     endpoints are the loosest of them and the reason is not this port: scipy's `t.ppf(0.975, 139)`
 *     is 8.1e-13 out IN PROBABILITY, so the reference's own t_crit differs from the correct one at
 *     7e-12 relative, which the 97.7-year lever arm turns into half a nanosecond of calendar time.
 *   - `p_value`: 1e-15 absolute, and it is the only field that cannot be held tighter.
 *     dating.py:1269 spells it `1 - f.cdf` where `f.sf` was meant, so it quantises at one ulp of
 *     1.0; on H1N1 the reference reports 1.11e-16 where the correctly rounded `1 - cdf` is 0 and the
 *     true survival is 1.59e-24 (fixtures/manifest.json, DATING Q4).
 *   - the SPLINE block: 1e-5 absolute. Not slack — DATING Q5: the reference solves UNCENTRED normal
 *     equations on a calendar axis at cond 9.3e12, so partial-pivot Gaussian elimination lands 1.1e-8
 *     years away on korber and 9.3e-7 on H1N1, and merely reassociating the reference's own products
 *     moves it 3.5e-10. Centring would change the answer and is forbidden.
 *   - `predicted_date` / `temporal_residual`: 1e-6 years (0.03 seconds). Under the spline these come
 *     from brentq at xtol 2e-12 on top of a beta that already differs at 1e-9 relative, so the bound
 *     is inherited from Q5 and not from the root finder, which reproduces the reference to 2.5e-11
 *     years given identical coefficients.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
	clockFittedAndPredicted,
	computeDeltaMrcaInterval,
	computeFiellerMrcaInterval,
	computeRcsBasis,
	computeTreeFreeDivergences,
	consensusSequence,
	parseAlignmentSequences,
	parseHeaderTimestamp,
	precisionWeightedEnsemble,
	residualScale,
	runOlsDating,
	runRestrictedSplineClockDating,
	timeDecayConsensusSequence,
	tn93CrossDistanceMatrix
} from '../src/index.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, '..', '..', 'fixtures');
const EXAMPLES = join(HERE, '..', '..', 'examples');
const load = (fn) => JSON.parse(readFileSync(join(FIXTURES, 'dating', `${fn}.json`), 'utf8'));
const MANIFEST = JSON.parse(readFileSync(join(FIXTURES, 'manifest.json'), 'utf8'));

/**
 * The reference bodies these tables were generated from, at the commit the fixtures were written.
 * An upstream edit that moves or resizes one of them changes this map, and the diff says which.
 */
const EXPECTED_SOURCE = {
	verify_coding_alignment: 'hyphaeon/dating.py:135-222',
	generate_consensus_sequence: 'hyphaeon/dating.py:225-245',
	generate_time_decay_consensus_sequence: 'hyphaeon/dating.py:248-310',
	compute_tree_free_divergences: 'hyphaeon/dating.py:624-698',
	compute_fieller_mrca_interval: 'hyphaeon/dating.py:839-887',
	run_ols_dating: 'hyphaeon/dating.py:1170-1296',
	compute_rcs_basis: 'hyphaeon/dating.py:1774-1806',
	run_restricted_spline_clock_dating: 'hyphaeon/dating.py:1809-1968',
	parse_header_timestamp: 'hyphaeon/dating.py:317-364',
	// The four bodies phase 4 added. They are lifted by the SAME `load_dating_reference`, so they
	// land in the same provenance map and this file asserts all of them even though
	// test/dating-model.test.js is what replays them — one map, one place it is checked.
	compute_neural_covariance_kernel: 'hyphaeon/dating.py:78-118',
	optimize_latent_convex_hull_root: 'hyphaeon/dating.py:701-832',
	run_pgls_dating: 'hyphaeon/dating.py:1299-1473',
	estimate_reml_pagel_lambda: 'hyphaeon/dating.py:1476-1547',
	parse_alignment_sequences: 'hyphaeon/dataset.py:252-364',
	compute_tn93_distance_matrix: 'hyphaeon/dataset.py:715-821',
	compute_tn93_cross_distance_matrix: 'hyphaeon/dataset.py:824-928',
	parse_date_to_decimal: 'hyphaeon/temporal.py:73-151',
	extract_date_from_string: 'hyphaeon/temporal.py:196-245'
};

/** Hard-coded so a table that quietly loses cases fails rather than passing with fewer. */
const EXPECTED_COUNTS = {
	generate_consensus_sequence: 6,
	generate_time_decay_consensus_sequence: 8,
	compute_tn93_cross_distance_matrix: 5,
	compute_tree_free_divergences: 7,
	compute_rcs_basis: 5,
	compute_fieller_mrca_interval: 9,
	run_ols_dating: 12,
	run_restricted_spline_clock_dating: 5,
	// The covariance arm, DATING Q10: the same function fitted with the neural kernel the PGLS fit
	// uses, in both distance modes. It lives in its own table because the model-free one is generated
	// without the checkpoint and could not carry a covariance.
	run_restricted_spline_clock_dating_gls: 2,
	run_mrca_dating: 3,
	// Phase 4's acceptance target: the same CLI with --method all, in both distance modes, so the
	// application can diff a whole record. Nothing in this library replays it — it is orchestration,
	// and `runtime/test/dating-model.test.js` in hyphaeon-app is what reads it.
	run_mrca_dating_model: 2,
	// phase 4, replayed by test/dating-model.test.js
	model_outputs: 1,
	compute_neural_covariance_kernel: 5,
	pairwise_acgt_hamming: 2,
	optimize_latent_convex_hull_root: 5,
	estimate_reml_pagel_lambda: 5,
	run_pgls_dating: 8
};

/** fixtures/README.md's convention: non-finite floats travel as strings. */
function decode(v) {
	if (v === 'NaN') return NaN;
	if (v === 'Infinity') return Infinity;
	if (v === '-Infinity') return -Infinity;
	return Array.isArray(v) ? v.map(decode) : v;
}

/** The same, for a taxon -> date object. A NaN date is how the reference says "undated". */
function decodeDates(m) {
	return new Map(Object.entries(m).map(([k, v]) => [k, decode(v)]));
}

/** Absolute closeness with NaN and ±Infinity compared by identity. Returns failure text or null. */
function near(got, want, tol, label) {
	const w = decode(want);
	if (Number.isNaN(w)) return Number.isNaN(got) ? null : `${label}: expected NaN, got ${got}`;
	if (!Number.isFinite(w)) return Object.is(got, w) ? null : `${label}: expected ${w}, got ${got}`;
	if (typeof w !== 'number') return Object.is(got, w) ? null : `${label}: expected ${w}, got ${got}`;
	const d = Math.abs(got - w);
	return d <= tol ? null : `${label}: |Δ| = ${d.toExponential(3)} > ${tol} (got ${got}, want ${w})`;
}

/** The largest |Δ| seen, for the non-vacuity assertions. */
function worstOf(got, want) {
	const w = decode(want);
	if (typeof w !== 'number' || !Number.isFinite(w) || !Number.isFinite(got)) return 0;
	return Math.abs(got - w);
}

// ------------------------------------------------------------------------------------------------
// 3. provenance
// ------------------------------------------------------------------------------------------------

describe('the fixtures describe the reference bodies they were generated from', () => {
	it('dating_reference_source names every lifted function and its line range', () => {
		expect(MANIFEST.dating_reference_source).toEqual(EXPECTED_SOURCE);
	});

	it('every dating table has the number of cases it was written with', () => {
		expect(MANIFEST.counts.dating).toEqual(EXPECTED_COUNTS);
		for (const [fn, n] of Object.entries(EXPECTED_COUNTS)) expect(load(fn).length, fn).toBe(n);
	});

	it('the fourteen DATING quirks these two ports replicate are named in the manifest', () => {
		// Q1-Q7 are the model-free half's (src/dating.js), Q8-Q14 the model-based half's
		// (src/datingModel.js). The count is asserted as well as the membership so a quirk that is
		// silently dropped from the manifest fails rather than passing with fewer.
		const quirks = MANIFEST.known_quirks.filter((q) => q.startsWith('DATING Q'));
		expect(quirks.length).toBe(14);
		for (let i = 1; i <= 14; i++) expect(quirks.some((q) => q.startsWith(`DATING Q${i}:`)), `DATING Q${i}`).toBe(true);
	});
});

// ------------------------------------------------------------------------------------------------
// 1. function replay
// ------------------------------------------------------------------------------------------------

describe('generate_consensus_sequence (dating.py:225-245)', () => {
	for (const c of load('generate_consensus_sequence')) {
		it(`${c.name}: the consensus string, exactly`, () => {
			expect(c.tolerance).toBe('exact');
			expect(consensusSequence(c.inputs.seq_dict, c.inputs.taxa ?? undefined)).toBe(c.outputs.consensus);
		});
	}

	it('refuses an empty taxon list, as the reference raises', () => {
		expect(() => consensusSequence({ a: 'ACGT' }, [])).toThrow(RangeError);
	});
});

describe('generate_time_decay_consensus_sequence (dating.py:248-310)', () => {
	for (const c of load('generate_time_decay_consensus_sequence')) {
		it(`${c.name}: the string exactly and eff_gamma to 1e-15`, () => {
			expect(c.tolerance).toBe('exact');
			const seqs = c.inputs.alignment ? alignmentFor(c.inputs.alignment).cleaned : c.inputs.seq_dict;
			const got = timeDecayConsensusSequence(seqs, decodeDates(c.inputs.dates_map), c.inputs.taxa ?? undefined, {
				gamma: c.inputs.gamma,
				halfLife: c.inputs.half_life
			});
			expect(got.sequence).toBe(c.outputs.consensus);
			expect(near(got.gamma, c.outputs.eff_gamma, 1e-15, 'eff_gamma')).toBeNull();
		});
	}
});

describe('compute_tn93_cross_distance_matrix (dataset.py:824-928)', () => {
	for (const c of load('compute_tn93_cross_distance_matrix')) {
		it(`${c.name}: the N x M float32 matrix, exactly`, () => {
			expect(c.tolerance).toBe('exact');
			const [n, m] = c.outputs.shape;
			const got = tn93CrossDistanceMatrix(c.inputs.seq_dict, c.inputs.taxa_all, c.inputs.taxa_landmarks);
			expect(got.length).toBe(n * m);
			for (let i = 0; i < n; i++) {
				for (let j = 0; j < m; j++) {
					expect(got[i * m + j], `[${i}][${j}]`).toBe(decode(c.outputs.matrix[i][j]));
				}
			}
		});
	}
});

describe('compute_tree_free_divergences (dating.py:624-698)', () => {
	for (const c of load('compute_tree_free_divergences')) {
		it(`${c.name}: ${c.outputs.root_description}`, () => {
			expect(c.tolerance).toBe('exact');
			const seqs = c.inputs.alignment ? alignmentFor(c.inputs.alignment).cleaned : c.inputs.seq_dict;
			const got = computeTreeFreeDivergences(seqs, c.inputs.dated_taxa, decodeDates(c.inputs.dates_map), {
				rootTaxon: c.inputs.root_taxon,
				decayGamma: c.inputs.decay_gamma,
				decayHalfLife: c.inputs.decay_half_life
			});
			expect(got.root_description).toBe(c.outputs.root_description);
			expect(got.divergences.length).toBe(c.outputs.divergences.length);
			for (let i = 0; i < got.divergences.length; i++) {
				expect(got.divergences[i], `divergence ${i}`).toBe(decode(c.outputs.divergences[i]));
			}
		});
	}
});

describe('compute_rcs_basis (dating.py:1774-1806)', () => {
	for (const c of load('compute_rcs_basis')) {
		it(`${c.name}: B and dB within 1e-9`, () => {
			expect(c.tolerance).toBe('1e-9');
			const { basis, derivative, ncols } = computeRcsBasis(c.inputs.x, c.inputs.knots);
			expect(ncols).toBe(c.outputs.ncols);
			const failures = [];
			for (let i = 0; i < c.inputs.x.length; i++) {
				for (let j = 0; j < ncols; j++) {
					const b = near(basis[i * ncols + j], c.outputs.B[i][j], 1e-9, `B[${i}][${j}]`);
					const d = near(derivative[i * ncols + j], c.outputs.dB[i][j], 1e-9, `dB[${i}][${j}]`);
					if (b) failures.push(b);
					if (d) failures.push(d);
				}
			}
			expect(failures, failures.slice(0, 5).join('\n')).toEqual([]);
		});
	}

	it('fewer than three knots is refused, as the reference raises', () => {
		expect(() => computeRcsBasis([1, 2, 3], [1, 2])).toThrow(RangeError);
	});

	it('every non-linear column and derivative is identically zero at or below the first knot', () => {
		const knots = [1983.5, 1992.5, 1995.5];
		const { basis, derivative } = computeRcsBasis([1900, 1983.5, 1983.49999], knots);
		for (let i = 0; i < 3; i++) {
			expect(basis[i]).toBe(0);
			expect(derivative[i]).toBe(0);
		}
	});
});

describe('compute_fieller_mrca_interval (dating.py:839-887)', () => {
	for (const c of load('compute_fieller_mrca_interval')) {
		it(`${c.name}: ${c.outputs.status}`, () => {
			expect(c.tolerance).toBe('1e-9');
			const cov = Float64Array.from([
				c.inputs.cov_beta[0][0],
				c.inputs.cov_beta[0][1],
				c.inputs.cov_beta[1][0],
				c.inputs.cov_beta[1][1]
			]);
			const got = computeFiellerMrcaInterval(c.inputs.mu, c.inputs.d0, cov, c.inputs.t_ref, c.inputs.df, {
				alpha: c.inputs.alpha,
				minTime: c.inputs.min_time
			});
			expect(got.status).toBe(c.outputs.status);
			expect(near(got.ci[0], c.outputs.ci[0], 1e-9, 'ci[0]')).toBeNull();
			expect(near(got.ci[1], c.outputs.ci[1], 1e-9, 'ci[1]')).toBeNull();
			expect(near(got.g, c.outputs.g, 1e-9, 'g')).toBeNull();
		});
	}
});

/** Fields of an OLS record whose classes differ from the table's own. */
const OLS_FIELD_TOL = { p_value: 1e-15 };
const OLS_EXACT = new Set(['method', 'status', 'ci_method', 'n']);

describe('run_ols_dating (dating.py:1170-1296)', () => {
	for (const c of load('run_ols_dating')) {
		it(`${c.name}: ${c.outputs.status ?? 'raises'}`, () => {
			if (c.outputs.raises) {
				// The reference raises LinAlgError from la.inv; the port refuses rather than
				// returning an infinite rate. There is no record to compare here, deliberately.
				expect(c.tolerance).toBe('exact');
				expect(() => runOlsDating(c.inputs.times, c.inputs.dists, { ciMethod: c.inputs.ci_method })).toThrow(RangeError);
				return;
			}
			expect(c.tolerance).toBe('1e-9');
			const got = runOlsDating(c.inputs.times, c.inputs.dists, {
				tRef: c.inputs.t_ref,
				ciMethod: c.inputs.ci_method
			});
			const failures = [];
			for (const [k, want] of Object.entries(c.outputs)) {
				const tol = OLS_FIELD_TOL[k] ?? 1e-9;
				if (OLS_EXACT.has(k)) {
					if (got[k] !== want) failures.push(`${k}: expected ${want}, got ${got[k]}`);
				} else if (Array.isArray(want)) {
					want.forEach((v, i) => {
						const m = near(got[k][i], v, tol, `${k}[${i}]`);
						if (m) failures.push(m);
					});
				} else if (want === null) {
					if (got[k] !== null) failures.push(`${k}: expected null (DATING Q7), got ${got[k]}`);
				} else {
					const m = near(got[k], want, tol, k);
					if (m) failures.push(m);
				}
			}
			expect(failures, failures.slice(0, 8).join('\n')).toEqual([]);
			// The key set IS the download contract (dating.py:3168 exports everything but three arrays).
			for (const k of Object.keys(c.outputs)) expect(Object.hasOwn(got, k), `key ${k}`).toBe(true);
		});
	}

	it('fewer than three observations is refused, as the reference raises', () => {
		expect(() => runOlsDating([2000, 2001], [0.01, 0.02])).toThrow(RangeError);
	});

	it('the three unported ci_methods are refused rather than silently becoming Fieller', () => {
		const t = [2000, 2001, 2002, 2003, 2004, 2005];
		const d = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1];
		for (const m of ['poisson', 'residual-boot', 'wild', 'jackknife', 'jack', 'loocv', 'POISSON']) {
			expect(() => runOlsDating(t, d, { ciMethod: m }), m).toThrow(RangeError);
		}
		// Anything else does take the reference's own else-branch into Fieller.
		const odd = runOlsDating(t, d, { ciMethod: 'not-a-method' });
		expect(odd.ci_mrca).toEqual(odd.ci_fieller);
		expect(odd.ci_method).toBe('not-a-method');
	});

	it('the delta interval agrees with the one run_ols_dating embeds', () => {
		const t = [1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018];
		const d = t.map((x, i) => 0.001 * (x - 1950) + (i % 3) * 1e-4);
		const ols = runOlsDating(t, d, { ciMethod: 'delta' });
		const delta = computeDeltaMrcaInterval(ols.mu, ols.d0, ols.cov_beta, ols.t_ref, t.length - 2, {
			minTime: Math.min(...t)
		});
		expect(delta.ci).toEqual(ols.ci_delta);
		expect(delta.seMrca).toBe(ols.se_mrca);
		expect(ols.ci_mrca).toEqual(ols.ci_delta);
	});
});

const SPLINE_EXACT = new Set(['method', 'n', 'is_nonlinear_preferred']);

describe('run_restricted_spline_clock_dating (dating.py:1809-1968)', () => {
	for (const c of load('run_restricted_spline_clock_dating')) {
		it(`${c.name}: preferred=${c.outputs.is_nonlinear_preferred}`, () => {
			expect(c.tolerance).toBe('1e-5');
			const got = runRestrictedSplineClockDating(c.inputs.times, c.inputs.dists);
			const failures = [];
			for (const [k, want] of Object.entries(c.outputs)) {
				if (SPLINE_EXACT.has(k)) {
					if (got[k] !== want) failures.push(`${k}: expected ${want}, got ${got[k]}`);
				} else if (Array.isArray(want)) {
					want.forEach((v, i) => {
						const m = near(got[k][i], v, k === 'knots' ? 0 : 1e-5, `${k}[${i}]`);
						if (m) failures.push(m);
					});
				} else {
					const m = near(got[k], want, 1e-5, k);
					if (m) failures.push(m);
				}
			}
			expect(failures, failures.slice(0, 8).join('\n')).toEqual([]);
			// DATING Q1: the upstream bootstrap raises on every replicate, so all four intervals
			// ARE their point estimates. A port that "fixed" the typo fails here, deliberately.
			expect(got.ci_mrca).toEqual([got.t_mrca, got.t_mrca]);
			expect(got.ci_rate_ancestral).toEqual([got.rate_ancestral, got.rate_ancestral]);
			expect(got.ci_rate_recent).toEqual([got.rate_recent, got.rate_recent]);
			expect(got.ci_beta_2).toEqual([got.beta_2, got.beta_2]);
		});
	}

	it('fewer than five observations is refused, as the reference raises', () => {
		expect(() => runRestrictedSplineClockDating([1, 2, 3, 4], [1, 2, 3, 4])).toThrow(RangeError);
	});

	it('a non-zero nBoot is refused rather than guessed at (DATING Q1)', () => {
		const t = [2000, 2001, 2002, 2003, 2004, 2005, 2006];
		const d = t.map((x) => 0.001 * (x - 1990));
		expect(() => runRestrictedSplineClockDating(t, d, { nBoot: 500 })).toThrow(RangeError);
	});
});

// ------------------------------------------------------------------------------------------------
// 2. the acceptance run
// ------------------------------------------------------------------------------------------------

/**
 * The application-side steps the reference performs before and after the library's functions, spelt
 * out here because they are the app's (PLAN.md §5.5) and this suite still has to walk them to reach
 * the reference's published record. Each is four lines and each cites its line in dating.py.
 */
const alignmentCache = new Map();
function alignmentFor(fasta) {
	if (alignmentCache.has(fasta)) return alignmentCache.get(fasta);
	const raw = parseAlignmentSequences(readFileSync(join(EXAMPLES, fasta), 'utf8'));
	const seqs = raw instanceof Map ? raw : new Map(Object.entries(raw));
	// verify_coding_alignment's silent trim, dating.py:163-174 (DATING Q6).
	const rem = seqs.get([...seqs.keys()][0]).length % 3;
	const trimmed = new Map(rem ? [...seqs].map(([k, v]) => [k, v.slice(0, v.length - rem)]) : [...seqs]);
	// The app's '*' → '-' rule on the dating path, DATING Q3.
	const cleaned = new Map([...trimmed].map(([k, v]) => [k, v.split('*').join('-')]));
	const out = { raw: seqs, trimmed, cleaned, trimmedBy: rem };
	alignmentCache.set(fasta, out);
	return out;
}

/** dating.py:2481-2497, 2687-2709, 2714, 2841, 2999-3062, with every library call in between. */
function runDatingChain(fasta, rootTaxon) {
	const { cleaned } = alignmentFor(fasta);
	const dates = new Map([...cleaned.keys()].map((t) => [t, parseHeaderTimestamp(t, { archival1959: true })]));
	const dated = [...cleaned.keys()].filter((t) => !Number.isNaN(dates.get(t)) && t !== rootTaxon);
	const tf = computeTreeFreeDivergences(cleaned, dated, dates, { rootTaxon });
	const times = Float64Array.from(dated, (t) => dates.get(t));

	// The coverage holdout, dating.py:2698-2709: upper-case ACGT only, a hard 0.50 cut, and only
	// when at least three training rows survive.
	const nCols = cleaned.get(dated[0]).length;
	const isTrain = dated.map((t) => {
		const s = cleaned.get(t);
		let c = 0;
		for (let i = 0; i < s.length; i++) {
			const ch = s[i];
			if (ch === 'A' || ch === 'C' || ch === 'G' || ch === 'T') c++;
		}
		return c / nCols >= 0.5;
	});
	const nTrain = isTrain.filter(Boolean).length;
	const trainIdx =
		isTrain.some((v) => !v) && nTrain >= 3 ? isTrain.map((v, i) => (v ? i : -1)).filter((i) => i >= 0) : dated.map((_, i) => i);

	const tFit = Float64Array.from(trainIdx, (i) => times[i]);
	const dFit = Float64Array.from(trainIdx, (i) => tf.divergences[i]);
	const ols = runOlsDating(tFit, dFit, { ciMethod: 'fieller' });
	const spline = runRestrictedSplineClockDating(tFit, dFit);
	// dating.py:2943-2996's selection, at the one branch --method ols reaches.
	const active = spline.is_nonlinear_preferred ? spline : ols;
	const fp = clockFittedAndPredicted(active, times, tf.divergences);
	const residuals = Float64Array.from(times, (_, i) => tf.divergences[i] - fp.fitted[i]);
	const scale = residualScale(residuals, trainIdx);
	const rows = dated.map((t, i) => {
		const holdout = !isTrain[i];
		const z = residuals[i] / scale;
		return {
			taxon: t,
			sampling_date: times[i],
			root_divergence: tf.divergences[i],
			fitted_divergence: fp.fitted[i],
			predicted_date: fp.predicted[i],
			divergence_residual: residuals[i],
			temporal_residual: Number.isNaN(fp.predicted[i]) ? NaN : fp.predicted[i] - times[i],
			z_score: z,
			is_outlier: holdout ? false : Math.abs(z) >= 2.5,
			is_holdout: holdout
		};
	});
	const ensemble = precisionWeightedEnsemble([{ name: 'ols', t_mrca: ols.t_mrca, ci_mrca: ols.ci_mrca }], Math.min(...times));
	return { tf, dated, times, trainIdx, ols, spline, active, rows, ensemble, methods: fp.methods };
}

/**
 * The worst |Δ| against the reference, per example, MEASURED at this commit. They are the real
 * assertion: the tolerance classes above say what would be acceptable, these say what actually
 * happens, and the gap between the two is where a regression would otherwise hide. korber's spline
 * and date figures are DATING Q5's conditioning, H1N1's are the same effect on a 0.67-year span.
 */
const MEASURED = {
	'korber_env_gp160.fasta': { ols: 4.5e-10, spline: 1.2e-8, fitted: 1.5e-11, date: 4.6e-7, z: 2e-9 },
	'H1N1_2009_pandemic.fasta': { ols: 1e-15, spline: 7.4e-6, fitted: 1e-18, date: 1e-18, z: 1e-14 }
};

describe('the acceptance run: `hyphaeon dating -a <fa> --root-taxon CONSENSUS --no-tree --method ols`', () => {
	const cases = load('run_mrca_dating').filter((c) => c.inputs.star_to_gap);

	for (const c of cases) {
		const fasta = c.outputs.result.alignment;
		it(`${fasta} reproduces the reference's published record`, () => {
			const ref = c.outputs.result;
			const got = runDatingChain(fasta, 'CONSENSUS');

			// -- ingestion -------------------------------------------------------------------
			expect(got.tf.root_description).toBe(ref.root_description);
			expect(ref.distance_mode).toBe('tn93');
			expect(got.dated.map((t) => t)).toEqual(ref.taxa_summary.map((r) => r.taxon));
			expect(got.ols.n).toBe(ref.ols.n);
			expect(ref.taxa_count).toBe(got.dated.length);

			// -- divergences and dates: EXACT ------------------------------------------------
			ref.taxa_summary.forEach((r, i) => {
				expect(got.rows[i].root_divergence, `root_divergence ${r.taxon}`).toBe(r.root_divergence);
				expect(got.rows[i].sampling_date, `sampling_date ${r.taxon}`).toBe(r.sampling_date);
			});

			// -- the OLS block ---------------------------------------------------------------
			let worstOls = 0;
			const failures = [];
			for (const [k, want] of Object.entries(ref.ols)) {
				if (['residuals', 'fitted', 'times'].includes(k)) continue;
				const tol = OLS_FIELD_TOL[k] ?? 1e-9;
				if (OLS_EXACT.has(k)) {
					if (got.ols[k] !== want) failures.push(`ols.${k}: ${want} vs ${got.ols[k]}`);
				} else if (Array.isArray(want)) {
					want.forEach((v, i) => {
						const m = near(got.ols[k][i], v, tol, `ols.${k}[${i}]`);
						if (m) failures.push(m);
						worstOls = Math.max(worstOls, worstOf(got.ols[k][i], v));
					});
				} else if (want === null) {
					if (got.ols[k] !== null) failures.push(`ols.${k}: expected null`);
				} else {
					const m = near(got.ols[k], want, tol, `ols.${k}`);
					if (m) failures.push(m);
					if (k !== 'p_value') worstOls = Math.max(worstOls, worstOf(got.ols[k], want));
				}
			}

			// -- the spline block ------------------------------------------------------------
			let worstSpline = 0;
			for (const [k, want] of Object.entries(ref.spline)) {
				if (['fitted', 'residuals'].includes(k)) continue;
				if (SPLINE_EXACT.has(k)) {
					if (got.spline[k] !== want) failures.push(`spline.${k}: ${want} vs ${got.spline[k]}`);
				} else if (Array.isArray(want)) {
					want.forEach((v, i) => {
						const m = near(got.spline[k][i], v, k === 'knots' ? 0 : 1e-5, `spline.${k}[${i}]`);
						if (m) failures.push(m);
						worstSpline = Math.max(worstSpline, worstOf(got.spline[k][i], v));
					});
				} else {
					const m = near(got.spline[k], want, 1e-5, `spline.${k}`);
					if (m) failures.push(m);
					worstSpline = Math.max(worstSpline, worstOf(got.spline[k], want));
				}
			}

			// -- the selected model, and the per-taxon table ----------------------------------
			const activeName = ref.active_model;
			expect(activeName === 'spline').toBe(got.spline.is_nonlinear_preferred);
			expect(near(got.active.t_mrca, ref.t_mrca, 1e-5, 't_mrca (headline)')).toBeNull();

			let worstFitted = 0;
			let worstDate = 0;
			let worstZ = 0;
			ref.taxa_summary.forEach((r, i) => {
				const g = got.rows[i];
				for (const [k, tol] of [
					['fitted_divergence', 1e-9],
					['divergence_residual', 1e-9],
					['predicted_date', 1e-6],
					['temporal_residual', 1e-6],
					['z_score', 1e-8]
				]) {
					const m = near(g[k], r[k], tol, `${r.taxon}.${k}`);
					if (m) failures.push(m);
				}
				if (g.is_outlier !== r.is_outlier) failures.push(`${r.taxon}.is_outlier: ${r.is_outlier} vs ${g.is_outlier}`);
				if (g.is_holdout !== r.is_holdout) failures.push(`${r.taxon}.is_holdout: ${r.is_holdout} vs ${g.is_holdout}`);
				worstFitted = Math.max(worstFitted, worstOf(g.fitted_divergence, r.fitted_divergence));
				worstDate = Math.max(worstDate, worstOf(g.predicted_date, r.predicted_date));
				worstZ = Math.max(worstZ, worstOf(g.z_score, r.z_score));
			});

			// -- the ensemble -----------------------------------------------------------------
			expect(Object.keys(got.ensemble.weights)).toEqual(Object.keys(ref.ensemble.weights));
			expect(near(got.ensemble.t_mrca, ref.ensemble.t_mrca, 1e-9, 'ensemble.t_mrca')).toBeNull();
			expect(near(got.ensemble.ci[0], ref.ensemble.ci_mrca[0], 1e-8, 'ensemble.ci[0]')).toBeNull();
			expect(near(got.ensemble.ci[1], ref.ensemble.ci_mrca[1], 1e-8, 'ensemble.ci[1]')).toBeNull();

			expect(failures, failures.slice(0, 10).join('\n')).toEqual([]);

			// -- the bounds are not vacuous ----------------------------------------------------
			// Every worst deviation is asserted against the value MEASURED at this commit rather
			// than against the bound it is checked at, so a regression that stayed inside the class
			// still fails here. 5 % of headroom absorbs a change in Node's libm, nothing more.
			const m = MEASURED[fasta];
			expect(worstOls, `ols worst |Δ| (bound 1e-9, measured ${m.ols})`).toBeLessThanOrEqual(m.ols * 1.05);
			expect(worstSpline, `spline worst |Δ| (bound 1e-5, measured ${m.spline})`).toBeLessThanOrEqual(m.spline * 1.05);
			expect(worstFitted, `fitted worst |Δ| (bound 1e-9, measured ${m.fitted})`).toBeLessThanOrEqual(m.fitted * 1.05 + 1e-18);
			expect(worstDate, `predicted_date worst |Δ| (bound 1e-6, measured ${m.date})`).toBeLessThanOrEqual(m.date * 1.05 + 1e-18);
			expect(worstZ, `z_score worst |Δ| (bound 1e-8, measured ${m.z})`).toBeLessThanOrEqual(m.z * 1.05 + 1e-18);
		});
	}

	it('korber: the flagship facts a reader acts on', () => {
		const ref = load('run_mrca_dating').find((c) => c.name === 'korber_env_gp160').outputs.result;
		const got = runDatingChain('korber_env_gp160.fasta', 'CONSENSUS');
		// 143 sequences, 142 dated (CONSENSUS is the miss), 141 in the fit: the one reserved row is
		// the 1959 Léopoldville isolate, at 17.6 % ACGT coverage.
		expect(alignmentFor('korber_env_gp160.fasta').cleaned.size).toBe(143);
		expect(got.dated.length).toBe(142);
		expect(got.trainIdx.length).toBe(141);
		const held = got.rows.filter((r) => r.is_holdout).map((r) => r.taxon);
		expect(held).toEqual(['Z59ZR.ZHU']);
		// A holdout is NEVER flagged (dating.py:3049), and on this example nothing else is either.
		expect(got.rows.filter((r) => r.is_outlier)).toEqual([]);
		expect(ref.taxa_summary.filter((r) => r.is_outlier)).toEqual([]);
		// The spline wins the selection test and has no interval to show for it (DATING Q1 + Q2),
		// while the ensemble weights OLS at 1.0 — the two numbers the same JSON presents.
		expect(got.spline.is_nonlinear_preferred).toBe(true);
		expect(got.spline.ci_mrca[0]).toBe(got.spline.ci_mrca[1]);
		expect(got.ensemble.weights).toEqual({ ols: 1 });
		expect(got.ols.t_mrca).toBeCloseTo(1893.911, 3);
		expect(got.spline.t_mrca).toBeCloseTo(1938.775, 3);
		// The per-taxon date inversion silently switches model for 12 of 142 rows (DATING Q2's
		// sibling defect): the bracket is refused and the ancestral LINEAR inverse stands in.
		const counts = got.methods.reduce((a, m) => ({ ...a, [m]: (a[m] ?? 0) + 1 }), {});
		expect(counts).toEqual({ spline: 118, linear_arm: 12, linear_fallback: 12 });
	});

	it('H1N1: the other branch of every rule korber exercises', () => {
		const got = runDatingChain('H1N1_2009_pandemic.fasta', 'CONSENSUS');
		// No root sequence in the file, so --root-taxon CONSENSUS falls through to case 4.
		expect(got.tf.root_description).toBe('time_decay_consensus_root (γ=3.0030)');
		expect(got.tf.case).toBe(4);
		// 100 sequences, 95 dated (fixtures/dates Q7 loses five), no holdout, one outlier.
		expect(got.dated.length).toBe(95);
		expect(got.trainIdx.length).toBe(95);
		expect(got.rows.filter((r) => r.is_holdout)).toEqual([]);
		expect(got.rows.filter((r) => r.is_outlier).map((r) => r.taxon)).toHaveLength(1);
		// The spline is REJECTED here, so the OLS record is the active model and every predicted
		// date comes from one division rather than from brentq.
		expect(got.spline.is_nonlinear_preferred).toBe(false);
		expect(got.active).toBe(got.ols);
		expect(new Set(got.methods)).toEqual(new Set(['ols']));
		// The L mod 3 trim fired here and not on korber (DATING Q6).
		expect(alignmentFor('H1N1_2009_pandemic.fasta').trimmedBy).toBe(2);
		expect(alignmentFor('korber_env_gp160.fasta').trimmedBy).toBe(0);
	});

	it("the '*' convention is load-bearing: leaving it alone moves the answer, measurably", () => {
		// fixtures/dating/run_mrca_dating.json carries BOTH korber runs so the size of DATING Q3 is
		// in the record. The package branch scores '*' as an unknown; the binary branch, which the
		// published numbers came from, rewrites it to a gap.
		const withGap = load('run_mrca_dating').find((c) => c.name === 'korber_env_gp160').outputs.result;
		const asIs = load('run_mrca_dating').find((c) => c.name === 'korber_env_gp160_star_unmodified').outputs.result;
		expect(Math.abs(withGap.ols.t_mrca - asIs.ols.t_mrca)).toBeGreaterThan(0.09);
		expect(Math.abs(withGap.ols.ci_fieller[0] - asIs.ols.ci_fieller[0])).toBeGreaterThan(0.28);

		// And the port reproduces the OTHER convention too, when handed the other sequences.
		const { trimmed } = alignmentFor('korber_env_gp160.fasta');
		const dates = new Map([...trimmed.keys()].map((t) => [t, parseHeaderTimestamp(t, { archival1959: true })]));
		const dated = [...trimmed.keys()].filter((t) => !Number.isNaN(dates.get(t)) && t !== 'CONSENSUS');
		const tf = computeTreeFreeDivergences(trimmed, dated, dates, { rootTaxon: 'CONSENSUS' });
		asIs.taxa_summary.forEach((r, i) => expect(tf.divergences[i], `${r.taxon} unmodified`).toBe(r.root_divergence));
		const differing = asIs.taxa_summary.filter((r, i) => r.root_divergence !== withGap.taxa_summary[i].root_divergence);
		expect(differing.length, 'sequences carrying at least one asterisk').toBe(18);
	});
});

describe('precisionWeightedEnsemble (the algebra of dating.py:2895-2921)', () => {
	it('one candidate still re-symmetrises its own interval (B10), which is why it is not the answer', () => {
		const ols = { name: 'ols', t_mrca: 1893.91095511759, ci_mrca: [1850.9002455209902, 1916.7928463014516] };
		const got = precisionWeightedEnsemble([ols], 1983.5);
		expect(got.weights).toEqual({ ols: 1 });
		expect(got.t_mrca).toBe(ols.t_mrca);
		// A skewed Fieller interval read as a symmetric Gaussian one comes back a DIFFERENT interval.
		expect(got.ci[0]).not.toBe(ols.ci_mrca[0]);
		expect(got.ci[1]).not.toBe(ols.ci_mrca[1]);
		expect(got.ci[0]).toBeCloseTo(1860.9646547, 6);
		expect(got.ci[1]).toBeCloseTo(1926.8572555, 6);
	});

	it('a half-infinite or zero-width interval is dropped, which is how a selected spline is ignored', () => {
		const spline = { name: 'spline', t_mrca: 1938.77, ci_mrca: [1938.77, 1938.77] };
		const unbounded = { name: 'ols', t_mrca: 1893.9, ci_mrca: [-Infinity, 1983.5] };
		expect(precisionWeightedEnsemble([spline], 1983.5)).toBeNull();
		expect(precisionWeightedEnsemble([unbounded], 1983.5)).toBeNull();
		expect(precisionWeightedEnsemble([{ name: 'x', t_mrca: 1, ci_mrca: null }], 10)).toBeNull();
	});

	it('two candidates: inverse-squared-width weights and the upper bound clamped to the earliest sample', () => {
		const a = { name: 'ols', t_mrca: 1900, ci_mrca: [1880, 1920] };
		const b = { name: 'pgls', t_mrca: 1910, ci_mrca: [1900, 1920] };
		const got = precisionWeightedEnsemble([a, b], 1930);
		expect(got.weights.ols + got.weights.pgls).toBeCloseTo(1, 15);
		expect(got.weights.pgls / got.weights.ols).toBeCloseTo(4, 12); // (40/20)² = 4
		expect(got.t_mrca).toBeCloseTo(0.2 * 1900 + 0.8 * 1910, 10);
		expect(got.ci[1]).toBeLessThanOrEqual(1930);
	});
});

describe('residualScale (dating.py:3041-3042)', () => {
	it('is the population standard deviation of the TRAINING residuals, ddof = 0', () => {
		const res = [1, -1, 2, -2, 100];
		const got = residualScale(res, [0, 1, 2, 3]);
		expect(got).toBeCloseTo(Math.sqrt((1 + 1 + 4 + 4) / 4), 15);
	});

	it('falls back to the whole set below three training rows, and then to a flat 1.0', () => {
		expect(residualScale([1, -1, 2, -2], [0, 1])).toBeCloseTo(Math.sqrt(2.5), 15);
		expect(residualScale([0, 0, 0, 0], [0, 1, 2, 3])).toBe(1.0);
	});
});
