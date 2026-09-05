/**
 * WHY THIS FILE EXISTS
 *
 * Replays every phenotype reference against src/phenotype.js, at three levels:
 *
 *   fixtures/phenotype/resolve_phenotype_vector.json   15 cases, tolerance `exact` (the one
 *       continuous case at 1e-9): four presets, six inline-foreground spellings (list, regex,
 *       pipe list, glob, `.*` patterns) and five CSV/TSV cases over `fixtures/phenotype/inputs/`.
 *       `y`, the `meta` dict key for key, and the fixture's `foreground_taxa`.
 *   test/data/phenotype/python_quirks.json            the library behaviour underneath it, taken
 *       from the libraries the reference actually calls: CPython `fnmatch` (globs, classes,
 *       negated classes, an unterminated `[`, metacharacters fnmatch treats as literals),
 *       `repr(list[str])`, `str(float)` across the bands where Python and JavaScript disagree,
 *       pandas' per-column dtype inference and the `str(val)` it implies, twenty more
 *       `resolve_phenotype_vector` calls and the four error texts.
 *   test/data/phenotype/run_phenotype_association_fake_model.json   the WHOLE DRIVER: five runs of
 *       the real `run_phenotype_association` against a deterministic fake model on a 10 x 14
 *       synthetic alignment, with the model's two outputs recorded so the port replays exactly
 *       what the reference saw. Every site record, co-selection pair, sector and gene-level field
 *       is compared. See that script's header for what each case exercises.
 *   fixtures/e2e/phenotype_RHO_marine_n_permutations_0.json         the real CLI on RHO.
 *
 * WHAT THE e2e FIXTURE CAN AND CANNOT PIN, and what the app runtime must therefore replay.
 * The fixture records `inputs.argv` and the CLI's JSON — 145 site records, 128 co-selection pairs
 * and one sector — and NOTHING from the model: the 349 x 655 attention matrix and the 349 LRTs
 * that produced them are not in the file, and neither is the alignment's token matrix. So this
 * file checks everything that is a function of the recorded columns, which is most of
 * phenotype.py:498-621:
 *
 *   - the record's key order and the report's shape (phenotype.py:624-646)
 *   - `phenotype_meta`, recomputed from `argv`'s `-fg` string over the 655 taxa that
 *     `examples/RHO.fasta` yields after duplicate pruning — the pillar's whole trait resolution
 *   - per site: `score`, `p_value` (ACAT of `p_lrt` and `p_assoc`), `q_value` (BH over the sorted
 *     column), `p_assoc == p_assoc_parametric` and `p_assoc_perm == null` at `--n-permutations 0`,
 *     the score-descending order, and the degrees of freedom `p_assoc_parametric` implies
 *   - the gene block: `max_assoc`, `p_evd_length_adjusted`, `score_track_a/b`,
 *     `dual_track_composite`, `compact_pars_signature`, `significant_sites_count`
 *   - per pair: `cesi`, `p_value` at df = N - 2, `q_value`, the CESI-descending order, and
 *     agreement between each pair's `ref_u`/`lrt_u` and the site record for that site
 *   - per sector: the signature format, ascending site lists, `size`, `isotropic_baseline`
 *
 * What it CANNOT check, and what the app's runtime must therefore replay through
 * `runPhenotypeAssociation` to close the loop: `association_rho`, `attribution_norm`,
 * `fg_mean_attn`, `bg_mean_attn`, `foreground_freq_pct`, `background_freq_pct`, `spectral_energy`,
 * `norm_spectral_ratio`, `similarity`, `shared_branches` and `spectral_coherence` are all
 * functions of the attention matrix. The fake-model case above is the substitute at the unit
 * level; at the integration level the app must run the same ONNX graph on `examples/RHO.fasta`
 * with `--mds-sign canonical` and compare those columns at the 1e-5 class (PLAN.md §5.4), which is
 * the four-way parity job of Phase 3, not something a JSON replay can do.
 *
 * TOLERANCE. Numeric fields of a site, pair or sector record are compared at the fixtures' 1e-6
 * class and strings, counts, site lists and orderings exactly. 1e-6 rather than 1e-9 for all of
 * them, not only the ones the reference stores as float32, because every numeric field in a site
 * record descends from `attribution_norm` — `np.linalg.norm(a_s)` through BLAS sdot, whose
 * accumulation order JavaScript cannot reproduce (src/phenotype.js header carries the
 * measurement). The relations BETWEEN the recorded columns, which involve no norm, are checked
 * against the e2e fixture at 1e-12.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
	PRESETS,
	PHENOTYPE_THRESHOLDS,
	resolvePhenotypeVector,
	runPhenotypeAssociation,
	parsePhenotypeTable,
	pyFnmatch,
	pyFloatStr,
	pyReprString,
	pyReprStringList
} from '../src/phenotype.js';
import { cauchyCombinationP, benjaminiHochberg } from '../src/stats.js';
import { tSf, normSf } from '../src/numeric/special.js';
import { readNewick } from '../src/preprocess/tree.js';
import { parseAlignmentSequences } from '../src/preprocess/parse.js';
import { pruneIdenticalSequences } from '../src/preprocess/downsample.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..', '..');
const FIXTURES = join(ROOT, 'fixtures', 'phenotype');
const resolveFixture = JSON.parse(readFileSync(join(FIXTURES, 'resolve_phenotype_vector.json'), 'utf8'));
const e2e = JSON.parse(readFileSync(join(ROOT, 'fixtures', 'e2e', 'phenotype_RHO_marine_n_permutations_0.json'), 'utf8'))[0];
const local = JSON.parse(readFileSync(join(HERE, 'data', 'phenotype', 'python_quirks.json'), 'utf8'));
const fake = JSON.parse(readFileSync(join(HERE, 'data', 'phenotype', 'run_phenotype_association_fake_model.json'), 'utf8'));

const readInput = (/** @type {string} */ name) => readFileSync(join(FIXTURES, 'inputs', name), 'utf8');

/** |a - b| <= tol * max(1, |b|) — the relative form the fixtures' classes use. */
function close(a, b, tol) {
	return Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
}

/**
 * The class for a numeric field of a site, pair or sector record: 1e-6, the fixtures' float32
 * class. It applies to EVERY such field, not only the ones the reference stores as float32,
 * because they all descend from one: `attribution_norm` is `np.linalg.norm(a_s)` through BLAS
 * sdot, whose accumulation order JavaScript cannot reproduce (src/phenotype.js header, with the
 * measurement: one float32 ulp, 3.7e-9 absolute on the fake-model matrix), and
 * `association_rho` divides by it, `p_assoc_parametric` is a t-tail of that, `p_value` an ACAT of
 * that, `q_value` a BH of that, and `score` a product with it. The relations BETWEEN the recorded
 * columns are checked to 1e-12 against the e2e fixture further down, where no norm is involved.
 */
const RECORD_TOL = 1e-6;

// =================================================================================================
// PRESETS
// =================================================================================================

describe('PRESETS — phenotype.py:48-111, verbatim', () => {
	it('has the reference keys in the reference order', () => {
		expect(Object.keys(PRESETS)).toEqual(local.preset_keys);
	});

	it('every title, description and species list matches the reference', () => {
		for (const [key, ref] of Object.entries(local.presets)) {
			const got = PRESETS[key];
			expect(got.title).toBe(ref.title);
			expect(got.description).toBe(ref.description);
			expect(got.foreground).toEqual(ref.foreground);
			if (ref.controls) expect(got.controls).toEqual(ref.controls);
			else expect(got.controls).toBeUndefined();
		}
	});

	it('is frozen, because a mutated preset would silently change every report', () => {
		expect(Object.isFrozen(PRESETS)).toBe(true);
	});
});

// =================================================================================================
// The CPython / pandas behaviour resolve_phenotype_vector rests on
// =================================================================================================

describe('CPython string behaviour (test/data/phenotype/gen.py)', () => {
	it('fnmatch.fnmatch over globs, classes and metacharacters', () => {
		for (const c of local.fnmatch) {
			expect({ ...c, match: pyFnmatch(c.name, c.pattern) }).toEqual(c);
		}
	});

	it('repr(list[str]), including the quote switch for an apostrophe', () => {
		for (const c of local.repr_list) expect(pyReprStringList(c.items)).toBe(c.repr);
		expect(pyReprString("it's")).toBe('"it\'s"');
		expect(pyReprString('plain')).toBe("'plain'");
	});

	it('str(float) across the bands where JavaScript disagrees with Python', () => {
		for (const c of local.float_str) expect(pyFloatStr(c.value)).toBe(c.str);
		// The two bands that motivate the function: JS would print '0.00001' and '1e+21'.
		expect(pyFloatStr(1e-5)).toBe('1e-05');
		expect(pyFloatStr(62)).toBe('62.0');
	});
});

describe('parsePhenotypeTable — pandas read_csv dtype inference', () => {
	for (const c of local.read_csv) {
		it(c.name, () => {
			const t = parsePhenotypeTable(c.text, c.sep);
			expect(t.columns).toEqual(c.columns);
			expect(t.rows.length).toBe(c.rows.length);
			for (let r = 0; r < c.rows.length; r++) {
				for (let k = 0; k < c.columns.length; k++) {
					const ref = c.rows[r][c.columns[k]];
					const got = t.rows[r][k];
					if (ref.isna) {
						expect(got).toBeNull();
					} else {
						const asStr = typeof got === 'number' ? (t.floatColumns[k] ? pyFloatStr(got) : String(got)) : String(got);
						expect(asStr).toBe(ref.str);
					}
				}
			}
		});
	}

	it('an integer column renders "1" and a float column "1.0" — the difference decides the trait', () => {
		expect(parsePhenotypeTable('s,t\na,1\n', ',').floatColumns[1]).toBe(false);
		expect(parsePhenotypeTable('s,t\na,1.0\n', ',').floatColumns[1]).toBe(true);
		// An NA forces float64 in pandas (int64 has no NA), so "1" becomes "1.0".
		expect(parsePhenotypeTable('s,t\na,1\nb,\n', ',').floatColumns[1]).toBe(true);
	});
});

// =================================================================================================
// resolve_phenotype_vector
// =================================================================================================

/** The fixture's Python keyword inputs as the port's options object. */
function resolveOptionsFrom(inp) {
	const opts = {
		preset: inp.preset,
		foreground: inp.foreground,
		background: inp.background,
		traitCol: inp.trait_col,
		speciesCol: inp.species_col,
		continuous: inp.continuous === true
	};
	if (inp.phenotype_file) {
		opts.phenotypeCsv = readInput(inp.phenotype_file);
		opts.phenotypeFile = inp.phenotype_file;
	}
	return opts;
}

describe('resolvePhenotypeVector — fixtures/phenotype/resolve_phenotype_vector.json', () => {
	it('replays all fifteen cases', () => {
		expect(resolveFixture.length).toBe(15);
	});

	for (const c of resolveFixture) {
		const tol = c.tolerance === 'exact' ? 0 : 1e-9;
		it(`${c.name} (${c.tolerance})`, () => {
			const got = resolvePhenotypeVector(c.inputs.taxa, resolveOptionsFrom(c.inputs));
			expect(got.y.length).toBe(c.outputs.y.length);
			for (let i = 0; i < got.y.length; i++) {
				if (tol === 0) expect(got.y[i]).toBe(c.outputs.y[i]);
				else expect(Math.abs(got.y[i] - c.outputs.y[i])).toBeLessThanOrEqual(tol);
			}
			expect(got.meta).toEqual(c.outputs.meta);
			// The library naming of the same four values.
			expect(got.mode).toBe(c.outputs.meta.mode);
			expect(got.fgCount).toBe(c.outputs.meta.foreground_count);
			expect(got.bgCount).toBe(c.outputs.meta.background_count);
			expect(got.description).toBe(c.outputs.meta.description);
			// The fixture's derived list: which taxa ended up positive, in taxon order.
			const fgTaxa = c.inputs.taxa.filter((_, i) => got.y[i] > 0);
			expect(fgTaxa).toEqual(c.outputs.foreground_taxa);
		});
	}

	it('foreground_glob_Smc6 is matched as a REGEX, not a glob (the quirk the fixture pins)', () => {
		// `pan*` is `pa` + zero-or-more `n`, so it matches papAnu; `mac*` matches rheMac. A glob
		// reading of the same patterns would give panPan/panTro4 and macFas/macMul only.
		const c = resolveFixture.find((x) => x.name === 'foreground_glob_Smc6');
		expect(c.outputs.foreground_taxa).toContain('papAnu');
		expect(c.outputs.foreground_taxa).toContain('rheMac');
		const got = resolvePhenotypeVector(c.inputs.taxa, resolveOptionsFrom(c.inputs));
		expect(c.inputs.taxa.filter((_, i) => got.y[i] > 0)).toEqual(c.outputs.foreground_taxa);
	});
});

describe('resolvePhenotypeVector — edge cases (test/data/phenotype/gen.py)', () => {
	for (const c of local.resolve) {
		it(c.name, () => {
			const opts = {
				preset: c.kwargs.preset ?? null,
				foreground: c.kwargs.foreground ?? null,
				background: c.kwargs.background ?? null,
				traitCol: c.kwargs.trait_col ?? null,
				speciesCol: c.kwargs.species_col ?? null,
				continuous: c.kwargs.continuous === true
			};
			if (c.phenotype_csv !== undefined) {
				opts.phenotypeCsv = c.phenotype_csv;
				opts.phenotypeFile = c.phenotype_file;
			}
			const got = resolvePhenotypeVector(c.taxa, opts);
			for (let i = 0; i < c.y.length; i++) expect(got.y[i]).toBeCloseTo(c.y[i], 12);
			expect(got.meta).toEqual(c.meta);
		});
	}

	for (const c of local.resolve_errors) {
		it(`throws: ${c.name}`, () => {
			expect(() =>
				resolvePhenotypeVector(c.taxa, { preset: c.kwargs.preset ?? null, foreground: c.kwargs.foreground ?? null })
			).toThrow(c.error.message);
		});
	}

	it('background is accepted and ignored', () => {
		const taxa = ['turTru', 'panTro4'];
		const withBg = resolvePhenotypeVector(taxa, { foreground: 'turTru', background: 'panTro4' });
		const without = resolvePhenotypeVector(taxa, { foreground: 'turTru' });
		expect(Array.from(withBg.y)).toEqual(Array.from(without.y));
		expect(withBg.meta).toEqual(without.meta);
	});
});

// =================================================================================================
// run_phenotype_association against the fake model
// =================================================================================================

describe('runPhenotypeAssociation — the whole driver against a deterministic fake model', () => {
	const L = fake.L;
	const N = fake.N;
	const loaded = { a: Int32Array.from(fake.a.flat()), c: Int32Array.from(fake.c.flat()), L, N, taxa: fake.taxa };
	const attention = Float32Array.from(fake.model_outputs.attention.flat());
	const lrt = Float64Array.from(fake.model_outputs.lrt_raw);

	/** @param {any} c */
	async function runCase(c) {
		const i = c.inputs;
		const input = {
			loaded,
			taxa: fake.taxa,
			attention,
			lrt,
			tree: readNewick(fake.newick),
			alignment: 'synthetic.fasta',
			treePath: 'synthetic.nwk'
		};
		input.phenotype = i.phenotype_file
			? { phenotypeCsv: fake.trait_csv, phenotypeFile: fake.trait_csv_name, continuous: true }
			: { foreground: i.foreground };
		return runPhenotypeAssociation(input, null, {
			permulations: i.permulations,
			nPermutations: i.n_permutations,
			seed: i.seed,
			alpha: i.alpha,
			minTaxaPerSite: i.min_taxa_per_site,
			continuous: i.continuous === true
		});
	}

	/** Compare one record against the reference, choosing the tolerance per field name. */
	function expectRecord(got, want, where) {
		expect(Object.keys(got), `${where}: key order`).toEqual(Object.keys(want));
		for (const [k, ref] of Object.entries(want)) {
			const g = got[k];
			if (ref === null || typeof ref === 'string' || typeof ref === 'boolean') expect(g, `${where}.${k}`).toBe(ref);
			else if (Array.isArray(ref)) expect(g, `${where}.${k}`).toEqual(ref);
			else if (Number.isInteger(ref) && Math.abs(ref) < 1e9 && !String(k).startsWith('p_')) expect(g, `${where}.${k}`).toBe(ref);
			else expect(close(g, ref, RECORD_TOL), `${where}.${k}: ${g} vs ${ref}`).toBe(true);
		}
	}

	for (const c of fake.cases.filter((x) => x.name !== 'permulations_200')) {
		it(`${c.name}: every field of the report`, async () => {
			const got = await runCase(c);
			const want = c.outputs;
			expect(Object.keys(got)).toEqual(Object.keys(want));
			for (const k of ['alignment', 'tree', 'taxa_count', 'codon_count', 'compact_pars_signature',
				'permulations_count', 'gene_p_value_perm', 'significant_sites_count', 'coselection_pairs_count',
				'trait_sectors_count']) {
				expect(got[k], k).toEqual(want[k]);
			}
			expect(got.phenotype_meta).toEqual(want.phenotype_meta);
			for (const k of ['spectral_energy', 'norm_spectral_ratio', 'max_assoc', 'p_evd_length_adjusted',
				'score_track_a', 'score_track_b', 'dual_track_composite']) {
				expect(close(got[k], want[k], 1e-6), `${k}: ${got[k]} vs ${want[k]}`).toBe(true);
			}
			expect(got.sites.length).toBe(want.sites.length);
			for (let s = 0; s < want.sites.length; s++) expectRecord(got.sites[s], want.sites[s], `sites[${s}]`);
			expect(got.coselection_pairs.length).toBe(want.coselection_pairs.length);
			for (let s = 0; s < want.coselection_pairs.length; s++) {
				expectRecord(got.coselection_pairs[s], want.coselection_pairs[s], `pairs[${s}]`);
			}
			expect(got.trait_sectors.length).toBe(want.trait_sectors.length);
			for (let s = 0; s < want.trait_sectors.length; s++) {
				expectRecord(got.trait_sectors[s], want.trait_sectors[s], `sectors[${s}]`);
			}
		});
	}

	it('permulations_200: the deterministic half matches and the Monte Carlo half has the right shape', async () => {
		const c = fake.cases.find((x) => x.name === 'permulations_200');
		const got = await runCase(c);
		const want = c.outputs;
		const P = c.inputs.permulations;
		expect(got.permulations_count).toBe(P);
		expect(want.permulations_count).toBe(P);
		expect(got.sites.length).toBe(want.sites.length);
		// The draws differ, so only the fields that do not depend on them are compared.
		const byS = new Map(want.sites.map((x) => [x.site, x]));
		for (const g of got.sites) {
			const w = byS.get(g.site);
			expect(w, `site ${g.site} missing from the reference`).toBeTruthy();
			for (const k of ['ref_aa', 'derived_aa', 'foreground_freq_pct', 'background_freq_pct']) expect(g[k]).toEqual(w[k]);
			for (const k of ['hyphaeon_lrt', 'p_lrt', 'attribution_norm', 'fg_mean_attn', 'bg_mean_attn']) {
				expect(close(g[k], w[k], 1e-6), `site ${g.site}.${k}`).toBe(true);
			}
			expect(close(g.association_rho, w.association_rho, 1e-6)).toBe(true);
			expect(close(g.p_assoc_parametric, w.p_assoc_parametric, 1e-6)).toBe(true);
			expect(close(g.score, w.score, 1e-6)).toBe(true);
			// The permulation p is (1 + #{null >= rho}) / (1 + P): a multiple of 1/(P+1) in
			// [1/(P+1), 1], and it IS p_assoc, which then feeds ACAT.
			expect(g.p_assoc_perm).not.toBeNull();
			expect(g.p_assoc).toBe(g.p_assoc_perm);
			const k = g.p_assoc_perm * (P + 1);
			expect(Math.abs(k - Math.round(k))).toBeLessThan(1e-9);
			expect(g.p_assoc_perm).toBeGreaterThanOrEqual(1 / (P + 1));
			expect(g.p_assoc_perm).toBeLessThanOrEqual(1);
			expect(close(g.p_value, cauchyCombinationP(Float64Array.of(g.p_lrt, g.p_assoc)), 1e-12)).toBe(true);
		}
		expect(got.gene_p_value_perm).not.toBeNull();
		const gk = got.gene_p_value_perm * (P + 1);
		expect(Math.abs(gk - Math.round(gk))).toBeLessThan(1e-9);
	});

	it('tree-free: with no tree the permulation branch is skipped, as phenotype.py:426 does', async () => {
		const c = fake.cases.find((x) => x.name === 'permulations_200');
		const got = await runPhenotypeAssociation(
			{ loaded, taxa: fake.taxa, attention, lrt, tree: null, phenotype: { foreground: c.inputs.foreground } },
			null,
			{ permulations: 200, nPermutations: 0, seed: 42 }
		);
		expect(got.permulations_count).toBe(0);
		expect(got.gene_p_value_perm).toBeNull();
		for (const s of got.sites) {
			expect(s.p_assoc_perm).toBeNull();
			expect(s.p_assoc).toBe(s.p_assoc_parametric);
		}
		// and it is then identical to the permulations = 0 run
		const base = await runCase(fake.cases.find((x) => x.name === 'base'));
		expect(got.sites.map((x) => x.site)).toEqual(base.sites.map((x) => x.site));
	});

	it('accepts a predict callback in place of recorded outputs', async () => {
		const batches = [];
		const predict = async (c, a, meta) => {
			batches.push(meta.batch);
			const l = new Float32Array(meta.batch);
			const at = new Float32Array(meta.batch * meta.N);
			for (let b = 0; b < meta.batch; b++) {
				const s = meta.siteIndices[b];
				l[b] = lrt[s];
				at.set(attention.subarray(s * N, (s + 1) * N), b * N);
			}
			return { lrt: l, attention: at };
		};
		const c = fake.cases.find((x) => x.name === 'base');
		const viaPredict = await runPhenotypeAssociation(
			{ loaded, taxa: fake.taxa, tree: readNewick(fake.newick), phenotype: { foreground: c.inputs.foreground },
				alignment: 'synthetic.fasta', treePath: 'synthetic.nwk' },
			predict,
			{ permulations: 0, nPermutations: 0, seed: 42, batchSize: 4 }
		);
		expect(batches).toEqual([4, 4, 4, 2]);
		expect(viaPredict.sites).toEqual((await runCase(c)).sites);
	});

	it('raises the reference\'s message when fewer than two foreground taxa match', async () => {
		await expect(
			runPhenotypeAssociation(
				{ loaded, taxa: fake.taxa, attention, lrt, phenotype: { foreground: 'fg_eta' } },
				null,
				{ permulations: 0, nPermutations: 0 }
			)
		).rejects.toThrow(`Insufficient foreground taxa (1) matching criteria among ${N} taxa.`);
	});

	it('the same check is skipped in continuous mode', async () => {
		const y = new Float64Array(N);
		y[N - 1] = 1;
		const got = await runPhenotypeAssociation(
			{ loaded, taxa: fake.taxa, attention, lrt, y },
			null,
			{ permulations: 0, nPermutations: 0, continuous: true }
		);
		expect(got.phenotype_meta.mode).toBe('continuous');
		expect(got.phenotype_meta.foreground_count).toBe(0);
		expect(got.sites.length).toBeGreaterThan(0);
	});
});

// =================================================================================================
// fixtures/e2e/phenotype_RHO_marine_n_permutations_0.json
// =================================================================================================

describe('fixtures/e2e/phenotype_RHO_marine_n_permutations_0.json', () => {
	const out = e2e.outputs;
	const argv = e2e.inputs.argv;
	const N = out.taxa_count;
	const L = out.codon_count;

	it('the report has the reference key order (phenotype.py:624-646)', () => {
		expect(Object.keys(out)).toEqual([
			'alignment', 'tree', 'taxa_count', 'codon_count', 'phenotype_meta', 'spectral_energy',
			'norm_spectral_ratio', 'max_assoc', 'p_evd_length_adjusted', 'score_track_a', 'score_track_b',
			'dual_track_composite', 'compact_pars_signature', 'permulations_count', 'gene_p_value_perm',
			'significant_sites_count', 'coselection_pairs_count', 'trait_sectors_count',
			'coselection_pairs', 'trait_sectors', 'sites'
		]);
		// and the port produces exactly those keys, in that order (checked against the fake-model
		// run, which is a real runPhenotypeAssociation return value)
		expect(Object.keys(fake.cases[0].outputs)).toEqual(Object.keys(out));
	});

	it('site records carry the reference keys in order (phenotype.py:499-515 plus q_value)', () => {
		for (const s of out.sites) {
			expect(Object.keys(s)).toEqual([
				'site', 'ref_aa', 'derived_aa', 'hyphaeon_lrt', 'p_lrt', 'attribution_norm', 'fg_mean_attn',
				'bg_mean_attn', 'association_rho', 'p_value', 'p_assoc', 'p_assoc_parametric', 'p_assoc_perm',
				'score', 'foreground_freq_pct', 'background_freq_pct', 'q_value'
			]);
		}
	});

	it('the trait vector is reproduced from argv over the pruned RHO taxa', () => {
		// dataset.load_alignment_and_tree prunes identical sequences before the pillar sees them:
		// 710 records in examples/RHO.fasta, 655 unique haplotypes, which is out.taxa_count.
		const seqs = parseAlignmentSequences(readFileSync(join(ROOT, 'examples', 'RHO.fasta'), 'utf8'));
		expect(seqs.size).toBe(710);
		const { uniqueTaxa } = pruneIdenticalSequences(seqs, [...seqs.keys()]);
		expect(uniqueTaxa.length).toBe(N);

		const fg = argv[argv.indexOf('-fg') + 1];
		const got = resolvePhenotypeVector(uniqueTaxa, { foreground: fg });
		expect(got.meta).toEqual(out.phenotype_meta);
		expect(got.fgCount).toBe(11);
		expect(got.bgCount).toBe(644);
		expect(got.description).toBe(`User-specified foreground patterns: ${pyReprStringList(fg.split(','))}`);
	});

	it('sites are unique, in range, and sorted by score descending', () => {
		const seen = new Set();
		for (const s of out.sites) {
			expect(s.site).toBeGreaterThanOrEqual(1);
			expect(s.site).toBeLessThanOrEqual(L);
			expect(seen.has(s.site)).toBe(false);
			seen.add(s.site);
		}
		for (let i = 1; i < out.sites.length; i++) expect(out.sites[i - 1].score).toBeGreaterThanOrEqual(out.sites[i].score);
	});

	it('score = sqrt(max(0, lrt)) * max(0, rho) at every site', () => {
		for (const s of out.sites) {
			const want = Math.sqrt(Math.max(0, s.hyphaeon_lrt)) * Math.max(0, s.association_rho);
			expect(close(s.score, want, 1e-12), `site ${s.site}`).toBe(true);
		}
	});

	it('p_value is the Cauchy combination of p_lrt and p_assoc', () => {
		for (const s of out.sites) {
			const want = cauchyCombinationP(Float64Array.of(s.p_lrt, s.p_assoc));
			expect(close(s.p_value, want, 1e-9), `site ${s.site}: ${s.p_value} vs ${want}`).toBe(true);
		}
	});

	it('q_value is Benjamini-Hochberg over the sorted p_value column', () => {
		const q = benjaminiHochberg(Float64Array.from(out.sites, (s) => s.p_value));
		for (let i = 0; i < out.sites.length; i++) {
			expect(close(out.sites[i].q_value, q[i], 1e-12), `site ${out.sites[i].site}`).toBe(true);
		}
	});

	it('at --n-permutations 0 the association p is the parametric one', () => {
		expect(out.permulations_count).toBe(0);
		expect(out.gene_p_value_perm).toBeNull();
		for (const s of out.sites) {
			expect(s.p_assoc_perm).toBeNull();
			expect(s.p_assoc).toBe(s.p_assoc_parametric);
		}
	});

	it('p_assoc_parametric is t.sf(rho * sqrt(df / (1 - rho^2)), df) for an integer df <= N - 2', () => {
		// N_valid is not recorded, so the test recovers the degrees of freedom the reference used
		// and checks that they are a legal per-site count: df = max(1, N_valid - 2) <= N - 2.
		for (const s of out.sites) {
			const rho = s.association_rho;
			let found = -1;
			for (let df = 1; df <= N - 2; df++) {
				const t = rho * Math.sqrt(df / Math.max(1e-15, 1 - rho * rho));
				if (close(tSf(t, df), s.p_assoc_parametric, 1e-9)) { found = df; break; }
			}
			expect(found, `site ${s.site}: no df reproduces p_assoc_parametric`).toBeGreaterThan(0);
		}
	});

	it('the dual-track extreme-value block follows from max_assoc, N and L', () => {
		expect(out.max_assoc).toBe(out.sites[0].association_rho);
		const sigma = 1.0 / Math.sqrt(Math.max(PHENOTYPE_THRESHOLDS.evdMinTaxa, N));
		let pSingle = 2.0 * normSf(Math.abs(out.max_assoc / sigma));
		pSingle = Math.max(1e-15, Math.min(1.0, pSingle));
		let pEvd = -Math.expm1(-L * pSingle);
		pEvd = Math.max(1e-15, Math.min(1.0, pEvd));
		expect(close(out.p_evd_length_adjusted, pEvd, 1e-12)).toBe(true);
		expect(close(out.score_track_a, -Math.log10(pEvd), 1e-12)).toBe(true);
		expect(out.score_track_b).toBe(out.norm_spectral_ratio);
		expect(close(out.dual_track_composite, Math.max(out.score_track_a / 10, out.score_track_b), 1e-12)).toBe(true);
	});

	it('the PARS bracket is the first 15 sites with rho >= 0.40 and score >= 0.50, in score order', () => {
		const top = [];
		for (const s of out.sites) {
			if (top.length >= PHENOTYPE_THRESHOLDS.parsMaxSites) break;
			if (s.association_rho >= PHENOTYPE_THRESHOLDS.parsMinRho && s.score >= PHENOTYPE_THRESHOLDS.parsMinScore) {
				top.push(`${s.ref_aa}${s.site}${s.derived_aa}`);
			}
		}
		expect(out.compact_pars_signature).toBe(top.length > 0 ? `[ ${top.join(' - ')} ]` : '[]');
		// RHO's best site is rho = 0.2496, below the 0.40 gate, so the bracket is empty.
		expect(out.compact_pars_signature).toBe('[]');
	});

	it('significant_sites_count is q <= alpha AND rho > 0 at the CLI default alpha', () => {
		const sig = out.sites.filter((s) => s.q_value <= 0.05 && s.association_rho > 0);
		expect(out.significant_sites_count).toBe(sig.length);
		expect(out.significant_sites_count).toBe(22);
	});

	it('co-selection pairs: CESI, the t p-value at df = N - 2, BH, and CESI-descending order', () => {
		expect(out.coselection_pairs.length).toBe(out.coselection_pairs_count);
		const dfPair = Math.max(1, N - 2);
		for (const p of out.coselection_pairs) {
			expect(Object.keys(p)).toEqual([
				'site_u', 'site_v', 'ref_u', 'ref_v', 'lrt_u', 'lrt_v', 'similarity', 'cesi', 'p_value',
				'shared_branches', 'q_value'
			]);
			expect(p.similarity).toBeGreaterThan(PHENOTYPE_THRESHOLDS.pairMinSimilarity);
			const cesi = p.similarity * Math.sqrt(Math.max(0.1, p.lrt_u) * Math.max(0.1, p.lrt_v));
			expect(close(p.cesi, cesi, 1e-9), `pair ${p.site_u}-${p.site_v} cesi`).toBe(true);
			const t = p.similarity * Math.sqrt(dfPair / Math.max(1e-15, 1 - p.similarity * p.similarity));
			expect(close(p.p_value, tSf(t, dfPair), 1e-9), `pair ${p.site_u}-${p.site_v} p`).toBe(true);
			expect(p.shared_branches).toBeGreaterThanOrEqual(0);
			expect(p.shared_branches).toBeLessThanOrEqual(N);
		}
		const q = benjaminiHochberg(Float64Array.from(out.coselection_pairs, (p) => p.p_value));
		for (let i = 0; i < out.coselection_pairs.length; i++) {
			expect(close(out.coselection_pairs[i].q_value, q[i], 1e-12)).toBe(true);
		}
		for (let i = 1; i < out.coselection_pairs.length; i++) {
			expect(out.coselection_pairs[i - 1].cesi).toBeGreaterThanOrEqual(out.coselection_pairs[i].cesi);
		}
	});

	it('pairs agree with the site table on ref and lrt, and their sites are the significant ones', () => {
		const byS = new Map(out.sites.map((s) => [s.site, s]));
		const sig = new Set(out.sites.filter((s) => s.q_value <= 0.05 && s.association_rho > 0).map((s) => s.site));
		for (const p of out.coselection_pairs) {
			expect(sig.has(p.site_u), `site_u ${p.site_u} not significant`).toBe(true);
			expect(sig.has(p.site_v), `site_v ${p.site_v} not significant`).toBe(true);
			expect(p.ref_u).toBe(byS.get(p.site_u).ref_aa);
			expect(p.ref_v).toBe(byS.get(p.site_v).ref_aa);
			expect(p.lrt_u).toBe(byS.get(p.site_u).hyphaeon_lrt);
			expect(p.lrt_v).toBe(byS.get(p.site_v).hyphaeon_lrt);
		}
		// QUIRK (phenotype.py:552): sub_indices is in SCORE order, so the pair (u, v) is not
		// ordered by position — the first recorded pair is (325, 83).
		expect(out.coselection_pairs[0].site_u).toBe(325);
		expect(out.coselection_pairs[0].site_v).toBe(83);
		expect(out.coselection_pairs.some((p) => p.site_u > p.site_v)).toBe(true);
	});

	it('trait sectors: signatures, ascending site lists, size and the isotropic baseline', () => {
		expect(out.trait_sectors.length).toBe(out.trait_sectors_count);
		const byS = new Map(out.sites.map((s) => [s.site, s]));
		for (const sec of out.trait_sectors) {
			expect(sec.size).toBe(sec.sites.length);
			expect(sec.size).toBeGreaterThanOrEqual(PHENOTYPE_THRESHOLDS.sectorMinCliqueSize);
			expect(sec.spectral_coherence).toBeGreaterThanOrEqual(PHENOTYPE_THRESHOLDS.sectorMinCoherence);
			expect([...sec.sites].sort((a, b) => a - b)).toEqual(sec.sites);
			expect(close(sec.isotropic_baseline, 1 / sec.size, 1e-12)).toBe(true);
			const sig = sec.sites.map((site) => `${byS.get(site).ref_aa}${site}`).join(' - ');
			expect(sec.consensus_signature).toBe(`[ ${sig} ]`);
			expect(sec.pars_signature).toBe(sec.consensus_signature);
			expect(sec.shared_branches).toBe(sec.shared_taxa);
		}
	});

	it('the CLI invocation the fixture records is the one the port models', () => {
		expect(argv.slice(0, 2)).toEqual(['hyphaeon', 'phenotype']);
		expect(argv).toContain('--n-permutations');
		expect(argv[argv.indexOf('--n-permutations') + 1]).toBe('0');
		expect(argv[argv.indexOf('--seed') + 1]).toBe('42');
		expect(argv[argv.indexOf('--mds-sign') + 1]).toBe('canonical');
		expect(out.tree).toBeNull(); // RHO has no tree, so the permulation branch was unreachable
	});
});
