/**
 * WHY THIS FILE EXISTS
 *
 * Pins src/writers.js to the reference CLI's own bytes. The parity class for writers is "byte-equal
 * after canonicalisation" (PLAN.md §5.4); these tests demand byte equality BEFORE canonicalisation
 * against strings the Python serialisers themselves produced (test/data/evaluate/gen.py:
 * json.dumps, pandas.DataFrame.to_csv, networkx.write_graphml via lxml, float.__repr__, format)
 * for the record shapes of fixtures/e2e/*.json — so a byte that differs is a bug in the port, not
 * a difference of canonical form.
 *
 * Every CSV writer is exercised on the e2e fixture records (meme sites with and without
 * attribution, the busted record, epistasis edges / sectors / plasticity, phenotype sites) by
 * re-serialising the fixture's own numbers and comparing headers and rows with the pandas output;
 * the GraphML writer is compared byte-for-byte on the Smc6 network and three synthetic graphs, and
 * additionally parsed structurally (keys, nodes, edges, data children) so a layout regression and a
 * content regression are reported separately.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
	pyFloatRepr,
	pyFormatFixed,
	pyFormatG,
	pyRepr,
	pyStr,
	pyJsonDumps,
	dataFrameCsv,
	resultJson,
	evaluateJson,
	memeSiteRecords,
	memeResult,
	memeJson,
	memeCsv,
	bustedJson,
	bustedCsv,
	phenotypeCsv,
	epistasisCsv,
	dmsCsv,
	graphml,
	PY_FLOAT_KEYS,
	PY_INT_KEYS,
} from '../src/writers.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..', '..');
const ref = JSON.parse(readFileSync(join(HERE, 'data', 'evaluate', 'python_formatting.json'), 'utf8'));
const e2e = (name) => JSON.parse(readFileSync(join(ROOT, 'fixtures', 'e2e', `${name}.json`), 'utf8'))[0].outputs;

describe('float.__repr__ (pyFloatRepr)', () => {
	it(`matches Python on ${ref.float_repr.length} values incl. the exponent-switch boundaries`, () => {
		for (const [v, s] of ref.float_repr) expect(pyFloatRepr(v), String(v)).toBe(s);
	});
	it('special values and negative zero', () => {
		expect(pyFloatRepr(Infinity)).toBe('inf');
		expect(pyFloatRepr(-Infinity)).toBe('-inf');
		expect(pyFloatRepr(NaN)).toBe('nan');
		expect(pyFloatRepr(-0)).toBe('-0.0');
		expect(pyFloatRepr(0)).toBe('0.0');
		expect(ref.float_repr_special).toEqual([['inf', 'inf'], ['-inf', '-inf'], ['nan', 'nan']]);
	});
});

describe("format(x, '.6f') and format(x, 'g')", () => {
	it('fixed: round-half-even on the exact binary value, sign kept', () => {
		for (const [v, s] of ref.format_fixed6) expect(pyFormatFixed(v, 6), String(v)).toBe(s);
		expect(pyFormatFixed(2.5, 0)).toBe('2');
		expect(pyFormatFixed(3.5, 0)).toBe('4');
		expect(pyFormatFixed(NaN, 6)).toBe('nan');
	});
	it('g: six significant digits, trailing zeros stripped, exponent below 1e-4 and at 1e6', () => {
		for (const [v, s] of ref.format_g) expect(pyFormatG(v), String(v)).toBe(s);
	});
});

describe('repr() / str() (pyRepr, pyStr)', () => {
	it('strings, containers, None, booleans', () => {
		for (const [o, s] of ref.repr) {
			const kind = o !== null && typeof o === 'object' && !Array.isArray(o) ? 'float' : undefined;
			expect(pyRepr(o, kind), JSON.stringify(o)).toBe(s);
		}
		expect(pyStr('a b')).toBe('a b');
		expect(pyStr(5)).toBe('5');
		expect(pyStr(5, 'float')).toBe('5.0');
		expect(pyStr(null)).toBe('None');
		expect(pyRepr(new Map([['k', 1]]))).toBe("{'k': 1}");
		expect(pyRepr(Float32Array.from([0.5, 1]), 'float')).toBe('[0.5, 1.0]');
		expect(pyRepr([1, 2], 'int')).toBe('[1, 2]');
	});
});

describe('json.dumps(indent=2) (pyJsonDumps)', () => {
	it('the meme document of fixtures/e2e/meme_bat_oas1_attribute_filter.json, byte for byte', () => {
		const d = e2e('meme_bat_oas1_attribute_filter');
		const [a, b] = ref.meme_json.sites_slice;
		const sub = {
			...d,
			runtime_sec: ref.meme_json.runtime_sec,
			attributions: new Map([[ref.meme_json.attribution_key, d.attributions[ref.meme_json.attribution_key]]]),
			sites: d.sites.slice(a, b),
		};
		expect(memeJson(sub)).toBe(ref.meme_json.text);
		expect(resultJson(sub)).toBe(ref.meme_json.text);
	});
	it('empties, escapes, NaN/Infinity literals, float keys', () => {
		const edge = {
			a: [],
			b: {},
			c: [1, 2.5, null, true, false, 'xé\n"y"\\/\t😀\x01\x7f'],
			d: NaN,
			e: Infinity,
			f: -Infinity,
			g: 1e16,
			h: 1e-5,
			i: -0,
			j: { k: [{ l: [] }] },
			ü: 1,
		};
		expect(pyJsonDumps(edge, { floatKeys: new Set(['g', 'h', 'i']) })).toBe(ref.json_edge.text);
		expect(pyJsonDumps({ x: 1, y: undefined, z: [undefined] })).toBe('{\n  "x": 1,\n  "z": [\n    null\n  ]\n}');
		expect(pyJsonDumps([], { indent: 4 })).toBe('[]');
		expect(pyJsonDumps({ a: { b: 1 } }, { indent: 4 })).toBe('{\n    "a": {\n        "b": 1\n    }\n}');
	});
	it('allow_nan=False raises (evaluateJson)', () => {
		expect(() => evaluateJson({ x: NaN })).toThrow(ref.json_allow_nan_error);
		expect(() => evaluateJson({ x: [Infinity] })).toThrow('Out of range float values');
		expect(evaluateJson({ pearson_r: null, ppv: 1, matched_genes: 1 })).toBe('{\n  "pearson_r": null,\n  "ppv": 1.0,\n  "matched_genes": 1\n}');
	});
	it('the schema names every int and float field the CLI writes, with no key in both sets', () => {
		for (const k of PY_FLOAT_KEYS) expect(PY_INT_KEYS.has(k), k).toBe(false);
		for (const k of ['hyphaeon_lrt', 'p_value', 'q_value', 'cesi', 'fdr_q', 'p_perm', 'pearson_r', 'roc_auc']) expect(PY_FLOAT_KEYS.has(k)).toBe(true);
		for (const k of ['site', 'site_u', 'shared_branches', 'sector_id', 'taxa_count', 'true_positive']) expect(PY_INT_KEYS.has(k)).toBe(true);
	});
});

describe('hyphaeon meme writers (cli.py:280-327)', () => {
	const d = e2e('meme_bat_oas1_attribute_filter');
	it('memeSiteRecords rebuilds the fixture site dicts from lrt/p/q/mask and 0-based attributions', () => {
		const attributions = new Map(Object.entries(d.attributions).map(([k, v]) => [Number(k) - 1, v]));
		const records = memeSiteRecords(
			Float32Array.from(d.sites.map((s) => s.hyphaeon_lrt)),
			Float32Array.from(d.sites.map((s) => s.p_value)),
			Float32Array.from(d.sites.map((s) => s.q_value)),
			Uint8Array.from(d.sites.map((s) => (s.is_invariable ? 1 : 0))),
			attributions,
		);
		expect(records).toEqual(d.sites);
		expect(Object.keys(records[140])).toEqual([
			'site', 'hyphaeon_lrt', 'p_value', 'q_value', 'is_invariable', 'evolutionary_epoch', 'adaptation_mode', 'top_driver', 'top_mutation', 'attribution_details',
		]);
		// A plain object keyed by 0-based index works too, and empty driving_species gives None fields.
		const rec = { ...d.attributions['141'], driving_species: [] };
		const one = memeSiteRecords([1], [0.5], [0.5], [false], { 0: rec })[0];
		expect(one.top_driver).toBeNull();
		expect(one.top_mutation).toBeNull();
	});
	it('memeResult + memeJson reproduce the fixture document (keys, order, 1-based attribution keys)', () => {
		const result = memeResult({
			alignment: d.alignment,
			tree: d.tree,
			taxaCount: d.taxa_count,
			codonCount: d.codon_count,
			runtimeSec: null,
			filterEnabled: d.filter_enabled,
			artifactsMasked: d.artifacts_masked,
			attributionEnabled: d.attribution_enabled,
			attributions: new Map(Object.entries(d.attributions).map(([k, v]) => [Number(k) - 1, v])),
			sites: d.sites,
		});
		expect(Object.keys(result)).toEqual(Object.keys(d));
		expect(JSON.parse(memeJson(result))).toEqual(d);
		expect(memeResult({ alignment: 'a.fa', taxaCount: 1, codonCount: 1, runtimeSec: 0, sites: [] }).tree).toBe('embedded_in_alignment');
		const [a, b] = ref.meme_json.sites_slice;
		const sub = memeResult({
			alignment: d.alignment,
			tree: d.tree,
			taxaCount: d.taxa_count,
			codonCount: d.codon_count,
			runtimeSec: ref.meme_json.runtime_sec,
			filterEnabled: true,
			artifactsMasked: [],
			attributionEnabled: true,
			attributions: { [Number(ref.meme_json.attribution_key) - 1]: d.attributions[ref.meme_json.attribution_key] },
			sites: d.sites.slice(a, b),
		});
		expect(memeJson(sub)).toBe(ref.meme_json.text);
	});
	it('memeCsv: attribution columns appear when any site carries them, empty elsewhere', () => {
		const [a, b] = ref.meme_csv_attr.sites_slice;
		expect(memeCsv(d.sites.slice(a, b))).toBe(ref.meme_csv_attr.text);
		expect(memeCsv(d.sites.slice(a, b), { attribution: false })).toBe(
			ref.meme_csv_attr.text
				.split('\n')
				.map((line) => line.split(',').slice(0, 5).join(','))
				.join('\n'),
		);
		expect(memeCsv(d.sites.slice(a, b), { attribution: true })).toBe(ref.meme_csv_attr.text);
	});
	it('memeCsv: plain sites of fixtures/e2e/meme_Smc6.json (the examples/Smc6_results.csv layout)', () => {
		const smc6 = e2e('meme_Smc6');
		const [a, b] = ref.meme_csv_plain.sites_slice;
		expect(memeCsv(smc6.sites.slice(a, b))).toBe(ref.meme_csv_plain.text);
		expect(memeCsv(smc6.sites.slice(a, b), { attribution: true })).toBe(ref.meme_csv_plain.text);
		const example = readFileSync(join(ROOT, 'examples', 'Smc6_results.csv'), 'utf8');
		expect(memeCsv(smc6.sites).split('\n')[0]).toBe(example.split('\n')[0]);
		expect(memeCsv(smc6.sites).split('\n').length).toBe(example.split('\n').length);
	});
});

describe('hyphaeon busted writers (cli.py:549-568)', () => {
	it('bustedCsv columns and formats', () => {
		expect(bustedCsv([ref.busted.record])).toBe(ref.busted.csv);
		expect(bustedCsv([ref.busted.record]).split('\n')[0]).toBe(
			'Gene,Taxa,Sites,p_ACAT,p_Simes,Selection_Prob,Pred_Gene_LRT,Omnibus_LRT,Omega_3,Prop_Positive,Sig_Sites_p05,Selected,Time_ms',
		);
		const nulled = e2e('busted_Smc6');
		const line = bustedCsv([nulled]).split('\n')[1];
		expect(line.startsWith('Smc6,20,1097,0.11831563373812454,1.0,,,3.285405158996582,,,5,,')).toBe(true);
	});
	it('bustedJson: a bare record for one alignment, a list in batch mode', () => {
		expect(bustedJson([ref.busted.record])).toBe(ref.busted.json_single);
		expect(bustedJson([ref.busted.record, ref.busted.record])).toBe(ref.busted.json_batch);
		expect(bustedJson([ref.busted.record], { batch: true })).toBe(ref.busted.json_batch.replace(/\n  \},\n  \{[\s\S]*\n  \}\n\]$/, '\n  }\n]'));
	});
});

describe('epistasis / dms / phenotype CSVs (cli.py:808-819, 898-902, 700-701)', () => {
	const e = e2e('epistasis_Smc6_n_permutations_1000');
	const p = e2e('phenotype_RHO_marine_n_permutations_0');
	it('edges are the table when present', () => {
		expect(epistasisCsv(e)).toBe(ref.edges_csv);
		expect(epistasisCsv({ edges: e.edges })).toBe(ref.edges_csv);
		const example = readFileSync(join(ROOT, 'examples', 'Smc6_edges.csv'), 'utf8').split('\n')[0].split(',');
		// examples/ predate the p_val/hyper_p/branches_u columns; the shared prefix and dtypes agree.
		expect(ref.edges_csv.split('\n')[0].split(',').slice(0, 7)).toEqual(example.slice(0, 7));
	});
	it('plasticity (with the mutant_deltas dict literal) when there are no edges, or under the dms subcommand', () => {
		const [a, b] = ref.plasticity_csv.slice;
		expect(epistasisCsv({ edges: [], sectors: e.sectors, plasticity: e.plasticity.slice(a, b) })).toBe(ref.plasticity_csv.text);
		expect(epistasisCsv({ edges: e.edges, plasticity: e.plasticity.slice(a, b) }, { command: 'dms' })).toBe(ref.plasticity_csv.text);
		expect(epistasisCsv({ edges: e.edges, plasticity: [] }, { command: 'dms' })).toBe(ref.edges_csv);
	});
	it('sectors (with the sites list literal) when there is nothing else', () => {
		expect(epistasisCsv({ edges: [], plasticity: [], sectors: e.sectors })).toBe(ref.sectors_csv);
		expect(epistasisCsv({})).toBe('\n');
	});
	it('dmsCsv drops mutant_deltas', () => {
		const [a, b] = ref.dms_csv.slice;
		expect(dmsCsv(e.plasticity.slice(a, b))).toBe(ref.dms_csv.text);
		expect(dmsCsv([{ site: 1, wt_aa: 'M' }])).toBe('site,wt_aa\n1,M\n');
	});
	it('phenotypeCsv', () => {
		const [a, b] = ref.phenotype_csv.slice;
		expect(phenotypeCsv(p.sites.slice(a, b))).toBe(ref.phenotype_csv.text);
		expect(ref.phenotype_csv.text.split('\n')[1]).toContain(',,'); // p_assoc_perm None -> empty cell
	});
});

describe('DataFrame.to_csv(index=False) dtype and quoting rules (dataFrameCsv)', () => {
	it('quirks: missing keys, None, NaN, quoting, list and dict cells, embedded newline', () => {
		const records = [
			{ a: 1, b: true, c: 'x,y', d: 'q"q', e: null, f: 1.5 },
			{ a: 2, f: NaN, g: [1, 2], h: { A: 0.5, B: -1.0 }, i: 'line\nbreak' },
		];
		expect(dataFrameCsv(records, { floatKeys: new Set(['h']) })).toBe(ref.csv_quirks);
	});
	it('float column formatting incl. inf and negative zero', () => {
		const values = [0.0, 1.0, 1e16, 1e15, 0.0001, 0.00001, 123456789012345678.0, 1.5e-300, -0.0, 2.5, 1e22, Infinity, -Infinity];
		expect(dataFrameCsv(values.map((v) => ({ v })), { floatKeys: new Set(['v']) })).toBe(ref.csv_floats);
	});
	it('a missing int becomes float64; a missing bool becomes object; a lone empty field is quoted', () => {
		expect(dataFrameCsv([{ a: 1, b: 1 }, { a: 2 }])).toBe(ref.csv_int_missing);
		expect(dataFrameCsv([{ a: true, b: false }, { a: false }])).toBe(ref.csv_bool_missing);
		expect(dataFrameCsv([{ a: '' }, { a: 'x' }, { a: null }])).toBe(ref.csv_single_column);
		expect(dataFrameCsv([])).toBe('\n');
		expect(dataFrameCsv([new Map([['k', 'v']])])).toBe('k\nv\n');
		expect(dataFrameCsv([{ x: 1, y: 2 }], { columns: ['y'] })).toBe('y\n2\n');
	});
});

describe('nx.write_graphml (cli.py:821-833), lxml layout', () => {
	for (const [name, c] of Object.entries(ref.graphml)) {
		it(`${name}: byte-equal`, () => {
			expect(graphml(c.edges)).toBe(c.text);
		});
	}
	it('the Smc6 co-selection network: byte-equal and structurally sound', () => {
		const e = e2e('epistasis_Smc6_n_permutations_1000');
		const text = graphml(e.edges);
		expect(text).toBe(ref.edges_graphml);
		const parsed = parseGraphml(text);
		expect(parsed.keys).toEqual([
			{ id: 'd3', for: 'edge', name: 'fdr_q', type: 'double' },
			{ id: 'd2', for: 'edge', name: 'shared', type: 'long' },
			{ id: 'd1', for: 'edge', name: 'cesi', type: 'double' },
			{ id: 'd0', for: 'edge', name: 'weight', type: 'double' },
		]);
		const sites = new Set();
		for (const edge of e.edges) {
			sites.add(String(edge.site_u));
			sites.add(String(edge.site_v));
		}
		expect(new Set(parsed.nodes)).toEqual(sites);
		expect(parsed.nodes.length).toBe(sites.size);
		expect(parsed.edges.length).toBe(e.edges.length);
		const pairs = new Set(e.edges.map((edge) => [edge.site_u, edge.site_v].map(String).sort().join('|')));
		for (const edge of parsed.edges) {
			expect(pairs.has([edge.source, edge.target].sort().join('|'))).toBe(true);
			expect(Object.keys(edge.data)).toEqual(['d0', 'd1', 'd2', 'd3']);
			const src = e.edges.find((x) => new Set([String(x.site_u), String(x.site_v)]).has(edge.source) && new Set([String(x.site_u), String(x.site_v)]).has(edge.target));
			expect(Number(edge.data.d0)).toBe(src.similarity);
			expect(Number(edge.data.d1)).toBe(src.cesi);
			expect(edge.data.d2).toBe(String(src.shared_branches));
			expect(Number(edge.data.d3)).toBe(src.fdr_q);
		}
		expect(text.endsWith('</graph></graphml>')).toBe(true);
		expect(text.startsWith("<?xml version='1.0' encoding='utf-8'?>\n<graphml ")).toBe(true);
	});
	it('optional pre-registered nodes come first and isolated ones are emitted', () => {
		const text = graphml(ref.graphml.small.edges, [999, 279]);
		const parsed = parseGraphml(text);
		expect(parsed.nodes.slice(0, 3)).toEqual(['999', '279', '930']);
		expect(parsed.edges.length).toBe(4);
	});
	it('examples/HIV1_RT_coselection.graphml has the layout this writer produces', () => {
		const example = readFileSync(join(ROOT, 'examples', 'HIV1_RT_coselection.graphml'), 'utf8');
		const head = example.slice(0, 700);
		const empty = graphml([]);
		expect(head.startsWith(empty.slice(0, empty.indexOf('<graph ')))).toBe(true);
		expect(head).toContain('<key id="d3" for="edge" attr.name="fdr_q" attr.type="double"/>\n<key id="d2" for="edge" attr.name="shared" attr.type="long"/>\n<key id="d1" for="edge" attr.name="cesi" attr.type="double"/>\n<key id="d0" for="edge" attr.name="weight" attr.type="double"/>\n<graph edgedefault="undirected"><node id="122"/>\n');
		expect(example.endsWith('</edge>\n</graph></graphml>')).toBe(true);
	});
});

/** A minimal GraphML reader for the exact element layout the writer emits. */
function parseGraphml(text) {
	const keys = [...text.matchAll(/<key id="([^"]*)" for="([^"]*)" attr\.name="([^"]*)" attr\.type="([^"]*)"\/>/g)].map((m) => ({
		id: m[1],
		for: m[2],
		name: m[3],
		type: m[4],
	}));
	const nodes = [...text.matchAll(/<node id="([^"]*)"\/>/g)].map((m) => m[1]);
	const edges = [...text.matchAll(/<edge source="([^"]*)" target="([^"]*)">\n((?:  <data key="[^"]*">[^<]*<\/data>\n)*)<\/edge>/g)].map((m) => {
		const data = {};
		for (const d of m[3].matchAll(/<data key="([^"]*)">([^<]*)<\/data>/g)) data[d[1]] = d[2];
		return { source: m[1], target: m[2], data };
	});
	return { keys, nodes, edges };
}
