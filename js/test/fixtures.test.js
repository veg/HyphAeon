/**
 * fixtures.test.js — the fixture replay's first rung.
 *
 * WHY THIS FILE EXISTS
 *
 * .github/workflows/js.yml calls its test step "vitest + fixture replay", and PLAN.md §5.5 rule 2
 * says fixtures are generated and consumed in the same commit. Until Phase 1 lands the ports that
 * consume every fixture (stats, filter, attribution, ...), this file is what makes that sentence
 * true: it reads fixtures/manifest.json, checks that every file the generator says it wrote is
 * present with the case count it recorded, and replays the ONE fixture the seeded library can
 * already answer — dataset/tokenizer.json — against tokenizer.js.
 *
 * THE TOKENIZER REPLAY IS PINNED AS A DIVERGENCE, NOT AS A PASS. The seeded tokenizer is DM3's
 * (AxoMEME 2.0's training vocabulary: 64 codons in TCAG order, stops as real tokens, gap 64,
 * unknown 65; amino acids stop 20 / gap 21 / unknown 22). hyphaeon/dataset.py numbers the 61
 * sense codons 0..60 with stops SKIPPED and sends stops, gaps and anything unrecognised to 64,
 * with a single amino-acid sentinel 20. Measured against fixtures/dataset/tokenizer.json at
 * Phase 0 integration: 54 of the 64 codons disagree (every codon from TAA onward is shifted by
 * the number of stops preceding it), all 20 standard residues agree, and 17 of 21 sentinel
 * inputs disagree at the codon level and 18 at the amino-acid level. The port-verbatim rule
 * kept tokenizer.js as DM3 wrote it; the runtime's bat_oas1 experiment (hyphaeon-app CLAUDE.md,
 * Phase 0 notes) shows the shipped checkpoint wants dataset.py's vocabulary. These numbers are
 * asserted EXACTLY so that the day tokenizer.js adopts dataset.py's tables this test fails and
 * is rewritten as a plain replay — the fixture is already the oracle; only the expectation
 * changes.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { codonToken, aaToken } from '../src/preprocess/tokenizer.js';

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'fixtures');

function loadJson(rel) {
	return JSON.parse(readFileSync(join(FIXTURES, rel), 'utf8'));
}

describe('fixtures/manifest.json agrees with the files on disk', () => {
	const manifest = loadJson('manifest.json');

	it('names the generator, the engine commit and the weights it ran with', () => {
		expect(manifest.generator).toBe('scripts/gen_fixtures.py');
		expect(manifest.engine_commit).toMatch(/^[0-9a-f]{40}$/);
		expect(manifest.model_safetensors_sha256).toMatch(/^[0-9a-f]{64}$/);
	});

	it('lists every fixture file with its byte size', () => {
		for (const [rel, bytes] of Object.entries(manifest.bytes)) {
			expect(existsSync(join(FIXTURES, rel)), `${rel} missing`).toBe(true);
			expect(statSync(join(FIXTURES, rel)).size, `${rel} size`).toBe(bytes);
		}
		const total = Object.values(manifest.bytes).reduce((a, b) => a + b, 0);
		expect(total).toBe(manifest.total_bytes);
	});

	it('records a case count that matches each file', () => {
		for (const [module, fns] of Object.entries(manifest.counts)) {
			for (const [fn, n] of Object.entries(fns)) {
				const cases = loadJson(`${module}/${fn}.json`);
				expect(Array.isArray(cases), `${module}/${fn}.json is a list`).toBe(true);
				expect(cases.length, `${module}/${fn}.json cases`).toBe(n);
				for (const c of cases) {
					expect(typeof c.name).toBe('string');
					expect(c).toHaveProperty('inputs');
					expect(c).toHaveProperty('outputs');
					expect(c).toHaveProperty('tolerance');
				}
			}
		}
	});

	it('carries no absolute paths', () => {
		for (const rel of Object.keys(manifest.bytes)) {
			expect(readFileSync(join(FIXTURES, rel), 'utf8')).not.toContain('/Users/');
		}
	});
});

describe('dataset/tokenizer.json replay (pinned divergence, see header)', () => {
	const cases = loadJson('dataset/tokenizer.json');
	const byName = Object.fromEntries(cases.map((c) => [c.name, c]));

	function mismatches(c) {
		let codon = 0;
		let aa = 0;
		c.inputs.codons.forEach((x, i) => {
			if (codonToken(x) !== c.outputs.codon_tokens[i]) codon += 1;
			if (aaToken(x) !== c.outputs.aa_tokens[i]) aa += 1;
		});
		return { codon, aa };
	}

	it('all_64_codons_TCAG_order: the 20 residues agree; 54 codon tokens do not', () => {
		const c = byName.all_64_codons_TCAG_order;
		expect(c.inputs.codons).toHaveLength(64);
		expect(mismatches(c)).toEqual({ codon: 54, aa: 0 });
	});

	it('gaps_ambiguity_lowercase_U: 17 of 21 codon sentinels and 18 of 21 aa sentinels differ', () => {
		const c = byName.gaps_ambiguity_lowercase_U;
		expect(c.inputs.codons).toHaveLength(21);
		expect(mismatches(c)).toEqual({ codon: 17, aa: 18 });
	});

	it('tables: dataset.py numbers 61 sense codons 0..60, sends the 3 stops to 64, and has a 20-entry AA_MAP', () => {
		const c = byName.tables;
		const code = c.outputs.GENETIC_CODE;
		expect(Object.keys(code)).toHaveLength(64);
		const sense = Object.entries(code).filter(([, v]) => v !== 64);
		expect(sense).toHaveLength(61);
		expect(new Set(sense.map(([, v]) => v)).size).toBe(61);
		expect(Math.max(...sense.map(([, v]) => v))).toBe(60);
		expect(Object.entries(code).filter(([, v]) => v === 64).map(([k]) => k).sort()).toEqual(['TAA', 'TAG', 'TGA']);
		expect(Object.keys(c.outputs.AA_MAP)).toHaveLength(20);
	});
});
