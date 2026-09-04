/**
 * WHY THIS FILE EXISTS
 *
 * Added for @veg/hyphaeon-js; it has no DataMonkey 3 original, because DataMonkey 3 had no package
 * boundary to defend — its callers reached into `src/lib/services/axomeme/` directly. Here the
 * boundary IS the product decision (PLAN.md §5.5), and three things about it fail silently without a
 * test:
 *
 *   1. NOTHING ELSE IMPORTS `src/index.js`. Every other test imports the module it exercises, so a
 *      broken entry point — a typo'd path, a module dropped from the barrel, a NAME COLLISION
 *      between two `export *` lines, which is a hard SyntaxError at link time — would leave the
 *      whole suite green and break every consumer on install. This file is the only place that
 *      loads the package the way an app does.
 *   2. THE PUBLIC SURFACE IS A COMMITMENT. `index.js` re-exports whole modules deliberately (its
 *      header says why), which means adding an export to any file under `src/preprocess/` publishes
 *      it, and deleting one breaks a pinned consumer. Listing all 41 names here turns both into a
 *      visible diff in the pull request that causes them, rather than a discovery in `hyphaeon-app`.
 *   3. THE EXPORTS MAP IS LOAD-BEARING, not decoration. `package.json` exposes exactly `.`, so a
 *      deep import into `src/preprocess/…` must FAIL — otherwise consumers pin internal paths, and
 *      the directory layout becomes a public API nobody agreed to. Node's self-reference resolution
 *      lets the package import itself by name, so the real resolver is what runs below rather than a
 *      relative path that would bypass the map entirely.
 *
 * The surface is asserted as a whole sorted list rather than symbol by symbol on purpose: a missing
 * name and an unintended extra one are the same class of mistake and both should fail here.
 */
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { describe, it, expect } from 'vitest';
import * as lib from '@veg/hyphaeon-js';
import * as modelContract from '../src/preprocess/modelContract.js';
import * as assemble from '../src/preprocess/assemble.js';
import * as variability from '../src/preprocess/variability.js';

/** Every public symbol of src/preprocess/*, sorted. Grouped by module for reviewable diffs. */
const PUBLIC_SURFACE = [
	// modelContract.js
	'AA_GAP',
	'AA_LIST',
	'AA_UNKNOWN',
	'AA_VALID_BELOW',
	'CODON_GAP',
	'CODON_ORDER',
	'CODON_UNKNOWN',
	'CODON_VALID_BELOW',
	'INPUT_NAMES',
	'INPUT_SPEC',
	'MAX_SPECIES_DEFAULT',
	'MDS_COMPONENTS',
	'NUM_CODON_TOKENS',
	'OUTPUT_NAMES_V1',
	'OUTPUT_SPEC',
	'OUTPUT_SPEC_V1',
	'VERIFIED_MODEL_SHA256',
	'WINDOW_SIZE_DEFAULT',
	'validateInputBundle',
	// newick.js
	'leafIndex',
	'normalizeTaxonName',
	'parseNewick',
	// tokenizer.js
	'AA_TO_IDX',
	'CODON_LIST',
	'CODON_TO_IDX',
	'GENETIC_CODE',
	'aaToken',
	'codonToken',
	'tokenizeSequence',
	// patristic.js
	'maxPdSelect',
	'patristicMatrix',
	'patristicRow',
	'rootDistances',
	// symmetricEigen.js / mds.js
	'computeMdsCoordinates',
	'symmetricEigen',
	// variability.js
	'isSiteVariable',
	'siteVariability',
	// assemble.js
	'batchSizeFor',
	'chooseReference',
	'orderSpecies',
	'prepareAlignment'
].sort();

describe('the package entry point', () => {
	it('exports exactly the documented public surface', () => {
		expect(Object.keys(lib).sort()).toEqual(PUBLIC_SURFACE);
	});

	it('re-exports the modules themselves, not copies', () => {
		// `export *` gives live bindings, so these are identity comparisons. If one ever came back as
		// a distinct object, something had wrapped or re-declared it and the two would drift.
		expect(lib.validateInputBundle).toBe(modelContract.validateInputBundle);
		expect(lib.INPUT_SPEC).toBe(modelContract.INPUT_SPEC);
		expect(lib.OUTPUT_SPEC_V1).toBe(modelContract.OUTPUT_SPEC_V1);
		expect(lib.prepareAlignment).toBe(assemble.prepareAlignment);
		expect(lib.isSiteVariable).toBe(variability.isSiteVariable);
	});

	it('runs the preprocessing pipeline through the entry point alone', () => {
		// The smallest end-to-end proof that the barrel is wired: parse, order, tokenise, distance,
		// MDS and validate, touching nothing but `lib`. Any missing re-export shows up here as a
		// TypeError rather than as an undefined that quietly becomes NaN downstream.
		const p = lib.prepareAlignment({
			names: ['alpha', 'beta', 'gamma'],
			sequences: ['ATGTTATCA', 'ATGCTATCA', 'ATGTTAAGC'],
			treeText: '((beta:0.3,alpha:0.15):0.02,gamma:0.1);',
			maxSpecies: 8
		});
		expect(p.totalCodons).toBe(3);
		expect(p.speciesCount).toBe(3);
		const v = lib.validateInputBundle(p.batch(0), {
			batch: p.totalCodons,
			numSpecies: p.speciesCount,
			windowSize: p.windowSize
		});
		expect(v.errors).toEqual([]);
		// Site 0 is ATG / ATG / ATG — one residue, invariable.
		// Site 1 is TTA / CTA / TTA — three codons, all Leucine, so invariable despite the change.
		// Site 2 is TCA / TCA / AGC — a serine island: variable ONLY under the TCN/AGY rule, and the
		// case `hyphaeon/dataset.py` currently calls invariable (see variability.js's header). If the
		// fixture harness settles that question against this port, this line is one of the two that
		// has to change.
		expect(lib.siteVariability(['ATGTTATCA', 'ATGCTATCA', 'ATGTTAAGC'], 3)).toEqual([
			false,
			false,
			true
		]);
	});
});

/**
 * IN A REAL NODE PROCESS, NOT IN VITEST'S MODULE RUNNER. Vite resolves specifiers itself and reports
 * a blocked deep import as a transform-time failure of the whole test file, which cannot be caught
 * with `rejects` and takes the rest of the suite down with it. What consumers actually run is Node's
 * own resolver, so that is what is exercised here: one child process, `--input-type=module`, the
 * package resolved by name through the same `exports` map npm will publish.
 */
const packageDir = fileURLToPath(new URL('..', import.meta.url));

/** @returns {{ok: boolean, code: string}} the child's verdict on one specifier. */
function resolveInNode(specifier) {
	const script = `import(${JSON.stringify(specifier)}).then(
		() => console.log(JSON.stringify({ ok: true, code: '' })),
		(e) => console.log(JSON.stringify({ ok: false, code: e.code ?? String(e) }))
	);`;
	const out = execFileSync(process.execPath, ['--input-type=module', '-e', script], {
		cwd: packageDir,
		encoding: 'utf8'
	});
	return JSON.parse(out);
}

describe('the exports map, resolved by Node itself', () => {
	it('serves the package by name', () => {
		// Self-reference resolution: this only works if `name` and `exports` agree and the entry file
		// is where the map says. A consumer's `import '@veg/hyphaeon-js'` takes the identical path.
		expect(resolveInNode('@veg/hyphaeon-js')).toEqual({ ok: true, code: '' });
	});

	it('publishes "." only, so internal paths cannot be pinned by a consumer', () => {
		// The layout under src/ must stay a private matter — a deep import that works once becomes an
		// API nobody agreed to. Asserting the error CODE rather than its message keeps this stable
		// across Node versions.
		expect(resolveInNode('@veg/hyphaeon-js/src/preprocess/newick.js')).toEqual({
			ok: false,
			code: 'ERR_PACKAGE_PATH_NOT_EXPORTED'
		});
	});
});
