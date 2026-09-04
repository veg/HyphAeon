#!/usr/bin/env node
/**
 * WHY THIS FILE EXISTS
 *
 * PLAN.md §5.3 rule 2 says every ported function gets a fixture replay, and the failure mode when
 * that slips is silent: a fixture file sits in fixtures/ and nothing reads it, so a Python change
 * regenerates it and no test notices. This script lists every fixtures/<module>/<function>.json
 * and says which test files actually READ it, printing the uncovered remainder, so the gap is a
 * line in a report rather than a discovery. It is a development tool, not part of the library
 * (js/scripts/ is outside package.json "files"), which is why it may use node:fs and spawn
 * processes.
 *
 * WHY A TRACE AND NOT A TEXT SCAN. The tests name fixtures in every shape — `'dataset/x.json'`,
 * `e2e/busted_${gene}.json`, a list of stems fed to a template — and a static scan either misses
 * the templates or, treating `${}` as a wildcard, credits every test file with every fixture (the
 * first version of this script did the latter). So the suite is run once with a setup file
 * (fixture-trace-setup.mjs) that wraps `fs.readFileSync` and appends each read under fixtures/
 * to a trace, together with the test file on the call stack (vitest takes setup files only from
 * a config, hence fixture-coverage.vitest.config.mjs). A "covered" verdict then means a test
 * opened the file while running; an "uncovered" one means none did.
 *
 * ONE READ IS NOT A REPLAY. fixtures.test.js's manifest block reads EVERY fixture file to assert
 * that no absolute path leaked into it, which would mark everything covered. The trace therefore
 * also carries the running test's title, and reads made under a title that names
 * `fixtures/manifest.json` are reported separately as "manifest check only" and count as
 * uncovered. Nothing else is filtered.
 *
 * Usage: node scripts/fixture-coverage.mjs [--json]      (exit code 1 when anything is uncovered)
 *        FIXTURE_TRACE=<file> to keep the raw trace.
 */
import { readdirSync, readFileSync, statSync, rmSync, existsSync, mkdtempSync } from 'node:fs';
import { join, relative } from 'node:path';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const PKG = join(HERE, '..');
const ROOT = join(PKG, '..');
const FIXTURES = join(ROOT, 'fixtures');

/** @param {string} dir @returns {string[]} */
function walk(dir) {
	const out = [];
	for (const name of readdirSync(dir)) {
		const p = join(dir, name);
		if (statSync(p).isDirectory()) out.push(...walk(p));
		else if (p.endsWith('.json')) out.push(p);
	}
	return out.sort();
}

/** Every fixture case file: fixtures/<module>/<function>.json, excluding inputs/ and the manifest. */
const fixtureFiles = walk(FIXTURES)
	.map((p) => relative(FIXTURES, p))
	.filter((rel) => !rel.includes('/inputs/') && rel !== 'manifest.json');

// Run the suite once with the tracing setup file. The trace is one "<fixture rel>\t<test file>"
// line per read; a fixture read outside any test file (e.g. from a helper at module load) is
// recorded with the test file that imported it, since that frame is on the stack too.
const keep = process.env.FIXTURE_TRACE;
const traceDir = keep ? null : mkdtempSync(join(tmpdir(), 'fixture-trace-'));
const trace = keep ?? join(traceDir, 'trace.tsv');
if (existsSync(trace)) rmSync(trace);
const run = spawnSync(
	process.execPath,
	[join(PKG, 'node_modules', 'vitest', 'vitest.mjs'), 'run', '--reporter=dot', '--config', join(HERE, 'fixture-coverage.vitest.config.mjs')],
	{ cwd: PKG, env: { ...process.env, FIXTURE_TRACE: trace, FIXTURE_TRACE_ROOT: FIXTURES }, encoding: 'utf8' }
);
if (run.status !== 0) {
	process.stderr.write(run.stdout + run.stderr);
	console.error('vitest failed; coverage not computed');
	process.exit(run.status ?? 1);
}

/** @type {Map<string, Set<string>>} */
const readers = new Map(fixtureFiles.map((f) => [f, new Set()]));
const lines = existsSync(trace) ? readFileSync(trace, 'utf8').split('\n').filter(Boolean) : [];
/** @type {Set<string>} fixtures touched only by the manifest sanity check */
const manifestOnly = new Set();
for (const line of lines) {
	const [rel, test, title] = line.split('\t');
	if (!readers.has(rel)) continue;
	if ((title ?? '').includes('fixtures/manifest.json')) manifestOnly.add(rel);
	else readers.get(rel).add(test || '(no test file on the stack)');
}
if (traceDir) rmSync(traceDir, { recursive: true, force: true });

const rows = fixtureFiles.map((f) => ({ fixture: f, tests: [...readers.get(f)].sort() }));
const uncovered = rows.filter((r) => r.tests.length === 0);

if (process.argv.includes('--json')) {
	console.log(JSON.stringify({ rows, uncovered: uncovered.map((r) => r.fixture) }, null, 2));
} else {
	const width = Math.max(...rows.map((r) => r.fixture.length));
	for (const r of rows) {
		const status = r.tests.length ? r.tests.join(', ') : manifestOnly.has(r.fixture) ? '-- manifest check only --' : '-- UNCOVERED --';
		console.log(`${r.fixture.padEnd(width)}  ${status}`);
	}
	console.log(`\n${rows.length - uncovered.length}/${rows.length} fixture files read by a test`);
	if (uncovered.length) {
		console.log('\nUncovered:');
		for (const r of uncovered) console.log(`  ${r.fixture}`);
	}
}
process.exitCode = uncovered.length ? 1 : 0;
