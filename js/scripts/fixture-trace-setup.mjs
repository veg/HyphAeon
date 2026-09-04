/**
 * WHY THIS FILE EXISTS
 *
 * The vitest setup file that fixture-coverage.mjs loads through fixture-coverage.vitest.config.mjs
 * to record which fixture files the suite actually reads. It wraps `fs.readFileSync` in every test
 * worker and appends "<path relative to FIXTURE_TRACE_ROOT>\t<test file>\t<running test title>"
 * to FIXTURE_TRACE for each read under the root. Node builtins are externalised by vite-node, so
 * patching the CJS module object and calling `syncBuiltinESMExports` makes
 * `import { readFileSync } from 'node:fs'` in the tests see the wrapper; the test title comes from
 * vitest's own `expect.getState()`. Not loaded by `npm test`; only the coverage script sets the two
 * variables.
 */
import fs from 'node:fs';
import { expect } from 'vitest';
import { syncBuiltinESMExports } from 'node:module';
import { relative, isAbsolute } from 'node:path';

const trace = process.env.FIXTURE_TRACE;
const root = process.env.FIXTURE_TRACE_ROOT;

if (trace && root) {
	const original = fs.readFileSync;
	const append = fs.appendFileSync;
	/** The first frame on the stack that is a test file, or ''. */
	const callingTest = () => {
		const stack = new Error().stack ?? '';
		const m = stack.match(/[/\\]test[/\\]([A-Za-z0-9_.-]+\.test\.js)/);
		return m ? m[1] : '';
	};
	fs.readFileSync = function tracedReadFileSync(path, ...rest) {
		if (typeof path === 'string' || path instanceof URL) {
			const p = path instanceof URL ? path.pathname : path;
			if (isAbsolute(p)) {
				const rel = relative(root, p);
				if (rel && !rel.startsWith('..') && !isAbsolute(rel)) append(trace, `${rel}\t${callingTest()}\t${(expect.getState().currentTestName ?? '').replace(/\s+/g, ' ')}\n`);
			}
		}
		return original.call(this, path, ...rest);
	};
	syncBuiltinESMExports();
}
