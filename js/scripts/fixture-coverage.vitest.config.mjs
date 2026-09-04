/**
 * WHY THIS FILE EXISTS
 *
 * vitest takes setup files from a config, not from a CLI flag. This is the one-line config
 * fixture-coverage.mjs passes with `--config` so the tracing setup file loads in every worker;
 * `npm test` does not use it and the package has no default vitest config on purpose.
 */
import { fileURLToPath } from 'node:url';

const PKG = fileURLToPath(new URL('..', import.meta.url));

export default {
	root: PKG,
	test: {
		setupFiles: [fileURLToPath(new URL('fixture-trace-setup.mjs', import.meta.url))]
	}
};
