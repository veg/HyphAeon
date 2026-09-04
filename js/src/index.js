/**
 * WHY THIS FILE EXISTS
 *
 * The one entry point package.json's `exports` map exposes, so `@veg/hyphaeon-js` has exactly one
 * public surface and every consumer — the browser workers, the MCP server, the Node runtime — reads
 * the same names. A deep import into `src/preprocess/…` would work today and pin an internal path
 * forever; the export map makes that a resolution error instead of a maintenance debt.
 *
 * WHY `export *` RATHER THAN A HAND-WRITTEN LIST. A curated list is a second place to update, and
 * the failure when someone forgets is silent: a function ported, tested and unreachable, which then
 * gets ported a second time in the app repository because it "wasn't in the library". Re-exporting
 * whole modules means what a module exports IS what the package exports, and the decision about
 * whether something is public happens once, in the module, where the reasoning already lives.
 *
 * The corollary is that adding an export to any file under src/preprocess/ publishes it. That is the
 * intended contract — see src/README.md for what is allowed in here in the first place, which is the
 * rule doing the real work.
 *
 * ORDER IS DEPENDENCY ORDER, leaves first, and there are no name collisions across these eight
 * modules: `vitest run` fails on a duplicate re-export, so a collision cannot reach a release.
 */

// Leaves — no imports of their own.
export * from './preprocess/modelContract.js'; // vocabulary constants, INPUT/OUTPUT specs, validateInputBundle
export * from './preprocess/symmetricEigen.js'; // symmetricEigen
export * from './preprocess/newick.js'; // parseNewick, leafIndex, normalizeTaxonName

// One level up.
export * from './preprocess/tokenizer.js'; // CODON_LIST, GENETIC_CODE, codonToken, aaToken, tokenizeSequence
export * from './preprocess/patristic.js'; // rootDistances, patristicRow, patristicMatrix, maxPdSelect
export * from './preprocess/mds.js'; // computeMdsCoordinates
export * from './preprocess/variability.js'; // isSiteVariable, siteVariability

// The joining layer.
export * from './preprocess/assemble.js'; // prepareAlignment, chooseReference, orderSpecies, batchSizeFor
