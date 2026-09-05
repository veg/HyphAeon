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
 * The corollary is that adding an export to any file listed here publishes it. That is the
 * intended contract — see src/README.md for what is allowed in here in the first place, which is the
 * rule doing the real work — and test/index.test.js pins the resulting surface name by name.
 *
 * ONE PYTHON FILE CAN BE SEVERAL MODULES. `hyphaeon/epistasis.py` is 776 lines and three unrelated
 * jobs, so PLAN.md §5.1 splits it the way it is used: `epistasis.js` (attributions and the
 * co-selection network, lines 51-222), `sectors.js` (the permutation null and sector mining,
 * 224-445) and `dms.js` (the in-silico DMS sweep, 446-628 and 724-768). The networkx primitives the
 * middle one needs are kernel code, not method code — every consumer of a graph wants the same
 * ordering and tie-breaking — so they sit in `numeric/graph.js` and reach here through
 * `numeric/index.js`. Each module's header names its own line range.
 *
 * ORDER IS DEPENDENCY ORDER, leaves first. There are no name collisions across the modules:
 * ES module linking treats two `export *` lines that export the SAME binding under one name as one
 * export (symmetricEigen, benjaminiHochberg and rocAuc are re-exported from the kernel by
 * numeric/index.js, stats.js and evaluate.js respectively, deliberately, so a reader of stats.py or
 * evaluation.py finds every function of that file in its port), and two DIFFERENT bindings under
 * one name are silently dropped from the namespace — which is why test/index.test.js asserts the
 * whole sorted surface rather than trusting the linker.
 *
 * THE CALLBACK CONVENTION. Nothing here loads a model. Functions that need the network
 * (predictSiteLrts, runAlignmentFilter, attributeSelection, runBusted) take an async
 * `predict(c, a, meta)` callback and apply the reference's clamp and float32 storage to what it
 * returns; the app's runtime/ supplies the onnxruntime session behind it (PLAN.md §5.5).
 */

// ---- preprocess/: hyphaeon/dataset.py ---------------------------------------------------------
// Leaves — no imports of their own.
export * from './preprocess/modelContract.js'; // vocabulary sentinels, INPUT/OUTPUT specs, MAX_SPECIES_*, validateInputBundle
export * from './preprocess/symmetricEigen.js'; // symmetricEigen (tred2/tql2)
export * from './preprocess/parse.js'; // parseAlignmentSequences + the CPython str helpers (pyStrip, pySplit, ...)
export * from './preprocess/tokenizer.js'; // CODON_LIST, GENETIC_CODE, AA_MAP, CODON_TO_AA, codonToken, aaToken, tokenizeSequence
export * from './preprocess/variability.js'; // isAaInvariable, invariableMask, isSiteVariable, siteVariability
export * from './preprocess/tree.js'; // NewickError, parseNewickTrees, readNewick, extractTree, matchTaxa, treeTaxa, branch-length predicates
// One level up.
export * from './preprocess/patristic.js'; // rootDistances, patristicRow, patristicMatrix, computeFastDistMatrix, rescaleDistances
export * from './preprocess/downsample.js'; // pruneIdenticalSequences, downsampleTaxaFaithPd, stridePreselect
export * from './preprocess/mds.js'; // computeMdsCoordinates
// The joining layer.
export * from './preprocess/assemble.js'; // loadAlignmentAndTree, siteBatch, siteBatches, batchSizeFor

// ---- numeric/: the kernel (PLAN.md §5.1) --------------------------------------------------------
export * from './numeric/index.js'; // special functions, Xoshiro256, ranks, BH, CCT, linalg, graph (networkx semantics), numpy reductions

// ---- method modules, one per hyphaeon/*.py ------------------------------------------------------
export * from './stats.js'; // pvalsFromLrtMeme, pvalsFromLrtSelfLiang, memeSitePq, cauchyCombinationP (+ benjaminiHochberg)
export * from './writers.js'; // Python-byte-equal JSON/CSV/GraphML writers and the py* formatting helpers (leaf)
export * from './evaluate.js'; // evaluation.py: loaders, matching, correlations, confusion, evaluatePairs/Files/Directories, formatTextReport
export * from './filter.js'; // filter.py: scanHypergeometricPatches, predictSiteLrts, consensusCodons, auditPatch, runAlignmentFilter
export * from './attribution.js'; // attribution.py: attributeSelection, attributionSiteFields, attributionsOneIndexed, INV_GENETIC_CODE
export * from './omnibus.js'; // cli.py cmd_busted statistics: bustedStatistics, bustedRecord, runBusted, simesP, omnibusLrt, BUSTED_*
export * from './epistasis.js'; // epistasis.py:51-222: consensusDelta, computeTransformerAttributions, runTransformerAttributions, computeBranchCoselectionNetwork, CoselectionGraph
export * from './sectors.js'; // epistasis.py:224-445: computeSectorPermutationTest, extractEpistaticSectorsTse, REV_AA_MAP, float32Percentile
export * from './dms.js'; // epistasis.py:446-628, 724-768: runInsilicoSelectionDms, runDigitalDmsAnalysis, digitalDmsRecord, CANONICAL_AA_TO_CODON
export * from './diagnostics.js'; // PLAN.md §4.3 pre-flight: diagnose, DIAGNOSTIC_CODES, DIAGNOSTIC_THRESHOLDS
