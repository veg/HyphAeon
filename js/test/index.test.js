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
 *      between two `export *` lines (two DIFFERENT bindings under one name are silently dropped
 *      from the namespace by the ES module linker; only a local declaration makes it an error)
 *      — would leave the whole suite green and break every consumer on install. This file is the
 *      only place that loads the package the way an app does.
 *   2. THE PUBLIC SURFACE IS A COMMITMENT. `index.js` re-exports whole modules deliberately (its
 *      header says why), which means adding an export to any file under `src/preprocess/` publishes
 *      it, and deleting one breaks a pinned consumer. Listing all 368 names here turns both into a
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
import * as symmetricEigenModule from '../src/preprocess/symmetricEigen.js';
import * as bh from '../src/numeric/bh.js';
import * as ranks from '../src/numeric/ranks.js';
import * as stats from '../src/stats.js';
import * as evaluate from '../src/evaluate.js';
import * as omnibus from '../src/omnibus.js';
import * as tn93 from '../src/preprocess/tn93.js';
import * as phenotype from '../src/phenotype.js';
import * as permulations from '../src/permulations.js';

/**
 * Every public symbol of the package, sorted. Grouped by module (first module to claim a name
 * wins; symmetricEigen, benjaminiHochberg and rocAuc are also re-exported by numeric/index.js,
 * stats.js and evaluate.js as the same bindings) for reviewable diffs. Regenerate with the same
 * loop over `Object.keys(await import(module))` when a module adds or removes an export.
 */
const PUBLIC_SURFACE = [
	'AA_GAP',
	'AA_LIST',
	'AA_STOP',
	'AA_UNKNOWN',
	'AA_VALID_BELOW',
	'CODON_GAP',
	'CODON_ORDER',
	'CODON_STOP',
	'CODON_UNKNOWN',
	'CODON_VALID_BELOW',
	'INPUT_NAMES',
	'INPUT_SPEC',
	'MAX_SPECIES_CAP',
	'MAX_SPECIES_DEFAULT',
	'MDS_COMPONENTS',
	'NUM_AA_TOKENS',
	'NUM_CODON_TOKENS',
	'OUTPUT_NAMES_V1',
	'OUTPUT_SPEC',
	'OUTPUT_SPEC_V1',
	'TAXA_OUTPUT_NAMES',
	'TAXA_OUTPUT_SPEC',
	'VERIFIED_MODEL_SHA256',
	'WINDOW_SIZE_DEFAULT',
	'taxaOutputDivisors',
	'validateInputBundle',
	// preprocess/symmetricEigen.js
	'symmetricEigen',
	// preprocess/parse.js
	'PY_NOT_WS',
	'PY_WS',
	'parseAlignmentSequences',
	'pyIsAlnum',
	'pyIsDigit',
	'pyLen',
	'pyLstrip',
	'pyRstrip',
	'pySplit',
	'pySplit1',
	'pySplitLines',
	'pyStrip',
	'pyStripChars',
	// preprocess/tokenizer.js
	'AA_MAP',
	'CODON_LIST',
	'CODON_TO_AA',
	'GENETIC_CODE',
	'aaToken',
	'codonToken',
	'tokenizeSequence',
	// preprocess/variability.js
	'invariableMask',
	'isAaInvariable',
	'isSiteVariable',
	'siteVariability',
	// preprocess/tree.js
	'NewickError',
	'enforceNonzeroBranchLengths',
	'extractTree',
	'findClades',
	'getTerminals',
	'hasNonzeroBranchLengths',
	'matchTaxa',
	'needsBranchLengths',
	'parseNewickTrees',
	'readNewick',
	'stripQuotes',
	'treeTaxa',
	// preprocess/tn93.js
	'tn93CrossDistanceMatrix',
	'TN93_FALLBACK_MAX',
	'TN93_MATCH_MODE',
	'TN93_MAX_AMBIG_FRACTION',
	'TN93_MIN_POSITIVE_DISTANCE',
	'TN93_SATURATION_SENTINEL',
	'TN93_TABLES',
	'ambigFractionTooHigh',
	'canResolve',
	'encodeSequence',
	'tn93CalculateDistance',
	'tn93Counts',
	'tn93Distance',
	'tn93DistanceMatrix',
	'tn93NucleotideFrequency',
	'tn93SaturatedPairs',
	// preprocess/consensus.js
	'CONSENSUS_EMPTY_CHAR',
	'CONSENSUS_SKIP',
	'consensusSequence',
	'timeDecayConsensusSequence',
	// preprocess/patristic.js
	'computeFastDistMatrix',
	'patristicMatrix',
	'patristicRow',
	'rescaleDistances',
	'rootDistances',
	// preprocess/downsample.js
	'downsampleTaxaFaithPd',
	'pruneIdenticalSequences',
	'stridePreselect',
	// preprocess/mds.js
	'computeMdsCoordinates',
	// preprocess/assemble.js
	'batchSizeFor',
	'loadAlignmentAndTree',
	'siteBatch',
	'siteBatches',
	// numeric/reduce.js
	'percentile',
	'float32Sum',
	'numpyMeanFloat32',
	'numpyPairwiseSum',
	// numeric/special.js
	'fCdf',
	'fSf',
	'normPpf',
	'tPpf',
	'betaincReg',
	'chi2Cdf',
	'chi2Sf',
	'erfc',
	'gammaincReg',
	'gammaincc',
	'hypergeomCdf',
	'hypergeomPmf',
	'hypergeomSf',
	'lgamma',
	'logChoose',
	'normCdf',
	'normSf',
	'tCdf',
	'tSf',
	// numeric/optimize.js
	'BRENTQ_MAXITER',
	'BRENTQ_RTOL',
	'BRENTQ_XTOL',
	'FMINBOUND_MAXITER',
	'FMINBOUND_XATOL',
	'brentq',
	'fminbound',
	// numeric/prng.js
	'Xoshiro256',
	// numeric/ranks.js
	'pearson',
	'rankdata',
	'rocAuc',
	'spearman',
	// numeric/bh.js
	'benjaminiHochberg',
	// numeric/cauchy.js
	'cauchyCombination',
	// numeric/linalg.js
	'cholesky',
	'largestEigenvalue',
	'symmetricEigenvalues',
	// numeric/graph.js
	'adjacencyFromGraph',
	'connectedComponents',
	'cpythonIntSetOrder',
	'degree',
	'graphSize',
	'greedyModularityCommunities',
	'inducedSubgraph',
	'numberOfEdges',
	// stats.js
	'cauchyCombinationP',
	'memeSitePq',
	'pvalsFromLrtMeme',
	'pvalsFromLrtSelfLiang',
	// writers.js
	'PY_FLOAT_KEYS',
	'PY_INT_KEYS',
	'bustedCsv',
	'bustedJson',
	'dataFrameCsv',
	'dmsCsv',
	'epistasisCsv',
	'evaluateJson',
	'graphml',
	'memeCsv',
	'memeJson',
	'memeResult',
	'memeSiteRecords',
	'phenotypeCsv',
	'pyFloatRepr',
	'pyFormatFixed',
	'pyFormatG',
	'pyJsonDumps',
	'pyRepr',
	'pyStr',
	'resultJson',
	// evaluate.js
	'EvaluationError',
	'confusion',
	'correlations',
	'dictReaderRows',
	'evaluateDirectories',
	'evaluateFiles',
	'evaluatePairs',
	'formatTextReport',
	'fpr',
	'loadMemeJson',
	'loadPredictionCsv',
	'matchGeneFiles',
	'normalizedHeader',
	'parseCsvRows',
	'ppv',
	'thresholdMetrics',
	// filter.js
	'auditPatch',
	'consensusCodons',
	'fastaText',
	'maskCodonSpan',
	'predictSiteLrts',
	'runAlignmentFilter',
	'scanHypergeometricPatches',
	// attribution.js
	'INV_GENETIC_CODE',
	'attributeSelection',
	'attributionSiteFields',
	'attributionsOneIndexed',
	// omnibus.js
	'BUSTED_ACAT_ALPHA',
	'BUSTED_EMBED_DIM',
	'BUSTED_HEAD_INPUT_NAMES',
	'BUSTED_HEAD_OUTPUT_NAMES',
	'BUSTED_LRT_THRESHOLD',
	'BUSTED_OMEGA_1',
	'BUSTED_OMEGA_2',
	'BUSTED_PROB_THRESHOLD',
	'BUSTED_SIMES_FLOOR',
	'bustedHeadFields',
	'bustedRecord',
	'bustedStatistics',
	'bustedVerdict',
	'gatherSiteBatch',
	'omnibusLrt',
	'runBusted',
	'simesP',
	'totalSelectionEnergy',
	'variableSiteIndices',
	// epistasis.js
	'CoselectionGraph',
	'computeBranchCoselectionNetwork',
	'computeTransformerAttributions',
	'consensusDelta',
	'runTransformerAttributions',
	// sectors.js
	'REV_AA_MAP',
	'computeSectorPermutationTest',
	'extractEpistaticSectorsTse',
	'float32Percentile',
	// dms.js
	'CANONICAL_AA_TO_CODON',
	'DMS_MUTANTS_PER_SITE',
	'digitalDmsRecord',
	'dmsTargetSites',
	'dmsWildTypeAa',
	'resolveFocalTaxon',
	'runDigitalDmsAnalysis',
	'runInsilicoSelectionDms',
	// permulations.js
	'computePhylogeneticCovariance',
	'findAnyByName',
	'generatePermulations',
	'pyRegexSource',
	// phenotype.js
	'PHENOTYPE_THRESHOLDS',
	'PRESETS',
	'parsePhenotypeTable',
	'pyFloatStr',
	'pyFnmatch',
	'pyReprString',
	'pyReprStringList',
	'resolvePhenotypeVector',
	'runPhenotypeAssociation',
	// diagnostics.js
	'DIAGNOSTIC_CODES',
	'DIAGNOSTIC_THRESHOLDS',
	'WORK_PER_SECOND',
	'diagnose',
	'meanPairwiseDivergence',
	'medianOffDiagonal',
	'sniffAlignmentFormat',
	// dating.js
	'CI_METHODS_PORTED',
	'CI_METHODS_UNPORTED',
	'DATING_STATUS',
	'EARLIEST_ALIASES',
	'FIELLER_STATUS',
	'SYNTHETIC_CONSENSUS_KEY',
	'TIME_DECAY_ROOT_KEY',
	'UNWEIGHTED_CONSENSUS_ALIASES',
	'clockFittedAndPredicted',
	'computeDeltaMrcaInterval',
	'computeFiellerMrcaInterval',
	'computeRcsBasis',
	'computeTreeFreeDivergences',
	'precisionWeightedEnsemble',
	'residualScale',
	'runOlsDating',
	'runRestrictedSplineClockDating',
	// datingModel.js
	'ADAM_DEFAULTS',
	'LATENT_ROOT_DEFAULTS',
	'LATENT_TRAJECTORY_STEPS',
	'PAGEL_LAMBDA_BOUNDS',
	'PGLS_DENSE_EIGEN_MAX',
	'REML_DENSE_MAX',
	'REML_STATUS',
	'computeNeuralCovarianceKernel',
	'estimateRemlPagelLambda',
	'optimizeLatentConvexHullRoot',
	'pairwiseAcgtHammingMatrix',
	'runPglsDating',
	// dates.js
	'ARCHIVAL_1959_DECIMAL',
	'ARCHIVAL_1959_MARKERS',
	'CALENDAR_TIME_UNITS',
	'CALENDAR_YEAR_MAX',
	'CALENDAR_YEAR_MIN',
	'DATE_RULES',
	'FEBRUARY_DAY_CAP',
	'IMPUTED_DAY',
	'IMPUTED_MONTH',
	'KORBER_YEAR_PIVOT',
	'MONTH_DAY_CAP',
	'NON_CALENDAR_TIME_UNITS',
	'NULL_DATE_TOKENS',
	'TIME_UNITS',
	'extractDate',
	'extractDateFromString',
	'isNonCalendarTimeUnits',
	'matchesArchival1959',
	'parseDate',
	'parseDateToDecimal',
	'parseFlexibleDate',
	'parseHeaderDate',
	'parseHeaderTimestamp',
	// numeric/calculus.js (numpy's linspace, gradient and trapezoid)
	'isUniformSpacing',
	'numpyGradient',
	'numpyGradientRows',
	'numpyLinspace',
	'numpyTrapezoid',
	'numpyTrapezoidRows',
	// numeric/reduce.js (the float64 reductions temporal.py needs)
	'numpyMeanFloat64',
	'numpyStdFloat64',
	'numpyVarFloat64',
	// numeric/svd.js (the thin time-side decomposition and D28's wave-sign convention)
	'WAVE_GAP_THRESHOLD',
	'WAVE_RANK_EPS',
	'WAVE_SIGN_MODES',
	'canonicalizeWaveSigns',
	'dominantTimeModes',
	'projectionResidual',
	'resolveWaveSign',
	'timeGram',
	// temporal.js
	'TEMPORAL_CLASSES',
	'TEMPORAL_CROSS_CLASSES',
	'TEMPORAL_ESCAPE_LRT',
	'TEMPORAL_ESCAPE_P',
	'TEMPORAL_NON_CALENDAR_UNITS',
	'TEMPORAL_Q_STATIC_CUT',
	'TEMPORAL_UNIT_LABELS',
	'TEMPORAL_UNKNOWN_AA',
	'TEMPORAL_WAVE_K',
	'classifyTemporalSites',
	'confirmSweeps',
	'directionalAttribution',
	'fpcaShapeGate',
	'inferRootSequence',
	'nadarayaWatsonWeights',
	'resolveEnergyFloors',
	'resolveTemporalBandwidth',
	'resolveTemporalRegime',
	'rootConsensusWindow',
	'rowStandardise',
	'smoothTrajectories',
	'stableArgsortAscending',
	'stableArgsortDescending',
	'stageOneMask',
	'sweepMetric',
	'taxonMajorWeights',
	'temporalDrawPermutation',
	'temporalMutationLabels',
	'temporalNullDraws',
	'temporalPermPValues',
	'temporalPermStat',
	'temporalTimeGrid',
	'temporalTrajectoryStatistics',
	'temporalWaveDecomposition',
	// writers.js (the temporal pillar's four output files and the float32 repr they need)
	'TEMPORAL_CURVES_COLUMNS',
	'TEMPORAL_SITES_COLUMNS',
	'TEMPORAL_SUMMARY_FLOAT_KEYS',
	'TEMPORAL_SUMMARY_INT_KEYS',
	'TEMPORAL_SUMMARY_KEYS',
	'TEMPORAL_WAVES_COLUMNS',
	'pyFloat32Repr',
	'pyRound',
	'temporalCurvesCsv',
	'temporalSitesCsv',
	'temporalSummaryJson',
	'temporalWavesCsv',
	'parseTimestampFlexible',
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
		expect(lib.loadAlignmentAndTree).toBe(assemble.loadAlignmentAndTree);
		expect(lib.isSiteVariable).toBe(variability.isSiteVariable);
		expect(lib.runBusted).toBe(omnibus.runBusted);
		// Phase 3a: the tree-free distances and the phenotype pillar reach the barrel unwrapped too.
		expect(lib.tn93DistanceMatrix).toBe(tn93.tn93DistanceMatrix);
		expect(lib.runPhenotypeAssociation).toBe(phenotype.runPhenotypeAssociation);
		expect(lib.generatePermulations).toBe(permulations.generatePermulations);
	});

	it('resolves the deliberately shared names to the kernel binding', () => {
		// Three names are exported by two modules each. The ES linker keeps a name only when every
		// `export *` that supplies it supplies the SAME binding, so these three being present at all
		// proves the re-exports are aliases and not copies. The identity checks make that explicit.
		expect(lib.symmetricEigen).toBe(symmetricEigenModule.symmetricEigen);
		expect(lib.benjaminiHochberg).toBe(bh.benjaminiHochberg);
		expect(lib.benjaminiHochberg).toBe(stats.benjaminiHochberg);
		expect(lib.rocAuc).toBe(ranks.rocAuc);
		expect(lib.rocAuc).toBe(evaluate.rocAuc);
	});

	it('runs the preprocessing pipeline through the entry point alone', () => {
		// The smallest end-to-end proof that the barrel is wired: parse, match, tokenise, distance,
		// MDS, the invariable mask, one site batch and validate, touching nothing but `lib`. Any
		// missing re-export shows up here as a TypeError rather than as an undefined that quietly
		// becomes NaN downstream.
		const loaded = lib.loadAlignmentAndTree(
			'>alpha\nATGTTATCA\n>beta\nATGCTATCA\n>gamma\nATGTTAAGC\n',
			'((beta:0.3,alpha:0.15):0.02,gamma:0.1);',
			{ maxSpecies: 8 }
		);
		expect(loaded.L).toBe(3);
		expect(loaded.N).toBe(3);
		const bundle = lib.siteBatch(loaded, 0, loaded.L);
		const v = lib.validateInputBundle(bundle, { batch: loaded.L, numSpecies: loaded.N });
		expect(v.errors).toEqual([]);
		// Site 0 is ATG / ATG / ATG — one residue, invariable.
		// Site 1 is TTA / CTA / TTA — three codons, all Leucine, so invariable despite the change.
		// Site 2 is TCA / TCA / AGC — a serine island: two codon families, ONE amino acid, which
		// dataset.py:1077-1083 calls invariable (no TCN/AGY rule; the fixture harness settled this
		// against the DataMonkey 3 port, PLAN.md §5.3).
		expect(Array.from(loaded.invariable)).toEqual([1, 1, 1]);
		expect(lib.siteVariability(['ATGTTATCA', 'ATGCTATCA', 'ATGTTAAGC'], 3)).toEqual([
			false,
			false,
			false
		]);
		// And one statistic through the method layer: an all-zero LRT vector is p = 2/3 everywhere.
		expect(Array.from(lib.pvalsFromLrtMeme(new Float32Array(3)))).toEqual([2 / 3, 2 / 3, 2 / 3]);
		// D22: the same alignment with NO tree goes tree-free through the barrel — TN93 distances,
		// alignment order, the tree slot null — rather than throwing as it did before Phase 3a.
		const treeFree = lib.loadAlignmentAndTree('>alpha\nATGTTATCA\n>beta\nATGCTATCA\n>gamma\nATGTTAAGC\n', null, {
			maxSpecies: 8
		});
		expect(treeFree.tree).toBeNull();
		expect(treeFree.notices.treeFree).toEqual({ reason: 'no_tree', taxaOrder: 'alignment' });
		expect(treeFree.taxa).toEqual(['alpha', 'beta', 'gamma']);
		expect(treeFree.d[1]).toBe(Math.fround(lib.tn93Distance('ATGTTATCA', 'ATGCTATCA')));
		expect(treeFree.d[1]).toBeGreaterThan(0);
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
		expect(resolveInNode('@veg/hyphaeon-js/src/preprocess/tree.js')).toEqual({
			ok: false,
			code: 'ERR_PACKAGE_PATH_NOT_EXPORTED'
		});
	});
});
