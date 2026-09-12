/**
 * WHY THIS FILE EXISTS
 *
 * The numeric kernel PLAN.md §5.1 calls out ("built once, tested against scipy fixtures"): the
 * special functions, PRNG, ranks, linear algebra, the graph primitives with networkx's ordering and
 * tie-breaking, and the two stats.py reductions that every port from stats.js onward imports. Ports code against the names re-exported here, never against the
 * files, so the kernel's layout can change without touching a port. src/index.js re-exports this
 * module in turn (the integrator's file).
 *
 * Nothing here is model-, method- or I/O-aware; see each module's header for what it mirrors.
 */

export * from './reduce.js'; // numpyPairwiseSum, float32Sum, numpyMeanFloat32, percentile (numpy reduction and quantile semantics)
export * from './special.js'; // lgamma, gammaincReg, gammaincc, betaincReg, erfc, chi2Sf, chi2Cdf, tSf, tCdf, normSf, normCdf, logChoose, hypergeomPmf, hypergeomCdf, hypergeomSf
export * from './prng.js'; // Xoshiro256
export * from './ranks.js'; // rankdata, pearson, spearman, rocAuc
export * from './bh.js'; // benjaminiHochberg
export * from './cauchy.js'; // cauchyCombination
export * from './linalg.js'; // cholesky, symmetricEigenvalues, largestEigenvalue, symmetricEigen (re-export)
export * from './calculus.js'; // numpyLinspace, numpyGradient(Rows), numpyTrapezoid(Rows), isUniformSpacing (numpy's own grid, derivative and quadrature)
export * from './svd.js'; // dominantTimeModes, timeGram, projectionResidual, canonicalizeWaveSigns, resolveWaveSign, WAVE_* (the thin time-side decomposition temporal.py needs, D28's sign convention)
export * from './optimize.js'; // brentq (scipy.optimize's bracketing root finder, transcribed from Zeros/brentq.c), fminbound (its bounded scalar minimiser)
export * from './graph.js'; // networkx 3.6.1 semantics: adjacency, subgraph views, components, greedy modularity, cpythonIntSetOrder
