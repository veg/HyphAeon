/**
 * WHY THIS FILE EXISTS
 *
 * The numeric kernel PLAN.md §5.1 calls out ("built once, tested against scipy fixtures"): the
 * special functions, PRNG, ranks, linear algebra and the two stats.py reductions that every port
 * from stats.js onward imports. Ports code against the names re-exported here, never against the
 * files, so the kernel's layout can change without touching a port. src/index.js re-exports this
 * module in turn (the integrator's file).
 *
 * Nothing here is model-, method- or I/O-aware; see each module's header for what it mirrors.
 */

export * from './reduce.js'; // numpyPairwiseSum, float32Sum, numpyMeanFloat32 (numpy reduction order)
export * from './special.js'; // lgamma, gammaincReg, gammaincc, betaincReg, erfc, chi2Sf, chi2Cdf, tSf, tCdf, normSf, normCdf, logChoose, hypergeomPmf, hypergeomCdf, hypergeomSf
export * from './prng.js'; // Xoshiro256
export * from './ranks.js'; // rankdata, pearson, spearman, rocAuc
export * from './bh.js'; // benjaminiHochberg
export * from './cauchy.js'; // cauchyCombination
export * from './linalg.js'; // cholesky, symmetricEigenvalues, largestEigenvalue, symmetricEigen (re-export)
