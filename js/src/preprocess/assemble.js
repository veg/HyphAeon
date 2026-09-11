/**
 * WHY THIS FILE EXISTS
 *
 * Mirrors `load_alignment_and_tree` of `hyphaeon/dataset.py:931-1144`, end to end, on strings: the
 * tree branch (dataset.py:994-1099) followed by the shared tail (1102-1144). EVERY LINE NUMBER IN
 * THIS FILE IS THE RECONCILED dataset.py's (branch reconcile/phase-5a); earlier revisions of this
 * header quoted two commits' numbering side by side. Step by step, with the line it mirrors:
 *
 *    1. parse_alignment_sequences(alignment)                                      951-952
 *    2. tree: extract from `treeText`, or from the alignment text when null        995-1011
 *    3. has_nonzero_branch_lengths -> (HyPhy, see below) -> enforce 1e-3 / 1e-4    1013-1027
 *    4. three-tier taxon matching, TREE TERMINAL ORDER, dropped counts             1029-1056
 *    5. prune_identical_sequences when prune_duplicates and > 1 taxon              1058-1063
 *    6. unequal-length warning; L = len(first taxon's sequence) // 3;
 *       < 3 bp raises; the `% 3` remainder is reported as trimmed                  1065-1080
 *    7. stride pre-selection to the first 2 * max_species taxa                     1082-1087
 *    8. compute_fast_dist_matrix (float32)                                         1090
 *    9. `if dist_mat.max() > 10.0: dist_mat /= L`                                  1092-1095
 *   10. downsample_taxa_faith_pd when still > max_species                          1097-1099
 *   11. compute_mds_coordinates(dist, 4, mds_sign)                                 1101-1102
 *   12. tokens [L, N, 1] with unknown / in-frame-stop counts                       1104-1130
 *   13. is_aa_invariable                                                           1132-1137
 *   14. tensors: c, a int64 [L,N,1]; d float32 [1,N,N]; z float32 [1,N,4]          1139-1144
 *
 * Everything the reference PRINTS becomes a field of `notices` so the runtime can show it; nothing
 * is printed here. The tokens are returned as Int32Array (values 0..64 / 0..20) in [L, N, 1]
 * layout; `siteBatch` converts a range of sites to the BigInt64Array / Float32Array bundle the ONNX
 * graph takes, with the site-invariant `dist_matrix` and `mds_coords` repeated per site because the
 * graph needs materialised data where torch broadcast a [1, N, N] view.
 *
 * THE TREE-FREE TN93 PATH (PLAN.md D22, resolved 2026-09-05), mirroring the `use_tn93` branch of
 * `load_alignment_and_tree` (dataset.py:956-993) with the same step order:
 *
 *    1. taxa = list(seq_dict.keys())        ALIGNMENT ORDER, the tree is not consulted        958
 *    2. prune_identical_sequences                                                          959-964
 *    3. unequal-length warning; L; < 3 bp raises; `% 3` remainder reported                  966-982
 *    4. stride pre-selection to the first 2 * max_species taxa                              984-987
 *    5. compute_tn93_distance_matrix (float32) -> tn93.js                                       989
 *    6. downsample_taxa_faith_pd when still > max_species                                   991-992
 *   then the shared tail (MDS, tokens, invariable mask) exactly as the tree path.
 *
 * There is NO `> 10` rescale on this path: dataset.py:1094-1095 sits in the tree branch only, and a
 * TN93 distance cannot exceed it anyway. There is no `enforce_nonzero_branch_lengths`, no taxon
 * matching and no tree order — `notices.matchTier` is null and `notices.droppedTaxa` is zero.
 *
 * WHEN IT IS TAKEN. The reference takes it only on request (`use_tn93=True`, or `nwk_path` in
 * "tn93" / "none" / "skip"). D22 widens that: it is also taken when there is NO TREE (the reference
 * raises at dataset.py:1006-1010) and when the tree HAS NO USABLE BRANCH LENGTHS (the reference calls
 * HyPhy at 655-668, or falls back to 1e-3/1e-4 defaults that are not distances at all). Those two
 * are DIVERGENCES FROM THE REFERENCE, recorded in `notices.treeFree.reason` as 'no_tree' and
 * 'no_branch_lengths' against 'requested'; the fixtures pin the requested case, which is the one
 * Python can produce. A tree that fails to PARSE still raises, as the reference does — that is a bad
 * input, not a missing one (diagnostics.js keeps TREE_UNPARSEABLE a warn/refuse and turns the other
 * two into the info-level TREE_FREE_TN93). When a tree was supplied and parsed it is still returned
 * in `tree` so a caller can draw it, but the MODEL never sees it: its branch lengths are left
 * exactly as parsed (no `enforceNonzeroBranchLengths`), and PLAN.md D22's display topology is a
 * neighbour-joining tree on these same distances (`nj.js`), not this one.
 *
 * WHAT IT DELIBERATELY DOES NOT DO:
 *   - HyPhy. dataset.py:1014-1026 shells out to `hyphy` for a tree without branch lengths. The
 *     library cannot, and under D22 it no longer needs to: that tree goes tree-free instead of
 *     taking the reference's "HyPhy not found" branch. `notices.branchLengthsMissing` still reports
 *     the fact; `needsBranchLengths` (tree.js) is still the predicate.
 *   - The tn93 BINARY. dataset.py:738-784 prefers it; tn93.js is the Python package's algorithm,
 *     which measured identical on the bundled examples (see its header).
 *   - Neighbour joining for display (PLAN.md D22): a separate module.
 *   - Files, gzip, printing.
 *   - A reference sequence. dataset.py has none: taxa are the tree/alignment intersection in tree
 *     order and L comes from the FIRST matched taxon. `referenceName` is accepted only so a caller
 *     can find where a taxon of interest landed (`referenceIndex`, -1 if absent); it changes
 *     nothing.
 *
 * DIVERGENCE FROM THE DATAMONKEY3 PORT (main@fac1330 src/lib/services/axomeme/assemble.js), which
 * this file replaces: DM3's `prepareAlignment` (1) ran MDS on the PADDED max_species matrix — now
 * the real N; (2) moved a reference sequence ('hg'/'hg38'/'human' heuristic, else the first) to
 * index 0 and seeded Max-PD there — gone; (3) clamped negative distances — gone (the reference has
 * none after enforce_nonzero_branch_lengths); (4) lacked the > 10 rescale, duplicate pruning, the
 * stride pre-selection and the three-tier matching — all present now; (5) took `{names, sequences}`
 * — this takes the alignment text, as the reference does. `chooseReference` and `orderSpecies` are
 * removed. `batchSizeFor` is kept unchanged; the per-site `batch()` builder is kept as `siteBatch`.
 */

import { parseAlignmentSequences } from './parse.js';
import {
	extractTree,
	hasNonzeroBranchLengths,
	enforceNonzeroBranchLengths,
	treeTaxa,
	matchTaxa
} from './tree.js';
import { computeFastDistMatrix, rescaleDistances } from './patristic.js';
import { pruneIdenticalSequences, downsampleTaxaFaithPd, stridePreselect } from './downsample.js';
import { computeMdsCoordinates } from './mds.js';
import { codonToken, aaToken, CODON_TO_AA } from './tokenizer.js';
import { invariableMask } from './variability.js';
import { MDS_COMPONENTS, CODON_UNKNOWN } from './modelContract.js';
import { tn93DistanceMatrix, tn93SaturatedPairs } from './tn93.js';

/**
 * @typedef {{
 *   c: Int32Array, a: Int32Array, d: Float32Array, z: Float32Array,
 *   invariable: Uint8Array, taxa: string[], L: number, N: number,
 *   referenceIndex: number,
 *   tree: import('./tree.js').PhyloTree|null,
 *   notices: {
 *     branchLengthsMissing: boolean,
 *     treeFree: {reason: 'requested'|'no_tree'|'no_branch_lengths', taxaOrder: 'alignment'}|null,
 *     tn93SaturatedPairs: number|null,
 *     matchTier: 'exact'|'quote_stripped'|'case_insensitive'|null,
 *     droppedTaxa: {alignment: number, tree: number},
 *     duplicatesCollapsed: number,
 *     duplicateMap: Map<string, string[]>,
 *     unequalLengths: number[]|null,
 *     codonsTrimmed: number,
 *     stridePreselected: boolean,
 *     distanceRescaled: boolean,
 *     rawDistMax: number,
 *     pdSubsampled: boolean,
 *     unknownCodons: number,
 *     unknownCodonFraction: number,
 *     inFrameStops: number,
 *     internalStops: {worstTaxon: string|null, worstCount: number},
 *     totalCodons: number
 *   }
 * }} LoadedAlignment
 */

/**
 * `load_alignment_and_tree(fa_path, nwk_path, max_species, prune_duplicates)` on strings.
 *
 * @param {string} alignmentText FASTA / PHYLIP / NEXUS content
 * @param {string|null} [treeText] Newick / NEXUS tree content; null to look for a tree embedded in
 *   the alignment text; 'tn93' / 'none' / 'skip' request the tree-free path, as `nwk_path` does at
 *   dataset.py:956
 * @param {{maxSpecies?: number|null, pruneDuplicates?: boolean, referenceName?: string,
 *   useTn93?: boolean, tn93Options?: {matchMode?: string, maxAmbigFraction?: number,
 *   ignoreGaps?: boolean}}} [options]
 * @returns {LoadedAlignment}
 */
export function loadAlignmentAndTree(alignmentText, treeText = null, options = {}) {
	const { maxSpecies = null, pruneDuplicates = true, referenceName, useTn93 = false, tn93Options = {} } = options;

	// 1. Alignment.
	const seqDict = parseAlignmentSequences(alignmentText);
	if (seqDict.size === 0) {
		throw new Error('Could not parse any sequences from alignment');
	}

	// 2. Tree, and the tree-free decision (D22; see the header).
	const treeMode = treeText === null ? null : String(treeText).trim().toLowerCase();
	const modeRequestsTn93 = treeMode !== null && ['tn93', 'none', 'skip'].includes(treeMode);
	const requestedTn93 = useTn93 || modeRequestsTn93;
	/** @type {import('./tree.js').PhyloTree|null} */
	let tree = null;
	if (!modeRequestsTn93) {
		tree = treeText === null ? extractTree(alignmentText) : extractTree(treeText);
		// A tree TEXT that will not parse is an error on every path, as it is in the reference
		// (dataset.py:1000-1001); only its ABSENCE goes tree-free.
		if (tree === null && treeText !== null && !requestedTn93) {
			throw new Error('Could not parse phylogenetic tree from specified tree text');
		}
	}
	const branchLengthsMissing = tree !== null && !hasNonzeroBranchLengths(tree);
	/** @type {'requested'|'no_tree'|'no_branch_lengths'|null} */
	const treeFreeReason = requestedTn93 ? 'requested' : tree === null ? 'no_tree' : branchLengthsMissing ? 'no_branch_lengths' : null;

	if (treeFreeReason !== null) return tn93Assembly(seqDict, tree, treeFreeReason, branchLengthsMissing, { maxSpecies, pruneDuplicates, referenceName, tn93Options });

	// 3. Branch lengths (a usable tree, so the reference's HyPhy branch is not reached).
	enforceNonzeroBranchLengths(/** @type {import('./tree.js').PhyloTree} */ (tree), 1e-4);

	// 4. Taxon matching.
	const match = matchTaxa(treeTaxa(/** @type {import('./tree.js').PhyloTree} */ (tree)), seqDict.keys());
	let taxa = match.taxa;

	// 5. Duplicates.
	let duplicatesCollapsed = 0;
	let duplicateMap = new Map();
	if (pruneDuplicates && taxa.length > 1) {
		const pruned = pruneIdenticalSequences(seqDict, taxa);
		duplicateMap = pruned.dupMap;
		if (pruned.numPruned > 0) {
			duplicatesCollapsed = pruned.numPruned;
			taxa = pruned.uniqueTaxa;
		}
	}

	// 6. Lengths and frame.
	const lengths = new Set(taxa.map((sp) => /** @type {string} */ (seqDict.get(sp)).length));
	const unequalLengths = lengths.size > 1 ? Array.from(lengths) : null;
	const rawLen = /** @type {string} */ (seqDict.get(taxa[0])).length;
	if (rawLen < 3) {
		throw new Error(`Alignment sequence length (${rawLen} bp) is less than 1 codon (3 bp).`);
	}
	const codonsTrimmed = rawLen % 3;
	const L = Math.floor(rawLen / 3);

	// 7. Stride pre-selection.
	let stridePreselected = false;
	if (maxSpecies !== null && taxa.length > maxSpecies) {
		taxa = stridePreselect(taxa, maxSpecies);
		stridePreselected = true;
	}

	// 8-9. Distances and the rescale rule.
	const rescaled = rescaleDistances(computeFastDistMatrix(tree, taxa), L);
	let dist = rescaled.dist;

	// 10. Faith's PD.
	let pdSubsampled = false;
	if (maxSpecies !== null && taxa.length > maxSpecies) {
		const ds = downsampleTaxaFaithPd(dist, taxa, maxSpecies);
		dist = ds.distMat;
		taxa = ds.taxa;
		pdSubsampled = true;
	}

	// 11-14. The shared tail.
	return assembleTail(seqDict, taxa, dist, L, referenceName, tree, {
		branchLengthsMissing,
		treeFree: null,
		tn93SaturatedPairs: null,
		matchTier: match.tier,
		droppedTaxa: { alignment: match.droppedAlignment, tree: match.droppedTree },
		duplicatesCollapsed,
		duplicateMap,
		unequalLengths,
		codonsTrimmed,
		stridePreselected,
		distanceRescaled: rescaled.rescaled,
		rawDistMax: rescaled.rawMax,
		pdSubsampled
	});
}

/**
 * The tree-free branch of `load_alignment_and_tree` (dataset.py:956-995): alignment-order taxa,
 * duplicate pruning, length/frame checks, stride pre-selection, the TN93 matrix, Faith's PD. No
 * rescale, no taxon matching, no branch-length enforcement (see the header).
 *
 * @param {Map<string, string>} seqDict
 * @param {import('./tree.js').PhyloTree|null} tree kept for display only; never read here
 * @param {'requested'|'no_tree'|'no_branch_lengths'} reason
 * @param {boolean} branchLengthsMissing
 * @param {{maxSpecies: number|null, pruneDuplicates: boolean, referenceName: string|undefined,
 *   tn93Options: object}} options
 * @returns {LoadedAlignment}
 */
function tn93Assembly(seqDict, tree, reason, branchLengthsMissing, { maxSpecies, pruneDuplicates, referenceName, tn93Options }) {
	// 1. taxa = list(seq_dict.keys()) (dataset.py:958).
	let taxa = Array.from(seqDict.keys());

	// 2. Duplicates (dataset.py:959-964).
	let duplicatesCollapsed = 0;
	let duplicateMap = new Map();
	if (pruneDuplicates && taxa.length > 1) {
		const pruned = pruneIdenticalSequences(seqDict, taxa);
		duplicateMap = pruned.dupMap;
		if (pruned.numPruned > 0) {
			duplicatesCollapsed = pruned.numPruned;
			taxa = pruned.uniqueTaxa;
		}
	}

	// 3. Lengths and frame (dataset.py:966-981).
	const lengths = new Set(taxa.map((sp) => /** @type {string} */ (seqDict.get(sp)).length));
	const unequalLengths = lengths.size > 1 ? Array.from(lengths) : null;
	const rawLen = /** @type {string} */ (seqDict.get(taxa[0])).length;
	if (rawLen < 3) {
		throw new Error(`Alignment sequence length (${rawLen} bp) is less than 1 codon (3 bp).`);
	}
	const codonsTrimmed = rawLen % 3;
	const L = Math.floor(rawLen / 3);

	// 4. Stride pre-selection (dataset.py:984-987).
	let stridePreselected = false;
	if (maxSpecies !== null && taxa.length > maxSpecies) {
		taxa = stridePreselect(taxa, maxSpecies);
		stridePreselected = true;
	}

	// 5. The TN93 matrix (dataset.py:989). No `> 10` rescale on this path.
	let dist = tn93DistanceMatrix(seqDict, taxa, tn93Options);
	let rawDistMax = 0;
	for (let i = 0; i < dist.length; i++) if (dist[i] > rawDistMax) rawDistMax = dist[i];
	let saturatedPairs = tn93SaturatedPairs(dist, taxa.length);

	// 6. Faith's PD (dataset.py:991-994).
	let pdSubsampled = false;
	if (maxSpecies !== null && taxa.length > maxSpecies) {
		const ds = downsampleTaxaFaithPd(dist, taxa, maxSpecies);
		dist = ds.distMat;
		taxa = ds.taxa;
		pdSubsampled = true;
		saturatedPairs = tn93SaturatedPairs(dist, taxa.length);
	}

	return assembleTail(seqDict, taxa, dist, L, referenceName, tree, {
		branchLengthsMissing,
		treeFree: { reason, taxaOrder: 'alignment' },
		tn93SaturatedPairs: saturatedPairs,
		matchTier: null,
		droppedTaxa: { alignment: 0, tree: 0 },
		duplicatesCollapsed,
		duplicateMap,
		unequalLengths,
		codonsTrimmed,
		stridePreselected,
		distanceRescaled: false,
		rawDistMax,
		pdSubsampled
	});
}

/**
 * The tail both paths share, dataset.py:1102-1107 and 749-775: MDS on the real N x N, the [L, N, 1]
 * token arrays with the unknown / in-frame-stop counts, and the amino-acid invariable mask.
 *
 * @param {Map<string, string>} seqDict
 * @param {string[]} taxa
 * @param {Float32Array} dist
 * @param {number} L
 * @param {string|undefined} referenceName
 * @param {import('./tree.js').PhyloTree|null} tree
 * @param {Record<string, any>} notices path-specific notices; the codon counts are added here
 * @returns {LoadedAlignment}
 */
function assembleTail(seqDict, taxa, dist, L, referenceName, tree, notices) {
	// MDS.
	const N = taxa.length;
	const z = computeMdsCoordinates(dist, N, MDS_COMPONENTS);

	// Tokens.
	const c = new Int32Array(L * N);
	const a = new Int32Array(L * N);
	let unknownCodons = 0;
	let stopCodons = 0;
	const totalCodons = L * N;
	for (let i = 0; i < N; i++) {
		const seq = seqDict.get(taxa[i]) ?? '-'.repeat(L * 3);
		for (let site = 0; site < L; site++) {
			const codon = seq.slice(site * 3, (site + 1) * 3).toUpperCase();
			const cTok = codonToken(codon);
			const aTok = aaToken(codon);
			if (cTok === CODON_UNKNOWN) {
				if (codon === 'TAA' || codon === 'TAG' || codon === 'TGA') stopCodons++;
				else unknownCodons++;
			}
			c[site * N + i] = cTok;
			a[site * N + i] = aTok;
		}
	}

	// Invariable sites.
	const invariable = invariableMask(a, L, N);

	// `_warn_internal_stops(seq_dict, L, taxa)`, dataset.py:235-249, which the reference calls from
	// both branches of `load_alignment_and_tree` (dataset.py:982 and :1080) and PRINTS. A DIFFERENT
	// statistic from `inFrameStops` above: per taxon, over the first L-1 codons only (the terminal
	// codon of a coding sequence is legitimately a stop), reporting the worst taxon and its count.
	// Ties go to the first taxon in `taxa` order, as the reference's strict `>` does.
	const internalLimit = Math.max(0, (L - 1) * 3);
	let worstTaxon = null;
	let worstCount = 0;
	for (const t of taxa) {
		const seq = seqDict.get(t) ?? '';
		let stops = 0;
		const limit = Math.min(seq.length, internalLimit);
		for (let i = 0; i < limit; i += 3) {
			if (CODON_TO_AA.get(seq.slice(i, i + 3).toUpperCase()) === '*') stops++;
		}
		if (stops > worstCount) {
			worstCount = stops;
			worstTaxon = t;
		}
	}

	return {
		c,
		a,
		d: dist,
		z,
		invariable,
		taxa,
		L,
		N,
		referenceIndex: referenceName === undefined ? -1 : taxa.indexOf(referenceName),
		tree,
		notices: /** @type {LoadedAlignment['notices']} */ ({
			...notices,
			unknownCodons,
			unknownCodonFraction: unknownCodons / Math.max(1, totalCodons),
			inFrameStops: stopCodons,
			internalStops: { worstTaxon, worstCount },
			totalCodons
		})
	};
}

/**
 * The graph bundle for sites [start, start + count): `msa_codons` / `msa_aas` as BigInt64Array
 * [b, N, 1] (onnxruntime wants int64), `dist_matrix` [b, N, N] and `mds_coords` [b, N, 4] repeated
 * per site. Sites are chunked because the tensors are O(sites x N^2).
 *
 * @param {LoadedAlignment} loaded
 * @param {number} start
 * @param {number} [count]
 * @returns {Record<string, {data: BigInt64Array|Float32Array, dims: number[]}>}
 */
export function siteBatch(loaded, start, count = loaded.L - start) {
	const { c, a, d, z, L, N } = loaded;
	const b = Math.max(0, Math.min(count, L - start));
	const codons = new BigInt64Array(b * N);
	const aas = new BigInt64Array(b * N);
	for (let k = 0; k < b * N; k++) {
		codons[k] = BigInt(c[start * N + k]);
		aas[k] = BigInt(a[start * N + k]);
	}
	const distData = new Float32Array(b * N * N);
	const mdsData = new Float32Array(b * N * MDS_COMPONENTS);
	for (let k = 0; k < b; k++) {
		distData.set(d, k * N * N);
		mdsData.set(z, k * N * MDS_COMPONENTS);
	}
	return {
		msa_codons: { data: codons, dims: [b, N, 1] },
		msa_aas: { data: aas, dims: [b, N, 1] },
		dist_matrix: { data: distData, dims: [b, N, N] },
		mds_coords: { data: mdsData, dims: [b, N, MDS_COMPONENTS] }
	};
}

/**
 * Every site batch in order, each of at most `batchSize` sites.
 *
 * @param {LoadedAlignment} loaded
 * @param {number} batchSize
 * @returns {Generator<{start: number, count: number, bundle: ReturnType<typeof siteBatch>}>}
 */
export function* siteBatches(loaded, batchSize) {
	const step = Math.max(1, Math.floor(batchSize));
	for (let start = 0; start < loaded.L; start += step) {
		const count = Math.min(step, loaded.L - start);
		yield { start, count, bundle: siteBatch(loaded, start, count) };
	}
}

/**
 * Site count per batch that keeps the materialised tensors near `budgetBytes`. dist_matrix
 * dominates at 4 * N^2 bytes per site; at least one site is always returned.
 *
 * @param {number} speciesCount
 * @param {number} [budgetBytes] default 64 MB
 * @returns {number}
 */
export function batchSizeFor(speciesCount, budgetBytes = 64 * 1024 * 1024) {
	const perSite = 4 * speciesCount * speciesCount;
	return Math.max(1, Math.floor(budgetBytes / Math.max(perSite, 1)));
}
