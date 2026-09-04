/**
 * WHY THIS FILE EXISTS
 *
 * Mirrors `load_alignment_and_tree` of `hyphaeon/dataset.py:523-730` at veg/HyphAeon 267f5cf, end
 * to end, on strings: the tree branch (dataset.py:581-685) followed by the shared tail
 * (687-730). Step by step, with the line it mirrors:
 *
 *    1. parse_alignment_sequences(alignment)                                      540-542
 *    2. tree: extract from `treeText`, or from the alignment text when null        582-598
 *    3. has_nonzero_branch_lengths -> (HyPhy, see below) -> enforce 1e-3 / 1e-4    600-614
 *    4. three-tier taxon matching, TREE TERMINAL ORDER, dropped counts             616-643
 *    5. prune_identical_sequences when prune_duplicates and > 1 taxon              645-650
 *    6. unequal-length warning; L = len(first taxon's sequence) // 3;
 *       < 3 bp raises; the `% 3` remainder is reported as trimmed                  652-667
 *    7. stride pre-selection to the first 2 * max_species taxa                     669-673
 *    8. compute_fast_dist_matrix (float32)                                         676
 *    9. `if dist_mat.max() > 10.0: dist_mat /= L`                                  678-681
 *   10. downsample_taxa_faith_pd when still > max_species                          683-685
 *   11. compute_mds_coordinates(dist, 4)                                           687-688
 *   12. tokens [L, N, 1] with unknown / in-frame-stop counts                       690-716
 *   13. is_aa_invariable                                                           718-723
 *   14. tensors: c, a int64 [L,N,1]; d float32 [1,N,N]; z float32 [1,N,4]          725-730
 *
 * Everything the reference PRINTS becomes a field of `notices` so the runtime can show it; nothing
 * is printed here. The tokens are returned as Int32Array (values 0..64 / 0..20) in [L, N, 1]
 * layout; `siteBatch` converts a range of sites to the BigInt64Array / Float32Array bundle the ONNX
 * graph takes, with the site-invariant `dist_matrix` and `mds_coords` repeated per site because the
 * graph needs materialised data where torch broadcast a [1, N, N] view.
 *
 * WHAT IT DELIBERATELY DOES NOT DO:
 *   - HyPhy. dataset.py:601-611 shells out to `hyphy` for a tree without branch lengths. The
 *     library cannot; it takes the reference's "HyPhy not found" branch (enforce defaults, go on)
 *     and sets `notices.branchLengthsMissing`. `needsBranchLengths` (tree.js) is the predicate; the
 *     runtime may estimate a tree and call again with it.
 *   - TN93. `use_tn93` / `nwk_path in ("tn93", "none", "skip")` (dataset.py:544-580) needs the tn93
 *     binary or package; `useTn93: true` throws.
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
import { codonToken, aaToken } from './tokenizer.js';
import { invariableMask } from './variability.js';
import { MDS_COMPONENTS, CODON_UNKNOWN } from './modelContract.js';

/**
 * @typedef {{
 *   c: Int32Array, a: Int32Array, d: Float32Array, z: Float32Array,
 *   invariable: Uint8Array, taxa: string[], L: number, N: number,
 *   referenceIndex: number,
 *   tree: import('./tree.js').PhyloTree,
 *   notices: {
 *     branchLengthsMissing: boolean,
 *     matchTier: 'exact'|'quote_stripped'|'case_insensitive',
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
 *     totalCodons: number
 *   }
 * }} LoadedAlignment
 */

/**
 * `load_alignment_and_tree(fa_path, nwk_path, max_species, prune_duplicates)` on strings.
 *
 * @param {string} alignmentText FASTA / PHYLIP / NEXUS content
 * @param {string|null} [treeText] Newick / NEXUS tree content; null to look for a tree embedded in
 *   the alignment text
 * @param {{maxSpecies?: number|null, pruneDuplicates?: boolean, referenceName?: string,
 *   useTn93?: boolean}} [options]
 * @returns {LoadedAlignment}
 */
export function loadAlignmentAndTree(alignmentText, treeText = null, options = {}) {
	const { maxSpecies = null, pruneDuplicates = true, referenceName, useTn93 = false } = options;

	// 1. Alignment.
	const seqDict = parseAlignmentSequences(alignmentText);
	if (seqDict.size === 0) {
		throw new Error('Could not parse any sequences from alignment');
	}

	if (useTn93 || (treeText !== null && ['tn93', 'none', 'skip'].includes(String(treeText).trim().toLowerCase()))) {
		throw new Error(
			'loadAlignmentAndTree: the TN93 path (dataset.py:544-580) needs the tn93 binary or package and is not in the library'
		);
	}

	// 2. Tree.
	const tree = treeText === null ? extractTree(alignmentText) : extractTree(treeText);
	if (tree === null) {
		throw new Error(
			treeText === null
				? 'No tree specified, and no embedded phylogenetic tree found in alignment. Please provide a tree.'
				: 'Could not parse phylogenetic tree from specified tree text'
		);
	}

	// 3. Branch lengths. Where the reference would call HyPhy, this records the fact and takes the
	//    "HyPhy not found" branch.
	const branchLengthsMissing = !hasNonzeroBranchLengths(tree);
	enforceNonzeroBranchLengths(tree, 1e-4);

	// 4. Taxon matching.
	const match = matchTaxa(treeTaxa(tree), seqDict.keys());
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

	// 11. MDS.
	const N = taxa.length;
	const z = computeMdsCoordinates(dist, N, MDS_COMPONENTS);

	// 12. Tokens.
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

	// 13. Invariable sites.
	const invariable = invariableMask(a, L, N);

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
		notices: {
			branchLengthsMissing,
			matchTier: match.tier,
			droppedTaxa: { alignment: match.droppedAlignment, tree: match.droppedTree },
			duplicatesCollapsed,
			duplicateMap,
			unequalLengths,
			codonsTrimmed,
			stridePreselected,
			distanceRescaled: rescaled.rescaled,
			rawDistMax: rescaled.rawMax,
			pdSubsampled,
			unknownCodons,
			unknownCodonFraction: unknownCodons / Math.max(1, totalCodons),
			inFrameStops: stopCodons,
			totalCodons
		}
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
