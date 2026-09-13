/**
 * assemble.test.js — loadAlignmentAndTree and the site-batch builder, on hand-checkable inputs.
 *
 * WHY THIS FILE EXISTS
 *
 * Assembly is where the separately-verified stages are joined, so the failures available here are
 * joining failures: right values in the wrong order, right shapes with the wrong species, MDS on the
 * wrong matrix. The fixture replay (Smc6, bat_oas1, the duplicate and rescale cases) is the oracle;
 * these cases say which step broke.
 *
 * Replaces DataMonkey 3's axomeme-assemble.test.js. Dropped with the code they tested:
 * `chooseReference` / `orderSpecies` (a reference sequence moved to index 0 — dataset.py has none),
 * "MDS on the PADDED matrix" (dataset.py:1047 uses the real N), "clamps a negative distance"
 * (dataset.py has no negative distances after enforce_nonzero_branch_lengths), "Max-PD seeded at
 * the reference" (dataset.py seeds with the most distant pair), "falls back to an all-zero matrix
 * without a tree" (dataset.py raises). `batchSizeFor` and the batching cases are kept.
 */
import { describe, it, expect } from 'vitest';
import { loadAlignmentAndTree, siteBatch, siteBatches, batchSizeFor } from '../src/preprocess/assemble.js';
import { stubTn93Options } from './helpers/tn93-engine.js';
import { computeMdsCoordinates } from '../src/preprocess/mds.js';
import { validateInputBundle } from '../src/preprocess/modelContract.js';

/** Four taxa, 3 codons each. Tree order is deliberately NOT alignment order. */
const FASTA = '>alpha\nATGTTATCA\n>beta\nATGCTATCA\n>gamma\nATGTTAAGC\n>delta\nATGGGGTCA\n';
const TREE = '((gamma:0.1,delta:0.2):0.05,(beta:0.3,alpha:0.15):0.02);';

/** The same four taxa over 20 codons: enough overlap for TN93 not to saturate (see tn93.js). */
const LONG_BASE = 'ATGTTATCAGGGCCCAAATTTCCCGGGAAATTTCCCGGGAAACCCTTTGGG';
const LONG_FASTA =
	`>alpha\n${LONG_BASE}AAACCCGGG\n` +
	`>beta\n${LONG_BASE.replace('TTATCA', 'CTATCC')}AAACCCGGT\n` +
	`>gamma\n${LONG_BASE.replace('GGGCCC', 'GGACCA')}AAACCCGGA\n` +
	`>delta\n${LONG_BASE.replace('AAATTT', 'AAGTTC')}AAACCCGGC\n`;
const LONG_TREE = '((gamma:0.1,delta:0.2):0.05,(beta:0.3,alpha:0.15):0.02);';

describe('loadAlignmentAndTree', () => {
	it('orders taxa by the TREE terminals, with no reference sequence moved to the front', () => {
		const r = loadAlignmentAndTree(FASTA, TREE);
		expect(r.taxa).toEqual(['gamma', 'delta', 'beta', 'alpha']);
		expect(r.N).toBe(4);
		expect(r.L).toBe(3);
		expect(r.notices.matchTier).toBe('exact');
		expect(r.notices.droppedTaxa).toEqual({ alignment: 0, tree: 0 });
	});

	it('builds the [L, N, 1] token arrays in taxa order', () => {
		const r = loadAlignmentAndTree(FASTA, TREE);
		const at = (site, i) => r.c[site * r.N + i];
		// site 1: gamma TTA, delta GGG, beta CTA, alpha TTA
		expect([at(1, 0), at(1, 1), at(1, 2), at(1, 3)]).toEqual([2, 60, 15, 2]);
		expect(r.a[1 * r.N + 1]).toBe(5); // GGG -> G
		expect(r.c).toBeInstanceOf(Int32Array);
		expect(r.c).toHaveLength(r.L * r.N);
	});

	it('orders the distance matrix rows to match the taxa, in float32', () => {
		const r = loadAlignmentAndTree(FASTA, TREE);
		const n = r.N;
		expect(r.d[0 * n + 1]).toBe(Math.fround(0.3)); // gamma-delta: 0.1 + 0.2
		expect(r.d[0 * n + 2]).toBe(Math.fround(0.47)); // gamma-beta: 0.1 + 0.05 + 0.02 + 0.3
		expect(r.d[0]).toBe(0);
		expect(r.notices.distanceRescaled).toBe(false);
		expect(r.notices.rawDistMax).toBe(Math.fround(0.57)); // delta-beta
	});

	it('runs MDS on the REAL N x N matrix it returns', () => {
		const r = loadAlignmentAndTree(FASTA, TREE);
		expect(Array.from(r.z)).toEqual(Array.from(computeMdsCoordinates(r.d, r.N, 4)));
		expect(r.z).toHaveLength(r.N * 4);
	});

	it('applies dataset.py"s invariable rule: a serine island is INVARIABLE', () => {
		// site 0: ATG x4; site 1: L, G, L, L; site 2: TCA, TCA, AGC, TCA -> all serine.
		const r = loadAlignmentAndTree(FASTA, TREE);
		expect(Array.from(r.invariable)).toEqual([1, 0, 1]);
	});

	it('finds a tree embedded in the alignment when treeText is null', () => {
		const r = loadAlignmentAndTree(FASTA + '\n' + TREE + '\n', null);
		expect(r.taxa).toEqual(['gamma', 'delta', 'beta', 'alpha']);
	});

	it('raises on an unparseable tree, no sequences, or fewer than 3 bp', () => {
		// "no tree" no longer raises: it is the tree-free TN93 path (D22), covered in tn93.test.js.
		expect(() => loadAlignmentAndTree(FASTA, 'not a tree')).toThrow(/Could not parse phylogenetic tree/);
		expect(() => loadAlignmentAndTree('', TREE)).toThrow(/Could not parse any sequences/);
		expect(() => loadAlignmentAndTree('>alpha\nAT\n>beta\nAT\n', '((alpha:0.1,beta:0.1):0.1,(x:1,y:1):1);')).toThrow(/less than 1 codon/);
	});

	it('takes the tree-free TN93 path on request, in ALIGNMENT order (D22)', () => {
		// 9 nucleotides saturate TN93, so the tree-free cases use a longer alignment; the full
		// tree-free assembly and its fixture replay are in tn93.test.js.
		// The subject is the BRANCH and the taxon order, not the distances; the engine is a test
		// double (test/helpers/tn93-engine.js) and `d[1] > 0` only asks that its answer arrived.
		const tn93Options = stubTn93Options();
		for (const r of [loadAlignmentAndTree(LONG_FASTA, 'tn93', { tn93Options }), loadAlignmentAndTree(LONG_FASTA, LONG_TREE, { useTn93: true, tn93Options })]) {
			expect(r.taxa).toEqual(['alpha', 'beta', 'gamma', 'delta']);
			expect(r.notices.treeFree).toEqual({ reason: 'requested', taxaOrder: 'alignment' });
			expect(r.notices.matchTier).toBeNull();
			expect(r.notices.distanceRescaled).toBe(false);
			expect(r.d[0 * 4 + 1]).toBeGreaterThan(0);
		}
	});

	it('goes tree-free for a tree without branch lengths instead of the "HyPhy not found" branch (D22)', () => {
		const r = loadAlignmentAndTree(LONG_FASTA, '((gamma,delta),(beta,alpha));', { tn93Options: stubTn93Options() });
		expect(r.notices.branchLengthsMissing).toBe(true);
		expect(r.notices.treeFree).toEqual({ reason: 'no_branch_lengths', taxaOrder: 'alignment' });
		// The topology is still returned for display, with its branch lengths untouched (null).
		expect(r.tree).not.toBeNull();
		expect(r.taxa).toEqual(['alpha', 'beta', 'gamma', 'delta']);
	});

	it('reports the length remainder it trims and unequal lengths', () => {
		const r = loadAlignmentAndTree('>alpha\nATGTTATCAG\n>beta\nATGCTATCA\n', '((alpha:0.1,beta:0.1):0.1,(x:1,y:1):1);');
		expect(r.L).toBe(3);
		expect(r.notices.codonsTrimmed).toBe(1);
		expect(r.notices.unequalLengths).toEqual([10, 9]);
		expect(r.notices.droppedTaxa).toEqual({ alignment: 0, tree: 2 });
	});

	it('counts unknown codons and in-frame stops the way dataset.py:1061-1075 does', () => {
		const r = loadAlignmentAndTree('>alpha\nATGTAANNN\n>beta\nATGTGA---\n', '((alpha:0.1,beta:0.1):0.1,(x:1,y:1):1);');
		expect(r.notices.inFrameStops).toBe(2);
		expect(r.notices.unknownCodons).toBe(2);
		expect(r.notices.unknownCodonFraction).toBeCloseTo(2 / 6, 12);
		expect(r.notices.totalCodons).toBe(6);
	});

	it('strides then Faith"s-PD downsamples when over maxSpecies, seeded with the most distant pair', () => {
		// delta-beta (0.57) is the largest distance, at row-major index (1, 2) of the tree-ordered
		// matrix, so the seed pair is [delta, beta].
		const r = loadAlignmentAndTree(FASTA, TREE, { maxSpecies: 2 });
		expect(r.taxa).toEqual(['delta', 'beta']);
		expect(r.N).toBe(2);
		expect(r.notices.stridePreselected).toBe(true);
		expect(r.notices.pdSubsampled).toBe(true);
		expect(r.d[1]).toBe(Math.fround(0.57));
		expect(r.c).toHaveLength(r.L * 2);
	});

	it('applies no cap by default (maxSpecies null)', () => {
		const r = loadAlignmentAndTree(FASTA, TREE);
		expect(r.notices.stridePreselected).toBe(false);
		expect(r.notices.pdSubsampled).toBe(false);
	});

	it('collapses identical sequences onto the first taxon in tree order unless told not to', () => {
		const fa = '>alpha\nATGTTATCA\n>beta\nATGTTATCA\n>gamma\nATGTTAAGC\n>delta\nATGGGGTCA\n';
		const r = loadAlignmentAndTree(fa, TREE);
		expect(r.taxa).toEqual(['gamma', 'delta', 'beta']);
		expect(r.notices.duplicatesCollapsed).toBe(1);
		expect(r.notices.duplicateMap.get('beta')).toEqual(['alpha']);
		const keep = loadAlignmentAndTree(fa, TREE, { pruneDuplicates: false });
		expect(keep.taxa).toEqual(['gamma', 'delta', 'beta', 'alpha']);
		expect(keep.notices.duplicatesCollapsed).toBe(0);
	});

	it('reports where a named taxon landed without moving it', () => {
		const r = loadAlignmentAndTree(FASTA, TREE, { referenceName: 'alpha' });
		expect(r.referenceIndex).toBe(3);
		expect(r.taxa[0]).toBe('gamma');
		expect(loadAlignmentAndTree(FASTA, TREE, { referenceName: 'nope' }).referenceIndex).toBe(-1);
		expect(loadAlignmentAndTree(FASTA, TREE).referenceIndex).toBe(-1);
	});
});

describe('siteBatch / siteBatches', () => {
	it('produces a bundle that satisfies the contract', () => {
		const r = loadAlignmentAndTree(FASTA, TREE);
		const bundle = siteBatch(r, 0);
		expect(validateInputBundle(bundle, { batch: r.L, numSpecies: r.N }).errors).toEqual([]);
		expect(bundle.msa_codons.data).toBeInstanceOf(BigInt64Array);
		expect(bundle.msa_codons.dims).toEqual([3, 4, 1]);
		expect(bundle.dist_matrix.dims).toEqual([3, 4, 4]);
		expect(bundle.mds_coords.dims).toEqual([3, 4, 4]);
	});

	it('slices sites without disturbing the per-alignment tensors', () => {
		const r = loadAlignmentAndTree(FASTA, TREE);
		const all = siteBatch(r, 0);
		const tail = siteBatch(r, 1, 2);
		expect(tail.msa_codons.dims).toEqual([2, 4, 1]);
		const n = r.N;
		for (let s = 0; s < n; s++) expect(tail.msa_codons.data[s]).toBe(all.msa_codons.data[n + s]);
		for (let k = 0; k < n * n; k++) expect(tail.dist_matrix.data[n * n + k]).toBe(tail.dist_matrix.data[k]);
		expect(Number(all.msa_codons.data[1 * n + 1])).toBe(60); // site 1, delta: GGG
	});

	it('clamps a range that runs past the end', () => {
		const r = loadAlignmentAndTree(FASTA, TREE);
		expect(siteBatch(r, 2, 99).msa_codons.dims[0]).toBe(1);
		expect(siteBatch(r, 3, 5).msa_codons.dims[0]).toBe(0);
	});

	it('iterates every site exactly once in batches of at most batchSize', () => {
		const r = loadAlignmentAndTree(FASTA, TREE);
		const seen = [...siteBatches(r, 2)].map((b) => [b.start, b.count]);
		expect(seen).toEqual([
			[0, 2],
			[2, 1]
		]);
	});

	it('sizes batches against the dist_matrix budget', () => {
		expect(batchSizeFor(512, 64 * 1024 * 1024)).toBe(64);
		expect(batchSizeFor(36, 64 * 1024 * 1024)).toBeGreaterThan(1000);
		expect(batchSizeFor(4096, 1024)).toBe(1);
	});
});
