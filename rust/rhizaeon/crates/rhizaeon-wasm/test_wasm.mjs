import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import wasmPkg, {
    initSync,
    engine_version,
    engine_build_timestamp,
    scan_fasta_json,
    count_informative_sites,
    scan_fasta_dossier_json,
    scan_fasta_html,
    scan_fasta_bundle,
    scan_fasta_bundle_with_progress,
} from './pkg/rhizaeon_wasm.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// In-repo empirical fixture — always resolvable, so CI and local runs both
// exercise the full scan/dossier/HTML surface on real data.
const KAL153_FASTA = path.resolve(
    __dirname, '../../tests/empirical/data/hiv1_kal153_crf02_ag.fasta'
);

// The TYLCV / adversarial-null / cassette ground-truth benchmarks live in the
// TOGA_MEME corpus, which is not vendored into the monorepo. Set
// RHIZAEON_BENCH_DIR to its benchmarks/ root to enable them; the historical
// sibling-checkout layout remains the default for local dev.
const BENCH_DIR = process.env.RHIZAEON_BENCH_DIR
    || path.resolve(__dirname, '../../../../TOGA_MEME/recombination/benchmarks');
const HAS_BENCH = fs.existsSync(BENCH_DIR);

// Ground-truth recombination events: { fixture, label, candidate, home, donor }.
// KAL153 uses the in-repo alignment; the rest require the TOGA_MEME corpus.
const GT_BENCH = [
    { file: '01_tylcv_is76/data/tylcv_is76_aligned.fasta', label: 'TYLCV IS76', cand: 'AF271234.1', home: 'Z25751.1', donor: 'X15656.1' },
    { file: '14_graduated_difficulty_ladder/alignments/t3_rec_cassette_rep0.fasta', label: '8-Taxon Cassette Swap', cand: 'R', home: 'P1', donor: 'P2' },
];

// Adversarial nulls — any emitted event is a false positive.
const NULL_BENCH = [
    't5_null_darren_trap_rep37.fasta',
    't5_null_apobec_burst_rep81.fasta',
    't5_null_codon_darwinian_rep92.fasta',
    't1_null_clock_rep62.fasta',
];

function logEvents(events) {
    console.log(`  Verified events: ${events.length}`);
    for (const ev of events) {
        console.log(`  -> Event: ${ev.candidate_name} | Home: ${ev.home_name} | Donor: ${ev.donor_name} | Tract: ${ev.u1}-${ev.u2} (${ev.tract_length} bp) | p-Fisher: ${ev.p_fisher.toExponential(2)} | Z: ${ev.z_phys.toFixed(2)}`);
    }
}

function fail(msg) {
    console.error(`  >>> FAIL: ${msg}`);
    process.exit(1);
}

async function run() {
    console.log("================================================================================");
    console.log("  RHIZAEON WASM CLIENT ENGINE VERIFICATION (Node.js & Browser Runtime)");
    console.log("================================================================================");

    const wasmPath = path.join(__dirname, 'pkg', 'rhizaeon_wasm_bg.wasm');
    const wasmBytes = fs.readFileSync(wasmPath);
    initSync({ module: wasmBytes });

    console.log("  WASM Binary loaded:", wasmPath);
    console.log("  WASM Binary size:  ", (wasmBytes.length / 1024).toFixed(1), "KB");
    console.log("--------------------------------------------------------------------------------");

    // Test 1: HIV-1 KAL153 CRF02_AG (Salminen 1997) — in-repo empirical fixture.
    // Ground truth: recombinant taxon R is a B/A mosaic.
    const hivFasta = fs.readFileSync(KAL153_FASTA, 'utf8');
    const infSites = count_informative_sites(hivFasta);

    console.log(`[TEST 1] HIV-1 KAL153 Full Genome (10 kb) Benchmark [in-repo fixture]`);
    const t0 = performance.now();
    const hivResult = JSON.parse(scan_fasta_json(hivFasta, 16));
    const t1 = performance.now();

    console.log(`  Execution time: ${(t1 - t0).toFixed(2)} ms (internal WASM: ${hivResult.run_time_ms.toFixed(2)} ms)`);
    console.log(`  Informative sites: ${infSites}`);
    console.log(`  Candidate taxa: ${hivResult.screening.candidate_recombinant_indices.length}`);
    logEvents(hivResult.events);

    const kal153 = hivResult.events.find(e => e.candidate_name === 'R');
    if (kal153 && kal153.home_name === 'B' && kal153.donor_name === 'A') {
        console.log("  >>> PASS: HIV-1 KAL153 exactly replicated ground truth (B/A mosaic)!");
    } else {
        fail("KAL153 mosaic not found or mismatched");
    }
    console.log("--------------------------------------------------------------------------------");

    if (HAS_BENCH) {
        // Tests 2..N: TOGA_MEME ground-truth recombination events
        for (const bt of GT_BENCH) {
            const fasta = fs.readFileSync(path.join(BENCH_DIR, bt.file), 'utf8');
            const bt0 = performance.now();
            const res = JSON.parse(scan_fasta_json(fasta, 16));
            const bt1 = performance.now();

            console.log(`[BENCH] ${bt.label} (${bt.file})`);
            console.log(`  Execution time: ${(bt1 - bt0).toFixed(2)} ms (internal WASM: ${res.run_time_ms.toFixed(2)} ms)`);
            console.log(`  Candidate taxa: ${res.screening.candidate_recombinant_indices.length}`);
            logEvents(res.events);

            const ev = res.events.find(e => e.candidate_name === bt.cand);
            if (ev && ev.home_name === bt.home && ev.donor_name === bt.donor) {
                console.log(`  >>> PASS: ${bt.label} replicated ground truth (${bt.cand} = ${bt.home}/${bt.donor} mosaic)!`);
            } else {
                fail(`${bt.label} mosaic not found or mismatched`);
            }
            console.log("--------------------------------------------------------------------------------");
        }

        // Adversarial Null Suite (zero false positive mandate)
        console.log(`[BENCH] Adversarial Null Calibration Suite`);
        for (const nf of NULL_BENCH) {
            const nFasta = fs.readFileSync(
                path.join(BENCH_DIR, '14_graduated_difficulty_ladder/alignments', nf), 'utf8');
            const nRes = JSON.parse(scan_fasta_json(nFasta, 16));
            console.log(`  ${nf.padEnd(35)} -> Events: ${nRes.events.length} (H0: ${nRes.is_non_recombinant_h0}) in ${nRes.run_time_ms.toFixed(2)} ms`);
            if (nRes.events.length > 0) {
                fail(`Adversarial null ${nf} leaked ${nRes.events.length} false positives!`);
            }
        }
        console.log("  >>> PASS: All adversarial nulls clean (zero false positives)");
        console.log("--------------------------------------------------------------------------------");
    } else {
        console.log(`[SKIP] TOGA_MEME benchmark corpus not found at ${BENCH_DIR}`);
        console.log("       Set RHIZAEON_BENCH_DIR to enable TYLCV/cassette/null ground-truth tests.");
        console.log("--------------------------------------------------------------------------------");
    }

    // Final test: Rich Visualization Dossier and Standalone HTML Export on the
    // in-repo fixture. As of engine beta.8 this 9-taxon CRF02_AG alignment
    // yields: query R, 6 breakpoints, 422 trajectory points. Engine accuracy
    // is asserted by TEST 1's ground-truth match; this block only verifies
    // dossier structure is complete and populated.
    console.log(`[TEST] Rich Visualization Dossier and Standalone HTML Export`);

    const dossierJsonStr = scan_fasta_dossier_json(hivFasta, "HIV-1 KAL153 WASM Dossier", 16);
    const dossier = JSON.parse(dossierJsonStr);

    console.log(`  Dossier JSON size: ${(dossierJsonStr.length / 1024).toFixed(1)} KB`);
    console.log(`  Dossier keys: ${Object.keys(dossier).join(', ')}`);
    console.log(`  Metadata title: "${dossier.metadata.title}"`);
    console.log(`  Recombinant query: "${dossier.metadata.query_id}"`);
    console.log(`  Taxa meta count: ${Object.keys(dossier.taxa_meta).length}`);
    console.log(`  Breakpoints count: ${dossier.breakpoints.length}`);
    console.log(`  Signals trajectory points: ${dossier.signals.trajectory_x.length}`);
    console.log(`  Informative SNPs count: ${dossier.signals.informative_snps.length}`);
    console.log(`  Subspaces partition 1 taxa: ${Object.keys(dossier.subspaces.partition_1).length}`);
    console.log(`  Canonical manifold colinear: ${dossier.canonical_manifold.is_colinear} (cos theta = ${dossier.canonical_manifold.cos_theta})`);

    if (dossier.metadata.query_id === 'R' && dossier.breakpoints.length >= 2 && dossier.signals.trajectory_x.length > 100) {
        console.log("  >>> PASS: WASM generated complete, valid Visualization Dossier JSON!");
    } else {
        fail("Dossier JSON validation failed");
    }

    const htmlStr = scan_fasta_html(hivFasta, "HIV-1 KAL153 WASM Dashboard", 16);
    console.log(`  Standalone HTML size: ${(htmlStr.length / 1024).toFixed(1)} KB`);
    if (htmlStr.includes('<script id="rhizaeon-data" type="application/json">') && htmlStr.includes('HIV-1 KAL153 WASM Dashboard')) {
        console.log("  >>> PASS: WASM generated complete standalone HTML dashboard!");
    } else {
        fail("Standalone HTML generation failed");
    }

    const bundle = scan_fasta_bundle(hivFasta, "HIV-1 KAL153 Bundle", 16);
    console.log(`  Engine version: ${engine_version()} (${engine_build_timestamp()})`);
    if (bundle && bundle.result && bundle.html && bundle.landscape && bundle.landscape.grid_x.length > 0) {
        console.log("  >>> PASS: scan_fasta_bundle returned result + html + landscape!");
    } else {
        fail("scan_fasta_bundle validation failed");
    }

    const progressSteps = [];
    const progBundle = scan_fasta_bundle_with_progress(
        hivFasta,
        "HIV-1 KAL153 Progress Bundle",
        16,
        undefined,
        1.0,
        1.0,
        (frac, msg) => {
            progressSteps.push({ frac, msg });
        }
    );
    console.log(`  Progress callbacks emitted: ${progressSteps.length} stages (${progressSteps.map(p => Math.round(p.frac * 100) + '%').join(' -> ')})`);
    if (
        progBundle &&
        progBundle.result &&
        progressSteps.length >= 6 &&
        progressSteps[progressSteps.length - 1].frac === 1.0 &&
        htmlStr.includes('function clearcutTree(') &&
        htmlStr.includes('shouldCenterTreesOnRec') &&
        htmlStr.includes('_hasMdsHoverListener') &&
        htmlStr.includes('saveAlignmentOption')
    ) {
        console.log("  >>> PASS: scan_fasta_bundle_with_progress + v0.1.2 dashboard features verified!");
    } else {
        fail("scan_fasta_bundle_with_progress or v0.1.2 dashboard feature verification failed");
    }

    console.log("================================================================================");
    console.log("  ALL WASM VERIFICATION CHECKS (SCAN + DOSSIER + HTML + BUNDLE) PASSED!");
    console.log("================================================================================");
}

run().catch(err => {
    console.error("Error running WASM test:", err);
    process.exit(1);
});
