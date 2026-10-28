use rhizaeon::{parse_fasta, RecombinationLandscape, RhizAeonEngine, ScanResult};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
pub struct WasmScanBundle {
    pub result: ScanResult,
    pub html: String,
    pub landscape: Option<RecombinationLandscape>,
}

#[wasm_bindgen]
pub fn init_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Returns the exact compiled RhizAeon WASM engine version string (e.g., "0.1.0-beta.10")
#[wasm_bindgen]
pub fn engine_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Returns the compilation timestamp of the RhizAeon WASM engine
#[wasm_bindgen]
pub fn engine_build_timestamp() -> String {
    option_env!("RHIZAEON_BUILD_DATE")
        .unwrap_or("2026-10-28")
        .to_string()
}

/// Scans a FASTA string for recombination events and returns a JS Object payload
#[wasm_bindgen]
pub fn scan_fasta(
    fasta_str: &str,
    target_landmarks: Option<usize>,
    compress_snps: Option<bool>,
    tier2_stringency: Option<f64>,
    tier3_stringency: Option<f64>,
) -> Result<JsValue, JsValue> {
    init_panic_hook();

    let start_time = js_sys::Date::now();
    let aln = parse_fasta(fasta_str).map_err(|e| JsValue::from_str(&e))?;

    let k = target_landmarks.unwrap_or(16);
    let mut engine = RhizAeonEngine::new()
        .with_landmarks(k)
        .with_tier2_stringency(tier2_stringency.unwrap_or(1.0))
        .with_tier3_stringency(tier3_stringency.unwrap_or(1.0));
    if let Some(cs) = compress_snps {
        engine = engine.with_compress_snps(cs);
    }
    let mut res = engine.scan(&aln);

    let end_time = js_sys::Date::now();
    res.run_time_ms = end_time - start_time;

    serde_wasm_bindgen::to_value(&res).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Scans a FASTA string and returns a formatted JSON string (optimal for zero-copy UI passing)
#[wasm_bindgen]
pub fn scan_fasta_json(
    fasta_str: &str,
    target_landmarks: Option<usize>,
    compress_snps: Option<bool>,
    tier2_stringency: Option<f64>,
    tier3_stringency: Option<f64>,
) -> Result<String, JsValue> {
    init_panic_hook();

    let start_time = js_sys::Date::now();
    let aln = parse_fasta(fasta_str).map_err(|e| JsValue::from_str(&e))?;

    let k = target_landmarks.unwrap_or(16);
    let mut engine = RhizAeonEngine::new()
        .with_landmarks(k)
        .with_tier2_stringency(tier2_stringency.unwrap_or(1.0))
        .with_tier3_stringency(tier3_stringency.unwrap_or(1.0));
    if let Some(cs) = compress_snps {
        engine = engine.with_compress_snps(cs);
    }
    let mut res = engine.scan(&aln);

    let end_time = js_sys::Date::now();
    res.run_time_ms = end_time - start_time;

    serde_json::to_string_pretty(&res).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Fast pre-screening: checks parsimony-informative sites in < 2 ms
#[wasm_bindgen]
pub fn count_informative_sites(fasta_str: &str) -> Result<usize, JsValue> {
    let aln = parse_fasta(fasta_str).map_err(|e| JsValue::from_str(&e))?;
    Ok(rhizaeon::fasta::count_parsimony_informative_sites(&aln))
}

/// Scans a FASTA string and returns a rich visualization dossier as a JS Object payload
#[wasm_bindgen]
pub fn scan_fasta_dossier(
    fasta_str: &str,
    title: Option<String>,
    target_landmarks: Option<usize>,
    compress_snps: Option<bool>,
    tier2_stringency: Option<f64>,
    tier3_stringency: Option<f64>,
) -> Result<JsValue, JsValue> {
    init_panic_hook();

    let start_time = js_sys::Date::now();
    let aln = parse_fasta(fasta_str).map_err(|e| JsValue::from_str(&e))?;

    let k = target_landmarks.unwrap_or(16);
    let mut engine = RhizAeonEngine::new()
        .with_landmarks(k)
        .with_tier2_stringency(tier2_stringency.unwrap_or(1.0))
        .with_tier3_stringency(tier3_stringency.unwrap_or(1.0));
    if let Some(cs) = compress_snps {
        engine = engine.with_compress_snps(cs);
    }
    let mut res = engine.scan(&aln);

    let end_time = js_sys::Date::now();
    res.run_time_ms = end_time - start_time;

    let dossier = rhizaeon::generate_visualization_dossier(&aln, &res, title.as_deref());
    serde_wasm_bindgen::to_value(&dossier).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Scans a FASTA string and returns a rich visualization dossier as a formatted JSON string
#[wasm_bindgen]
pub fn scan_fasta_dossier_json(
    fasta_str: &str,
    title: Option<String>,
    target_landmarks: Option<usize>,
    compress_snps: Option<bool>,
    tier2_stringency: Option<f64>,
    tier3_stringency: Option<f64>,
) -> Result<String, JsValue> {
    init_panic_hook();

    let start_time = js_sys::Date::now();
    let aln = parse_fasta(fasta_str).map_err(|e| JsValue::from_str(&e))?;

    let k = target_landmarks.unwrap_or(16);
    let mut engine = RhizAeonEngine::new()
        .with_landmarks(k)
        .with_tier2_stringency(tier2_stringency.unwrap_or(1.0))
        .with_tier3_stringency(tier3_stringency.unwrap_or(1.0));
    if let Some(cs) = compress_snps {
        engine = engine.with_compress_snps(cs);
    }
    let mut res = engine.scan(&aln);

    let end_time = js_sys::Date::now();
    res.run_time_ms = end_time - start_time;

    let dossier = rhizaeon::generate_visualization_dossier(&aln, &res, title.as_deref());
    serde_json::to_string_pretty(&dossier).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Scans a FASTA string and returns a complete, self-contained standalone HTML interactive dashboard
#[wasm_bindgen]
pub fn scan_fasta_html(
    fasta_str: &str,
    title: Option<String>,
    target_landmarks: Option<usize>,
    compress_snps: Option<bool>,
    tier2_stringency: Option<f64>,
    tier3_stringency: Option<f64>,
) -> Result<String, JsValue> {
    init_panic_hook();

    let start_time = js_sys::Date::now();
    let aln = parse_fasta(fasta_str).map_err(|e| JsValue::from_str(&e))?;

    let k = target_landmarks.unwrap_or(16);
    let mut engine = RhizAeonEngine::new()
        .with_landmarks(k)
        .with_tier2_stringency(tier2_stringency.unwrap_or(1.0))
        .with_tier3_stringency(tier3_stringency.unwrap_or(1.0));
    if let Some(cs) = compress_snps {
        engine = engine.with_compress_snps(cs);
    }
    let mut res = engine.scan(&aln);

    let end_time = js_sys::Date::now();
    res.run_time_ms = end_time - start_time;

    let dossier = rhizaeon::generate_visualization_dossier(&aln, &res, title.as_deref());
    rhizaeon::generate_standalone_html(&dossier).map_err(|e| JsValue::from_str(&e))
}

/// Scans a FASTA string in a single pass and returns `{ result, html, landscape }`
#[wasm_bindgen]
pub fn scan_fasta_bundle(
    fasta_str: &str,
    title: Option<String>,
    target_landmarks: Option<usize>,
    compress_snps: Option<bool>,
    tier2_stringency: Option<f64>,
    tier3_stringency: Option<f64>,
) -> Result<JsValue, JsValue> {
    init_panic_hook();

    let start_time = js_sys::Date::now();
    let aln = parse_fasta(fasta_str).map_err(|e| JsValue::from_str(&e))?;

    let k = target_landmarks.unwrap_or(16);
    let mut engine = RhizAeonEngine::new()
        .with_landmarks(k)
        .with_tier2_stringency(tier2_stringency.unwrap_or(1.0))
        .with_tier3_stringency(tier3_stringency.unwrap_or(1.0));
    if let Some(cs) = compress_snps {
        engine = engine.with_compress_snps(cs);
    }
    let mut res = engine.scan(&aln);

    let end_time = js_sys::Date::now();
    res.run_time_ms = end_time - start_time;

    let dossier = rhizaeon::generate_visualization_dossier(&aln, &res, title.as_deref());
    let html = rhizaeon::generate_standalone_html(&dossier).map_err(|e| JsValue::from_str(&e))?;
    let landscape = dossier.recombination_landscape.clone();

    let bundle = WasmScanBundle {
        result: res,
        html,
        landscape,
    };
    serde_wasm_bindgen::to_value(&bundle).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Scans a FASTA string with a JavaScript progress callback `(fraction, stage_label)` and returns `{ result, html, landscape }`
#[wasm_bindgen]
pub fn scan_fasta_bundle_with_progress(
    fasta_str: &str,
    title: Option<String>,
    target_landmarks: Option<usize>,
    compress_snps: Option<bool>,
    tier2_stringency: Option<f64>,
    tier3_stringency: Option<f64>,
    progress_cb: Option<js_sys::Function>,
) -> Result<JsValue, JsValue> {
    init_panic_hook();

    let report_progress = |frac: f64, msg: &str| {
        if let Some(ref cb) = progress_cb {
            let _ = cb.call2(&JsValue::NULL, &JsValue::from_f64(frac), &JsValue::from_str(msg));
        }
    };

    report_progress(0.02, "Parsing FASTA alignment...");
    let start_time = js_sys::Date::now();
    let aln = parse_fasta(fasta_str).map_err(|e| JsValue::from_str(&e))?;

    let k = target_landmarks.unwrap_or(16);
    let mut engine = RhizAeonEngine::new()
        .with_landmarks(k)
        .with_tier2_stringency(tier2_stringency.unwrap_or(1.0))
        .with_tier3_stringency(tier3_stringency.unwrap_or(1.0));
    if let Some(cs) = compress_snps {
        engine = engine.with_compress_snps(cs);
    }
    let mut res = engine.scan_with_progress(&aln, |frac, msg| {
        report_progress(0.05 + frac * 0.80, msg);
    });

    let end_time = js_sys::Date::now();
    res.run_time_ms = end_time - start_time;

    report_progress(0.88, "Synthesizing visualization dossier & 2D MDS subspaces...");
    let dossier = rhizaeon::generate_visualization_dossier(&aln, &res, title.as_deref());
    report_progress(0.96, "Rendering interactive HTML dashboard...");
    let html = rhizaeon::generate_standalone_html(&dossier).map_err(|e| JsValue::from_str(&e))?;
    let landscape = dossier.recombination_landscape.clone();
    report_progress(1.0, "Complete");

    let bundle = WasmScanBundle {
        result: res,
        html,
        landscape,
    };
    serde_wasm_bindgen::to_value(&bundle).map_err(|e| JsValue::from_str(&e.to_string()))
}
