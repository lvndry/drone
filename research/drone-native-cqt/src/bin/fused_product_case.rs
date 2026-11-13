//! One fixed frontend-only candidate; preserves the archived native negative control.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use drone_native_cached::{bytes_sha256, check_file, file_sha256};
use drone_native_cqt::{
    plan::{PreparedPlan, Soxr},
    read_f32,
};
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

const MANIFEST: &str = "462205de24ff451e040afe45a3409cfc3fa9b74afc4f37d1725024cee59dc1fc";
const PCM: &str = "0b1234cb5e3c1ec2573d379660d7078ee7bffb1db029b87ffa58605836e2dde0";
const PYTHON_CORE: &str = "4024b6238ef874f7f6e44a3355f384b4d2bdc881ada425d544107a5f7516023e";
const NATIVE_CORE: &str = "6b6efa2647bcbb63e1f0d0a3d96816ed5c43662d087795393437b02cbf46c390";

#[derive(Parser)]
struct Args {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    python_core: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    soxr_wrapper: PathBuf,
    #[arg(long)]
    soxr_library: PathBuf,
    #[arg(long)]
    runtime_proof: PathBuf,
    #[arg(long)]
    runtime_proof_sha256: String,
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(bytes)?;
    Ok(())
}

fn core_bytes(core: &[f32], frames: usize) -> Result<Vec<u8>> {
    ensure!(
        frames == 399 && core.len() == 192 * 399,
        "fixed core geometry"
    );
    ensure!(core.iter().all(|value| value.is_finite()), "nonfinite core");
    Ok(core.iter().flat_map(|value| value.to_le_bytes()).collect())
}

fn compare(actual: &[f32], expected: &[f32]) -> Result<Value> {
    ensure!(actual.len() == expected.len(), "reference core geometry");
    let mut count = 0;
    let mut maximum = 0.0f64;
    let mut first = Vec::new();
    for (offset, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        if a.to_bits() != b.to_bits() {
            count += 1;
            maximum = maximum.max((f64::from(a) - f64::from(b)).abs());
            if first.len() < 32 {
                first.push(json!({"bin":offset / 399,"frame":offset % 399}));
            }
        }
    }
    Ok(
        json!({"elements":actual.len(),"unequal_bits":count,"max_absolute_difference":maximum,"first_different_coordinates":first}),
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    check_file(&args.manifest, MANIFEST)?;
    check_file(&args.python_core, PYTHON_CORE)?;
    check_file(&args.runtime_proof, &args.runtime_proof_sha256)?;
    let proof: Value = serde_json::from_slice(&fs::read(&args.runtime_proof)?)?;
    ensure!(
        proof["format"] == "cqt-fused-product-runtime-proof-v1",
        "candidate runtime proof"
    );
    let files = proof["native_files"]
        .as_object()
        .context("native file pins")?;
    for role in [
        "executable",
        "candidate_source",
        "plan_source",
        "cargo_lock",
        "rustc",
        "cargo",
        "soxr_wrapper",
        "soxr_library",
        "soxr_wrapper_source",
        "soxr_source_receipt",
    ] {
        let record = files.get(role).context("missing candidate runtime role")?;
        check_file(
            Path::new(record["path"].as_str().context("runtime path")?),
            record["sha256"].as_str().context("runtime hash")?,
        )?;
    }
    ensure!(
        files["executable"]["sha256"] == file_sha256(&std::env::current_exe()?)?,
        "actual candidate executable"
    );
    ensure!(
        files["candidate_source"]["sha256"]
            == bytes_sha256(include_bytes!("fused_product_case.rs")),
        "compiled candidate source"
    );
    ensure!(
        files["plan_source"]["sha256"] == bytes_sha256(include_bytes!("../plan.rs")),
        "compiled plan source"
    );
    ensure!(
        files["cargo_lock"]["sha256"] == bytes_sha256(include_bytes!("../../Cargo.lock")),
        "compiled Cargo lock"
    );
    ensure!(
        proof["historical_gaps"]
            == json!([
                "scipy_fft_backend",
                "jit_runtime",
                "system_math",
                "rustfft_dispatch"
            ]),
        "retain historical gaps"
    );
    check_file(
        &args.soxr_wrapper,
        "49c7903d21711aff76afaaef122f7e71120f901427acb314f721f5b663b53880",
    )?;
    check_file(
        &args.soxr_library,
        "b98d5dcb7214509496968e72f42b1fa18a3365dce251f6eb56331c0c7f618445",
    )?;
    let input_root = args.manifest.parent().context("manifest parent")?;
    let manifest: Value = serde_json::from_slice(&fs::read(&args.manifest)?)?;
    let cases = manifest["cases"].as_array().context("cases")?;
    ensure!(
        cases.len() == 1 && cases[0]["id"] == "fmak:091711" && cases[0]["frames"] == 399,
        "fixed0308 only"
    );
    let case = &cases[0];
    ensure!(case["pcm"]["sha256"] == PCM, "fixed PCM identity");
    let pcm = read_f32(input_root, &case["pcm"])?;
    ensure!(pcm.len() == 1_633_354, "fixed PCM samples");
    let reference_bytes = fs::read(&args.python_core)?;
    ensure!(
        reference_bytes.len() == 192 * 399 * 4,
        "fixed Python core bytes"
    );
    let reference: Vec<f32> = reference_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    ensure!(
        reference.iter().all(|value| value.is_finite()),
        "nonfinite Python reference"
    );
    // Require a new output directory: no old arrays, reports or executables are overwritten.
    fs::create_dir(&args.output)?;
    let soxr = Soxr::new(&args.soxr_wrapper)?;
    let plan = PreparedPlan::load(input_root, case)?;
    let (baseline, frames) = plan.compute(&pcm, &soxr, None, true)?;
    let baseline_bytes = core_bytes(&baseline, frames)?;
    let baseline_hash = bytes_sha256(&baseline_bytes);
    write_new(
        &args.output.join("separate-product-core.f32"),
        &baseline_bytes,
    )?;
    let baseline_report = json!({"format":"cqt-fused-product-control-v1","case":"fmak:091711","product_rule":"separate","core_sha256":baseline_hash,"archived_native_core_preserved":baseline_hash == NATIVE_CORE,"python_comparison":compare(&baseline,&reference)?});
    write_new(
        &args.output.join("separate-product.json"),
        &serde_json::to_vec_pretty(&baseline_report)?,
    )?;
    ensure!(
        baseline_hash == NATIVE_CORE,
        "new build failed archived native control; candidate not admitted"
    );

    let (candidate, frames) = plan.research_fused_product_core(&pcm, &soxr)?;
    let candidate_bytes = core_bytes(&candidate, frames)?;
    let candidate_hash = bytes_sha256(&candidate_bytes);
    write_new(
        &args.output.join("fused-product-core.f32"),
        &candidate_bytes,
    )?;
    let report = json!({
        "format":"cqt-fused-product-case-v1", "case":"fmak:091711",
        "candidate_identity":"fused-product-separate-accumulation-v1",
        "manifest_sha256":MANIFEST, "pcm_sha256":PCM,
        "python_core_sha256":PYTHON_CORE, "archived_native_core_sha256":NATIVE_CORE,
        "runtime_proof_sha256":args.runtime_proof_sha256,
        "executable_sha256":file_sha256(&std::env::current_exe()?)?,
        "source_sha256":bytes_sha256(include_bytes!("fused_product_case.rs")),
        "plan_source_sha256":bytes_sha256(include_bytes!("../plan.rs")),
        "baseline":baseline_report, "candidate_core_sha256":candidate_hash,
        "candidate_matches_archived_python_core":candidate_hash == PYTHON_CORE,
        "candidate_python_comparison":compare(&candidate,&reference)?,
        "candidate_baseline_comparison":compare(&candidate,&baseline)?,
        "default_product_rule":"separate", "no_model_or_ort_invoked":true,
        "no_release_promotion":true, "historical_gaps":proof["historical_gaps"]
    });
    write_new(
        &args.output.join("result.json"),
        &serde_json::to_vec_pretty(&report)?,
    )?;
    // A completed mismatch is evidence, not a reason to change tolerances or retry.
    Ok(())
}
