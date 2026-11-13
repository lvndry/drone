//! Fixed nine-case frontend candidate validation; no classifier or reference regeneration.
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

const MANIFEST: &str = "14bd55b7e8f6eb4a034cb05253cd1f5391eac3f438740a26edfc6596e17bb732";
const CONTROLS: [(&str, &str); 9] = [
    (
        "silence",
        "fd9243e1ba57263ed469c3bdbd7ade6ec5254e7ed924a9f5737fa44749933cc0",
    ),
    (
        "impulses",
        "41963cf9e240f4fb7c98518874ce6af9ca6e5ea5415094fac8c0bd61cfe4a5f5",
    ),
    (
        "tone",
        "5557c4aaccbd4320b650e65a4eb3170f7bc8fe29276f019d8d0143aece512272",
    ),
    (
        "major",
        "b9d2811f69dc6f6c4579359dc987fe70a10b0dd7d49113594c36523d9b31c51d",
    ),
    (
        "minor",
        "3488c45dfea7d427417af86287d4f1371129c00e495de1ebb77be562e5061532",
    ),
    (
        "noise",
        "73614cfaaca14a4ef3f106fc01624408eccd0512cef2a08de6bdd53b2c28b77b",
    ),
    (
        "fmak-110544",
        "196bc8a2c054dfc7ae254bda6f2eac50339eb4d78fbe76ed9a2a4bb93312a7e7",
    ),
    (
        "fmak-107354",
        "3587b441b3e280c854a4ef5c8594e7ae939265303c0b94c1dae47fb9b6459622",
    ),
    (
        "fmak-087373",
        "87b6019faa61bb55766f1a0df8d80e209e96317c29377b84d71fb0a197f4c482",
    ),
];

#[derive(Parser)]
struct Args {
    #[arg(long)]
    manifest: PathBuf,
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
fn comparison(actual: &[f32], reference: &[f32], frames: usize) -> Result<Value> {
    ensure!(
        frames > 0 && actual.len() == 192 * frames && actual.len() == reference.len(),
        "core geometry"
    );
    ensure!(
        actual.iter().chain(reference).all(|v| v.is_finite()),
        "nonfinite core"
    );
    let mut count = 0;
    let mut maximum = 0.0f64;
    let mut first = Vec::new();
    for (offset, (&a, &b)) in actual.iter().zip(reference).enumerate() {
        if a.to_bits() != b.to_bits() {
            count += 1;
            maximum = maximum.max((f64::from(a) - f64::from(b)).abs());
            if first.len() < 32 {
                first.push(json!({"bin":offset / frames,"frame":offset % frames}));
            }
        }
    }
    Ok(
        json!({"elements":actual.len(),"unequal_bits":count,"max_absolute_difference":maximum,"first_different_coordinates":first}),
    )
}
fn save_core(output: &Path, name: &str, core: &[f32]) -> Result<String> {
    let bytes: Vec<u8> = core.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_new(&output.join(name), &bytes)?;
    Ok(bytes_sha256(&bytes))
}
fn verify_runtime(args: &Args) -> Result<Value> {
    check_file(&args.runtime_proof, &args.runtime_proof_sha256)?;
    let proof: Value = serde_json::from_slice(&fs::read(&args.runtime_proof)?)?;
    ensure!(
        proof["format"] == "cqt-fused-goldens-runtime-proof-v1",
        "runtime proof format"
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
        let record = files.get(role).context("missing runtime role")?;
        check_file(
            Path::new(record["path"].as_str().context("runtime path")?),
            record["sha256"].as_str().context("runtime hash")?,
        )?;
    }
    ensure!(
        files["executable"]["sha256"] == file_sha256(&std::env::current_exe()?)?,
        "actual executable"
    );
    ensure!(
        files["candidate_source"]["sha256"]
            == bytes_sha256(include_bytes!("fused_golden_cases.rs")),
        "compiled candidate source"
    );
    ensure!(
        files["plan_source"]["sha256"] == bytes_sha256(include_bytes!("../plan.rs")),
        "compiled plan source"
    );
    ensure!(
        files["cargo_lock"]["sha256"] == bytes_sha256(include_bytes!("../../Cargo.lock")),
        "compiled lockfile"
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
    Ok(proof)
}
fn main() -> Result<()> {
    let args = Args::parse();
    check_file(&args.manifest, MANIFEST)?;
    let proof = verify_runtime(&args)?;
    let root = args.manifest.parent().context("manifest parent")?;
    let manifest: Value = serde_json::from_slice(&fs::read(&args.manifest)?)?;
    let cases = manifest["cases"].as_array().context("cases")?;
    ensure!(cases.len() == CONTROLS.len(), "fixed nine-case membership");
    for (case, (id, _)) in cases.iter().zip(CONTROLS) {
        ensure!(case["id"] == id, "fixed order");
    }
    fs::create_dir(&args.output)?;
    let soxr = Soxr::new(&args.soxr_wrapper)?;
    let mut reports = Vec::new();
    for (index, (case, (id, archived_native))) in cases.iter().zip(CONTROLS).enumerate() {
        let output = args.output.join(format!("{index:04}"));
        fs::create_dir(&output)?;
        let pcm = read_f32(root, &case["pcm"])?;
        let reference = read_f32(root, &case["core"])?;
        let frames = usize::try_from(case["frames"].as_u64().context("frames")?)?;
        let plan = PreparedPlan::load(root, case)?;
        let (baseline, actual_frames) = plan.compute(&pcm, &soxr, None, true)?;
        ensure!(actual_frames == frames, "baseline frame count");
        let baseline_comparison = comparison(&baseline, &reference, frames)?;
        let baseline_hash = save_core(&output, "separate-product-core.f32", &baseline)?;
        let admitted = baseline_hash == archived_native;
        let mut report = json!({"id":id,"frames":frames,"pcm_samples":pcm.len(),"manifest_core_sha256":case["core"]["sha256"],"archived_native_core_sha256":archived_native,"baseline_core_sha256":baseline_hash,"baseline_python_comparison":baseline_comparison,"original_control_preserved":admitted,"candidate_admitted":admitted});
        // Persist admission before candidate work; a failed control remains a failed case.
        write_new(
            &output.join("control.json"),
            &serde_json::to_vec_pretty(&report)?,
        )?;
        if admitted {
            let (candidate, actual_frames) = plan.research_fused_product_core(&pcm, &soxr)?;
            ensure!(actual_frames == frames, "candidate frame count");
            let candidate_comparison = comparison(&candidate, &reference, frames)?;
            let candidate_hash = save_core(&output, "fused-product-core.f32", &candidate)?;
            report["candidate_core_sha256"] = json!(candidate_hash);
            report["candidate_matches_archived_python_core"] =
                json!(report["candidate_core_sha256"] == case["core"]["sha256"]);
            report["candidate_python_comparison"] = candidate_comparison;
        } else {
            report["candidate_withheld_reason"] = json!("Original native control not reproduced");
        }
        write_new(
            &output.join("result.json"),
            &serde_json::to_vec_pretty(&report)?,
        )?;
        reports.push(report);
        // All per-case buffers/plans are dropped before processing the next fixed case.
    }
    let all_controls = reports
        .iter()
        .all(|r| r["original_control_preserved"] == true);
    let all_exact = reports
        .iter()
        .all(|r| r["candidate_matches_archived_python_core"] == true);
    let result = json!({"format":"cqt-fused-goldens-v1","complete":true,"manifest_sha256":MANIFEST,"runtime_proof_sha256":args.runtime_proof_sha256,"candidate_identity":"fused-product-separate-accumulation-v1","cases":reports,"all_controls_preserved":all_controls,"all_candidate_cores_exact":all_exact,"default_product_rule":"separate","no_model_or_ort_invoked":true,"no_release_promotion":true,"historical_gaps":proof["historical_gaps"]});
    write_new(
        &args.output.join("result.json"),
        &serde_json::to_vec_pretty(&result)?,
    )?;
    // Completed mismatches are evidence; the report, not exit status, carries the numerical verdict.
    Ok(())
}
