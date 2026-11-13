//! One admitted member of the fixed public376 frontend panel; no Python reference or model.
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

const MEMBERSHIP: &[u8] = include_bytes!("../../composed/FUSED_376_MEMBERSHIP.json");

#[derive(Parser)]
struct Args {
    #[arg(long)]
    index: usize,
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
fn save_core(output: &Path, name: &str, core: &[f32]) -> Result<String> {
    let bytes: Vec<u8> = core.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_new(&output.join(name), &bytes)?;
    Ok(bytes_sha256(&bytes))
}
fn verify_runtime(args: &Args) -> Result<Value> {
    check_file(&args.runtime_proof, &args.runtime_proof_sha256)?;
    let proof: Value = serde_json::from_slice(&fs::read(&args.runtime_proof)?)?;
    ensure!(
        proof["format"] == "cqt-fused-376-runtime-proof-v1",
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
        files["candidate_source"]["sha256"] == bytes_sha256(include_bytes!("fused_public_case.rs")),
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
    let membership: Value = serde_json::from_slice(MEMBERSHIP)?;
    let rows = membership["cases"].as_array().context("fixed membership")?;
    ensure!(
        rows.len() == 376 && args.index < rows.len(),
        "fixed public376 index"
    );
    let row = &rows[args.index];
    ensure!(
        row["index"].as_u64() == Some(args.index as u64),
        "fixed ordered membership"
    );
    check_file(
        &args.manifest,
        row["mac_manifest_sha256"]
            .as_str()
            .context("manifest hash")?,
    )?;
    let proof = verify_runtime(&args)?;
    ensure!(
        proof["membership_sha256"] == bytes_sha256(MEMBERSHIP),
        "compiled membership identity"
    );
    let root = args.manifest.parent().context("manifest parent")?;
    let manifest: Value = serde_json::from_slice(&fs::read(&args.manifest)?)?;
    let cases = manifest["cases"].as_array().context("case array")?;
    ensure!(cases.len() == 1, "one restored case per worker");
    let case = &cases[0];
    ensure!(
        case["id"] == row["id"] && case["frames"] == row["frames"] && case["public"] == true,
        "case membership"
    );
    ensure!(
        case["pcm"]["sha256"] == row["pcm_sha256"] && case["pcm"]["bytes"] == row["pcm_bytes"],
        "archived PCM identity"
    );
    ensure!(
        case["core"]["sha256"] == row["archived_mac_core_sha256"],
        "archived reference identity"
    );
    let pcm = read_f32(root, &case["pcm"])?;
    ensure!(
        Some(pcm.len() as u64) == row["pcm_samples"].as_u64(),
        "PCM sample count"
    );
    ensure!(pcm.iter().all(|v| v.is_finite()), "nonfinite PCM");
    let frames = usize::try_from(row["frames"].as_u64().context("frames")?)?;
    fs::create_dir(&args.output)?;
    let soxr = Soxr::new(&args.soxr_wrapper)?;
    let plan = PreparedPlan::load(root, case)?;
    let (baseline, actual_frames) = plan.compute(&pcm, &soxr, None, true)?;
    ensure!(
        actual_frames == frames
            && baseline.len() == 192 * frames
            && baseline.iter().all(|v| v.is_finite()),
        "baseline core geometry/finiteness"
    );
    let baseline_hash = save_core(&args.output, "separate-product-core.f32", &baseline)?;
    let admitted = row["archived_native_core_sha256"] == baseline_hash;
    let mut report = json!({
        "format":"cqt-fused-public-case-v1", "index":args.index, "id":row["id"],
        "frames":frames, "core_elements":baseline.len(), "pcm_sha256":row["pcm_sha256"],
        "membership_sha256":bytes_sha256(MEMBERSHIP), "manifest_sha256":row["mac_manifest_sha256"],
        "runtime_proof_sha256":args.runtime_proof_sha256,
        "archived_mac_core_sha256":row["archived_mac_core_sha256"],
        "archived_linux_core_sha256":row["archived_linux_core_sha256"],
        "archived_native_core_sha256":row["archived_native_core_sha256"],
        "baseline_core_sha256":baseline_hash, "original_control_preserved":admitted,
        "candidate_admitted":admitted, "default_product_rule":"separate",
        "no_model_or_ort_invoked":true, "no_new_python_reference":true,
        "no_release_promotion":true, "historical_gaps":proof["historical_gaps"]
    });
    write_new(
        &args.output.join("control.json"),
        &serde_json::to_vec_pretty(&report)?,
    )?;
    if admitted {
        let (candidate, actual_frames) = plan.research_fused_product_core(&pcm, &soxr)?;
        ensure!(
            actual_frames == frames
                && candidate.len() == baseline.len()
                && candidate.iter().all(|v| v.is_finite()),
            "candidate core geometry/finiteness"
        );
        let candidate_hash = save_core(&args.output, "fused-product-core.f32", &candidate)?;
        let exact = row["archived_mac_core_sha256"] == candidate_hash;
        let linux_exact = row["archived_linux_core_sha256"] == candidate_hash;
        report["candidate_core_sha256"] = json!(candidate_hash);
        report["candidate_matches_archived_mac_core"] = json!(exact);
        report["candidate_matches_archived_linux_core"] = json!(linux_exact);
        report["primary_reference"] = json!("archived_mac_python");
        report["linux_comparison_is_secondary"] = json!(true);
        report["candidate_reference_unequal_values"] = if exact { json!(0) } else { Value::Null };
        report["candidate_reference_max_absolute_difference"] =
            if exact { json!(0.0) } else { Value::Null };
        report["reference_comparison_note"] = json!(
            "Reference hash is retained; full archived reference array is absent. A different hash fails exact equality; its element count/error magnitude is unavailable."
        );
    } else {
        report["candidate_withheld_reason"] = json!("Original native control not reproduced");
    }
    write_new(
        &args.output.join("result.json"),
        &serde_json::to_vec_pretty(&report)?,
    )?;
    // Completed numerical failure remains a result; the fixed controller continues other cases.
    Ok(())
}
