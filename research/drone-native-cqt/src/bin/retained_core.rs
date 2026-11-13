//! Fixed retained-core regression through the existing native full/chunked adapters.
//! No decoding, CQT, reference regeneration, training, or label scoring.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use drone_native_cached::{
    ATOL, Bundle, Chunked, Full, RTOL, bytes_sha256, check_file, compare, environment, file_sha256,
    process_usage,
};
use serde_json::{Value, json};
use std::{fs, io::Write, path::PathBuf};

const MODEL: &str = "35fc1bfe5d67f7aed2c46221c15519d0f3d5822351932b5bd9899d47d48cb613";
const CHECKPOINT: &str = "dcf8a7a292b7c12076ffff63c8c6550cb09a6c41ce4d000fd7949fc6bb4b452c";
const MEMBERSHIP: &str = "08a90fd7686ca354a6fecd30e65a5dca90a666d16fdd336590ba78b60b6459c2";
const ORT: &str = "d586cbe89effc085a09d3341b3cfbd5c322517f9d397aeb20814af21433f2b36";

#[derive(Parser)]
struct Args {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    manifest_sha256: String,
    #[arg(long)]
    full_model: PathBuf,
    #[arg(long)]
    bundle: PathBuf,
    #[arg(long)]
    ort_library: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn vector(value: &Value) -> Result<[f32; 24]> {
    let values = value.as_array().context("Reference array")?;
    ensure!(values.len() == 24, "Reference length");
    let mut result = [0.0; 24];
    for (target, value) in result.iter_mut().zip(values) {
        *target = value.as_f64().context("Reference number")? as f32;
    }
    ensure!(result.iter().all(|x| x.is_finite()), "Nonfinite reference");
    Ok(result)
}

fn run(args: &Args) -> Result<()> {
    check_file(&args.manifest, &args.manifest_sha256)?;
    let manifest: Value = serde_json::from_slice(&fs::read(&args.manifest)?)?;
    ensure!(
        manifest["schema"] == "drone-retained-model376-v1",
        "Manifest contract"
    );
    let membership_path = PathBuf::from(
        manifest["membership_path"]
            .as_str()
            .context("Membership path")?,
    );
    check_file(&membership_path, MEMBERSHIP)?;
    let membership: Value = serde_json::from_slice(&fs::read(membership_path)?)?;
    let members = membership["cases"].as_array().context("Membership rows")?;
    let cases = manifest["cases"].as_array().context("Cases")?;
    ensure!(
        cases.len() == 376 && members.len() == 376,
        "Exact 376 membership required"
    );
    check_file(&args.full_model, MODEL)?;
    check_file(&args.ort_library, ORT)?;
    let bundle = Bundle::load(&args.bundle, CHECKPOINT)?;
    let mut records = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(args.output.with_extension("jsonl"))?;
    ensure!(!args.output.exists(), "Terminal output already exists");
    let runtime = environment(&args.ort_library)?;
    let mut full = Full::new(&runtime, &args.full_model)?;
    let mut chunked = Chunked::new(&runtime, &args.bundle, &bundle)?;
    let mut rows = Vec::new();
    let mut passed = true;
    for (index, (case, member)) in cases.iter().zip(members).enumerate() {
        ensure!(
            member["index"] == index
                && case["index"] == index
                && case["id"] == member["id"]
                && case["frames"] == member["frames"],
            "Changed order/identity/frames"
        );
        let archived_path = PathBuf::from(
            case["mac_manifest_path"]
                .as_str()
                .context("Archived manifest path")?,
        );
        check_file(
            &archived_path,
            member["mac_manifest_sha256"]
                .as_str()
                .context("Archived manifest digest")?,
        )?;
        let archived: Value = serde_json::from_slice(&fs::read(archived_path)?)?;
        let reference = &archived["cases"][0];
        ensure!(
            archived["complete"] == true
                && archived["full_model_sha256"] == MODEL
                && reference["id"] == member["id"]
                && reference["frames"] == member["frames"]
                && reference["core"]["sha256"] == member["archived_mac_core_sha256"],
            "Archived reference lineage"
        );
        let expected = vector(&reference["reference_logits"])?;
        let frames = usize::try_from(member["frames"].as_u64().context("Frames")?)?;
        let core_path = PathBuf::from(case["core_path"].as_str().context("Core path")?);
        ensure!(!core_path.is_symlink(), "Core symlink");
        let raw = fs::read(&core_path)?;
        ensure!(
            raw.len() == frames.checked_mul(192 * 4).context("Size overflow")?
                && bytes_sha256(&raw) == member["archived_mac_core_sha256"],
            "Candidate core shape/hash"
        );
        let core: Vec<f32> = raw
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect();
        ensure!(core.iter().all(|x| x.is_finite()), "Nonfinite core");
        let full_logits = full.predict(&core, frames)?;
        let chunked_logits = chunked.predict(&core, frames)?;
        let chunked_vs_full = compare(&chunked_logits, &full_logits)?;
        passed &= chunked_vs_full.allclose && chunked_vs_full.argmax_agrees;
        let mut predictions = Vec::new();
        for (backend, prediction) in [("full", full_logits), ("chunked", chunked_logits)] {
            let comparison = compare(&prediction, &expected)?;
            passed &= comparison.allclose && comparison.argmax_agrees;
            let failed_elements: Vec<_> = prediction
                .iter()
                .zip(expected)
                .enumerate()
                .filter_map(|(i, (&actual, reference))| {
                    let error = (f64::from(actual) - f64::from(reference)).abs();
                    (error > f64::from(ATOL) + f64::from(RTOL) * f64::from(reference).abs())
                        .then_some(i)
                })
                .collect();
            predictions.push(json!({"backend":backend,"logits":prediction,"comparison":comparison,"failed_elements":failed_elements}));
        }
        let row = json!({"index":index,"id":member["id"],"frames":frames,"core_sha256":member["archived_mac_core_sha256"],"reference_logits":expected,"predictions":predictions,"chunked_vs_full":chunked_vs_full});
        writeln!(records, "{}", serde_json::to_string(&row)?)?;
        records.flush()?;
        rows.push(row);
    }
    let (cpu, rss_kib) = process_usage()?;
    let result = json!({"complete":true,"passed":passed,"tracks":rows.len(),"cases":rows,"atol":ATOL,"rtol":RTOL,"cpu_seconds":cpu,"peak_rss_bytes":rss_kib*1024,"model_sha256":MODEL,"membership_sha256":MEMBERSHIP,"manifest_sha256":args.manifest_sha256,"ort_sha256":ORT,"native_library_source_sha256":bytes_sha256(include_bytes!("../../../drone-native-cached/src/lib.rs")),"entrypoint_source_sha256":bytes_sha256(include_bytes!("retained_core.rs")),"executable_sha256":file_sha256(&std::env::current_exe()?)?,"decode_or_cqt_executed":false,"reference_regenerated":false,"key_accuracy_scored":false,"release_promotion":false});
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    serde_json::to_writer_pretty(&mut output, &result)?;
    writeln!(output)?;
    ensure!(
        passed,
        "Strict numerical regression failed; all 376 comparisons retained"
    );
    Ok(())
}

fn main() -> Result<()> {
    run(&Args::parse())
}
