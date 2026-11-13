//! Fixed-case frontend observations only. No ORT initialization or model API.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use drone_native_cached::{bytes_sha256, check_file, file_sha256};
use drone_native_cqt::{
    plan::{Intermediate, PreparedPlan, Soxr},
    read_f32,
};
use rustfft::num_complex::Complex;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

const MANIFEST: &str = "462205de24ff451e040afe45a3409cfc3fa9b74afc4f37d1725024cee59dc1fc";
const CORE: &str = "6b6efa2647bcbb63e1f0d0a3d96816ed5c43662d087795393437b02cbf46c390";
const LIMIT: u64 = 64 * 1024 * 1024;

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

fn owned_bytes(root: &Path) -> Result<u64> {
    let mut size = 0;
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        let meta = fs::symlink_metadata(&path)?;
        ensure!(!meta.file_type().is_symlink(), "trace output symlink");
        size += if meta.is_dir() {
            owned_bytes(&path)?
        } else {
            meta.len()
        };
    }
    Ok(size)
}

struct Writer {
    root: PathBuf,
    records: BTreeMap<String, Value>,
}
impl Writer {
    fn raw(&mut self, name: &str, data: Vec<u8>, dtype: &str, shape: &[usize]) -> Result<()> {
        ensure!(!self.records.contains_key(name), "duplicate boundary");
        let hash = bytes_sha256(&data);
        let relative = format!("objects/{hash}");
        let path = self.root.join(&relative);
        if path.exists() {
            ensure!(
                fs::symlink_metadata(&path)?.is_file(),
                "invalid existing object"
            );
            check_file(&path, &hash)?;
        } else {
            ensure!(
                owned_bytes(&self.root)? + data.len() as u64 <= LIMIT - 262144,
                "trace disk cap"
            );
            fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&path)?
                .write_all(&data)?;
        }
        self.records.insert(
            name.into(),
            json!({"path":relative,"sha256":hash,"dtype":dtype,"shape":shape,"bytes":data.len()}),
        );
        Ok(())
    }
    fn real(&mut self, name: &str, data: &[f32], shape: &[usize]) -> Result<()> {
        self.raw(
            name,
            data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            "<f4",
            shape,
        )
    }
    fn complex(
        &mut self,
        name: &str,
        data: &[Complex<f32>],
        rows: usize,
        frames: usize,
        probes: bool,
    ) -> Result<()> {
        ensure!(data.len() == rows * frames, "complex observation shape");
        self.raw(
            name,
            data.iter()
                .flat_map(|v| v.re.to_le_bytes().into_iter().chain(v.im.to_le_bytes()))
                .collect(),
            "<c8",
            &[rows, frames],
        )?;
        if probes {
            let width = frames.min(16);
            let mut starts = vec![0, (frames - width) / 2, frames - width];
            starts.sort();
            starts.dedup();
            for (index, start) in starts.iter().enumerate() {
                let values: Vec<_> = (0..rows)
                    .flat_map(|r| {
                        data[r * frames + *start..r * frames + *start + width]
                            .iter()
                            .copied()
                    })
                    .collect();
                self.complex(&format!("{name}.probe{index}"), &values, rows, width, false)?;
            }
        }
        Ok(())
    }
    fn resample(&mut self, name: &str, data: &[f32]) -> Result<()> {
        self.real(name, data, &[data.len()])?;
        let width = data.len().min(256);
        let mut starts = vec![0, (data.len() - width) / 2, data.len() - width];
        starts.sort();
        starts.dedup();
        for (index, start) in starts.iter().enumerate() {
            self.real(
                &format!("{name}.probe{index}"),
                &data[*start..*start + width],
                &[width],
            )?;
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    check_file(&args.manifest, MANIFEST)?;
    check_file(&args.runtime_proof, &args.runtime_proof_sha256)?;
    let proof: Value = serde_json::from_slice(&fs::read(&args.runtime_proof)?)?;
    ensure!(
        proof["format"] == "cqt-boundary-runtime-proof-v1",
        "runtime proof format"
    );
    let required = [
        "executable",
        "trace_source",
        "plan_source",
        "cargo_lock",
        "rustc",
        "cargo",
        "soxr_wrapper",
        "soxr_library",
        "soxr_wrapper_source",
        "soxr_source_receipt",
    ];
    let files = proof["native_files"]
        .as_object()
        .context("native file pins")?;
    for role in required {
        let record = files.get(role).context("missing native role")?;
        check_file(
            Path::new(record["path"].as_str().context("native path")?),
            record["sha256"].as_str().context("native hash")?,
        )?;
    }
    ensure!(
        files["executable"]["sha256"] == file_sha256(&std::env::current_exe()?)?,
        "actual executable pin"
    );
    ensure!(
        files["trace_source"]["sha256"] == bytes_sha256(include_bytes!("boundary_trace.rs")),
        "actual trace source pin"
    );
    ensure!(
        files["plan_source"]["sha256"] == bytes_sha256(include_bytes!("../plan.rs")),
        "actual plan source pin"
    );
    ensure!(
        files["cargo_lock"]["sha256"] == bytes_sha256(include_bytes!("../../Cargo.lock")),
        "actual Cargo lock pin"
    );
    ensure!(
        proof["historical_gaps"]
            == json!([
                "scipy_fft_backend",
                "jit_runtime",
                "system_math",
                "rustfft_dispatch"
            ]),
        "declare historical closure gaps"
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
    ensure!(
        case["pcm"]["sha256"] == "0b1234cb5e3c1ec2573d379660d7078ee7bffb1db029b87ffa58605836e2dde0",
        "fixed PCM hash"
    );
    let pcm = read_f32(input_root, &case["pcm"])?;
    ensure!(pcm.len() == 1633354, "fixed PCM length");
    if args.output.exists() {
        ensure!(
            !fs::symlink_metadata(&args.output)?.file_type().is_symlink(),
            "trace root symlink"
        );
    }
    fs::create_dir_all(args.output.join("objects"))?;
    owned_bytes(&args.output)?;
    let report = args.output.join("native-trace.json");
    ensure!(!report.exists(), "immutable native report exists");
    let mut writer = Writer {
        root: args.output.clone(),
        records: BTreeMap::new(),
    };
    let soxr = Soxr::new(&args.soxr_wrapper)?;
    let plan = PreparedPlan::load(input_root, case)?;
    let mut observer = |event: Intermediate<'_>| -> Result<()> {
        match event {
            Intermediate::Input(stage, values) => {
                writer.real(&format!("stage{stage}.input"), values, &[values.len()])
            }
            Intermediate::Spectrum(stage, frames, values) => writer.complex(
                &format!("stage{stage}.spectrum"),
                values,
                values.len() / frames,
                frames,
                false,
            ),
            Intermediate::Response(stage, values) => {
                let rows = case["stages"][stage]["basis_shape"][0]
                    .as_u64()
                    .context("stage rows")? as usize;
                writer.complex(
                    &format!("stage{stage}.response"),
                    values,
                    rows,
                    values.len() / rows,
                    true,
                )
            }
            Intermediate::Resampled(stage, values) => {
                writer.resample(&format!("after_stage{stage}.resampled"), values)
            }
            Intermediate::PreNormalization(frames, values) => {
                writer.complex("terminal.pre_normalization", values, 192, frames, false)
            }
            Intermediate::Scales(values) => writer.raw(
                "terminal.scales",
                values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                "<f8",
                &[192],
            ),
            Intermediate::Normalized(frames, values) => {
                writer.complex("terminal.normalized", values, 192, frames, false)
            }
            Intermediate::Magnitudes(values) => {
                writer.real("terminal.magnitude", values, &[192, 399])
            }
            Intermediate::Logged(values) => writer.real("terminal.log1p", values, &[192, 399]),
            Intermediate::Core(values) => {
                writer.real("terminal.core", values, &[192, 399])?;
                writer.raw(
                    "terminal.fp16",
                    values
                        .iter()
                        .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
                        .collect(),
                    "<f2",
                    &[192, 399],
                )
            }
        }
    };
    let (core, frames) = plan.trace(&pcm, &soxr, &mut observer)?;
    let core_hash = bytes_sha256(
        &core
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let preserved = frames == 399 && core_hash == CORE;
    for (suffix, count) in [(".spectrum", 9), (".response", 9), (".resampled", 7)] {
        ensure!(
            writer
                .records
                .keys()
                .filter(|name| name.ends_with(suffix))
                .count()
                == count,
            "incomplete boundary coverage"
        );
    }
    ensure!(
        writer
            .records
            .keys()
            .filter(|name| name.contains(".response.probe"))
            .count()
            == 27,
        "response probes"
    );
    ensure!(
        writer
            .records
            .keys()
            .filter(|name| name.contains(".resampled.probe"))
            .count()
            == 21,
        "resample probes"
    );
    let output = json!({"format":"cqt-boundary-trace-v1","implementation":"native","case":"fmak:091711","runtime_proof_sha256":args.runtime_proof_sha256,"manifest_sha256":MANIFEST,"executable_sha256":file_sha256(&std::env::current_exe()?)?,"source_sha256":bytes_sha256(include_bytes!("boundary_trace.rs")),"plan_source_sha256":bytes_sha256(include_bytes!("../plan.rs")),"records":writer.records,"core_sha256":core_hash,"archived_core_preserved":preserved,"no_model_or_ort_invoked":true});
    fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(report)?
        .write_all(&serde_json::to_vec_pretty(&output)?)?;
    ensure!(owned_bytes(&args.output)? <= LIMIT, "final trace disk cap");
    ensure!(
        preserved,
        "instrumentation/build/runtime changed archived native core; no causal admission"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_observation_keeps_full_array_and_three_exact_probes() -> Result<()> {
        let root = std::env::temp_dir().join(format!("cqt-boundary-writer-{}", std::process::id()));
        ensure!(!root.exists(), "test output collision");
        fs::create_dir_all(root.join("objects"))?;
        let mut writer = Writer {
            root: root.clone(),
            records: BTreeMap::new(),
        };
        let data: Vec<_> = (0..40)
            .map(|i| Complex::new(i as f32, -(i as f32)))
            .collect();
        let result = (|| -> Result<()> {
            writer.complex("stage0.response", &data, 2, 20, true)?;
            ensure!(writer.records.len() == 4, "full array plus probes");
            let full = &writer.records["stage0.response"];
            let raw = fs::read(root.join(full["path"].as_str().context("test path")?))?;
            ensure!(
                raw.len() == 40 * 8 && raw[8..12] == 1.0_f32.to_le_bytes(),
                "complex LE layout"
            );
            let probe = &writer.records["stage0.response.probe1"];
            let raw = fs::read(root.join(probe["path"].as_str().context("test path")?))?;
            ensure!(raw[..4] == 2.0_f32.to_le_bytes(), "middle frame offset");
            ensure!(
                raw[16 * 8..16 * 8 + 4] == 22.0_f32.to_le_bytes(),
                "row-major probe"
            );
            Ok(())
        })();
        fs::remove_dir_all(root)?;
        result
    }
}
