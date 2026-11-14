// SPDX-License-Identifier: GPL-3.0-or-later
//! Opt-in, finite original-file measurement. No default frontend changes.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use drone_native_cached::{
    Bundle, Chunked, argmax, check_file, compare, environment, process_usage,
};
use drone_native_cqt::{
    artifact_root,
    plan::{PreparedPlan, Soxr},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, path::PathBuf, process::Command, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    case_index: usize,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    execute_research_candidate: bool,
}
const PANEL: &str = include_str!("../../composed/END_TO_END_PANEL.json");
const CHECKPOINT: &str = "dcf8a7a292b7c12076ffff63c8c6550cb09a6c41ce4d000fd7949fc6bb4b452c";

fn pinned(value: &Value) -> Result<PathBuf> {
    let path = PathBuf::from(value["path"].as_str().context("pinned path")?);
    ensure!(path.is_absolute(), "pinned paths must be absolute");
    check_file(&path, value["sha256"].as_str().context("pinned SHA256")?)?;
    Ok(path)
}

fn cpu_seconds() -> Result<f64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes this live output on successful return.
    let status = unsafe { libc::getrusage(libc::RUSAGE_CHILDREN, usage.as_mut_ptr()) };
    ensure!(status == 0, "child rusage");
    // SAFETY: the call above succeeded.
    let usage = unsafe { usage.assume_init() };
    Ok(process_usage()?.0
        + usage.ru_utime.tv_sec as f64
        + usage.ru_utime.tv_usec as f64 / 1e6
        + usage.ru_stime.tv_sec as f64
        + usage.ru_stime.tv_usec as f64 / 1e6)
}

/// Parse the same S16 mono WAV contract as drone.features; reject malformed chunks.
fn wav_pcm(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure!(
        bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "RIFF/WAVE required"
    );
    let mut pos = 12usize;
    let mut format_seen = false;
    while pos + 8 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into()?) as usize;
        let end = (pos + 8).checked_add(size).context("WAV overflow")?;
        ensure!(end <= bytes.len(), "truncated WAV");
        let body = &bytes[pos + 8..end];
        if &bytes[pos..pos + 4] == b"fmt " {
            ensure!(body.len() >= 16, "short fmt");
            ensure!(
                u16::from_le_bytes(body[0..2].try_into()?) == 1
                    && u16::from_le_bytes(body[2..4].try_into()?) == 1
                    && u32::from_le_bytes(body[4..8].try_into()?) == 22050
                    && u16::from_le_bytes(body[14..16].try_into()?) == 16,
                "expected PCM mono 22050 S16"
            );
            format_seen = true;
        } else if &bytes[pos..pos + 4] == b"data" {
            ensure!(
                format_seen && !body.is_empty() && body.len().is_multiple_of(2),
                "invalid data"
            );
            return Ok(body
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from(i16::from_le_bytes(*b)) / 32768.0)
                .collect());
        }
        pos = end.checked_add(size % 2).context("WAV padding overflow")?;
    }
    anyhow::bail!("missing WAV data")
}

fn float_hash(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    // Keep evidence hashing from adding a second full PCM allocation to measured RSS.
    for chunk in values.chunks(4096) {
        let mut bytes = [0u8; 4096 * 4];
        for (value, target) in chunk.iter().zip(bytes.as_chunks_mut::<4>().0) {
            target.copy_from_slice(&value.to_le_bytes());
        }
        hash.update(&bytes[..chunk.len() * 4]);
    }
    format!("{:x}", hash.finalize())
}

fn reference(value: &Value) -> Result<[f32; 24]> {
    let values = value
        .as_array()
        .context("reference vector")?
        .iter()
        .map(|v| {
            v.as_f64()
                .filter(|v| v.is_finite())
                .map(|v| v as f32)
                .context("finite logit")
        })
        .collect::<Result<Vec<_>>>()?;
    values
        .try_into()
        .map_err(|_| anyhow::anyhow!("24 reference logits required"))
}

fn decoder_command(ffmpeg: &Path, audio: &Path, output: &Path) -> Command {
    // Prospective thread policy; unchanged decode/filter format, checked against historical PCM.
    let mut command = Command::new(ffmpeg);
    command
        .args([
            "-loglevel",
            "error",
            "-y",
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-i",
        ])
        .arg(audio)
        .args(["-ac", "1", "-ar", "22050"])
        .arg(output)
        .stdin(std::process::Stdio::null());
    command
}

fn decode(ffmpeg: &Path, audio: &Path, output: &Path) -> Result<Vec<f32>> {
    let status = decoder_command(ffmpeg, audio, output).status()?;
    ensure!(status.success(), "FFmpeg failed");
    wav_pcm(&fs::read(output)?)
}

fn decoder_lane(config: &Value) -> Result<bool> {
    if config["format"] == "drone-original-file-measurement-v1" {
        ensure!(
            config["input_lane"].is_null() || config["input_lane"] == "canonical",
            "v1 remains canonical-only"
        );
        return Ok(false);
    }
    ensure!(
        config["format"] == "drone-original-file-measurement-v2"
            && config["input_lane"] == "decoder_fidelity",
        "explicit v2 decoder-fidelity lane required"
    );
    Ok(true)
}

fn admit_decoded_pcm(
    pcm: &[f32],
    expected: &Value,
    decoder_fidelity: bool,
    observed: &mut Option<(usize, String)>,
) -> Result<Value> {
    ensure!(!pcm.is_empty(), "empty decoded PCM");
    ensure!(
        Some(pcm.len() as u64) == expected["pcm_samples"].as_u64(),
        "PCM length changed"
    );
    let hash = float_hash(pcm);
    let canonical_matches = Some(pcm.len() as u64) == expected["pcm_samples"].as_u64()
        && Some(hash.as_str()) == expected["pcm_sha256"].as_str();
    ensure!(decoder_fidelity || canonical_matches, "decoded PCM changed");
    if let Some((samples, previous_hash)) = observed {
        ensure!(
            *samples == pcm.len() && *previous_hash == hash,
            "local decoder changed between repetitions"
        );
    } else {
        *observed = Some((pcm.len(), hash.clone()));
    }
    Ok(json!({"samples":pcm.len(),"sha256":hash,"canonical_pcm_matches":canonical_matches}))
}

fn retire_after_geometry(frames: usize, expected: &Value, wav: &Path) -> Result<()> {
    ensure!(
        Some(frames as u64) == expected["frames"].as_u64(),
        "frame count changed"
    );
    fs::remove_file(wav)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.execute_research_candidate,
        "explicit research execution opt-in required"
    );
    ensure!(!args.output.exists(), "output already exists");
    let config: Value = serde_json::from_slice(&fs::read(&args.config)?)?;
    let decoder_fidelity = decoder_lane(&config)?;
    ensure!(
        config["warm_repetitions"] == 3,
        "fixed three warm repetitions"
    );
    let panel: Value = serde_json::from_str(PANEL)?;
    let expected = panel["cases"]
        .as_array()
        .context("panel")?
        .get(args.case_index)
        .context("case index")?;
    let input = config["cases"]
        .as_array()
        .context("cases")?
        .get(args.case_index)
        .context("configured case")?;
    ensure!(input["id"] == expected["id"], "panel order");
    let audio = pinned(&input["audio"])?;
    ensure!(
        input["audio"]["sha256"] == expected["audio_sha256"],
        "original audio identity"
    );
    let manifest_path = pinned(&input["manifest"])?;
    ensure!(
        input["manifest"]["sha256"] == expected["mac_manifest_sha256"],
        "historical plan identity"
    );
    let ffmpeg = pinned(&config["ffmpeg"])?;
    let ort = pinned(&config["ort"])?;
    let soxr_path = pinned(&config["soxr"])?;
    let bundle_contract = pinned(&config["bundle_contract"])?;
    ensure!(
        bundle_contract
            .file_name()
            .is_some_and(|s| s == "contract.json"),
        "bundle contract filename"
    );
    let bundle_root = bundle_contract.parent().context("bundle root")?;
    let manifest: Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    let case = manifest["cases"]
        .as_array()
        .context("plan cases")?
        .first()
        .context("plan case")?;
    ensure!(
        manifest["cases"].as_array().context("plan cases")?.len() == 1,
        "single plan"
    );
    let mac_reference = reference(&case["reference_logits"])?;
    let linux_reference = reference(&case["canonical_linux_logits"])?;
    let total_started = Instant::now();
    let total_cpu = cpu_seconds()?;
    let setup_started = Instant::now();
    let setup_cpu = cpu_seconds()?;
    let runtime = environment(&ort)?;
    let bundle = Bundle::load(bundle_root, CHECKPOINT)?;
    let mut model = Chunked::new(&runtime, bundle_root, &bundle)?;
    let soxr = Soxr::new(&soxr_path)?;
    let plan = PreparedPlan::load(artifact_root(&manifest_path)?, case)?;
    let setup = json!({"wall_seconds":setup_started.elapsed().as_secs_f64(),"cpu_seconds":cpu_seconds()?-setup_cpu});
    let wav = args.output.with_extension("wav");
    ensure!(!wav.exists(), "owned WAV exists");
    let mut runs = Vec::new();
    let mut observed_pcm = None;
    for repetition in 0..4 {
        let started = Instant::now();
        let cpu = cpu_seconds()?;
        let stage = Instant::now();
        let stage_cpu = cpu_seconds()?;
        let pcm = decode(&ffmpeg, &audio, &wav)?;
        let decode_stage = json!({"wall_seconds":stage.elapsed().as_secs_f64(),"cpu_seconds":cpu_seconds()?-stage_cpu});
        let verify = Instant::now();
        let pcm_identity = admit_decoded_pcm(&pcm, expected, decoder_fidelity, &mut observed_pcm)?;
        let verify_seconds = verify.elapsed().as_secs_f64();
        let stage = Instant::now();
        let stage_cpu = cpu_seconds()?;
        let (core, frames) = plan.research_fused_product_core(&pcm, &soxr)?;
        let cqt = json!({"wall_seconds":stage.elapsed().as_secs_f64(),"cpu_seconds":cpu_seconds()?-stage_cpu});
        let retirement = Instant::now();
        retire_after_geometry(frames, expected, &wav)?;
        let retirement_seconds = retirement.elapsed().as_secs_f64();
        let stage = Instant::now();
        let stage_cpu = cpu_seconds()?;
        let logits = model.predict(&core, frames)?;
        let ort_stage = json!({"wall_seconds":stage.elapsed().as_secs_f64(),"cpu_seconds":cpu_seconds()?-stage_cpu});
        let wall = started.elapsed().as_secs_f64();
        let used_cpu = cpu_seconds()? - cpu;
        let core_hash = float_hash(&core);
        runs.push(json!({"repetition":repetition,"kind":if repetition==0 {"first_in_fresh_process"} else {"warm_session"},
            "pcm":pcm_identity,"frames":frames,"canonical_frames_match":Some(frames as u64)==expected["frames"].as_u64(),
            "wav_retirement_wall_seconds":retirement_seconds,
            "decode":decode_stage,"identity_verification_wall_seconds":verify_seconds,"cqt":cqt,"ort":ort_stage,
            "wall_seconds":wall,"cpu_seconds":used_cpu,"logits":logits,"key_id":argmax(&logits)?,
            "comparison_mac":compare(&logits,&mac_reference)?,"comparison_linux":compare(&logits,&linux_reference)?,
            "core_sha256":core_hash,"matches_mac_core":Some(core_hash.as_str())==expected["archived_mac_core_sha256"].as_str(),
            "matches_linux_core":Some(core_hash.as_str())==expected["archived_linux_core_sha256"].as_str()}));
    }
    let numerical_pass = runs.iter().all(|run| {
        ["comparison_mac", "comparison_linux"]
            .iter()
            .all(|name| run[name]["allclose"] == true)
    });
    let unchanged_key = runs.iter().all(|run| {
        ["comparison_mac", "comparison_linux"]
            .iter()
            .all(|name| run[name]["argmax_agrees"] == true)
    });
    let result = json!({"format":"drone-original-file-result-v1","complete":true,"id":expected["id"],
        "timing_complete":true,"input_lane":if decoder_fidelity {"decoder_fidelity"} else {"canonical"},
        "canonical_pcm_matches":runs.iter().all(|r|r["pcm"]["canonical_pcm_matches"]==true),
        "historical_reference_qualified":numerical_pass && unchanged_key,
        "numerical_pass":numerical_pass,"unchanged_key":unchanged_key,"benchmark_qualified":numerical_pass && unchanged_key,
        "setup":setup,"runs":runs,"measurement_wall_seconds":total_started.elapsed().as_secs_f64(),
        "measurement_cpu_seconds":cpu_seconds()?-total_cpu,"process_cpu_seconds_including_waited_children":cpu_seconds()?,
        "self_peak_rss_bytes":process_usage()?.1*1024,"note":"CPU includes waited decoder children; RSS here is worker-only. Supervisor reports sampled process-tree RSS. Admission and diagnostic hashes are not production work."});
    fs::write(&args.output, serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Vec<u8> {
        let mut b = b"RIFF\0\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0".to_vec();
        b.extend(22050u32.to_le_bytes());
        b.extend(44100u32.to_le_bytes());
        b.extend([2, 0, 16, 0]);
        b.extend(b"data\x06\0\0\0");
        for x in [i16::MIN, 0, i16::MAX] {
            b.extend(x.to_le_bytes());
        }
        b
    }
    #[test]
    fn s16_conversion_is_exact() -> Result<()> {
        assert_eq!(wav_pcm(&fixture())?, vec![-1.0, 0.0, 32767.0 / 32768.0]);
        Ok(())
    }
    #[test]
    fn rejects_truncation_and_wrong_sample_rate() {
        let mut b = fixture();
        b.pop();
        assert!(wav_pcm(&b).is_err());
        let mut b = fixture();
        b[24..28].copy_from_slice(&44100u32.to_le_bytes());
        assert!(wav_pcm(&b).is_err());
    }

    #[test]
    fn decoder_threads_are_input_scoped_and_output_contract_is_unchanged() {
        let cmd = decoder_command(
            Path::new("ffmpeg"),
            Path::new("input.mp3"),
            Path::new("out.wav"),
        );
        let args: Vec<_> = cmd
            .get_args()
            .map(|v| v.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            args,
            [
                "-loglevel",
                "error",
                "-y",
                "-threads",
                "1",
                "-filter_threads",
                "1",
                "-i",
                "input.mp3",
                "-ac",
                "1",
                "-ar",
                "22050",
                "out.wav"
            ]
        );
    }

    #[test]
    fn bounded_hash_matches_canonical_bytes_across_chunk_boundaries() {
        let values: Vec<_> = (0..8200).map(|v| v as f32 / 32768.0).collect();
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            float_hash(&values),
            drone_native_cached::bytes_sha256(&bytes)
        );
    }

    #[test]
    fn decoder_evidence_is_retained_until_canonical_pcm_admission() -> Result<()> {
        use std::io::Write;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "drone-decoder-admission-{}-{nonce}.wav",
            std::process::id()
        ));
        fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?
            .write_all(b"retained decoder evidence")?;
        let pcm = [0.0f32, 1.0];
        let wrong = json!({"pcm_samples":2,"pcm_sha256":"wrong"});
        assert!(admit_decoded_pcm(&pcm, &wrong, false, &mut None).is_err());
        assert_eq!(fs::read(&path)?, b"retained decoder evidence");
        let right = json!({"pcm_samples":2,"pcm_sha256":float_hash(&pcm),"frames":1});
        admit_decoded_pcm(&pcm, &right, false, &mut None)?;
        assert!(retire_after_geometry(2, &right, &path).is_err());
        assert!(path.exists());
        retire_after_geometry(1, &right, &path)?;
        assert!(!path.exists());
        Ok(())
    }

    #[test]
    fn prospective_lane_cannot_relabel_v1_and_requires_repeatable_pcm() -> Result<()> {
        assert!(decoder_lane(&json!({"format":"drone-original-file-measurement-v1","input_lane":"decoder_fidelity"})).is_err());
        assert!(decoder_lane(&json!({"format":"drone-original-file-measurement-v2"})).is_err());
        assert!(decoder_lane(
            &json!({"format":"drone-original-file-measurement-v2","input_lane":"decoder_fidelity"})
        )?);
        let path =
            std::env::temp_dir().join(format!("drone-repeatable-pcm-{}.wav", std::process::id()));
        fs::write(&path, b"retained mismatch")?;
        let mut observed = Some((1, float_hash(&[0.0])));
        let expected = json!({"pcm_samples":1,"pcm_sha256":"historical","frames":1});
        assert!(admit_decoded_pcm(&[0.0, 0.0], &expected, true, &mut None).is_err());
        assert!(admit_decoded_pcm(&[1.0], &expected, true, &mut observed).is_err());
        assert!(path.exists());
        let identity = admit_decoded_pcm(&[0.0], &expected, true, &mut observed)?;
        assert_eq!(identity["canonical_pcm_matches"], false);
        retire_after_geometry(1, &expected, &path)?;
        assert!(!path.exists());
        Ok(())
    }
}
