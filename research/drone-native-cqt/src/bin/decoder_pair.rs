// SPDX-License-Identifier: GPL-3.0-or-later
//! One fixed decoder comparison. No decode, labels, training or default changes.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use drone_native_cached::{
    Bundle, Chunked, argmax, bytes_sha256, check_file, compare, environment, key_name,
    process_usage,
};
use drone_native_cqt::{
    artifact_root,
    plan::{PreparedPlan, Soxr},
};
use serde_json::{Value, json};
use std::{fs, path::PathBuf, time::Instant};

const PANEL: &str = include_str!("../../composed/END_TO_END_PANEL.json");
const CHECKPOINT: &str = "dcf8a7a292b7c12076ffff63c8c6550cb09a6c41ce4d000fd7949fc6bb4b452c";

#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    config_sha256: String,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    execute: bool,
}

fn pin(value: &Value) -> Result<PathBuf> {
    let path = PathBuf::from(value["path"].as_str().context("path")?);
    ensure!(path.is_absolute(), "absolute pinned path required");
    check_file(&path, value["sha256"].as_str().context("SHA256")?)?;
    Ok(path)
}

fn pcm(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure!(
        !bytes.is_empty() && bytes.len().is_multiple_of(4),
        "nonempty aligned float32 PCM required"
    );
    let samples: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    ensure!(samples.iter().all(|x| x.is_finite()), "nonfinite PCM");
    Ok(samples)
}

fn vector(value: &Value) -> Result<[f32; 24]> {
    value
        .as_array()
        .context("reference vector")?
        .iter()
        .map(|v| v.as_f64().map(|x| x as f32).context("logit"))
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| anyhow::anyhow!("24 logits required"))
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.execute, "explicit paired-PCM execution required");
    ensure!(!args.output.exists(), "retain prior output");
    check_file(&args.config, &args.config_sha256)?;
    let config: Value = serde_json::from_slice(&fs::read(&args.config)?)?;
    ensure!(config["format"] == "drone-decoder-pair-v1", "config format");
    let panel: Value = serde_json::from_str(PANEL)?;
    let expected = &panel["cases"][0];
    ensure!(config["id"] == expected["id"], "fixed case0 only");
    let manifest = pin(&config["manifest"])?;
    ensure!(
        config["manifest"]["sha256"] == expected["mac_manifest_sha256"],
        "historical plan"
    );
    let source: Value = serde_json::from_slice(&fs::read(&manifest)?)?;
    ensure!(
        source["cases"].as_array().context("cases")?.len() == 1,
        "one plan"
    );
    let case = &source["cases"][0];
    let canonical = pin(&config["canonical_pcm"])?;
    let mac = pin(&config["mac_pcm"])?;
    ensure!(
        config["canonical_pcm"]["sha256"] == expected["pcm_sha256"],
        "canonical PCM identity"
    );
    let canonical_pcm = pcm(&fs::read(canonical)?)?;
    let mac_pcm = pcm(&fs::read(mac)?)?;
    ensure!(
        Some(canonical_pcm.len() as u64) == expected["pcm_samples"].as_u64()
            && mac_pcm.len() == canonical_pcm.len(),
        "fixed diagnostic geometry"
    );
    let ort = pin(&config["ort"])?;
    let soxr = pin(&config["soxr"])?;
    let contract = pin(&config["bundle_contract"])?;
    ensure!(
        contract.file_name().is_some_and(|n| n == "contract.json"),
        "contract filename"
    );
    let bundle_root = contract.parent().context("bundle root")?;
    let runtime = environment(&ort)?;
    let bundle = Bundle::load(bundle_root, CHECKPOINT)?;
    let mut model = Chunked::new(&runtime, bundle_root, &bundle)?;
    let soxr = Soxr::new(&soxr)?;
    let plan = PreparedPlan::load(artifact_root(&manifest)?, case)?;
    let mac_reference = vector(&case["reference_logits"])?;
    let linux_reference = vector(&case["canonical_linux_logits"])?;
    let mut rows = Vec::new();
    let mut predictions = Vec::new();
    for (lane, samples) in [("canonical", &canonical_pcm), ("mac", &mac_pcm)] {
        let start = Instant::now();
        let (core, frames) = plan.research_fused_product_core(samples, &soxr)?;
        let hash = bytes_sha256(
            &core
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let canonical_core_matches =
            Some(hash.as_str()) == expected["archived_mac_core_sha256"].as_str();
        ensure!(
            lane != "canonical" || canonical_core_matches,
            "canonical native core no longer reproduces archived Mac reference"
        );
        let logits = model.predict(&core, frames)?;
        let key = argmax(&logits)?;
        rows.push(json!({"lane":lane,"samples":samples.len(),"frames":frames,"core_sha256":hash,
            "canonical_core_matches":canonical_core_matches,"logits":logits,"key_id":key,"key":key_name(key)?,
            "comparison_mac_reference":compare(&logits,&mac_reference)?,"comparison_linux_reference":compare(&logits,&linux_reference)?,
            "diagnostic_compute_wall_seconds":start.elapsed().as_secs_f64()}));
        predictions.push(logits);
    }
    let result = json!({"format":"drone-decoder-pair-result-v1","complete":true,"id":expected["id"],
        "config_sha256":args.config_sha256,"inputs":{"canonical":config["canonical_pcm"],"mac":config["mac_pcm"]},
        "lanes":rows,"decoder_prediction_drift":compare(&predictions[1],&predictions[0])?,
        "original_tolerances":{"atol":1e-5,"rtol":1e-5},"key_accuracy_scored":false,"reference_regenerated":false,
        "no_decode":true,"cpu_seconds":process_usage()?.0,"self_peak_rss_bytes":process_usage()?.1*1024});
    fs::write(args.output, serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_malformed_pcm() {
        assert!(pcm(&[]).is_err());
        assert!(pcm(&[1, 2, 3]).is_err());
        assert!(pcm(&f32::NAN.to_le_bytes()).is_err());
    }
    #[test]
    fn reads_exact_little_endian_pcm() -> Result<()> {
        assert_eq!(pcm(&[0, 0, 128, 63, 0, 0, 128, 191])?, vec![1.0, -1.0]);
        Ok(())
    }
}
