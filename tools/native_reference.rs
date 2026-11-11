//! Task-local Euphoria benchmark driver. Copy into euphoria-bench/src/bin/.
//! Detector and decoder are the unmodified Euphoria implementations.
use anyhow::{Context, Result, ensure};
use cpu_time::ThreadTime;
use euphoria_core::detect::{KeyDetector, KeyFinderDetector};
use euphoria_core::{Mode, decode_mono};
use rayon::prelude::*;
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 3,
        "usage: native_reference DEVELOPMENT_MANIFEST.tsv OUTPUT.csv"
    );
    let manifest = fs::read_to_string(&args[1])?;
    let rows: Vec<_> = manifest
        .lines()
        .map(|line| line.split_once('\t').context("expected id and path"))
        .collect::<Result<_>>()?;
    let done = AtomicUsize::new(0);
    let results: Vec<_> = rows
        .par_iter()
        .map(|(id, path)| {
            let wall = Instant::now();
            let cpu = ThreadTime::now();
            let prediction = (|| {
                let audio = decode_mono(Path::new(path))?;
                KeyFinderDetector.detect_key(&audio)
            })();
            let code = match prediction {
                Ok(Some(key)) => {
                    i32::from(key.tonic()) * 2 + i32::from(matches!(key.mode(), Mode::Minor))
                }
                Ok(None) => -1,
                Err(error) => {
                    eprintln!("{id}: {error}");
                    -1
                }
            };
            let cpu_seconds = cpu.elapsed().as_secs_f64();
            let wall_seconds = wall.elapsed().as_secs_f64();
            let count = done.fetch_add(1, Ordering::Relaxed) + 1;
            if count % 100 == 0 {
                eprintln!("Native reference: {count}/{}", rows.len());
            }
            format!("{id},{code},{wall_seconds},{cpu_seconds}\n")
        })
        .collect();
    let mut out = BufWriter::new(fs::File::create(&args[2])?);
    writeln!(out, "id,prediction,wall_seconds,cpu_seconds")?;
    for row in results {
        out.write_all(row.as_bytes())?;
    }
    out.flush()?;
    Ok(())
}
