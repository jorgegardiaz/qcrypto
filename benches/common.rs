//! Shared utilities for all `qcrypto` benchmarks.
//!
//! This module centralises three concerns that must be consistent across every
//! benchmark binary:
//!
//! 1. **Environment capture** (`write_environment`): CPU, core count, OS,
//!    crate version, and toolchain.  Without this, timing results lack context.
//! 2. **Deterministic seeding** (`SEED` + `seed_thread`): all randomness in
//!    `qcrypto` is thread-local, so the RNG must be seeded on every thread that
//!    executes protocol code.  For single-threaded criterion runs (the default),
//!    seeding once per measurement batch is sufficient.
//! 3. **Raw CSV output** (`RawCsv`): criterion produces its own statistical
//!    reports (`estimates.json`, HTML plots), but raw per-run samples in CSV
//!    format are convenient for downstream analysis with external tools.
//!
//! This file is **not** a benchmark binary (it has no `criterion_main!`).
//! Other bench files include it with `#[path = "common.rs"] mod common;`.

#![allow(dead_code)]

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Global seed used across all benchmarks for reproducibility.
pub const SEED: u64 = 0x5150_2025_0517;

/// Directory where raw CSV files are written (qcrypto-side data only).
pub const DATA_DIR: &str = "benches/data/qcrypto";

/// Seeds the thread-local RNG of `qcrypto` on the calling thread.
///
/// Because `qcrypto`'s randomness is thread-local, this must be called on
/// every thread that invokes protocol code.  In single-threaded criterion
/// runs the timed body executes on the main thread, so seeding once per
/// measurement batch is sufficient.
pub fn seed_thread() {
    qcrypto::set_global_seed(SEED);
}

/// Ensures the data directory exists and returns its path.
pub fn ensure_data_dir() -> PathBuf {
    let dir = PathBuf::from(DATA_DIR);
    fs::create_dir_all(&dir).expect("could not create benches/data");
    dir
}

/// Writes an `environment.txt` file capturing system information.
///
/// Call this once per benchmark run (e.g. from the first bench function).
/// Records the minimum information needed to contextualise timing results.
pub fn write_environment() {
    let dir = ensure_data_dir();
    let path = dir.join("environment.txt");
    let mut f = File::create(&path).expect("could not create environment.txt");

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    writeln!(f, "# qcrypto benchmark environment").unwrap();
    writeln!(f, "unix_timestamp = {ts}").unwrap();
    writeln!(f, "qcrypto_version = {}", env!("CARGO_PKG_VERSION")).unwrap();
    writeln!(f, "seed = {:#x}", SEED).unwrap();
    writeln!(f, "rustc_target = {}", current_target()).unwrap();
    writeln!(f, "os = {}", std::env::consts::OS).unwrap();
    writeln!(f, "arch = {}", std::env::consts::ARCH).unwrap();
    writeln!(
        f,
        "logical_cpus = {}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0)
    )
    .unwrap();

    // Linux-only: CPU model and total memory (best-effort; silently skipped on other OSes).
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        if let Some(model) = cpuinfo
            .lines()
            .find(|l| l.starts_with("model name"))
            .and_then(|l| l.split(':').nth(1))
        {
            writeln!(f, "cpu_model = {}", model.trim()).unwrap();
        }
    }
    if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
        if let Some(total) = meminfo
            .lines()
            .find(|l| l.starts_with("MemTotal"))
            .and_then(|l| l.split(':').nth(1))
        {
            writeln!(f, "mem_total = {}", total.trim()).unwrap();
        }
    }

    eprintln!("[bench] environment written to {}", path.display());
}

fn current_target() -> String {
    // No stable API exposes the full target triple at runtime; reconstruct an
    // approximation from the available compile-time constants.
    format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
}

/// Buffered CSV writer with a fixed header.
///
/// # Example
///
/// ```ignore
/// let mut csv = RawCsv::create("core_gates.csv", "qubits,state_type,gate,sample_ns");
/// csv.row(format_args!("{n},StateVector,X,{ns}"));
/// ```
pub struct RawCsv {
    writer: BufWriter<File>,
}

impl RawCsv {
    /// Creates (or truncates) a CSV file with the given header inside `DATA_DIR`.
    pub fn create(filename: &str, header: &str) -> Self {
        let dir = ensure_data_dir();
        let path = dir.join(filename);
        let file = File::create(&path)
            .unwrap_or_else(|e| panic!("could not create {}: {e}", path.display()));
        let mut writer = BufWriter::new(file);
        writeln!(writer, "{header}").unwrap();
        Self { writer }
    }

    /// Opens an existing CSV in append mode (does not rewrite the header).
    pub fn append(filename: &str) -> Self {
        let dir = ensure_data_dir();
        let path = dir.join(filename);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap_or_else(|e| panic!("could not open {}: {e}", path.display()));
        Self {
            writer: BufWriter::new(file),
        }
    }

    /// Writes a pre-formatted row (newline appended automatically).
    pub fn row(&mut self, args: std::fmt::Arguments) {
        writeln!(self.writer, "{args}").unwrap();
    }

    pub fn flush(&mut self) {
        self.writer.flush().unwrap();
    }
}

impl Drop for RawCsv {
    fn drop(&mut self) {
        let _ = self.writer.flush();
    }
}

/// Returns `true` if `environment.txt` already exists in the data directory.
pub fn environment_written() -> bool {
    Path::new(DATA_DIR).join("environment.txt").exists()
}
