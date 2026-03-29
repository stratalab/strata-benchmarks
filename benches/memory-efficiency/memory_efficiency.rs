//! Memory efficiency benchmark — Strata performance under constrained memory budgets.
//!
//! Tests how Strata performs when configured for memory-constrained environments
//! (e.g., Raspberry Pi, edge devices) by varying block_cache_size and
//! write_buffer_size across several memory budgets.
//!
//! Memory budgets tested: 32MB, 64MB, 128MB, 256MB, 1GB (+ unlimited baseline)
//!
//! For each budget, measures:
//! - Write throughput (sequential inserts)
//! - Read throughput (random point reads)
//! - Scan throughput (range scans)
//! - Peak RSS
//! - Disk usage
//!
//! Usage:
//!   cargo bench --bench memory_efficiency
//!   cargo bench --bench memory_efficiency -- --quick -q
//!   cargo bench --bench memory_efficiency -- --records 1000000

#[allow(unused)]
#[path = "../harness/mod.rs"]
mod harness;

use harness::recorder::ResultRecorder;
use harness::{kv_key, kv_value, print_hardware_info};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use stratadb::{Database, Strata, StorageConfig, StrataConfig};
use tempfile::TempDir;

// =============================================================================
// Configuration
// =============================================================================

const DEFAULT_RECORDS: usize = 500_000;
const QUICK_RECORDS: usize = 50_000;
const READ_OPS: usize = 50_000;
const SCAN_OPS: usize = 5_000;
const SCAN_LIMIT: u64 = 10;

/// Memory budget profiles: (label, total_mb, block_cache_mb, write_buffer_mb, max_immutable, target_file_mb, level_base_mb, bg_threads)
const PROFILES: &[(&str, usize, usize, usize, usize, u64, u64, usize)] = &[
    // label       total  cache  wbuf  imm  file  L1base  threads
    ("32mb",        32,    16,    4,    1,    4,    32,     1),
    ("64mb",        64,    32,    8,    2,    8,    64,     1),
    ("128mb",      128,    64,   16,    2,   16,   128,     2),
    ("256mb",      256,   128,   32,    3,   32,   256,     2),
    ("1gb",       1024,   512,  128,    4,   64,   256,     4),
    ("unlimited",    0,     0,    0,    0,    0,     0,     0), // all defaults
];

const QUICK_PROFILES: &[(&str, usize, usize, usize, usize, u64, u64, usize)] = &[
    ("32mb",        32,    16,    4,    1,    4,    32,     1),
    ("128mb",      128,    64,   16,    2,   16,   128,     2),
    ("unlimited",    0,     0,    0,    0,    0,     0,     0),
];

// =============================================================================
// Helpers
// =============================================================================

fn rss_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<f64>() {
                            return kb / 1024.0;
                        }
                    }
                }
            }
        }
    }
    0.0
}

fn dir_size_mb(path: &std::path::Path) -> f64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let ft = match entry.file_type() {
                Ok(ft) => ft,
                Err(_) => continue,
            };
            if ft.is_file() {
                total += entry.metadata().map(|m| m.len()).unwrap_or(0);
            } else if ft.is_dir() {
                total += (dir_size_mb(&entry.path()) * 1024.0 * 1024.0) as u64;
            }
        }
    }
    total as f64 / (1024.0 * 1024.0)
}

fn make_config(profile: &(&str, usize, usize, usize, usize, u64, u64, usize)) -> StrataConfig {
    let (_label, _total, cache_mb, wbuf_mb, max_imm, file_mb, base_mb, threads) = *profile;

    if _total == 0 {
        // Unlimited — use all defaults
        return StrataConfig::default();
    }

    StrataConfig {
        storage: StorageConfig {
            block_cache_size: cache_mb * 1024 * 1024,
            write_buffer_size: wbuf_mb * 1024 * 1024,
            max_immutable_memtables: max_imm,
            background_threads: threads,
            target_file_size: file_mb << 20,
            level_base_bytes: base_mb << 20,
            ..StorageConfig::default()
        },
        ..StrataConfig::default()
    }
}

fn open_db(config: StrataConfig, dir: &TempDir) -> Strata {
    let db = Database::open_with_config(dir.path(), config).unwrap();
    Strata::from_database(db).unwrap()
}

// =============================================================================
// LCG RNG for deterministic random reads
// =============================================================================

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_bounded(&mut self, n: u64) -> u64 {
        self.0 = self.0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) % n
    }
}

// =============================================================================
// Benchmark phases
// =============================================================================

struct ProfileResult {
    label: String,
    total_budget_mb: usize,
    records: usize,
    write_ops_sec: f64,
    read_ops_sec: f64,
    scan_ops_sec: f64,
    peak_rss_mb: f64,
    disk_mb: f64,
}

fn bench_profile(
    profile: &(&str, usize, usize, usize, usize, u64, u64, usize),
    records: usize,
    read_ops: usize,
    scan_ops: usize,
) -> ProfileResult {
    let (label, total_mb, ..) = *profile;
    let config = make_config(profile);
    let dir = TempDir::new().unwrap();
    let db = open_db(config, &dir);

    let rss_before = rss_mb();

    // Phase 1: Write
    let write_start = Instant::now();
    for i in 0..records as u64 {
        db.kv_put(&kv_key(i), kv_value()).unwrap();
    }
    let write_elapsed = write_start.elapsed();
    let write_ops_sec = records as f64 / write_elapsed.as_secs_f64();

    let rss_after_write = rss_mb();

    // Phase 2: Random reads
    let mut rng = Lcg::new(0xBEEF);
    let read_start = Instant::now();
    for _ in 0..read_ops {
        let idx = rng.next_bounded(records as u64);
        let _ = db.kv_get(&kv_key(idx));
    }
    let read_elapsed = read_start.elapsed();
    let read_ops_sec = read_ops as f64 / read_elapsed.as_secs_f64();

    // Phase 3: Range scans
    let mut rng = Lcg::new(0xCAFE);
    let scan_start = Instant::now();
    for _ in 0..scan_ops {
        let idx = rng.next_bounded(records as u64);
        let _ = db.kv_scan(Some(&kv_key(idx)), Some(SCAN_LIMIT));
    }
    let scan_elapsed = scan_start.elapsed();
    let scan_ops_sec = scan_ops as f64 / scan_elapsed.as_secs_f64();

    let peak_rss = rss_mb();
    let disk = dir_size_mb(dir.path());

    ProfileResult {
        label: label.to_string(),
        total_budget_mb: total_mb,
        records,
        write_ops_sec,
        read_ops_sec,
        scan_ops_sec,
        peak_rss_mb: peak_rss - rss_before + rss_after_write - rss_before,
        disk_mb: disk,
    }
}

// =============================================================================
// Output
// =============================================================================

fn print_header() {
    eprintln!(
        "  {:<12} {:>10} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "profile", "records", "write ops/s", "read ops/s", "scan ops/s", "RSS (MB)", "disk (MB)",
    );
    eprintln!(
        "  {:<12} {:>10} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "-------", "-------", "-----------", "----------", "----------", "--------", "---------",
    );
}

fn print_row(r: &ProfileResult) {
    eprintln!(
        "  {:<12} {:>10} {:>12.0} {:>12.0} {:>12.0} {:>10.1} {:>10.1}",
        r.label, r.records, r.write_ops_sec, r.read_ops_sec, r.scan_ops_sec, r.peak_rss_mb, r.disk_mb,
    );
}

// =============================================================================
// CLI
// =============================================================================

struct Config {
    records: usize,
    quick: bool,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut config = Config {
        records: DEFAULT_RECORDS,
        quick: false,
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-q" | "--quick" => {
                config.quick = true;
                config.records = QUICK_RECORDS;
            }
            "--records" => {
                i += 1;
                if i < args.len() {
                    config.records = args[i].parse().unwrap_or(DEFAULT_RECORDS);
                }
            }
            _ => {}
        }
        i += 1;
    }
    config
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    print_hardware_info();
    let config = parse_args();

    let profiles = if config.quick { QUICK_PROFILES } else { PROFILES };

    eprintln!(
        "\n=== Memory Efficiency Benchmark ({} records, {} mode) ===\n",
        config.records,
        if config.quick { "quick" } else { "full" },
    );

    print_header();

    let mut recorder = ResultRecorder::new("memory-efficiency");

    for profile in profiles {
        let r = bench_profile(profile, config.records, READ_OPS, SCAN_OPS);
        print_row(&r);

        let mut params = HashMap::new();
        params.insert("profile".into(), serde_json::json!(r.label));
        params.insert("total_budget_mb".into(), serde_json::json!(r.total_budget_mb));
        params.insert("records".into(), serde_json::json!(r.records));
        params.insert("write_ops_sec".into(), serde_json::json!(r.write_ops_sec));
        params.insert("read_ops_sec".into(), serde_json::json!(r.read_ops_sec));
        params.insert("scan_ops_sec".into(), serde_json::json!(r.scan_ops_sec));
        params.insert("peak_rss_mb".into(), serde_json::json!(r.peak_rss_mb));
        params.insert("disk_mb".into(), serde_json::json!(r.disk_mb));

        let bench_name = format!("memory/{}/{}", r.label, r.records);
        let p = harness::Percentiles {
            p50: std::time::Duration::from_nanos(0),
            p95: std::time::Duration::from_nanos(0),
            p99: std::time::Duration::from_nanos(0),
            min: std::time::Duration::from_nanos(0),
            max: std::time::Duration::from_nanos(0),
            samples: r.records,
        };
        recorder.record_latency(&bench_name, params, &p, None, r.records as u64);
    }

    eprintln!();
    if let Ok(path) = recorder.save() {
        eprintln!("Results saved to: {}", path.display());
    }
}
