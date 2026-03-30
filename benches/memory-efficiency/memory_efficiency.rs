//! Memory efficiency benchmark — Strata performance under constrained memory budgets.
//!
//! Tests how Strata performs when configured for memory-constrained environments
//! (e.g., Raspberry Pi, edge devices) by varying block_cache_size and
//! write_buffer_size across several memory budgets.
//!
//! Usage:
//!   cargo bench --bench memory_efficiency                          # all profiles
//!   cargo bench --bench memory_efficiency -- --profile 32mb        # single profile
//!   cargo bench --bench memory_efficiency -- --profile unlimited   # baseline
//!   cargo bench --bench memory_efficiency -- -q                    # quick mode
//!   cargo bench --bench memory_efficiency -- --records 1000000     # custom size
//!
//! Available profiles: 32mb, 64mb, 128mb, 256mb, 1gb, unlimited

#[allow(unused)]
#[path = "../harness/mod.rs"]
mod harness;

use harness::recorder::ResultRecorder;
use harness::{kv_key, kv_value, print_hardware_info};
use std::collections::HashMap;
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

struct Profile {
    label: &'static str,
    total_mb: usize,
    cache_mb: usize,
    wbuf_mb: usize,
    max_imm: usize,
    file_mb: u64,
    base_mb: u64,
    threads: usize,
}

const ALL_PROFILES: &[Profile] = &[
    Profile { label: "32mb",      total_mb: 32,   cache_mb: 16,  wbuf_mb: 4,   max_imm: 1, file_mb: 4,  base_mb: 32,  threads: 1 },
    Profile { label: "64mb",      total_mb: 64,   cache_mb: 32,  wbuf_mb: 8,   max_imm: 2, file_mb: 8,  base_mb: 64,  threads: 1 },
    Profile { label: "128mb",     total_mb: 128,  cache_mb: 64,  wbuf_mb: 16,  max_imm: 2, file_mb: 16, base_mb: 128, threads: 2 },
    Profile { label: "256mb",     total_mb: 256,  cache_mb: 128, wbuf_mb: 32,  max_imm: 3, file_mb: 32, base_mb: 256, threads: 2 },
    Profile { label: "1gb",       total_mb: 1024, cache_mb: 512, wbuf_mb: 128, max_imm: 4, file_mb: 64, base_mb: 256, threads: 4 },
    Profile { label: "unlimited", total_mb: 0,    cache_mb: 0,   wbuf_mb: 0,   max_imm: 0, file_mb: 0,  base_mb: 0,   threads: 0 },
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

fn make_config(profile: &Profile) -> StrataConfig {
    if profile.total_mb == 0 {
        return StrataConfig::default();
    }

    StrataConfig {
        storage: StorageConfig {
            block_cache_size: profile.cache_mb * 1024 * 1024,
            write_buffer_size: profile.wbuf_mb * 1024 * 1024,
            max_immutable_memtables: profile.max_imm,
            background_threads: profile.threads,
            target_file_size: profile.file_mb << 20,
            level_base_bytes: profile.base_mb << 20,
            ..StorageConfig::default()
        },
        ..StrataConfig::default()
    }
}

// =============================================================================
// LCG RNG
// =============================================================================

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_bounded(&mut self, n: u64) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) % n
    }
}

// =============================================================================
// Benchmark
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

fn bench_profile(profile: &Profile, records: usize) -> ProfileResult {
    let config = make_config(profile);
    let dir = TempDir::new().unwrap();
    let db = Database::open_with_config(dir.path(), config).unwrap();
    let db = Strata::from_database(db).unwrap();

    // Phase 1: Write
    let write_start = Instant::now();
    for i in 0..records as u64 {
        db.kv_put(&kv_key(i), kv_value()).unwrap();
    }
    let write_elapsed = write_start.elapsed();
    let write_ops_sec = records as f64 / write_elapsed.as_secs_f64();

    // Phase 2: Random reads
    let mut rng = Lcg::new(0xBEEF);
    let read_start = Instant::now();
    for _ in 0..READ_OPS {
        let idx = rng.next_bounded(records as u64);
        let _ = db.kv_get(&kv_key(idx));
    }
    let read_elapsed = read_start.elapsed();
    let read_ops_sec = READ_OPS as f64 / read_elapsed.as_secs_f64();

    // Phase 3: Range scans
    let mut rng = Lcg::new(0xCAFE);
    let scan_start = Instant::now();
    for _ in 0..SCAN_OPS {
        let idx = rng.next_bounded(records as u64);
        let _ = db.kv_scan(Some(&kv_key(idx)), Some(SCAN_LIMIT));
    }
    let scan_elapsed = scan_start.elapsed();
    let scan_ops_sec = SCAN_OPS as f64 / scan_elapsed.as_secs_f64();

    let peak_rss = rss_mb();
    let disk = dir_size_mb(dir.path());

    ProfileResult {
        label: profile.label.to_string(),
        total_budget_mb: profile.total_mb,
        records,
        write_ops_sec,
        read_ops_sec,
        scan_ops_sec,
        peak_rss_mb: peak_rss,
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
    profile: Option<String>,
    quick: bool,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut config = Config {
        records: DEFAULT_RECORDS,
        profile: None,
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
            "--profile" => {
                i += 1;
                if i < args.len() {
                    config.profile = Some(args[i].clone());
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

    let profiles: Vec<&Profile> = match &config.profile {
        Some(name) => {
            match ALL_PROFILES.iter().find(|p| p.label == name.as_str()) {
                Some(p) => vec![p],
                None => {
                    eprintln!(
                        "Unknown profile '{}'. Available: {}",
                        name,
                        ALL_PROFILES.iter().map(|p| p.label).collect::<Vec<_>>().join(", "),
                    );
                    std::process::exit(1);
                }
            }
        }
        None if config.quick => {
            ALL_PROFILES.iter().filter(|p| matches!(p.label, "32mb" | "128mb" | "unlimited")).collect()
        }
        None => ALL_PROFILES.iter().collect(),
    };

    eprintln!(
        "\n=== Memory Efficiency Benchmark ({} records) ===\n",
        config.records,
    );

    print_header();

    let mut recorder = ResultRecorder::new("memory-efficiency");

    for profile in &profiles {
        let r = bench_profile(profile, config.records);
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
