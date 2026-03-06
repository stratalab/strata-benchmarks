//! Scaling Benchmark for StrataDB
//!
//! Measures how performance degrades as dataset size grows from "fits in cache"
//! to "exceeds RAM." Covers KV, JSON, Vector, and Graph tiers at scale levels
//! from 1K to 1B records, measuring write throughput, read latency,
//! scan/query performance, RSS, disk usage, page faults, I/O bytes, CPU time,
//! WAL counters, space amplification, and load throughput curve at each level.
//!
//! Run:    `cargo bench --bench scaling`
//! Quick:  `cargo bench --bench scaling -- --quick`
//! Quiet:  `cargo bench --bench scaling -- --quick -q`
//! CSV:    `cargo bench --bench scaling -- --quick --csv`
//! Tiers:  `cargo bench --bench scaling -- --tiers kv,json`
//! Scales: `cargo bench --bench scaling -- --scales 1000,10000`

#[allow(unused)]
#[path = "../harness/mod.rs"]
mod harness;

use harness::recorder::ResultRecorder;
use harness::{create_db, print_hardware_info, BenchDb, DurabilityConfig};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};
use strata_benchmarks::schema::{BenchmarkMetrics, BenchmarkResult};
use stratadb::{DistanceMetric, Value};

// =============================================================================
// Configuration
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tier {
    Kv,
    Json,
    Vector,
    Graph,
}

impl Tier {
    fn label(&self) -> &'static str {
        match self {
            Self::Kv => "kv",
            Self::Json => "json",
            Self::Vector => "vector",
            Self::Graph => "graph",
        }
    }
}

struct Config {
    tiers: Vec<Tier>,
    scales: Vec<usize>,
    sample_ops: usize,
    durability: DurabilityConfig,
    kv_value_size: usize,
    vector_dims: usize,
    batch_size: usize,
    csv: bool,
    quiet: bool,
    quick: bool,
}

const QUICK_SCALES: &[usize] = &[1_000, 10_000, 100_000];
const FULL_SCALES: &[usize] = &[
    1_000, 10_000, 100_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000,
    1_000_000_000,
];
const VECTOR_MAX_SCALE: usize = 1_000_000_000;
const ALL_TIERS: &[Tier] = &[Tier::Kv, Tier::Json, Tier::Vector, Tier::Graph];
const GRAPH_EDGES_PER_VERTEX: usize = 5;

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut config = Config {
        tiers: ALL_TIERS.to_vec(),
        scales: Vec::new(),
        sample_ops: 10_000,
        durability: DurabilityConfig::Standard,
        kv_value_size: 1024,
        vector_dims: 128,
        batch_size: 50_000,
        csv: false,
        quiet: false,
        quick: false,
    };

    let mut i = 1;
    let mut scales_override: Option<Vec<usize>> = None;

    while i < args.len() {
        match args[i].as_str() {
            "--quick" => config.quick = true,
            "--tiers" => {
                i += 1;
                if i < args.len() {
                    config.tiers = args[i]
                        .split(',')
                        .filter_map(|s| match s.trim().to_lowercase().as_str() {
                            "kv" => Some(Tier::Kv),
                            "json" => Some(Tier::Json),
                            "vector" => Some(Tier::Vector),
                            "graph" => Some(Tier::Graph),
                            _ => None,
                        })
                        .collect();
                }
            }
            "--scales" => {
                i += 1;
                if i < args.len() {
                    scales_override = Some(
                        args[i]
                            .split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect(),
                    );
                }
            }
            "--sample-ops" => {
                i += 1;
                if i < args.len() {
                    config.sample_ops = args[i].parse().unwrap_or(10_000);
                }
            }
            "--durability" => {
                i += 1;
                if i < args.len() {
                    config.durability = match args[i].as_str() {
                        "cache" => DurabilityConfig::Cache,
                        "standard" => DurabilityConfig::Standard,
                        "always" => DurabilityConfig::Always,
                        _ => DurabilityConfig::Standard,
                    };
                }
            }
            "--kv-value-size" => {
                i += 1;
                if i < args.len() {
                    config.kv_value_size = args[i].parse().unwrap_or(1024);
                }
            }
            "--vector-dims" => {
                i += 1;
                if i < args.len() {
                    config.vector_dims = args[i].parse().unwrap_or(128);
                }
            }
            "--batch-size" => {
                i += 1;
                if i < args.len() {
                    config.batch_size = args[i].parse().unwrap_or(50_000);
                }
            }
            "--csv" => config.csv = true,
            "-q" => config.quiet = true,
            _ => {}
        }
        i += 1;
    }

    if config.quick {
        config.sample_ops = config.sample_ops.min(1_000);
    }

    config.scales = scales_override.unwrap_or_else(|| {
        if config.quick {
            QUICK_SCALES.to_vec()
        } else {
            FULL_SCALES.to_vec()
        }
    });

    config
}

/// Effective scales for a tier (vector caps at VECTOR_MAX_SCALE).
fn effective_scales(tier: Tier, config: &Config) -> Vec<usize> {
    match tier {
        Tier::Vector => config
            .scales
            .iter()
            .copied()
            .filter(|&s| s <= VECTOR_MAX_SCALE)
            .collect(),
        _ => config.scales.clone(),
    }
}

// =============================================================================
// Data Generators
// =============================================================================

fn scaling_kv_key(i: u64) -> String {
    format!("scale:{:012}", i)
}

/// LCG-seeded pseudo-random bytes to prevent compression artifacts.
fn scaling_kv_value(size: usize, i: u64) -> Value {
    let mut state: u64 = i.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut bytes = Vec::with_capacity(size);
    for _ in 0..size {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        bytes.push((state >> 33) as u8);
    }
    Value::Bytes(bytes)
}

/// Generate a deterministic vector of configurable dimensions.
fn scaling_vector(dims: usize, i: u64) -> Vec<f32> {
    let seed = i as f32;
    (0..dims)
        .map(|d| (seed * 0.1 + d as f32 * 0.7).sin() * 0.5 + 0.5)
        .collect()
}

/// Generate a synthetic graph: ring backbone + random edges (LCG).
fn generate_synthetic_graph(
    num_vertices: usize,
    edges_per_vertex: usize,
) -> (Vec<u64>, Vec<(u64, u64)>) {
    let vertices: Vec<u64> = (0..num_vertices as u64).collect();
    let total_random_edges = num_vertices * (edges_per_vertex.saturating_sub(1));
    let mut edges = Vec::with_capacity(num_vertices + total_random_edges);

    // Ring backbone (ensures connectivity)
    for i in 0..num_vertices {
        let src = i as u64;
        let dst = ((i + 1) % num_vertices) as u64;
        edges.push((src, dst));
    }

    // Random edges via LCG
    let mut rng_state: u64 = 0xdeadbeef_u64;
    for _ in 0..total_random_edges {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let src = (rng_state >> 33) % num_vertices as u64;
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let dst = (rng_state >> 33) % num_vertices as u64;
        edges.push((src, dst));
    }

    (vertices, edges)
}

// =============================================================================
// Resource Tracking
// =============================================================================

/// Snapshot of all process-level resource counters at a point in time.
#[derive(Debug, Clone, Default)]
struct ResourceSnapshot {
    rss_mb: f64,
    minor_faults: u64,
    major_faults: u64,
    io_read_bytes: u64,
    io_write_bytes: u64,
    cpu_user_ms: u64,
    cpu_sys_ms: u64,
    wal_appends: u64,
    wal_syncs: u64,
}

/// Delta between two resource snapshots.
#[derive(Debug, Clone, Default)]
struct ResourceDelta {
    minor_faults: u64,
    major_faults: u64,
    io_read_mb: f64,
    io_write_mb: f64,
    cpu_user_ms: u64,
    cpu_sys_ms: u64,
    wal_appends: u64,
    wal_syncs: u64,
}

fn snapshot_resources(db: &BenchDb) -> ResourceSnapshot {
    let (minor, major) = read_page_faults();
    let (io_r, io_w) = read_io_bytes();
    let (cpu_u, cpu_s) = read_cpu_time();
    let wal = db.db.durability_counters().unwrap_or_default();
    ResourceSnapshot {
        rss_mb: rss_mb(),
        minor_faults: minor,
        major_faults: major,
        io_read_bytes: io_r,
        io_write_bytes: io_w,
        cpu_user_ms: cpu_u,
        cpu_sys_ms: cpu_s,
        wal_appends: wal.wal_appends,
        wal_syncs: wal.sync_calls,
    }
}

fn resource_delta(before: &ResourceSnapshot, after: &ResourceSnapshot) -> ResourceDelta {
    ResourceDelta {
        minor_faults: after.minor_faults.saturating_sub(before.minor_faults),
        major_faults: after.major_faults.saturating_sub(before.major_faults),
        io_read_mb: (after.io_read_bytes.saturating_sub(before.io_read_bytes)) as f64
            / (1024.0 * 1024.0),
        io_write_mb: (after.io_write_bytes.saturating_sub(before.io_write_bytes)) as f64
            / (1024.0 * 1024.0),
        cpu_user_ms: after.cpu_user_ms.saturating_sub(before.cpu_user_ms),
        cpu_sys_ms: after.cpu_sys_ms.saturating_sub(before.cpu_sys_ms),
        wal_appends: after.wal_appends.saturating_sub(before.wal_appends),
        wal_syncs: after.wal_syncs.saturating_sub(before.wal_syncs),
    }
}

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

/// Read minor and major page faults from /proc/self/stat.
/// Fields 10 (minflt) and 12 (majflt) in 1-indexed /proc/self/stat.
fn read_page_faults() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/self/stat") {
            // Skip past comm field (in parentheses, may contain spaces)
            if let Some(close_paren) = contents.rfind(')') {
                let rest = &contents[close_paren + 2..];
                let fields: Vec<&str> = rest.split_whitespace().collect();
                // After comm: field[0]=state, field[7]=minflt, field[9]=majflt
                if fields.len() > 9 {
                    let minor = fields[7].parse().unwrap_or(0);
                    let major = fields[9].parse().unwrap_or(0);
                    return (minor, major);
                }
            }
        }
    }
    (0, 0)
}

/// Read I/O bytes from /proc/self/io (read_bytes, write_bytes).
fn read_io_bytes() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/self/io") {
            let mut read_bytes = 0u64;
            let mut write_bytes = 0u64;
            for line in contents.lines() {
                if let Some(val) = line.strip_prefix("read_bytes: ") {
                    read_bytes = val.trim().parse().unwrap_or(0);
                } else if let Some(val) = line.strip_prefix("write_bytes: ") {
                    write_bytes = val.trim().parse().unwrap_or(0);
                }
            }
            return (read_bytes, write_bytes);
        }
    }
    (0, 0)
}

/// Read user and system CPU time in milliseconds from /proc/self/stat.
fn read_cpu_time() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/self/stat") {
            if let Some(close_paren) = contents.rfind(')') {
                let rest = &contents[close_paren + 2..];
                let fields: Vec<&str> = rest.split_whitespace().collect();
                // field[11]=utime, field[12]=stime (in clock ticks)
                if fields.len() > 12 {
                    let ticks_per_sec = {
                        const SC_CLK_TCK: i32 = 2;
                        let ticks = unsafe {
                            extern "C" {
                                fn sysconf(name: i32) -> i64;
                            }
                            sysconf(SC_CLK_TCK)
                        };
                        if ticks > 0 { ticks as u64 } else { 100 }
                    };
                    let utime: u64 = fields[11].parse().unwrap_or(0);
                    let stime: u64 = fields[12].parse().unwrap_or(0);
                    return (utime * 1000 / ticks_per_sec, stime * 1000 / ticks_per_sec);
                }
            }
        }
    }
    (0, 0)
}

fn dir_size_bytes(path: &Path) -> u64 {
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
                total += dir_size_bytes(&entry.path());
            }
        }
    }
    total
}

fn disk_mb(db: &BenchDb) -> f64 {
    match db.db_path() {
        Some(p) => dir_size_bytes(p) as f64 / (1024.0 * 1024.0),
        None => 0.0,
    }
}

/// Estimate raw data size in MB for a given tier and scale.
fn raw_data_mb(tier: Tier, scale: usize, config: &Config) -> f64 {
    let bytes = match tier {
        Tier::Kv => {
            // key (~19 bytes "scale:000000000000") + value
            scale * (19 + config.kv_value_size)
        }
        Tier::Json => {
            // ~500 bytes per json_document
            scale * 500
        }
        Tier::Vector => {
            // dims * 4 bytes (f32) + ~10 byte key
            scale * (config.vector_dims * 4 + 10)
        }
        Tier::Graph => {
            // vertices: 8 bytes each, edges: 16 bytes each (~edges_per_vertex * scale edges)
            scale * 8 + scale * GRAPH_EDGES_PER_VERTEX * 16
        }
    };
    bytes as f64 / (1024.0 * 1024.0)
}

// =============================================================================
// Measurement Result
// =============================================================================

struct ScaleResult {
    tier: Tier,
    operation: String,
    scale: usize,
    ops_per_sec: f64,
    total_time_secs: f64,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    min: Duration,
    max: Duration,
    avg: Duration,
    samples: usize,
    // Resource metrics
    rss_mb: f64,
    disk_mb: f64,
    resources: ResourceDelta,
    space_amplification: f64,
    // Load throughput curve (load ops only)
    first_batch_ops_sec: f64,
    last_batch_ops_sec: f64,
}

impl ScaleResult {
    fn is_load(&self) -> bool {
        self.operation == "load"
    }
}

// =============================================================================
// LCG for random key access
// =============================================================================

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0xdeadbeef),
        }
    }

    #[inline]
    fn next_bounded(&mut self, bound: u64) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 33) % bound
    }
}

// =============================================================================
// Load result builder (shared by all tier load functions)
// =============================================================================

struct LoadTracker {
    first_batch_ops_sec: f64,
    last_batch_ops_sec: f64,
}

impl LoadTracker {
    fn new() -> Self {
        Self {
            first_batch_ops_sec: 0.0,
            last_batch_ops_sec: 0.0,
        }
    }

    fn record_batch(&mut self, batch_idx: usize, batch_count: usize, batch_elapsed: Duration) {
        let ops_sec = batch_count as f64 / batch_elapsed.as_secs_f64();
        if batch_idx == 0 {
            self.first_batch_ops_sec = ops_sec;
        }
        // Always update last — final call is the last batch
        self.last_batch_ops_sec = ops_sec;
    }
}

fn build_load_result(
    tier: Tier,
    scale: usize,
    total_items: usize,
    elapsed: Duration,
    snap_before: &ResourceSnapshot,
    snap_after: &ResourceSnapshot,
    disk: f64,
    tracker: &LoadTracker,
    config: &Config,
) -> ScaleResult {
    let delta = resource_delta(snap_before, snap_after);
    let raw = raw_data_mb(tier, scale, config);
    let space_amp = if raw > 0.0 { disk / raw } else { 0.0 };

    ScaleResult {
        tier,
        operation: "load".to_string(),
        scale,
        ops_per_sec: total_items as f64 / elapsed.as_secs_f64(),
        total_time_secs: elapsed.as_secs_f64(),
        p50: Duration::ZERO,
        p95: Duration::ZERO,
        p99: Duration::ZERO,
        min: Duration::ZERO,
        max: Duration::ZERO,
        avg: Duration::ZERO,
        samples: total_items,
        rss_mb: snap_after.rss_mb,
        disk_mb: disk,
        resources: delta,
        space_amplification: space_amp,
        first_batch_ops_sec: tracker.first_batch_ops_sec,
        last_batch_ops_sec: tracker.last_batch_ops_sec,
    }
}

// =============================================================================
// Tier Operations
// =============================================================================

fn run_kv_load(db: &BenchDb, scale: usize, config: &Config) -> ScaleResult {
    let snap_before = snapshot_resources(db);
    let mut tracker = LoadTracker::new();
    let start = Instant::now();

    let mut batch_idx = 0usize;
    for batch_start in (0..scale).step_by(config.batch_size) {
        let batch_end = (batch_start + config.batch_size).min(scale);
        let batch_t = Instant::now();
        for i in batch_start..batch_end {
            let key = scaling_kv_key(i as u64);
            let val = scaling_kv_value(config.kv_value_size, i as u64);
            db.db.kv_put(&key, val).unwrap();
        }
        tracker.record_batch(batch_idx, batch_end - batch_start, batch_t.elapsed());
        batch_idx += 1;
        if !config.quiet && !config.csv && scale >= config.batch_size {
            eprintln!("    loaded {}/{}", batch_end, scale);
        }
    }

    let elapsed = start.elapsed();
    let snap_after = snapshot_resources(db);
    let disk = disk_mb(db);

    build_load_result(Tier::Kv, scale, scale, elapsed, &snap_before, &snap_after, disk, &tracker, config)
}

fn run_kv_random_read(db: &BenchDb, scale: usize, config: &Config) -> ScaleResult {
    let n = config.sample_ops;
    let mut rng = Lcg::new(42);

    let snap_before = snapshot_resources(db);
    let mut latencies = Vec::with_capacity(n);
    let wall = Instant::now();

    for _ in 0..n {
        let idx = rng.next_bounded(scale as u64);
        let key = scaling_kv_key(idx);
        let t = Instant::now();
        let _ = db.db.kv_get(&key);
        latencies.push(t.elapsed());
    }

    let wall_elapsed = wall.elapsed();
    let snap_after = snapshot_resources(db);
    build_latency_result(Tier::Kv, "random_read", scale, latencies, wall_elapsed, db, &snap_before, &snap_after, config)
}

fn run_kv_random_write(db: &BenchDb, scale: usize, config: &Config) -> ScaleResult {
    let n = config.sample_ops;
    let mut rng = Lcg::new(99);

    let snap_before = snapshot_resources(db);
    let mut latencies = Vec::with_capacity(n);
    let wall = Instant::now();

    for _ in 0..n {
        let idx = rng.next_bounded(scale as u64);
        let key = scaling_kv_key(idx);
        let val = scaling_kv_value(config.kv_value_size, idx.wrapping_add(1_000_000_000));
        let t = Instant::now();
        db.db.kv_put(&key, val).unwrap();
        latencies.push(t.elapsed());
    }

    let wall_elapsed = wall.elapsed();
    let snap_after = snapshot_resources(db);
    build_latency_result(Tier::Kv, "random_write", scale, latencies, wall_elapsed, db, &snap_before, &snap_after, config)
}

fn run_kv_scan(db: &BenchDb, scale: usize, config: &Config) -> ScaleResult {
    let n = config.sample_ops;

    let snap_before = snapshot_resources(db);
    let mut latencies = Vec::with_capacity(n);
    let wall = Instant::now();

    for _ in 0..n {
        let t = Instant::now();
        let _ = db.db.kv_list(Some("scale:"));
        latencies.push(t.elapsed());
    }

    let wall_elapsed = wall.elapsed();
    let snap_after = snapshot_resources(db);
    build_latency_result(Tier::Kv, "scan", scale, latencies, wall_elapsed, db, &snap_before, &snap_after, config)
}

fn run_json_load(db: &BenchDb, scale: usize, config: &Config) -> ScaleResult {
    let snap_before = snapshot_resources(db);
    let mut tracker = LoadTracker::new();
    let start = Instant::now();

    let mut batch_idx = 0usize;
    for batch_start in (0..scale).step_by(config.batch_size) {
        let batch_end = (batch_start + config.batch_size).min(scale);
        let batch_t = Instant::now();
        for i in batch_start..batch_end {
            let key = format!("jdoc:{:012}", i);
            let doc = harness::json_document(i as u64);
            db.db.json_set(&key, "$", doc).unwrap();
        }
        tracker.record_batch(batch_idx, batch_end - batch_start, batch_t.elapsed());
        batch_idx += 1;
        if !config.quiet && !config.csv && scale >= config.batch_size {
            eprintln!("    loaded {}/{}", batch_end, scale);
        }
    }

    let elapsed = start.elapsed();
    let snap_after = snapshot_resources(db);
    let disk = disk_mb(db);

    build_load_result(Tier::Json, scale, scale, elapsed, &snap_before, &snap_after, disk, &tracker, config)
}

fn run_json_random_read(db: &BenchDb, scale: usize, config: &Config) -> ScaleResult {
    let n = config.sample_ops;
    let mut rng = Lcg::new(42);

    let snap_before = snapshot_resources(db);
    let mut latencies = Vec::with_capacity(n);
    let wall = Instant::now();

    for _ in 0..n {
        let idx = rng.next_bounded(scale as u64);
        let key = format!("jdoc:{:012}", idx);
        let t = Instant::now();
        let _ = db.db.json_get(&key, "$");
        latencies.push(t.elapsed());
    }

    let wall_elapsed = wall.elapsed();
    let snap_after = snapshot_resources(db);
    build_latency_result(Tier::Json, "random_read", scale, latencies, wall_elapsed, db, &snap_before, &snap_after, config)
}

fn run_json_path_read(db: &BenchDb, scale: usize, config: &Config) -> ScaleResult {
    let n = config.sample_ops;
    let mut rng = Lcg::new(77);

    let snap_before = snapshot_resources(db);
    let mut latencies = Vec::with_capacity(n);
    let wall = Instant::now();

    for _ in 0..n {
        let idx = rng.next_bounded(scale as u64);
        let key = format!("jdoc:{:012}", idx);
        let t = Instant::now();
        let _ = db.db.json_get(&key, "$.metadata.mid_score");
        latencies.push(t.elapsed());
    }

    let wall_elapsed = wall.elapsed();
    let snap_after = snapshot_resources(db);
    build_latency_result(Tier::Json, "path_read", scale, latencies, wall_elapsed, db, &snap_before, &snap_after, config)
}

fn run_json_path_update(db: &BenchDb, scale: usize, config: &Config) -> ScaleResult {
    let n = config.sample_ops;
    let mut rng = Lcg::new(88);

    let snap_before = snapshot_resources(db);
    let mut latencies = Vec::with_capacity(n);
    let wall = Instant::now();

    for _ in 0..n {
        let idx = rng.next_bounded(scale as u64);
        let key = format!("jdoc:{:012}", idx);
        let new_val = Value::Float(idx as f64 * 2.5);
        let t = Instant::now();
        let _ = db.db.json_set(&key, "$.metadata.mid_score", new_val);
        latencies.push(t.elapsed());
    }

    let wall_elapsed = wall.elapsed();
    let snap_after = snapshot_resources(db);
    build_latency_result(Tier::Json, "path_update", scale, latencies, wall_elapsed, db, &snap_before, &snap_after, config)
}

fn run_vector_load(db: &BenchDb, scale: usize, config: &Config) -> ScaleResult {
    db.db
        .vector_create_collection("scale_vec", config.vector_dims as u64, DistanceMetric::Cosine)
        .unwrap();

    let snap_before = snapshot_resources(db);
    let mut tracker = LoadTracker::new();
    let start = Instant::now();

    let mut batch_idx = 0usize;
    for batch_start in (0..scale).step_by(config.batch_size) {
        let batch_end = (batch_start + config.batch_size).min(scale);
        let batch_t = Instant::now();
        for i in batch_start..batch_end {
            let vec = scaling_vector(config.vector_dims, i as u64);
            db.db
                .vector_upsert("scale_vec", &format!("v_{}", i), vec, None)
                .unwrap();
        }
        tracker.record_batch(batch_idx, batch_end - batch_start, batch_t.elapsed());
        batch_idx += 1;
        if !config.quiet && !config.csv && scale >= config.batch_size {
            eprintln!("    loaded {}/{}", batch_end, scale);
        }
    }

    let elapsed = start.elapsed();
    let snap_after = snapshot_resources(db);
    let disk = disk_mb(db);

    build_load_result(Tier::Vector, scale, scale, elapsed, &snap_before, &snap_after, disk, &tracker, config)
}

fn run_vector_search(db: &BenchDb, scale: usize, config: &Config) -> ScaleResult {
    let n = config.sample_ops;
    let mut rng = Lcg::new(55);

    let snap_before = snapshot_resources(db);
    let mut latencies = Vec::with_capacity(n);
    let wall = Instant::now();

    for _ in 0..n {
        let idx = rng.next_bounded(scale as u64);
        let query = scaling_vector(config.vector_dims, idx);
        let t = Instant::now();
        let _ = db.db.vector_search("scale_vec", query, 10);
        latencies.push(t.elapsed());
    }

    let wall_elapsed = wall.elapsed();
    let snap_after = snapshot_resources(db);
    build_latency_result(Tier::Vector, "search", scale, latencies, wall_elapsed, db, &snap_before, &snap_after, config)
}

fn run_graph_load(db: &BenchDb, scale: usize, config: &Config) -> ScaleResult {
    let (vertices, edges) = generate_synthetic_graph(scale, GRAPH_EDGES_PER_VERTEX);

    db.db.graph_create("scale_graph").unwrap();

    let snap_before = snapshot_resources(db);
    let mut tracker = LoadTracker::new();
    let start = Instant::now();

    // Bulk insert nodes in batches
    let node_batch = config.batch_size;
    let mut batch_idx = 0usize;
    for chunk in vertices.chunks(node_batch) {
        let batch_t = Instant::now();
        let node_ids: Vec<String> = chunk.iter().map(|v| v.to_string()).collect();
        let nodes: Vec<(&str, Option<&str>, Option<Value>)> =
            node_ids.iter().map(|id| (id.as_str(), None, None)).collect();
        db.db.graph_bulk_insert("scale_graph", &nodes, &[]).unwrap();
        tracker.record_batch(batch_idx, chunk.len(), batch_t.elapsed());
        batch_idx += 1;
    }

    if !config.quiet && !config.csv {
        eprintln!("    nodes done ({} vertices)", vertices.len());
    }

    // Bulk insert edges in batches
    let edge_batch = config.batch_size;
    let mut edges_done = 0usize;
    for chunk in edges.chunks(edge_batch) {
        let batch_t = Instant::now();
        let edge_strs: Vec<(String, String)> = chunk
            .iter()
            .map(|(s, d)| (s.to_string(), d.to_string()))
            .collect();
        let edge_tuples: Vec<(&str, &str, &str, Option<f64>, Option<Value>)> = edge_strs
            .iter()
            .map(|(s, d)| (s.as_str(), d.as_str(), "E", None, None))
            .collect();
        db.db
            .graph_bulk_insert("scale_graph", &[], &edge_tuples)
            .unwrap();
        tracker.record_batch(batch_idx, chunk.len(), batch_t.elapsed());
        batch_idx += 1;
        edges_done += chunk.len();
        if !config.quiet && !config.csv && edges.len() >= edge_batch {
            eprintln!("    edges {}/{}", edges_done, edges.len());
        }
    }

    let elapsed = start.elapsed();
    let snap_after = snapshot_resources(db);
    let disk = disk_mb(db);
    let total_items = vertices.len() + edges.len();

    build_load_result(Tier::Graph, scale, total_items, elapsed, &snap_before, &snap_after, disk, &tracker, config)
}

fn run_graph_bfs(db: &BenchDb, scale: usize, config: &Config) -> ScaleResult {
    let n = config.sample_ops;
    let mut rng = Lcg::new(33);

    let snap_before = snapshot_resources(db);
    let mut latencies = Vec::with_capacity(n);
    let wall = Instant::now();

    for _ in 0..n {
        let src = rng.next_bounded(scale as u64);
        let t = Instant::now();
        let _ = db.db.graph_bfs(
            "scale_graph",
            &src.to_string(),
            usize::MAX,
            None,
            None,
            Some("both"),
        );
        latencies.push(t.elapsed());
    }

    let wall_elapsed = wall.elapsed();
    let snap_after = snapshot_resources(db);
    build_latency_result(Tier::Graph, "bfs", scale, latencies, wall_elapsed, db, &snap_before, &snap_after, config)
}

// =============================================================================
// Common latency result builder
// =============================================================================

fn build_latency_result(
    tier: Tier,
    op: &str,
    scale: usize,
    mut latencies: Vec<Duration>,
    wall_elapsed: Duration,
    db: &BenchDb,
    snap_before: &ResourceSnapshot,
    snap_after: &ResourceSnapshot,
    config: &Config,
) -> ScaleResult {
    latencies.sort_unstable();
    let len = latencies.len();
    let sum: Duration = latencies.iter().sum();
    let delta = resource_delta(snap_before, snap_after);
    let disk = disk_mb(db);
    let raw = raw_data_mb(tier, scale, config);
    let space_amp = if raw > 0.0 { disk / raw } else { 0.0 };

    ScaleResult {
        tier,
        operation: op.to_string(),
        scale,
        ops_per_sec: len as f64 / wall_elapsed.as_secs_f64(),
        total_time_secs: wall_elapsed.as_secs_f64(),
        p50: latencies[len * 50 / 100],
        p95: latencies[(len * 95 / 100).min(len - 1)],
        p99: latencies[(len * 99 / 100).min(len - 1)],
        min: latencies[0],
        max: latencies[len - 1],
        avg: sum / len as u32,
        samples: len,
        rss_mb: snap_after.rss_mb,
        disk_mb: disk,
        resources: delta,
        space_amplification: space_amp,
        first_batch_ops_sec: 0.0,
        last_batch_ops_sec: 0.0,
    }
}

// =============================================================================
// Output Formatting
// =============================================================================

fn fmt_num(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn fmt_scale(n: usize) -> String {
    fmt_num(n as u64)
}

fn fmt_duration(d: Duration) -> String {
    let nanos = d.as_nanos();
    if nanos == 0 {
        "\u{2014}".to_string()
    } else if nanos < 1_000 {
        format!("{}ns", nanos)
    } else if nanos < 1_000_000 {
        format!("{:.1}us", nanos as f64 / 1_000.0)
    } else if nanos < 1_000_000_000 {
        format!("{:.1}ms", nanos as f64 / 1_000_000.0)
    } else {
        format!("{:.1}s", nanos as f64 / 1_000_000_000.0)
    }
}

fn fmt_time(secs: f64) -> String {
    if secs < 0.001 {
        format!("{:.0}us", secs * 1_000_000.0)
    } else if secs < 1.0 {
        format!("{:.1}ms", secs * 1000.0)
    } else if secs < 60.0 {
        format!("{:.1}s", secs)
    } else {
        format!("{:.1}m", secs / 60.0)
    }
}

fn fmt_size(mb: f64) -> String {
    if mb < 1.0 {
        format!("{:.0} KB", mb * 1024.0)
    } else if mb < 1024.0 {
        format!("{:.0} MB", mb)
    } else {
        format!("{:.1} GB", mb / 1024.0)
    }
}

fn print_load_header() {
    eprintln!(
        "  {:>12}  {:>11}  {:>8}  {:>10}  {:>10}  {:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>12}  {:>12}",
        "scale", "ops/sec", "time", "RSS", "disk", "sp.amp",
        "maj_flt", "io_read", "io_write", "cpu(u+s)",
        "1st_batch", "last_batch"
    );
}

fn print_load_row(r: &ScaleResult) {
    let cpu = format!(
        "{}+{}ms",
        r.resources.cpu_user_ms,
        r.resources.cpu_sys_ms,
    );
    eprintln!(
        "  {:>12}  {:>11}  {:>8}  {:>10}  {:>10}  {:>5.1}x  {:>10}  {:>10}  {:>10}  {:>10}  {:>12}  {:>12}",
        fmt_scale(r.scale),
        fmt_num(r.ops_per_sec as u64),
        fmt_time(r.total_time_secs),
        fmt_size(r.rss_mb),
        fmt_size(r.disk_mb),
        r.space_amplification,
        fmt_num(r.resources.major_faults),
        fmt_size(r.resources.io_read_mb),
        fmt_size(r.resources.io_write_mb),
        cpu,
        fmt_num(r.first_batch_ops_sec as u64),
        fmt_num(r.last_batch_ops_sec as u64),
    );
}

fn print_latency_header() {
    eprintln!(
        "  {:>12}  {:>11}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "scale", "ops/sec", "p50", "p95", "p99",
        "maj_flt", "io_read", "io_write", "cpu(u+s)"
    );
}

fn print_latency_row(r: &ScaleResult) {
    let cpu = format!(
        "{}+{}ms",
        r.resources.cpu_user_ms,
        r.resources.cpu_sys_ms,
    );
    eprintln!(
        "  {:>12}  {:>11}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        fmt_scale(r.scale),
        fmt_num(r.ops_per_sec as u64),
        fmt_duration(r.p50),
        fmt_duration(r.p95),
        fmt_duration(r.p99),
        fmt_num(r.resources.major_faults),
        fmt_size(r.resources.io_read_mb),
        fmt_size(r.resources.io_write_mb),
        cpu,
    );
}

fn print_quiet(r: &ScaleResult) {
    if r.is_load() {
        eprintln!(
            "{}/{}/{}: {} ops/sec, {}, RSS={}, disk={}, sp.amp={:.1}x, maj_flt={}, io_w={}, curve={}->{}",
            r.tier.label(),
            r.operation,
            fmt_scale(r.scale),
            fmt_num(r.ops_per_sec as u64),
            fmt_time(r.total_time_secs),
            fmt_size(r.rss_mb),
            fmt_size(r.disk_mb),
            r.space_amplification,
            fmt_num(r.resources.major_faults),
            fmt_size(r.resources.io_write_mb),
            fmt_num(r.first_batch_ops_sec as u64),
            fmt_num(r.last_batch_ops_sec as u64),
        );
    } else {
        eprintln!(
            "{}/{}/{}: {} ops/sec, p50={}, p99={}, maj_flt={}, io_r={}",
            r.tier.label(),
            r.operation,
            fmt_scale(r.scale),
            fmt_num(r.ops_per_sec as u64),
            fmt_duration(r.p50),
            fmt_duration(r.p99),
            fmt_num(r.resources.major_faults),
            fmt_size(r.resources.io_read_mb),
        );
    }
}

fn print_csv_header() {
    println!(
        "tier,operation,scale,ops_sec,p50_ns,p95_ns,p99_ns,rss_mb,disk_mb,\
         space_amp,major_faults,minor_faults,io_read_mb,io_write_mb,\
         cpu_user_ms,cpu_sys_ms,wal_appends,wal_syncs,\
         first_batch_ops_sec,last_batch_ops_sec"
    );
}

fn print_csv_row(r: &ScaleResult) {
    println!(
        "{},{},{},{:.2},{},{},{},{:.1},{:.1},\
         {:.2},{},{},{:.1},{:.1},\
         {},{},{},{},\
         {:.0},{:.0}",
        r.tier.label(),
        r.operation,
        r.scale,
        r.ops_per_sec,
        r.p50.as_nanos(),
        r.p95.as_nanos(),
        r.p99.as_nanos(),
        r.rss_mb,
        r.disk_mb,
        r.space_amplification,
        r.resources.major_faults,
        r.resources.minor_faults,
        r.resources.io_read_mb,
        r.resources.io_write_mb,
        r.resources.cpu_user_ms,
        r.resources.cpu_sys_ms,
        r.resources.wal_appends,
        r.resources.wal_syncs,
        r.first_batch_ops_sec,
        r.last_batch_ops_sec,
    );
}

// =============================================================================
// Recording
// =============================================================================

fn record_result(recorder: &mut ResultRecorder, r: &ScaleResult, config: &Config) {
    let mut params = HashMap::new();
    params.insert("tier".into(), serde_json::json!(r.tier.label()));
    params.insert("operation".into(), serde_json::json!(r.operation));
    params.insert("scale".into(), serde_json::json!(r.scale));
    params.insert(
        "durability".into(),
        serde_json::json!(config.durability.label()),
    );
    params.insert("rss_mb".into(), serde_json::json!(r.rss_mb));
    params.insert("disk_mb".into(), serde_json::json!(r.disk_mb));
    params.insert("space_amplification".into(), serde_json::json!(r.space_amplification));
    params.insert("major_faults".into(), serde_json::json!(r.resources.major_faults));
    params.insert("minor_faults".into(), serde_json::json!(r.resources.minor_faults));
    params.insert("io_read_mb".into(), serde_json::json!(r.resources.io_read_mb));
    params.insert("io_write_mb".into(), serde_json::json!(r.resources.io_write_mb));
    params.insert("cpu_user_ms".into(), serde_json::json!(r.resources.cpu_user_ms));
    params.insert("cpu_sys_ms".into(), serde_json::json!(r.resources.cpu_sys_ms));
    params.insert("wal_appends".into(), serde_json::json!(r.resources.wal_appends));
    params.insert("wal_syncs".into(), serde_json::json!(r.resources.wal_syncs));

    if r.is_load() {
        params.insert("first_batch_ops_sec".into(), serde_json::json!(r.first_batch_ops_sec));
        params.insert("last_batch_ops_sec".into(), serde_json::json!(r.last_batch_ops_sec));
    }

    match r.tier {
        Tier::Kv => {
            params.insert("kv_value_size".into(), serde_json::json!(config.kv_value_size));
        }
        Tier::Vector => {
            params.insert("dims".into(), serde_json::json!(config.vector_dims));
        }
        Tier::Graph => {
            params.insert(
                "edges_per_vertex".into(),
                serde_json::json!(GRAPH_EDGES_PER_VERTEX),
            );
        }
        _ => {}
    }

    let scale_label = if r.scale >= 1_000_000_000 {
        format!("{}b", r.scale / 1_000_000_000)
    } else if r.scale >= 1_000_000 {
        format!("{}m", r.scale / 1_000_000)
    } else if r.scale >= 1_000 {
        format!("{}k", r.scale / 1_000)
    } else {
        r.scale.to_string()
    };

    recorder.record(BenchmarkResult {
        benchmark: format!("scaling/{}/{}/{}", r.tier.label(), r.operation, scale_label),
        category: "scaling".to_string(),
        parameters: params,
        metrics: BenchmarkMetrics {
            ops_per_sec: Some(r.ops_per_sec),
            p50_ns: if r.p50 > Duration::ZERO {
                Some(r.p50.as_nanos() as u64)
            } else {
                None
            },
            p95_ns: if r.p95 > Duration::ZERO {
                Some(r.p95.as_nanos() as u64)
            } else {
                None
            },
            p99_ns: if r.p99 > Duration::ZERO {
                Some(r.p99.as_nanos() as u64)
            } else {
                None
            },
            min_ns: if r.min > Duration::ZERO {
                Some(r.min.as_nanos() as u64)
            } else {
                None
            },
            max_ns: if r.max > Duration::ZERO {
                Some(r.max.as_nanos() as u64)
            } else {
                None
            },
            avg_ns: if r.avg > Duration::ZERO {
                Some(r.avg.as_nanos() as u64)
            } else {
                None
            },
            samples: Some(r.samples as u64),
            ..Default::default()
        },
    });
}

// =============================================================================
// Tier runners
// =============================================================================

fn operations_for_tier(tier: Tier) -> Vec<&'static str> {
    match tier {
        Tier::Kv => vec!["random_read", "random_write", "scan"],
        Tier::Json => vec!["random_read", "path_read", "path_update"],
        Tier::Vector => vec!["search"],
        Tier::Graph => vec!["bfs"],
    }
}

fn run_operation(
    db: &BenchDb,
    tier: Tier,
    op: &str,
    scale: usize,
    config: &Config,
) -> ScaleResult {
    match (tier, op) {
        (Tier::Kv, "random_read") => run_kv_random_read(db, scale, config),
        (Tier::Kv, "random_write") => run_kv_random_write(db, scale, config),
        (Tier::Kv, "scan") => run_kv_scan(db, scale, config),
        (Tier::Json, "random_read") => run_json_random_read(db, scale, config),
        (Tier::Json, "path_read") => run_json_path_read(db, scale, config),
        (Tier::Json, "path_update") => run_json_path_update(db, scale, config),
        (Tier::Vector, "search") => run_vector_search(db, scale, config),
        (Tier::Graph, "bfs") => run_graph_bfs(db, scale, config),
        _ => unreachable!("unknown operation: {}/{}", tier.label(), op),
    }
}

fn run_load(db: &BenchDb, tier: Tier, scale: usize, config: &Config) -> ScaleResult {
    match tier {
        Tier::Kv => run_kv_load(db, scale, config),
        Tier::Json => run_json_load(db, scale, config),
        Tier::Vector => run_vector_load(db, scale, config),
        Tier::Graph => run_graph_load(db, scale, config),
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let config = parse_args();
    print_hardware_info();

    if !config.csv {
        eprintln!("=== StrataDB Scaling Benchmark ===");
        eprintln!(
            "Durability: {}, KV value size: {} bytes, Vector dims: {}",
            config.durability.label(),
            config.kv_value_size,
            config.vector_dims,
        );
        eprintln!(
            "Scales: {:?}, Sample ops: {}",
            config.scales, config.sample_ops
        );
        eprintln!(
            "Tiers: {:?}",
            config.tiers.iter().map(|t| t.label()).collect::<Vec<_>>()
        );
        eprintln!();
    }

    if config.csv {
        print_csv_header();
    }

    let mut recorder = ResultRecorder::new("scaling");

    for &tier in &config.tiers {
        let scales = effective_scales(tier, &config);

        if !config.csv && !config.quiet {
            let size_info = match tier {
                Tier::Kv => format!(", {} byte values", config.kv_value_size),
                Tier::Vector => format!(", {}d", config.vector_dims),
                Tier::Graph => format!(", ~{} edges/vertex", GRAPH_EDGES_PER_VERTEX),
                _ => String::new(),
            };
            eprintln!(
                "=== {} Scaling ({}{}) ===",
                tier.label().to_uppercase(),
                config.durability.label(),
                size_info,
            );
            eprintln!();
        }

        // Collect results per operation for table output
        let ops = {
            let mut v = vec!["load"];
            v.extend(operations_for_tier(tier));
            v
        };

        // Group results by operation for table display
        let mut op_results: HashMap<String, Vec<ScaleResult>> = HashMap::new();
        for op in &ops {
            op_results.insert(op.to_string(), Vec::new());
        }

        for &scale in &scales {
            // Fresh DB per (tier, scale)
            let db = create_db(config.durability);

            if !config.csv && !config.quiet {
                eprintln!(
                    "  --- {} @ {} ---",
                    tier.label(),
                    fmt_scale(scale)
                );
            }

            // Phase 1: Load
            let load_result = run_load(&db, tier, scale, &config);
            if config.csv {
                print_csv_row(&load_result);
            } else if config.quiet {
                print_quiet(&load_result);
            }
            record_result(&mut recorder, &load_result, &config);
            op_results.get_mut("load").unwrap().push(load_result);

            // Phase 2: Measure each operation
            for &op in operations_for_tier(tier).iter() {
                let result = run_operation(&db, tier, op, scale, &config);
                if config.csv {
                    print_csv_row(&result);
                } else if config.quiet {
                    print_quiet(&result);
                }
                record_result(&mut recorder, &result, &config);
                op_results.get_mut(op).unwrap().push(result);
            }
            // db dropped here, temp dir cleaned
        }

        // Print tables (non-csv, non-quiet)
        if !config.csv && !config.quiet {
            eprintln!();
            for op in &ops {
                let results = op_results.get(*op).unwrap();
                if results.is_empty() {
                    continue;
                }
                eprintln!("--- {} ---", op);
                if *op == "load" {
                    print_load_header();
                    for r in results {
                        print_load_row(r);
                    }
                } else {
                    print_latency_header();
                    for r in results {
                        print_latency_row(r);
                    }
                }
                eprintln!();
            }
        }
    }

    if !config.csv {
        eprintln!("=== Scaling benchmark complete ===");
    }
    let _ = recorder.save();
}
