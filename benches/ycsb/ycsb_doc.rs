//! YCSB-Doc (Yahoo! Cloud Serving Benchmark) for StrataDB JSON Document API
//!
//! Measures native Rust JSON document performance using standard YCSB workloads A-F.
//! Uses path-level mutations for updates instead of full-document overwrite.
//!
//! Run:    `cargo bench --bench ycsb_doc`
//! Quick:  `cargo bench --bench ycsb_doc -- -q`
//! Single: `cargo bench --bench ycsb_doc -- --workload a`
//! Custom: `cargo bench --bench ycsb_doc -- --records 1000000 --ops 1000000`
//! CSV:    `cargo bench --bench ycsb_doc -- --csv`

#[allow(unused)]
#[path = "../harness/mod.rs"]
mod harness;

mod workloads;

use harness::recorder::ResultRecorder;
use harness::{create_db, json_document, print_hardware_info, BenchDb, DurabilityConfig};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use strata_benchmarks::schema::{BenchmarkMetrics, BenchmarkResult};
use stratadb::Value;
use workloads::{
    ycsb_key, FastRng, KeyChooser, Operation, WorkloadSpec, workload_by_label,
};

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const DEFAULT_RECORDS: usize = 100_000;
const DEFAULT_OPS: usize = 100_000;

// ---------------------------------------------------------------------------
// Per-operation latency collection
// ---------------------------------------------------------------------------

#[derive(Default)]
struct OpLatencies {
    read: Vec<Duration>,
    update: Vec<Duration>,
    insert: Vec<Duration>,
    scan: Vec<Duration>,
    rmw: Vec<Duration>,
}

impl OpLatencies {
    fn all(&self) -> Vec<Duration> {
        let mut all = Vec::with_capacity(
            self.read.len()
                + self.update.len()
                + self.insert.len()
                + self.scan.len()
                + self.rmw.len(),
        );
        all.extend_from_slice(&self.read);
        all.extend_from_slice(&self.update);
        all.extend_from_slice(&self.insert);
        all.extend_from_slice(&self.scan);
        all.extend_from_slice(&self.rmw);
        all
    }
}

struct LatencyStats {
    count: usize,
    ops_per_sec: f64,
    avg: Duration,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    min: Duration,
    max: Duration,
}

fn compute_stats(mut latencies: Vec<Duration>, wall_elapsed: Option<Duration>) -> Option<LatencyStats> {
    if latencies.is_empty() {
        return None;
    }
    latencies.sort_unstable();
    let len = latencies.len();
    let sum: Duration = latencies.iter().sum();
    let elapsed = wall_elapsed.unwrap_or(sum);

    Some(LatencyStats {
        count: len,
        ops_per_sec: len as f64 / elapsed.as_secs_f64(),
        avg: sum / len as u32,
        p50: latencies[len * 50 / 100],
        p95: latencies[(len * 95 / 100).min(len - 1)],
        p99: latencies[(len * 99 / 100).min(len - 1)],
        min: latencies[0],
        max: latencies[len - 1],
    })
}

// ---------------------------------------------------------------------------
// Load phase
// ---------------------------------------------------------------------------

struct LoadResult {
    record_count: usize,
    elapsed: Duration,
    ops_per_sec: f64,
}

fn run_load_phase(db: &BenchDb, record_count: usize) -> LoadResult {
    let start = Instant::now();

    for i in 0..record_count {
        let key = ycsb_key(i);
        db.db.json_set(&key, "$", json_document(i as u64)).unwrap();
    }

    let elapsed = start.elapsed();
    LoadResult {
        record_count,
        elapsed,
        ops_per_sec: record_count as f64 / elapsed.as_secs_f64(),
    }
}

// ---------------------------------------------------------------------------
// Run phase
// ---------------------------------------------------------------------------

struct RunResult {
    latencies: OpLatencies,
    wall_elapsed: Duration,
}

fn run_workload_phase(
    db: &BenchDb,
    workload: &WorkloadSpec,
    record_count: usize,
    operation_count: usize,
) -> RunResult {
    let mut rng = FastRng::new(0xABCD_2026);
    let mut key_chooser = KeyChooser::new(workload.distribution, record_count);
    let mut insert_counter = record_count; // next key to insert

    let mut latencies = OpLatencies::default();

    let wall_start = Instant::now();

    for _ in 0..operation_count {
        let op = workload.choose_operation(rng.next_f64());

        match op {
            Operation::Read => {
                let idx = key_chooser.next(&mut rng);
                let key = ycsb_key(idx);
                let start = Instant::now();
                let _ = db.db.json_get(&key, "$");
                latencies.read.push(start.elapsed());
            }
            Operation::Update => {
                let idx = key_chooser.next(&mut rng);
                let key = ycsb_key(idx);
                let new_score = Value::Float(idx as f64 * 99.9);
                let start = Instant::now();
                db.db
                    .json_set(&key, "$.metadata.mid_score", new_score)
                    .unwrap();
                latencies.update.push(start.elapsed());
            }
            Operation::Insert => {
                let key = ycsb_key(insert_counter);
                insert_counter += 1;
                key_chooser.set_max_key(insert_counter);
                let start = Instant::now();
                db.db
                    .json_set(&key, "$", json_document(insert_counter as u64))
                    .unwrap();
                latencies.insert.push(start.elapsed());
            }
            Operation::Scan => {
                let idx = key_chooser.next(&mut rng);
                let prefix = format!("user{:010}", idx);
                let start = Instant::now();
                let _ = db.db.json_list(Some(prefix), None, 100);
                latencies.scan.push(start.elapsed());
            }
            Operation::ReadModifyWrite => {
                let idx = key_chooser.next(&mut rng);
                let key = ycsb_key(idx);
                let start = Instant::now();
                let _ = db.db.json_get(&key, "$");
                let new_score = Value::Float(idx as f64 * 99.9);
                db.db
                    .json_set(&key, "$.metadata.mid_score", new_score)
                    .unwrap();
                latencies.rmw.push(start.elapsed());
            }
        }
    }

    let wall_elapsed = wall_start.elapsed();
    RunResult {
        latencies,
        wall_elapsed,
    }
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

fn duration_ms(d: Duration) -> f64 {
    d.as_nanos() as f64 / 1_000_000.0
}

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

fn print_workload_header(workload: &WorkloadSpec, config: &Config) {
    eprintln!();
    eprintln!(
        "=== YCSB-Doc Workload {}: {} ({}) ===",
        workload.label.to_ascii_uppercase(),
        workload.name,
        workload.mix_label(),
    );
    eprintln!(
        "  records={}  ops={}  durability={}",
        fmt_num(config.records as u64),
        fmt_num(config.ops as u64),
        config.durability.label(),
    );
    eprintln!();
}

fn print_load_result(load: &LoadResult) {
    eprintln!(
        "  Load: {} records in {:.3}s ({} ops/s)",
        fmt_num(load.record_count as u64),
        load.elapsed.as_secs_f64(),
        fmt_num(load.ops_per_sec as u64),
    );
    eprintln!();
}

fn print_run_table(latencies: &OpLatencies, wall_elapsed: Duration) {
    eprintln!(
        "  {:<14} {:>8}  {:>10}  {:>9}  {:>9}  {:>9}  {:>9}",
        "Operation", "count", "ops/sec", "p50", "p95", "p99", "max"
    );
    eprintln!(
        "  {}",
        "-".repeat(74)
    );

    // Overall
    let all = latencies.all();
    if let Some(s) = compute_stats(all, Some(wall_elapsed)) {
        eprintln!(
            "  {:<14} {:>8}  {:>10}  {:>8.3}ms  {:>8.3}ms  {:>8.3}ms  {:>8.3}ms",
            "overall",
            fmt_num(s.count as u64),
            fmt_num(s.ops_per_sec as u64),
            duration_ms(s.p50),
            duration_ms(s.p95),
            duration_ms(s.p99),
            duration_ms(s.max),
        );
    }

    // Per-op breakdown
    let ops: &[(&str, &[Duration])] = &[
        ("read", &latencies.read),
        ("update", &latencies.update),
        ("insert", &latencies.insert),
        ("scan", &latencies.scan),
        ("rmw", &latencies.rmw),
    ];

    for (name, lats) in ops {
        if let Some(s) = compute_stats(lats.to_vec(), None) {
            eprintln!(
                "  {:<14} {:>8}  {:>10}  {:>8.3}ms  {:>8.3}ms  {:>8.3}ms  {:>8.3}ms",
                name,
                fmt_num(s.count as u64),
                "",
                duration_ms(s.p50),
                duration_ms(s.p95),
                duration_ms(s.p99),
                duration_ms(s.max),
            );
        }
    }
    eprintln!();
}

fn print_quiet(workload: &WorkloadSpec, overall: &LatencyStats, load: &LoadResult) {
    eprintln!(
        "workload-{}: load={} ops/s, run={} ops/s, p50={:.3}ms, p99={:.3}ms",
        workload.label.to_ascii_uppercase(),
        fmt_num(load.ops_per_sec as u64),
        fmt_num(overall.ops_per_sec as u64),
        duration_ms(overall.p50),
        duration_ms(overall.p99),
    );
}

fn print_csv_header() {
    println!(
        "\"workload\",\"phase\",\"operation\",\"count\",\"ops_sec\",\"p50_ms\",\"p95_ms\",\"p99_ms\",\"max_ms\""
    );
}

fn print_csv_load(workload: &WorkloadSpec, load: &LoadResult) {
    println!(
        "\"{}\",\"load\",\"insert\",{},{:.2},,,",
        workload.label, load.record_count, load.ops_per_sec,
    );
}

fn print_csv_run(workload: &WorkloadSpec, name: &str, stats: &LatencyStats) {
    println!(
        "\"{}\",\"run\",\"{}\",{},{:.2},{:.3},{:.3},{:.3},{:.3}",
        workload.label,
        name,
        stats.count,
        stats.ops_per_sec,
        duration_ms(stats.p50),
        duration_ms(stats.p95),
        duration_ms(stats.p99),
        duration_ms(stats.max),
    );
}

// ---------------------------------------------------------------------------
// JSON recording
// ---------------------------------------------------------------------------

fn record_workload_result(
    recorder: &mut ResultRecorder,
    workload: &WorkloadSpec,
    config: &Config,
    load: &LoadResult,
    run: &RunResult,
) {
    let all_latencies = run.latencies.all();
    let overall = match compute_stats(all_latencies, Some(run.wall_elapsed)) {
        Some(s) => s,
        None => return,
    };

    let mut params = HashMap::new();
    params.insert("workload".into(), serde_json::json!(format!("{}", workload.label)));
    params.insert("workload_name".into(), serde_json::json!(workload.name));
    params.insert("record_count".into(), serde_json::json!(config.records));
    params.insert("operation_count".into(), serde_json::json!(config.ops));
    params.insert("durability".into(), serde_json::json!(config.durability.label()));
    params.insert("distribution".into(), serde_json::json!(workload.distribution.label()));
    params.insert("auto_embed".into(), serde_json::json!(!config.no_embed));
    params.insert(
        "load_ops_per_sec".into(),
        serde_json::json!(load.ops_per_sec),
    );
    params.insert(
        "load_elapsed_ms".into(),
        serde_json::json!(load.elapsed.as_millis() as u64),
    );

    // Per-op type stats in parameters
    let op_vecs: &[(&str, &[Duration])] = &[
        ("read", &run.latencies.read),
        ("update", &run.latencies.update),
        ("insert", &run.latencies.insert),
        ("scan", &run.latencies.scan),
        ("rmw", &run.latencies.rmw),
    ];
    for (name, lats) in op_vecs {
        if let Some(s) = compute_stats(lats.to_vec(), None) {
            params.insert(
                format!("{}_count", name),
                serde_json::json!(s.count),
            );
            params.insert(
                format!("{}_p50_ns", name),
                serde_json::json!(s.p50.as_nanos() as u64),
            );
            params.insert(
                format!("{}_p95_ns", name),
                serde_json::json!(s.p95.as_nanos() as u64),
            );
            params.insert(
                format!("{}_p99_ns", name),
                serde_json::json!(s.p99.as_nanos() as u64),
            );
            params.insert(
                format!("{}_avg_ns", name),
                serde_json::json!(s.avg.as_nanos() as u64),
            );
        }
    }

    let record_label = if config.records >= 1_000_000 {
        format!("{}m", config.records / 1_000_000)
    } else {
        format!("{}k", config.records / 1_000)
    };

    recorder.record(BenchmarkResult {
        benchmark: format!(
            "ycsb-doc/workload-{}/{}-{}",
            workload.label,
            record_label,
            config.durability.label()
        ),
        category: "ycsb-doc".to_string(),
        parameters: params,
        metrics: BenchmarkMetrics {
            ops_per_sec: Some(overall.ops_per_sec),
            p50_ns: Some(overall.p50.as_nanos() as u64),
            p95_ns: Some(overall.p95.as_nanos() as u64),
            p99_ns: Some(overall.p99.as_nanos() as u64),
            min_ns: Some(overall.min.as_nanos() as u64),
            max_ns: Some(overall.max.as_nanos() as u64),
            avg_ns: Some(overall.avg.as_nanos() as u64),
            samples: Some(overall.count as u64),
            ..Default::default()
        },
    });
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Config {
    workloads: Vec<char>,
    records: usize,
    ops: usize,
    durability: DurabilityConfig,
    no_embed: bool,
    csv: bool,
    quiet: bool,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut config = Config {
        workloads: vec!['a', 'b', 'c', 'd', 'e', 'f'],
        records: DEFAULT_RECORDS,
        ops: DEFAULT_OPS,
        durability: DurabilityConfig::Standard,
        no_embed: false,
        csv: false,
        quiet: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--workload" | "-w" => {
                i += 1;
                if i < args.len() {
                    config.workloads = args[i]
                        .split(',')
                        .filter_map(|s| s.trim().chars().next())
                        .map(|c| c.to_ascii_lowercase())
                        .collect();
                }
            }
            "--records" => {
                i += 1;
                if i < args.len() {
                    config.records = args[i].parse().unwrap_or(DEFAULT_RECORDS);
                }
            }
            "--ops" => {
                i += 1;
                if i < args.len() {
                    config.ops = args[i].parse().unwrap_or(DEFAULT_OPS);
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
            "--no-embed" | "--raw" => config.no_embed = true,
            "--csv" => config.csv = true,
            "-q" => config.quiet = true,
            _ => {}
        }
        i += 1;
    }

    config
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let config = parse_args();
    print_hardware_info();

    if !config.csv && !config.quiet {
        eprintln!("=== StrataDB YCSB-Doc Benchmark ===");
        eprintln!(
            "Parameters: {} records, {} ops, {} durability{}",
            fmt_num(config.records as u64),
            fmt_num(config.ops as u64),
            config.durability.label(),
            if config.no_embed { ", auto_embed=off" } else { "" },
        );
    }

    if config.csv {
        print_csv_header();
    }

    let mut recorder = ResultRecorder::new("ycsb-doc");

    for &label in &config.workloads {
        let workload = match workload_by_label(label) {
            Some(w) => w,
            None => {
                eprintln!("Unknown workload: '{}', skipping", label);
                continue;
            }
        };

        // Create a fresh database for each workload
        let db = create_db(config.durability);

        // Disable search features for pure document benchmarking
        if config.no_embed {
            db.db.config_set("auto_embed", "false").unwrap();
        }

        // --- Load phase ---
        if !config.csv && !config.quiet {
            print_workload_header(workload, &config);
            eprint!("  Loading {} records...", fmt_num(config.records as u64));
        }

        let load = run_load_phase(&db, config.records);

        if !config.csv && !config.quiet {
            eprintln!(" done");
            print_load_result(&load);
        }

        // --- Run phase ---
        let run = run_workload_phase(&db, workload, config.records, config.ops);

        // --- Output ---
        if config.csv {
            print_csv_load(workload, &load);
            let all = run.latencies.all();
            if let Some(s) = compute_stats(all, Some(run.wall_elapsed)) {
                print_csv_run(workload, "overall", &s);
            }
            let ops: &[(&str, &[Duration])] = &[
                ("read", &run.latencies.read),
                ("update", &run.latencies.update),
                ("insert", &run.latencies.insert),
                ("scan", &run.latencies.scan),
                ("rmw", &run.latencies.rmw),
            ];
            for (name, lats) in ops {
                if let Some(s) = compute_stats(lats.to_vec(), None) {
                    print_csv_run(workload, name, &s);
                }
            }
        } else if config.quiet {
            let all = run.latencies.all();
            if let Some(s) = compute_stats(all, Some(run.wall_elapsed)) {
                print_quiet(workload, &s, &load);
            }
        } else {
            print_run_table(&run.latencies, run.wall_elapsed);
        }

        // --- Record ---
        record_workload_result(&mut recorder, workload, &config, &load, &run);
    }

    if !config.csv && !config.quiet {
        eprintln!("=== YCSB-Doc benchmark complete ===");
    }
    let _ = recorder.save();
}
