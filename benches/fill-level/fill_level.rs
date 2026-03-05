//! Fill-Level Benchmark for StrataDB
//!
//! Measures how operation latency and throughput degrade as database size grows.
//! Tests run at fill levels of 0, 10K, 50K, 100K, and 250K pre-existing keys,
//! showing the performance curve for each operation.
//!
//! Uses a custom harness (like redis_compare.rs) instead of Criterion because:
//! - Clean table output showing fill level vs latency per operation
//! - Fill-level population has non-trivial setup time
//! - The comparison axis is fill level, not statistical convergence
//!
//! Run:    `cargo bench --bench fill_level`
//! Quick:  `cargo bench --bench fill_level -- -q`
//! CSV:    `cargo bench --bench fill_level -- --csv`
//! Custom: `cargo bench --bench fill_level -- --levels 0,1000,5000,10000`
//! Single: `cargo bench --bench fill_level -- -t kv_put`

#[allow(unused)]
#[path = "../harness/mod.rs"]
mod harness;

use harness::recorder::ResultRecorder;
use harness::{create_db, kv_value, print_hardware_info, BenchDb, DurabilityConfig};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use strata_benchmarks::schema::{BenchmarkMetrics, BenchmarkResult};
use stratadb::Value;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

const DEFAULT_OPS: usize = 10_000;
const DEFAULT_LEVELS: &[usize] = &[0, 10_000, 50_000, 100_000, 250_000];
const BENCH_VALUE_SIZE: usize = 64; // smaller bench values to focus on engine overhead

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct FillResult {
    name: String,
    fill_level: usize,
    total_ops: usize,
    elapsed: Duration,
    ops_per_sec: f64,
    avg: Duration,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    min: Duration,
    max: Duration,
}

// ---------------------------------------------------------------------------
// Core measurement (same pattern as redis_compare.rs)
// ---------------------------------------------------------------------------

fn run_bench(
    name: &str,
    fill_level: usize,
    total_ops: usize,
    mut bench_fn: impl FnMut(),
) -> FillResult {
    let mut latencies = Vec::with_capacity(total_ops);
    let wall_start = Instant::now();

    for _ in 0..total_ops {
        let op_start = Instant::now();
        bench_fn();
        latencies.push(op_start.elapsed());
    }

    let elapsed = wall_start.elapsed();
    latencies.sort_unstable();
    let len = latencies.len();
    let sum: Duration = latencies.iter().sum();

    FillResult {
        name: name.to_string(),
        fill_level,
        total_ops: len,
        elapsed,
        ops_per_sec: len as f64 / elapsed.as_secs_f64(),
        avg: sum / len as u32,
        p50: latencies[len * 50 / 100],
        p95: latencies[(len * 95 / 100).min(len - 1)],
        p99: latencies[(len * 99 / 100).min(len - 1)],
        min: latencies[0],
        max: latencies[len - 1],
    }
}

// ---------------------------------------------------------------------------
// Fill strategy
// ---------------------------------------------------------------------------

fn fill_database(db: &BenchDb, count: usize) {
    let fill_value = kv_value(); // 1KB
    for i in 0..count {
        let key = format!("fill:{:012}", i);
        db.db.kv_put(&key, fill_value.clone()).unwrap();
        if count >= 50_000 && (i + 1) % 50_000 == 0 {
            eprintln!("  filled {}/{} keys...", i + 1, count);
        }
    }
}

// ---------------------------------------------------------------------------
// Recording helper
// ---------------------------------------------------------------------------

fn record_fill_result(
    recorder: &mut ResultRecorder,
    r: &FillResult,
    mode: &DurabilityConfig,
) {
    let mut params = HashMap::new();
    params.insert("durability".into(), serde_json::json!(mode.label()));
    params.insert("fill_level".into(), serde_json::json!(r.fill_level));

    recorder.record(BenchmarkResult {
        benchmark: format!("fill-level/{}/{}keys", r.name, r.fill_level),
        category: "fill-level".to_string(),
        parameters: params,
        metrics: BenchmarkMetrics {
            ops_per_sec: Some(r.ops_per_sec),
            p50_ns: Some(r.p50.as_nanos() as u64),
            p95_ns: Some(r.p95.as_nanos() as u64),
            p99_ns: Some(r.p99.as_nanos() as u64),
            min_ns: Some(r.min.as_nanos() as u64),
            max_ns: Some(r.max.as_nanos() as u64),
            avg_ns: Some(r.avg.as_nanos() as u64),
            samples: Some(r.total_ops as u64),
            fill_level: Some(r.fill_level),
            ..Default::default()
        },
    });
}

// ---------------------------------------------------------------------------
// Benchmark functions
// ---------------------------------------------------------------------------

fn bench_kv_put(db: &BenchDb, n: usize, fill_level: usize) -> FillResult {
    let val = Value::Bytes(vec![0x42; BENCH_VALUE_SIZE]);
    let mut i = 0u64;
    run_bench("kv_put", fill_level, n, || {
        let key = format!("bench:{:012}", i);
        db.db.kv_put(&key, val.clone()).unwrap();
        i += 1;
    })
}

fn bench_kv_get(db: &BenchDb, n: usize, fill_level: usize) -> FillResult {
    // Pre-populate 100 read-target keys
    let val = Value::Bytes(vec![0x42; BENCH_VALUE_SIZE]);
    for i in 0..100u64 {
        let key = format!("read:{:012}", i);
        db.db.kv_put(&key, val.clone()).unwrap();
    }

    let mut i = 0u64;
    run_bench("kv_get", fill_level, n, || {
        let key = format!("read:{:012}", i % 100);
        let _ = db.db.kv_get(&key);
        i += 1;
    })
}

fn bench_kv_delete(db: &BenchDb, n: usize, fill_level: usize) -> FillResult {
    // Delete from fill keys (they exist from fill_database)
    // If fill_level is 0, pre-populate some keys to delete
    let delete_count = if fill_level == 0 { n } else { fill_level };
    if fill_level == 0 {
        let val = Value::Bytes(vec![0x42; BENCH_VALUE_SIZE]);
        for i in 0..n {
            let key = format!("fill:{:012}", i);
            db.db.kv_put(&key, val.clone()).unwrap();
        }
    }

    let mut i = 0u64;
    run_bench("kv_delete", fill_level, n, || {
        let key = format!("fill:{:012}", i % delete_count as u64);
        let _ = db.db.kv_delete(&key);
        i += 1;
    })
}

fn bench_kv_list(mode: DurabilityConfig, n: usize, fill_level: usize) -> FillResult {
    // Fresh database per fill level (same pattern as LRANGE_100 in redis_compare)
    let db = create_db(mode);
    fill_database(&db, fill_level);

    // Pre-populate 100 keys with scan: prefix
    let val = Value::Bytes(vec![0x42; BENCH_VALUE_SIZE]);
    for i in 0..100u64 {
        let key = format!("scan:{:012}", i);
        db.db.kv_put(&key, val.clone()).unwrap();
    }

    run_bench("kv_list", fill_level, n, || {
        let _ = db.db.kv_list(Some("scan:")).unwrap();
    })
}

fn bench_state_set(db: &BenchDb, n: usize, fill_level: usize) -> FillResult {
    let val = Value::Bytes(vec![0x53; BENCH_VALUE_SIZE]);
    let mut i = 0u64;
    run_bench("state_set", fill_level, n, || {
        let cell = format!("cell:{:012}", i);
        db.db.state_set(&cell, val.clone()).unwrap();
        i += 1;
    })
}

fn bench_state_read(db: &BenchDb, n: usize, fill_level: usize) -> FillResult {
    // Pre-populate 100 state cells
    let val = Value::Bytes(vec![0x53; BENCH_VALUE_SIZE]);
    for i in 0..100u64 {
        let cell = format!("rcell:{:012}", i);
        db.db.state_set(&cell, val.clone()).unwrap();
    }

    let mut i = 0u64;
    run_bench("state_read", fill_level, n, || {
        let cell = format!("rcell:{:012}", i % 100);
        let _ = db.db.state_get(&cell).unwrap();
        i += 1;
    })
}

fn bench_event_append(db: &BenchDb, n: usize, fill_level: usize) -> FillResult {
    let mut payload_map = HashMap::new();
    payload_map.insert(
        "data".to_string(),
        Value::Bytes(vec![0x45; BENCH_VALUE_SIZE]),
    );
    let payload = Value::object(payload_map);

    run_bench("event_append", fill_level, n, || {
        db.db.event_append("bench_stream", payload.clone()).unwrap();
    })
}

fn bench_event_read(db: &BenchDb, n: usize, fill_level: usize) -> FillResult {
    // Pre-append 1000 events
    let mut payload_map = HashMap::new();
    payload_map.insert("data".to_string(), Value::Int(0));
    let payload = Value::object(payload_map);
    for _ in 0..1000u64 {
        db.db
            .event_append("read_stream", payload.clone())
            .unwrap();
    }

    let mut rng: u64 = 0xdeadbeef;
    run_bench("event_read", fill_level, n, || {
        // Simple LCG for sequence selection
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let seq = (rng >> 33) % 1000 + 1; // 1-indexed
        let _ = db.db.event_get(seq).unwrap();
    })
}

// ---------------------------------------------------------------------------
// Output formatters
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

fn print_table_header() {
    eprintln!(
        "  {:>10}  {:>11}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}",
        "fill_level", "ops/sec", "avg", "p50", "p95", "p99", "max"
    );
}

fn print_table_row(r: &FillResult) {
    eprintln!(
        "  {:>10}  {:>11}  {:>8.3}ms  {:>8.3}ms  {:>8.3}ms  {:>8.3}ms  {:>8.3}ms",
        fmt_num(r.fill_level as u64),
        fmt_num(r.ops_per_sec as u64),
        duration_ms(r.avg),
        duration_ms(r.p50),
        duration_ms(r.p95),
        duration_ms(r.p99),
        duration_ms(r.max),
    );
}

fn print_quiet(r: &FillResult) {
    eprintln!(
        "{} @ {}: {} ops/sec, p50={:.3}ms",
        r.name,
        fmt_num(r.fill_level as u64),
        fmt_num(r.ops_per_sec as u64),
        duration_ms(r.p50),
    );
}

fn print_csv_header() {
    println!(
        "\"test\",\"fill_level\",\"ops_sec\",\"avg_ms\",\"p50_ms\",\"p95_ms\",\"p99_ms\",\"max_ms\""
    );
}

fn print_csv_row(r: &FillResult) {
    println!(
        "\"{}\",{},{:.2},{:.3},{:.3},{:.3},{:.3},{:.3}",
        r.name,
        r.fill_level,
        r.ops_per_sec,
        duration_ms(r.avg),
        duration_ms(r.p50),
        duration_ms(r.p95),
        duration_ms(r.p99),
        duration_ms(r.max),
    );
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Config {
    ops: usize,
    durability: DurabilityConfig,
    tests: Option<Vec<String>>,
    levels: Vec<usize>,
    csv: bool,
    quiet: bool,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut config = Config {
        ops: DEFAULT_OPS,
        durability: DurabilityConfig::Cache,
        tests: None,
        levels: DEFAULT_LEVELS.to_vec(),
        csv: false,
        quiet: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-n" => {
                i += 1;
                config.ops = args[i].parse().unwrap_or(DEFAULT_OPS);
            }
            "--durability" => {
                i += 1;
                config.durability = match args[i].as_str() {
                    "cache" => DurabilityConfig::Cache,
                    "standard" => DurabilityConfig::Standard,
                    "always" => DurabilityConfig::Always,
                    _ => DurabilityConfig::Cache,
                };
            }
            "-t" => {
                i += 1;
                let names: Vec<String> = args[i]
                    .split(',')
                    .map(|s| s.trim().to_lowercase())
                    .collect();
                config.tests = Some(names);
            }
            "--levels" => {
                i += 1;
                config.levels = args[i]
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
            }
            "--csv" => config.csv = true,
            "-q" => config.quiet = true,
            _ => {}
        }
        i += 1;
    }

    config
}

fn test_is_selected(name: &str, filter: &Option<Vec<String>>) -> bool {
    match filter {
        None => true,
        Some(names) => names
            .iter()
            .any(|f| name.to_lowercase().starts_with(&f.to_lowercase())),
    }
}

// ---------------------------------------------------------------------------
// Test names
// ---------------------------------------------------------------------------

const ALL_TESTS: &[&str] = &[
    "kv_put",
    "kv_get",
    "kv_delete",
    "kv_list",
    "state_set",
    "state_read",
    "event_append",
    "event_read",
];

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let config = parse_args();
    print_hardware_info();

    if !config.csv {
        eprintln!("=== StrataDB Fill-Level Benchmark ===");
        eprintln!("Measures operation latency as database size grows.");
        eprintln!();
        eprintln!(
            "Parameters: {} ops per measurement, {} bytes payload, {} mode",
            config.ops, BENCH_VALUE_SIZE, config.durability.label()
        );
        eprintln!(
            "Fill levels: {:?}",
            config.levels
        );
        eprintln!();
    }

    if config.csv {
        print_csv_header();
    }

    let mut recorder = ResultRecorder::new("fill-level");

    for test_name in ALL_TESTS {
        if !test_is_selected(test_name, &config.tests) {
            continue;
        }

        let mut results = Vec::new();

        for &level in &config.levels {
            if !config.csv && !config.quiet {
                eprint!("  populating {} fill keys for {}...", fmt_num(level as u64), test_name);
            }

            // kv_list uses a fresh database per fill level
            if *test_name == "kv_list" {
                let result = bench_kv_list(config.durability, config.ops, level);
                if !config.csv && !config.quiet {
                    eprintln!(" done");
                }
                record_fill_result(&mut recorder, &result, &config.durability);
                results.push(result);
                continue;
            }

            let db = create_db(config.durability);
            fill_database(&db, level);

            if !config.csv && !config.quiet {
                eprintln!(" done");
            }

            let result = match *test_name {
                "kv_put" => bench_kv_put(&db, config.ops, level),
                "kv_get" => bench_kv_get(&db, config.ops, level),
                "kv_delete" => bench_kv_delete(&db, config.ops, level),
                "state_set" => bench_state_set(&db, config.ops, level),
                "state_read" => bench_state_read(&db, config.ops, level),
                "event_append" => bench_event_append(&db, config.ops, level),
                "event_read" => bench_event_read(&db, config.ops, level),
                _ => unreachable!(),
            };

            record_fill_result(&mut recorder, &result, &config.durability);
            results.push(result);
        }

        // Output results
        if config.csv {
            for r in &results {
                print_csv_row(r);
            }
        } else if config.quiet {
            for r in &results {
                print_quiet(r);
            }
        } else {
            eprintln!();
            eprintln!("--- {} ---", test_name);
            print_table_header();
            for r in &results {
                print_table_row(r);
            }
            eprintln!();
        }
    }

    if !config.csv {
        eprintln!("=== Benchmark complete ===");
    }
    let _ = recorder.save();
}
