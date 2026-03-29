//! Branch isolation benchmark — establishes a new benchmark category.
//!
//! No competitor has git-style branching. This benchmark quantifies:
//! 1. Branch create latency at scale (1..1000 existing branches)
//! 2. Fork latency (snapshot of populated branch)
//! 3. Write isolation overhead (branch writes vs default writes)
//! 4. Concurrent writers on separate branches
//! 5. Merge latency at various divergence sizes
//!
//! Usage:
//!   cargo bench --bench branch_isolation
//!   cargo bench --bench branch_isolation -- --quick -q

#[allow(unused)]
#[path = "../harness/mod.rs"]
mod harness;

use std::collections::HashMap;
use std::sync::{Arc, Barrier};
use std::time::Instant;

use harness::recorder::ResultRecorder;
use harness::{
    create_db, kv_key, kv_value, measure_percentiles, print_hardware_info, report_percentiles,
    DurabilityConfig,
};
use stratadb::MergeStrategy;

// =============================================================================
// Configuration
// =============================================================================

const MEASURE_ITERATIONS: usize = 50;
const WRITE_OPS_PER_TEST: usize = 1_000;

const BRANCH_SCALES: &[usize] = &[1, 10, 100, 500, 1000];
const QUICK_BRANCH_SCALES: &[usize] = &[1, 10, 100];

const DIVERGENCE_SIZES: &[usize] = &[10, 100, 1_000, 10_000];
const QUICK_DIVERGENCE_SIZES: &[usize] = &[10, 100, 1_000];

const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8];

// =============================================================================
// 1. Branch create latency at scale
// =============================================================================

fn bench_branch_create_at_scale(
    recorder: &mut ResultRecorder,
    scales: &[usize],
) {
    eprintln!("\n=== Branch Create Latency at Scale ===");

    for &existing in scales {
        let bench_db = create_db(DurabilityConfig::Cache);

        // Pre-create branches to reach the target count
        for i in 0..existing {
            bench_db
                .db
                .create_branch(&format!("pre_{}", i))
                .unwrap();
        }

        let label = format!("branch/create_at_{}", existing);
        let mut counter = existing;
        let p = measure_percentiles(MEASURE_ITERATIONS, || {
            bench_db
                .db
                .create_branch(&format!("bench_{}", counter))
                .unwrap();
            counter += 1;
        });
        report_percentiles(&label, &p);

        let mut params = HashMap::new();
        params.insert("existing_branches".into(), serde_json::json!(existing));
        recorder.record_latency(
            &label,
            params,
            &p,
            None,
            MEASURE_ITERATIONS as u64,
        );
    }
}

// =============================================================================
// 2. Fork latency (snapshot of populated branch)
// =============================================================================

fn bench_fork_latency(
    recorder: &mut ResultRecorder,
    divergence_sizes: &[usize],
) {
    eprintln!("\n=== Fork Latency (populated branch) ===");

    for &data_size in divergence_sizes {
        // Fork requires disk-backed database
        let bench_db = create_db(DurabilityConfig::Standard);

        // Populate default branch
        for i in 0..data_size as u64 {
            bench_db.db.kv_put(&kv_key(i), kv_value()).unwrap();
        }

        let label = format!("branch/fork_at_{}", data_size);
        let mut counter = 0usize;
        let p = measure_percentiles(MEASURE_ITERATIONS, || {
            bench_db
                .db
                .fork_branch(&format!("fork_{}", counter))
                .unwrap();
            counter += 1;
        });
        report_percentiles(&label, &p);

        let mut params = HashMap::new();
        params.insert("data_size".into(), serde_json::json!(data_size));
        recorder.record_latency(
            &label,
            params,
            &p,
            None,
            MEASURE_ITERATIONS as u64,
        );
    }
}

// =============================================================================
// 3. Write isolation overhead
// =============================================================================

fn bench_write_isolation_overhead(recorder: &mut ResultRecorder) {
    eprintln!("\n=== Write Isolation Overhead ===");

    // Baseline: writes on default branch (use Standard for fair comparison with fork)
    let bench_db = create_db(DurabilityConfig::Standard);
    let label_default = "branch/write_default";
    let mut counter = 0u64;
    let p_default = measure_percentiles(WRITE_OPS_PER_TEST, || {
        bench_db.db.kv_put(&kv_key(counter), kv_value()).unwrap();
        counter += 1;
    });
    report_percentiles(label_default, &p_default);

    let mut params = HashMap::new();
    params.insert("branch".into(), serde_json::json!("default"));
    recorder.record_latency(
        label_default,
        params,
        &p_default,
        None,
        WRITE_OPS_PER_TEST as u64,
    );

    // Writes on a forked branch (fork requires disk-backed DB)
    let mut bench_db2 = create_db(DurabilityConfig::Standard);
    // Populate some data first
    for i in 0..1000u64 {
        bench_db2.db.kv_put(&kv_key(i), kv_value()).unwrap();
    }
    bench_db2.db.fork_branch("experiment").unwrap();
    bench_db2.db.set_branch("experiment").unwrap();

    let label_branch = "branch/write_forked";
    counter = 10_000;
    let p_branch = measure_percentiles(WRITE_OPS_PER_TEST, || {
        bench_db2.db.kv_put(&kv_key(counter), kv_value()).unwrap();
        counter += 1;
    });
    report_percentiles(label_branch, &p_branch);

    let mut params = HashMap::new();
    params.insert("branch".into(), serde_json::json!("experiment"));
    recorder.record_latency(
        label_branch,
        params,
        &p_branch,
        None,
        WRITE_OPS_PER_TEST as u64,
    );

    // Report overhead
    let overhead_pct = if p_default.p50.as_nanos() > 0 {
        ((p_branch.p50.as_nanos() as f64 / p_default.p50.as_nanos() as f64) - 1.0) * 100.0
    } else {
        0.0
    };
    eprintln!(
        "  Branch write overhead: {:.1}% (default p50={:?}, branch p50={:?})",
        overhead_pct, p_default.p50, p_branch.p50,
    );
}

// =============================================================================
// 4. Concurrent writers on separate branches
// =============================================================================

fn bench_concurrent_branch_writers(recorder: &mut ResultRecorder, thread_counts: &[usize]) {
    eprintln!("\n=== Concurrent Branch Writers ===");

    let ops_per_thread = WRITE_OPS_PER_TEST;

    for &num_threads in thread_counts {
        let bench_db = create_db(DurabilityConfig::Cache);
        let db = Arc::new(bench_db.db);

        // Create one branch per thread
        for t in 0..num_threads {
            db.create_branch(&format!("thread_{}", t)).unwrap();
        }

        let barrier = Arc::new(Barrier::new(num_threads));

        let start = Instant::now();
        std::thread::scope(|s| {
            for t in 0..num_threads {
                let db = db.clone();
                let barrier = barrier.clone();
                s.spawn(move || {
                    // Each thread gets its own Strata handle pointed at its branch
                    let mut local_db =
                        stratadb::Strata::from_database(db.database()).unwrap();
                    local_db
                        .set_branch(&format!("thread_{}", t))
                        .unwrap();

                    barrier.wait();

                    for i in 0..ops_per_thread {
                        let key = format!("t{}_{:08}", t, i);
                        local_db.kv_put(&key, kv_value()).unwrap();
                    }
                });
            }
        });
        let elapsed = start.elapsed();

        let total_ops = num_threads * ops_per_thread;
        let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();

        let label = format!("branch/concurrent_writers/{}_threads", num_threads);
        eprintln!(
            "  {:<45} {} ops in {:?} ({:.0} ops/sec)",
            label, total_ops, elapsed, ops_per_sec,
        );

        let mut params = HashMap::new();
        params.insert("threads".into(), serde_json::json!(num_threads));
        params.insert("ops_per_thread".into(), serde_json::json!(ops_per_thread));
        params.insert("total_ops".into(), serde_json::json!(total_ops));
        params.insert("ops_per_sec".into(), serde_json::json!(ops_per_sec));

        // Use a simple percentile from the total duration
        let avg_ns = elapsed.as_nanos() as u64 / total_ops as u64;
        let p = harness::Percentiles {
            p50: std::time::Duration::from_nanos(avg_ns),
            p95: std::time::Duration::from_nanos(avg_ns),
            p99: std::time::Duration::from_nanos(avg_ns),
            min: std::time::Duration::from_nanos(avg_ns),
            max: elapsed,
            samples: total_ops,
        };
        recorder.record_latency(
            &label,
            params,
            &p,
            None,
            total_ops as u64,
        );
    }
}

// =============================================================================
// 5. Merge latency at various divergence sizes
// =============================================================================

fn bench_merge_latency(
    recorder: &mut ResultRecorder,
    divergence_sizes: &[usize],
) {
    eprintln!("\n=== Merge Latency ===");

    for &div_size in divergence_sizes {
        let label = format!("branch/merge/{}", div_size);
        let p = measure_percentiles(std::cmp::min(MEASURE_ITERATIONS, 20), || {
            // Merge requires disk-backed database (fork_branch dependency)
            let mut bench_db = create_db(DurabilityConfig::Standard);

            // Populate default branch with base data
            for i in 0..100u64 {
                bench_db
                    .db
                    .kv_put(&format!("base_{}", i), kv_value())
                    .unwrap();
            }

            // Fork and diverge
            bench_db.db.fork_branch("feature").unwrap();
            bench_db.db.set_branch("feature").unwrap();

            for i in 0..div_size as u64 {
                bench_db
                    .db
                    .kv_put(&format!("feature_{}", i), kv_value())
                    .unwrap();
            }

            bench_db.db.set_branch("default").unwrap();

            // Measure merge
            bench_db
                .db
                .merge_branches("feature", "default", MergeStrategy::LastWriterWins)
                .unwrap();
        });
        report_percentiles(&label, &p);

        let mut params = HashMap::new();
        params.insert("divergence_size".into(), serde_json::json!(div_size));
        recorder.record_latency(
            &label,
            params,
            &p,
            None,
            MEASURE_ITERATIONS as u64,
        );
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    print_hardware_info();

    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick" || a == "-q");

    let branch_scales = if quick { QUICK_BRANCH_SCALES } else { BRANCH_SCALES };
    let divergence_sizes = if quick {
        QUICK_DIVERGENCE_SIZES
    } else {
        DIVERGENCE_SIZES
    };

    eprintln!(
        "\n=== Branch Isolation Benchmark ({} mode) ===",
        if quick { "quick" } else { "full" }
    );

    let mut recorder = ResultRecorder::new("branch-isolation");

    bench_branch_create_at_scale(&mut recorder, branch_scales);
    bench_fork_latency(&mut recorder, divergence_sizes);
    bench_write_isolation_overhead(&mut recorder);
    bench_concurrent_branch_writers(&mut recorder, THREAD_COUNTS);
    bench_merge_latency(&mut recorder, divergence_sizes);

    eprintln!("\n=== Summary ===");
    if let Ok(path) = recorder.save() {
        eprintln!("  Results saved to: {}", path.display());
    }
}
