//! ANN (Approximate Nearest Neighbor) Benchmark for StrataDB
//!
//! Measures the standard ANN trade-off: Recall@k vs Queries Per Second (QPS),
//! following ann-benchmarks.com methodology with synthetic clustered data.
//! Compares Strata's HNSW against instant-distance as a baseline.
//!
//! Run:    `cargo bench --bench ann`
//! Quick:  `cargo bench --bench ann -- --quick -q`
//! Custom: `cargo bench --bench ann -- --scales 10000,50000 --dims 32,128 --ks 1,10`
//! Strata only: `cargo bench --bench ann -- --strata-only`
//! CSV:    `cargo bench --bench ann -- --csv`

#[allow(unused)]
#[path = "../harness/mod.rs"]
mod harness;

mod dataset;

use dataset::{compute_ground_truth, compute_recall, generate_dataset};
use harness::recorder::ResultRecorder;
use harness::{create_db, print_hardware_info, DurabilityConfig};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use strata_benchmarks::schema::{BenchmarkMetrics, BenchmarkResult};
use stratadb::DistanceMetric;

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const QUICK_SCALES: &[usize] = &[10_000, 50_000, 100_000];
const FULL_SCALES: &[usize] = &[10_000, 50_000, 100_000, 500_000, 1_000_000];
const DEFAULT_DIMS: &[usize] = &[32, 128, 512];
const DEFAULT_KS: &[usize] = &[1, 10, 100];
const DEFAULT_QUERIES: usize = 100;
const SEED: u64 = 0xA00_2026;

// ---------------------------------------------------------------------------
// instant-distance wrapper
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct CosinePoint(Vec<f32>);

impl instant_distance::Point for CosinePoint {
    fn distance(&self, other: &Self) -> f32 {
        // Dataset is L2-normalized, so cosine_distance = 1 - dot_product
        let dot: f32 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();
        1.0 - dot
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

struct AnnResult {
    engine: String,
    scale: usize,
    dim: usize,
    k: usize,
    build_qps: f64,
    search_qps: f64,
    recall: f64,
    latencies: Vec<Duration>,
    p50: Duration,
    p95: Duration,
    p99: Duration,
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

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

fn fmt_duration(d: Duration) -> String {
    let nanos = d.as_nanos();
    if nanos < 1_000 {
        format!("{:>7}ns", nanos)
    } else if nanos < 1_000_000 {
        format!("{:>6.1}us", nanos as f64 / 1_000.0)
    } else if nanos < 1_000_000_000 {
        format!("{:>6.1}ms", nanos as f64 / 1_000_000.0)
    } else {
        format!("{:>6.2}s ", nanos as f64 / 1_000_000_000.0)
    }
}

fn scale_label(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}m", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}k", n / 1_000)
    } else {
        format!("{}", n)
    }
}

fn print_table_header() {
    eprintln!(
        "  {:>16}  {:>10}  {:>5}  {:>5}  {:>10}  {:>10}  {:>8}  {:>10}  {:>10}  {:>10}",
        "engine", "scale", "dim", "k", "build QPS", "search QPS", "recall", "p50", "p95", "p99"
    );
}

fn print_table_row(r: &AnnResult) {
    eprintln!(
        "  {:>16}  {:>10}  {:>5}  {:>5}  {:>10}  {:>10}  {:>8.4}  {:>10}  {:>10}  {:>10}",
        r.engine,
        fmt_num(r.scale as u64),
        r.dim,
        r.k,
        fmt_num(r.build_qps as u64),
        fmt_num(r.search_qps as u64),
        r.recall,
        fmt_duration(r.p50),
        fmt_duration(r.p95),
        fmt_duration(r.p99),
    );
}

fn print_quiet(r: &AnnResult) {
    eprintln!(
        "{} ann {}@k={}/{}d: recall={:.4}, search={} QPS, build={} QPS, p50={}",
        r.engine,
        fmt_num(r.scale as u64),
        r.k,
        r.dim,
        r.recall,
        fmt_num(r.search_qps as u64),
        fmt_num(r.build_qps as u64),
        fmt_duration(r.p50),
    );
}

fn print_csv_header() {
    println!(
        "\"engine\",\"scale\",\"k\",\"dim\",\"build_qps\",\"search_qps\",\"recall\",\"p50_us\",\"p95_us\",\"p99_us\""
    );
}

fn print_csv_row(r: &AnnResult) {
    println!(
        "\"{}\",{},{},{},{:.2},{:.2},{:.6},{:.1},{:.1},{:.1}",
        r.engine,
        r.scale,
        r.k,
        r.dim,
        r.build_qps,
        r.search_qps,
        r.recall,
        r.p50.as_nanos() as f64 / 1_000.0,
        r.p95.as_nanos() as f64 / 1_000.0,
        r.p99.as_nanos() as f64 / 1_000.0,
    );
}

fn print_reference_points() {
    eprintln!();
    eprintln!("  Published reference points (ann-benchmarks.com, 128d cosine, ~1M vectors):");
    eprintln!("    hnswlib   ~25,000 QPS @ 0.95 recall");
    eprintln!("    FAISS-IVF ~10,000 QPS @ 0.90 recall");
    eprintln!("    Annoy     ~ 5,000 QPS @ 0.85 recall");
    eprintln!("    ScaNN     ~30,000 QPS @ 0.95 recall");
}

// ---------------------------------------------------------------------------
// JSON recording
// ---------------------------------------------------------------------------

fn record_result(recorder: &mut ResultRecorder, r: &AnnResult, config: &Config) {
    let mut params = HashMap::new();
    params.insert("engine".into(), serde_json::json!(r.engine));
    params.insert("scale".into(), serde_json::json!(r.scale));
    params.insert("k".into(), serde_json::json!(r.k));
    params.insert("dim".into(), serde_json::json!(r.dim));
    params.insert("recall".into(), serde_json::json!(r.recall));
    params.insert("build_qps".into(), serde_json::json!(r.build_qps));
    params.insert("queries".into(), serde_json::json!(config.queries));
    params.insert(
        "durability".into(),
        serde_json::json!(config.durability.label()),
    );
    params.insert("metric".into(), serde_json::json!("cosine"));

    let bench_name = if r.engine == "strata" {
        format!("ann/{}/k{}/{}d", scale_label(r.scale), r.k, r.dim)
    } else {
        format!(
            "ann/{}/{}/k{}/{}d",
            r.engine,
            scale_label(r.scale),
            r.k,
            r.dim
        )
    };

    recorder.record(BenchmarkResult {
        benchmark: bench_name,
        category: "ann".to_string(),
        parameters: params,
        metrics: BenchmarkMetrics {
            ops_per_sec: Some(r.search_qps),
            p50_ns: Some(r.p50.as_nanos() as u64),
            p95_ns: Some(r.p95.as_nanos() as u64),
            p99_ns: Some(r.p99.as_nanos() as u64),
            samples: Some(r.latencies.len() as u64),
            ..Default::default()
        },
    });
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Config {
    scales: Vec<usize>,
    dims: Vec<usize>,
    ks: Vec<usize>,
    queries: usize,
    durability: DurabilityConfig,
    csv: bool,
    quiet: bool,
    strata_only: bool,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();

    let quick = args.iter().any(|a| a == "--quick");
    let default_scales = if quick { QUICK_SCALES } else { FULL_SCALES };

    let mut config = Config {
        scales: default_scales.to_vec(),
        dims: DEFAULT_DIMS.to_vec(),
        ks: DEFAULT_KS.to_vec(),
        queries: DEFAULT_QUERIES,
        durability: DurabilityConfig::Cache,
        csv: false,
        quiet: false,
        strata_only: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--scales" => {
                i += 1;
                if i < args.len() {
                    config.scales = args[i]
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                }
            }
            "--dims" => {
                i += 1;
                if i < args.len() {
                    config.dims = args[i]
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                }
            }
            "--ks" => {
                i += 1;
                if i < args.len() {
                    config.ks = args[i]
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                }
            }
            "--queries" => {
                i += 1;
                if i < args.len() {
                    config.queries = args[i].parse().unwrap_or(DEFAULT_QUERIES);
                }
            }
            "--durability" => {
                i += 1;
                if i < args.len() {
                    config.durability = match args[i].as_str() {
                        "cache" => DurabilityConfig::Cache,
                        "standard" => DurabilityConfig::Standard,
                        "always" => DurabilityConfig::Always,
                        _ => DurabilityConfig::Cache,
                    };
                }
            }
            "--csv" => config.csv = true,
            "-q" => config.quiet = true,
            "--strata-only" => config.strata_only = true,
            "--quick" => {} // already handled above
            _ => {}
        }
        i += 1;
    }

    config
}

// ---------------------------------------------------------------------------
// Percentile helper
// ---------------------------------------------------------------------------

fn compute_percentiles(latencies: &mut Vec<Duration>) -> (Duration, Duration, Duration) {
    latencies.sort_unstable();
    let len = latencies.len();
    let p50 = latencies[len * 50 / 100];
    let p95 = latencies[(len * 95 / 100).min(len - 1)];
    let p99 = latencies[(len * 99 / 100).min(len - 1)];
    (p50, p95, p99)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let config = parse_args();
    print_hardware_info();

    if !config.csv && !config.quiet {
        eprintln!("=== StrataDB ANN Benchmark ===");
        eprintln!("Measures Recall@k vs QPS (ann-benchmarks.com methodology)");
        eprintln!();
        eprintln!(
            "Parameters: cosine, {} queries, {} mode",
            config.queries,
            config.durability.label()
        );
        eprintln!("Dimensions: {:?}", config.dims);
        eprintln!("Scales: {:?}", config.scales);
        eprintln!("k values: {:?}", config.ks);
        if config.strata_only {
            eprintln!("Engine: strata only");
        } else {
            eprintln!("Engines: strata, instant-distance");
        }
        eprintln!();
    }

    if config.csv {
        print_csv_header();
    }

    let mut recorder = ResultRecorder::new("ann");
    let max_k = *config.ks.iter().max().unwrap_or(&10);

    for &dim in &config.dims {
        for &scale in &config.scales {
            // Phase 1: Generate dataset
            if !config.csv && !config.quiet {
                eprint!(
                    "  Generating {} vectors ({}d, {} clusters)...",
                    fmt_num(scale as u64),
                    dim,
                    10
                );
            }
            let gen_start = Instant::now();
            let dataset = generate_dataset(scale, config.queries, dim, SEED);
            let gen_elapsed = gen_start.elapsed();
            if !config.csv && !config.quiet {
                eprintln!(" {:.2}s", gen_elapsed.as_secs_f64());
            }

            // Phase 2: Compute brute-force ground truth (at max k)
            if !config.csv && !config.quiet {
                eprint!("  Computing ground truth (brute-force, k={})...", max_k);
            }
            let gt_start = Instant::now();
            let ground_truth = compute_ground_truth(&dataset, max_k);
            let gt_elapsed = gt_start.elapsed();
            if !config.csv && !config.quiet {
                eprintln!(" {:.2}s", gt_elapsed.as_secs_f64());
            }

            // ---- Strata ----
            if !config.csv && !config.quiet {
                eprint!(
                    "  Building Strata index ({} vectors, {}d)...",
                    fmt_num(scale as u64),
                    dim
                );
            }
            let db = create_db(config.durability);
            db.db
                .vector_create_collection("ann_bench", dim as u64, DistanceMetric::Cosine)
                .unwrap();

            let build_start = Instant::now();
            for i in 0..scale {
                db.db
                    .vector_upsert(
                        "ann_bench",
                        &dataset.train_keys[i],
                        dataset.train_vectors[i].clone(),
                        None,
                    )
                    .unwrap();
            }
            let build_elapsed = build_start.elapsed();
            let strata_build_qps = scale as f64 / build_elapsed.as_secs_f64();

            if !config.csv && !config.quiet {
                eprintln!(
                    " {:.2}s ({} inserts/s)",
                    build_elapsed.as_secs_f64(),
                    fmt_num(strata_build_qps as u64)
                );
            }

            // Print scale/dim header
            if !config.csv && !config.quiet {
                eprintln!();
                eprintln!(
                    "--- {} vectors, {}d, cosine ---",
                    fmt_num(scale as u64),
                    dim
                );
                print_table_header();
            }

            // Search Strata for each k
            for &k in &config.ks {
                let gt_k = dataset::GroundTruth {
                    neighbors: ground_truth
                        .neighbors
                        .iter()
                        .map(|nn| nn.iter().take(k).copied().collect())
                        .collect(),
                    k,
                };

                let mut latencies = Vec::with_capacity(config.queries);
                let mut ann_results = Vec::with_capacity(config.queries);

                let search_start = Instant::now();
                for q in 0..config.queries {
                    let query = dataset.query_vectors[q].clone();
                    let op_start = Instant::now();
                    let results = db.db.vector_search("ann_bench", query, k as u64).unwrap();
                    latencies.push(op_start.elapsed());

                    let keys: Vec<String> = results.iter().map(|m| m.key.clone()).collect();
                    ann_results.push(keys);
                }
                let search_elapsed = search_start.elapsed();
                let search_qps = config.queries as f64 / search_elapsed.as_secs_f64();

                let recall = compute_recall(&ann_results, &gt_k, &dataset);
                let (p50, p95, p99) = compute_percentiles(&mut latencies);

                let result = AnnResult {
                    engine: "strata".to_string(),
                    scale,
                    dim,
                    k,
                    build_qps: strata_build_qps,
                    search_qps,
                    recall,
                    latencies,
                    p50,
                    p95,
                    p99,
                };

                if config.csv {
                    print_csv_row(&result);
                } else if config.quiet {
                    print_quiet(&result);
                } else {
                    print_table_row(&result);
                }
                record_result(&mut recorder, &result, &config);
            }

            // ---- instant-distance ----
            if !config.strata_only {
                if !config.csv && !config.quiet {
                    eprint!(
                        "  Building instant-distance index ({} vectors, {}d)...",
                        fmt_num(scale as u64),
                        dim
                    );
                }

                let points: Vec<CosinePoint> = dataset
                    .train_vectors
                    .iter()
                    .map(|v| CosinePoint(v.clone()))
                    .collect();
                let values: Vec<usize> = (0..scale).collect();

                let id_build_start = Instant::now();
                let hnsw = instant_distance::Builder::default().build(points, values);
                let id_build_elapsed = id_build_start.elapsed();
                let id_build_qps = scale as f64 / id_build_elapsed.as_secs_f64();

                if !config.csv && !config.quiet {
                    eprintln!(
                        " {:.2}s ({} inserts/s)",
                        id_build_elapsed.as_secs_f64(),
                        fmt_num(id_build_qps as u64)
                    );
                }

                for &k in &config.ks {
                    let gt_k = dataset::GroundTruth {
                        neighbors: ground_truth
                            .neighbors
                            .iter()
                            .map(|nn| nn.iter().take(k).copied().collect())
                            .collect(),
                        k,
                    };

                    let mut latencies = Vec::with_capacity(config.queries);
                    let mut ann_results = Vec::with_capacity(config.queries);
                    let mut search = instant_distance::Search::default();

                    let search_start = Instant::now();
                    for q in 0..config.queries {
                        let query_point = CosinePoint(dataset.query_vectors[q].clone());
                        let op_start = Instant::now();
                        let results: Vec<String> = hnsw
                            .search(&query_point, &mut search)
                            .take(k)
                            .map(|item| dataset.train_keys[*item.value].clone())
                            .collect();
                        latencies.push(op_start.elapsed());
                        ann_results.push(results);
                    }
                    let search_elapsed = search_start.elapsed();
                    let search_qps = config.queries as f64 / search_elapsed.as_secs_f64();

                    let recall = compute_recall(&ann_results, &gt_k, &dataset);
                    let (p50, p95, p99) = compute_percentiles(&mut latencies);

                    let result = AnnResult {
                        engine: "instant-distance".to_string(),
                        scale,
                        dim,
                        k,
                        build_qps: id_build_qps,
                        search_qps,
                        recall,
                        latencies,
                        p50,
                        p95,
                        p99,
                    };

                    if config.csv {
                        print_csv_row(&result);
                    } else if config.quiet {
                        print_quiet(&result);
                    } else {
                        print_table_row(&result);
                    }
                    record_result(&mut recorder, &result, &config);
                }
            }

            if !config.csv && !config.quiet {
                eprintln!();
            }
        }
    }

    if !config.csv && !config.quiet {
        print_reference_points();
        eprintln!();
        eprintln!("=== ANN benchmark complete ===");
    }
    let _ = recorder.save();
}
