//! Embedding Benchmark for StrataDB
//!
//! Measures embedding throughput (embeddings/sec) and quality (STS-B Spearman
//! correlation) for Strata's embedding subsystem vs fastembed (ONNX runtime).
//! Both engines use all-MiniLM-L6-v2 (384d) for comparison; Strata can also
//! benchmark nomic-embed, bge-m3, and gemma-embed.
//!
//! Run:    `cargo bench --bench embed`
//! Quick:  `cargo bench --bench embed -- --quick -q`
//! Custom: `cargo bench --bench embed -- --models miniLM,nomic-embed --batch-sizes 1,128`
//! Strata only: `cargo bench --bench embed -- --strata-only`
//! No quality:  `cargo bench --bench embed -- --no-quality`

#[allow(unused)]
#[path = "../harness/mod.rs"]
mod harness;

use harness::recorder::ResultRecorder;
use harness::{create_db, print_hardware_info, DurabilityConfig};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use strata_benchmarks::schema::{BenchmarkMetrics, BenchmarkResult};

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const QUICK_SCALES: &[usize] = &[100, 500];
const FULL_SCALES: &[usize] = &[100, 500, 1000, 5000];
const QUICK_BATCH_SIZES: &[usize] = &[1, 128];
const FULL_BATCH_SIZES: &[usize] = &[1, 32, 128, 512];
const QUICK_TEXT_LENGTHS: &[usize] = &[50, 200];
const FULL_TEXT_LENGTHS: &[usize] = &[50, 200, 1000];
const WARMUP_EMBEDS: usize = 10;

const ALL_STRATA_MODELS: &[&str] = &["miniLM", "nomic-embed", "bge-m3", "gemma-embed"];

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Config {
    models: Vec<String>,
    scales: Vec<usize>,
    text_lengths: Vec<usize>,
    batch_sizes: Vec<usize>,
    run_quality: bool,
    strata_only: bool,
    csv: bool,
    quiet: bool,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");

    let mut config = Config {
        models: vec!["miniLM".to_string()],
        scales: if quick {
            QUICK_SCALES.to_vec()
        } else {
            FULL_SCALES.to_vec()
        },
        text_lengths: if quick {
            QUICK_TEXT_LENGTHS.to_vec()
        } else {
            FULL_TEXT_LENGTHS.to_vec()
        },
        batch_sizes: if quick {
            QUICK_BATCH_SIZES.to_vec()
        } else {
            FULL_BATCH_SIZES.to_vec()
        },
        run_quality: true,
        strata_only: false,
        csv: false,
        quiet: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--models" => {
                i += 1;
                if i < args.len() {
                    if args[i] == "all" {
                        config.models =
                            ALL_STRATA_MODELS.iter().map(|s| s.to_string()).collect();
                    } else {
                        config.models = args[i]
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .collect();
                    }
                }
            }
            "--scales" => {
                i += 1;
                if i < args.len() {
                    config.scales = args[i]
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                }
            }
            "--text-lengths" => {
                i += 1;
                if i < args.len() {
                    config.text_lengths = args[i]
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                }
            }
            "--batch-sizes" => {
                i += 1;
                if i < args.len() {
                    config.batch_sizes = args[i]
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                }
            }
            "--no-quality" => config.run_quality = false,
            "--strata-only" => config.strata_only = true,
            "--csv" => config.csv = true,
            "-q" => config.quiet = true,
            "--quick" => {} // already handled
            _ => {}
        }
        i += 1;
    }

    config
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

struct ThroughputResult {
    engine: String,
    model: String,
    text_length: usize,
    batch_size: usize,
    total_embeds: usize,
    embeds_per_sec: f64,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    latencies: Vec<Duration>,
}

struct QualityResult {
    engine: String,
    model: String,
    spearman: f64,
    pairs: usize,
    embed_time: Duration,
}

// ---------------------------------------------------------------------------
// Text generator
// ---------------------------------------------------------------------------

const VOCABULARY: &[&str] = &[
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "a", "large", "dataset", "contains", "many", "records", "that",
    "need", "processing", "with", "efficient", "algorithms", "to",
    "achieve", "optimal", "performance", "in", "real", "world",
    "applications", "machine", "learning", "models", "require",
    "significant", "computational", "resources", "for", "training",
    "and", "inference", "phases", "of", "deep", "neural", "networks",
    "natural", "language", "understanding", "enables", "computers",
    "comprehend", "human", "text", "semantic", "similarity",
    "measures", "how", "closely", "related", "two", "sentences",
    "are", "meaning", "embedding", "vectors", "capture",
    "dense", "representations", "words", "phrases", "documents",
    "transformer", "architecture", "has", "revolutionized", "field",
    "information", "retrieval", "search", "engines", "use", "these",
    "techniques", "provide", "relevant", "results", "users",
    "benchmark", "evaluation", "compares", "different", "systems",
    "accuracy", "speed", "throughput", "latency", "scalability",
    "database", "stores", "structured", "unstructured", "data",
    "indexing", "fast", "lookups", "queries", "across",
    "millions", "billions", "entries", "modern", "recent",
    "cloud", "computing", "provides", "elastic", "infrastructure",
    "distributed", "handles", "scale", "capacity", "volume",
    "workloads", "efficiently", "parallel", "execution", "reduces",
    "time", "complex", "operations", "batch", "pipeline",
    "streaming", "analytics", "continuous", "monitoring",
    "artificial", "intelligence", "cognitive", "automation",
    "knowledge", "graphs", "connect", "entities", "relationships",
    "contextual", "improves", "recommendations",
    "personalized", "experiences", "user", "engagement",
];

fn generate_text(id: u64, word_count: usize) -> String {
    let vocab_len = VOCABULARY.len() as u64;
    let mut result = String::with_capacity(word_count * 7);
    let mut state = id.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x123456789ABCDEF0);

    for i in 0..word_count {
        if i > 0 {
            // Insert punctuation roughly every 8-15 words
            if state % 12 == 0 {
                result.push('.');
                // Capitalize next word
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                result.push(' ');
                let idx = (state % vocab_len) as usize;
                let word = VOCABULARY[idx];
                let mut chars = word.chars();
                if let Some(first) = chars.next() {
                    result.extend(first.to_uppercase());
                    result.push_str(chars.as_str());
                }
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                continue;
            } else if state % 20 == 0 {
                result.push(',');
            }
            result.push(' ');
        }
        let idx = (state % vocab_len) as usize;
        result.push_str(VOCABULARY[idx]);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    }
    result.push('.');
    result
}

// ---------------------------------------------------------------------------
// STS-B loader
// ---------------------------------------------------------------------------

struct StsbPair {
    sentence1: String,
    sentence2: String,
    score: f64,
}

fn load_stsb() -> Option<Vec<StsbPair>> {
    let path = "data/stsb/sts-test.tsv";
    let contents = std::fs::read_to_string(path).ok()?;
    let mut pairs = Vec::new();
    for line in contents.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 3 {
            if let Ok(score) = parts[2].parse::<f64>() {
                pairs.push(StsbPair {
                    sentence1: parts[0].to_string(),
                    sentence2: parts[1].to_string(),
                    score,
                });
            }
        }
    }
    if pairs.is_empty() {
        None
    } else {
        Some(pairs)
    }
}

// ---------------------------------------------------------------------------
// Spearman correlation
// ---------------------------------------------------------------------------

fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 2 {
        return 0.0;
    }

    let rank_x = compute_ranks(x);
    let rank_y = compute_ranks(y);

    pearson_correlation(&rank_x, &rank_y)
}

fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n - 1 && indexed[j + 1].1 == indexed[j].1 {
            j += 1;
        }
        // Average rank for ties
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j + 1;
    }
    ranks
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        cov / denom
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y as f64).sum();
    let norm_a: f64 = a.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
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

fn print_throughput_header() {
    eprintln!(
        "  {:>12}  {:>12}  {:>6}  {:>6}  {:>10}  {:>10}  {:>10}  {:>10}",
        "engine", "model", "text", "batch", "embed/s", "p50", "p95", "p99"
    );
}

fn print_throughput_row(r: &ThroughputResult) {
    eprintln!(
        "  {:>12}  {:>12}  {:>5}w  {:>6}  {:>10}  {:>10}  {:>10}  {:>10}",
        r.engine,
        r.model,
        r.text_length,
        r.batch_size,
        fmt_num(r.embeds_per_sec as u64),
        fmt_duration(r.p50),
        fmt_duration(r.p95),
        fmt_duration(r.p99),
    );
}

fn print_throughput_quiet(r: &ThroughputResult) {
    eprintln!(
        "{} {} {}w batch{}: {}/s p50={}",
        r.engine,
        r.model,
        r.text_length,
        r.batch_size,
        fmt_num(r.embeds_per_sec as u64),
        fmt_duration(r.p50),
    );
}

fn print_quality_header() {
    eprintln!(
        "  {:>12}  {:>12}  {:>10}  {:>6}  {:>10}",
        "engine", "model", "spearman", "pairs", "embed_time"
    );
}

fn print_quality_row(r: &QualityResult) {
    eprintln!(
        "  {:>12}  {:>12}  {:>10.4}  {:>6}  {:>10.2}s",
        r.engine,
        r.model,
        r.spearman,
        r.pairs,
        r.embed_time.as_secs_f64(),
    );
}

fn print_quality_quiet(r: &QualityResult) {
    eprintln!(
        "{} {} quality: spearman={:.4} ({} pairs, {:.2}s)",
        r.engine,
        r.model,
        r.spearman,
        r.pairs,
        r.embed_time.as_secs_f64(),
    );
}

fn print_csv_throughput_header() {
    println!(
        "\"type\",\"engine\",\"model\",\"text_length\",\"batch_size\",\"embeds_per_sec\",\"p50_us\",\"p95_us\",\"p99_us\""
    );
}

fn print_csv_throughput_row(r: &ThroughputResult) {
    println!(
        "\"throughput\",\"{}\",\"{}\",{},{},{:.2},{:.1},{:.1},{:.1}",
        r.engine,
        r.model,
        r.text_length,
        r.batch_size,
        r.embeds_per_sec,
        r.p50.as_nanos() as f64 / 1_000.0,
        r.p95.as_nanos() as f64 / 1_000.0,
        r.p99.as_nanos() as f64 / 1_000.0,
    );
}

fn print_csv_quality_header() {
    println!(
        "\"type\",\"engine\",\"model\",\"spearman\",\"pairs\",\"embed_time_ms\""
    );
}

fn print_csv_quality_row(r: &QualityResult) {
    println!(
        "\"quality\",\"{}\",\"{}\",{:.6},{},{:.1}",
        r.engine,
        r.model,
        r.spearman,
        r.pairs,
        r.embed_time.as_millis(),
    );
}

// ---------------------------------------------------------------------------
// Percentile helper
// ---------------------------------------------------------------------------

fn compute_percentiles(latencies: &mut Vec<Duration>) -> (Duration, Duration, Duration) {
    if latencies.is_empty() {
        return (Duration::ZERO, Duration::ZERO, Duration::ZERO);
    }
    latencies.sort_unstable();
    let len = latencies.len();
    let p50 = latencies[len * 50 / 100];
    let p95 = latencies[(len * 95 / 100).min(len - 1)];
    let p99 = latencies[(len * 99 / 100).min(len - 1)];
    (p50, p95, p99)
}

// ---------------------------------------------------------------------------
// JSON recording
// ---------------------------------------------------------------------------

fn scale_label(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}m", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}k", n / 1_000)
    } else {
        format!("{}", n)
    }
}

fn record_throughput(recorder: &mut ResultRecorder, r: &ThroughputResult) {
    let mut params = HashMap::new();
    params.insert("engine".into(), serde_json::json!(r.engine));
    params.insert("model".into(), serde_json::json!(r.model));
    params.insert("text_length".into(), serde_json::json!(r.text_length));
    params.insert("batch_size".into(), serde_json::json!(r.batch_size));
    params.insert("total_embeds".into(), serde_json::json!(r.total_embeds));

    let bench_name = format!(
        "embed/{}/{}/{}/{}w/batch{}",
        r.engine, r.model, scale_label(r.total_embeds), r.text_length, r.batch_size
    );

    recorder.record(BenchmarkResult {
        benchmark: bench_name,
        category: "embed".to_string(),
        parameters: params,
        metrics: BenchmarkMetrics {
            ops_per_sec: Some(r.embeds_per_sec),
            p50_ns: Some(r.p50.as_nanos() as u64),
            p95_ns: Some(r.p95.as_nanos() as u64),
            p99_ns: Some(r.p99.as_nanos() as u64),
            samples: Some(r.latencies.len() as u64),
            ..Default::default()
        },
    });
}

fn record_quality(recorder: &mut ResultRecorder, r: &QualityResult) {
    let mut params = HashMap::new();
    params.insert("engine".into(), serde_json::json!(r.engine));
    params.insert("model".into(), serde_json::json!(r.model));
    params.insert("spearman".into(), serde_json::json!(r.spearman));
    params.insert("pairs".into(), serde_json::json!(r.pairs));
    params.insert(
        "embed_time_ms".into(),
        serde_json::json!(r.embed_time.as_millis()),
    );

    let bench_name = format!("embed/quality/{}/{}", r.engine, r.model);

    recorder.record(BenchmarkResult {
        benchmark: bench_name,
        category: "embed".to_string(),
        parameters: params,
        metrics: BenchmarkMetrics {
            ..Default::default()
        },
    });
}

// ---------------------------------------------------------------------------
// Strata embedding helpers
// ---------------------------------------------------------------------------

fn strata_embed(db: &stratadb::Strata, text: &str) -> Result<Vec<f32>, String> {
    db.embed(text).map_err(|e| format!("{}", e))
}

fn strata_embed_batch(db: &stratadb::Strata, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
    db.embed_batch(texts).map_err(|e| format!("{}", e))
}

/// Try a single embed to check that the Strata embedding model is available.
/// Returns false (with reason) if libllama.so or the model can't be loaded.
fn strata_embed_check(db: &stratadb::Strata) -> Result<(), String> {
    strata_embed(db, "test").map(|_| ())
}

// ---------------------------------------------------------------------------
// fastembed helpers
// ---------------------------------------------------------------------------

fn fastembed_embed_batch(model: &mut TextEmbedding, texts: &[String]) -> Vec<Vec<f32>> {
    let docs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    model
        .embed(docs, None)
        .expect("fastembed embed failed")
}

// ---------------------------------------------------------------------------
// Throughput benchmark
// ---------------------------------------------------------------------------

fn run_throughput(config: &Config, recorder: &mut ResultRecorder) {
    // Pre-check Strata embed library availability (try miniLM as a known baseline)
    let strata_available = {
        let db = create_db(DurabilityConfig::Cache);
        let _ = db.db.config_set("embed_model", "miniLM");
        match strata_embed_check(&db.db) {
            Ok(()) => true,
            Err(e) => {
                if !config.quiet {
                    eprintln!("  WARNING: Strata embedding unavailable: {}", e);
                    eprintln!("  Skipping Strata throughput.");
                    eprintln!();
                }
                false
            }
        }
    };

    // Pre-create fastembed model once (expensive init)
    let mut fe_model = if !config.strata_only {
        match TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                .with_show_download_progress(!config.quiet)
        ) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("  WARNING: Cannot create fastembed model: {}", e);
                None
            }
        }
    } else {
        None
    };

    if !strata_available && fe_model.is_none() {
        eprintln!("  WARNING: No embedding engine available. Nothing to benchmark.");
        return;
    }

    for &text_length in &config.text_lengths {
        // Generate texts up to the max scale
        let max_scale = *config.scales.iter().max().unwrap_or(&100);
        let texts: Vec<String> = (0..max_scale as u64)
            .map(|id| generate_text(id, text_length))
            .collect();

        for &batch_size in &config.batch_sizes {
            // --- Strata ---
            if strata_available {
                for model_name in &config.models {
                    // Check if this specific model is available
                    let db = create_db(DurabilityConfig::Cache);
                    if let Err(e) = db.db.config_set("embed_model", model_name) {
                        eprintln!("  Skipping strata model '{}': {}", model_name, e);
                        continue;
                    }
                    if let Err(e) = strata_embed_check(&db.db) {
                        eprintln!("  Skipping strata model '{}': {}", model_name, e);
                        continue;
                    }
                    drop(db);

                    for &scale in &config.scales {
                        let n_texts = scale.min(texts.len());
                        let bench_texts = &texts[..n_texts];

                        let db = create_db(DurabilityConfig::Cache);
                        db.db
                            .config_set("embed_model", model_name)
                            .expect("failed to set embed_model");

                        // Warm up
                        for i in 0..WARMUP_EMBEDS.min(n_texts) {
                            let _ = strata_embed(&db.db, &bench_texts[i]);
                        }

                        // Measure: embed in batches, record per-batch latency
                        let mut latencies = Vec::new();
                        let total_start = Instant::now();

                        let mut offset = 0;
                        while offset < n_texts {
                            let end = (offset + batch_size).min(n_texts);
                            let batch: Vec<&str> =
                                bench_texts[offset..end].iter().map(|s| s.as_str()).collect();

                            let batch_start = Instant::now();
                            if batch.len() == 1 {
                                let _ = strata_embed(&db.db, batch[0]);
                            } else {
                                let _ = strata_embed_batch(&db.db, &batch);
                            }
                            latencies.push(batch_start.elapsed());
                            offset = end;
                        }

                        let total_elapsed = total_start.elapsed();
                        let embeds_per_sec = n_texts as f64 / total_elapsed.as_secs_f64();
                        let (p50, p95, p99) = compute_percentiles(&mut latencies);

                        let result = ThroughputResult {
                            engine: "strata".to_string(),
                            model: model_name.clone(),
                            text_length,
                            batch_size,
                            total_embeds: n_texts,
                            embeds_per_sec,
                            p50,
                            p95,
                            p99,
                            latencies,
                        };

                        if config.csv {
                            print_csv_throughput_row(&result);
                        } else if config.quiet {
                            print_throughput_quiet(&result);
                        } else {
                            print_throughput_row(&result);
                        }
                        record_throughput(recorder, &result);
                    }
                }
            }

            // --- fastembed (miniLM only) ---
            if let Some(ref mut model) = fe_model {
                // Warm up once per (text_length, batch_size) combination
                {
                    let warmup_n = WARMUP_EMBEDS.min(texts.len());
                    let warmup_texts: Vec<String> = texts[..warmup_n].to_vec();
                    let _ = fastembed_embed_batch(model, &warmup_texts);
                }

                for &scale in &config.scales {
                    let n_texts = scale.min(texts.len());
                    let bench_texts = &texts[..n_texts];

                    // Measure
                    let mut latencies = Vec::new();
                    let total_start = Instant::now();

                    let mut offset = 0;
                    while offset < n_texts {
                        let end = (offset + batch_size).min(n_texts);
                        let batch: Vec<String> = bench_texts[offset..end].to_vec();

                        let batch_start = Instant::now();
                        let _ = fastembed_embed_batch(model, &batch);
                        latencies.push(batch_start.elapsed());
                        offset = end;
                    }

                    let total_elapsed = total_start.elapsed();
                    let embeds_per_sec = n_texts as f64 / total_elapsed.as_secs_f64();
                    let (p50, p95, p99) = compute_percentiles(&mut latencies);

                    let result = ThroughputResult {
                        engine: "fastembed".to_string(),
                        model: "miniLM".to_string(),
                        text_length,
                        batch_size,
                        total_embeds: n_texts,
                        embeds_per_sec,
                        p50,
                        p95,
                        p99,
                        latencies,
                    };

                    if config.csv {
                        print_csv_throughput_row(&result);
                    } else if config.quiet {
                        print_throughput_quiet(&result);
                    } else {
                        print_throughput_row(&result);
                    }
                    record_throughput(recorder, &result);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Quality benchmark (STS-B)
// ---------------------------------------------------------------------------

fn run_quality(config: &Config, recorder: &mut ResultRecorder) {
    let pairs = match load_stsb() {
        Some(p) => p,
        None => {
            eprintln!("  WARNING: STS-B dataset not found at data/stsb/sts-test.tsv");
            eprintln!("  Run: ./scripts/download-stsb.sh");
            return;
        }
    };

    if !config.csv && !config.quiet {
        eprintln!();
        eprintln!("--- Quality Benchmark (STS-B, {} pairs) ---", pairs.len());
        print_quality_header();
    }
    if config.csv {
        print_csv_quality_header();
    }

    let human_scores: Vec<f64> = pairs.iter().map(|p| p.score).collect();

    // Collect unique sentences
    let mut unique_sentences: Vec<String> = Vec::new();
    let mut sentence_index: HashMap<String, usize> = HashMap::new();
    for pair in &pairs {
        if !sentence_index.contains_key(&pair.sentence1) {
            sentence_index.insert(pair.sentence1.clone(), unique_sentences.len());
            unique_sentences.push(pair.sentence1.clone());
        }
        if !sentence_index.contains_key(&pair.sentence2) {
            sentence_index.insert(pair.sentence2.clone(), unique_sentences.len());
            unique_sentences.push(pair.sentence2.clone());
        }
    }

    // --- Strata quality ---
    // Check Strata embed library availability (try miniLM as a known baseline)
    let strata_available = {
        let db = create_db(DurabilityConfig::Cache);
        let _ = db.db.config_set("embed_model", "miniLM");
        strata_embed_check(&db.db).is_ok()
    };

    if strata_available {
        for model_name in &config.models {
            let db = create_db(DurabilityConfig::Cache);
            if let Err(e) = db.db.config_set("embed_model", model_name) {
                eprintln!("  Skipping strata quality for '{}': {}", model_name, e);
                continue;
            }
            if let Err(e) = strata_embed_check(&db.db) {
                eprintln!("  Skipping strata quality for '{}': {}", model_name, e);
                continue;
            }

            let embed_start = Instant::now();
            let refs: Vec<&str> = unique_sentences.iter().map(|s| s.as_str()).collect();
            let embeddings = match strata_embed_batch(&db.db, &refs) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("  Strata quality for '{}' failed: {}", model_name, e);
                    continue;
                }
            };
            let embed_time = embed_start.elapsed();

            // Compute cosine similarities
            let mut predicted: Vec<f64> = Vec::with_capacity(pairs.len());
            for pair in &pairs {
                let idx1 = sentence_index[&pair.sentence1];
                let idx2 = sentence_index[&pair.sentence2];
                predicted.push(cosine_similarity(&embeddings[idx1], &embeddings[idx2]));
            }

            let spearman = spearman_correlation(&predicted, &human_scores);

            let result = QualityResult {
                engine: "strata".to_string(),
                model: model_name.clone(),
                spearman,
                pairs: pairs.len(),
                embed_time,
            };

            if config.csv {
                print_csv_quality_row(&result);
            } else if config.quiet {
                print_quality_quiet(&result);
            } else {
                print_quality_row(&result);
            }
            record_quality(recorder, &result);
        }
    } else {
        eprintln!("  Skipping Strata quality (embedding model unavailable).");
    }

    // --- fastembed quality ---
    if !config.strata_only {
        let mut fe_model = match TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                .with_show_download_progress(!config.quiet)
        ) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("  WARNING: Cannot create fastembed model: {}", e);
                return;
            }
        };

        let embed_start = Instant::now();
        let embeddings = fastembed_embed_batch(&mut fe_model, &unique_sentences);
        let embed_time = embed_start.elapsed();

        let mut predicted: Vec<f64> = Vec::with_capacity(pairs.len());
        for pair in &pairs {
            let idx1 = sentence_index[&pair.sentence1];
            let idx2 = sentence_index[&pair.sentence2];
            predicted.push(cosine_similarity(&embeddings[idx1], &embeddings[idx2]));
        }

        let spearman = spearman_correlation(&predicted, &human_scores);

        let result = QualityResult {
            engine: "fastembed".to_string(),
            model: "miniLM".to_string(),
            spearman,
            pairs: pairs.len(),
            embed_time,
        };

        if config.csv {
            print_csv_quality_row(&result);
        } else if config.quiet {
            print_quality_quiet(&result);
        } else {
            print_quality_row(&result);
        }
        record_quality(recorder, &result);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let config = parse_args();
    print_hardware_info();

    if !config.csv && !config.quiet {
        eprintln!("=== StrataDB Embedding Benchmark ===");
        eprintln!("Measures throughput (embed/s) and quality (STS-B Spearman)");
        eprintln!();
        eprintln!("Models: {:?}", config.models);
        eprintln!("Text lengths: {:?}", config.text_lengths);
        eprintln!("Batch sizes: {:?}", config.batch_sizes);
        eprintln!("Scales: {:?}", config.scales);
        if config.strata_only {
            eprintln!("Engine: strata only");
        } else {
            eprintln!("Engines: strata, fastembed");
        }
        eprintln!("Quality (STS-B): {}", if config.run_quality { "yes" } else { "no" });
        eprintln!();
    }

    let mut recorder = ResultRecorder::new("embed");

    // Part A: Throughput
    if !config.csv && !config.quiet {
        eprintln!("--- Throughput Benchmark ---");
        print_throughput_header();
    }
    if config.csv {
        print_csv_throughput_header();
    }

    run_throughput(&config, &mut recorder);

    // Part B: Quality (STS-B)
    if config.run_quality {
        run_quality(&config, &mut recorder);
    }

    if !config.csv && !config.quiet {
        eprintln!();
        eprintln!("=== Embedding benchmark complete ===");
    }
    let _ = recorder.save();
}
