//! LDBC Graphalytics BFS Benchmark — Strata vs petgraph head-to-head
//!
//! Validates BFS traversal correctness and measures throughput via EVPS
//! (Edges + Vertices processed per second). Compares Strata against petgraph
//! (pure in-memory adjacency list) on the same dataset, same machine.
//!
//! Uses a custom harness (matching fill-level and redis-compare patterns) since BFS
//! is a whole-graph operation, not per-operation latency.
//!
//! Run:           `cargo bench --bench graph_bfs`
//! Quick:         `cargo bench --bench graph_bfs -- -q`
//! Validate only: `cargo bench --bench graph_bfs -- --validate-only`
//! CSV:           `cargo bench --bench graph_bfs -- --csv`
//! Custom data:   `cargo bench --bench graph_bfs -- --dataset path/to/ldbc/dir`
//! Strata only:   `cargo bench --bench graph_bfs -- --strata-only`

#[allow(unused)]
#[path = "../harness/mod.rs"]
mod harness;

#[allow(unused)]
mod ldbc;

use harness::recorder::ResultRecorder;
use harness::{create_db, print_hardware_info, BenchDb, DurabilityConfig};
use ldbc::{petgraph_bfs, BfsReference, LdbcDataset, UNREACHABLE};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use strata_benchmarks::schema::{BenchmarkMetrics, BenchmarkResult};

// ---------------------------------------------------------------------------
// Default parameters
// ---------------------------------------------------------------------------

const DEFAULT_RUNS: usize = 10;

fn default_dataset_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/graph/example-directed")
}

// ---------------------------------------------------------------------------
// CLI configuration
// ---------------------------------------------------------------------------

struct Config {
    dataset: PathBuf,
    source: Option<u64>,
    runs: usize,
    validate_only: bool,
    no_validate: bool,
    csv: bool,
    quiet: bool,
    strata_only: bool,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut config = Config {
        dataset: default_dataset_dir(),
        source: None,
        runs: DEFAULT_RUNS,
        validate_only: false,
        no_validate: false,
        csv: false,
        quiet: false,
        strata_only: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dataset" => {
                i += 1;
                if i < args.len() {
                    config.dataset = PathBuf::from(&args[i]);
                }
            }
            "--source" => {
                i += 1;
                if i < args.len() {
                    config.source = args[i].parse().ok();
                }
            }
            "--runs" => {
                i += 1;
                if i < args.len() {
                    config.runs = args[i].parse::<usize>().unwrap_or(DEFAULT_RUNS).max(1);
                }
            }
            "--validate-only" => config.validate_only = true,
            "--no-validate" => config.no_validate = true,
            "--csv" => config.csv = true,
            "-q" => config.quiet = true,
            "--strata-only" => config.strata_only = true,
            _ => {}
        }
        i += 1;
    }

    config
}

// ---------------------------------------------------------------------------
// Graph loading (Strata)
// ---------------------------------------------------------------------------

fn load_graph(db: &BenchDb, dataset: &LdbcDataset) -> std::time::Duration {
    ldbc::load_graph_into_strata(&db.db, dataset)
}

// ---------------------------------------------------------------------------
// BFS execution (Strata)
// ---------------------------------------------------------------------------

struct BfsRun {
    elapsed: std::time::Duration,
    depths: HashMap<String, usize>,
}

fn run_bfs(db: &BenchDb, source: u64) -> BfsRun {
    let start = Instant::now();
    let result = db
        .db
        .graph_bfs(
            "ldbc",
            &source.to_string(),
            usize::MAX,
            None,
            None,
            Some("both"),
        )
        .expect("graph_bfs failed");
    let elapsed = start.elapsed();

    BfsRun {
        elapsed,
        depths: result.depths,
    }
}

// ---------------------------------------------------------------------------
// Validation against LDBC reference
// ---------------------------------------------------------------------------

struct ValidationResult {
    pass: bool,
    mismatches: usize,
    details: Vec<String>,
}

fn validate_bfs(
    dataset: &LdbcDataset,
    bfs_depths: &HashMap<String, usize>,
    reference: &BfsReference,
) -> ValidationResult {
    let mut mismatches = 0;
    let mut details = Vec::new();

    for &vid in &dataset.vertices {
        let ref_depth = reference.depths.get(&vid).copied().unwrap_or(UNREACHABLE);
        let actual_depth = bfs_depths.get(&vid.to_string()).copied();

        if ref_depth == UNREACHABLE {
            if let Some(actual) = actual_depth {
                mismatches += 1;
                if details.len() < 10 {
                    details.push(format!(
                        "vertex {}: expected unreachable, got depth {}",
                        vid, actual
                    ));
                }
            }
        } else {
            match actual_depth {
                None => {
                    mismatches += 1;
                    if details.len() < 10 {
                        details.push(format!(
                            "vertex {}: expected depth {}, but not visited",
                            vid, ref_depth
                        ));
                    }
                }
                Some(actual) if actual as i64 != ref_depth => {
                    mismatches += 1;
                    if details.len() < 10 {
                        details.push(format!(
                            "vertex {}: expected depth {}, got {}",
                            vid, ref_depth, actual
                        ));
                    }
                }
                _ => {}
            }
        }
    }

    ValidationResult {
        pass: mismatches == 0,
        mismatches,
        details,
    }
}

// ---------------------------------------------------------------------------
// Cross-validation: Strata vs petgraph
// ---------------------------------------------------------------------------

/// Compare Strata BFS depths against petgraph BFS depths.
/// Returns (pass, num_vertices_checked, mismatches).
fn cross_validate(
    dataset: &LdbcDataset,
    strata_depths: &HashMap<String, usize>,
    petgraph_depths: &HashMap<petgraph::graph::NodeIndex, usize>,
    id_map: &HashMap<u64, petgraph::graph::NodeIndex>,
) -> (bool, usize, usize) {
    let mut mismatches = 0;
    let mut checked = 0;

    for &vid in &dataset.vertices {
        checked += 1;
        let strata_d = strata_depths.get(&vid.to_string()).copied();
        let pg_d = id_map
            .get(&vid)
            .and_then(|idx| petgraph_depths.get(idx).copied());

        match (strata_d, pg_d) {
            (Some(a), Some(b)) if a != b => mismatches += 1,
            (None, Some(_)) | (Some(_), None) => mismatches += 1,
            _ => {} // both None (unreachable) or both equal
        }
    }

    (mismatches == 0, checked, mismatches)
}

// ---------------------------------------------------------------------------
// Run statistics
// ---------------------------------------------------------------------------

struct RunStats {
    avg: std::time::Duration,
    p50: std::time::Duration,
    p95: std::time::Duration,
    p99: std::time::Duration,
    min: std::time::Duration,
    max: std::time::Duration,
    avg_evps: f64,
    count: usize,
}

fn compute_stats(times: &mut Vec<std::time::Duration>, total_elements: f64) -> RunStats {
    assert!(!times.is_empty(), "compute_stats requires at least one run");
    times.sort_unstable();
    let len = times.len();
    let sum: std::time::Duration = times.iter().sum();
    let avg = sum / len as u32;
    let avg_secs = avg.as_secs_f64();
    let avg_evps = if avg_secs > 0.0 {
        total_elements / avg_secs
    } else {
        0.0
    };
    RunStats {
        avg,
        p50: times[len * 50 / 100],
        p95: times[(len * 95 / 100).min(len - 1)],
        p99: times[(len * 99 / 100).min(len - 1)],
        min: times[0],
        max: times[len - 1],
        avg_evps,
        count: len,
    }
}

// ---------------------------------------------------------------------------
// Output formatters
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

fn fmt_ms(d: std::time::Duration) -> String {
    format!("{:.1}ms", d.as_secs_f64() * 1000.0)
}

fn print_csv_header() {
    println!("\"engine\",\"run\",\"bfs_time_ms\",\"evps\",\"vertices\",\"edges\"");
}

fn print_csv_row(engine: &str, run: usize, bfs_ms: f64, evps: f64, vertices: usize, edges: usize) {
    println!("\"{}\",{},{:.3},{:.0},{},{}", engine, run, bfs_ms, evps, vertices, edges);
}

fn print_published_references(strata_evps: f64, dataset_name: &str) {
    eprintln!();
    eprintln!("--- Published Reference Points (different hardware, for context) ---");
    eprintln!("  {:30} {:>14}  {}", "System", "~EVPS", "Notes");
    eprintln!("  {:30} {:>14}  {}", "------", "-----", "-----");
    eprintln!(
        "  {:30} {:>14}  {}",
        "GraphBLAS/SuiteSparse", "~7,000,000,000", "128-core server, 4.3B edges"
    );
    eprintln!(
        "  {:30} {:>14}  {}",
        "Oracle PGX.D", "~500,000,000", "graph500-22, 16-core server"
    );
    eprintln!(
        "  {:30} {:>14}  {}",
        "Neo4j", "~2,000,000", "estimated, single machine"
    );
    eprintln!(
        "  {:30} {:>14}  {}",
        "Strata (this run)",
        fmt_num(strata_evps as u64),
        format!("{}, this machine", dataset_name),
    );
}

// ---------------------------------------------------------------------------
// Markdown report generation
// ---------------------------------------------------------------------------

fn write_markdown_report(
    json_path: &std::path::Path,
    dataset: &LdbcDataset,
    source: u64,
    runs: usize,
    strata_only: bool,
    strata_load_time: std::time::Duration,
    petgraph_load_time: Option<std::time::Duration>,
    strata_stats: &RunStats,
    petgraph_stats: Option<&RunStats>,
    ldbc_validation: Option<bool>,
    cross_validation: Option<bool>,
) -> std::io::Result<PathBuf> {
    let json_name = json_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    let md_name = json_name
        .replace("graph-bfs-", "graph-bfs-report-")
        .replace(".json", ".md");
    let md_path = json_path.with_file_name(md_name);

    let cpu = harness::read_cpu_model();
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let ram_gb = harness::read_total_ram_gb();
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    let mut md = String::new();

    // Title
    md.push_str("# BFS Benchmark Baseline Report\n\n");

    // Hardware
    md.push_str("## Hardware\n\n");
    md.push_str("| Field | Value |\n");
    md.push_str("|-------|-------|\n");
    md.push_str(&format!("| CPU | {} |\n", cpu));
    md.push_str(&format!("| Cores | {} |\n", cores));
    md.push_str(&format!("| RAM | {} GB |\n", ram_gb));
    md.push_str(&format!("| OS | {} ({}) |\n", os, arch));
    md.push_str("\n");

    // Dataset
    md.push_str("## Dataset\n\n");
    md.push_str("| Field | Value |\n");
    md.push_str("|-------|-------|\n");
    md.push_str(&format!("| Name | {} |\n", dataset.name));
    md.push_str(&format!(
        "| Vertices | {} |\n",
        fmt_num(dataset.vertices.len() as u64)
    ));
    md.push_str(&format!(
        "| Edges | {} |\n",
        fmt_num(dataset.edges.len() as u64)
    ));
    md.push_str(&format!(
        "| Type | {} |\n",
        if dataset.directed {
            "directed"
        } else {
            "undirected"
        }
    ));
    md.push_str(&format!("| BFS Source | {} |\n", source));
    md.push_str("\n");

    // Configuration
    md.push_str("## Configuration\n\n");
    md.push_str("| Field | Value |\n");
    md.push_str("|-------|-------|\n");
    md.push_str(&format!("| Runs | {} |\n", runs));
    md.push_str("| Direction | both |\n");
    if strata_only {
        md.push_str("| Engine(s) | Strata |\n");
    } else {
        md.push_str("| Engine(s) | Strata, petgraph |\n");
    }
    md.push_str("\n");

    // Load Phase
    md.push_str("## Load Phase\n\n");
    if strata_only {
        md.push_str("| Engine | Load Time |\n");
        md.push_str("|--------|-----------|\n");
        md.push_str(&format!("| Strata | {} |\n", fmt_ms(strata_load_time)));
    } else {
        md.push_str("| Engine | Load Time | Ratio |\n");
        md.push_str("|--------|-----------|-------|\n");
        md.push_str(&format!(
            "| Strata | {} | — |\n",
            fmt_ms(strata_load_time)
        ));
        if let Some(pg_load) = petgraph_load_time {
            let ratio = if pg_load.as_secs_f64() > 0.0 {
                format!(
                    "{:.2}x",
                    strata_load_time.as_secs_f64() / pg_load.as_secs_f64()
                )
            } else {
                "—".to_string()
            };
            md.push_str(&format!(
                "| petgraph | {} | {} |\n",
                fmt_ms(pg_load),
                ratio
            ));
        }
    }
    md.push_str("\n");

    // BFS Phase
    md.push_str(&format!("## BFS Phase ({} runs)\n\n", runs));
    if strata_only {
        md.push_str("| Metric | Strata |\n");
        md.push_str("|--------|--------|\n");
        md.push_str(&format!("| avg | {} |\n", fmt_ms(strata_stats.avg)));
        md.push_str(&format!("| p50 | {} |\n", fmt_ms(strata_stats.p50)));
        md.push_str(&format!("| p95 | {} |\n", fmt_ms(strata_stats.p95)));
        md.push_str(&format!("| p99 | {} |\n", fmt_ms(strata_stats.p99)));
        md.push_str(&format!("| min | {} |\n", fmt_ms(strata_stats.min)));
        md.push_str(&format!("| max | {} |\n", fmt_ms(strata_stats.max)));
        md.push_str(&format!(
            "| EVPS | {} |\n",
            fmt_num(strata_stats.avg_evps as u64)
        ));
    } else if let Some(ref pg) = petgraph_stats {
        let ratio =
            |s: f64, p: f64| -> String {
                if p > 0.0 {
                    format!("{:.1}x", s / p)
                } else {
                    "—".to_string()
                }
            };
        md.push_str("| Metric | Strata | petgraph | Ratio |\n");
        md.push_str("|--------|--------|----------|-------|\n");
        md.push_str(&format!(
            "| avg | {} | {} | {} |\n",
            fmt_ms(strata_stats.avg),
            fmt_ms(pg.avg),
            ratio(strata_stats.avg.as_secs_f64(), pg.avg.as_secs_f64())
        ));
        md.push_str(&format!(
            "| p50 | {} | {} | {} |\n",
            fmt_ms(strata_stats.p50),
            fmt_ms(pg.p50),
            ratio(strata_stats.p50.as_secs_f64(), pg.p50.as_secs_f64())
        ));
        md.push_str(&format!(
            "| p95 | {} | {} | {} |\n",
            fmt_ms(strata_stats.p95),
            fmt_ms(pg.p95),
            ratio(strata_stats.p95.as_secs_f64(), pg.p95.as_secs_f64())
        ));
        md.push_str(&format!(
            "| p99 | {} | {} | {} |\n",
            fmt_ms(strata_stats.p99),
            fmt_ms(pg.p99),
            ratio(strata_stats.p99.as_secs_f64(), pg.p99.as_secs_f64())
        ));
        md.push_str(&format!(
            "| min | {} | {} | {} |\n",
            fmt_ms(strata_stats.min),
            fmt_ms(pg.min),
            ratio(strata_stats.min.as_secs_f64(), pg.min.as_secs_f64())
        ));
        md.push_str(&format!(
            "| max | {} | {} | {} |\n",
            fmt_ms(strata_stats.max),
            fmt_ms(pg.max),
            ratio(strata_stats.max.as_secs_f64(), pg.max.as_secs_f64())
        ));
        let evps_ratio = if strata_stats.avg_evps > 0.0 {
            format!("{:.1}x", pg.avg_evps / strata_stats.avg_evps)
        } else {
            "—".to_string()
        };
        md.push_str(&format!(
            "| EVPS | {} | {} | {} |\n",
            fmt_num(strata_stats.avg_evps as u64),
            fmt_num(pg.avg_evps as u64),
            evps_ratio,
        ));
    }
    md.push_str("\n");

    // Validation
    md.push_str("## Validation\n\n");
    md.push_str("| Check | Result |\n");
    md.push_str("|-------|--------|\n");
    match ldbc_validation {
        Some(true) => md.push_str("| LDBC Reference | PASS |\n"),
        Some(false) => md.push_str("| LDBC Reference | **FAIL** |\n"),
        None => md.push_str("| LDBC Reference | skipped |\n"),
    }
    match cross_validation {
        Some(true) => md.push_str("| Cross-validation (Strata vs petgraph) | PASS |\n"),
        Some(false) => md.push_str("| Cross-validation (Strata vs petgraph) | **FAIL** |\n"),
        None => md.push_str("| Cross-validation | skipped |\n"),
    }
    md.push_str("\n");

    // Published References
    md.push_str("## Published References\n\n");
    md.push_str("| System | ~EVPS | Notes |\n");
    md.push_str("|--------|-------|-------|\n");
    md.push_str("| GraphBLAS/SuiteSparse | ~7,000,000,000 | 128-core server, 4.3B edges |\n");
    md.push_str("| Oracle PGX.D | ~500,000,000 | graph500-22, 16-core server |\n");
    md.push_str("| Neo4j | ~2,000,000 | estimated, single machine |\n");
    md.push_str(&format!(
        "| Strata (this run) | {} | {}, this machine |\n",
        fmt_num(strata_stats.avg_evps as u64),
        dataset.name,
    ));
    md.push_str("\n");

    // Raw JSON pointer
    md.push_str("## Raw Data\n\n");
    md.push_str(&format!(
        "Machine-readable results: `{}`\n",
        json_path.file_name().unwrap_or_default().to_string_lossy()
    ));

    std::fs::write(&md_path, &md)?;
    eprintln!("Markdown report saved to {}", md_path.display());
    Ok(md_path)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // Init tracing subscriber so strata-engine profiling logs are visible.
    tracing_subscriber::fmt().init();

    let config = parse_args();
    print_hardware_info();

    // Load dataset
    let dataset = LdbcDataset::load(&config.dataset).unwrap_or_else(|e| {
        eprintln!("Failed to load dataset from {}: {}", config.dataset.display(), e);
        std::process::exit(1);
    });

    let source = config
        .source
        .or(dataset.bfs_source)
        .unwrap_or(dataset.vertices[0]);

    let total_elements = (dataset.vertices.len() + dataset.edges.len()) as f64;

    if !config.csv {
        eprintln!("=== LDBC Graphalytics BFS Benchmark ===");
        eprintln!(
            "Dataset:  {} ({} vertices, {} edges, {})",
            dataset.name,
            fmt_num(dataset.vertices.len() as u64),
            fmt_num(dataset.edges.len() as u64),
            if dataset.directed { "directed" } else { "undirected" },
        );
        eprintln!("Source:   {}", source);
        eprintln!("Runs:     {}", config.runs);
        eprintln!("Direction: both (LDBC BFS treats edges as undirected)");
        if config.strata_only {
            eprintln!("Mode:     strata-only (petgraph comparison skipped)");
        }
        eprintln!();
    }

    // -----------------------------------------------------------------------
    // Load phase
    // -----------------------------------------------------------------------

    // Strata
    let db = create_db(DurabilityConfig::Cache);

    if !config.csv && !config.quiet {
        eprint!("Loading graph into Strata...");
    }
    let strata_load_time = load_graph(&db, &dataset);
    if !config.csv && !config.quiet {
        eprintln!(" done ({:.1}ms)", strata_load_time.as_secs_f64() * 1000.0);
    }

    // petgraph (unless --strata-only)
    let petgraph_state = if !config.strata_only {
        if !config.csv && !config.quiet {
            eprint!("Loading graph into petgraph...");
        }
        let pg_start = Instant::now();
        let (pg_graph, id_map) = dataset.to_petgraph();
        let pg_load_time = pg_start.elapsed();
        if !config.csv && !config.quiet {
            eprintln!(" done ({:.1}ms)", pg_load_time.as_secs_f64() * 1000.0);
        }
        Some((pg_graph, id_map, pg_load_time))
    } else {
        None
    };

    // Print load comparison
    if !config.csv && !config.quiet {
        eprintln!();
        eprintln!("--- Load Phase ---");
        eprintln!("  {:12} {}", "Strata:", fmt_ms(strata_load_time));
        if let Some((_, _, pg_load_time)) = &petgraph_state {
            let pg_secs = pg_load_time.as_secs_f64();
            if pg_secs > 0.0 {
                let ratio = strata_load_time.as_secs_f64() / pg_secs;
                eprintln!(
                    "  {:12} {}  ({:.2}x faster)",
                    "petgraph:", fmt_ms(*pg_load_time), ratio
                );
            } else {
                eprintln!("  {:12} {}", "petgraph:", fmt_ms(*pg_load_time));
            }
        }
    }

    // Load BFS reference for validation
    let reference = if !config.no_validate {
        let bfs_path = config.dataset.join(format!("{}-BFS", dataset.name));
        if bfs_path.exists() {
            Some(BfsReference::load(&bfs_path).unwrap_or_else(|e| {
                eprintln!("Failed to load BFS reference: {}", e);
                std::process::exit(1);
            }))
        } else {
            if !config.csv && !config.quiet {
                eprintln!("No BFS reference file found, skipping LDBC validation.");
            }
            None
        }
    } else {
        None
    };

    // -----------------------------------------------------------------------
    // BFS phase — Strata
    // -----------------------------------------------------------------------

    let mut strata_times = Vec::with_capacity(config.runs);
    let mut ldbc_validation_pass: Option<bool> = None;
    let mut cross_validation_pass: Option<bool> = None;

    if config.csv {
        print_csv_header();
    }

    for run in 0..config.runs {
        let bfs_run = run_bfs(&db, source);
        let bfs_ms = bfs_run.elapsed.as_secs_f64() * 1000.0;
        let evps = total_elements / bfs_run.elapsed.as_secs_f64();
        strata_times.push(bfs_run.elapsed);

        // Validate first run against LDBC reference
        if let Some(ref reference) = reference {
            if run == 0 || config.validate_only {
                let validation = validate_bfs(&dataset, &bfs_run.depths, reference);
                if run == 0 {
                    ldbc_validation_pass = Some(validation.pass);
                }
                if !config.csv {
                    if validation.pass {
                        eprintln!(
                            "LDBC Validation: PASS ({} vertices checked)",
                            dataset.vertices.len()
                        );
                    } else {
                        eprintln!(
                            "LDBC Validation: FAIL ({} mismatches out of {} vertices)",
                            validation.mismatches,
                            dataset.vertices.len()
                        );
                        for detail in &validation.details {
                            eprintln!("  {}", detail);
                        }
                    }
                }
                if !validation.pass && config.validate_only {
                    std::process::exit(1);
                }
            }
        }

        // Cross-validate first run against petgraph
        if run == 0 {
            if let Some((ref pg_graph, ref id_map, _)) = petgraph_state {
                let pg_source = id_map[&source];
                let pg_depths = petgraph_bfs(pg_graph, pg_source);
                let (pass, checked, mismatches) =
                    cross_validate(&dataset, &bfs_run.depths, &pg_depths, id_map);
                cross_validation_pass = Some(pass);
                if !config.csv {
                    if pass {
                        eprintln!(
                            "Cross-validation: PASS (depths match on all {} vertices)",
                            fmt_num(checked as u64)
                        );
                    } else {
                        eprintln!(
                            "Cross-validation: FAIL ({} mismatches out of {} vertices)",
                            mismatches, checked
                        );
                    }
                }
            }
        }

        if config.validate_only {
            if !config.csv {
                eprintln!("Validate-only mode, skipping remaining runs.");
            }
            return;
        }

        if config.csv {
            print_csv_row(
                "strata",
                run + 1,
                bfs_ms,
                evps,
                dataset.vertices.len(),
                dataset.edges.len(),
            );
        } else if config.quiet && run == 0 {
            eprintln!(
                "Strata BFS: {:.3}ms, EVPS: {:.0}, |V|={}, |E|={}",
                bfs_ms, evps, dataset.vertices.len(), dataset.edges.len()
            );
        }
    }

    let strata_stats = compute_stats(&mut strata_times, total_elements);

    // -----------------------------------------------------------------------
    // BFS phase — petgraph
    // -----------------------------------------------------------------------

    let petgraph_stats = if let Some((ref pg_graph, ref id_map, _)) = petgraph_state {
        let pg_source = id_map[&source];
        let mut pg_times = Vec::with_capacity(config.runs);

        for run in 0..config.runs {
            let start = Instant::now();
            let _ = petgraph_bfs(pg_graph, pg_source);
            let elapsed = start.elapsed();
            pg_times.push(elapsed);

            if config.csv {
                let bfs_ms = elapsed.as_secs_f64() * 1000.0;
                let evps = total_elements / elapsed.as_secs_f64();
                print_csv_row(
                    "petgraph",
                    run + 1,
                    bfs_ms,
                    evps,
                    dataset.vertices.len(),
                    dataset.edges.len(),
                );
            } else if config.quiet && run == 0 {
                let bfs_ms = elapsed.as_secs_f64() * 1000.0;
                let evps = total_elements / elapsed.as_secs_f64();
                eprintln!(
                    "petgraph BFS: {:.3}ms, EVPS: {:.0}, |V|={}, |E|={}",
                    bfs_ms, evps, dataset.vertices.len(), dataset.edges.len()
                );
            }
        }

        Some(compute_stats(&mut pg_times, total_elements))
    } else {
        None
    };

    // -----------------------------------------------------------------------
    // Output comparison table
    // -----------------------------------------------------------------------

    if !config.csv && !config.quiet {
        eprintln!();
        eprintln!(
            "--- BFS Phase ({} runs, direction=both) ---",
            strata_stats.count
        );
        eprintln!(
            "  {:16} {:>10} {:>10} {:>14}",
            "", "avg", "p50", "EVPS"
        );
        eprintln!(
            "  {:16} {:>10} {:>10} {:>14}",
            "Strata:",
            fmt_ms(strata_stats.avg),
            fmt_ms(strata_stats.p50),
            fmt_num(strata_stats.avg_evps as u64),
        );

        if let Some(ref pg) = petgraph_stats {
            eprintln!(
                "  {:16} {:>10} {:>10} {:>14}",
                "petgraph:",
                fmt_ms(pg.avg),
                fmt_ms(pg.p50),
                fmt_num(pg.avg_evps as u64),
            );
            let avg_ratio = strata_stats.avg.as_secs_f64() / pg.avg.as_secs_f64();
            let p50_ratio = strata_stats.p50.as_secs_f64() / pg.p50.as_secs_f64();
            eprintln!(
                "  {:16} {:>10} {:>10}",
                "Ratio:",
                format!("{:.1}x", avg_ratio),
                format!("{:.1}x", p50_ratio),
            );
        }

        // Full Strata percentile table
        eprintln!();
        eprintln!("--- Strata Detailed ({} runs) ---", strata_stats.count);
        eprintln!(
            "  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
            "avg", "p50", "p95", "p99", "min", "max"
        );
        eprintln!(
            "  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
            fmt_ms(strata_stats.avg),
            fmt_ms(strata_stats.p50),
            fmt_ms(strata_stats.p95),
            fmt_ms(strata_stats.p99),
            fmt_ms(strata_stats.min),
            fmt_ms(strata_stats.max),
        );
        eprintln!(
            "  EVPS (avg): {}  (|V|+|E|={} / {:.6}s)",
            fmt_num(strata_stats.avg_evps as u64),
            total_elements as u64,
            strata_stats.avg.as_secs_f64(),
        );

        // Published reference points
        print_published_references(strata_stats.avg_evps, &dataset.name);
    }

    // -----------------------------------------------------------------------
    // Record results
    // -----------------------------------------------------------------------

    let mut recorder = ResultRecorder::new("graph-bfs");

    // Strata result
    {
        let mut params = HashMap::new();
        params.insert("dataset".into(), serde_json::json!(dataset.name));
        params.insert("engine".into(), serde_json::json!("strata"));
        params.insert("source".into(), serde_json::json!(source));
        params.insert("vertices".into(), serde_json::json!(dataset.vertices.len()));
        params.insert("edges".into(), serde_json::json!(dataset.edges.len()));
        params.insert("direction".into(), serde_json::json!("both"));

        recorder.record(BenchmarkResult {
            benchmark: format!(
                "graph-bfs/strata/{}/{}V-{}E",
                dataset.name,
                dataset.vertices.len(),
                dataset.edges.len()
            ),
            category: "graph-bfs".to_string(),
            parameters: params,
            metrics: BenchmarkMetrics {
                ops_per_sec: Some(strata_stats.avg_evps),
                p50_ns: Some(strata_stats.p50.as_nanos() as u64),
                p95_ns: Some(strata_stats.p95.as_nanos() as u64),
                p99_ns: Some(strata_stats.p99.as_nanos() as u64),
                min_ns: Some(strata_stats.min.as_nanos() as u64),
                max_ns: Some(strata_stats.max.as_nanos() as u64),
                avg_ns: Some(strata_stats.avg.as_nanos() as u64),
                samples: Some(strata_stats.count as u64),
                ..Default::default()
            },
        });
    }

    // petgraph result
    if let Some(ref pg) = petgraph_stats {
        let mut params = HashMap::new();
        params.insert("dataset".into(), serde_json::json!(dataset.name));
        params.insert("engine".into(), serde_json::json!("petgraph"));
        params.insert("source".into(), serde_json::json!(source));
        params.insert("vertices".into(), serde_json::json!(dataset.vertices.len()));
        params.insert("edges".into(), serde_json::json!(dataset.edges.len()));
        params.insert("direction".into(), serde_json::json!("both"));

        recorder.record(BenchmarkResult {
            benchmark: format!(
                "graph-bfs/petgraph/{}/{}V-{}E",
                dataset.name,
                dataset.vertices.len(),
                dataset.edges.len()
            ),
            category: "graph-bfs".to_string(),
            parameters: params,
            metrics: BenchmarkMetrics {
                ops_per_sec: Some(pg.avg_evps),
                p50_ns: Some(pg.p50.as_nanos() as u64),
                p95_ns: Some(pg.p95.as_nanos() as u64),
                p99_ns: Some(pg.p99.as_nanos() as u64),
                min_ns: Some(pg.min.as_nanos() as u64),
                max_ns: Some(pg.max.as_nanos() as u64),
                avg_ns: Some(pg.avg.as_nanos() as u64),
                samples: Some(pg.count as u64),
                ..Default::default()
            },
        });
    }

    if !config.csv {
        eprintln!();
        eprintln!("=== Benchmark complete ===");
    }
    if let Ok(json_path) = recorder.save() {
        if !config.csv {
            let _ = write_markdown_report(
                &json_path,
                &dataset,
                source,
                config.runs,
                config.strata_only,
                strata_load_time,
                petgraph_state.as_ref().map(|(_, _, t)| *t),
                &strata_stats,
                petgraph_stats.as_ref(),
                ldbc_validation_pass,
                cross_validation_pass,
            );
        }
    }
}
