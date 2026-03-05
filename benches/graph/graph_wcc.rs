//! LDBC Graphalytics WCC Benchmark — Strata vs petgraph
//!
//! Weakly Connected Components: assigns each vertex to a component ID
//! (the minimum vertex ID in that component).
//!
//! Run:           `cargo bench --bench graph_wcc`
//! Quick:         `cargo bench --bench graph_wcc -- -q`
//! Custom data:   `cargo bench --bench graph_wcc -- --dataset path/to/ldbc/dir`

#[allow(unused)]
#[path = "../harness/mod.rs"]
mod harness;

#[allow(unused)]
mod ldbc;

use harness::recorder::ResultRecorder;
use harness::print_hardware_info;
use ldbc::*;
use std::collections::HashMap;
use std::time::Instant;
use strata_benchmarks::schema::{BenchmarkMetrics, BenchmarkResult};

fn record(
    recorder: &mut ResultRecorder,
    engine: &str,
    dataset: &LdbcDataset,
    stats: &RunStats,
) {
    let mut params = HashMap::new();
    params.insert("dataset".into(), serde_json::json!(dataset.name));
    params.insert("engine".into(), serde_json::json!(engine));
    params.insert("vertices".into(), serde_json::json!(dataset.vertices.len()));
    params.insert("edges".into(), serde_json::json!(dataset.edges.len()));

    recorder.record(BenchmarkResult {
        benchmark: format!(
            "graph-wcc/{}/{}/{}V-{}E",
            engine, dataset.name, dataset.vertices.len(), dataset.edges.len()
        ),
        category: "graph-wcc".to_string(),
        parameters: params,
        metrics: BenchmarkMetrics {
            ops_per_sec: Some(stats.avg_evps),
            p50_ns: Some(stats.p50.as_nanos() as u64),
            p95_ns: Some(stats.p95.as_nanos() as u64),
            p99_ns: Some(stats.p99.as_nanos() as u64),
            min_ns: Some(stats.min.as_nanos() as u64),
            max_ns: Some(stats.max.as_nanos() as u64),
            avg_ns: Some(stats.avg.as_nanos() as u64),
            samples: Some(stats.count as u64),
            ..Default::default()
        },
    });
}

fn main() {
    let config = parse_args();
    print_hardware_info();

    let dataset = LdbcDataset::load(&config.dataset).unwrap_or_else(|e| {
        eprintln!("Failed to load dataset from {}: {}", config.dataset.display(), e);
        std::process::exit(1);
    });

    let total = (dataset.vertices.len() + dataset.edges.len()) as f64;

    if !config.csv {
        eprintln!("=== LDBC Graphalytics WCC Benchmark ===");
        eprintln!(
            "Dataset:  {} ({} vertices, {} edges, {})",
            dataset.name,
            fmt_num(dataset.vertices.len() as u64),
            fmt_num(dataset.edges.len() as u64),
            if dataset.directed { "directed" } else { "undirected" },
        );
        eprintln!("Runs:     {}", config.runs);
        eprintln!();
    }

    // Load reference for validation
    let reference = if !config.no_validate {
        let ref_path = config.dataset.join(format!("{}-WCC", dataset.name));
        if ref_path.exists() {
            Some(U64Reference::load(&ref_path).unwrap_or_else(|e| {
                eprintln!("Failed to load WCC reference: {}", e);
                std::process::exit(1);
            }))
        } else {
            if !config.csv && !config.quiet {
                eprintln!("No WCC reference file found, skipping validation.");
            }
            None
        }
    } else {
        None
    };

    let mut recorder = ResultRecorder::new("graph-wcc");

    // --- Strata ---
    {
        let db = harness::create_db(harness::DurabilityConfig::Cache);
        let strata = &db.db;

        if !config.csv && !config.quiet {
            eprint!("Loading graph into Strata...");
        }
        let load_time = load_graph_into_strata(strata, &dataset);
        if !config.csv && !config.quiet {
            eprintln!(" done ({})", fmt_ms(load_time));
        }

        let mut times = Vec::with_capacity(config.runs);
        for run in 0..config.runs {
            let start = Instant::now();
            let result = strata.graph_wcc("ldbc").expect("graph_wcc failed");
            times.push(start.elapsed());

            if run == 0 {
                if let Some(ref reference) = reference {
                    let (pass, mismatches) = validate_u64(&dataset, &result.result, reference);
                    if !config.csv {
                        if pass {
                            eprintln!("Strata LDBC Validation: PASS ({} vertices checked)", dataset.vertices.len());
                        } else {
                            eprintln!("Strata LDBC Validation: FAIL ({} mismatches out of {} vertices)", mismatches, dataset.vertices.len());
                        }
                    }
                }

                if config.csv {
                    let ms = times[0].as_secs_f64() * 1000.0;
                    let evps = total / times[0].as_secs_f64();
                    println!("\"strata\",1,{:.3},{:.0},{},{}", ms, evps, dataset.vertices.len(), dataset.edges.len());
                } else if config.quiet {
                    let ms = times[0].as_secs_f64() * 1000.0;
                    let evps = total / times[0].as_secs_f64();
                    eprintln!("Strata WCC: {:.3}ms, EVPS: {:.0}, |V|={}, |E|={}", ms, evps, dataset.vertices.len(), dataset.edges.len());
                }
            }
        }

        let stats = compute_stats(&mut times, total);
        if !config.csv && !config.quiet {
            print_stats_table("Strata WCC", &stats);
        }
        record(&mut recorder, "strata", &dataset, &stats);
    }

    // --- petgraph ---
    {
        if !config.csv && !config.quiet {
            eprint!("Loading graph into petgraph...");
        }
        let pg_start = Instant::now();
        let (pg_graph, id_map) = dataset.to_petgraph();
        let pg_load_time = pg_start.elapsed();
        let rev_map = reverse_id_map(&id_map);
        if !config.csv && !config.quiet {
            eprintln!(" done ({})", fmt_ms(pg_load_time));
        }

        let mut times = Vec::with_capacity(config.runs);
        for run in 0..config.runs {
            let start = Instant::now();
            let result = petgraph_wcc(&pg_graph, &rev_map);
            times.push(start.elapsed());

            if run == 0 {
                if let Some(ref reference) = reference {
                    let mut mismatches = 0;
                    for &vid in &dataset.vertices {
                        let expected = reference.values.get(&vid).copied();
                        let actual = id_map.get(&vid).and_then(|idx| result.get(idx).copied());
                        match (expected, actual) {
                            (Some(e), Some(a)) if e != a => mismatches += 1,
                            (Some(_), None) | (None, Some(_)) => mismatches += 1,
                            _ => {}
                        }
                    }
                    if !config.csv {
                        if mismatches == 0 {
                            eprintln!("petgraph LDBC Validation: PASS ({} vertices checked)", dataset.vertices.len());
                        } else {
                            eprintln!("petgraph LDBC Validation: FAIL ({} mismatches out of {} vertices)", mismatches, dataset.vertices.len());
                        }
                    }
                }

                if config.csv {
                    let ms = times[0].as_secs_f64() * 1000.0;
                    let evps = total / times[0].as_secs_f64();
                    println!("\"petgraph\",1,{:.3},{:.0},{},{}", ms, evps, dataset.vertices.len(), dataset.edges.len());
                } else if config.quiet {
                    let ms = times[0].as_secs_f64() * 1000.0;
                    let evps = total / times[0].as_secs_f64();
                    eprintln!("petgraph WCC: {:.3}ms, EVPS: {:.0}, |V|={}, |E|={}", ms, evps, dataset.vertices.len(), dataset.edges.len());
                }
            }
        }

        let stats = compute_stats(&mut times, total);
        if !config.csv && !config.quiet {
            print_stats_table("petgraph WCC", &stats);
        }
        record(&mut recorder, "petgraph", &dataset, &stats);
    }

    if !config.csv {
        eprintln!();
        eprintln!("=== Benchmark complete ===");
    }
    let _ = recorder.save();
}
