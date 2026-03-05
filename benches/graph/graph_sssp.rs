//! LDBC Graphalytics SSSP Benchmark — Strata vs petgraph
//!
//! Single-Source Shortest Path using Dijkstra on weighted graphs.
//! Requires a dataset with edge weights (3-column .e file).
//!
//! Run:           `cargo bench --bench graph_sssp`
//! Quick:         `cargo bench --bench graph_sssp -- -q`
//! Custom data:   `cargo bench --bench graph_sssp -- --dataset data/graph/kgs`

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
    source: u64,
) {
    let mut params = HashMap::new();
    params.insert("dataset".into(), serde_json::json!(dataset.name));
    params.insert("engine".into(), serde_json::json!(engine));
    params.insert("source".into(), serde_json::json!(source));
    params.insert("vertices".into(), serde_json::json!(dataset.vertices.len()));
    params.insert("edges".into(), serde_json::json!(dataset.edges.len()));

    recorder.record(BenchmarkResult {
        benchmark: format!(
            "graph-sssp/{}/{}/{}V-{}E",
            engine, dataset.name, dataset.vertices.len(), dataset.edges.len()
        ),
        category: "graph-sssp".to_string(),
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

    // SSSP requires weighted edges
    if dataset.edge_weights.is_none() {
        eprintln!("=== LDBC Graphalytics SSSP Benchmark ===");
        eprintln!(
            "Dataset: {} — no edge weights found (unweighted graph).",
            dataset.name
        );
        eprintln!("SSSP requires a weighted dataset (3-column .e file). Skipping.");
        eprintln!("Try: --dataset data/graph/kgs  or  --dataset data/graph/dota-league");
        return;
    }

    let source = dataset.sssp_source.unwrap_or_else(|| {
        let fallback = dataset.bfs_source.unwrap_or(dataset.vertices[0]);
        eprintln!(
            "No SSSP source vertex in properties, falling back to {}",
            fallback
        );
        fallback
    });

    let total = (dataset.vertices.len() + dataset.edges.len()) as f64;

    if !config.csv {
        eprintln!("=== LDBC Graphalytics SSSP Benchmark ===");
        eprintln!(
            "Dataset:  {} ({} vertices, {} edges, {})",
            dataset.name,
            fmt_num(dataset.vertices.len() as u64),
            fmt_num(dataset.edges.len() as u64),
            if dataset.directed { "directed" } else { "undirected" },
        );
        eprintln!("Source:   {}", source);
        eprintln!("Weighted: yes");
        eprintln!("Runs:     {}", config.runs);
        eprintln!();
    }

    let reference = if !config.no_validate {
        let ref_path = config.dataset.join(format!("{}-SSSP", dataset.name));
        if ref_path.exists() {
            Some(F64Reference::load(&ref_path).unwrap_or_else(|e| {
                eprintln!("Failed to load SSSP reference: {}", e);
                std::process::exit(1);
            }))
        } else {
            if !config.csv && !config.quiet {
                eprintln!("No SSSP reference file found, skipping validation.");
            }
            None
        }
    } else {
        None
    };

    let mut recorder = ResultRecorder::new("graph-sssp");

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

        let source_str = source.to_string();
        let mut times = Vec::with_capacity(config.runs);
        for run in 0..config.runs {
            let start = Instant::now();
            let result = strata
                .graph_sssp("ldbc", &source_str, None)
                .expect("graph_sssp failed");
            times.push(start.elapsed());

            if run == 0 {
                if let Some(ref reference) = reference {
                    let (pass, mismatches) = validate_f64(&dataset, &result.result, reference, 1e-6);
                    if !config.csv {
                        if pass {
                            eprintln!("Strata LDBC Validation: PASS ({} vertices checked, tolerance=1e-6)", dataset.vertices.len());
                        } else {
                            eprintln!("Strata LDBC Validation: FAIL ({} mismatches out of {} vertices, tolerance=1e-6)", mismatches, dataset.vertices.len());
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
                    eprintln!("Strata SSSP: {:.3}ms, EVPS: {:.0}, |V|={}, |E|={}", ms, evps, dataset.vertices.len(), dataset.edges.len());
                }
            }
        }

        let stats = compute_stats(&mut times, total);
        if !config.csv && !config.quiet {
            print_stats_table("Strata SSSP", &stats);
        }
        record(&mut recorder, "strata", &dataset, &stats, source);
    }

    // --- petgraph ---
    {
        if !config.csv && !config.quiet {
            eprint!(
                "Loading weighted graph into petgraph ({})...",
                if dataset.directed { "directed" } else { "undirected" }
            );
        }
        let pg_start = Instant::now();

        // Build the appropriate graph type based on directedness
        let (pg_graph_u, id_map_u) = if !dataset.directed {
            let r = dataset.to_petgraph_weighted();
            (Some(r.0), Some(r.1))
        } else {
            (None, None)
        };
        let (pg_graph_d, id_map_d) = if dataset.directed {
            let r = dataset.to_petgraph_weighted_directed();
            (Some(r.0), Some(r.1))
        } else {
            (None, None)
        };

        let pg_load_time = pg_start.elapsed();
        if !config.csv && !config.quiet {
            eprintln!(" done ({})", fmt_ms(pg_load_time));
        }

        let id_map = if dataset.directed {
            id_map_d.as_ref().unwrap()
        } else {
            id_map_u.as_ref().unwrap()
        };

        let pg_source = match id_map.get(&source) {
            Some(&idx) => idx,
            None => {
                eprintln!("Source vertex {} not found in dataset", source);
                std::process::exit(1);
            }
        };

        let mut times = Vec::with_capacity(config.runs);
        for run in 0..config.runs {
            let start = Instant::now();
            let result = if dataset.directed {
                petgraph_sssp_directed(pg_graph_d.as_ref().unwrap(), pg_source)
            } else {
                petgraph_sssp(pg_graph_u.as_ref().unwrap(), pg_source)
            };
            times.push(start.elapsed());

            if run == 0 {
                if let Some(ref reference) = reference {
                    let mut mismatches = 0;
                    for &vid in &dataset.vertices {
                        let expected = reference.values.get(&vid).copied();
                        let actual = id_map.get(&vid).and_then(|idx| result.get(idx).copied());
                        match (expected, actual) {
                            (Some(e), Some(a)) => {
                                if e.is_infinite() {
                                    mismatches += 1;
                                } else {
                                    let rel_err = if e.abs() > 1e-15 {
                                        (a - e).abs() / e.abs()
                                    } else {
                                        (a - e).abs()
                                    };
                                    if rel_err > 1e-6 {
                                        mismatches += 1;
                                    }
                                }
                            }
                            (Some(e), None) => {
                                if !e.is_infinite() {
                                    mismatches += 1;
                                }
                            }
                            (None, Some(_)) => mismatches += 1,
                            _ => {}
                        }
                    }
                    if !config.csv {
                        if mismatches == 0 {
                            eprintln!("petgraph LDBC Validation: PASS ({} vertices checked, tolerance=1e-6)", dataset.vertices.len());
                        } else {
                            eprintln!("petgraph LDBC Validation: FAIL ({} mismatches out of {} vertices, tolerance=1e-6)", mismatches, dataset.vertices.len());
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
                    eprintln!("petgraph SSSP: {:.3}ms, EVPS: {:.0}, |V|={}, |E|={}", ms, evps, dataset.vertices.len(), dataset.edges.len());
                }
            }
        }

        let stats = compute_stats(&mut times, total);
        if !config.csv && !config.quiet {
            print_stats_table("petgraph SSSP", &stats);
        }
        record(&mut recorder, "petgraph", &dataset, &stats, source);
    }

    if !config.csv {
        eprintln!();
        eprintln!("=== Benchmark complete ===");
    }
    let _ = recorder.save();
}
