//! LDBC Graphalytics LCC Benchmark — Strata vs petgraph
//!
//! Local Clustering Coefficient: for each vertex v with degree d >= 2,
//! LCC(v) = triangles(v) / (d*(d-1)/2). Vertices with d < 2 get LCC = 0.
//!
//! Warning: O(V * d^2) — can be slow on dense graphs (e.g., dota-league).
//!
//! Run:           `cargo bench --bench graph_lcc`
//! Quick:         `cargo bench --bench graph_lcc -- -q`
//! Custom data:   `cargo bench --bench graph_lcc -- --dataset path/to/ldbc/dir`

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
            "graph-lcc/{}/{}/{}V-{}E",
            engine, dataset.name, dataset.vertices.len(), dataset.edges.len()
        ),
        category: "graph-lcc".to_string(),
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

    let avg_degree = if !dataset.vertices.is_empty() {
        dataset.edges.len() as f64 / dataset.vertices.len() as f64
    } else {
        0.0
    };

    if !config.csv {
        eprintln!("=== LDBC Graphalytics LCC Benchmark ===");
        eprintln!(
            "Dataset:    {} ({} vertices, {} edges, {})",
            dataset.name,
            fmt_num(dataset.vertices.len() as u64),
            fmt_num(dataset.edges.len() as u64),
            if dataset.directed { "directed" } else { "undirected" },
        );
        eprintln!("Avg degree: {:.1}", avg_degree);
        eprintln!("Runs:       {}", config.runs);
        if avg_degree > 500.0 {
            eprintln!(
                "WARNING: High average degree ({:.0}) — LCC is O(V*d^2), expect slow runtime.",
                avg_degree
            );
        }
        eprintln!();
    }

    let reference = if !config.no_validate {
        let ref_path = config.dataset.join(format!("{}-LCC", dataset.name));
        if ref_path.exists() {
            Some(F64Reference::load(&ref_path).unwrap_or_else(|e| {
                eprintln!("Failed to load LCC reference: {}", e);
                std::process::exit(1);
            }))
        } else {
            if !config.csv && !config.quiet {
                eprintln!("No LCC reference file found, skipping validation.");
            }
            None
        }
    } else {
        None
    };

    let mut recorder = ResultRecorder::new("graph-lcc");

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
            let result = strata.graph_lcc("ldbc").expect("graph_lcc failed");
            times.push(start.elapsed());

            if run == 0 {
                if let Some(ref reference) = reference {
                    let (pass, mismatches) = validate_f64(&dataset, &result.result, reference, 1e-4);
                    if !config.csv {
                        if pass {
                            eprintln!("Strata LDBC Validation: PASS ({} vertices checked, tolerance=1e-4)", dataset.vertices.len());
                        } else {
                            eprintln!("Strata LDBC Validation: FAIL ({} mismatches out of {} vertices, tolerance=1e-4)", mismatches, dataset.vertices.len());
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
                    eprintln!("Strata LCC: {:.3}ms, EVPS: {:.0}, |V|={}, |E|={}", ms, evps, dataset.vertices.len(), dataset.edges.len());
                }
            }
        }

        let stats = compute_stats(&mut times, total);
        if !config.csv && !config.quiet {
            print_stats_table("Strata LCC", &stats);
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
        let dir_edges = if dataset.directed {
            Some(directed_edge_set(&dataset, &id_map))
        } else {
            None
        };
        let pg_load_time = pg_start.elapsed();
        if !config.csv && !config.quiet {
            eprintln!(" done ({})", fmt_ms(pg_load_time));
        }

        let mut times = Vec::with_capacity(config.runs);
        for run in 0..config.runs {
            let start = Instant::now();
            let result = petgraph_lcc(&pg_graph, dir_edges.as_ref());
            times.push(start.elapsed());

            if run == 0 {
                if let Some(ref reference) = reference {
                    let mut mismatches = 0;
                    for &vid in &dataset.vertices {
                        let expected = reference.values.get(&vid).copied();
                        let actual = id_map.get(&vid).and_then(|idx| result.get(idx).copied());
                        match (expected, actual) {
                            (Some(e), Some(a)) => {
                                let rel_err = if e.abs() > 1e-15 {
                                    (a - e).abs() / e.abs()
                                } else {
                                    (a - e).abs()
                                };
                                if rel_err > 1e-4 {
                                    mismatches += 1;
                                }
                            }
                            (Some(_), None) | (None, Some(_)) => mismatches += 1,
                            _ => {}
                        }
                    }
                    if !config.csv {
                        if mismatches == 0 {
                            eprintln!("petgraph LDBC Validation: PASS ({} vertices checked, tolerance=1e-4)", dataset.vertices.len());
                        } else {
                            eprintln!("petgraph LDBC Validation: FAIL ({} mismatches out of {} vertices, tolerance=1e-4)", mismatches, dataset.vertices.len());
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
                    eprintln!("petgraph LCC: {:.3}ms, EVPS: {:.0}, |V|={}, |E|={}", ms, evps, dataset.vertices.len(), dataset.edges.len());
                }
            }
        }

        let stats = compute_stats(&mut times, total);
        if !config.csv && !config.quiet {
            print_stats_table("petgraph LCC", &stats);
        }
        record(&mut recorder, "petgraph", &dataset, &stats);
    }

    if !config.csv {
        eprintln!();
        eprintln!("=== Benchmark complete ===");
    }
    let _ = recorder.save();
}
