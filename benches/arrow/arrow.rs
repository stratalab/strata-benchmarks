//! Arrow export benchmarks: every permutation of primitive × format × dataset size.
//!
//! Primitives: KV, JSON, Events, Graph
//! Formats: CSV, JSON, JSONL
//! Dataset sizes: 100, 1_000, 10_000 (quick); adds 50_000, 100_000 (full)
//!
//! Usage:
//!   cargo bench --bench arrow            # full suite
//!   cargo bench --bench arrow -- --quick # quick mode (smaller datasets)
//!   cargo bench --bench arrow -- -q      # same as --quick

#[allow(unused)]
#[path = "../harness/mod.rs"]
mod harness;

use std::collections::HashMap;
use std::time::Instant;

use harness::recorder::ResultRecorder;
use harness::{
    create_db, event_payload, json_document, kv_key, kv_value, measure_percentiles,
    print_hardware_info, report_percentiles, DurabilityConfig,
};
use stratadb::{Command, ExportFormat, ExportPrimitive, Executor, Output, Value};

// =============================================================================
// Configuration
// =============================================================================

const QUICK_SCALES: &[usize] = &[100, 1_000, 10_000];
const FULL_SCALES: &[usize] = &[100, 1_000, 10_000, 50_000, 100_000];

const ALL_PRIMITIVES: &[ExportPrimitive] = &[
    ExportPrimitive::Kv,
    ExportPrimitive::Json,
    ExportPrimitive::Events,
    ExportPrimitive::Graph,
];

const ALL_FORMATS: &[ExportFormat] = &[ExportFormat::Csv, ExportFormat::Json, ExportFormat::Jsonl];

const MEASURE_ITERATIONS: usize = 10;

fn format_label(f: &ExportFormat) -> &'static str {
    match f {
        ExportFormat::Csv => "csv",
        ExportFormat::Json => "json",
        ExportFormat::Jsonl => "jsonl",
    }
}

fn primitive_label(p: &ExportPrimitive) -> &'static str {
    match p {
        ExportPrimitive::Kv => "kv",
        ExportPrimitive::Json => "json",
        ExportPrimitive::Events => "events",
        ExportPrimitive::Graph => "graph",
    }
}

// =============================================================================
// Data population
// =============================================================================

fn populate_kv(db: &stratadb::Strata, n: usize) {
    for i in 0..n as u64 {
        db.kv_put(&kv_key(i), kv_value()).unwrap();
    }
}

fn populate_json(db: &stratadb::Strata, n: usize) {
    for i in 0..n as u64 {
        db.json_set(&format!("doc:{}", i), "$", json_document(i))
            .unwrap();
    }
}

fn populate_events(db: &stratadb::Strata, n: usize) {
    let event_types = ["action", "observation", "tool_call", "response"];
    for i in 0..n as u64 {
        db.event_append(event_types[(i % 4) as usize], event_payload())
            .unwrap();
    }
}

fn populate_graph(db: &stratadb::Strata, n: usize) {
    let graph_name = "bench_graph";
    db.graph_create(graph_name).unwrap();

    // Create n nodes with properties
    for i in 0..n as u64 {
        let mut props = HashMap::new();
        props.insert("weight".to_string(), Value::Float(i as f64 * 0.1));
        props.insert(
            "label".to_string(),
            Value::String(format!("node_{}", i)),
        );
        db.graph_add_node(
            graph_name,
            &format!("n{}", i),
            None,
            Some(Value::object(props)),
        )
        .unwrap();
    }
    // Add edges between consecutive nodes (linear chain)
    for i in 0..(n.saturating_sub(1)) as u64 {
        db.graph_add_edge(
            graph_name,
            &format!("n{}", i),
            &format!("n{}", i + 1),
            "connects",
            Some(1.0),
            None,
        )
        .unwrap();
    }
}

fn populate(db: &stratadb::Strata, primitive: &ExportPrimitive, n: usize) {
    match primitive {
        ExportPrimitive::Kv => populate_kv(db, n),
        ExportPrimitive::Json => populate_json(db, n),
        ExportPrimitive::Events => populate_events(db, n),
        ExportPrimitive::Graph => populate_graph(db, n),
    }
}

// =============================================================================
// Export benchmark runner
// =============================================================================

fn run_export(
    executor: &Executor,
    primitive: ExportPrimitive,
    format: ExportFormat,
) -> (u64, u64) {
    let cmd = Command::DbExport {
        branch: None,
        space: None,
        primitive,
        format,
        prefix: None,
        limit: None,
        path: None, // inline — avoid filesystem overhead
    };
    match executor.execute(cmd).unwrap() {
        Output::Exported(result) => {
            let size = result
                .data
                .as_ref()
                .map(|d| d.len() as u64)
                .unwrap_or(0);
            (result.row_count, size)
        }
        other => panic!("Unexpected output: {:?}", other),
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    print_hardware_info();

    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick" || a == "-q");

    let scales = if quick { QUICK_SCALES } else { FULL_SCALES };

    eprintln!(
        "\n=== Arrow Export Benchmark ({} mode) ===",
        if quick { "quick" } else { "full" }
    );
    eprintln!(
        "Permutations: {} primitives × {} formats × {} scales = {} combinations",
        ALL_PRIMITIVES.len(),
        ALL_FORMATS.len(),
        scales.len(),
        ALL_PRIMITIVES.len() * ALL_FORMATS.len() * scales.len(),
    );
    eprintln!("Iterations per combination: {}\n", MEASURE_ITERATIONS);

    let mut recorder = ResultRecorder::new("arrow-export");

    let total = ALL_PRIMITIVES.len() * ALL_FORMATS.len() * scales.len();
    let mut idx = 0;

    for &primitive in ALL_PRIMITIVES {
        for &scale in scales {
            // Create one DB per (primitive, scale) — reuse across formats
            let bench_db = create_db(DurabilityConfig::Cache);
            let populate_start = Instant::now();
            populate(&bench_db.db, &primitive, scale);
            let populate_elapsed = populate_start.elapsed();

            eprintln!(
                "  Populated {} with {} rows in {:.2?}",
                primitive_label(&primitive),
                scale,
                populate_elapsed,
            );

            let executor = Executor::new(bench_db.db.database());

            for &format in ALL_FORMATS {
                idx += 1;
                let label = format!(
                    "export/{}/{}/{}",
                    primitive_label(&primitive),
                    format_label(&format),
                    scale,
                );

                // Warm up: one export to prime any caches
                let (row_count, output_bytes) = run_export(&executor, primitive, format);

                // Measure
                let p = measure_percentiles(MEASURE_ITERATIONS, || {
                    run_export(&executor, primitive, format);
                });

                let throughput_rows_per_sec = if p.p50.as_secs_f64() > 0.0 {
                    row_count as f64 / p.p50.as_secs_f64()
                } else {
                    f64::INFINITY
                };

                let throughput_mb_per_sec = if p.p50.as_secs_f64() > 0.0 {
                    (output_bytes as f64 / (1024.0 * 1024.0)) / p.p50.as_secs_f64()
                } else {
                    f64::INFINITY
                };

                eprintln!(
                    "  [{:>3}/{}] {:<45} rows={:<8} bytes={:<10} rows/s={:<12.0} MB/s={:<8.1}",
                    idx,
                    total,
                    label,
                    row_count,
                    output_bytes,
                    throughput_rows_per_sec,
                    throughput_mb_per_sec,
                );
                report_percentiles(&format!("  {}", label), &p);

                // Record to JSON
                let mut params = HashMap::new();
                params.insert("primitive".into(), serde_json::json!(primitive_label(&primitive)));
                params.insert("format".into(), serde_json::json!(format_label(&format)));
                params.insert("scale".into(), serde_json::json!(scale));
                params.insert("row_count".into(), serde_json::json!(row_count));
                params.insert("output_bytes".into(), serde_json::json!(output_bytes));
                params.insert(
                    "throughput_rows_per_sec".into(),
                    serde_json::json!(throughput_rows_per_sec),
                );
                params.insert(
                    "throughput_mb_per_sec".into(),
                    serde_json::json!(throughput_mb_per_sec),
                );
                recorder.record_latency(
                    &label,
                    params,
                    &p,
                    None,
                    MEASURE_ITERATIONS as u64,
                );
            }
        }
    }

    // Summary table
    eprintln!("\n=== Summary ===");
    eprintln!(
        "  Total combinations tested: {} ({} primitives × {} formats × {} scales)",
        total,
        ALL_PRIMITIVES.len(),
        ALL_FORMATS.len(),
        scales.len(),
    );

    if let Ok(path) = recorder.save() {
        eprintln!("  Results saved to: {}", path.display());
    }
}
