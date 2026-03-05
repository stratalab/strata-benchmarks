//! Memory profiling: Graph storage vs KV storage.
//!
//! Test 1: Put ~1 GB of raw KV data into Strata, measure RSS.
//! Test 2: Load graph500-22 (1.2 GB raw) into Strata graph, measure RSS.
//!
//! Run: `cargo bench --bench mem_profile`
//! KV only: `cargo bench --bench mem_profile -- --kv-only`
//! Graph only: `cargo bench --bench mem_profile -- --graph-only`

#[allow(unused)]
#[path = "../harness/mod.rs"]
mod harness;

#[allow(unused)]
mod ldbc;

use harness::{create_db, print_hardware_info, DurabilityConfig};
use std::path::PathBuf;
use stratadb::{DistanceMetric, Strata};

const RSS_LIMIT_MB: f64 = 50_000.0;

fn rss_mb() -> f64 {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<f64>() {
                        return kb / 1024.0;
                    }
                }
            }
        }
    }
    0.0
}

fn check_rss(stage: &str) -> f64 {
    let rss = rss_mb();
    eprintln!("  RSS: {:.0} MB  [{}]", rss, stage);
    if rss > RSS_LIMIT_MB {
        eprintln!(
            "ABORTING: RSS {:.0} MB exceeds {:.0} MB limit at: {}",
            rss, RSS_LIMIT_MB, stage
        );
        std::process::exit(1);
    }
    rss
}

/// Put ~1 GB of raw KV data into Strata.
/// Uses 1,000,000 keys × 1,000-byte values = ~1 GB raw payload.
fn test_kv() {
    eprintln!("\n========================================");
    eprintln!("  TEST: KV Storage (~1 GB raw data)");
    eprintln!("========================================");

    const NUM_KEYS: usize = 1_000_000;
    const VALUE_SIZE: usize = 1_000;
    const BATCH_SIZE: usize = 100_000;

    let raw_bytes = NUM_KEYS * (VALUE_SIZE + 10); // ~10 bytes per key name
    eprintln!(
        "Plan: {} keys × {} byte values = {:.0} MB raw data",
        NUM_KEYS,
        VALUE_SIZE,
        raw_bytes as f64 / 1_048_576.0
    );

    check_rss("kv: start");

    let db = create_db(DurabilityConfig::Cache);
    check_rss("kv: after DB create");

    let value = stratadb::Value::Bytes(vec![0x42; VALUE_SIZE]);
    let t0 = std::time::Instant::now();

    for batch in 0..(NUM_KEYS / BATCH_SIZE) {
        let start_key = batch * BATCH_SIZE;
        let end_key = start_key + BATCH_SIZE;
        for i in start_key..end_key {
            let key = format!("key-{:07}", i);
            db.db.kv_put(&key, value.clone()).unwrap();
        }
        check_rss(&format!(
            "kv: after {} keys ({:.0} MB raw)",
            end_key,
            (end_key * (VALUE_SIZE + 10)) as f64 / 1_048_576.0
        ));
    }

    let elapsed = t0.elapsed();
    let final_rss = check_rss("kv: done");

    eprintln!("\n--- KV Summary ---");
    eprintln!("Keys:       {}", NUM_KEYS);
    eprintln!("Raw data:   {:.0} MB", raw_bytes as f64 / 1_048_576.0);
    eprintln!("RSS used:   {:.0} MB", final_rss);
    eprintln!("Blowup:     {:.1}x", final_rss / (raw_bytes as f64 / 1_048_576.0));
    eprintln!("Time:       {:.2}s", elapsed.as_secs_f64());
    eprintln!(
        "Throughput: {:.0} keys/s",
        NUM_KEYS as f64 / elapsed.as_secs_f64()
    );
}

/// Load graph500-22 into Strata graph layer.
/// Uses a persistent DB at data/graph/graph500-22/strata-db/ to avoid
/// re-loading 64M edges (691s) on every run.
fn test_graph() {
    eprintln!("\n========================================");
    eprintln!("  TEST: Graph Storage (graph500-22)");
    eprintln!("========================================");

    let dataset_path = PathBuf::from("data/graph/graph500-22");
    let db_path = dataset_path.join("strata-db");

    let t0 = std::time::Instant::now();
    let dataset = ldbc::LdbcDataset::load(&dataset_path).expect("failed to load dataset");
    let parse_elapsed = t0.elapsed();

    let raw_bytes_edges = dataset.edges.len() * 16; // 2 × u64 per edge
    let raw_bytes_verts = dataset.vertices.len() * 8; // u64 per vertex
    let raw_total = raw_bytes_edges + raw_bytes_verts;

    eprintln!(
        "Parsed: {} vertices, {} edges in {:.2}s ({:.0} MB raw)",
        dataset.vertices.len(),
        dataset.edges.len(),
        parse_elapsed.as_secs_f64(),
        raw_total as f64 / 1_048_576.0
    );
    check_rss("graph: after parse");

    print_hardware_info();

    // Try to open existing DB, or create fresh
    let db = if db_path.exists() {
        eprintln!("Opening existing Strata DB at {}", db_path.display());
        Strata::open(&db_path).expect("failed to open existing strata DB")
    } else {
        eprintln!("Creating new Strata DB at {}", db_path.display());
        std::fs::create_dir_all(&db_path).expect("failed to create DB directory");
        Strata::open(&db_path).expect("failed to create strata DB")
    };
    let rss_before = check_rss("graph: after DB open");

    // Check if graph already loaded
    let graph_exists = db.graph_list().map(|g| g.contains(&"ldbc".to_string())).unwrap_or(false);

    if graph_exists {
        eprintln!("Graph 'ldbc' already exists — skipping load");
        check_rss("graph: after DB open (cached)");
    } else {
        eprintln!("Loading graph into Strata...");
        db.graph_create("ldbc").expect("graph_create failed");

        const NODE_BATCH: usize = 500_000;
        const EDGE_BATCH: usize = 4_000_000;

        let t1 = std::time::Instant::now();

        // Nodes
        let total_nodes = dataset.vertices.len();
        for (i, chunk) in dataset.vertices.chunks(NODE_BATCH).enumerate() {
            let node_ids: Vec<String> = chunk.iter().map(|v| v.to_string()).collect();
            let nodes: Vec<(&str, Option<&str>, Option<stratadb::Value>)> =
                node_ids.iter().map(|id| (id.as_str(), None, None)).collect();
            db.graph_bulk_insert("ldbc", &nodes, &[]).unwrap();
            let done = ((i + 1) * NODE_BATCH).min(total_nodes);
            check_rss(&format!("graph: nodes {}/{}", done, total_nodes));
        }

        // Edges — batch in 4M-edge chunks to limit peak string allocations
        let total_edges = dataset.edges.len();
        let mut edges_done = 0usize;
        for edge_chunk in dataset.edges.chunks(EDGE_BATCH) {
            let edge_strs: Vec<(String, String)> = edge_chunk
                .iter()
                .map(|(s, d)| (s.to_string(), d.to_string()))
                .collect();
            let base_idx = edges_done;
            let edges: Vec<(&str, &str, &str, Option<f64>, Option<stratadb::Value>)> = edge_strs
                .iter()
                .enumerate()
                .map(|(j, (s, d))| {
                    let w = dataset.edge_weights.as_ref().map(|ws| ws[base_idx + j]);
                    (s.as_str(), d.as_str(), "E", w, None)
                })
                .collect();
            db.graph_bulk_insert("ldbc", &[], &edges).unwrap();
            edges_done += edge_chunk.len();
            check_rss(&format!("graph: edges {}/{}", edges_done, total_edges));
        }

        let load_elapsed = t1.elapsed();
        let final_rss = check_rss("graph: done");
        let graph_rss = final_rss - rss_before;

        eprintln!("\n--- Graph Load Summary ---");
        eprintln!("Vertices:   {}", dataset.vertices.len());
        eprintln!("Edges:      {}", dataset.edges.len());
        eprintln!("Raw data:   {:.0} MB", raw_total as f64 / 1_048_576.0);
        eprintln!("RSS used:   {:.0} MB (graph layer only: ~{:.0} MB)", final_rss, graph_rss);
        eprintln!("Blowup:     {:.1}x", final_rss / (raw_total as f64 / 1_048_576.0));
        eprintln!("Load time:  {:.2}s", load_elapsed.as_secs_f64());
    }

    // --- Run all LDBC Graphalytics algorithms ---
    eprintln!("\n========================================");
    eprintln!("  LDBC Graphalytics Algorithms");
    eprintln!("========================================");

    // 1. BFS
    let source = dataset.bfs_source.expect("no BFS source vertex");
    eprintln!("\n--- BFS (source: {}) ---", source);
    let t = std::time::Instant::now();
    let bfs_result = db
        .graph_bfs("ldbc", &source.to_string(), usize::MAX, None, None, Some("both"))
        .expect("graph_bfs failed");
    let bfs_elapsed = t.elapsed();
    eprintln!("  Visited:  {} vertices", bfs_result.depths.len());
    eprintln!("  Time:     {:.2}s", bfs_elapsed.as_secs_f64());
    let evps = (dataset.vertices.len() + dataset.edges.len()) as f64 / bfs_elapsed.as_secs_f64();
    eprintln!("  EVPS:     {:.0}", evps);
    check_rss("after BFS");

    // 2. WCC
    eprintln!("\n--- WCC ---");
    let t = std::time::Instant::now();
    let wcc_result = db.graph_wcc("ldbc").expect("graph_wcc failed");
    let wcc_elapsed = t.elapsed();
    let num_components: std::collections::HashSet<_> = wcc_result.result.values().collect();
    eprintln!("  Components: {}", num_components.len());
    eprintln!("  Vertices:   {}", wcc_result.result.len());
    eprintln!("  Time:       {:.2}s", wcc_elapsed.as_secs_f64());
    check_rss("after WCC");

    // 3. CDLP
    let max_iterations = dataset.cdlp_max_iterations.unwrap_or(10);
    eprintln!("\n--- CDLP (max_iterations: {}) ---", max_iterations);
    let t = std::time::Instant::now();
    let cdlp_result = db
        .graph_cdlp("ldbc", max_iterations, Some("both"))
        .expect("graph_cdlp failed");
    let cdlp_elapsed = t.elapsed();
    let num_labels: std::collections::HashSet<_> = cdlp_result.result.values().collect();
    eprintln!("  Labels:   {}", num_labels.len());
    eprintln!("  Vertices: {}", cdlp_result.result.len());
    eprintln!("  Time:     {:.2}s", cdlp_elapsed.as_secs_f64());
    check_rss("after CDLP");

    // 4. PageRank
    let damping = dataset.pr_damping_factor.unwrap_or(0.85);
    let iterations = dataset.pr_num_iterations.unwrap_or(10);
    eprintln!("\n--- PageRank (damping: {}, iterations: {}) ---", damping, iterations);
    let t = std::time::Instant::now();
    let pr_result = db
        .graph_pagerank("ldbc", Some(damping), Some(iterations), None)
        .expect("graph_pagerank failed");
    let pr_elapsed = t.elapsed();
    let max_rank = pr_result.result.values().cloned().fold(0.0_f64, f64::max);
    eprintln!("  Vertices: {}", pr_result.result.len());
    eprintln!("  Max rank: {:.6}", max_rank);
    eprintln!("  Time:     {:.2}s", pr_elapsed.as_secs_f64());
    check_rss("after PageRank");

    // 5. LCC
    eprintln!("\n--- LCC ---");
    eprintln!("  (O(V*d^2) — may be slow on dense graphs)");
    let t = std::time::Instant::now();
    let lcc_result = db.graph_lcc("ldbc").expect("graph_lcc failed");
    let lcc_elapsed = t.elapsed();
    let nonzero = lcc_result.result.values().filter(|&&v| v > 0.0).count();
    eprintln!("  Vertices: {}", lcc_result.result.len());
    eprintln!("  Non-zero: {}", nonzero);
    eprintln!("  Time:     {:.2}s", lcc_elapsed.as_secs_f64());
    check_rss("after LCC");

    // Summary
    eprintln!("\n========================================");
    eprintln!("  Algorithm Summary");
    eprintln!("========================================");
    eprintln!("  BFS:      {:.2}s  ({} vertices reached)", bfs_elapsed.as_secs_f64(), bfs_result.depths.len());
    eprintln!("  WCC:      {:.2}s  ({} components)", wcc_elapsed.as_secs_f64(), num_components.len());
    eprintln!("  CDLP:     {:.2}s  ({} labels)", cdlp_elapsed.as_secs_f64(), num_labels.len());
    eprintln!("  PageRank: {:.2}s", pr_elapsed.as_secs_f64());
    eprintln!("  LCC:      {:.2}s", lcc_elapsed.as_secs_f64());
    eprintln!("  Peak RSS: {:.0} MB", rss_mb());
}

/// Insert vectors into Strata and measure RSS growth.
/// Tests 100K and 500K vectors at 128d.
fn test_vector() {
    eprintln!("\n========================================");
    eprintln!("  TEST: Vector Storage (128d, cosine)");
    eprintln!("========================================");

    const DIM: usize = 128;
    const SCALES: &[usize] = &[100_000, 500_000];
    const BATCH_SIZE: usize = 50_000;

    for &total in SCALES {
        let raw_bytes = total * DIM * 4; // f32 = 4 bytes
        eprintln!(
            "\nPlan: {} vectors x {}d x 4 bytes = {:.0} MB raw data",
            total,
            DIM,
            raw_bytes as f64 / 1_048_576.0
        );

        let rss_before = check_rss(&format!("vector-{}k: start", total / 1000));

        let db = create_db(DurabilityConfig::Cache);
        db.db
            .vector_create_collection("vec_bench", DIM as u64, DistanceMetric::Cosine)
            .unwrap();
        check_rss(&format!("vector-{}k: after DB create", total / 1000));

        let t0 = std::time::Instant::now();

        let mut inserted = 0;
        while inserted < total {
            let end = (inserted + BATCH_SIZE).min(total);
            for i in inserted..end {
                let vec = harness::vector_128d(i as u64);
                db.db
                    .vector_upsert("vec_bench", &format!("vec_{}", i), vec, None)
                    .unwrap();
            }
            inserted = end;
            check_rss(&format!(
                "vector-{}k: after {} vectors",
                total / 1000,
                inserted
            ));
        }

        let elapsed = t0.elapsed();
        let final_rss = check_rss(&format!("vector-{}k: done", total / 1000));

        eprintln!("\n--- Vector Summary ({} vectors) ---", total);
        eprintln!("Vectors:    {}", total);
        eprintln!("Dimensions: {}", DIM);
        eprintln!("Raw data:   {:.0} MB", raw_bytes as f64 / 1_048_576.0);
        eprintln!("RSS used:   {:.0} MB", final_rss);
        eprintln!(
            "Blowup:     {:.1}x",
            final_rss / (raw_bytes as f64 / 1_048_576.0)
        );
        eprintln!("Time:       {:.2}s", elapsed.as_secs_f64());
        eprintln!(
            "Throughput: {:.0} vectors/s",
            total as f64 / elapsed.as_secs_f64()
        );
        let _ = rss_before; // suppress unused warning
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kv_only = args.iter().any(|a| a == "--kv-only");
    let graph_only = args.iter().any(|a| a == "--graph-only");
    let vector_only = args.iter().any(|a| a == "--vector-only");

    eprintln!("=== Strata Memory Profile (limit: {:.0} GB) ===", RSS_LIMIT_MB / 1024.0);

    if !graph_only && !vector_only {
        test_kv();
    }

    if !kv_only && !vector_only {
        test_graph();
    }

    if !kv_only && !graph_only {
        test_vector();
    }
}
