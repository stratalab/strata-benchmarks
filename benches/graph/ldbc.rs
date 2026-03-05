//! LDBC Graphalytics dataset parser.
//!
//! Parses the standard LDBC file formats:
//! - `.v`  — one vertex ID (u64) per line
//! - `.e`  — `src dst [weight]` per line (space-separated, optional 3rd column)
//! - `.properties` — Java properties format with graph metadata
//! - BFS reference — `vertex_id depth` per line
//! - U64 reference — `vertex_id u64_value` per line (WCC, CDLP)
//! - F64 reference — `vertex_id f64_value` per line (PR, LCC, SSSP)

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::time::Duration;

use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;

/// Sentinel value for unreachable vertices in LDBC BFS output.
pub const UNREACHABLE: i64 = 9223372036854775807; // i64::MAX

/// An LDBC Graphalytics dataset (vertices + edges + metadata).
pub struct LdbcDataset {
    pub vertices: Vec<u64>,
    pub edges: Vec<(u64, u64)>,
    pub directed: bool,
    pub name: String,
    pub bfs_source: Option<u64>,
    pub edge_weights: Option<Vec<f64>>,
    pub sssp_source: Option<u64>,
    pub cdlp_max_iterations: Option<usize>,
    pub pr_damping_factor: Option<f64>,
    pub pr_num_iterations: Option<usize>,
}

/// BFS reference output for validation.
pub struct BfsReference {
    pub source: u64,
    pub depths: HashMap<u64, i64>, // i64 to hold UNREACHABLE sentinel
}

impl LdbcDataset {
    /// Load an LDBC dataset from a directory.
    ///
    /// Expects files named `<name>.v`, `<name>.e`, and optionally `<name>.properties`
    /// where `<name>` is the directory's basename.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| "invalid dataset directory name".to_string())?
            .to_string();

        let v_path = dir.join(format!("{}.v", name));
        let e_path = dir.join(format!("{}.e", name));
        let props_path = dir.join(format!("{}.properties", name));

        // Parse vertices
        let v_content = std::fs::read_to_string(&v_path)
            .map_err(|e| format!("failed to read {}: {}", v_path.display(), e))?;
        let vertices: Vec<u64> = v_content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                l.trim()
                    .parse::<u64>()
                    .map_err(|e| format!("bad vertex id '{}': {}", l.trim(), e))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Parse edges (2-column unweighted or 3-column weighted)
        let e_content = std::fs::read_to_string(&e_path)
            .map_err(|e| format!("failed to read {}: {}", e_path.display(), e))?;
        let mut edges: Vec<(u64, u64)> = Vec::new();
        let mut edge_weights: Vec<f64> = Vec::new();
        let mut has_weights = false;

        for l in e_content.lines() {
            let l = l.trim();
            if l.is_empty() {
                continue;
            }
            let parts: Vec<&str> = l.split_whitespace().collect();
            match parts.len() {
                2 => {
                    let src = parts[0]
                        .parse::<u64>()
                        .map_err(|e| format!("bad edge src '{}': {}", parts[0], e))?;
                    let dst = parts[1]
                        .parse::<u64>()
                        .map_err(|e| format!("bad edge dst '{}': {}", parts[1], e))?;
                    edges.push((src, dst));
                    edge_weights.push(1.0);
                }
                3 => {
                    let src = parts[0]
                        .parse::<u64>()
                        .map_err(|e| format!("bad edge src '{}': {}", parts[0], e))?;
                    let dst = parts[1]
                        .parse::<u64>()
                        .map_err(|e| format!("bad edge dst '{}': {}", parts[1], e))?;
                    let w = parts[2]
                        .parse::<f64>()
                        .map_err(|e| format!("bad edge weight '{}': {}", parts[2], e))?;
                    edges.push((src, dst));
                    edge_weights.push(w);
                    has_weights = true;
                }
                _ => return Err(format!("bad edge line: '{}'", l)),
            }
        }

        let edge_weights = if has_weights {
            Some(edge_weights)
        } else {
            None
        };

        // Parse properties (optional)
        // Supports both simple format (example-directed) and fully-qualified
        // format (graph.<name>.<key>) used by real LDBC datasets.
        let mut directed = true;
        let mut bfs_source = None;
        let mut expected_vertices: Option<usize> = None;
        let mut expected_edges: Option<usize> = None;
        let mut sssp_source = None;
        let mut cdlp_max_iterations = None;
        let mut pr_damping_factor = None;
        let mut pr_num_iterations = None;

        if props_path.exists() {
            let props_content = std::fs::read_to_string(&props_path)
                .map_err(|e| format!("failed to read {}: {}", props_path.display(), e))?;

            // Prefix to strip for fully-qualified keys: "graph.<name>."
            let fq_prefix = format!("graph.{}.", name);

            for line in props_content.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if let Some((raw_key, value)) = line.split_once('=') {
                    let raw_key = raw_key.trim();
                    let value = value.trim();

                    // Normalize: strip fully-qualified prefix if present
                    let key = raw_key
                        .strip_prefix(&fq_prefix)
                        .unwrap_or(raw_key);

                    match key {
                        "graph.directed" | "directed" => directed = value == "true",
                        "meta.vertices" => expected_vertices = value.parse().ok(),
                        "meta.edges" => expected_edges = value.parse().ok(),
                        "algorithms.bfs.source-vertex" | "bfs.source-vertex" => {
                            bfs_source = value.parse().ok();
                        }
                        "cdlp.max-iterations" => {
                            cdlp_max_iterations = value.parse().ok();
                        }
                        "pr.damping-factor" => {
                            pr_damping_factor = value.parse().ok();
                        }
                        "pr.num-iterations" => {
                            pr_num_iterations = value.parse().ok();
                        }
                        "sssp.source-vertex" => {
                            sssp_source = value.parse().ok();
                        }
                        _ => {}
                    }
                }
            }
        }

        // Validate counts if properties file provided them
        if let Some(ev) = expected_vertices {
            if vertices.len() != ev {
                return Err(format!(
                    "vertex count mismatch: file has {}, properties says {}",
                    vertices.len(),
                    ev
                ));
            }
        }
        if let Some(ee) = expected_edges {
            if edges.len() != ee {
                return Err(format!(
                    "edge count mismatch: file has {}, properties says {}",
                    edges.len(),
                    ee
                ));
            }
        }

        Ok(LdbcDataset {
            vertices,
            edges,
            directed,
            name,
            bfs_source,
            edge_weights,
            sssp_source,
            cdlp_max_iterations,
            pr_damping_factor,
            pr_num_iterations,
        })
    }

    /// Build a petgraph undirected graph from this dataset.
    ///
    /// Returns the graph and a mapping from LDBC vertex ID to petgraph NodeIndex.
    pub fn to_petgraph(&self) -> (UnGraph<(), ()>, HashMap<u64, NodeIndex>) {
        let mut graph = UnGraph::new_undirected();
        let mut id_map: HashMap<u64, NodeIndex> = HashMap::with_capacity(self.vertices.len());

        for &vid in &self.vertices {
            let idx = graph.add_node(());
            id_map.insert(vid, idx);
        }

        for &(src, dst) in &self.edges {
            if let (Some(&si), Some(&di)) = (id_map.get(&src), id_map.get(&dst)) {
                graph.add_edge(si, di, ());
            }
        }

        (graph, id_map)
    }

    /// Build a petgraph undirected graph with f64 edge weights from this dataset.
    ///
    /// Uses `self.edge_weights` if present, defaulting to 1.0 for each edge.
    /// Returns the graph, vertex-ID-to-NodeIndex map, and reverse map.
    pub fn to_petgraph_weighted(
        &self,
    ) -> (
        petgraph::Graph<(), f64, petgraph::Undirected>,
        HashMap<u64, NodeIndex>,
    ) {
        let mut graph = petgraph::Graph::new_undirected();
        let mut id_map: HashMap<u64, NodeIndex> = HashMap::with_capacity(self.vertices.len());

        for &vid in &self.vertices {
            let idx = graph.add_node(());
            id_map.insert(vid, idx);
        }

        let weights = self.edge_weights.as_deref();
        for (i, &(src, dst)) in self.edges.iter().enumerate() {
            if let (Some(&si), Some(&di)) = (id_map.get(&src), id_map.get(&dst)) {
                let w = weights.map_or(1.0, |ws| ws[i]);
                graph.add_edge(si, di, w);
            }
        }

        (graph, id_map)
    }

    /// Build a petgraph directed graph from this dataset.
    ///
    /// Used for algorithms that respect edge direction (PageRank, SSSP).
    pub fn to_petgraph_directed(
        &self,
    ) -> (petgraph::Graph<(), (), petgraph::Directed>, HashMap<u64, NodeIndex>) {
        let mut graph = petgraph::Graph::new();
        let mut id_map: HashMap<u64, NodeIndex> = HashMap::with_capacity(self.vertices.len());

        for &vid in &self.vertices {
            let idx = graph.add_node(());
            id_map.insert(vid, idx);
        }

        for &(src, dst) in &self.edges {
            if let (Some(&si), Some(&di)) = (id_map.get(&src), id_map.get(&dst)) {
                graph.add_edge(si, di, ());
            }
        }

        (graph, id_map)
    }

    /// Build a petgraph directed graph with f64 edge weights from this dataset.
    ///
    /// Used for weighted directed algorithms (SSSP).
    pub fn to_petgraph_weighted_directed(
        &self,
    ) -> (petgraph::Graph<(), f64, petgraph::Directed>, HashMap<u64, NodeIndex>) {
        let mut graph = petgraph::Graph::new();
        let mut id_map: HashMap<u64, NodeIndex> = HashMap::with_capacity(self.vertices.len());

        for &vid in &self.vertices {
            let idx = graph.add_node(());
            id_map.insert(vid, idx);
        }

        let weights = self.edge_weights.as_deref();
        for (i, &(src, dst)) in self.edges.iter().enumerate() {
            if let (Some(&si), Some(&di)) = (id_map.get(&src), id_map.get(&dst)) {
                let w = weights.map_or(1.0, |ws| ws[i]);
                graph.add_edge(si, di, w);
            }
        }

        (graph, id_map)
    }
}

/// Build a reverse mapping from NodeIndex to LDBC vertex ID.
pub fn reverse_id_map(id_map: &HashMap<u64, NodeIndex>) -> HashMap<NodeIndex, u64> {
    id_map.iter().map(|(&vid, &idx)| (idx, vid)).collect()
}

/// Build a set of directed edges as (NodeIndex, NodeIndex) pairs.
///
/// Used for directed LCC computation.
pub fn directed_edge_set(
    dataset: &LdbcDataset,
    id_map: &HashMap<u64, NodeIndex>,
) -> HashSet<(NodeIndex, NodeIndex)> {
    dataset
        .edges
        .iter()
        .filter_map(|&(src, dst)| {
            let si = id_map.get(&src)?;
            let di = id_map.get(&dst)?;
            Some((*si, *di))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Reference file types
// ---------------------------------------------------------------------------

/// Reference output with u64 values (WCC component IDs, CDLP labels).
pub struct U64Reference {
    pub values: HashMap<u64, u64>,
}

impl U64Reference {
    /// Load a 2-column reference file: `vertex_id u64_value` per line.
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;

        let mut values = HashMap::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                return Err(format!("bad reference line: '{}'", line));
            }
            let vid = parts[0]
                .parse::<u64>()
                .map_err(|e| format!("bad vertex id '{}': {}", parts[0], e))?;
            let val = parts[1]
                .parse::<u64>()
                .map_err(|e| format!("bad u64 value '{}': {}", parts[1], e))?;
            values.insert(vid, val);
        }

        Ok(U64Reference { values })
    }
}

/// Reference output with f64 values (PageRank scores, LCC coefficients, SSSP distances).
pub struct F64Reference {
    pub values: HashMap<u64, f64>,
}

impl F64Reference {
    /// Load a 2-column reference file: `vertex_id f64_value` per line.
    /// Handles scientific notation (e.g., `3.911456800211408e-07`).
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;

        let mut values = HashMap::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                return Err(format!("bad reference line: '{}'", line));
            }
            let vid = parts[0]
                .parse::<u64>()
                .map_err(|e| format!("bad vertex id '{}': {}", parts[0], e))?;
            let val = parts[1]
                .parse::<f64>()
                .map_err(|e| format!("bad f64 value '{}': {}", parts[1], e))?;
            values.insert(vid, val);
        }

        Ok(F64Reference { values })
    }
}

// ---------------------------------------------------------------------------
// Petgraph algorithm implementations
// ---------------------------------------------------------------------------

/// Run BFS on a petgraph graph using a manual VecDeque-based traversal.
///
/// Returns a map from NodeIndex to BFS depth (0 for source).
pub fn petgraph_bfs(graph: &UnGraph<(), ()>, source: NodeIndex) -> HashMap<NodeIndex, usize> {
    let mut depths: HashMap<NodeIndex, usize> = HashMap::with_capacity(graph.node_count());
    let mut queue = VecDeque::new();

    depths.insert(source, 0);
    queue.push_back(source);

    while let Some(node) = queue.pop_front() {
        let d = depths[&node];
        for neighbor in graph.neighbors(node) {
            if !depths.contains_key(&neighbor) {
                depths.insert(neighbor, d + 1);
                queue.push_back(neighbor);
            }
        }
    }

    depths
}

/// Weakly Connected Components using BFS flood-fill.
///
/// Component ID = minimum LDBC vertex ID in the component (LDBC convention).
/// Returns a map from NodeIndex to component ID.
pub fn petgraph_wcc(
    graph: &UnGraph<(), ()>,
    reverse_map: &HashMap<NodeIndex, u64>,
) -> HashMap<NodeIndex, u64> {
    let mut component: HashMap<NodeIndex, u64> = HashMap::with_capacity(graph.node_count());
    let mut visited: HashSet<NodeIndex> = HashSet::with_capacity(graph.node_count());

    for node in graph.node_indices() {
        if visited.contains(&node) {
            continue;
        }

        // BFS to find all nodes in this component
        let mut queue = VecDeque::new();
        let mut members = Vec::new();
        queue.push_back(node);
        visited.insert(node);

        while let Some(current) = queue.pop_front() {
            members.push(current);
            for neighbor in graph.neighbors(current) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        // Component ID = minimum LDBC vertex ID in the component
        let comp_id = members
            .iter()
            .map(|&n| reverse_map[&n])
            .min()
            .unwrap();

        for &m in &members {
            component.insert(m, comp_id);
        }
    }

    component
}

/// Community Detection via Label Propagation (CDLP).
///
/// Synchronous/simultaneous update: in each iteration, every vertex adopts
/// the most frequent label among its neighbors (tie-break: smallest label).
/// Initial label = LDBC vertex ID.
pub fn petgraph_cdlp(
    graph: &UnGraph<(), ()>,
    reverse_map: &HashMap<NodeIndex, u64>,
    max_iterations: usize,
) -> HashMap<NodeIndex, u64> {
    let mut labels: HashMap<NodeIndex, u64> = graph
        .node_indices()
        .map(|n| (n, reverse_map[&n]))
        .collect();

    for _ in 0..max_iterations {
        let mut new_labels: HashMap<NodeIndex, u64> =
            HashMap::with_capacity(graph.node_count());

        for node in graph.node_indices() {
            let neighbors: Vec<_> = graph.neighbors(node).collect();
            if neighbors.is_empty() {
                // Isolated vertex keeps its own label
                new_labels.insert(node, labels[&node]);
                continue;
            }

            // Count label frequencies among neighbors
            let mut freq: HashMap<u64, usize> = HashMap::new();
            for &nbr in &neighbors {
                *freq.entry(labels[&nbr]).or_insert(0) += 1;
            }

            // Find max frequency, tie-break by smallest label
            let max_count = *freq.values().max().unwrap();
            let best_label = freq
                .iter()
                .filter(|&(_, &count)| count == max_count)
                .map(|(&label, _)| label)
                .min()
                .unwrap();

            new_labels.insert(node, best_label);
        }

        labels = new_labels;
    }

    labels
}

/// PageRank with fixed iterations on a directed graph (LDBC Graphalytics specification).
///
/// `PR(v) = (1 - d) / |V| + d * Σ(PR(u) / out_deg(u))` for each u with edge u→v.
/// Dangling nodes (out_deg=0) redistribute their rank uniformly to all nodes.
/// Uses the specified number of iterations with no convergence check.
pub fn petgraph_pagerank(
    graph: &petgraph::Graph<(), (), petgraph::Directed>,
    damping: f64,
    iterations: usize,
) -> HashMap<NodeIndex, f64> {
    let n = graph.node_count() as f64;
    let base = (1.0 - damping) / n;

    // Pre-compute out-degrees
    let out_deg: HashMap<NodeIndex, usize> = graph
        .node_indices()
        .map(|node| {
            let deg = graph
                .edges_directed(node, petgraph::Direction::Outgoing)
                .count();
            (node, deg)
        })
        .collect();

    let mut rank: HashMap<NodeIndex, f64> = graph
        .node_indices()
        .map(|node| (node, 1.0 / n))
        .collect();

    for _ in 0..iterations {
        let mut contributions: HashMap<NodeIndex, f64> =
            graph.node_indices().map(|n| (n, 0.0)).collect();

        // Sum of rank from dangling nodes (out_deg=0)
        let dangling_sum: f64 = graph
            .node_indices()
            .filter(|node| out_deg[node] == 0)
            .map(|node| rank[&node])
            .sum();

        // Each node distributes its rank equally across its out-edges
        for node in graph.node_indices() {
            let deg = out_deg[&node];
            if deg == 0 {
                continue;
            }
            let share = rank[&node] / deg as f64;
            for edge in graph.edges_directed(node, petgraph::Direction::Outgoing) {
                *contributions.get_mut(&edge.target()).unwrap() += share;
            }
        }

        // Dangling nodes redistribute their rank uniformly to all nodes
        let dangling_share = dangling_sum / n;

        let mut new_rank: HashMap<NodeIndex, f64> = HashMap::with_capacity(graph.node_count());
        for node in graph.node_indices() {
            new_rank.insert(
                node,
                base + damping * (contributions[&node] + dangling_share),
            );
        }

        rank = new_rank;
    }

    rank
}

/// Local Clustering Coefficient.
///
/// For undirected graphs:
///   LCC(v) = 2 * T(v) / (d(v) * (d(v) - 1))
///   where T(v) = number of undirected edges among neighbors.
///
/// For directed graphs (LDBC Graphalytics spec):
///   LCC(v) = T_dir(v) / (d(v) * (d(v) - 1))
///   where T_dir(v) = number of directed edges (u,w) among neighbors,
///   N(v) = undirected neighbors, d(v) = |N(v)|.
///
/// `directed_edges` should be Some with the original directed edge set for
/// directed graphs, or None for undirected graphs.
pub fn petgraph_lcc(
    graph: &UnGraph<(), ()>,
    directed_edges: Option<&HashSet<(NodeIndex, NodeIndex)>>,
) -> HashMap<NodeIndex, f64> {
    let mut result: HashMap<NodeIndex, f64> = HashMap::with_capacity(graph.node_count());

    // Pre-compute deduplicated neighbor sets
    let neighbor_sets: HashMap<NodeIndex, HashSet<NodeIndex>> = graph
        .node_indices()
        .map(|n| (n, graph.neighbors(n).collect()))
        .collect();

    for node in graph.node_indices() {
        let neighbors = &neighbor_sets[&node];
        let deg = neighbors.len();
        if deg < 2 {
            result.insert(node, 0.0);
            continue;
        }

        let nbr_vec: Vec<NodeIndex> = neighbors.iter().copied().collect();

        if let Some(dir_edges) = directed_edges {
            // Directed: count directed edges (u→w) between neighbor pairs
            let mut dir_count: usize = 0;
            for i in 0..nbr_vec.len() {
                for j in 0..nbr_vec.len() {
                    if i != j && dir_edges.contains(&(nbr_vec[i], nbr_vec[j])) {
                        dir_count += 1;
                    }
                }
            }
            let max_directed = deg * (deg - 1);
            result.insert(node, dir_count as f64 / max_directed as f64);
        } else {
            // Undirected: count undirected edges among neighbors
            let mut triangles: usize = 0;
            for i in 0..nbr_vec.len() {
                for j in (i + 1)..nbr_vec.len() {
                    if neighbor_sets[&nbr_vec[i]].contains(&nbr_vec[j]) {
                        triangles += 1;
                    }
                }
            }
            let max_pairs = deg * (deg - 1) / 2;
            result.insert(node, triangles as f64 / max_pairs as f64);
        }
    }

    result
}

/// Single-Source Shortest Path using Dijkstra on an undirected weighted graph.
///
/// Returns a map from NodeIndex to shortest distance from source.
pub fn petgraph_sssp(
    graph: &petgraph::Graph<(), f64, petgraph::Undirected>,
    source: NodeIndex,
) -> HashMap<NodeIndex, f64> {
    petgraph::algo::dijkstra(graph, source, None, |e| *e.weight())
}

/// Single-Source Shortest Path using Dijkstra on a directed weighted graph.
///
/// Returns a map from NodeIndex to shortest distance from source.
pub fn petgraph_sssp_directed(
    graph: &petgraph::Graph<(), f64, petgraph::Directed>,
    source: NodeIndex,
) -> HashMap<NodeIndex, f64> {
    petgraph::algo::dijkstra(graph, source, None, |e| *e.weight())
}

// ---------------------------------------------------------------------------
// Shared benchmark scaffolding
// ---------------------------------------------------------------------------

pub const DEFAULT_RUNS: usize = 10;

pub fn default_dataset_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/graph/example-directed")
}

pub struct Config {
    pub dataset: PathBuf,
    pub runs: usize,
    pub no_validate: bool,
    pub csv: bool,
    pub quiet: bool,
}

pub fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut config = Config {
        dataset: default_dataset_dir(),
        runs: DEFAULT_RUNS,
        no_validate: false,
        csv: false,
        quiet: false,
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
            "--runs" => {
                i += 1;
                if i < args.len() {
                    config.runs = args[i].parse::<usize>().unwrap_or(DEFAULT_RUNS).max(1);
                }
            }
            "--no-validate" => config.no_validate = true,
            "--csv" => config.csv = true,
            "-q" => config.quiet = true,
            _ => {}
        }
        i += 1;
    }

    config
}

pub fn fmt_num(n: u64) -> String {
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

pub fn fmt_ms(d: Duration) -> String {
    format!("{:.1}ms", d.as_secs_f64() * 1000.0)
}

pub struct RunStats {
    pub avg: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub min: Duration,
    pub max: Duration,
    pub avg_evps: f64,
    pub count: usize,
}

pub fn compute_stats(times: &mut Vec<Duration>, total_elements: f64) -> RunStats {
    assert!(!times.is_empty());
    times.sort_unstable();
    let len = times.len();
    let sum: Duration = times.iter().sum();
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

/// Load an LDBC dataset into a Strata graph using batched bulk insert.
///
/// Inserts nodes and edges in chunks to avoid OOM on large graphs.
/// The old approach materialised all 64M edges as `(String, String)` at once,
/// requiring ~15 GB before Strata even started; batching keeps peak overhead
/// proportional to `EDGE_BATCH_SIZE`.
pub fn load_graph_into_strata(strata: &stratadb::Strata, dataset: &LdbcDataset) -> Duration {
    const NODE_BATCH_SIZE: usize = 500_000;
    const EDGE_BATCH_SIZE: usize = 1_000_000;

    let start = std::time::Instant::now();
    strata.graph_create("ldbc").expect("graph_create failed");

    // --- nodes (batched) ---
    for chunk in dataset.vertices.chunks(NODE_BATCH_SIZE) {
        let node_ids: Vec<String> = chunk.iter().map(|v| v.to_string()).collect();
        let nodes: Vec<(&str, Option<&str>, Option<stratadb::Value>)> =
            node_ids.iter().map(|id| (id.as_str(), None, None)).collect();
        strata
            .graph_bulk_insert("ldbc", &nodes, &[])
            .expect("graph_bulk_insert (nodes) failed");
    }

    // --- edges (batched) ---
    for (batch_start, chunk) in dataset.edges.chunks(EDGE_BATCH_SIZE).enumerate() {
        let offset = batch_start * EDGE_BATCH_SIZE;
        let edge_strs: Vec<(String, String)> = chunk
            .iter()
            .map(|(s, d)| (s.to_string(), d.to_string()))
            .collect();
        let edges: Vec<(&str, &str, &str, Option<f64>, Option<stratadb::Value>)> = edge_strs
            .iter()
            .enumerate()
            .map(|(i, (s, d))| {
                let w = dataset.edge_weights.as_ref().map(|ws| ws[offset + i]);
                (s.as_str(), d.as_str(), "E", w, None)
            })
            .collect();
        strata
            .graph_bulk_insert("ldbc", &[], &edges)
            .expect("graph_bulk_insert (edges) failed");
    }

    start.elapsed()
}

/// Validate a Strata U64 analytics result against an LDBC reference.
pub fn validate_u64(
    dataset: &LdbcDataset,
    strata_result: &HashMap<String, u64>,
    reference: &U64Reference,
) -> (bool, usize) {
    let mut mismatches = 0;
    for &vid in &dataset.vertices {
        let expected = reference.values.get(&vid).copied();
        let actual = strata_result.get(&vid.to_string()).copied();
        match (expected, actual) {
            (Some(e), Some(a)) if e != a => mismatches += 1,
            (Some(_), None) | (None, Some(_)) => mismatches += 1,
            _ => {}
        }
    }
    (mismatches == 0, mismatches)
}

/// Validate a Strata F64 analytics result against an LDBC reference.
pub fn validate_f64(
    dataset: &LdbcDataset,
    strata_result: &HashMap<String, f64>,
    reference: &F64Reference,
    tolerance: f64,
) -> (bool, usize) {
    let mut mismatches = 0;
    for &vid in &dataset.vertices {
        let expected = reference.values.get(&vid).copied();
        let actual = strata_result.get(&vid.to_string()).copied();
        match (expected, actual) {
            (Some(e), Some(a)) => {
                if e.is_infinite() && a.is_infinite() {
                    // Both infinite — OK
                } else if e.is_infinite() || a.is_infinite() {
                    mismatches += 1;
                } else {
                    let rel_err = if e.abs() > 1e-15 {
                        (a - e).abs() / e.abs()
                    } else {
                        (a - e).abs()
                    };
                    if rel_err > tolerance {
                        mismatches += 1;
                    }
                }
            }
            (Some(e), None) => {
                // SSSP: reference says infinity (unreachable) and Strata didn't include it: OK
                if !e.is_infinite() {
                    mismatches += 1;
                }
            }
            (None, Some(_)) => mismatches += 1,
            _ => {}
        }
    }
    (mismatches == 0, mismatches)
}

/// Print the stats table header + values for a benchmark phase.
pub fn print_stats_table(phase: &str, stats: &RunStats) {
    eprintln!();
    eprintln!("--- {} ({} runs) ---", phase, stats.count);
    eprintln!(
        "  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "avg", "p50", "p95", "p99", "min", "max"
    );
    eprintln!(
        "  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        fmt_ms(stats.avg),
        fmt_ms(stats.p50),
        fmt_ms(stats.p95),
        fmt_ms(stats.p99),
        fmt_ms(stats.min),
        fmt_ms(stats.max),
    );
    eprintln!("  EVPS (avg): {}", fmt_num(stats.avg_evps as u64));
}

impl BfsReference {
    /// Load a BFS reference output file.
    ///
    /// Format: `vertex_id depth` per line, space-separated.
    /// The source vertex is inferred as the one with depth 0.
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;

        let mut depths = HashMap::new();
        let mut source = None;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                return Err(format!("bad BFS reference line: '{}'", line));
            }
            let vid = parts[0]
                .parse::<u64>()
                .map_err(|e| format!("bad vertex id '{}': {}", parts[0], e))?;
            let depth = parts[1]
                .parse::<i64>()
                .map_err(|e| format!("bad depth '{}': {}", parts[1], e))?;

            if depth == 0 {
                source = Some(vid);
            }
            depths.insert(vid, depth);
        }

        let source = source.ok_or_else(|| "no source vertex (depth=0) in BFS reference".to_string())?;

        Ok(BfsReference { source, depths })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn example_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/graph/example-directed")
    }

    fn example_dataset() -> LdbcDataset {
        LdbcDataset::load(&example_dir()).unwrap()
    }

    // -----------------------------------------------------------------------
    // Dataset loading tests
    // -----------------------------------------------------------------------

    #[test]
    fn load_example_dataset() {
        let ds = example_dataset();
        assert_eq!(ds.name, "example-directed");
        assert_eq!(ds.vertices.len(), 10);
        assert_eq!(ds.edges.len(), 17);
        assert!(ds.directed);
        assert_eq!(ds.bfs_source, Some(1));
        assert!(ds.edge_weights.is_none());
    }

    #[test]
    fn load_bfs_reference() {
        let path = example_dir().join("example-directed-BFS");
        let bfs = BfsReference::load(&path).unwrap();
        assert_eq!(bfs.source, 1);
        assert_eq!(bfs.depths.len(), 10);
        assert_eq!(bfs.depths[&1], 0);
        assert_eq!(bfs.depths[&2], 1);
        assert_eq!(bfs.depths[&3], 1);
        assert_eq!(bfs.depths[&4], 2);
    }

    #[test]
    fn no_unreachable_in_example() {
        let path = example_dir().join("example-directed-BFS");
        let bfs = BfsReference::load(&path).unwrap();
        for (_vid, &depth) in &bfs.depths {
            assert_ne!(depth, UNREACHABLE, "example dataset should have no unreachable vertices");
        }
    }

    // -----------------------------------------------------------------------
    // to_petgraph tests
    // -----------------------------------------------------------------------

    #[test]
    fn to_petgraph_node_count() {
        let ds = example_dataset();
        let (graph, id_map) = ds.to_petgraph();
        assert_eq!(graph.node_count(), 10);
        assert_eq!(id_map.len(), 10);
    }

    #[test]
    fn to_petgraph_edge_count() {
        let ds = example_dataset();
        let (graph, _) = ds.to_petgraph();
        // Each directed edge from the .e file becomes one undirected petgraph edge.
        // Pairs like (1,2) and (2,1) create two parallel undirected edges.
        assert_eq!(graph.edge_count(), 17);
    }

    #[test]
    fn to_petgraph_id_map_covers_all_vertices() {
        let ds = example_dataset();
        let (_, id_map) = ds.to_petgraph();
        for &vid in &ds.vertices {
            assert!(
                id_map.contains_key(&vid),
                "vertex {} missing from id_map",
                vid
            );
        }
    }

    #[test]
    fn to_petgraph_edges_are_traversable() {
        let ds = example_dataset();
        let (graph, id_map) = ds.to_petgraph();

        // Verify that for each original edge (src, dst), src and dst are
        // neighbors in the petgraph (undirected, so both directions).
        for &(src, dst) in &ds.edges {
            let si = id_map[&src];
            let di = id_map[&dst];
            let neighbors: Vec<_> = graph.neighbors(si).collect();
            assert!(
                neighbors.contains(&di),
                "edge ({}, {}): dst not in neighbors of src",
                src, dst
            );
        }
    }

    // -----------------------------------------------------------------------
    // petgraph_bfs tests
    // -----------------------------------------------------------------------

    #[test]
    fn petgraph_bfs_reaches_all_vertices() {
        let ds = example_dataset();
        let (graph, id_map) = ds.to_petgraph();
        let source = id_map[&1];
        let depths = petgraph_bfs(&graph, source);
        // All 10 vertices are reachable from vertex 1 in the undirected view
        assert_eq!(
            depths.len(),
            10,
            "BFS should reach all 10 vertices, reached {}",
            depths.len()
        );
    }

    #[test]
    fn petgraph_bfs_source_has_depth_zero() {
        let ds = example_dataset();
        let (graph, id_map) = ds.to_petgraph();
        let source = id_map[&1];
        let depths = petgraph_bfs(&graph, source);
        assert_eq!(depths[&source], 0);
    }

    #[test]
    fn petgraph_bfs_depths_match_ldbc_reference() {
        let ds = example_dataset();
        let (graph, id_map) = ds.to_petgraph();
        let source = id_map[&1];
        let depths = petgraph_bfs(&graph, source);

        // Expected depths from the LDBC reference file (BFS from vertex 1,
        // treating edges as undirected):
        let expected: HashMap<u64, usize> = [
            (1, 0),
            (2, 1),
            (3, 1),
            (4, 2),
            (5, 3),
            (6, 3),
            (7, 4),
            (8, 4),
            (9, 5),
            (10, 5),
        ]
        .into_iter()
        .collect();

        for (&vid, &expected_depth) in &expected {
            let idx = id_map[&vid];
            let actual = depths.get(&idx).copied();
            assert_eq!(
                actual,
                Some(expected_depth),
                "vertex {}: expected depth {}, got {:?}",
                vid,
                expected_depth,
                actual
            );
        }
    }

    #[test]
    fn petgraph_bfs_isolated_vertex() {
        // Build a graph with 3 nodes but only an edge between 0 and 1.
        // BFS from node 2 should only reach node 2.
        let mut graph = UnGraph::new_undirected();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n0, n1, ());

        let depths = petgraph_bfs(&graph, n2);
        assert_eq!(depths.len(), 1, "isolated source should only reach itself");
        assert_eq!(depths[&n2], 0);
    }

    #[test]
    fn petgraph_bfs_disconnected_components() {
        // Two disconnected pairs: (0,1) and (2,3)
        let mut graph = UnGraph::new_undirected();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        graph.add_edge(n0, n1, ());
        graph.add_edge(n2, n3, ());

        // BFS from n0 should reach n0 and n1, but NOT n2 or n3
        let depths = petgraph_bfs(&graph, n0);
        assert_eq!(depths.len(), 2);
        assert_eq!(depths[&n0], 0);
        assert_eq!(depths[&n1], 1);
        assert!(!depths.contains_key(&n2));
        assert!(!depths.contains_key(&n3));
    }

    #[test]
    fn petgraph_bfs_single_node() {
        let mut graph = UnGraph::new_undirected();
        let n0 = graph.add_node(());

        let depths = petgraph_bfs(&graph, n0);
        assert_eq!(depths.len(), 1);
        assert_eq!(depths[&n0], 0);
    }

    #[test]
    fn petgraph_bfs_linear_chain() {
        // 0 -- 1 -- 2 -- 3 -- 4
        let mut graph = UnGraph::new_undirected();
        let nodes: Vec<_> = (0..5).map(|_| graph.add_node(())).collect();
        for i in 0..4 {
            graph.add_edge(nodes[i], nodes[i + 1], ());
        }

        let depths = petgraph_bfs(&graph, nodes[0]);
        assert_eq!(depths.len(), 5);
        for (i, node) in nodes.iter().enumerate() {
            assert_eq!(depths[node], i, "node {} should be at depth {}", i, i);
        }
    }
}
