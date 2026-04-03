//! Isolated compaction profiler. Pre-loads data, then opens the DB and
//! profiles ONLY the compaction convergence phase.
//!
//! Step 1: Load data (if DB doesn't exist)
//! Step 2: Flamegraph this: only compaction runs, no load overhead
//!
//! Usage:
//!   # First, create the DB:
//!   cargo run --release --bin compact_profile -- --db /tmp/compact-bench --records 5000000 --load
//!   # Then flamegraph just the compaction:
//!   flamegraph --freq 997 -o compact.svg -- target/release/compact_profile --db /tmp/compact-bench

use std::time::Instant;

const KEY_SIZE: usize = 24;
const VALUE_SIZE: usize = 150;
const RNG_SEED: u64 = 3;
const BATCH_SIZE: usize = 50_000;

fn random_pair(rng: &mut fastrand::Rng) -> ([u8; KEY_SIZE], Vec<u8>) {
    let mut key = [0u8; KEY_SIZE];
    rng.fill(&mut key);
    let mut value = vec![0u8; VALUE_SIZE];
    rng.fill(&mut value);
    (key, value)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut db_path = String::from("/tmp/compact-bench");
    let mut records: usize = 5_000_000;
    let mut load = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--db" => { i += 1; db_path = args[i].clone(); }
            "--records" => { i += 1; records = args[i].parse().unwrap(); }
            "--load" => { load = true; }
            _ => {}
        }
        i += 1;
    }

    let path = std::path::PathBuf::from(&db_path);

    if load {
        // Phase 1: Load data only, then exit (no compaction waiting)
        eprintln!("Loading {records} records into {db_path}...");
        std::fs::create_dir_all(&path).unwrap();
        let config = stratadb::StrataConfig { durability: "always".into(), ..Default::default() };
        let engine = stratadb::Database::open_with_config(&path, config).unwrap();
        let db = stratadb::Strata::from_database(engine).unwrap();

        let start = Instant::now();
        let mut rng = fastrand::Rng::with_seed(RNG_SEED);
        let mut batch = Vec::with_capacity(BATCH_SIZE);
        for i in 0..records {
            let (key, value) = random_pair(&mut rng);
            batch.push(stratadb::BatchKvEntry {
                key: hex::encode(key),
                value: stratadb::Value::Bytes(value),
            });
            if batch.len() >= BATCH_SIZE {
                db.kv_batch_put(std::mem::take(&mut batch)).unwrap();
                if (i + 1) % 1_000_000 == 0 { eprintln!("  loaded {}M...", (i + 1) / 1_000_000); }
            }
        }
        if !batch.is_empty() { db.kv_batch_put(batch).unwrap(); }
        eprintln!("Loaded in {:.1}s. DB at {db_path}", start.elapsed().as_secs_f64());
        eprintln!("Now run without --load to profile compaction.");
    } else {
        // Phase 2: Open DB, wait for compaction to converge (this is what you flamegraph)
        eprintln!("Opening {db_path} — compaction will run...");
        let config = stratadb::StrataConfig { durability: "always".into(), ..Default::default() };
        let engine = stratadb::Database::open_with_config(&path, config).unwrap();
        let db = stratadb::Strata::from_database(engine).unwrap();

        let l0_before = db.database().storage().max_l0_segment_count();
        eprintln!("L0 at open: {l0_before}");

        let start = Instant::now();
        loop {
            let l0 = db.database().storage().max_l0_segment_count();
            if l0 == 0 { break; }
            std::thread::sleep(std::time::Duration::from_millis(100));
            if start.elapsed().as_secs() > 300 {
                eprintln!("Timed out (L0={})", l0);
                break;
            }
        }
        eprintln!("Compaction converged in {:.1}s (L0: {} → 0)", start.elapsed().as_secs_f64(), l0_before);
    }
}
