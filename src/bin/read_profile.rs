//! Concurrent read profiler for Strata.

use std::sync::Arc;
use std::time::Instant;

const KEY_SIZE: usize = 24;
const VALUE_SIZE: usize = 150;
const RNG_SEED: u64 = 3;
const THREAD_COUNTS: &[usize] = &[4, 8, 16, 32];
const BATCH_SIZE: usize = 50_000;

fn random_pair(rng: &mut fastrand::Rng) -> ([u8; KEY_SIZE], Vec<u8>) {
    let mut key = [0u8; KEY_SIZE];
    rng.fill(&mut key);
    let mut value = vec![0u8; VALUE_SIZE];
    rng.fill(&mut value);
    (key, value)
}

fn make_rng() -> fastrand::Rng {
    fastrand::Rng::with_seed(RNG_SEED)
}

fn dump_state(db: &stratadb::Strata, label: &str) {
    let engine = db.database();
    let l0 = engine.storage().max_l0_segment_count();
    let sched = engine.scheduler().stats();
    let cache = strata_storage::block_cache::global_cache().stats();
    let total = cache.hits + cache.misses;
    let hit_rate = if total > 0 { cache.hits as f64 / total as f64 * 100.0 } else { 0.0 };
    eprintln!(
        "[{label}] L0={l0}  sched: q={} active={} done={}  cache: {:.0}MB/{:.0}MB hit={:.1}%",
        sched.queue_depth, sched.active_tasks, sched.tasks_completed,
        cache.size_bytes as f64 / (1024.0 * 1024.0),
        cache.capacity_bytes as f64 / (1024.0 * 1024.0),
        hit_rate,
    );
}

fn run_reads(db: &Arc<stratadb::Strata>, records: usize) {
    for &num_threads in THREAD_COUNTS {
        dump_state(db, &format!("before {num_threads}t reads"));

        let reads_per_thread = records / num_threads;
        let total_reads = reads_per_thread * num_threads;
        let barrier = Arc::new(std::sync::Barrier::new(num_threads));
        let start = Instant::now();
        std::thread::scope(|s| {
            for t in 0..num_threads {
                let db = Arc::clone(db);
                let barrier = Arc::clone(&barrier);
                s.spawn(move || {
                    let mut rng = make_rng();
                    for _ in 0..(t * reads_per_thread) { random_pair(&mut rng); }
                    barrier.wait();
                    let mut found = 0u64;
                    for _ in 0..reads_per_thread {
                        let (key, _) = random_pair(&mut rng);
                        if db.kv_get(&hex::encode(key)).ok().flatten().is_some() { found += 1; }
                    }
                    let _ = found;
                });
            }
        });
        let elapsed = start.elapsed();

        dump_state(db, &format!("after  {num_threads}t reads"));
        eprintln!("{:>2} threads × {}  = {} reads in {:.1}s  ({:.1}μs/read)\n",
            num_threads, reads_per_thread, total_reads,
            elapsed.as_secs_f64(), elapsed.as_micros() as f64 / total_reads as f64);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut db_path = String::from("/tmp/strata-read-profile");
    let mut records: usize = 20_000_000;
    let mut read_only = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--db" => { i += 1; db_path = args[i].clone(); }
            "--records" => { i += 1; records = args[i].parse().unwrap(); }
            "--read-only" => { read_only = true; }
            _ => {}
        }
        i += 1;
    }
    let path = std::path::PathBuf::from(&db_path);

    if read_only {
        println!("=== READ-ONLY (compacted DB) ===");
        let config = stratadb::StrataConfig { durability: "always".into(), ..Default::default() };
        let engine = stratadb::Database::open_with_config(&path, config).unwrap();
        let db = Arc::new(stratadb::Strata::from_database(engine).unwrap());
        std::thread::sleep(std::time::Duration::from_secs(2));
        run_reads(&db, records);
    } else {
        println!("=== LOAD + IMMEDIATE READS ===");
        std::fs::create_dir_all(&path).unwrap();
        let config = stratadb::StrataConfig { durability: "always".into(), ..Default::default() };
        let engine = stratadb::Database::open_with_config(&path, config).unwrap();
        let db = Arc::new(stratadb::Strata::from_database(engine).unwrap());

        let start = Instant::now();
        let mut rng = make_rng();
        let mut batch = Vec::with_capacity(BATCH_SIZE);
        for i in 0..records {
            let (key, value) = random_pair(&mut rng);
            batch.push(stratadb::BatchKvEntry { key: hex::encode(key), value: stratadb::Value::Bytes(value) });
            if batch.len() >= BATCH_SIZE {
                db.kv_batch_put(std::mem::take(&mut batch)).unwrap();
                if (i + 1) % 5_000_000 == 0 {
                    dump_state(&db, &format!("loaded {}M", (i + 1) / 1_000_000));
                }
            }
        }
        if !batch.is_empty() { db.kv_batch_put(batch).unwrap(); }
        println!("Loaded {records} records in {:.1}s\n", start.elapsed().as_secs_f64());

        dump_state(&db, "load complete");
        run_reads(&db, records);
    }
}
