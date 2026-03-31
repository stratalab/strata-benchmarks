//! Industry-standard embedded database benchmark (redb suite).
//!
//! Compares Strata against redb, lmdb, rocksdb, sled, fjall, sqlite, and Redis
//! on bulk load, individual writes, batch writes, random reads, range scans,
//! multi-threaded reads, and deletes.
//!
//! Redis requires a running server (localhost:6379). If unavailable, the Redis
//! column is skipped and the table prints without it.
//!
//! Adapted from: https://github.com/cberner/redb/tree/master/crates/redb-bench

use std::env::current_dir;
use std::{fs, process};
use tempfile::{NamedTempFile, TempDir};

mod common;
use common::*;

fn parse_records() -> usize {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--records" {
            i += 1;
            if i < args.len() {
                return args[i].parse().unwrap_or(100_000);
            }
        }
        i += 1;
    }
    100_000
}

fn parse_only() -> Option<Vec<String>> {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--only" {
            i += 1;
            if i < args.len() {
                return Some(args[i].split(',').map(|s| s.to_string()).collect());
            }
        }
        i += 1;
    }
    None
}

fn should_run(db_name: &str, only: &Option<Vec<String>>) -> bool {
    match only {
        None => true,
        Some(list) => list.iter().any(|s| s == db_name),
    }
}

fn main() {
    let _ = env_logger::try_init();
    let records = parse_records();
    let only = parse_only();
    let cfg = BenchConfig::for_records(records);

    println!("=== redb benchmark: {} records ===\n", records);

    let tmpdir = current_dir().unwrap().join(".benchmark");
    fs::create_dir_all(&tmpdir).unwrap();

    let tmpdir2 = tmpdir.clone();
    ctrlc::set_handler(move || {
        fs::remove_dir_all(&tmpdir2).unwrap();
        process::exit(1);
    })
    .unwrap();

    // ── Strata ──────────────────────────────────────────────────────────────
    let strata_results = if !should_run("strata", &only) { None } else { Some({
        let tmpdir_strata: TempDir = tempfile::tempdir_in(&tmpdir).unwrap();
        // Use "always" durability for fair comparison — other DBs fsync every commit
        let config = stratadb::StrataConfig {
            durability: "always".into(),
            ..stratadb::StrataConfig::default()
        };
        let engine = stratadb::Database::open_with_config(tmpdir_strata.path(), config).unwrap();
        let db = stratadb::Strata::from_database(engine).unwrap();
        let table = StrataBenchDatabase::new(db);
        benchmark(table, tmpdir_strata.path(), &cfg)
    })};

    // ── redb ────────────────────────────────────────────────────────────────
    let redb_results = if !should_run("redb", &only) { None } else { Some({
        let tmpfile: NamedTempFile = NamedTempFile::new_in(&tmpdir).unwrap();
        let mut db = redb::Database::builder()
            .set_cache_size(CACHE_SIZE)
            .create(tmpfile.path())
            .unwrap();
        let table = RedbBenchDatabase::new(&mut db);
        benchmark(table, tmpfile.path(), &cfg)
    })};

    // ── lmdb (heed) ─────────────────────────────────────────────────────────
    let lmdb_results = if !should_run("lmdb", &only) { None } else { Some({
        let tempdir: TempDir = tempfile::tempdir_in(&tmpdir).unwrap();
        let env = unsafe {
            heed::EnvOpenOptions::new()
                .map_size(4096 * 1024 * 1024)
                .open(tempdir.path())
                .unwrap()
        };
        let table = HeedBenchDatabase::new(env);
        benchmark(table, tempdir.path(), &cfg)
    })};

    // ── rocksdb ─────────────────────────────────────────────────────────────
    let rocksdb_results = if !should_run("rocksdb", &only) { None } else { Some({
        let tmpfile: TempDir = tempfile::tempdir_in(&tmpdir).unwrap();

        let cache = rocksdb::Cache::new_lru_cache(CACHE_SIZE);
        let write_buffer = rocksdb::WriteBufferManager::new_write_buffer_manager_with_cache(
            CACHE_SIZE / 2,
            false,
            cache.clone(),
        );

        let mut bb = rocksdb::BlockBasedOptions::default();
        bb.set_block_cache(&cache);
        bb.set_bloom_filter(10.0, false);
        bb.set_cache_index_and_filter_blocks(true);
        bb.set_pin_l0_filter_and_index_blocks_in_cache(false);
        bb.set_pin_top_level_index_and_filter(false);

        let mut opts = rocksdb::Options::default();
        opts.set_block_based_table_factory(&bb);
        opts.set_write_buffer_manager(&write_buffer);
        opts.set_max_write_buffer_size_to_maintain((CACHE_SIZE / 2) as i64);
        opts.create_if_missing(true);
        opts.increase_parallelism(
            std::thread::available_parallelism().map_or(1, |n| n.get()) as i32,
        );

        let db = rocksdb::OptimisticTransactionDB::open(&opts, tmpfile.path()).unwrap();
        let table = RocksdbBenchDatabase::new(&db);
        benchmark(table, tmpfile.path(), &cfg)
    })};

    // ── sled ────────────────────────────────────────────────────────────────
    let sled_results = if !should_run("sled", &only) { None } else { Some({
        let tmpfile: TempDir = tempfile::tempdir_in(&tmpdir).unwrap();

        let db = sled::Config::new()
            .path(tmpfile.path())
            .cache_capacity(CACHE_SIZE as u64)
            .open()
            .unwrap();

        let table = SledBenchDatabase::new(&db, tmpfile.path());
        benchmark(table, tmpfile.path(), &cfg)
    })};

    // ── fjall ───────────────────────────────────────────────────────────────
    let fjall_results = if !should_run("fjall", &only) { None } else { Some({
        let tmpfile: TempDir = tempfile::tempdir_in(&tmpdir).unwrap();

        let mut db = fjall::Config::new(tmpfile.path())
            .cache_size(CACHE_SIZE.try_into().unwrap())
            .open_transactional()
            .unwrap();

        let table = FjallBenchDatabase::new(&mut db);
        benchmark(table, tmpfile.path(), &cfg)
    })};

    // ── sqlite ──────────────────────────────────────────────────────────────
    let sqlite_results = if !should_run("sqlite", &only) { None } else { Some({
        let tmpfile: NamedTempFile = NamedTempFile::new_in(&tmpdir).unwrap();
        let table = SqliteBenchDatabase::new(tmpfile.path());
        benchmark(table, tmpfile.path(), &cfg)
    })};

    // ── Redis (optional — skipped if server unavailable) ────────────────────
    let redis_results = if !should_run("redis", &only) { None } else { RedisBenchDatabase::new("redis://127.0.0.1/").map(|table| {
        let tmpdir_redis: TempDir = tempfile::tempdir_in(&tmpdir).unwrap();
        benchmark(table, tmpdir_redis.path(), &cfg)
    })};
    if redis_results.is_none() && should_run("redis", &only) {
        println!("Redis: skipped (server not available at 127.0.0.1:6379)");
    }

    fs::remove_dir_all(&tmpdir).unwrap();

    // ── Print comparison table ──────────────────────────────────────────────
    let mut rows = Vec::new();

    // Find first non-None result to get benchmark names
    let all_optional: Vec<Option<Vec<(String, ResultType)>>> = vec![
        strata_results,
        redb_results,
        lmdb_results,
        rocksdb_results,
        sled_results,
        fjall_results,
        sqlite_results,
        redis_results,
    ];
    let all_headers = vec![
        "strata", "redb", "lmdb", "rocksdb", "sled", "fjall", "sqlite", "redis",
    ];

    // Collect only the databases that were run
    let mut all_results: Vec<Vec<(String, ResultType)>> = Vec::new();
    let mut headers: Vec<&str> = vec![""];
    for (i, opt) in all_optional.into_iter().enumerate() {
        if let Some(results) = opt {
            if rows.is_empty() {
                for (benchmark, _) in &results {
                    rows.push(vec![benchmark.to_string()]);
                }
            }
            all_results.push(results);
            headers.push(all_headers[i]);
        }
    }

    if rows.is_empty() {
        println!("No databases were benchmarked.");
        return;
    }

    let mut identified_smallests = vec![vec![false; all_results.len()]; rows.len()];
    for (i, identified_smallests_row) in identified_smallests.iter_mut().enumerate() {
        let mut smallest = None;
        for (j, _) in identified_smallests_row.iter().enumerate() {
            let (_, rt) = &all_results[j][i];
            smallest = match smallest {
                Some((_, prev)) if rt < prev => Some((j, rt)),
                Some((pi, prev)) => Some((pi, prev)),
                None => Some((j, rt)),
            };
        }
        let (j, _rt) = smallest.unwrap();
        identified_smallests_row[j] = true;
    }

    for (j, results) in all_results.iter().enumerate() {
        for (i, (_benchmark, result_type)) in results.iter().enumerate() {
            rows[i].push(if identified_smallests[i][j] {
                format!("**{result_type}**")
            } else {
                result_type.to_string()
            });
        }
    }

    let mut table = comfy_table::Table::new();
    table.load_preset(comfy_table::presets::ASCII_MARKDOWN);
    table.set_width(140);
    table.set_header(headers);
    for row in rows {
        table.add_row(row);
    }

    println!();
    println!("{table}");
}
