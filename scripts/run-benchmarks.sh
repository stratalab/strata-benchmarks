#!/usr/bin/env bash
# Run the full benchmark suite and optionally save a tagged baseline snapshot.
#
# Usage:
#   ./scripts/run-benchmarks.sh [--tag NAME] [--skip-graph] [--skip-criterion] [--skip-concurrency]
#
# Examples:
#   ./scripts/run-benchmarks.sh --tag pre-refactor       # full run, save baseline
#   ./scripts/run-benchmarks.sh --skip-graph              # latency + throughput only
#   ./scripts/run-benchmarks.sh --skip-criterion          # skip slow Criterion benches
#   ./scripts/run-benchmarks.sh --skip-concurrency        # skip concurrency (~28 min)
#   ./scripts/run-benchmarks.sh                           # run everything, don't tag

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/results"

# ── Parse arguments ──────────────────────────────────────────────────────────

TAG=""
SKIP_GRAPH=false
SKIP_CRITERION=false
SKIP_CONCURRENCY=false

while [ $# -gt 0 ]; do
    case "$1" in
        --tag)
            shift
            if [ $# -eq 0 ]; then
                echo "ERROR: --tag requires a name argument" >&2
                exit 1
            fi
            TAG="$1"
            ;;
        --skip-graph)
            SKIP_GRAPH=true
            ;;
        --skip-criterion)
            SKIP_CRITERION=true
            ;;
        --skip-concurrency)
            SKIP_CONCURRENCY=true
            ;;
        -h|--help)
            head -11 "$0" | tail -9
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            exit 1
            ;;
    esac
    shift
done

# ── Define benchmark suite ───────────────────────────────────────────────────
#
# Format: "bench_name|category|flags|group"
#   group: criterion | custom | graph

BENCHMARKS=(
    "kv|latency||criterion"
    "state|latency||criterion"
    "event|latency||criterion"
    "json|latency||criterion"
    "vector|latency||criterion"
    "branch|latency||criterion"
    "fill_level|fill-level|-q|custom"
    "ycsb|ycsb|-q|custom"
    "concurrency|concurrency||concurrency"
    "graph_bfs|graph-bfs|-q|graph"
    "graph_wcc|graph-wcc|-q|graph"
    "graph_cdlp|graph-cdlp|-q|graph"
    "graph_pr|graph-pr|-q|graph"
    "graph_lcc|graph-lcc|-q|graph"
    "ann|ann|--quick -q|custom"
    "embed|embed|--quick -q|custom"
    "scaling|scaling|--quick -q|custom"
)

# ── Build only the benchmarks we'll run ───────────────────────────────────────

echo "=== Building benchmarks (release) ==="
build_start=$(date +%s)
build_args=()
for entry in "${BENCHMARKS[@]}"; do
    IFS='|' read -r name category flags group <<< "$entry"
    if [ "$SKIP_GRAPH" = true ] && [ "$group" = "graph" ]; then continue; fi
    if [ "$SKIP_CRITERION" = true ] && [ "$group" = "criterion" ]; then continue; fi
    if [ "$SKIP_CONCURRENCY" = true ] && [ "$group" = "concurrency" ]; then continue; fi
    build_args+=(--bench "$name")
done
if ! cargo build --release "${build_args[@]}" 2>&1; then
    echo "ERROR: Build failed. Aborting." >&2
    exit 1
fi
build_elapsed=$(( $(date +%s) - build_start ))
echo "Build completed in ${build_elapsed}s"
echo ""

# ── Run benchmarks ───────────────────────────────────────────────────────────

# Record the run start time so we can identify result files from this run.
# Files created after this timestamp belong to the current run.
RUN_START=$(date -u +%Y-%m-%dT%H-%M-%SZ)
RUN_START_EPOCH=$(date +%s)

total=0
for entry in "${BENCHMARKS[@]}"; do
    IFS='|' read -r name category flags group <<< "$entry"
    if [ "$SKIP_GRAPH" = true ] && [ "$group" = "graph" ]; then continue; fi
    if [ "$SKIP_CRITERION" = true ] && [ "$group" = "criterion" ]; then continue; fi
    if [ "$SKIP_CONCURRENCY" = true ] && [ "$group" = "concurrency" ]; then continue; fi
    total=$((total + 1))
done

passed=0
failed=0
failed_names=()
idx=0

for entry in "${BENCHMARKS[@]}"; do
    IFS='|' read -r name category flags group <<< "$entry"

    # Apply skip filters
    if [ "$SKIP_GRAPH" = true ] && [ "$group" = "graph" ]; then continue; fi
    if [ "$SKIP_CRITERION" = true ] && [ "$group" = "criterion" ]; then continue; fi
    if [ "$SKIP_CONCURRENCY" = true ] && [ "$group" = "concurrency" ]; then continue; fi

    idx=$((idx + 1))
    printf "[%d/%d] Running %s..." "$idx" "$total" "$name"
    bench_start=$(date +%s)

    # Run the benchmark; capture output but don't abort on failure
    if [ -n "$flags" ]; then
        if cargo bench --bench "$name" -- $flags > /dev/null 2>&1; then
            bench_elapsed=$(( $(date +%s) - bench_start ))
            echo " done (${bench_elapsed}s)"
            passed=$((passed + 1))
        else
            bench_elapsed=$(( $(date +%s) - bench_start ))
            echo " FAILED (${bench_elapsed}s)"
            failed=$((failed + 1))
            failed_names+=("$name")
        fi
    else
        if cargo bench --bench "$name" > /dev/null 2>&1; then
            bench_elapsed=$(( $(date +%s) - bench_start ))
            echo " done (${bench_elapsed}s)"
            passed=$((passed + 1))
        else
            bench_elapsed=$(( $(date +%s) - bench_start ))
            echo " FAILED (${bench_elapsed}s)"
            failed=$((failed + 1))
            failed_names+=("$name")
        fi
    fi
done

# ── Summary ──────────────────────────────────────────────────────────────────

total_elapsed=$(( $(date +%s) - RUN_START_EPOCH ))
echo ""
echo "=== Summary ==="
echo "Passed: $passed / $total"
if [ "$failed" -gt 0 ]; then
    echo "Failed: $failed (${failed_names[*]})"
fi
echo "Total time: ${total_elapsed}s"

# ── Tag baseline ─────────────────────────────────────────────────────────────

if [ -n "$TAG" ]; then
    BASELINE_DIR="$REPO_ROOT/baselines/$TAG"
    mkdir -p "$BASELINE_DIR"

    # Copy result JSON files created during this run (modified after RUN_START_EPOCH).
    # For each category, keep only the newest file (ls -t gives newest first).
    copied=0
    declare -A seen_categories

    for f in $(ls -t "$RESULTS_DIR"/*.json 2>/dev/null); do
        # stat -c works on Linux, stat -f on macOS
        file_epoch=$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null)
        [ -n "$file_epoch" ] && [ "$file_epoch" -ge "$RUN_START_EPOCH" ] || continue

        fname=$(basename "$f")
        # Extract category: everything before the timestamp pattern  -YYYY-MM-DD
        category=$(echo "$fname" | sed 's/-[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}T.*//')

        if [ -z "${seen_categories[$category]+x}" ]; then
            cp "$f" "$BASELINE_DIR/"
            seen_categories[$category]=1
            copied=$((copied + 1))
        fi
    done

    echo ""
    echo "Baseline '$TAG' saved: $copied result files -> $BASELINE_DIR/"
    ls "$BASELINE_DIR/"
fi
