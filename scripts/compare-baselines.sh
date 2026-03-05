#!/usr/bin/env bash
# Compare two tagged baselines category-by-category via bench-compare.
#
# Usage:
#   ./scripts/compare-baselines.sh <baseline-tag> <candidate-tag>
#
# Example:
#   ./scripts/compare-baselines.sh pre-refactor post-refactor

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ $# -ne 2 ]; then
    echo "Usage: $0 <baseline-tag> <candidate-tag>" >&2
    exit 1
fi

BASELINE_TAG="$1"
CANDIDATE_TAG="$2"
BASELINE_DIR="$REPO_ROOT/baselines/$BASELINE_TAG"
CANDIDATE_DIR="$REPO_ROOT/baselines/$CANDIDATE_TAG"

if [ ! -d "$BASELINE_DIR" ]; then
    echo "ERROR: Baseline '$BASELINE_TAG' not found at $BASELINE_DIR" >&2
    exit 1
fi

if [ ! -d "$CANDIDATE_DIR" ]; then
    echo "ERROR: Candidate '$CANDIDATE_TAG' not found at $CANDIDATE_DIR" >&2
    exit 1
fi

# ── Build bench-compare ─────────────────────────────────────────────────────

echo "Building bench-compare..."
cargo build --release --bin bench-compare 2>/dev/null
echo ""

# ── Extract categories from baseline ────────────────────────────────────────
#
# Filename pattern: <category>-<YYYY>-<MM>-<DD>T<HH>-<MM>-<SS>Z-<commit>.json
# Category is everything before the first  -YYYY-MM-DD  timestamp.

declare -A baseline_files
declare -A candidate_files

for f in "$BASELINE_DIR"/*.json; do
    [ -f "$f" ] || continue
    fname=$(basename "$f")
    category=$(echo "$fname" | sed 's/-[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}T.*//')
    baseline_files[$category]="$f"
done

for f in "$CANDIDATE_DIR"/*.json; do
    [ -f "$f" ] || continue
    fname=$(basename "$f")
    category=$(echo "$fname" | sed 's/-[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}T.*//')
    candidate_files[$category]="$f"
done

# ── Compare matched categories ──────────────────────────────────────────────

matched=0
baseline_only=()
candidate_only=()

# Sort categories for consistent output
all_categories=()
for cat in "${!baseline_files[@]}"; do
    all_categories+=("$cat")
done
IFS=$'\n' sorted_categories=($(sort <<< "${all_categories[*]}")); unset IFS

for category in "${sorted_categories[@]}"; do
    if [ -n "${candidate_files[$category]+x}" ]; then
        echo "=== $category ==="
        cargo run --release --bin bench-compare -- \
            "${baseline_files[$category]}" \
            "${candidate_files[$category]}" \
            2>/dev/null || true
        echo ""
        matched=$((matched + 1))
    else
        baseline_only+=("$category")
    fi
done

# Find candidate-only categories
for cat in "${!candidate_files[@]}"; do
    if [ -z "${baseline_files[$cat]+x}" ]; then
        candidate_only+=("$cat")
    fi
done

# ── Summary ──────────────────────────────────────────────────────────────────

total_categories=$(( matched + ${#baseline_only[@]} + ${#candidate_only[@]} ))
echo "Summary: $matched/$total_categories categories compared, ${#baseline_only[@]} baseline-only, ${#candidate_only[@]} candidate-only"

if [ ${#baseline_only[@]} -gt 0 ]; then
    echo "  Baseline-only: ${baseline_only[*]}"
fi

if [ ${#candidate_only[@]} -gt 0 ]; then
    echo "  Candidate-only: ${candidate_only[*]}"
fi
