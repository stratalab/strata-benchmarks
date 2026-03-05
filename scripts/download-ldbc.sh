#!/usr/bin/env bash
# Download and decompress LDBC Graphalytics datasets.
#
# Usage:
#   ./scripts/download-ldbc.sh [dataset-name...]
#
# Examples:
#   ./scripts/download-ldbc.sh graph500-22              # S-scale only
#   ./scripts/download-ldbc.sh graph500-22 graph500-23  # S + M scale
#   ./scripts/download-ldbc.sh                          # defaults: S + M
#
# Datasets are downloaded from https://datasets.ldbcouncil.org/graphalytics/
# and decompressed into data/graph/<name>/.

set -euo pipefail

BASE_URL="https://datasets.ldbcouncil.org/graphalytics"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data/graph"

DEFAULTS=(graph500-22 graph500-23)

# Use arguments or defaults
if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${DEFAULTS[@]}")
fi

# Check for required tools
for tool in curl zstd tar; do
    if ! command -v "$tool" &>/dev/null; then
        echo "ERROR: '$tool' is required but not found. Install it and retry." >&2
        exit 1
    fi
done

mkdir -p "$DATA_DIR"

for name in "${DATASETS[@]}"; do
    dest="$DATA_DIR/$name"
    archive="$DATA_DIR/$name.tar.zst"
    url="$BASE_URL/$name.tar.zst"

    if [ -d "$dest" ] && [ -f "$dest/$name.v" ] && [ -f "$dest/$name.e" ]; then
        echo "==> $name already exists at $dest, skipping."
        continue
    fi

    echo "==> Downloading $name..."
    curl --fail --location --progress-bar -o "$archive" "$url"

    echo "==> Decompressing $name..."
    mkdir -p "$dest"
    zstd -d "$archive" --stdout | tar xf - -C "$dest"

    rm -f "$archive"

    # Verify expected files exist
    if [ ! -f "$dest/$name.v" ] || [ ! -f "$dest/$name.e" ]; then
        echo "WARNING: Expected files ($name.v, $name.e) not found in $dest" >&2
        echo "  Contents:" >&2
        ls -la "$dest" >&2
    else
        v_count=$(wc -l < "$dest/$name.v" | tr -d ' ')
        e_count=$(wc -l < "$dest/$name.e" | tr -d ' ')
        echo "==> $name ready: $v_count vertices, $e_count edges"
    fi
done

echo ""
echo "Done. Datasets in: $DATA_DIR"
