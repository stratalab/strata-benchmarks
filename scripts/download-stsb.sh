#!/usr/bin/env bash
# Download the STS-B (Semantic Textual Similarity Benchmark) test set.
#
# Usage:
#   ./scripts/download-stsb.sh
#
# Downloads the STS-B test split from HuggingFace and saves it as a TSV file
# at data/stsb/sts-test.tsv (sentence1\tsentence2\tscore, 1,379 pairs).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data/stsb"
OUTPUT="$DATA_DIR/sts-test.tsv"

if [ -f "$OUTPUT" ]; then
    lines=$(wc -l < "$OUTPUT" | tr -d ' ')
    echo "==> STS-B test set already exists at $OUTPUT ($lines pairs), skipping."
    exit 0
fi

# Check for required tools
for tool in curl python3; do
    if ! command -v "$tool" &>/dev/null; then
        echo "ERROR: '$tool' is required but not found. Install it and retry." >&2
        exit 1
    fi
done

mkdir -p "$DATA_DIR"

# ---------------------------------------------------------------------------
# Method 1: Download parquet and convert with pandas
# ---------------------------------------------------------------------------
PARQUET_URL="https://huggingface.co/datasets/sentence-transformers/stsb/resolve/main/test.parquet"
TEMP_PARQUET="$DATA_DIR/test.parquet"
parquet_ok=false

echo "==> Trying parquet download from HuggingFace..."
if curl --fail --location --silent --progress-bar -o "$TEMP_PARQUET" "$PARQUET_URL" 2>/dev/null; then
    if python3 -c "
import sys
try:
    import pandas as pd
    df = pd.read_parquet('$TEMP_PARQUET')
    with open('$OUTPUT', 'w') as out:
        for _, row in df.iterrows():
            s1 = str(row.get('sentence1', '')).replace('\t', ' ').replace('\n', ' ')
            s2 = str(row.get('sentence2', '')).replace('\t', ' ').replace('\n', ' ')
            score = float(row.get('score', 0.0))
            out.write(f'{s1}\t{s2}\t{score}\n')
    print(f'Extracted {len(df)} pairs', file=sys.stderr)
except ImportError:
    print('pandas/pyarrow not available, will try API fallback', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'Parquet conversion failed: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1; then
        parquet_ok=true
    fi
fi
rm -f "$TEMP_PARQUET"

# ---------------------------------------------------------------------------
# Method 2: HuggingFace datasets API with pagination
# ---------------------------------------------------------------------------
if [ "$parquet_ok" = false ]; then
    echo "==> Parquet method failed, trying HuggingFace API..."
    API_BASE="https://datasets-server.huggingface.co/rows?dataset=sentence-transformers%2Fstsb&config=default&split=test"

    python3 -c "
import json, sys, urllib.request

all_rows = []
offset = 0
batch = 100

while True:
    url = '${API_BASE}&offset=' + str(offset) + '&length=' + str(batch)
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f'ERROR: API request failed at offset {offset}: {e}', file=sys.stderr)
        if all_rows:
            break
        sys.exit(1)

    rows = data.get('rows', [])
    if not rows:
        break
    all_rows.extend(rows)
    offset += len(rows)
    if len(rows) < batch:
        break

if not all_rows:
    print('ERROR: No rows fetched from API', file=sys.stderr)
    sys.exit(1)

with open('$OUTPUT', 'w') as out:
    for row in all_rows:
        r = row.get('row', {})
        s1 = r.get('sentence1', '').replace('\t', ' ').replace('\n', ' ')
        s2 = r.get('sentence2', '').replace('\t', ' ').replace('\n', ' ')
        score = r.get('score', 0.0)
        out.write(f'{s1}\t{s2}\t{score}\n')

print(f'Extracted {len(all_rows)} pairs', file=sys.stderr)
"
fi

if [ ! -f "$OUTPUT" ]; then
    echo "ERROR: Failed to create $OUTPUT" >&2
    exit 1
fi

lines=$(wc -l < "$OUTPUT" | tr -d ' ')
echo "==> STS-B test set ready: $lines sentence pairs at $OUTPUT"
