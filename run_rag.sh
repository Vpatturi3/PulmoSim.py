#!/usr/bin/env bash
set -euo pipefail

# Simple helper to run the RAG airflow diagnosis pipeline
# Edit the variables below or set them via environment before running.

# Load .env if present (for GOOGLE_API_KEY and configuration overrides)
if [ -f ./.env ]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi

# Path to the folder with research PDFs
PAPERS_DIR="${PAPERS_DIR:-./articles}"

# Comma-separated image paths (quantitative plots/time series)
IMAGES_PATHS="${IMAGES_PATHS:-./outputs/quantitative.png,./outputs/time_series.png}"

# Directory where the FAISS index & metadata are stored
INDEX_DIR="${INDEX_DIR:-./rag_db}"

# Output JSON
OUT_JSON="${OUT_JSON:-./diagnosis.json}"

# Number of evidence chunks to retrieve
TOP_K=${TOP_K:-8}

# If you have an environment variable GOOGLE_API_KEY set, it will be used.
if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "ERROR: Please set GOOGLE_API_KEY in your environment before running."
  echo "Example: export GOOGLE_API_KEY=your_api_key_here"
  exit 1
fi

python rag_airflow_diagnosis.py \
  --papers "$PAPERS_DIR" \
  --images "$IMAGES_PATHS" \
  --index-dir "$INDEX_DIR" \
  --out "$OUT_JSON" \
  --top-k $TOP_K \
  --full-eval

echo "Wrote: $OUT_JSON"

# Use --rebuild when you want to recreate the FAISS index