#!/usr/bin/env bash
set -euo pipefail
# 1) Build indexes (edit data_dir to where your MSMARCO is)
python backend/index_build.py --data_dir "./data" --out_dir "backend/index" --gzipped --chunk_size 500 --chunk_overlap 50 --max_docs 50000
# 2) Start API
uvicorn backend.main:app --host 0.0.0.0 --port 8008 --reload