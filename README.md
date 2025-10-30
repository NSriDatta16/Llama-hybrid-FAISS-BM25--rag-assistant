# MSMARCO Hybrid RAG (FAISS + BM25) with Chat UI

End-to-end template to index MSMARCO-like datasets and chat via a RAG pipeline.
- Chunking → Embeddings (SentenceTransformers) → FAISS (dense) + BM25 (lexical)
- Hybrid scoring: `score = α * dense + (1-α) * lexical`
- LlamaIndex used for the LLM call + prompt templating (OpenAI-compatible)
- Streamlit UI in a ChatGPT-like style

## Setup

```bash
python -m venv .venv && source .venv/bin/activate  # (on Windows: .venv\Scripts\activate)
pip install -r requirements.txt
cp backend/.env.example backend/.env  # put your OPENAI_API_KEY
```

Put your dataset under `backend/data` (e.g., `msmarco-docs.tsv.gz`). The builder will scan `*.tsv*`, `*.txt`, `*.jsonl`.

## Build Index

```bash
# Example (adjust --max_docs for speed on first run)
python backend/index_build.py --data_dir backend/data --out_dir backend/index --gzipped --chunk_size 500 --chunk_overlap 50 --max_docs 100000
```

This creates:
- `backend/index/corpus.parquet` – chunk text + doc IDs
- `backend/index/faiss.index` – dense vectors
- `backend/index/bm25.pkl` – lexical model

## Run Backend API

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8008 --reload
```

Test:
```bash
curl -s -X POST "http://localhost:8008/ask" -H "Content-Type: application/json" -d '{"query":"What is attention?", "top_k":5}'
```

## Run Chat UI

In another terminal:

```bash
export API_BASE=http://127.0.0.1:8008   # Windows: set API_BASE=http://127.0.0.1:8008
streamlit run ui/app.py
```

## Notes

- If `msmarco-docs.tsv.gz` has no header, the loader assumes first column is ID and second is text.
- To use a local OpenAI-compatible LLM, set `OPENAI_BASE_URL` and `OPENAI_MODEL` in `.env`.
- Tune α in the sidebar to balance semantic vs lexical retrieval.