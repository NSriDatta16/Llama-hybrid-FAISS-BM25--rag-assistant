import arxiv, os, time, json, re
from backend.retrieval.hybrid import simple_clean, strip_urls
OUT_DIR = "backend/data/arxiv_ai_json"
os.makedirs(OUT_DIR, exist_ok=True)

SEARCH_QUERIES = [
    "machine learning", "deep learning", "neural network",
    "reinforcement learning", "transformer", "gan", "rnn",
    "autoencoder", "retrieval augmented generation", "agents"
]

def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

results = []
for q in SEARCH_QUERIES:
    print(f"🔍 Querying arXiv: {q}")
    search = arxiv.Search(query=q, max_results=3000, sort_by=arxiv.SortCriterion.SubmittedDate)
    for paper in search.results():
        text = simple_clean(strip_urls(f"{paper.title}. {paper.summary}"))
        results.append({"doc_id": paper.entry_id, "text": text})
    time.sleep(3)

save_jsonl(os.path.join(OUT_DIR, "arxiv_ai.jsonl"), results)
print(f"✅ Saved {len(results)} docs to {OUT_DIR}")
