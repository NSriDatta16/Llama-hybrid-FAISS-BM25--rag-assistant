# backend/index_build.py
from retrieval.embeddings import EmbeddingManager
from retrieval.hybrid import HybridIndex, read_local_dataset, chunk_text, strip_urls

def build_all(
    data_dir,
    out_dir,
    max_docs=None,
    chunk_size=500,
    chunk_overlap=50,
    text_col=None,
    id_col=None,
    gzipped=True,
):
    """
    Builds FAISS + BM25 hybrid index with live progress updates.
    """
    emb = EmbeddingManager()
    idx = HybridIndex(out_dir, dim=emb.dim)

    print(f"📂 Reading dataset from: {data_dir}")
    docs = read_local_dataset(
        data_dir,
        max_docs=max_docs,
        text_col=text_col,
        id_col=id_col,
        gzipped=gzipped,
    )
    print(f"✅ Loaded {len(docs):,} documents")

    chunks = []
    print("🔧 Chunking documents...")
    for i, d in enumerate(docs, 1):
        cleaned = strip_urls(d["text"])
        cs = chunk_text(cleaned, size=chunk_size, overlap=chunk_overlap)
        chunks.extend({"doc_id": d["doc_id"], "text": c} for c in cs)
        if i % 1000 == 0:
            print(f"  → Processed {i:,} docs, total chunks so far: {len(chunks):,}")

    if not chunks:
        raise RuntimeError("No chunks produced. Check dataset / reader options.")

    print(f"✅ Total {len(chunks):,} chunks ready for embedding and indexing.")
    idx.build(chunks, emb.encode)
    print(f"✅ Index built with {len(chunks):,} chunks at '{out_dir}'.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="backend/index")
    ap.add_argument("--max_docs", type=int, default=None)
    ap.add_argument("--chunk_size", type=int, default=500)
    ap.add_argument("--chunk_overlap", type=int, default=50)
    ap.add_argument("--text_col", default=None)
    ap.add_argument("--id_col", default=None)
    ap.add_argument("--gzipped", action="store_true")
    args = ap.parse_args()
    build_all(**vars(args))
