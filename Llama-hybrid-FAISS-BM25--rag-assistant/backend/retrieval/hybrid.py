# backend/retrieval/hybrid.py
import os, pickle, re, glob, gzip, json, csv, sys
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
import faiss

# ----------------- Text cleanup & tokenization -----------------

URL_RE = re.compile(r"https?://\S+")
ALNUM_RE = re.compile(r"[A-Za-z0-9]+")


def strip_urls(text: str) -> str:
    """Remove raw URLs so the LLM doesn't regurgitate them."""
    return URL_RE.sub("", text or "")


def simple_clean(s: str) -> str:
    """Normalize whitespace and strip NULs etc."""
    s = (s or "").replace("\u0000", " ")
    return re.sub(r"\s+", " ", s).strip()


def tokenize(text: str) -> List[str]:
    """
    Lightweight tokenizer for BM25 (no NLTK download hassles).
    Lowercases and keeps alphanumerics.
    """
    return ALNUM_RE.findall((text or "").lower())


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    """
    Simple fixed-size chunker with overlap; cleans & strips URLs defensively.
    """
    text = strip_urls(simple_clean(text))
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks, i = [], 0
    step = max(1, size - overlap)
    while i < len(text):
        chunks.append(text[i : i + size])
        i += step
    return chunks


# ----------------- Dataset reader (MSMARCO-friendly) -----------------

def read_local_dataset(
    data_dir: str,
    pattern: str = "*.tsv*",
    max_docs: int | None = None,
    text_col: str | None = None,
    id_col: str | None = None,
    gzipped: bool = True,
) -> List[Dict[str, str]]:
    """
    Reads MSMARCO-style TSV/TSV.GZ (with or without header) and .txt files.
    Increases CSV field size limit to handle very long passages.
    Returns a list of dicts: {"doc_id": str, "text": str}
    """
    # raise CSV field size limit to handle long rows
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)

    docs: List[Dict[str, str]] = []

    for path in glob.glob(os.path.join(data_dir, pattern)):
        # decide handle
        if path.endswith(".gz") and gzipped:
            f = gzip.open(path, "rt", encoding="utf-8", newline="")
        else:
            f = open(path, "r", encoding="utf-8", newline="")

        with f:
            reader = csv.reader(f, delimiter="\t")

            # Try to detect header (best effort)
            has_header = False
            try:
                pos = f.tell()
                sample = f.read(4096)
                f.seek(pos)
                has_header = csv.Sniffer().has_header(sample)
            except Exception:
                has_header = False

            header = None
            if has_header:
                try:
                    header = next(reader)
                except StopIteration:
                    continue

            count = 0
            for row in reader:
                if max_docs and count >= max_docs:
                    break
                if not row:
                    continue

                if header and text_col in (header or []):
                    # named columns
                    t = row[header.index(text_col)] if text_col in header else (row[1] if len(row) > 1 else "")
                    did = row[header.index(id_col)] if (id_col and id_col in header) else (row[0] if row else str(count))
                else:
                    # MSMARCO default: col0=id, col1=text
                    did = row[0]
                    t = row[1] if len(row) > 1 else ""

                docs.append({"doc_id": str(did), "text": simple_clean(t)})
                count += 1

    # Add any plain .txt files
    for path in glob.glob(os.path.join(data_dir, "*.txt")):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as tf:
                docs.append({"doc_id": os.path.basename(path), "text": simple_clean(tf.read())})
        except Exception:
            pass

    return docs


# ----------------- Hybrid Index (FAISS + BM25) -----------------

class HybridIndex:
    def __init__(self, base_dir: str, dim: int):
        self.base = base_dir
        os.makedirs(self.base, exist_ok=True)
        self.dim = dim
        self.df: pd.DataFrame | None = None
        self.faiss_index: faiss.Index | None = None
        self.bm25: BM25Okapi | None = None

    @property
    def corpus_parquet(self): return os.path.join(self.base, "corpus.parquet")

    @property
    def faiss_index_path(self): return os.path.join(self.base, "faiss.index")

    @property
    def bm25_pickle(self): return os.path.join(self.base, "bm25.pkl")

    def build(self, chunks: List[Dict[str, Any]], embed_fn, batch_size: int = 2048):
        """
        Build FAISS (inner product) + BM25 over sanitized chunks in batches.
        Avoids holding all embeddings in RAM at once.
        """
        # 1) Persist corpus first (so we can mmap / reload)
        self.df = pd.DataFrame(chunks)
        self.df.to_parquet(self.corpus_parquet, index=False)

        texts = self.df["text"].tolist()
        n = len(texts)

        # 2) Create FAISS index and add in batches
        index = faiss.IndexFlatIP(self.dim)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_texts = texts[start:end]
            X = embed_fn(batch_texts)          # returns float32, normalized
            index.add(X)                       # add incrementally
            if start % (batch_size * 5) == 0:
                print(f"FAISS: added {end}/{n} vectors")

        faiss.write_index(index, self.faiss_index_path)
        self.faiss_index = index

        # 3) BM25 (tokenize in batches to reduce peak memory)
        tokenized = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            tokenized.extend([tokenize(t) for t in texts[start:end]])
            if start % (batch_size * 10) == 0:
                print(f"BM25: tokenized {end}/{n} docs")

        self.bm25 = BM25Okapi(tokenized)
        with open(self.bm25_pickle, "wb") as f:
            pickle.dump(self.bm25, f)

        print(f"✅ Built hybrid index with {n} chunks.")

    def load(self):
        import pyarrow.parquet as pq
        self.df = pq.read_table(self.corpus_parquet).to_pandas()
        self.faiss_index = faiss.read_index(self.faiss_index_path)
        with open(self.bm25_pickle, "rb") as f:
            self.bm25 = pickle.load(f)
        print(f"✅ Loaded index with {len(self.df)} chunks.")

    def search(self, query: str, embed_fn, k: int = 5, alpha: float = 0.6):
        """
        Hybrid scoring:
            score = alpha * dense_sim + (1 - alpha) * bm25_score

        Performance note:
            On very large corpora, computing BM25 scores across *all* docs is slow.
            If alpha >= 0.999, we short-circuit and return FAISS results only.
        """
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0,1]"
        if self.df is None or self.faiss_index is None:
            raise RuntimeError("Index not loaded or built.")
        use_bm25 = self.bm25 is not None and alpha < 0.999

        # ------ Dense (FAISS) ------
        qv = embed_fn([query])
        if qv.dtype != np.float32:
            qv = qv.astype("float32", copy=False)
        # when mixing with BM25, fetch a bit more dense candidates
        dense_fetch = k if not use_bm25 else max(k, 100)
        D, I = self.faiss_index.search(qv, dense_fetch)
        dense_idx = I[0]
        dense_scores = D[0]
        dense_map = {int(idx): float(sc) for idx, sc in zip(dense_idx, dense_scores)}

        # Fast path: dense-only
        if not use_bm25:
            results = []
            for i, sc in zip(dense_idx[:k], dense_scores[:k]):
                i = int(i)
                row = self.df.iloc[i]
                results.append({
                    "doc_id": row["doc_id"],
                    "text": row["text"],
                    "score": float(sc),
                })
            return results

        # ------ Lexical (BM25) with candidate cap ------
        q_tokens = tokenize(query)
        # WARNING: bm25.get_scores is O(N); this is still heavy on huge corpora.
        bm25_all = np.array(self.bm25.get_scores(q_tokens), dtype="float32")

        # Candidate union: FAISS dense set + top-N lexical
        klex = max(500, k * 50)  # keep reasonable to avoid timeouts
        top_lex_idx = np.argsort(-bm25_all)[:klex]
        candidates = np.union1d(dense_idx, top_lex_idx)

        results = []
        for i in candidates:
            i = int(i)
            ds = dense_map.get(i, 0.0)  # dense score (0 if not in FAISS set)
            ls = float(bm25_all[i])     # lexical score
            score = alpha * ds + (1.0 - alpha) * ls
            row = self.df.iloc[i]
            results.append({
                "doc_id": row["doc_id"],
                "text": row["text"],
                "score": score,
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:k]
