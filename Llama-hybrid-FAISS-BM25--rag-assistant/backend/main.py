# backend/main.py
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# our schemas & retrieval
from backend.models.schemas import AskRequest, AskResponse, RetrievedChunk
from backend.retrieval.embeddings import EmbeddingManager
from backend.retrieval.hybrid import HybridIndex

# Optional OpenAI (via LlamaIndex) for /ask
try:
    from llama_index.llms.openai import OpenAI
    from llama_index.core import PromptTemplate
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# Groq LLM + simple RAG for /ask_groq
from backend.llm_groq import get_groq_llm, rag_simple as groq_rag_simple

# --- Force-load env from backend/.env ---
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# -------- FastAPI app --------
app = FastAPI(title="Hybrid RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock this down for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Config --------
INDEX_DIR = os.getenv("INDEX_DIR", "backend/index_wiki20k")
DEFAULT_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))

# -------- Lazy-load globals --------
_EMB = None
_IDX = None


def ensure_loaded():
    """Load the embedder and hybrid index once, reuse thereafter."""
    global _EMB, _IDX
    if _EMB is None:
        _EMB = EmbeddingManager()
    if _IDX is None:
        _IDX = HybridIndex(base_dir=INDEX_DIR, dim=_EMB.dim)
        _IDX.load()
    return _EMB, _IDX


@app.get("/health")
def health():
    return {"status": "ok", "index_dir": INDEX_DIR}


# ---------- OpenAI route (optional) ----------
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    OpenAI-backed answer generation using LlamaIndex.
    If OpenAI isn't configured, returns a graceful message.
    """
    emb, idx = ensure_loaded()
    alpha = req.alpha or DEFAULT_ALPHA

    # Retrieve from hybrid index
    rows = idx.search(req.query, emb.encode, k=req.top_k, alpha=alpha)
    context = "\n\n".join([r["text"] for r in rows])
    citations = [
        RetrievedChunk(doc_id=r["doc_id"], score=float(r["score"]), text=r["text"])
        for r in rows
    ]

    if not _HAS_OPENAI or not os.getenv("OPENAI_API_KEY"):
        msg = (
            "OpenAI not configured. Set OPENAI_API_KEY and (optionally) OPENAI_MODEL "
            "in backend/.env, or use the /ask_groq route from the UI."
        )
        return AskResponse(answer=msg, citations=citations, used_alpha=alpha)

    llm = OpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    tmpl = PromptTemplate(
        "You are a precise assistant. Use ONLY the context.\n"
        "If unsure, say you don't know.\n\n"
        "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    answer = llm.complete(tmpl.format(context=context, query=req.query)).text

    return AskResponse(answer=answer, citations=citations, used_alpha=alpha)


# ---------- Groq route ----------
class HybridRetriever:
    """
    Small adapter so Groq RAG can call `retrieve_query(...)`
    and get list of objects with page_content/metadata/score.
    """
    def __init__(self, index: HybridIndex, embedder: EmbeddingManager):
        self.index = index
        self.embedder = embedder

    def retrieve_query(self, query: str, top_k: int = 5):
        rows = self.index.search(
            query,
            self.embedder.encode,
            k=top_k,
            alpha=float(os.getenv("HYBRID_ALPHA", DEFAULT_ALPHA)),
        )

        class Doc:
            pass

        docs = []
        for r in rows:
            d = Doc()
            d.page_content = r["text"]
            d.metadata = {"doc_id": r["doc_id"]}
            d.score = r["score"]
            docs.append(d)
        return docs


@app.post("/ask_groq", response_model=AskResponse)
def ask_groq(req: AskRequest):
    """
    Groq-backed answer generation using rag_simple() in llm_groq.py
    for text generation and the same hybrid retriever for citations.
    """
    emb, idx = ensure_loaded()
    alpha = req.alpha or DEFAULT_ALPHA

    # Adapter retriever for Groq
    retriever = HybridRetriever(idx, emb)

    # Fetch rows for citations first (even if LLM fails, you see context)
    rows = idx.search(req.query, emb.encode, k=req.top_k, alpha=alpha)
    citations = [
        RetrievedChunk(doc_id=r["doc_id"], score=float(r["score"]), text=r["text"])
        for r in rows
    ]

    try:
        groq_llm = get_groq_llm(req.model)   # model can come from UI or default
        answer = groq_rag_simple(req.query, retriever, groq_llm, top_k=req.top_k)
        if not answer.strip():
            answer = "(Groq) The model returned an empty response."
    except Exception as e:
        # Never 500 — show the reason in the answer
        answer = f"(Groq) Could not generate an answer: {e}"

    return AskResponse(answer=answer, citations=citations, used_alpha=alpha)

