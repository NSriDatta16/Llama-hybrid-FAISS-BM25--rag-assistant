# backend/retrieval/embeddings.py
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        print(f"✅ Embedding model loaded. Dimension = {self.dim}")

    def encode(self, texts, batch_size: int = 256):
        # convert_to_numpy + normalize for IP sim with FAISS
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if vecs.dtype != np.float32:
            vecs = vecs.astype("float32", copy=False)
        return vecs
