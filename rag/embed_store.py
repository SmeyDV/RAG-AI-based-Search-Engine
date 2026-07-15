"""
Vector store: turn chunks into vectors and support similarity search over them.

Uses sentence-transformers for semantic embeddings (all-MiniLM-L6-v2, 384-dim).
This replaces the TF-IDF starter backend with real dense embeddings for better
retrieval quality.

Upgrade path:
- Swap the in-memory cosine_similarity search below for FAISS or Chroma once your
  chunk count grows past a few thousand.
- Keep the VectorStore interface (`build`, `query`) the same so app.py doesn't change.
"""

from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .ingest import Chunk


class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        """Encode all chunk texts into dense embeddings."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the top_k (chunk, similarity_score) pairs for a query string."""
        if self.embeddings is None:
            raise RuntimeError("VectorStore.build() must be called before query().")
        query_vec = self.model.encode([query_text])
        scores = cosine_similarity(query_vec, self.embeddings).flatten()
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[int(i)], float(scores[i])) for i in ranked_idx]
