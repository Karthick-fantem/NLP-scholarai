"""
Module 4B: Semantic and Lexical Retrieval
Implements FAISS dense index retrieval, cosine similarity fallback,
and TF-IDF lexical retrieval baseline for scientific literature search.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import config

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class DocumentRetriever:
    """
    Retrieves the most semantically relevant document chunks given a natural language query.
    Tracks page provenance and enforces relevance thresholds.
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        chunk_embeddings: Optional[np.ndarray] = None,
        similarity_threshold: float = config.RELEVANCE_SIMILARITY_THRESHOLD,
        use_faiss: bool = True
    ):
        self.chunks = chunks
        self.similarity_threshold = similarity_threshold
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.embeddings = chunk_embeddings
        self.index = None

        if self.embeddings is not None and len(self.embeddings) > 0:
            self._build_index()

    def _build_index(self):
        """
        Builds FAISS inner-product index (equivalent to cosine similarity on normalized vectors).
        """
        if self.use_faiss:
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings.astype(np.float32))

    def update_embeddings(self, embeddings: np.ndarray):
        """
        Updates the stored embeddings and rebuilds the index.
        """
        self.embeddings = embeddings
        self._build_index()

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = config.DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k chunks ranked by semantic cosine similarity.
        """
        if not self.chunks or self.embeddings is None or len(self.embeddings) == 0:
            return []

        top_k = min(top_k, len(self.chunks))

        if self.use_faiss and self.index is not None:
            # Query FAISS
            q = np.expand_dims(query_embedding.astype(np.float32), axis=0)
            distances, indices = self.index.search(q, top_k)
            retrieved_scores = distances[0]
            retrieved_indices = indices[0]
        else:
            # NumPy / scikit-learn cosine similarity fallback
            sims = np.dot(self.embeddings, query_embedding)
            retrieved_indices = np.argsort(sims)[::-1][:top_k]
            retrieved_scores = sims[retrieved_indices]

        results = []
        for rank, (idx, score) in enumerate(zip(retrieved_indices, retrieved_scores)):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = dict(self.chunks[idx])
            float_score = float(score)
            chunk["rank"] = rank + 1
            chunk["similarity_score"] = round(float_score, 4)
            chunk["is_relevant"] = float_score >= self.similarity_threshold
            results.append(chunk)

        return results


class TFIDFRetriever:
    """
    Lexical baseline retriever using Term Frequency-Inverse Document Frequency (TF-IDF).
    Used for comparative ablation experiments (Dense vs. Sparse Retrieval).
    """

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = None

        if self.chunks:
            texts = [c["text"] for c in self.chunks]
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Retrieves top-k chunks by TF-IDF cosine similarity.
        """
        if not self.chunks or self.tfidf_matrix is None:
            return []

        top_k = min(top_k, len(self.chunks))
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []

        for rank, idx in enumerate(top_indices):
            chunk = dict(self.chunks[idx])
            chunk["rank"] = rank + 1
            chunk["similarity_score"] = round(float(sims[idx]), 4)
            chunk["is_relevant"] = float(sims[idx]) > 0.05
            results.append(chunk)

        return results

