"""
Module 4A: Dense Sentence Embeddings
Generates dense vector embeddings using Sentence Transformers (all-MiniLM-L6-v2).
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import config


class EmbeddingModel:
    """
    Manages sentence embedding generation with L2 normalization for fast cosine similarity.
    """

    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model: SentenceTransformer = None
        self._is_loaded = False

    def load_model(self):
        """
        Lazy-loads the SentenceTransformer model from local cache.
        """
        if self._is_loaded and self.model is not None:
            return
        try:
            self.model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(self.model_name)
        self._is_loaded = True

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encodes a list of passages into normalized float32 embeddings.
        """
        self.load_model()
        if not texts:
            return np.empty((0, config.EMBEDDING_DIM), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # Crucial: enables dot product as cosine similarity
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encodes a single query into a 1D normalized float32 embedding.
        """
        self.load_model()
        emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return emb[0].astype(np.float32)

