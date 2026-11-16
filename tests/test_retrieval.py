"""
Unit Tests for Module 4: Embeddings and Retrieval
"""

import unittest
import numpy as np
from modules.embeddings import EmbeddingModel
from modules.retriever import DocumentRetriever, TFIDFRetriever


class TestRetrieval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.embedding_model = EmbeddingModel()
        cls.embedding_model.load_model()

    def setUp(self):
        self.mock_chunks = [
            {"chunk_id": 0, "text": "Multi-head attention allows the model to jointly attend to information from different representation subspaces.", "page_number": 1},
            {"chunk_id": 1, "text": "We evaluate on SciTLDR and QASPER scientific paper benchmarks on CPU architecture.", "page_number": 3},
            {"chunk_id": 2, "text": "The learning rate was initialized to 5e-4 with AdamW optimizer.", "page_number": 3}
        ]
        texts = [c["text"] for c in self.mock_chunks]
        self.embeddings = self.embedding_model.embed_texts(texts)
        self.retriever = DocumentRetriever(self.mock_chunks, self.embeddings, use_faiss=True)
        self.tfidf_retriever = TFIDFRetriever(self.mock_chunks)

    def test_embedding_dimensions(self):
        self.assertEqual(self.embeddings.shape, (3, 384))
        # Verify L2 normalization
        norms = np.linalg.norm(self.embeddings, axis=1)
        for n in norms:
            self.assertAlmostEqual(n, 1.0, places=4)

    def test_semantic_retrieval_ranking(self):
        query = "How does multi-head attention operate?"
        q_emb = self.embedding_model.embed_query(query)
        results = self.retriever.retrieve(q_emb, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], 0)
        self.assertEqual(results[0]["page_number"], 1)
        self.assertTrue(results[0]["is_relevant"])

    def test_out_of_domain_query_threshold(self):
        query = "What is the capital city of Australia?"
        q_emb = self.embedding_model.embed_query(query)
        results = self.retriever.retrieve(q_emb, top_k=1)

        self.assertLess(results[0]["similarity_score"], 0.30)

    def test_tfidf_retriever(self):
        results = self.tfidf_retriever.retrieve("SciTLDR benchmark", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], 1)


if __name__ == "__main__":
    unittest.main()

