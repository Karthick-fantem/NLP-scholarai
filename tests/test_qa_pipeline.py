"""
Unit Tests for Summarization, QA, and Evaluation
"""

import unittest
from modules.extractive_summarizer import TextRankSummarizer
from modules.evaluation import (
    compute_exact_match,
    compute_token_f1,
    SummarizationEvaluator,
    RetrievalEvaluator,
    FaithfulnessEvaluator
)


class TestSummarizationAndQA(unittest.TestCase):

    def setUp(self):
        self.textrank = TextRankSummarizer()
        self.evaluator = SummarizationEvaluator()

    def test_textrank_summarizer(self):
        sample_doc = (
            "Natural language processing allows computers to comprehend human language. "
            "Transformers have revolutionized the field of scientific literature comprehension. "
            "Extractive summarization ranks sentences based on graph centrality. "
            "Abstractive summarization synthesizes novel phrasings. "
            "Deep neural networks achieve high benchmark scores across domain datasets."
        )
        res = self.textrank.summarize(sample_doc, num_sentences=2)
        self.assertEqual(res["sentence_count"], 2)
        self.assertGreater(len(res["summary"]), 20)

    def test_evaluation_metrics_qa(self):
        pred = "Sparse-Scale Multi-Head Attention"
        ref = "sparse-scale multi-head attention"
        self.assertEqual(compute_exact_match(pred, ref), 1.0)

        pred_f1 = "Multi-Head Attention"
        ref_f1 = "Sparse Multi-Head Attention"
        f1 = compute_token_f1(pred_f1, ref_f1)
        self.assertGreater(f1, 0.70)

    def test_evaluation_metrics_summarization(self):
        cand = "Transformers revolutionized natural language processing."
        ref = "Transformers revolutionized the field of natural language processing."
        scores = self.evaluator.evaluate_rouge(cand, ref)
        self.assertIn("rouge1", scores)
        self.assertIn("rouge2", scores)
        self.assertIn("rougeL", scores)
        self.assertGreater(scores["rouge1"], 0.5)

    def test_retrieval_metrics(self):
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc2"}
        p1 = RetrievalEvaluator.precision_at_k(retrieved, relevant, k=1)
        p3 = RetrievalEvaluator.precision_at_k(retrieved, relevant, k=3)
        rr = RetrievalEvaluator.reciprocal_rank(retrieved, relevant)

        self.assertEqual(p1, 0.0)
        self.assertAlmostEqual(p3, 1.0 / 3, places=3)
        self.assertEqual(rr, 0.5)  # Found at rank 2 -> 1/2 = 0.5


if __name__ == "__main__":
    unittest.main()

