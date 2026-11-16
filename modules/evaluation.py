"""
Module 5: Experimental Evaluation Metrics
Implements standard academic NLP evaluation metrics:
- Summarization: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore
- Question Answering: SQuAD-style Exact Match (EM), Token-level F1
- Retrieval: Precision@K, Recall@K, Mean Reciprocal Rank (MRR)
- Grounding: Faithfulness, Citation accuracy, Latency
"""

import re
import string
import time
from typing import List, Dict, Any, Set, Tuple, Optional
import numpy as np

# ROUGE
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

# BERTScore
try:
    import bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


def normalize_answer(s: str) -> str:
    """
    Standard SQuAD text normalization: lowercase, remove punctuation, remove articles, unify whitespace.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Computes Exact Match (EM) metric (1.0 if identical normalized strings, else 0.0).
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """
    Computes token-level F1 score between prediction and ground-truth answer strings.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return float(prediction_tokens == ground_truth_tokens)

    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_same = sum(min(prediction_tokens.count(w), ground_truth_tokens.count(w)) for w in common)

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(float(f1), 4)


class SummarizationEvaluator:
    """
    Evaluator for text summarization systems.
    """

    def __init__(self):
        if ROUGE_AVAILABLE:
            self.scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"],
                use_stemmer=True
            )
        else:
            self.scorer = None

    def evaluate_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
        """
        if not self.scorer or not prediction or not reference:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        scores = self.scorer.score(reference, prediction)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4)
        }

    def evaluate_bertscore(
        self,
        predictions: List[str],
        references: List[str],
        lang: str = "en"
    ) -> Dict[str, float]:
        """
        Computes semantic BERTScore (Precision, Recall, F1) across candidate and reference pairs.
        """
        if not BERTSCORE_AVAILABLE or not predictions or not references:
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}

        try:
            P, R, F1 = bert_score.score(
                predictions,
                references,
                lang=lang,
                verbose=False,
                rescale_with_baseline=False
            )
            return {
                "bertscore_precision": round(float(P.mean()), 4),
                "bertscore_recall": round(float(R.mean()), 4),
                "bertscore_f1": round(float(F1.mean()), 4)
            }
        except Exception as e:
            print(f"BERTScore computation note: {e}")
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}


class RetrievalEvaluator:
    """
    Evaluator for passage retrieval pipelines.
    Computes Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).
    """

    @staticmethod
    def precision_at_k(retrieved_ids: List[Any], relevant_ids: Set[Any], k: int) -> float:
        """
        Precision@K = (# of relevant items in top-k) / K
        """
        if k <= 0:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_in_top_k = sum(1 for item in top_k if item in relevant_ids)
        return round(float(relevant_in_top_k) / k, 4)

    @staticmethod
    def recall_at_k(retrieved_ids: List[Any], relevant_ids: Set[Any], k: int) -> float:
        """
        Recall@K = (# of relevant items in top-k) / (Total relevant items)
        """
        if not relevant_ids or k <= 0:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_in_top_k = sum(1 for item in top_k if item in relevant_ids)
        return round(float(relevant_in_top_k) / len(relevant_ids), 4)

    @staticmethod
    def reciprocal_rank(retrieved_ids: List[Any], relevant_ids: Set[Any]) -> float:
        """
        Reciprocal Rank (RR) = 1 / rank of first relevant item.
        """
        for rank, item in enumerate(retrieved_ids, start=1):
            if item in relevant_ids:
                return round(1.0 / rank, 4)
        return 0.0

    @classmethod
    def evaluate_retrieval_batch(
        cls,
        queries_results: List[Dict[str, Any]],
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, float]:
        """
        Evaluates a batch of retrieval queries and returns mean metrics.
        Each entry in queries_results should contain:
        - 'retrieved': list of item IDs
        - 'relevant': set or list of relevant item IDs
        """
        if not queries_results:
            return {}

        metrics: Dict[str, List[float]] = {f"P@{k}": [] for k in k_values}
        metrics.update({f"R@{k}": [] for k in k_values})
        metrics["MRR"] = []

        for item in queries_results:
            ret = item.get("retrieved", [])
            rel = set(item.get("relevant", []))

            for k in k_values:
                metrics[f"P@{k}"].append(cls.precision_at_k(ret, rel, k))
                metrics[f"R@{k}"].append(cls.recall_at_k(ret, rel, k))

            metrics["MRR"].append(cls.reciprocal_rank(ret, rel))

        summary = {}
        for metric_name, values in metrics.items():
            summary[metric_name] = round(float(np.mean(values)), 4) if values else 0.0

        return summary


class FaithfulnessEvaluator:
    """
    Evaluates whether an answer is grounded in retrieved context passages
    by calculating lexical overlap and named entity/keyword presence.
    """

    @staticmethod
    def compute_lexical_overlap(answer: str, context: str) -> float:
        """
        Computes token unigram recall: fraction of content words in answer present in context.
        """
        a_clean = normalize_answer(answer).split()
        c_clean = set(normalize_answer(context).split())

        if not a_clean:
            return 1.0  # Empty answer is trivially faithful or N/A

        overlap = sum(1 for w in a_clean if w in c_clean)
        return round(float(overlap) / len(a_clean), 4)

