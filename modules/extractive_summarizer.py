"""
Module 3A: Extractive Summarization via TextRank
Implements the graph-based TextRank algorithm (Mihalcea & Tarau, 2004)
for unsupervised salient sentence extraction from research papers.
"""

import math
import re
from typing import List, Dict, Any, Optional
import networkx as nx
import numpy as np
import nltk
from nltk.corpus import stopwords
from modules.preprocessing import AcademicTextPreprocessor

# Ensure stopwords are loaded
try:
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    nltk.download("stopwords", quiet=True)
    try:
        STOPWORDS = set(stopwords.words("english"))
    except Exception:
        STOPWORDS = {"the", "a", "an", "in", "on", "of", "to", "for", "and", "is", "was", "are"}


class TextRankSummarizer:
    """
    Extractive summarization using the classic graph-based TextRank algorithm.
    Builds a complete sentence-similarity graph and ranks nodes using PageRank.
    """

    def __init__(self, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-4):
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.preprocessor = AcademicTextPreprocessor()

    def _sentence_tokens(self, sentence: str) -> List[str]:
        """
        Tokenizes and removes punctuation and stopwords for sentence similarity calculation.
        """
        words = re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower())
        return [w for w in words if w not in STOPWORDS]

    def _similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """
        Mihalcea & Tarau (2004) TextRank sentence similarity:
        Sim(S_i, S_j) = |tokens(S_i) ∩ tokens(S_j)| / (log(|tokens(S_i)|) + log(|tokens(S_j)|))
        """
        if not tokens1 or not tokens2:
            return 0.0

        common_count = len(set(tokens1).intersection(set(tokens2)))
        if common_count == 0:
            return 0.0

        len1 = len(tokens1)
        len2 = len(tokens2)

        # Avoid log(1) == 0 division by adding small epsilon or checking length > 1
        denom = (math.log(max(len1, 2)) + math.log(max(len2, 2)))
        return float(common_count) / denom if denom > 0 else 0.0

    def summarize(
        self,
        text: str,
        num_sentences: int = 5,
        preserve_order: bool = True
    ) -> Dict[str, Any]:
        """
        Extracts top-ranked sentences from raw text.
        """
        sentences = self.preprocessor.tokenize_sentences(text)
        if len(sentences) == 0:
            return {
                "summary": "",
                "selected_sentences": [],
                "scores": [],
                "sentence_count": 0
            }

        if len(sentences) <= num_sentences:
            return {
                "summary": " ".join(sentences),
                "selected_sentences": sentences,
                "scores": [1.0] * len(sentences),
                "sentence_count": len(sentences)
            }

        # Tokenize each sentence
        tokenized_sentences = [self._sentence_tokens(s) for s in sentences]

        # Construct weighted sentence graph
        graph = nx.Graph()
        n = len(sentences)
        for i in range(n):
            graph.add_node(i)

        for i in range(n):
            for j in range(i + 1, n):
                sim = self._similarity(tokenized_sentences[i], tokenized_sentences[j])
                if sim > 0.01:
                    graph.add_edge(i, j, weight=sim)

        # Run PageRank
        try:
            pagerank_scores = nx.pagerank(
                graph,
                alpha=self.damping,
                max_iter=self.max_iter,
                tol=self.tol,
                weight="weight"
            )
        except Exception:
            # Fallback if graph is disconnected / degenerate: uniform scoring
            pagerank_scores = {i: 1.0 / n for i in range(n)}

        # Sort sentences by PageRank score
        ranked_indices = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in ranked_indices[:num_sentences]]

        if preserve_order:
            top_indices = sorted(top_indices)

        selected_sentences = [sentences[idx] for idx in top_indices]
        selected_scores = [pagerank_scores[idx] for idx in top_indices]

        summary_text = " ".join(selected_sentences)

        return {
            "summary": summary_text,
            "selected_sentences": selected_sentences,
            "scores": selected_scores,
            "sentence_count": len(selected_sentences)
        }

    def summarize_pages(
        self,
        pages_data: List[Dict[str, Any]],
        num_sentences: int = 6
    ) -> Dict[str, Any]:
        """
        Extractive summarization across pages while preserving sentence page numbers.
        """
        all_sentences = []
        for p in pages_data:
            p_num = p["page_number"]
            s_list = self.preprocessor.tokenize_sentences(p["text"])
            for s in s_list:
                all_sentences.append({"sentence": s, "page_number": p_num})

        if not all_sentences:
            return {"summary": "", "citations": [], "sentence_count": 0}

        plain_text = " ".join([item["sentence"] for item in all_sentences])
        result = self.summarize(plain_text, num_sentences=num_sentences, preserve_order=True)

        # Map selected sentences back to page citations
        citations = []
        for s in result["selected_sentences"]:
            matched_page = next((item["page_number"] for item in all_sentences if item["sentence"] == s), 1)
            citations.append({"sentence": s, "page": matched_page})

        return {
            "summary": result["summary"],
            "selected_sentences": result["selected_sentences"],
            "citations": citations,
            "sentence_count": len(result["selected_sentences"])
        }

