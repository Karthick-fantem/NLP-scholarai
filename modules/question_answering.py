"""
Module 4C: Question Answering via Retrieval-Augmented Generation (RAG)
Generates strictly grounded answers to natural language questions about research papers
using FLAN-T5 via direct seq2seq generation, citing supporting PDF pages and strictly preventing hallucination.
"""

from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import config
from modules.embeddings import EmbeddingModel
from modules.retriever import DocumentRetriever


class RAGQuestionAnswerer:
    """
    RAG-based Question Answering engine designed for academic literature.
    """

    def __init__(
        self,
        model_name: str = config.QA_GENERATOR_MODEL,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._is_loaded = False

    def load_model(self):
        """
        Loads the Seq2Seq generation model (FLAN-T5) on demand from local cache.
        """
        if self._is_loaded and self.model is not None:
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=True)
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
        except Exception as e:
            print(f"Loading {self.model_name} offline note: {e}. Trying fallback...")
            self.model_name = "google/flan-t5-small"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=True)
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True

    def build_grounded_prompt(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Assembles strict RAG prompt for FLAN-T5 zero-shot QA.
        """
        context_text = " ".join([c["text"] for c in retrieved_chunks])
        words = context_text.split()
        if len(words) > 450:
            context_text = " ".join(words[:450])

        prompt = f"Read the research paper text below and answer the question.\n\nText:\n{context_text}\n\nBased on the text above, {question}"
        return prompt

    def answer_question(
        self,
        question: str,
        retriever: DocumentRetriever,
        embedding_model: EmbeddingModel,
        top_k: int = config.DEFAULT_TOP_K
    ) -> Dict[str, Any]:
        """
        Executes end-to-end RAG pipeline:
        1. Encodes user question.
        2. Retrieves relevant chunks with page citations.
        3. Enforces relevance threshold.
        4. Synthesizes grounded answer.
        """
        if not question.strip():
            return {
                "question": question,
                "answer": "Please provide a valid question.",
                "is_grounded": False,
                "evidence": [],
                "citations": []
            }

        # 1. Embed question and retrieve candidates
        q_emb = embedding_model.embed_query(question)
        retrieved_chunks = retriever.retrieve(q_emb, top_k=top_k)

        if not retrieved_chunks:
            return {
                "question": question,
                "answer": config.FALLBACK_NOT_FOUND_MESSAGE,
                "is_grounded": False,
                "evidence": [],
                "citations": [],
                "max_similarity": 0.0
            }

        max_sim = retrieved_chunks[0]["similarity_score"]

        # 2. Strict Grounding Check: If highest similarity is below threshold, return fallback
        if max_sim < config.RELEVANCE_SIMILARITY_THRESHOLD:
            return {
                "question": question,
                "answer": config.FALLBACK_NOT_FOUND_MESSAGE,
                "is_grounded": False,
                "evidence": retrieved_chunks,
                "citations": [],
                "max_similarity": max_sim,
                "reason": f"Top similarity score ({max_sim:.2f}) fell below confidence threshold ({config.RELEVANCE_SIMILARITY_THRESHOLD:.2f})."
            }

        # Filter only relevant chunks for context
        relevant_chunks = [c for c in retrieved_chunks if c["is_relevant"]]
        if not relevant_chunks:
            relevant_chunks = retrieved_chunks[:1]

        # 3. Build Prompt and Generate Answer using direct model.generate()
        self.load_model()
        prompt = self.build_grounded_prompt(question, relevant_chunks)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=80,
                    min_length=2,
                    num_beams=2,
                    no_repeat_ngram_size=2
                )
            raw_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            # Clean possible repetitive answer prefixes
            import re
            raw_answer = re.sub(r"^(Answer:\s*)+", "", raw_answer, flags=re.IGNORECASE).strip()
            print(f"DEBUG_RAW_GENERATION: '{raw_answer}'")
        except Exception as e:
            raw_answer = f"Error during neural generation: {str(e)}"

        # 4. Check for unanswerability markers
        unanswerable_cues = [
            "not found",
            "cannot be found",
            "not mentioned",
            "no information",
            "does not provide",
            "unanswerable"
        ]

        if any(cue in raw_answer.lower() for cue in unanswerable_cues) or len(raw_answer) < 3:
            final_answer = config.FALLBACK_NOT_FOUND_MESSAGE
            is_grounded = False
        else:
            final_answer = raw_answer
            is_grounded = True

        # Extract unique citation pages
        cited_pages = sorted(list(set(c["page_number"] for c in relevant_chunks)))
        citations_str = ", ".join([f"Page {p}" for p in cited_pages])

        return {
            "question": question,
            "answer": final_answer,
            "is_grounded": is_grounded,
            "evidence": relevant_chunks,
            "citations": cited_pages,
            "citations_display": citations_str,
            "max_similarity": max_sim,
            "model_used": self.model_name
        }
