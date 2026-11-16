"""
Module 3B: Abstractive Summarization via Transformer Models
Implements abstractive summarization using Hugging Face Transformers
(DistilBART / FLAN-T5) with direct model.generate() for fast, reliable offline execution.
"""

import os
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import config


class AbstractiveSummarizer:
    """
    Abstractive neural summarizer leveraging sequence-to-sequence transformers.
    Supports short, detailed, and section-wise academic summaries.
    """

    def __init__(
        self,
        model_name: str = config.ABSTRACTIVE_SUMMARY_MODEL,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._is_loaded = False

    def load_model(self):
        """
        Lazy-loads the model and tokenizer from local cache.
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
            # Fallback to local flan-t5-small
            print(f"Loading {self.model_name} offline note: {e}. Switching to flan-t5-small...")
            self.model_name = "google/flan-t5-small"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=True)
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True

    def summarize_chunk(
        self,
        text: str,
        max_length: int = 120,
        min_length: int = 25,
        num_beams: int = 2
    ) -> str:
        """
        Summarizes an individual text passage using direct model generation.
        """
        self.load_model()
        if not text or len(text.strip().split()) < 15:
            return text.strip()

        # If using T5, prefix with instruction
        prefix = "summarize: " if "t5" in self.model_name.lower() else ""
        prompt_input = prefix + text

        try:
            inputs = self.tokenizer(
                prompt_input,
                return_tensors="pt",
                max_length=600,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )

            summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
            return summary_text
        except Exception as e:
            return f"[Summarization Note: {str(e)}]"

    def summarize_document(
        self,
        full_text: str,
        mode: str = "detailed"
    ) -> Dict[str, Any]:
        """
        Hierarchical summarization for full research papers.
        mode: 'short' (~60 words) or 'detailed' (~150 words).
        """
        self.load_model()
        words = full_text.split()
        total_word_count = len(words)

        if total_word_count == 0:
            return {"summary": "", "mode": mode, "chunks_processed": 0}

        if mode == "short":
            target_max = 75
            target_min = 25
            chunk_target = 50
        else:
            target_max = 160
            target_min = 55
            chunk_target = 80

        if total_word_count <= 500:
            direct_summary = self.summarize_chunk(
                full_text,
                max_length=target_max,
                min_length=target_min
            )
            return {
                "summary": direct_summary,
                "mode": mode,
                "chunks_processed": 1,
                "original_word_count": total_word_count,
                "summary_word_count": len(direct_summary.split())
            }

        # Long document: chunk and hierarchical summarize
        chunk_size = 450
        text_chunks = []
        for i in range(0, total_word_count, chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            text_chunks.append(chunk)

        # Select most salient chunks: intro, methods, conclusion
        selected_chunks = text_chunks[:2] + text_chunks[-2:] if len(text_chunks) > 4 else text_chunks

        chunk_summaries = []
        for ch in selected_chunks:
            if len(ch.split()) >= 20:
                s = self.summarize_chunk(ch, max_length=chunk_target, min_length=20)
                chunk_summaries.append(s)

        combined_intermediate = " ".join(chunk_summaries)
        final_summary = self.summarize_chunk(
            combined_intermediate,
            max_length=target_max,
            min_length=target_min
        )

        return {
            "summary": final_summary,
            "mode": mode,
            "chunks_processed": len(selected_chunks),
            "original_word_count": total_word_count,
            "summary_word_count": len(final_summary.split())
        }

    def summarize_sections(
        self,
        sections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generates individual abstractive summaries for distinct detected sections.
        """
        self.load_model()
        section_summaries = []

        for sec in sections:
            sec_title = sec["section_title"]
            if "reference" in sec_title.lower() or "bibliography" in sec_title.lower():
                continue

            sec_text = sec["text"]
            if len(sec_text.split()) < 25:
                continue

            sec_summary = self.summarize_chunk(
                sec_text,
                max_length=95,
                min_length=20
            )

            section_summaries.append({
                "section_title": sec_title,
                "start_page": sec.get("start_page", 1),
                "summary": sec_summary,
                "original_words": len(sec_text.split()),
                "summary_words": len(sec_summary.split())
            })

        return section_summaries
