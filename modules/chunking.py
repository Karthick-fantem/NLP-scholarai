"""
Module 2B: Text Chunking
Sentence-aware chunking with sliding window overlap, preserving original PDF page citations.
"""

from typing import List, Dict, Any, Optional
from modules.preprocessing import AcademicTextPreprocessor


class AcademicChunker:
    """
    Creates context-preserving text chunks mapped strictly to their original PDF pages.
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120, min_chunk_len: int = 60):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_len = min_chunk_len
        self.preprocessor = AcademicTextPreprocessor()

    def chunk_document_pages(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunks the document while keeping 100% strict tracking of page numbers.
        Processes page by page or continuous flow with page tags.
        """
        chunks: List[Dict[str, Any]] = []
        chunk_id = 0

        # We first build a list of sentences tagged with their origin page
        sentence_units = []
        for page in pages_data:
            page_num = page["page_number"]
            page_text = page["text"]
            if not page_text or not page.get("has_text", True):
                continue

            sentences = self.preprocessor.tokenize_sentences(page_text)
            for s in sentences:
                sentence_units.append({
                    "sentence": s,
                    "page_number": page_num
                })

        if not sentence_units:
            return []

        # Sliding window assembly respecting sentence boundaries
        idx = 0
        n_sentences = len(sentence_units)

        while idx < n_sentences:
            curr_text_parts = []
            curr_pages = set()
            curr_len = 0
            start_idx = idx

            # Add sentences until reaching or exceeding chunk_size
            while idx < n_sentences:
                s_obj = sentence_units[idx]
                s_text = s_obj["sentence"]
                s_page = s_obj["page_number"]

                curr_text_parts.append(s_text)
                curr_pages.add(s_page)
                curr_len += len(s_text) + 1  # count space

                idx += 1
                if curr_len >= self.chunk_size:
                    break

            combined_text = " ".join(curr_text_parts).strip()

            if len(combined_text) >= self.min_chunk_len:
                sorted_pages = sorted(list(curr_pages))
                primary_page = sorted_pages[0] if sorted_pages else 1

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": combined_text,
                    "page_number": primary_page,
                    "pages": sorted_pages,
                    "char_count": len(combined_text),
                    "word_count": len(combined_text.split()),
                    "page_citation": f"Page {primary_page}" if len(sorted_pages) == 1 else f"Pages {sorted_pages[0]}-{sorted_pages[-1]}"
                })
                chunk_id += 1

            # Compute overlap rewind
            if idx >= n_sentences:
                break

            # Rewind index based on overlap target
            overlap_acc = 0
            rewind_steps = 0
            for back_idx in range(idx - 1, start_idx, -1):
                overlap_acc += len(sentence_units[back_idx]["sentence"])
                rewind_steps += 1
                if overlap_acc >= self.chunk_overlap:
                    break

            # If no rewind made progress, ensure we move forward by at least 1 sentence
            if rewind_steps > 0 and (idx - rewind_steps) > start_idx:
                idx = idx - rewind_steps

        return chunks

    def chunk_section(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunks an isolated section while preserving section context.
        """
        sec_title = section.get("section_title", "Section")
        sec_text = section.get("text", "")
        start_page = section.get("start_page", 1)

        sentences = self.preprocessor.tokenize_sentences(sec_text)
        if not sentences:
            return []

        chunks = []
        curr_parts = []
        curr_len = 0
        chunk_idx = 0

        for s in sentences:
            curr_parts.append(s)
            curr_len += len(s)
            if curr_len >= self.chunk_size:
                text = " ".join(curr_parts)
                chunks.append({
                    "chunk_id": chunk_idx,
                    "section": sec_title,
                    "text": text,
                    "page_number": start_page,
                    "pages": section.get("pages", [start_page])
                })
                chunk_idx += 1
                curr_parts = []
                curr_len = 0

        if curr_parts:
            text = " ".join(curr_parts)
            if len(text) >= self.min_chunk_len:
                chunks.append({
                    "chunk_id": chunk_idx,
                    "section": sec_title,
                    "text": text,
                    "page_number": start_page,
                    "pages": section.get("pages", [start_page])
                })

        return chunks

