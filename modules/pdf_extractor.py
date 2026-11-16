"""
Module 1: PDF Upload and Text Extraction
Extracts text page-by-page from academic PDFs using PyMuPDF (fitz),
preserves page numbers, detects scanned/empty documents, and prepares OCR hooks.
"""

import io
import re
from typing import Dict, List, Any, Optional, Tuple, Union
import pymupdf  # PyMuPDF


class PDFExtractor:
    """
    Handles robust PDF text extraction, metadata parsing, and scanned document detection.
    """

    def __init__(self, ocr_enabled: bool = False):
        self.ocr_enabled = ocr_enabled

    def extract_from_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
        """
        Extract text and structure from a PDF byte stream (e.g., uploaded via Streamlit).
        """
        stream = io.BytesIO(pdf_bytes)
        return self._process_doc(pymupdf.open(stream=stream, filetype="pdf"), filename=filename)

    def extract_from_path(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text and structure from a local PDF file path.
        """
        return self._process_doc(pymupdf.open(file_path), filename=file_path)

    def _process_doc(self, doc: pymupdf.Document, filename: str) -> Dict[str, Any]:
        """
        Internal worker that extracts text page-by-page while preserving page numbers.
        """
        pages_data: List[Dict[str, Any]] = []
        total_chars = 0
        total_words = 0
        empty_pages = 0

        # Read document metadata
        meta = doc.metadata or {}
        doc_title = meta.get("title", "").strip()
        doc_author = meta.get("author", "").strip()

        for page_idx in range(len(doc)):
            page_num = page_idx + 1  # 1-indexed for standard citation
            page = doc.load_page(page_idx)
            text = page.get_text("text")

            # Basic whitespace normalization per page
            clean_text = self._basic_clean(text)
            char_count = len(clean_text)
            words = clean_text.split()
            word_count = len(words)

            if word_count == 0:
                empty_pages += 1

            pages_data.append({
                "page_number": page_num,
                "text": clean_text,
                "char_count": char_count,
                "word_count": word_count,
                "has_text": word_count > 0
            })

            total_chars += char_count
            total_words += word_count

        total_pages = len(doc)
        doc.close()

        # Detect Scanned / Non-selectable PDF
        is_scanned = (total_words < 15 and total_pages > 0) or (empty_pages == total_pages and total_pages > 0)
        
        # Title heuristic if metadata title is absent or generic
        if not doc_title and len(pages_data) > 0 and pages_data[0]["has_text"]:
            first_lines = [l.strip() for l in pages_data[0]["text"].split("\n") if len(l.strip()) > 5]
            if first_lines:
                doc_title = first_lines[0][:150]

        warning_message = None
        if is_scanned:
            warning_message = (
                "Warning: This document contains little or no selectable digital text. "
                "It appears to be a scanned image-based PDF. "
                "Text-based NLP analysis requires machine-readable text or an Optical Character Recognition (OCR) layer."
            )

        full_document_text = "\n\n".join(
            [f"--- Page {p['page_number']} ---\n{p['text']}" for p in pages_data if p['has_text']]
        )

        return {
            "filename": filename,
            "title": doc_title or "Untitled Research Paper",
            "author": doc_author or "Unknown Author",
            "total_pages": total_pages,
            "empty_pages": empty_pages,
            "total_chars": total_chars,
            "total_words": total_words,
            "is_scanned": is_scanned,
            "warning": warning_message,
            "pages": pages_data,
            "full_text": full_document_text
        }

    @staticmethod
    def _basic_clean(text: str) -> str:
        """
        Removes binary/null bytes and unifies unicode newlines without losing academic content.
        """
        if not text:
            return ""
        # Remove null characters and non-printable control codes
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Normalize newlines
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        return cleaned.strip()

    def ocr_fallback(self, page_image_bytes: bytes) -> str:
        """
        Placeholder hook for OCR engine (e.g., Tesseract / EasyOCR).
        Future enhancement ready for MCA viva demonstration.
        """
        if not self.ocr_enabled:
            return ""
        # In a production environment with pytesseract installed:
        # from PIL import Image
        # import pytesseract
        # return pytesseract.image_to_string(Image.open(io.BytesIO(page_image_bytes)))
        return "[OCR Not configured in current environment]"

