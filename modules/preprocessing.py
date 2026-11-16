"""
Module 2A: Text Preprocessing
Cleans academic PDF text, repairs line-break hyphenations, preserves scientific
notations/citations, and identifies academic section headers.
"""

import re
from typing import List, Dict, Tuple, Optional, Any
import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK punkt is initialized safely
try:
    _ = sent_tokenize("Test sentence.")
except Exception:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


class AcademicTextPreprocessor:
    """
    Preprocessor tailored specifically for academic research literature.
    """

    # Common canonical academic paper section headers
    SECTION_PATTERNS = [
        (r"(?i)\b(abstract)\b", "Abstract"),
        (r"(?i)\b(\d+[\.\s]+)?(introduction)\b", "Introduction"),
        (r"(?i)\b(\d+[\.\s]+)?(related\s+work|literature\s+review|background)\b", "Related Work"),
        (r"(?i)\b(\d+[\.\s]+)?(methodology|methods|proposed\s+(method|approach|system|architecture)|system\s+model)\b", "Methodology"),
        (r"(?i)\b(\d+[\.\s]+)?(experiments?|experimental\s+setup|evaluation|implementation)\b", "Experiments & Evaluation"),
        (r"(?i)\b(\d+[\.\s]+)?(results?|findings|results\s+and\s+discussion)\b", "Results & Discussion"),
        (r"(?i)\b(\d+[\.\s]+)?(discussion)\b", "Discussion"),
        (r"(?i)\b(\d+[\.\s]+)?(conclusion|conclusions|concluding\s+remarks)\b", "Conclusion"),
        (r"(?i)\b(\d+[\.\s]+)?(future\s+work|limitations)\b", "Future Work & Limitations"),
        (r"(?i)\b(\d+[\.\s]+)?(references|bibliography)\b", "References")
    ]

    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        """
        Applies academic text cleaning:
        1. Joins hyphenated words broken across line wraps (e.g. 'classifi-\ncation' -> 'classification').
        2. Normalizes multiple spaces and excessive blank lines.
        3. Preserves mathematical expressions, Greek symbols, brackets, and citations.
        """
        if not text:
            return ""

        # Repair hyphenated word wrap: 'exam-\nple' -> 'example'
        cleaned = re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)", r"\1\2", text)

        # Remove header/footer line noise like bare page numbers on single lines
        cleaned = re.sub(r"(?m)^\s*\d+\s*$", "", cleaned)

        # Normalize line breaks: replace solitary newlines within sentences with spaces
        lines = cleaned.split("\n")
        rebuilt_lines = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                rebuilt_lines.append("")
            else:
                rebuilt_lines.append(line_str)

        text_unified = "\n".join(rebuilt_lines)
        # Collapse 3 or more consecutive newlines into 2
        text_unified = re.sub(r"\n{3,}", "\n\n", text_unified)

        # Normalize inline whitespace (multiple spaces/tabs to single space)
        text_unified = re.sub(r"[ \t]+", " ", text_unified)

        return text_unified.strip()

    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Splits text into coherent sentences using NLTK, handling academic abbreviations
        (e.g., 'et al.', 'i.e.', 'e.g.', 'Fig.', 'Eq.').
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        # Mask common academic abbreviations before sentence tokenization
        # to prevent erroneous sentence breaks
        replacements = {
            "et al.": "et al<DOT>",
            "Et al.": "Et al<DOT>",
            "e.g.": "e<DOT>g<DOT>",
            "E.g.": "E<DOT>g<DOT>",
            "i.e.": "i<DOT>e<DOT>",
            "I.e.": "I<DOT>e<DOT>",
            "Fig.": "Fig<DOT>",
            "Eq.": "Eq<DOT>",
            "Ref.": "Ref<DOT>",
            "vs.": "vs<DOT>",
            "al.": "al<DOT>"
        }

        masked = cleaned
        for abbr, token in replacements.items():
            masked = masked.replace(abbr, token)

        try:
            sentences = sent_tokenize(masked)
        except Exception:
            # Fallback regex-based splitter if NLTK fails
            sentences = re.split(r"(?<=[.!?])\s+", masked)

        # Unmask
        final_sentences = []
        for s in sentences:
            unmasked = s
            for abbr, token in replacements.items():
                unmasked = unmasked.replace(token, abbr)
            s_clean = unmasked.strip()
            if len(s_clean) > 5:
                final_sentences.append(s_clean)

        return final_sentences

    def detect_sections(self, pages_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Identifies academic sections across pages and maps content to section names.
        """
        sections: List[Dict[str, Any]] = []
        current_section = "Abstract / Overview"
        current_text: List[str] = []
        current_start_page = 1
        current_pages = set()

        for page in pages_data:
            page_num = page["page_number"]
            lines = page["text"].split("\n")

            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue

                matched_section = None
                # Check if this line looks like a section header (short line, matches regex)
                if len(line_str) < 80:
                    for pattern, sec_name in self.SECTION_PATTERNS:
                        if re.search(pattern, line_str):
                            matched_section = sec_name
                            break

                if matched_section and matched_section != current_section:
                    # Save accumulated section
                    if current_text:
                        sec_body = self.clean_text("\n".join(current_text))
                        if len(sec_body) > 30:
                            sections.append({
                                "section_title": current_section,
                                "text": sec_body,
                                "start_page": current_start_page,
                                "pages": sorted(list(current_pages)),
                                "word_count": len(sec_body.split())
                            })
                    # Start new section
                    current_section = matched_section
                    current_text = [line_str]
                    current_start_page = page_num
                    current_pages = {page_num}
                else:
                    current_text.append(line_str)
                    current_pages.add(page_num)

        # Flush final section
        if current_text:
            sec_body = self.clean_text("\n".join(current_text))
            if len(sec_body) > 30:
                sections.append({
                    "section_title": current_section,
                    "text": sec_body,
                    "start_page": current_start_page,
                    "pages": sorted(list(current_pages)),
                    "word_count": len(sec_body.split())
                })

        return sections

