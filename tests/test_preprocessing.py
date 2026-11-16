"""
Unit Tests for Module 2: Preprocessing and Chunking
"""

import unittest
from modules.preprocessing import AcademicTextPreprocessor
from modules.chunking import AcademicChunker


class TestPreprocessingAndChunking(unittest.TestCase):

    def setUp(self):
        self.preprocessor = AcademicTextPreprocessor()
        self.chunker = AcademicChunker(chunk_size=300, chunk_overlap=60)

    def test_clean_text_hyphen_repair(self):
        raw = "This is a trans-\nformer neural archi-\ntecture."
        cleaned = self.preprocessor.clean_text(raw)
        self.assertIn("transformer", cleaned)
        self.assertIn("architecture", cleaned)
        self.assertNotIn("-\n", cleaned)

    def test_sentence_tokenization_with_abbreviations(self):
        text = "As shown by Vaswani et al. in 2017, attention is effective. E.g. for translation."
        sentences = self.preprocessor.tokenize_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertIn("et al.", sentences[0])
        self.assertIn("E.g.", sentences[1])

    def test_chunking_with_page_retention(self):
        mock_pages = [
            {"page_number": 1, "text": "Page one sentence one. Page one sentence two. Page one sentence three.", "has_text": True},
            {"page_number": 2, "text": "Page two discusses experimental results and benchmarks. Second sentence.", "has_text": True}
        ]
        chunks = self.chunker.chunk_document_pages(mock_pages)
        self.assertGreater(len(chunks), 0)

        # Ensure every chunk has page provenance
        for c in chunks:
            self.assertIn("page_number", c)
            self.assertIn(c["page_number"], [1, 2])
            self.assertIn("page_citation", c)
            self.assertIn("Page", c["page_citation"])


if __name__ == "__main__":
    unittest.main()

