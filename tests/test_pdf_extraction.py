"""
Unit Tests for Module 1: PDF Extraction
"""

import unittest
from pathlib import Path
from modules.pdf_extractor import PDFExtractor

SAMPLE_DIR = Path("data/samples")


class TestPDFExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = PDFExtractor()
        self.valid_pdf = SAMPLE_DIR / "sample_transformer_paper.pdf"
        self.scanned_pdf = SAMPLE_DIR / "sample_scanned_paper.pdf"

    def test_extract_valid_paper(self):
        self.assertTrue(self.valid_pdf.exists(), "Sample PDF must exist.")
        result = self.extractor.extract_from_path(str(self.valid_pdf))

        self.assertEqual(result["total_pages"], 4)
        self.assertFalse(result["is_scanned"])
        self.assertGreater(result["total_words"], 300)
        self.assertEqual(len(result["pages"]), 4)

        # Verify page numbers are 1-indexed
        for idx, page in enumerate(result["pages"], start=1):
            self.assertEqual(page["page_number"], idx)
            self.assertTrue(page["has_text"])

    def test_scanned_empty_detection(self):
        self.assertTrue(self.scanned_pdf.exists(), "Scanned PDF must exist.")
        result = self.extractor.extract_from_path(str(self.scanned_pdf))

        self.assertTrue(result["is_scanned"])
        self.assertIsNotNone(result["warning"])
        self.assertIn("scanned image-based PDF", result["warning"])


if __name__ == "__main__":
    unittest.main()

