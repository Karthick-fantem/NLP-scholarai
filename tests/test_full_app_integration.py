"""
End-to-End System Integration Test
Simulates a full user session:
1. Upload & parse 4-page academic PDF
2. Section detection & chunking with page tracking
3. Extractive summarization (TextRank)
4. Abstractive summarization (DistilBART)
5. Grounded Question Answering (FLAN-T5)
6. Out-of-Domain Hallucination Prevention
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from modules.pdf_extractor import PDFExtractor
from modules.preprocessing import AcademicTextPreprocessor
from modules.chunking import AcademicChunker
from modules.extractive_summarizer import TextRankSummarizer
from modules.abstractive_summarizer import AbstractiveSummarizer
from modules.embeddings import EmbeddingModel
from modules.retriever import DocumentRetriever
from modules.question_answering import RAGQuestionAnswerer


def run_integration_test():
    print("=" * 65)
    print("STARTING END-TO-END PIPELINE INTEGRATION TEST")
    print("=" * 65)

    pdf_path = BASE_DIR / "data/samples/sample_transformer_paper.pdf"
    assert pdf_path.exists(), "Sample PDF missing!"

    # 1. Extraction
    print("\n[Step 1] Extracting text from PDF...")
    extractor = PDFExtractor()
    doc_data = extractor.extract_from_path(str(pdf_path))
    print(f"-> Extracted {doc_data['total_pages']} pages, {doc_data['total_words']} words.")
    assert doc_data["total_pages"] == 4
    assert not doc_data["is_scanned"]

    # 2. Section detection & Chunking
    print("\n[Step 2] Academic section detection & chunking...")
    preprocessor = AcademicTextPreprocessor()
    sections = preprocessor.detect_sections(doc_data["pages"])
    print(f"-> Detected {len(sections)} sections: {[s['section_title'] for s in sections]}")

    chunker = AcademicChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_document_pages(doc_data["pages"])
    print(f"-> Generated {len(chunks)} chunks with page tracking.")
    for c in chunks[:2]:
        print(f"   * Chunk {c['chunk_id']} -> {c['page_citation']}: {c['text'][:65]}...")

    # 3. TextRank Extractive Summary
    print("\n[Step 3] Running TextRank Extractive Summarizer...")
    textrank = TextRankSummarizer()
    tr_res = textrank.summarize_pages(doc_data["pages"], num_sentences=3)
    print(f"-> Extractive Summary ({tr_res['sentence_count']} sentences):")
    print(f"   \"{tr_res['summary'][:160]}...\"")

    # 4. Neural Abstractive Summary
    print("\n[Step 4] Running DistilBART Abstractive Summarizer...")
    abstractive = AbstractiveSummarizer()
    abs_res = abstractive.summarize_document(doc_data["full_text"], mode="short")
    print(f"-> Abstractive Summary ({abs_res['summary_word_count']} words):")
    print(f"   \"{abs_res['summary']}\"")

    # 5. Embedding & Vector Indexing
    print("\n[Step 5] Dense Sentence Embeddings & FAISS Index...")
    emb_model = EmbeddingModel()
    emb_model.load_model()
    texts = [c["text"] for c in chunks]
    embeddings = emb_model.embed_texts(texts)
    retriever = DocumentRetriever(chunks, embeddings, use_faiss=True)
    print(f"-> Indexed {embeddings.shape[0]} vectors of dimension {embeddings.shape[1]}.")

    # 6. Question Answering (In-Domain Grounded Query)
    print("\n[Step 6] Grounded Question Answering Test...")
    qa_bot = RAGQuestionAnswerer(model_name="google/flan-t5-small")
    q1 = "By what percentage does the proposed attention mechanism reduce memory overhead?"
    ans1 = qa_bot.answer_question(q1, retriever, emb_model, top_k=4)
    print(f"   Question: {q1}")
    print(f"   Answer: {ans1['answer']}")
    print(f"   Citations: {ans1['citations_display']}")
    print(f"   Relevance: {ans1['max_similarity']:.2f}")
    assert "38.4" in ans1["answer"]
    assert ans1["is_grounded"]
    print("   -> PASSED Grounded Verification!")

    # 7. Out-of-Domain Query (Hallucination Rejection Test)
    print("\n[Step 7] Out-of-Domain Hallucination Prevention Test...")
    q2 = "What are the rules of playing professional cricket?"
    ans2 = qa_bot.answer_question(q2, retriever, emb_model, top_k=2)
    print(f"   Question: {q2}")
    print(f"   Answer: {ans2['answer']}")
    print(f"   Is Grounded: {ans2['is_grounded']}")
    assert not ans2["is_grounded"]
    assert "not found" in ans2["answer"].lower()
    print("   -> PASSED Hallucination Rejection Verification!")

    print("\n" + "=" * 65)
    print("ALL INTEGRATION CHECKS PASSED WITH 100% SUCCESS!")
    print("=" * 65)


if __name__ == "__main__":
    run_integration_test()

