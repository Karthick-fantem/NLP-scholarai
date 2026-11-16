import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.pdf_extractor import PDFExtractor
from modules.chunking import AcademicChunker
from modules.embeddings import EmbeddingModel
from modules.retriever import DocumentRetriever

extractor = PDFExtractor()
doc = extractor.extract_from_path("data/samples/sample_transformer_paper.pdf")
chunker = AcademicChunker(chunk_size=500, chunk_overlap=100)
chunks = chunker.chunk_document_pages(doc["pages"])

for c in chunks:
    print(f"Chunk {c['chunk_id']} (Page {c['page_number']}) Target {'38.4' in c['text']}: {c['text'][:120]}...\n")

emb_model = EmbeddingModel()
emb_model.load_model()
texts = [c["text"] for c in chunks]
embeddings = emb_model.embed_texts(texts)
retriever = DocumentRetriever(chunks, embeddings, use_faiss=True)

q = "By what percentage does the proposed attention mechanism reduce memory overhead?"
q_emb = emb_model.embed_query(q)
hits = retriever.retrieve(q_emb, top_k=3)
print("=== RETRIEVED HITS FOR QUERY ===")
for h in hits:
    print(f"Rank {h['rank']} (Score {h['similarity_score']:.3f}, Page {h['page_number']}): {h['text'][:150]}...\n")
