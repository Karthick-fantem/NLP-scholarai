"""
Utility script to generate realistic academic research paper PDFs
with proper paragraph text-box wrapping in PyMuPDF.
"""

from pathlib import Path
import pymupdf

SAMPLES_DIR = Path("data/samples")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def create_paper_1():
    doc = pymupdf.open()
    rect = pymupdf.Rect(50, 50, 550, 780)

    # Page 1: Title, Abstract, Introduction
    p1 = doc.new_page(width=595, height=842)  # Standard A4
    text_p1 = (
        "Efficient Multi-Head Self-Attention Mechanisms for Scientific Text Processing\n\n"
        "Dr. Eleanor Vance, Prof. Marcus Sterling, Dr. Alicia Chen\n"
        "Department of Computer Science and Artificial Intelligence\n\n"
        "Abstract\n"
        "Natural language processing in specialized scientific domains faces severe challenges due to high lexical density and complex syntactic structures. Standard recurrent architectures struggle with quadratic complexity across lengthy technical documents. In this paper, we propose Sparse-Scale Multi-Head Attention (SS-MHA), an optimized self-attention variant tailored for scientific literature. By dynamically pruning low-entropy attention weights, our mechanism reduces memory overhead by 38.4% while improving ROUGE-L summarization scores by 2.7 points over standard BART baselines.\n\n"
        "1. Introduction\n"
        "Scientific literature is expanding exponentially, with over three million peer-reviewed articles published annually. Automated comprehension systems, including document summarization and question answering, are essential for assisting researchers in navigating large corpora. Early methods relied heavily on frequency-based heuristics such as TF-IDF and graph-based centrality measures like TextRank. While computationally efficient, these extractive techniques lack the capacity to synthesize diffuse findings.\n\n"
        "Recent breakthroughs in Transformer architectures (Vaswani et al., 2017) demonstrated that multi-head self-attention enables rich contextual representation learning. However, the standard scaled dot-product attention scales quadratically with sequence length, making long-document processing expensive. This paper addresses this bottleneck through an adaptive sparse projection method.\n\n"
        "The primary contributions of this paper are:\n"
        "1. We introduce SS-MHA, a sparse attention layer that preserves scientific named entities.\n"
        "2. We formulate an entropy-guided token importance filter for passage retrieval.\n"
        "3. We conduct extensive empirical evaluations demonstrating state-of-the-art efficiency on SciTLDR and QASPER."
    )
    p1.insert_textbox(rect, text_p1, fontsize=10, fontname="helv")

    # Page 2: Related Work & Methodology
    p2 = doc.new_page(width=595, height=842)
    text_p2 = (
        "2. Related Work\n"
        "Dense document retrieval and extractive summarization have evolved from classic statistical formulations to dense neural representations. Erkan and Radev (2004) introduced LexRank, which represents documents as graph networks where sentences serve as vertices. Mihalcea and Tarau (2004) formalized TextRank, applying PageRank iterations over sentence co-occurrence graphs. While unsupervised, extractive approaches cannot alter sentence grammar or combine disjoint facts.\n\n"
        "Abstractive summarization witnessed dramatic advancement with sequence-to-sequence models such as BART (Lewis et al., 2020) and T5 (Raffel et al., 2020). BART employs a bidirectional encoder combined with an autoregressive decoder. FLAN-T5 (Chung et al., 2022) extended this through instruction fine-tuning, exhibiting superior zero-shot performance on complex academic reasoning tasks.\n\n"
        "3. Proposed Methodology\n"
        "Our proposed architecture incorporates two interconnected modules: (a) a Hierarchical Semantic Chunking Layer, and (b) a Grounded Retrieval-Augmented Generation (RAG) framework.\n\n"
        "3.1 Hierarchical Semantic Chunking\n"
        "Let D represent a document consisting of M pages. We first tokenize sentences using punctuation-aware academic filters. A sliding window of size L_c = 600 characters with an overlap delta = 120 characters slides across the text. Each chunk preserves its origin page metadata, ensuring that downstream citations can trace directly to the source page.\n\n"
        "3.2 Dense Embedding and Indexing\n"
        "Each chunk c_i is mapped to a d-dimensional vector using an all-MiniLM-L6-v2 model: v_i = E(c_i) / ||E(c_i)||_2, where d = 384. Normalized vectors allow inner product calculation to represent exact cosine similarity. We index these vectors using FAISS FlatIP."
    )
    p2.insert_textbox(rect, text_p2, fontsize=10, fontname="helv")

    # Page 3: Experimental Evaluation
    p3 = doc.new_page(width=595, height=842)
    text_p3 = (
        "4. Experimental Evaluation\n\n"
        "4.1 Datasets and Experimental Setup\n"
        "We benchmarked our system using SciTLDR (scientific paper summarization) and QASPER (academic question answering). All experiments were conducted on an Intel Core i7 machine with 16GB RAM using CPU-only inference to reflect typical student workstation constraints.\n\n"
        "For summarization, we compared unsupervised TextRank against DistilBART-CNN-12-6. Evaluation metrics include ROUGE-1, ROUGE-2, and ROUGE-L F1 scores. For passage retrieval, we compared traditional TF-IDF against dense Sentence-Transformer retrieval evaluated via Precision@3 and Mean Reciprocal Rank (MRR).\n\n"
        "4.2 Quantitative Findings\n"
        "Table 1 summarizes the summarization performance on test papers. DistilBART achieved a ROUGE-1 of 0.384, significantly outperforming TextRank (0.291). However, TextRank required an average inference time of only 0.004 seconds compared to 7.2 seconds for DistilBART, illustrating a clear trade-off between semantic abstraction and computational speed.\n\n"
        "In retrieval experiments, dense embeddings achieved an MRR of 0.812 compared to 0.548 for TF-IDF, primarily because scientific queries frequently use synonyms that fail under exact lexical matching.\n\n"
        "4.3 Grounding and Hallucination Control\n"
        "When out-of-domain questions were posed (questions whose answers do not appear in the text), our similarity threshold mechanism (cutoff = 0.32) successfully triggered fallback responses in 100% of test queries, verifying strict hallucination mitigation."
    )
    p3.insert_textbox(rect, text_p3, fontsize=10, fontname="helv")

    # Page 4: Conclusion & References
    p4 = doc.new_page(width=595, height=842)
    text_p4 = (
        "5. Conclusion and Future Work\n\n"
        "In this study, we presented an integrated NLP framework combining hierarchical chunking, dense semantic retrieval, and grounded abstractive synthesis. By enforcing strict relevance filtering and page citation tracking, the system provides verifiable, transparent answers for academic literature search. Future enhancements include implementing full multimodal parsing for mathematical formulas and figures, as well as incorporating active user feedback loops.\n\n"
        "References\n"
        "[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.\n"
        "[2] Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2020). BART: Denoising sequence-to-sequence pre-training. ACL 2020.\n"
        "[3] Mihalcea, R., & Tarau, P. (2004). TextRank: Bringing order into text. EMNLP 2004.\n"
        "[4] Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., ... & Wei, J. (2022). Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416.\n"
        "[5] Dasigi, P., Lo, K., Beltagy, I., Cohan, A., Smith, N. A., & Gardner, M. (2021). A dataset of information-seeking questions and answers anchored in research papers. NAACL 2021."
    )
    p4.insert_textbox(rect, text_p4, fontsize=10, fontname="helv")

    output_path = SAMPLES_DIR / "sample_transformer_paper.pdf"
    doc.save(str(output_path))
    doc.close()
    print(f"Re-generated {output_path} with full textbox wrapping.")


def create_scanned_empty_paper():
    doc = pymupdf.open()
    p = doc.new_page(width=595, height=842)
    p.draw_rect(pymupdf.Rect(50, 50, 545, 792), color=(0.8, 0.8, 0.8), fill=(0.96, 0.96, 0.96))
    output_path = SAMPLES_DIR / "sample_scanned_paper.pdf"
    doc.save(str(output_path))
    doc.close()
    print(f"Re-generated {output_path}")


if __name__ == "__main__":
    create_paper_1()
    create_scanned_empty_paper()
