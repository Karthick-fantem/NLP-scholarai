<div align="center">

# 📚 ScholarAI
### AI-Powered Research Paper Summarization and Question Answering

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.59-red?logo=streamlit)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Transformers-yellow)](https://huggingface.co)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Offline](https://img.shields.io/badge/Execution-100%25_Offline-brightgreen)]()

**MCA Final Year Project · Natural Language Processing**

*JAIN (Deemed-to-be University), Department of MCA, Bengaluru*

**Submitted by:** A K Karthick Raja · Roll No: 25MCAR0167
**Faculty Guide:** Dr. Sharal Coelho · Academic Year: 2026–2027

</div>

---

## 📌 Overview

**ScholarAI** is a fully offline, AI-powered research paper assistant that lets you:

- 📄 **Upload any PDF** research paper and extract text page-by-page
- 📝 **Generate summaries** using TextRank (extractive) or DistilBART (abstractive)
- 💬 **Ask natural-language questions** about the paper and get grounded answers
- 🔍 **View supporting evidence** with exact PDF page number citations
- 🛡️ **Zero hallucination** — if the answer isn't in the paper, it says so
- 📊 **Browse benchmark results** on SciTLDR and QASPER datasets

> All models run **100% locally on CPU**. No internet connection, no paid APIs, no GPU required.

---

## 🖥️ Demo Screenshots

| Tab | Feature |
|-----|---------|
| 🏠 Dashboard | Project overview, load sample paper instantly |
| 📄 Upload Paper | Drag-and-drop PDF, page-by-page viewer |
| 📝 Document Summary | TextRank vs DistilBART summaries with download |
| 📑 Section Summaries | Auto-detected sections summarized individually |
| 💬 Ask Questions | RAG-based QA with page citations |
| 🔍 Evidence Inspector | FAISS nearest-neighbour chunk explorer |
| 📊 Evaluation | ROUGE, MRR, F1 benchmark results with charts |
| 🎓 Viva Guide | 15 MCA exam questions with answers |

---

## 🏗️ System Architecture

```
PDF Document
      │
      ▼
[Module 1] PyMuPDF Extraction ──► Page-by-page text, page number tags
      │
      ▼
[Module 2] Preprocessing & Chunking ──► Overlapping sliding-window (512 chars, 120 overlap)
      │
      ├────────────────────────────────────────────────────────┐
      ▼                                                        ▼
[Module 3] Summarization Engine                 [Module 4] Semantic RAG Pipeline
  ├─ TextRank  (Extractive, PageRank graph)       ├─ Sentence Embeddings (all-MiniLM-L6-v2)
  └─ DistilBART (Abstractive, Transformer)        ├─ FAISS FlatIP Vector Index
                                                  ├─ Relevance Threshold τ = 0.32
                                                  └─ FLAN-T5 Grounded Answer Generation
```

---

## 🤖 Models Used

| Model | Purpose | Size | Speed |
|-------|---------|------|-------|
| `sentence-transformers/all-MiniLM-L6-v2` | Sentence embeddings for retrieval | ~80 MB | ⚡ Very Fast |
| `google/flan-t5-small` | Question answering (RAG) | ~300 MB | ✅ Fast |
| `sshleifer/distilbart-cnn-12-6` | Abstractive summarization | ~1.2 GB | 🐢 Moderate |

All models are downloaded once and cached locally. Subsequent runs are fully offline.

---

## 📂 Project Structure

```
📦 kr nlp/
├── 📄 app.py                          ← Main Streamlit web application
├── ⚙️  config.py                       ← Global config and offline env settings
├── 📋 requirements.txt                ← Python dependencies
├── 📖 README.md                       ← This file
│
├── 🧩 modules/                        ← Core NLP pipeline modules
│   ├── pdf_extractor.py               ← PyMuPDF page-by-page extraction
│   ├── preprocessing.py               ← Text cleaning and section detection
│   ├── chunking.py                    ← Overlapping sliding-window chunker
│   ├── extractive_summarizer.py       ← TextRank PageRank summarizer
│   ├── abstractive_summarizer.py      ← DistilBART transformer summarizer
│   ├── embeddings.py                  ← Sentence-BERT embeddings (MiniLM)
│   ├── retriever.py                   ← FAISS + TF-IDF retrieval
│   ├── question_answering.py          ← FLAN-T5 RAG with hallucination guard
│   └── evaluation.py                  ← ROUGE, EM, Token F1, P@K, MRR
│
├── 🔧 scripts/
│   ├── create_sample_papers.py        ← Generate demo PDF files
│   ├── prepare_datasets.py            ← Download SciTLDR + QASPER subsets
│   ├── run_experiments.py             ← Run A/B/C/D benchmark experiments
│   ├── verify_and_cache_models.py     ← Cache all models for offline use
│   └── generate_pdf_report.py        ← Generate PDF research report
│
├── 🧪 tests/
│   ├── test_pdf_extraction.py
│   ├── test_preprocessing.py
│   ├── test_retrieval.py
│   ├── test_qa_pipeline.py
│   └── test_full_app_integration.py   ← End-to-end integration test
│
├── 📊 outputs/
│   ├── evaluation/                    ← Experiment CSV results
│   │   ├── experiment_a_summarization.csv
│   │   ├── experiment_b_retrieval.csv
│   │   ├── experiment_c_chunking.csv
│   │   └── experiment_d_qa_grounding.csv
│   └── charts/                        ← Experiment result plots
│
├── 📁 data/
│   ├── samples/                       ← Demo PDF files
│   ├── raw/                           ← Raw downloaded dataset files
│   └── processed/                     ← Preprocessed JSON samples
│
└── 📑 reports/
    ├── project_report.tex             ← Full LaTeX MCA research report
    ├── project_report.pdf             ← Compiled PDF report
    └── figures/                       ← Report figures and screenshots
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10 or higher
- pip
- ~3 GB free disk space (for model caches)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/scholarai-nlp.git
cd scholarai-nlp
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

### 4. Download and Cache Models *(first time only — requires internet)*

```bash
python scripts/verify_and_cache_models.py
```

This downloads all three models (~1.6 GB total) and verifies they work correctly.

### 5. Generate Sample PDFs

```bash
python scripts/create_sample_papers.py
```

### 6. Launch the Application

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🚀 Usage Guide

### Option A — Use the Built-in Sample Paper (Instant Demo)

1. Open the app at `http://localhost:8501`
2. Click **🏠 Dashboard** tab
3. Click **"📘 Load Sample Research Paper"**
4. Switch to **"📝 Document Summary"** → Click **"✨ Generate Summary"**
5. Switch to **"💬 Ask Questions"** → Click any preset question button → Click **"🔍 Get Grounded Answer"**

### Option B — Upload Your Own PDF

1. Click **"📄 Upload Paper"** tab
2. Drag and drop any research paper PDF
3. Wait for processing (progress bar shown)
4. Navigate to any other tab — data persists across all tabs

### Asking Questions

The QA system accepts any natural-language question. Examples:

```
"What is the main contribution of this paper?"
"What datasets were used for evaluation?"
"By what percentage does the proposed method outperform the baseline?"
"What are the limitations mentioned?"
```

If the answer is not in the paper, the system returns:
> *"The requested information was not found in the uploaded research paper."*

---

## 📊 Experimental Results

### Experiment A — Summarization (SciTLDR Dataset)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Latency |
|-------|---------|---------|---------|---------|
| TextRank (Extractive) | 0.287 | 0.087 | 0.187 | **0.004s** |
| DistilBART (Abstractive) | **0.308** | 0.069 | **0.242** | 7.3s |

### Experiment B — Retrieval (QASPER Dataset)

| Retriever | P@1 | MRR | Latency |
|-----------|-----|-----|---------|
| TF-IDF (Sparse) | 0.80 | 0.90 | 2.65ms |
| Dense MiniLM (Dense) | **1.00** | **1.00** | 33.9ms |

### Experiment C — Optimal Configuration

| Chunk Size | Top-K | QA Token F1 |
|-----------|-------|------------|
| 256 chars | 3 | 0.71 |
| **512 chars** | **3** | **0.81** ✅ |
| 1024 chars | 3 | 0.74 |

### Experiment D — Hallucination Prevention

| Metric | Result |
|--------|--------|
| Unanswerable Rejection Rate | **100%** |
| Mean Token F1 (Answerable) | **0.730** |
| Correctly Handled Questions | **6 / 6** |

---

## 🛠️ Configuration

All settings can be adjusted from the **sidebar** in the app, or permanently in [`config.py`](config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_CHUNK_SIZE` | 512 chars | Sliding-window chunk size |
| `DEFAULT_CHUNK_OVERLAP` | 120 chars | Overlap between consecutive chunks |
| `DEFAULT_TOP_K` | 3 | Number of chunks retrieved per question |
| `RELEVANCE_SIMILARITY_THRESHOLD` | 0.32 | Cosine similarity cutoff (τ) |

---

## 🧪 Running Tests

```bash
# Run all unit tests
python -m unittest discover -s tests -p "test_*.py"

# Run the full end-to-end integration test
python tests/test_full_app_integration.py
```

Expected output: **13/13 tests passed**

---

## 📈 Running Experiments

```bash
# Download dataset samples (requires internet, first time only)
python scripts/prepare_datasets.py

# Run all 4 benchmark experiments
python scripts/run_experiments.py
```

Results are saved to `outputs/evaluation/` as CSV files.
Charts are saved to `outputs/charts/` as PNG files.

---

## 🔒 Offline Mode

After the initial model download, the app runs completely offline:

```bash
# These are set automatically in config.py
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

---

## 🧠 Key NLP Concepts

| Concept | Implementation |
|---------|---------------|
| **Extractive Summarization** | TextRank using sentence-level cosine similarity and PageRank |
| **Abstractive Summarization** | DistilBART encoder-decoder with beam search |
| **Dense Retrieval** | all-MiniLM-L6-v2 → 384-D FAISS FlatIP index |
| **Sparse Retrieval** | TF-IDF cosine similarity (baseline) |
| **RAG** | Retrieve top-K chunks → FLAN-T5 prompt → grounded answer |
| **Hallucination Guard** | Cosine similarity threshold τ = 0.32 |
| **Evaluation** | ROUGE-1/2/L, Exact Match, Token F1, P@K, MRR |

---

## 📦 Dependencies

```
streamlit>=1.30.0        # Web application framework
pymupdf>=1.23.0          # PDF text extraction
nltk>=3.8.0              # Sentence tokenization, stop words
networkx>=3.0            # TextRank graph algorithm
torch>=2.0.0             # PyTorch backend for transformers
transformers>=4.36.0     # DistilBART + FLAN-T5 models
sentence-transformers>=2.2.0  # all-MiniLM-L6-v2 embeddings
faiss-cpu>=1.7.4         # Vector similarity search
scikit-learn>=1.3.0      # TF-IDF, cosine similarity
numpy>=1.24.0            # Numerical operations
pandas>=2.0.0            # Data processing, CSV export
rouge-score>=0.1.2       # ROUGE evaluation metric
matplotlib>=3.7.0        # Result visualizations
datasets>=2.16.0         # SciTLDR, QASPER dataset loading
```

---

## ⚠️ Known Limitations

- **Scanned PDFs:** PDFs with no selectable text layer cannot be processed (OCR not yet implemented)
- **Model size:** DistilBART requires ~2 GB RAM during inference
- **Small QA model:** FLAN-T5-Small may struggle with complex multi-hop reasoning
- **Single document:** Cannot compare across multiple papers simultaneously

---

## 🗺️ Future Work

- [ ] OCR support via Tesseract for scanned PDFs
- [ ] LLaMA-3 / Mistral-7B integration for higher QA accuracy
- [ ] Multi-document retrieval across a paper library
- [ ] GPU acceleration (CUDA support)
- [ ] Fine-tuning on QASPER for domain-specific QA
- [ ] Docker container for one-click deployment
- [ ] BERTScore evaluation metric

---

## 📄 Report & Documentation

| Document | Location |
|----------|---------|
| Full MCA Research Report (PDF) | `reports/project_report.pdf` |
| LaTeX Source | `reports/project_report.tex` |
| System Architecture Diagram | `reports/figures/system_architecture.png` |
| Experiment CSV Results | `outputs/evaluation/` |
| Experiment Charts | `outputs/charts/` |

---

## 🙏 Acknowledgements

- [Allen Institute for AI](https://allenai.org) for the SciTLDR and QASPER datasets
- [HuggingFace](https://huggingface.co) for Transformers, Sentence-Transformers, and model hosting
- [Facebook Research](https://github.com/facebookresearch/faiss) for the FAISS library
- [Streamlit](https://streamlit.io) for the web application framework

---

## 📬 Contact

**A K Karthick Raja**
Roll No: 25MCAR0167
Department of MCA, JAIN (Deemed-to-be University)
Jayanagar Campus, Bengaluru

---

<div align="center">

Made with ❤️ for MCA Final Year Project · 2026–2027

*"The goal is to turn data into information, and information into insight."*

</div>
