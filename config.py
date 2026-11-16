"""
AI-Powered Research Paper Summarization and Question Answering
Configuration Module
"""

import os
import tempfile
from pathlib import Path

# Enforce Offline Mode and Silence Warnings for local inference
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DATA_DIR = DATA_DIR / "samples"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
SUMMARIES_DIR = OUTPUTS_DIR / "summaries"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
CHARTS_DIR = OUTPUTS_DIR / "charts"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure runtime directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLES_DATA_DIR, MODELS_DIR, SUMMARIES_DIR, EVALUATION_DIR, CHARTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Model Selections (Balanced for Laptop CPU Execution)
# 1. Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# 2. Abstractive Summarization Model
# DistilBART-CNN-12-6 is an industry standard distilled version of BART, fast & high quality
ABSTRACTIVE_SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"

# 3. Question Answering / RAG Generator Model
# FLAN-T5 Small - fast CPU inference (~300MB) with high instruction-following quality
QA_GENERATOR_MODEL = "google/flan-t5-small"
QA_GENERATOR_MODEL_FALLBACK = "t5-small"

# Text Processing & Chunking Configuration
DEFAULT_CHUNK_SIZE = 600          # Characters per chunk (~100-120 words)
DEFAULT_CHUNK_OVERLAP = 120       # Overlapping characters between consecutive chunks
MIN_CHUNK_LENGTH = 60             # Ignore noise chunks smaller than this

# Retrieval Configuration
DEFAULT_TOP_K = 3
RELEVANCE_SIMILARITY_THRESHOLD = 0.32  # Cosine similarity cutoff for groundedness
MAX_CONTEXT_TOKENS = 1024

# Fallback Response
FALLBACK_NOT_FOUND_MESSAGE = (
    "The requested information was not found in the uploaded research paper. "
    "To prevent hallucination, the system only generates answers strictly supported by the document's text."
)

# UI Configuration
PAGE_TITLE = "ScholarAI | Research Paper Assistant"
PAGE_ICON = "📚"
LAYOUT = "wide"

