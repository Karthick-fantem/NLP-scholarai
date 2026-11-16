"""
Utility script to pre-cache transformer models locally for offline operation.
"""

import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

print("1. Caching Sentence-Transformer (all-MiniLM-L6-v2)...")
try:
    _ = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Sentence-Transformer successfully cached!")
except Exception as e:
    print(f"Error caching SentenceTransformer: {e}")

print("\n2. Caching QA Model (google/flan-t5-small)...")
try:
    _ = AutoTokenizer.from_pretrained("google/flan-t5-small")
    _ = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    print("google/flan-t5-small successfully cached!")
except Exception as e:
    print(f"Error caching flan-t5-small: {e}")

print("\n3. Caching Summarization Model (sshleifer/distilbart-cnn-12-6)...")
try:
    _ = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    _ = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
    print("sshleifer/distilbart-cnn-12-6 successfully cached!")
except Exception as e:
    print(f"Error caching distilbart: {e}")

print("\nAll target models cached successfully.")

