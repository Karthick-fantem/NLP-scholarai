"""
Verification and Offline Testing Script for Hugging Face Transformer Models
Validates:
1. sentence-transformers/all-MiniLM-L6-v2
2. google/flan-t5-small
3. sshleifer/distilbart-cnn-12-6
Tests strict offline mode (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1).
"""

import os
import sys
import time
from pathlib import Path
import psutil

# Ensure warnings don't clutter output
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def get_dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total_bytes / (1024 * 1024), 2)


def main():
    print("=" * 70)
    print("STAGE 1: ENSURING COMPLETE LOCAL MODEL DOWNLOAD")
    print("=" * 70)

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from sentence_transformers import SentenceTransformer

    models_to_download = [
        ("sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers"),
        ("google/flan-t5-small", "seq2seq"),
        ("sshleifer/distilbart-cnn-12-6", "seq2seq"),
    ]

    for model_id, m_type in models_to_download:
        print(f"\n[DOWNLOADING/VERIFYING] {model_id}...")
        t0 = time.time()
        try:
            loc = snapshot_download(
                repo_id=model_id,
                local_files_only=False,
                resume_download=True
            )
            print(f"-> Snapshot verified at: {loc} in {time.time() - t0:.2f}s")
        except Exception as e:
            print(f"-> Error downloading {model_id}: {e}")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("STAGE 2: STRICT OFFLINE INFERENCE VERIFICATION")
    print("Enforcing: HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1")
    print("=" * 70)

    # Enable offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    hub_cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    results = []

    # 1. Test sentence-transformers/all-MiniLM-L6-v2
    print("\n--- Model 1: sentence-transformers/all-MiniLM-L6-v2 ---")
    m1_name = "sentence-transformers/all-MiniLM-L6-v2"
    m1_folder_name = "models--sentence-transformers--all-MiniLM-L6-v2"
    m1_cache_path = hub_cache_dir / m1_folder_name
    m1_size_mb = get_dir_size_mb(m1_cache_path)

    mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
    t_load_start = time.time()
    st_model = SentenceTransformer(m1_name, local_files_only=True)
    t_load = time.time() - t_load_start
    mem_after = psutil.Process().memory_info().rss / (1024 * 1024)

    # Inference test
    sample_texts = [
        "Attention mechanisms improve context representation in scientific texts.",
        "Deep learning architectures process sequential token embeddings."
    ]
    t_inf_start = time.time()
    embeddings = st_model.encode(sample_texts, normalize_embeddings=True)
    t_inf = time.time() - t_inf_start

    sim = float(embeddings[0] @ embeddings[1])
    print(f"Status: SUCCESS (Offline)")
    print(f"Cache Location: {m1_cache_path}")
    print(f"Disk Size: {m1_size_mb} MB")
    print(f"RAM Overhead: ~{mem_after - mem_before:.1f} MB")
    print(f"Load Time: {t_load:.3f} s")
    print(f"Inference Time: {t_inf:.3f} s (for 2 sentences)")
    print(f"Embedding Shape: {embeddings.shape}, Cosine Sim: {sim:.4f}")

    results.append({
        "model": m1_name,
        "type": "Dense Embedding",
        "cache_path": str(m1_cache_path),
        "disk_mb": m1_size_mb,
        "load_time_s": round(t_load, 3),
        "inf_time_s": round(t_inf, 3),
        "sample_output": f"Embedding shape {embeddings.shape}, Cosine Sim {sim:.4f}"
    })

    # 2. Test google/flan-t5-small
    print("\n--- Model 2: google/flan-t5-small ---")
    m2_name = "google/flan-t5-small"
    m2_folder_name = "models--google--flan-t5-small"
    m2_cache_path = hub_cache_dir / m2_folder_name
    m2_size_mb = get_dir_size_mb(m2_cache_path)

    mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
    t_load_start = time.time()
    tok_t5 = AutoTokenizer.from_pretrained(m2_name, local_files_only=True)
    model_t5 = AutoModelForSeq2SeqLM.from_pretrained(m2_name, local_files_only=True)
    t_load = time.time() - t_load_start
    mem_after = psutil.Process().memory_info().rss / (1024 * 1024)

    # QA Inference Test
    prompt = (
        "Context: Sparse-Scale Multi-Head Attention reduces memory overhead by 38.4%.\n"
        "Question: By what percentage does Sparse-Scale Multi-Head Attention reduce memory overhead?\n"
        "Answer:"
    )
    t_inf_start = time.time()
    inputs = tok_t5(prompt, return_tensors="pt")
    outputs = model_t5.generate(**inputs, max_new_tokens=40)
    answer_t5 = tok_t5.decode(outputs[0], skip_special_tokens=True).strip()
    t_inf = time.time() - t_inf_start

    print(f"Status: SUCCESS (Offline)")
    print(f"Cache Location: {m2_cache_path}")
    print(f"Disk Size: {m2_size_mb} MB")
    print(f"RAM Overhead: ~{mem_after - mem_before:.1f} MB")
    print(f"Load Time: {t_load:.3f} s")
    print(f"Inference Time: {t_inf:.3f} s")
    print(f"Sample Answer: '{answer_t5}'")

    results.append({
        "model": m2_name,
        "type": "Grounded QA / Seq2Seq",
        "cache_path": str(m2_cache_path),
        "disk_mb": m2_size_mb,
        "load_time_s": round(t_load, 3),
        "inf_time_s": round(t_inf, 3),
        "sample_output": answer_t5
    })

    # 3. Test sshleifer/distilbart-cnn-12-6
    print("\n--- Model 3: sshleifer/distilbart-cnn-12-6 ---")
    m3_name = "sshleifer/distilbart-cnn-12-6"
    m3_folder_name = "models--sshleifer--distilbart-cnn-12-6"
    m3_cache_path = hub_cache_dir / m3_folder_name
    m3_size_mb = get_dir_size_mb(m3_cache_path)

    mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
    t_load_start = time.time()
    tok_bart = AutoTokenizer.from_pretrained(m3_name, local_files_only=True)
    model_bart = AutoModelForSeq2SeqLM.from_pretrained(m3_name, local_files_only=True)
    t_load = time.time() - t_load_start
    mem_after = psutil.Process().memory_info().rss / (1024 * 1024)

    # Summarization Inference Test
    article_text = (
        "Scientific literature is expanding exponentially with millions of papers published annually. "
        "Standard recurrent architectures struggle with quadratic complexity across long technical documents. "
        "In this work, we propose Sparse-Scale Multi-Head Attention, an optimized self-attention variant. "
        "Our mechanism reduces memory overhead by 38.4% while improving ROUGE-L summarization scores by 2.7 points."
    )
    t_inf_start = time.time()
    inputs_b = tok_bart(article_text, return_tensors="pt", max_length=512, truncation=True)
    outputs_b = model_bart.generate(**inputs_b, max_length=70, min_length=20, num_beams=2)
    summary_bart = tok_bart.decode(outputs_b[0], skip_special_tokens=True).strip()
    t_inf = time.time() - t_inf_start

    print(f"Status: SUCCESS (Offline)")
    print(f"Cache Location: {m3_cache_path}")
    print(f"Disk Size: {m3_size_mb} MB")
    print(f"RAM Overhead: ~{mem_after - mem_before:.1f} MB")
    print(f"Load Time: {t_load:.3f} s")
    print(f"Inference Time: {t_inf:.3f} s")
    print(f"Sample Summary: '{summary_bart}'")

    results.append({
        "model": m3_name,
        "type": "Abstractive Summarization",
        "cache_path": str(m3_cache_path),
        "disk_mb": m3_size_mb,
        "load_time_s": round(t_load, 3),
        "inf_time_s": round(t_inf, 3),
        "sample_output": summary_bart
    })

    print("\n" + "=" * 70)
    print("ALL 3 MODELS VERIFIED WORKING STRICTLY OFFLINE!")
    print("=" * 70)

    # Save summary report to JSON/CSV for documentation
    import json
    out_file = Path("outputs/evaluation/model_caching_verification.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Verification log written to {out_file}")


if __name__ == "__main__":
    main()

