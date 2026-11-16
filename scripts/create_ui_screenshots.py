"""
Generates clean visual diagrams and application interface previews for report documentation.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Title
    ax.text(50, 96, "ScholarAI: System Architecture & Data Flow", ha="center", fontsize=15, fontweight="bold", color="#1E3A8A")

    # Boxes
    boxes = [
        (10, 75, 80, 12, "Layer 1: User Interface & Ingestion (Streamlit, PyMuPDF)", "#EFF6FF", "#3B82F6"),
        (10, 56, 80, 12, "Layer 2: Academic Preprocessing & Chunking (Page-Preserving Sliding Window)", "#F8FAFC", "#64748B"),
        (10, 37, 38, 12, "Layer 3A: Summarization\nTextRank & DistilBART", "#FEF3C7", "#D97706"),
        (52, 37, 38, 12, "Layer 3B: Dense RAG Pipeline\nSentence Transformers + FAISS", "#ECFDF5", "#059669"),
        (10, 18, 80, 12, "Layer 4: Verification & Evaluation (ROUGE, MRR, Faithfulness, Page Citations)", "#F5F3FF", "#7C3AED"),
    ]

    for x, y, w, h, text, bg, border in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.5", facecolor=bg, edgecolor=border, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1E293B")

    # Arrows
    arrow_props = dict(facecolor="#1E3A8A", edgecolor="#1E3A8A", width=1.5, headwidth=6, shrink=0.05)
    ax.annotate("", xy=(50, 68), xytext=(50, 75), arrowprops=arrow_props)
    ax.annotate("", xy=(29, 49), xytext=(29, 56), arrowprops=arrow_props)
    ax.annotate("", xy=(71, 49), xytext=(71, 56), arrowprops=arrow_props)
    ax.annotate("", xy=(50, 30), xytext=(50, 37), arrowprops=arrow_props)

    plt.tight_layout()
    out_path = FIG_DIR / "system_architecture.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Created {out_path}")


def create_ui_preview():
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Streamlit Browser Window Mockup
    window = patches.FancyBboxPatch((2, 2), 96, 94, boxstyle="round,pad=1.0", facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=2)
    ax.add_patch(window)

    # Header bar
    hbar = patches.Rectangle((2, 90), 96, 6, facecolor="#F1F5F9", edgecolor="none")
    ax.add_patch(hbar)
    ax.text(6, 92.5, "● ● ●  http://localhost:8501  —  ScholarAI | Research Paper Assistant", fontsize=9, color="#64748B")

    # Sidebar
    sidebar = patches.Rectangle((2, 2), 26, 88, facecolor="#0F172A", edgecolor="none")
    ax.add_patch(sidebar)
    ax.text(4, 82, "📚 ScholarAI", fontsize=13, fontweight="bold", color="#F8FAFC")
    ax.text(4, 78, "MCA NLP Project", fontsize=9, color="#94A3B8")
    ax.text(4, 68, "● Home Dashboard\n● Upload Paper\n● Document Summary\n● Section Summary\n✔ Ask Questions (QA)\n● Evidence Inspector\n● Benchmark Metrics", fontsize=9, color="#E2E8F0", linespacing=2)

    # Main Area
    ax.text(32, 82, "💬 Intelligent Question Answering (RAG)", fontsize=14, fontweight="bold", color="#1E3A8A")
    ax.text(32, 77, "Loaded: sample_transformer_paper.pdf (4 Pages, 872 Words)", fontsize=9, color="#64748B")

    # Question Box
    qbox = patches.FancyBboxPatch((32, 64), 62, 9, boxstyle="round,pad=0.8", facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1)
    ax.add_patch(qbox)
    ax.text(34, 69, "Question: By what percentage does the proposed attention mechanism reduce memory overhead?", fontsize=8.5, color="#0F172A")

    # Answer Box (Green grounded)
    abox = patches.FancyBboxPatch((32, 42), 62, 18, boxstyle="round,pad=0.8", facecolor="#F0FDF4", edgecolor="#22C55E", linewidth=1.5)
    ax.add_patch(abox)
    ax.text(34, 55, "🎯 Grounded Answer:", fontsize=9.5, fontweight="bold", color="#15803D")
    ax.text(34, 49, "38.4%", fontsize=13, fontweight="bold", color="#166534")
    ax.text(34, 44, "Citation: [Page 1]  |  Relevance Score: 0.46  |  Model: google/flan-t5-small", fontsize=8, color="#15803D")

    # Evidence Box
    ebox = patches.FancyBboxPatch((32, 14), 62, 24, boxstyle="round,pad=0.8", facecolor="#F8FAFC", edgecolor="#3B82F6", linewidth=1.5)
    ax.add_patch(ebox)
    ax.text(34, 33, "🔎 Supporting Document Evidence (Passage 1 — Page 1):", fontsize=8.5, fontweight="bold", color="#1D4ED8")
    evidence_text = (
        "\"By dynamically pruning low-entropy attention weights, our mechanism reduces memory overhead\n"
        "by 38.4% while improving ROUGE-L summarization scores by 2.7 points over standard BART baselines.\""
    )
    ax.text(34, 24, evidence_text, fontsize=8, color="#334155", style="italic")

    plt.tight_layout()
    out_path = FIG_DIR / "ui_qa_screenshot.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Created {out_path}")


if __name__ == "__main__":
    create_architecture_diagram()
    create_ui_preview()

