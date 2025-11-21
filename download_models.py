import sys
sys.path.insert(0, r"C:\tmp\libs")

from sentence_transformers import SentenceTransformer
from transformers import pipeline

def download_models():
    print("1. Downloading Embedding Model (all-MiniLM-L6-v2)...")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("   - Embedding Model Downloaded Successfully.")
    except Exception as e:
        print(f"   - Failed to download Embedding Model: {e}")

    print("\n2. Downloading Summarization Model (facebook/bart-base)...")
    try:
        summarizer = pipeline("summarization", model="facebook/bart-base")
        print("   - Summarization Model Downloaded Successfully.")
    except Exception as e:
        print(f"   - Failed to download Summarization Model: {e}")

if __name__ == "__main__":
    download_models()
