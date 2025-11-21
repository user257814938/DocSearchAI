import os
import sys
# Add custom libs path for torch/transformers if needed
sys.path.insert(0, r"C:\tmp\libs")

import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Optional imports
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import docx
except ImportError:
    docx = None

# --- Configuration ---
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
INDEX_DIR = os.path.join(DATA_DIR, "index")

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, INDEX_DIR]:
    os.makedirs(d, exist_ok=True)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SUMMARIZATION_MODEL_NAME = "facebook/bart-base"
CHUNK_SIZE = 250  # tokens approx (words)
OVERLAP = 20

# --- Global Models (Lazy Loading) ---
_embedder = None
_summarizer = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL_NAME)
    return _summarizer

# --- Text Extraction ---
def extract_text_from_file(file_path):
    """Extracts text from .txt, .pdf, or .docx files."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    
    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif ext == ".pdf":
            if fitz is None:
                return None, "PyMuPDF (fitz) is not installed."
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
        elif ext == ".docx":
            if docx is None:
                return None, "python-docx is not installed."
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            return None, f"Unsupported file format: {ext}"
    except Exception as e:
        return None, str(e)
        
    return text, None

# --- Text Processing ---
def clean_text(text):
    """Basic text cleaning."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Splits text into chunks of approximately chunk_size words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# --- Embeddings & FAISS ---
def generate_embeddings(chunks):
    """Generates embeddings for a list of text chunks."""
    model = get_embedder()
    embeddings = model.encode(chunks)
    return embeddings

def create_index(dimension):
    """Creates a new FAISS index."""
    index = faiss.IndexFlatL2(dimension)
    return index

def save_index(index, index_path):
    """Saves FAISS index to disk."""
    faiss.write_index(index, index_path)

def load_index(index_path):
    """Loads FAISS index from disk."""
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    return None

def search_index(index, query_embedding, k=5):
    """Searches the FAISS index for the k nearest neighbors."""
    D, I = index.search(np.array([query_embedding]), k)
    return I[0], D[0]

# --- Summarization ---
def summarize_text(text, max_length=200, min_length=50):
    """Summarizes the given text using BART."""
    summarizer = get_summarizer()
    # Truncate input if too long for the model (BART limit is usually 1024 tokens)
    # We'll just take the first 1024 chars * 3 approx for now to be safe or rely on pipeline truncation
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False, truncation=True)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error summarizing: {e}"

# --- Pipeline Helper ---
def process_and_index_document(file_path, index, chunks_metadata):
    """
    Full pipeline: Extract -> Clean -> Chunk -> Embed -> Index
    Returns updated index and metadata.
    """
    text, error = extract_text_from_file(file_path)
    if error:
        return index, chunks_metadata, error

    text = clean_text(text)
    chunks = chunk_text(text)
    
    if not chunks:
        return index, chunks_metadata, "No text extracted."

    embeddings = generate_embeddings(chunks)
    
    if index is None:
        index = create_index(embeddings.shape[1])
        
    index.add(embeddings)
    
    # Store metadata (mapping index id to chunk text)
    start_id = len(chunks_metadata)
    for i, chunk in enumerate(chunks):
        chunks_metadata[start_id + i] = {
            "source": os.path.basename(file_path),
            "chunk_id": i,
            "text": chunk
        }
        
    return index, chunks_metadata, None
