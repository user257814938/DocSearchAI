# PSTB-DI-Bootcamp_Hackathon_2

CPU-optimized document ingestion, search, and summarization tool.

## Architecture
- **Frontend**: Streamlit
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS (CPU)
- **Summarization**: Transformers (facebook/bart-base)

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run app: `streamlit run app.py`
