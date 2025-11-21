import streamlit as st
import sys
sys.path.insert(0, r"C:\tmp\libs")
import os
import pickle
import numpy as np
from utils import (
    process_and_index_document, 
    generate_embeddings, 
    search_index, 
    summarize_text, 
    save_index, 
    load_index,
    RAW_DIR,
    INDEX_DIR,
    DATA_DIR
)

# --- App Config ---
st.set_page_config(page_title="DocSearch AI", page_icon="🔍", layout="wide")

# --- Session State ---
if "index" not in st.session_state:
    st.session_state.index = None
if "metadata" not in st.session_state:
    st.session_state.metadata = {}

# --- Paths ---
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.pkl")

# --- Load Existing Data ---
if st.session_state.index is None:
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        st.session_state.index = load_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            st.session_state.metadata = pickle.load(f)
        st.toast("Loaded existing index.", icon="✅")

# --- Sidebar: Upload ---
with st.sidebar:
    st.header("📂 Document Ingestion")
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
    
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing..."):
                # Save file
                file_path = os.path.join(RAW_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process
                index, metadata, error = process_and_index_document(
                    file_path, 
                    st.session_state.index, 
                    st.session_state.metadata
                )
                
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.session_state.index = index
                    st.session_state.metadata = metadata
                    
                    # Save to disk
                    save_index(index, INDEX_FILE)
                    with open(METADATA_FILE, "wb") as f:
                        pickle.dump(metadata, f)
                        
                    st.success(f"Indexed {uploaded_file.name}!")

    st.divider()
    st.markdown("### Stats")
    st.write(f"Total Documents: {len(set(m['source'] for m in st.session_state.metadata.values())) if st.session_state.metadata else 0}")
    st.write(f"Total Chunks: {len(st.session_state.metadata) if st.session_state.metadata else 0}")

# --- Main Area: Search ---
st.title("🔍 Semantic Document Search")
st.markdown("Ask a question about your uploaded documents.")

query = st.text_input("Enter your query:", placeholder="What is the main topic of the document?")

if query:
    if st.session_state.index is None:
        st.warning("Please upload and process a document first.")
    else:
        with st.spinner("Searching..."):
            # Generate query embedding
            query_embedding = generate_embeddings([query])[0]
            
            # Search
            k = 5
            indices, distances = search_index(st.session_state.index, query_embedding, k=k)
            
            # Retrieve results
            results = []
            context_text = ""
            for idx, dist in zip(indices, distances):
                if idx != -1 and idx in st.session_state.metadata:
                    meta = st.session_state.metadata[idx]
                    results.append({
                        "text": meta["text"],
                        "source": meta["source"],
                        "score": dist
                    })
                    context_text += meta["text"] + " "
            
            # Display Results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📝 Summary")
                if context_text:
                    summary = summarize_text(context_text)
                    st.info(summary)
                else:
                    st.write("No relevant content found to summarize.")

            with col2:
                st.subheader("📄 Relevant Chunks")
                for res in results:
                    with st.expander(f"Source: {res['source']} (Score: {res['score']:.4f})"):
                        st.write(res["text"])

