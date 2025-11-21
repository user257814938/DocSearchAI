import os
import sys
sys.path.insert(0, r"C:\tmp\libs")
from utils import process_and_index_document, search_index, generate_embeddings, summarize_text, DATA_DIR

# Create a dummy file
test_file = "test_doc.txt"
with open(test_file, "w", encoding="utf-8") as f:
    f.write("Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.")

# Initialize
index = None
metadata = {}

print("1. Processing document...")
index, metadata, error = process_and_index_document(test_file, index, metadata)

if error:
    print(f"Error: {error}")
    exit(1)

print("2. Document processed. Metadata:", metadata)

print("3. Searching...")
query = "What is AI?"
query_embedding = generate_embeddings([query])[0]
indices, distances = search_index(index, query_embedding, k=1)

print(f"Search results: Indices={indices}, Distances={distances}")

if indices[0] != -1:
    retrieved_text = metadata[indices[0]]["text"]
    print(f"Retrieved text: {retrieved_text}")
    
    print("4. Summarizing...")
    summary = summarize_text(retrieved_text)
    print(f"Summary: {summary}")
else:
    print("No results found.")

# Cleanup
os.remove(test_file)
print("Done.")
