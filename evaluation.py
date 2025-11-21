import os
import sys
# Add custom libs path
sys.path.insert(0, r"C:\tmp\libs")

import json
import numpy as np
from sklearn.metrics import precision_score, recall_score
import evaluate
from utils import search_index, generate_embeddings, summarize_text, load_index, INDEX_DIR, DATA_DIR

def evaluate_search(ground_truth, index, metadata, k=5):
    """
    Evaluates search precision and recall.
    ground_truth: list of {query: str, expected_chunk_ids: list[int]}
    """
    y_true = []
    y_pred = []
    
    precisions = []
    recalls = []
    
    print(f"Evaluating search on {len(ground_truth)} queries...")
    
    for item in ground_truth:
        query = item["query"]
        expected_ids = set(item["expected_chunk_ids"])
        
        # Search
        query_embedding = generate_embeddings([query])[0]
        indices, _ = search_index(index, query_embedding, k=k)
        
        retrieved_ids = set(indices)
        
        # Calculate hits
        hits = len(expected_ids.intersection(retrieved_ids))
        
        # Precision@k = hits / k
        precision = hits / k
        precisions.append(precision)
        
        # Recall@k = hits / total_relevant
        if len(expected_ids) > 0:
            recall = hits / len(expected_ids)
            recalls.append(recall)
        else:
            recalls.append(0.0)
            
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    
    return avg_precision, avg_recall

def evaluate_summarization(ground_truth):
    """
    Evaluates summarization using ROUGE.
    ground_truth: list of {text: str, reference_summary: str}
    """
    rouge = evaluate.load("rouge")
    
    predictions = []
    references = []
    
    print(f"Evaluating summarization on {len(ground_truth)} samples...")
    
    for item in ground_truth:
        text = item["text"]
        ref = item["reference_summary"]
        
        # Generate summary
        pred = summarize_text(text)
        
        predictions.append(pred)
        references.append(ref)
        
    results = rouge.compute(predictions=predictions, references=references)
    return results

if __name__ == "__main__":
    # Example usage with dummy data
    # In a real scenario, load this from a file
    
    # Load index
    index_path = os.path.join(INDEX_DIR, "faiss.index")
    index = load_index(index_path)
    
    if index:
        print("Index loaded.")
        # Dummy evaluation (since we don't have real ground truth yet)
        print("Skipping actual evaluation execution as no ground truth data is available.")
        print("To use: Create a JSON file with ground truth and call evaluate_search/evaluate_summarization.")
    else:
        print("Index not found. Run app.py to create one.")
