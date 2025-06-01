import sys
import json
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import KMeans  # Add scikit-learn for clustering

# Usage: python get_embedding.py <dataset> <model_path_or_name> [num_clusters]
dataset = sys.argv[1]
model_path = sys.argv[2] if len(sys.argv) > 2 else "bert-base-uncased"
num_clusters = int(sys.argv[3]) if len(sys.argv) > 3 else 5  # Default to 5 clusters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16).to(device)

def get_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        feature = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return feature.squeeze().cpu().tolist()

def get_embeddings_for_passages(passages):
    return [get_embedding(p["text"]) for p in passages]

def cluster_embeddings(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels

input_file = f"../../{dataset}/datasets/{dataset}_results_w_negative_passages_full.json"
output_file = f"../../{dataset}/datasets/{dataset}_results_w_negative_passages_full_embedding.json"

with open(input_file, "r", encoding="utf-8") as lines:
    cases = json.load(lines)
    for case in tqdm(cases):
        # Embeddings for positive and negative passages (documents)
        for passage in case.get("positive_passages", []):
            passage["embedding"] = get_embedding(passage["text"])
        for passage in case.get("negative_passages", []):
            passage["embedding"] = get_embedding(passage["text"])
        # Embedding for query (if present)
        if "query" in case:
            case["query_embedding"] = get_embedding(case["query"])
        # Example: Cluster positive passages before summarization
        pos_embeddings = [p["embedding"] for p in case.get("positive_passages", [])]
        if pos_embeddings:
            cluster_labels = cluster_embeddings(pos_embeddings, num_clusters)
            for passage, label in zip(case["positive_passages"], cluster_labels):
                passage["cluster"] = int(label)

    json.dump(cases, json_file, ensure_ascii=False, indent=4)