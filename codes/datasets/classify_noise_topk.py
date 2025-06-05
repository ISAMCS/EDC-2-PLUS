import json
import sys
import ast
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import random

# Set seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from codes.datasets.utils import (
    gpt35_turbo_0613_request,
    llama4_maverick_request,
    Microsoft_Phi4_request,
    mistral7b_instruct_request,
    local_hf_request,
)

# Parse arguments
dataset = sys.argv[1]
topk_list = ast.literal_eval(sys.argv[2])  # e.g. "[20]"
noise_list = ast.literal_eval(sys.argv[3]) # e.g. "[0,20,40,60,80,100]"
eval_model = sys.argv[4]

# Select LLM request function
if eval_model == "llama4":
    assess_model = llama4_maverick_request
elif eval_model == "Phi":
    assess_model = Microsoft_Phi4_request
elif eval_model == "mistral7b_instruct":
    assess_model = mistral7b_instruct_request
elif eval_model == "gpt35_turbo":
    assess_model = gpt35_turbo_0613_request
elif eval_model == "local_hf":
    assess_model = local_hf_request
else:
    raise ValueError(f"Unknown eval_model: {eval_model}")

input_file = f"{dataset}/datasets/{dataset}_results_w_negative_passages_full.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(path, torch_dtype=torch.float16).to(device)

def get_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        feature = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return feature.squeeze().cpu().numpy()

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def get_llm_ranked_indices(question, passages, k):
    if not passages:
        return []
    passage_list = "\n".join([f"[{i}] {p['text']}" for i, p in enumerate(passages)])
    prompt = f"""
You are given a question and a list of passages, each already pre-ranked by embedding similarity to the question. Your task is to further rank these passages by how directly and verifiably they support the most canonical answer.

Question:
"{question}"

Passages:
{passage_list}

Instructions:
- Focus on passages that most directly and explicitly support the canonical answer.
- Use embedding similarity as a guide, but prioritize passages that contain exact facts, names, or codes matching the expected answer.
- Ignore passages that are off-topic or do not provide clear evidence.
- Output only the indices of the passages in order of relevance.

Answer format: [index1, index2, ..., index{k}]
"""
    response = assess_model(prompt)
    try:
        indices = ast.literal_eval(response.strip())
        if isinstance(indices, list) and all(isinstance(i, int) for i in indices):
            indices = indices[:k]
            if len(indices) < k:
                indices += [i for i in range(len(passages)) if i not in indices][:k - len(indices)]
            return indices
    except Exception:
        pass
    return list(range(min(k, len(passages))))

with open(input_file, "r", encoding="utf-8") as f:
    cases = json.load(f)

# Optional: Count unique passage IDs for analysis
unique_ids = set()
for case in cases:
    for passage in case.get("positive_passages", []) + case.get("negative_passages", []):
        if "id" in passage:
            unique_ids.add(passage["id"])
print(f"Total unique passage ids: {len(unique_ids)}")

TOP_N_EMBEDDING = 50  # Number of passages to shortlist by embedding before LLM ranking

for topk in topk_list:
    for noise in noise_list:
        outs = []
        output_file = f"{dataset}/datasets/{dataset}_results_random_full_w_negative_passages_noise{noise}_topk{topk}_embedding.json"
        for case in tqdm(cases, desc=f"topk={topk}, noise={noise}"):
            out = dict(case)
            positives = case.get("positive_passages", [])
            negatives = case.get("negative_passages", [])
            all_passages = positives + negatives

            if len(all_passages) < topk:
                raise ValueError(f"Not enough passages for case: need {topk}, got {len(all_passages)}")

           # if TOP_N_EMBEDDING > len(all_passages):
            #    print(f"Warning: TOP_N_EMBEDDING ({TOP_N_EMBEDDING}) > number of passages ({len(all_passages)}). Using {len(all_passages)}.")

            # --- Step 1: Embedding-based pre-ranking ---
            q_emb = get_embedding(case["question"])
            passage_scores = [
                (i, cosine_similarity(q_emb, get_embedding(p['text'])))
                for i, p in enumerate(all_passages)
            ]
            passage_scores.sort(key=lambda x: x[1], reverse=True)
            top_n = min(TOP_N_EMBEDDING, len(passage_scores))
            top_indices_by_embedding = [i for i, _ in passage_scores[:top_n]]
            top_passages = [all_passages[i] for i in top_indices_by_embedding]

            # --- Step 2: LLM-based ranking on top-N ---
            llm_indices = get_llm_ranked_indices(case["question"], top_passages, topk)
            if len(llm_indices) < topk:
                unused = [i for i in range(len(top_passages)) if i not in llm_indices]
                llm_indices += unused[:topk - len(llm_indices)]
            selected = [top_passages[i] for i in llm_indices[:topk]]

            # --- Step 3: Add noise: replace n passages with random negatives ---
            n = topk * noise // 100
            p = topk - n
            selected_main = selected[:p]
            remaining_negatives = [x for x in negatives if x not in selected_main]
            if n > 0 and remaining_negatives:
                np.random.shuffle(remaining_negatives)
                selected_main += remaining_negatives[:n]
            assert len(selected_main) == topk, f"selected_main has {len(selected_main)} but topk={topk}"

            out["passages"] = selected_main
            out.pop("positive_passages", None)
            out.pop("negative_passages", None)
            outs.append(out)

        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(outs, json_file, ensure_ascii=False, indent=4)