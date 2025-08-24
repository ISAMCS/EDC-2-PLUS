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

from Codespace.LLMs.utils import (
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

# Use a stronger embedding model
path = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"  # or "BAAI/bge-large-en" for even stronger recall
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(path, torch_dtype=torch.float16).to(device)

# Embedding cache
embedding_cache = {}


def get_embedding(text):
    if text in embedding_cache:
        return embedding_cache[text]
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        feature = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    emb = feature.squeeze().cpu().numpy()
    embedding_cache[text] = emb
    return emb

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def get_llm_ranked_indices(question, passages, k):
    if not passages:
        return []
    passage_list = "\n".join([f"[{i}] {p['text']}" for i, p in enumerate(passages)])
    # Short, direct prompt with few-shot example
    prompt = f"""
You are given a question and a list of passages. Each passage is pre-ranked by embedding similarity.  
Your task: reorder them by how directly they support the best possible answer.

Question:
"{question}"

Passages:
{passage_list}

Guidelines:
1. Identify exactly what the question is asking (entity, number, date, etc.).  
2. Prefer passages that:
    - Contain the answer explicitly or provide strong evidence.
    - Include clear keywords, entities, numbers, or dates tied to the question.  
    - Use unambiguous language (avoid vague references).  
    - Prefer passages with explicit answers (numbers, named entities) over general context.
3. Deprioritize passages that:
    - Are irrelevant or only loosely related.  
    - Contain popular but off-topic entities.  
    - Provide general context without answering.  
4. If multiple passages are equally good, keep their original order.  
5. Output only the passage indices in ranked order, no extra text.

Answer format: [index1, index2, ..., index{k}]
"""
    response = assess_model(prompt)
    # Post-process: deduplicate, clip, fill, and validate
    try:
        indices = ast.literal_eval(response.strip())
        if isinstance(indices, list):
            indices = [int(i) for i in indices if isinstance(i, int) and 0 <= i < len(passages)]
            indices = list(dict.fromkeys(indices))[:k]
            if len(indices) < k:
                indices += [i for i in range(len(passages)) if i not in indices][:k - len(indices)]
            return indices
    except Exception:
        # Retry with stricter prompt
        retry_prompt = prompt + "\nReturn a JSON list of integers only."
        response = assess_model(retry_prompt)
        try:
            indices = ast.literal_eval(response.strip())
            if isinstance(indices, list):
                indices = [int(i) for i in indices if isinstance(i, int) and 0 <= i < len(passages)]
                indices = list(dict.fromkeys(indices))[:k]
                if len(indices) < k:
                    indices += [i for i in range(len(passages)) if i not in indices][:k - len(indices)]
                return indices
        except Exception:
            pass
    # Fallback: top-k
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
            passage_embs = [get_embedding(p['text']) for p in all_passages]
            passage_scores = [
                (i, cosine_similarity(q_emb, emb))
                for i, emb in enumerate(passage_embs)
            ]
            passage_scores.sort(key=lambda x: x[1], reverse=True)
            top_n = min(TOP_N_EMBEDDING, len(passage_scores))
            top_indices_by_embedding = [i for i, _ in passage_scores[:top_n]]
            top_passages = [all_passages[i] for i in top_indices_by_embedding]
            top_passage_embs = [passage_embs[i] for i in top_indices_by_embedding]

            # --- Step 2: LLM-based ranking on top-N ---
            llm_indices = get_llm_ranked_indices(case["question"], top_passages, topk)

            # --- Score Fusion: combine embedding and LLM scores ---
            # Assign LLM score: higher rank = higher score
            llm_scores = np.zeros(len(top_passages))
            for rank, idx in enumerate(llm_indices):
                if 0 <= idx < len(top_passages):
                    llm_scores[idx] = len(top_passages) - rank
            # Normalize embedding scores
            emb_scores = np.array([cosine_similarity(q_emb, emb) for emb in top_passage_embs])
            emb_scores = (emb_scores - emb_scores.min()) / (emb_scores.max() - emb_scores.min() + 1e-8)
            # Fusion: weighted sum (alpha controls balance)
            alpha = 0.35
            fusion_scores = alpha * emb_scores + (1 - alpha) * (llm_scores / llm_scores.max() if llm_scores.max() > 0 else llm_scores)
            fusion_indices = np.argsort(-fusion_scores)[:topk]
            selected = [top_passages[i] for i in fusion_indices]


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