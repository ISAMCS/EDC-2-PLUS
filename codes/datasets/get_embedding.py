import sys
import json
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

dataset = sys.argv[1]
model_path = sys.argv[2] if len(sys.argv) > 2 else "bert-base-uncased"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16).to(device)

def get_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        feature = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return feature.squeeze().cpu().tolist()

input_file = f"{dataset}/datasets/{dataset}_results_w_negative_passages_full.json"
output_file = f"{dataset}/datasets/{dataset}_results_w_negative_passages_full_embedding.json"

with open(input_file, "r", encoding="utf-8") as lines:
    cases = json.load(lines)
    for case in tqdm(cases):
        for passage in case.get("positive_passages", []):
            passage["embedding"] = get_embedding(passage["text"])
        for passage in case.get("negative_passages", []):
            passage["embedding"] = get_embedding(passage["text"])

with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(cases, json_file, ensure_ascii=False, indent=4)