import json
import sys
import ast
from nltk.tokenize import word_tokenize
import pandas as pd
import re, string

results = []
date = sys.argv[1]
dataset = sys.argv[2]
eval_method = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]

# --- Normalization helper ---
def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return ' '.join(text.split())

# --- Compute F1 ---
def compute_f1(pred, true):
    pred_tokens = word_tokenize(pred)
    true_tokens = word_tokenize(true)
    common = set(pred_tokens) & set(true_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(true_tokens)
    return 2 * (precision * recall) / (precision + recall)

# --- EM/F1 evaluator ---
def compute_metrics(dataset):
    em_total, f1_total = 0, 0
    for data in dataset:
        ans_field = data.get("extracted_answer") or data.get("edc2plus_response") or data.get("answer") or ""
        if isinstance(ans_field, list):
            # if list of strings -> join; if list of dicts -> try common keys; else join stringified elements
            if all(isinstance(x, str) for x in ans_field):
                norm = " ".join([x.strip() for x in ans_field if x.strip()])
            elif all(isinstance(x, dict) for x in ans_field) and any("answer" in x for x in ans_field):
                norm = next((x.get("answer","") for x in ans_field if x.get("answer","")), "")
            else:
                norm = " ".join(map(str, ans_field))
        elif isinstance(ans_field, dict):
            norm = ans_field.get("answer") or ans_field.get("final_answer") or json.dumps(ans_field, ensure_ascii=False)
        else:
            norm = str(ans_field)

        pred = norm.split(":")[-1].strip()
        answers = data["answers"]

        pred = normalize_text(pred)
        answers = [normalize_text(ans) for ans in answers]

        # EM: if pred matches any gold answer
        em_total += int(any(pred == ans for ans in answers))

        # F1: take max score across possible gold answers
        f1_total += max(compute_f1(pred, ans) for ans in answers)

    em_score = em_total / len(dataset)
    f1_score = f1_total / len(dataset)
    return round(em_score*100,2), round(f1_score*100,2)

# --- Main loop ---
for topk in topkk:
    for noise in noises:
        custom_file = sys.argv[7] if len(sys.argv) > 7 else None
        if custom_file and custom_file.endswith('.json'):
            input_file = f"triviaq/extracted_answer/{custom_file}"
        elif benchmark and benchmark != "baseline":
            input_file = f"triviaq/extracted_answer/{date}_{dataset}_edc2plus_compress_{eval_method}_noise{noise}_topk{topk}_{benchmark}.json"
        else:
            input_file = f"triviaq/extracted_answer/{date}_{dataset}_compress_{eval_method}_noise{noise}_topk{topk}.json"

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                datasets = json.load(f)
        except Exception as e:
            print(f"Could not open {input_file}: {e}")
            exit(1)

        em_score, f1_score = compute_metrics(datasets)
        results.append([topk, noise, em_score, f1_score])
        print(f"{input_file}: EM: {em_score}, F1: {f1_score}")

# --- Save to Excel ---
df = pd.DataFrame(results, columns=["TopK", "Noise", "EM Score", "F1 Score"])
output_file = f"{dataset}/tables/{date}_{dataset}_edc2plus_compress_{eval_method}_noise{noises}_topk{topkk}.xlsx"
df.to_excel(output_file, index=False)
print(f"Results saved to {output_file}")
