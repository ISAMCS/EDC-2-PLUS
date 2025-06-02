import json
import os
from sklearn.metrics import precision_score, recall_score
import sys
from nltk.tokenize import word_tokenize

date = sys.argv[1]
dataset = sys.argv[2]
eval_method = sys.argv[3]
benchmark = sys.argv[4]

def normalize_text(text):
    text = text.lower()
    return ' '.join(text.split())

def compute_f1(pred, true):
    pred_tokens = word_tokenize(pred)
    true_tokens = word_tokenize(true)
    common = set(pred_tokens) & set(true_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(true_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def compute_metrics(dataset):
    em_total = 0
    f1_total = 0
    for data in dataset:
        pred = data["extracted_answer"].split(":")[-1].strip()
        answers = data["answers"]
        pred = normalize_text(pred)
        answers = [normalize_text(ans) for ans in answers]
        em = int(any(pred == ans for ans in answers))
        em_total += em

        f1 = max(compute_f1(pred, ans) for ans in answers)
        f1_total += f1

    em_score = em_total / len(dataset)
    f1_score = f1_total / len(dataset)
    return round(em_score*100,2), round(f1_score*100,2)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

eval_method_clean = eval_method.replace("eval_", "").replace("eval", "")
input_file = os.path.join(
    REPO_ROOT,
    benchmark,
    "extracted_answer",
    f"{date}_{dataset}_baseline_wo_retrieve_{eval_method_clean}.json"
)
with open(input_file, "r", encoding="utf-8") as f:
    datasets = json.load(f)
em_score, f1_score = compute_metrics(datasets)
print(f"{input_file}: EM: {em_score}, F1: {f1_score}")