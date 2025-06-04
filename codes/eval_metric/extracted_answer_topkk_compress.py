import torch
import json
from tqdm import tqdm
import os
import sys
import requests
from requests.auth import HTTPBasicAuth
import sys
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from codes.text_utils import llama4_maverick_request, Microsoft_Phi4_request, mistral7b_instruct_request, gpt35_turbo_0613_request, local_hf_request
# CHANGE THIS TO THE DESIRED MODEL FUNCTION
eval_model = gpt35_turbo_0613_request
import ast
import time

date = sys.argv[1]
dataset = sys.argv[2]
eval_method = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]

import re
import string

ANSWER_PREFIXES = [
    r"^\s*reformatted answer\s*:\s*",
    r"^\s*final answer\s*:\s*",
    r"^\s*answer\s*:\s*"
]

def normalize(text: str) -> str:
    """Lower-case, drop articles, punctuation, and collapse spaces."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def extract_clean_answer(raw: str) -> str:
    """Pull the concise answer out of the model’s response."""
    # 1) Keep only the first non-empty line
    first_line = next((ln for ln in raw.splitlines() if ln.strip()), "")
    # 2) Strip our prefixes (“Reformatted Answer:”, “Answer:”, …)
    for p in ANSWER_PREFIXES:
        first_line = re.sub(p, "", first_line, flags=re.I)
    # 3) Trim quotes / trailing punctuation like the period after ‘piano.’
    first_line = first_line.strip().strip('“”"\'').rstrip(".")
    return first_line

def _run_nli_GPT3turbo(case):
    prompt = f"""Task:
    You are given a question, a golden answer, and a generated answer.  
    Extract the minimum text from the generated answer that answers the question

    Inputs:
    - Question: {case.get("question", "")}
    - Golden Answer: {case.get("answers", [""])[0]}
    - Generated Answer: {case.get("response", "")}

    Guidelines:
    1. Keep only information essential to the question.
    3. Ignore extra or unrelated details.  
    4. Output a single line after **Reformatted Answer:** with no surrounding quotes or added punctuation (unless required by the golden answer).  
    5. If no valid answer is present, leave the line blank after the colon.

    Output:  
    Reformatted Answer:"""
    while True:
        try:
            text = eval_model(prompt)
            time.sleep(3)
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            if "429" in str(e):
                print("Rate limit hit. Waiting 90 seconds before retrying...")
                sys.exit(1)
            else:
                time.sleep(10)
    return text

def process_slice(slice_cases):
    for case in tqdm(slice_cases):
        res = 0
        text = _run_nli_GPT3turbo(case)
        case["extracted_answer"] = extract_clean_answer(text)
    return slice_cases

def run(topk, noise):
    global eval_method, date, dataset
    case_file = f"{benchmark}/results/{date}_{dataset}_compress_{eval_method}_noise{noise}_topk{topk}.json"
    res_file = f"{benchmark}/extracted_answer/{date}_{dataset}_compress_{eval_method}_noise{noise}_topk{topk}.json"
    with open(case_file, "r", encoding="utf-8") as lines:
        cases = json.load(lines)
        # Process all cases sequentially (no threading)
        final_result = process_slice(cases)
        with open(res_file, "w", encoding="utf-8") as json_file:
            json.dump(final_result, json_file, ensure_ascii=False, indent=4)

for topk in topkk:
    for noise in noises:
        run(topk, noise)
        print(f"In Extracted Answer TopkK:{topk} Noise:{noise}")
