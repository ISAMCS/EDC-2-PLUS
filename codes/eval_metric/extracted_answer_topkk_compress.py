# python codes/eval_metric/extracted_answer_topkk_compress.py 0601 full eval_llama3 "[20]" "[40]" triviaq

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
from codes.text_utils import llama4_maverick_request
eval_model = llama4_maverick_request
import ast
import time

date = sys.argv[1]
dataset = sys.argv[2]
eval_method = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]

def _run_nli_GPT3turbo(case):
    prompt = f"""Task Description:
You need to extract the essential information from a generated answer and reformat it to match the structure of the golden answer. We will provide a question, a golden answer, and a generated answer. Carefully compare the generated answer with the golden answer, and extract key information from the generated answer to make it as close as possible to the golden answer in format. This will facilitate subsequent evaluation using Exact Match (EM) and F1 metrics.

Input:

Question: {case.get("question", "")}
Golden Answer: {case.get("answers", [""])[0]}
Generated Answer: {case.get("response", "")}
Requirements:

Extract information from the generated answer that corresponds to the essential content of the golden answer.
Reorganize the extracted content to align with the structure of the golden answer, including phrasing and order of information where relevant.
If the generated answer contains information not covered in the golden answer, include only information crucial to answering the question. Disregard redundant or irrelevant details.
Output Format:
Provide a reformatted answer, aligned as closely as possible with the golden answer:

Reformatted Answer: """
    while True:
        try:
            text = eval_model(prompt)
            time.sleep(4)
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
        case["extracted_answer"] = text
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
