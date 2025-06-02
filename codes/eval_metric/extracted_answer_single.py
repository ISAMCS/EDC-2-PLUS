import torch
import json
from tqdm import tqdm
import os
import sys
import requests
from requests.auth import HTTPBasicAuth
import concurrent.futures
import sys
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from codes.text_utils import llama4_maverick_request
eval_model = llama4_maverick_request

date = sys.argv[1]
dataset = sys.argv[2]
eval_method = sys.argv[3]
benchmark = sys.argv[4]

def _run_nli_GPT3turbo(case):
    prompt = f"""Task Description:
You need to extract the essential information from a generated answer and reformat it to match the structure of the golden answer. We will provide a question, a golden nnswer, and a generated answer. Carefully compare the generated answer with the golden answer, and extract key information from the generated answer to make it as close as possible to the golden answer in format. This will facilitate subsequent evaluation using Exact Match (EM) and F1 metrics.

Input:

Question: {case["question"]}
Golden Answer: {case["answers"][0]}
Generated Answer: {case["response"]}
Requirements:

Extract information from the generated answer that corresponds to the essential content of the golden answer.
Reorganize the extracted content to align with the structure of the golden answer, including phrasing and order of information where relevant.
If the generated answer contains information not covered in the golden answer, include only information crucial to answering the question. Disregard redundant or irrelevant details.
Output Format:
Provide a reformatted answer, aligned as closely as possible with the golden answer:

Reformatted Answer: """
    res = 0
    while (True):
        try:
            text = eval_model(prompt)
            break
        except Exception as e:
            print(f"An error occurred: {e}")
    return text

def process_slice(slice_cases):
    for case in tqdm(slice_cases):
        res=0
        text = _run_nli_GPT3turbo(case)
        case["extracted_answer"] = text
    return slice_cases

def run():
    global eval_method, date, dataset
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    res_file = os.path.join(
        base_dir,
        benchmark,
        "extracted_answer",
        f"{date}_full_baseline_wo_retrieve_llama3.json"
    )
    case_file = os.path.join(
        base_dir,
        benchmark,
        "results",
        f"{date}_{dataset}_baseline_wo_retrieve_eval_llama3.json"
    )
    os.makedirs(os.path.dirname(res_file), exist_ok=True)
    with open(case_file, "r", encoding="utf-8") as lines:
        cases = json.load(lines)
    json_data = []
    num_slices = 10
    slice_length = len(cases) // num_slices
    slices = [cases[i:i+slice_length] for i in range(0, len(cases), slice_length)]
    final_result = []
    results = []
    for slice_cases in slices:
        result = process_slice(slice_cases)
        final_result.extend(result)
    with open(res_file, "w", encoding="utf-8") as json_file:
        json.dump(final_result, json_file, ensure_ascii=False, indent=4) 
run()
