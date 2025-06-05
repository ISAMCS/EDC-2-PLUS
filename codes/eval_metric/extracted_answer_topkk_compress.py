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
import ast
import time

# python codes/eval_metric/extracted_answer_topkk_compress.py 0604 triviaq gpt35_turbo "[20]" "[0]" full
date = sys.argv[1]
dataset = sys.argv[2]
eval_model = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]

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
    while True:
        try:
            text = assess_model(prompt)
            #time.sleep(3)
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
    case_file = f"{dataset}/results/{date}_{dataset}_compress_{eval_model}_noise{noise}_topk{topk}.json"
    res_file = f"{dataset}/extracted_answer/{date}_{dataset}_compress_{eval_model}_noise{noise}_topk{topk}.json"
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
