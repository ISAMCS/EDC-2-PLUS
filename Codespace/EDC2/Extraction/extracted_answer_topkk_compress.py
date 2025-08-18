import torch
import json
from tqdm import tqdm
import os
import sys
import requests
from requests.auth import HTTPBasicAuth
import sys
from sklearn.metrics import roc_auc_score
from Codespace.LLMs import utils
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
    assess_model = utils.llama4_maverick_request
elif eval_model == "Phi":
    assess_model = utils.Microsoft_Phi4_request
elif eval_model == "mistral7b_instruct":
    assess_model = utils.mistral7b_instruct_request
elif eval_model == "gpt35_turbo":
    assess_model = utils.gpt35_turbo_0613_request
elif eval_model == "local_hf":
    assess_model = utils.local_hf_request
    
def _run_nli_GPT3turbo(case):
    prompt = f"""Task Description:
    You need to extract the essential information from a generated answer and reformat it to match the structure of the golden answer. We will provide a question, a golden answer, and a generated answer. Carefully compare the generated answer with the golden answer, and extract key information ONLY from the generated answer. Do NOT copy or use the golden answer directly. The reformatted answer must be strictly derived from the generated answer, reorganized to match the structure and style of the golden answer as much as possible, but only using information present in the generated answer.

    Input:

    Question: {case["question"]}
    Golden Answer: {case["answers"][0]}
    Generated Answer: {case["response"]}
    Requirements:

    - Extract information from the generated answer that corresponds to the essential content of the golden answer.
    - Reorganize the extracted content to align with the structure of the golden answer, including phrasing and order of information where relevant.
    - If the generated answer contains information not covered in the golden answer, include only information crucial to answering the question. Disregard redundant or irrelevant details.
    - Do NOT copy or use the golden answer directly. The reformatted answer must be strictly extracted and reorganized from the generated answer.
    Output Format:
    Provide a reformatted answer, derived ONLY from the generated answer and NOT copied from the golden answer:

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