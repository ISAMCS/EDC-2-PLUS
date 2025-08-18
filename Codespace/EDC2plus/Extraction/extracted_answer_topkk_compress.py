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
Extract the essential information from the provided evidence and reformat your answer to match the style, brevity, and phrasing of the golden answer. Use only the evidence in the compressed blocks. Your answer should be as concise and direct as the golden answer, and use similar wording and format whenever possible.

Input:
Question: {case['question']}
Golden Answer: {case['answers'][0]}
Evidence (compressed blocks):\n{chr(10).join(case.get('compressed_blocks', []))}

Requirements:
- Use only the evidence in the compressed blocks to answer the question.
- Reformat your answer to match the style, brevity, and phrasing of the golden answer.
- If the evidence does not support the golden answer, answer strictly based on the evidence provided.
- Do NOT use outside knowledge.

Output Format:
Provide a single reformatted answer, matching the golden answer's style and format, and ONLY using the evidence:

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
    case_file = f"{dataset}/results/{date}_{dataset}_edc2plus_compress_{eval_model}_noise{noise}_topk{topk}.json"
    res_file = f"{dataset}/extracted_answer/{date}_{dataset}_edc2plus_compress_{eval_model}_noise{noise}_topk{topk}.json"
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