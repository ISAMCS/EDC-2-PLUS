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
from codes.text_utils import llama4_maverick_request, Microsoft_Phi4_request, mistral7b_instruct_request, gpt35_turbo_0613_request 

eval_model = sys.argv[1]
date = sys.argv[2]
dataset = sys.argv[3]
benchmark = sys.argv[4]

# Model selection
if eval_model == "llama4_request":
    assess_model = llama4_maverick_request
elif eval_model == "mistral7b_instruct_request":
    assess_model = mistral7b_instruct_request
elif eval_model == "gpt35_turbo_0613_request":
    assess_model = gpt35_turbo_0613_request
elif eval_model == "Phi":
    assess_model = Microsoft_Phi4_request
else:
    raise ValueError(f"Unknown model name: {eval_model}")

def _run_nli_GPT3turbo(case):
    prompt = "Question:\n\n{}Answer:".format(case["question"]) 
    res = 0
    while (True):
        try:
            text = assess_model(prompt)
            break
        except Exception as e:
            print(f"An error occurred: {e}")
    return text

def process_slice(slice_cases):
    for case in tqdm(slice_cases):
        res=0
        text = _run_nli_GPT3turbo(case)
        case["response"] = text
    return slice_cases

eval_method = {
    "llama4_request": "llama4",
    "Phi_request": "Phi",
    "mistral7b_instruct_request": "mistral7b",
    "gpt35_turbo_0613_request": "3.5turbo"
}.get(eval_model, eval_model)

def run():
    global eval_method, date, dataset, benchmark
    res_file = f"{benchmark}/results/{date}_{dataset}_baseline_wo_retrieve_{eval_method}.json"
    if dataset == "500":
        case_file = "codes/datasets/case_webq_nq_ddtags_noise20_topk5_1101.json"
    elif dataset == "113":
        case_file = "codes/datasets/case_113_webq_ddtags_noise20_topk20_simcse_0.65.json"
    elif dataset == "400":
        case_file = "codes/datasets/webq_results_random_400_w_negative_passages_noise0_topk5.json"
    elif dataset == "full":
        case_file = f"{benchmark}/datasets/{benchmark}_results_w_negative_passages_{dataset}.json"
    with open(case_file, "r", encoding="utf-8") as lines:
        cases = json.load(lines)
        # Sequentially process all cases
        final_result = process_slice(cases)
        with open(res_file, "w", encoding="utf-8") as json_file:
            json.dump(final_result, json_file, ensure_ascii=False, indent=4)

run()
print(f"In Eval Baseline Wo Retrieve: {eval_model} {date} {dataset}")    

# 这个脚本的目的是使用不同的模型（GPT-4、GPT-3.5、GPT-3.5-turbo和自定义的T5模型）来评估案例中的前提和断言之间的逻辑关系