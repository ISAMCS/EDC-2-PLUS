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
import ast

eval_model = sys.argv[1]#llama3_request, GPT_Instruct_request, ChatGPT_request
date = sys.argv[2]
dataset = sys.argv[3]#496 or 300 or full
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]

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
    global topk, dataset
    topk = int(topk)
    if dataset == "redundancy":
        ref_text = "\n".join([f"{i+1}.{case['docs'][i].strip()}" for i in range(topk)])
    else:
        ref_text = "\n".join([f"{i+1}.{case['passages'][i]['text'].strip()}" for i in range(topk)])
    prompt = "Instruction:\nPlease refer to the following text and answer the following question in simple words.\n\nQuestion:\n{}\n\nReference text:\n{}\n\nAnswer:".format(case["question"], ref_text) 
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
        for passage in case['passages']:
            if 'embedding' in passage:
                del passage['embedding']
        text = _run_nli_GPT3turbo(case)
        case["response"] = text
    return slice_cases

eval_method = {
    "llama4_request": "llama4",
    "Phi_request": "Phi",
    "mistral7b_instruct_request": "mistral7b",
    "gpt35_turbo_0613_request": "3.5turbo"
}.get(eval_model, eval_model)

def run(topk, noise):
    global eval_method, date, dataset
    res_file = f"{benchmark}/results/{date}_{dataset}_rag_{eval_method}_noise{noise}_topk{topk}.json"
    if dataset == "redundancy":
        if topk == 30:
            case_file = f"codes/datasets/case_0329_rewrite_3.5turbo_webq_noise{noise}_topk{topk}.json"
        else:
            case_file = f"codes/datasets/case_0327_rewrite_3.5turbo_webq_noise{noise}_topk{topk}.json"
    else:
        case_file = f"{benchmark}/datasets/{benchmark}_results_random_{dataset}_w_negative_passages_noise{noise}_topk{topk}_embedding.json"
    with open(case_file, "r", encoding="utf-8") as lines:
        cases = json.load(lines)
        # Sequentially process all cases
        final_result = process_slice(cases)
        with open(res_file, "w", encoding="utf-8") as json_file:
            json.dump(final_result, json_file, ensure_ascii=False, indent=4)

for topk in topkk:
    for noise in noises:
        run(topk, noise)
        print(f"In Eval Baseline RAG: {topk}, {noise}")

# 这个脚本的目的是使用不同的模型（GPT-4、GPT-3.5、GPT-3.5-turbo和自定义的T5模型）来评估案例中的前提和断言之间的逻辑关系