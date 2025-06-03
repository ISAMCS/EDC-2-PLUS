import torch
import json
from tqdm import tqdm
import os
import requests
from requests.auth import HTTPBasicAuth
import concurrent.futures
import sys
import ast
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from codes.text_utils import llama4_maverick_request, Microsoft_Phi4_request, mistral7b_instruct_request, gpt35_turbo_0613_request 
date = sys.argv[1]
dataset = sys.argv[2]
eval_model = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
length = sys.argv[6]
summary_prompt = sys.argv[7]
clustering_type = sys.argv[8]
benchmark = sys.argv[9]

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

def _run_nli_GPT3turbo(question, ref_text):
    prompt = "Instruction:\nPlease refer to the following text and answer the following question in simple words.\n\nQuestion:\n{}\n\nReference text:\n{}\n\nAnswer:".format(question, ref_text) 
    res = 0
    while (True):
        try:
            text = assess_model(prompt)
            break
        except Exception as e:
            print(f"An error occurred: {e}")
    return text

def process_slice(cases):
    global topk, dataset
    topk = int(topk)
    for i, case in enumerate(tqdm(cases)):
        if case["summary_docs"]:
            ref_text = "\n".join([f"{i+1}.{case['summary_docs'][i]}" for i in range(len(case['summary_docs']))])
        else:
            if dataset == "redundancy":
                ref_text = "\n".join([f"{i+1}.{case['docs'][i].strip()}" for i in range(topk)])
            else:
                ref_text = "\n".join([f"{i+1}.{case['passages'][i]['text'].strip()}" for i in range(topk)])
        text= _run_nli_GPT3turbo(case["question"], ref_text)
        case["response"] = text
    return cases

eval_method = {
    "llama4_request": "llama4",
    "Phi_request": "Phi",
    "mistral7b_instruct_request": "mistral7b",
    "gpt35_turbo_0613_request": "3.5turbo"
}.get(eval_model, eval_model)

def run(topk, noise):
    global eval_method, date, dataset, length, summary_prompt
    res_file = f"{benchmark}/results/{date}_{dataset}_ours_summary_{summary_prompt}_ddtags_{clustering_type}_{length}_{eval_method}_noise{noise}_topk{topk}.json"
    eval_method_1 = eval_method.split("_")[-1]
    case_file = f"{benchmark}/datasets/case_{date}_summary_{eval_method_1}_{summary_prompt}_{dataset}_results_ddtags_{clustering_type}_{length}_noise{noise}_topk{topk}.json"
    with open(case_file, "r", encoding="utf-8") as lines:
        cases = json.load(lines)
        # Sequentially process all cases
        final_result = process_slice(cases)
        with open(res_file, "w", encoding="utf-8") as json_file:
            json.dump(final_result, json_file, ensure_ascii=False, indent=4)

for topk in topkk:
    for noise in noises:
        run(topk, noise)
        print(f"Finished running for topk={topk} and noise={noise} and length={length} and summary prompt={summary_prompt} In Eval Ours")