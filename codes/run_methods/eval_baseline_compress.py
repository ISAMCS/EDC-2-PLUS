import torch
import json
from tqdm import tqdm
import os
import sys
import requests
import ast
import time
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from codes.text_utils import llama4_maverick_request, Microsoft_Phi4_request, mistral7b_instruct_request, gpt35_turbo_0613_request, local_hf_request 

eval_model = sys.argv[1] 
date = sys.argv[2]
dataset = sys.argv[3]
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
elif eval_model == "local_hf_request":
    assess_model = local_hf_request
else:
    raise ValueError(f"Unknown model name: {eval_model}")

def _run_nli_GPT3turbo(case):
    global topk
    topk = int(topk)
    ref_text = "\n".join([f"{i+1}.{case['summary_docs_baseline'][i]}" for i in range(topk)])
    prompt = (
        "## Instruction ##\n"
        "Read ONLY the reference text below and answer the question in clear, simple words.\n"
        "• If the question requires counting (e.g., “How many teams end in United?”), count unique items carefully and list them.\n"
        "• If deduction is required, think step-by-step **internally**, but **do NOT show your reasoning**.\n"
        "• Do NOT use any information not present in the reference text.\n"
        "Return only the final answer after the tag **Final Answer:**, on a single line. If the answer cannot be determined from the reference text, write 'Unknown' after Final Answer:.\n\n"
        "## Question ##\n{question}\n\n"
        "## Reference Text ##\n{ref_text}\n\n"
        "Final Answer:"
    ).format(question=case["question"], ref_text=ref_text)
    res = 0
    while (True):
        try:
            text = assess_model(prompt)
            break
        except Exception as e:
            print(f"An error occurred: {e}")
    return text


def process_slice(slice_cases):
    outs = []
    for case_1 in tqdm(slice_cases):
        case = deepcopy(case_1)
        for passage in case['passages']:
            if 'embedding' in passage:
                del passage['embedding']
        text = _run_nli_GPT3turbo(case)
        case["response"] = text
        outs.append(case)
    return outs

eval_method = {
    "llama4_request": "llama4",
    "Phi_request": "Phi",
    "mistral7b_instruct_request": "mistral7b",
    "gpt35_turbo_0613_request": "3.5turbo",
    "local_hf_request" : "local_hf"
}.get(eval_model, eval_model)

def run(topk, noise):
    global eval_method, date, dataset, benchmark
    res_file = f"{benchmark}/results/{date}_{dataset}_compress_{eval_method}_noise{noise}_topk{topk}.json"
    case_file = f"{benchmark}/datasets/case_{date}_{dataset}_summary_baseline_compress_{eval_method}_noise{noise}_topk{topk}.json"
    with open(case_file, "r", encoding="utf-8") as lines:
        cases = json.load(lines)
        num_slices = 10
        slice_length = len(cases) // num_slices
        slices = [cases[i:i+slice_length] for i in range(0, len(cases), slice_length)]
        final_result = []
        for slice_cases in slices:
            result = process_slice(slice_cases)
            final_result.extend(result)
        with open(res_file, "w", encoding="utf-8") as json_file:
            json.dump(final_result, json_file, ensure_ascii=False, indent=4)

for topk in topkk:
    for noise in noises:
        run(topk, noise)
        print(f"In Eval Baseline RAG: {topk}, {noise}")