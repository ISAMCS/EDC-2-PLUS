import torch
import json
from tqdm import tqdm
import os
import sys
import requests
import ast
import time
import os
from Codespace.LLMs import utils
from copy import deepcopy

# python codes/run_methods/eval_baseline_compress.py gpt35_turbo 0604 triviaq "[20]" "[0]" full

eval_model = sys.argv[1] 
date = sys.argv[2]
dataset = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]

# Model selection
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
    global topk
    topk = int(topk)
    ref_text = "\n".join([f"{i+1}.{case['summary_docs_baseline'][i]}" for i in range(topk)])
    prompt = (
        "## Task ##\n"
        "You are an expert question-answering assistant. Your job is to extract and provide the most canonical, standard, or widely accepted answer to the question, using ONLY the information provided in the reference text below.\n"
        "\n"
        "## Instructions ##\n"
        "• Carefully read the question and identify all requirements (entities, numbers, dates, etc).\n"
        "• Determine what would be considered the most canonical or standard answer in a reputable reference source.\n"
        "• Use all relevant evidence from the reference text to extract and provide this canonical answer. Combine information from multiple sentences if needed.\n"
        "• Prefer facts, entities, or statements that use standard names, codes, or formats (e.g., ISO codes, official names, standard spellings) when relevant.\n"
        "• Match the style and level of detail of a high-quality ground truth answer.\n"
        "• If the question requires counting or listing (e.g., \"How many teams end in United?\"), list the entities that meet ALL criteria, then provide the count or list as the answer.\n"
        "• If the answer is a name, number, date, or list, provide it directly and concisely.\n"
        "• If the answer cannot be determined from the reference text, write 'Unknown' after Final Answer:.\n"
        "• Do NOT use any outside knowledge. Only use the reference text.\n"
        "• Output ONLY the most canonical answer after the tag 'Final Answer:', on a single line. Do not include explanations or reasoning in your output.\n"
        "\n"
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

def run(topk, noise):
    global eval_method, date, dataset, benchmark
    res_file = f"{dataset}/results/{date}_{dataset}_compress_{eval_model}_noise{noise}_topk{topk}.json"
    case_file = f"{dataset}/datasets/case_{date}_{dataset}_summary_baseline_compress_{eval_model}_noise{noise}_topk{topk}.json"
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