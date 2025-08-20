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

# Parse arguments
# Usage: python extracted_answer_topkk_compress.py date dataset eval_model topkk noises benchmark [toggle] [output_file]
date = sys.argv[1]
dataset = sys.argv[2]
eval_model = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]
toggle = sys.argv[7] if len(sys.argv) > 7 else None
custom_output_file = sys.argv[8] if len(sys.argv) > 8 else None

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
        # Ablation logic: construct evidence for each toggle
        evidence_blocks = case.get('compressed_blocks', [])
        extra_info = ""
        if toggle == "stability":
            extra_info = f"Stability: {case.get('stability', '')}"
            evidence_blocks = [extra_info] if extra_info.strip() else []
        elif toggle == "temporal":
            extra_info = f"Temporal Class: {case.get('temporal_class', '')}"
            evidence_blocks = [extra_info] if extra_info.strip() else []
        elif toggle == "conflict":
            # Use claims/conflict info if available
            conflict_info = case.get('conflict', case.get('claims', ''))
            extra_info = f"Conflict Info: {conflict_info}"
            evidence_blocks = [extra_info] if extra_info.strip() else []
        elif toggle == "calibration":
            extra_info = f"Calibration/Confidence: {case.get('calibration', case.get('confidence', ''))}"
            evidence_blocks = [extra_info] if extra_info.strip() else []
        elif toggle == "all":
            # Use all available info
            all_blocks = []
            if 'stability' in case:
                all_blocks.append(f"Stability: {case['stability']}")
            if 'temporal_class' in case:
                all_blocks.append(f"Temporal Class: {case['temporal_class']}")
            if 'conflict' in case:
                all_blocks.append(f"Conflict Info: {case['conflict']}")
            elif 'claims' in case:
                all_blocks.append(f"Claims: {case['claims']}")
            if 'calibration' in case:
                all_blocks.append(f"Calibration: {case['calibration']}")
            elif 'confidence' in case:
                all_blocks.append(f"Confidence: {case['confidence']}")
            # Add original compressed blocks
            all_blocks.extend(case.get('compressed_blocks', []))
            evidence_blocks = all_blocks
        # Compose a new case for the ablation prompt
        ablation_case = case.copy()
        ablation_case['compressed_blocks'] = evidence_blocks
        text = _run_nli_GPT3turbo(ablation_case)
        case["extracted_answer"] = text
    return slice_cases

def run(topk, noise):
    global eval_model, date, dataset, toggle, custom_output_file
    # Compose input file name based on benchmark and toggle
    file_base = f"{dataset}/datasets/case_{date}_{dataset}_summary_edc2plus_compress_{eval_model}_noise{noise}_topk{topk}_{benchmark}"
    if toggle:
        file_base += f"_{toggle}"
    case_file = file_base + ".json"
    if custom_output_file:
        res_file = custom_output_file
    elif toggle:
        res_file = f"{dataset}/extracted_answer/{date}_{dataset}_edc2plus_compress_{eval_model}_noise{noise}_topk{topk}_{toggle}.json"
    else:
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