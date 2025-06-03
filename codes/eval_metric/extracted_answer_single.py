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
from codes.text_utils import llama4_maverick_request, Microsoft_Phi4_request, mistral7b_instruct_request, gpt35_turbo_0613_request, local_hf_request
# CHANGE THIS TO THE DESIRED MODEL FUNCTION
eval_model =  gpt35_turbo_0613_request

date = sys.argv[1]
dataset = sys.argv[2]
eval_method = sys.argv[3]
benchmark = sys.argv[4]

def extract_clean_answer(raw_text):
    for line in raw_text.splitlines():
        if "Reformatted Answer:" in line:
            return line.split("Reformatted Answer:")[-1].strip()
    return raw_text.strip()


def _run_nli_GPT3turbo(case):
    prompt = f"""Task Description:
    You need to extract the essential answer from a generated answer and reformat it to match the style and structure of the golden answer. We will provide a question, a list of acceptable golden answers, and a generated answer. Your job is to extract the key answer from the generated answer, and rewrite it as concisely as possible, matching the phrasing and format of the golden answers. Do not include extra context or full sentences—just the answer itself.

    Input:

    Question: {case["question"]}
    Golden Answers: {case["answers"]}
    Generated Answer: {case["response"]}

    Requirements:
    - Extract only the essential answer from the generated answer.
    - Reformat it to match the style and structure of the golden answers (e.g., short phrase or noun, not a full sentence).
    - If possible, select or closely match one of the golden answers.
    - Do not include any extra explanation or context.

    Output Format:
    Reformatted Answer: <your concise answer here>
    """
    res = 0
    while (True):
        try:
            text = eval_model(prompt)
            break
        except Exception as e:
            print(f"An error occurred: {e}")
    return text
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
        case["extracted_answer"] = extract_clean_answer(text)
    return slice_cases

def run():
    global eval_method, date, dataset
    res_file = f"{benchmark}/extracted_answer/{date}_{dataset}_baseline_wo_retrieve_{eval_method}.json"
    case_file = f"{benchmark}/results/{date}_{dataset}_baseline_wo_retrieve_{eval_method}.json"
    with open(case_file, "r", encoding="utf-8") as lines:
        cases = json.load(lines)
        # Sequentially process all cases
        final_result = process_slice(cases)
        with open(res_file, "w", encoding="utf-8") as json_file:
            json.dump(final_result, json_file, ensure_ascii=False, indent=4)

run()