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
eval_model =  llama4_maverick_request

date = sys.argv[1]
dataset = sys.argv[2]
eval_method = sys.argv[3]
benchmark = sys.argv[4]
def _run_nli_GPT3turbo(case):
    prompt = f"""Task Description:
You are given a question, a list of possible answer choices (the "Golden Answer" list), and a generated answer. Your task is to extract only the essential keywords or phrases from the generated answer that best match one of the provided answer choices, with no extra words or rephrasing. Select the answer choice that is the closest match in meaning and wording, guided by the context and the answer list. Do not add information not present in the answer choices. Your output should be the single best-matching answer choice, exactly as it appears in the list.

Input:

Question: {case["question"]}
Answer Choices: {case["answers"]}
Generated Answer: {case["response"]}

Instructions:

- Compare the generated answer to the list of answer choices.
- Select and output only the answer choice that matches the generated answer most closely in meaning and wording.
- If there are multiple close matches, choose the one that is most similar in form and content.
- Do not add extra words, explanations, or rephrase the answer.
- Output only the selected answer choice, exactly as it appears in the list.

Output Format:
Reformatted Answer: """
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
        case["extracted_answer"] = text
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