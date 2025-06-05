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

date = sys.argv[1]
dataset = sys.argv[2]
eval_model = sys.argv[3]
benchmark = sys.argv[4]

def _run_nli_GPT3turbo(case):
    prompt = f"""Task Description:

    You are given a question, a golden answer, and a generated answer.  

    Inputs:

    Question: {case["question"]}
    Golden Answer: {case["answers"][0]}
    Generated Answer: {case["response"]}

    Instructions:

    1. Carefully identify exactly what the question is asking, including all constraints (such as entity type, date, location, number, or other attributes).
    2. Determine the type of answer the question is looking for (numeric, entity, date, etc.)
    3. Determine how certain ambigious answers may be important to the question (e.g if it is asking "How many" then pay attention to the entity that is is asking for and key words for that)
    4. Extract the essential information from the {case["response"]}
    5. View the main concept, idea, concept, and structure of the golden answer
    6. Compare the main concept, idea, concept, and structure of the generated answer with the golden answer
    7. Reformat it to match the structure of the golden answer only if it is present in the {case["response"]}
    8. Carefully compare the generated answer with the golden answer, and extract key information from the generated answer 
    9. Generate an output that you extracted from the generated answer that closely matches the style and structure of the golden answer
    
    Guidelines:
    1. Keep only information essential to the question.
    3. Ignore extra or unrelated details.  
    4. If no valid answer is present, return "No valid answer found."
    5. If the generated answers does not contain any information or does not contain relevant information return "None"
    6. Only extract the information out of the generatted answer, use other information as a guide for the format but not for the content

    Output Format:
    <your concise, reformatted answer here>
    """

    Question: {case["question"]}
    Golden Answer: {case["answers"][0]}
    Generated Answer: {case["response"]}
    
    Requirements:

    1. Carefully identify exactly what the question is asking, including all constraints (such as entity type, date, location, number, or other attributes).
    2. Determine the type of answer the question is looking for (numeric, entity, date, etc.)
    3. Determine how certain ambigious answers may be important to the question (e.g if it is asking "How many" then pay attention to the entity that is is asking for and key words for that)

    Extract information from the generated answer that corresponds to the essential content of the golden answer.
    Reorganize the extracted content to align with the structure of the golden answer, including phrasing and order of information where relevant.
    If the generated answer contains information not covered in the golden answer, include only information crucial to answering the question. Disregard redundant or irrelevant details.

    Output Format:

    Reformatted Answer:"""
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
    case_file = f"{dataset}/results/{date}_{dataset}_compress_{eval_model}_noise{noise}_topk{topk}.json"
    res_file = f"{dataset}/extracted_answer/{date}_{dataset}_compress_{eval_model}_noise{noise}_topk{topk}.json"
    with open(case_file, "r", encoding="utf-8") as lines:
        cases = json.load(lines)
        # Sequentially process all cases
        final_result = process_slice(cases)
        with open(res_file, "w", encoding="utf-8") as json_file:
            json.dump(final_result, json_file, ensure_ascii=False, indent=4)

run()