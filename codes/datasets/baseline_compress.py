#python codes/datasets/baseline_compress.py llama3_request 0601 full "[20]" "[40]" triviaq

import torch
import json
from tqdm import tqdm
import os
import sys
import requests
from requests.auth import HTTPBasicAuth
import concurrent.futures
from codes.datasets.utils import llama4_maverick_request, Microsoft_Phi4_request, mistral7b_instruct_request, gpt35_turbo_0613_request
import re
import ast


eval_model = sys.argv[1] 
date = sys.argv[2]
dataset = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]
if eval_model == "llama4_request":
    assess_model = llama4_maverick_request
elif eval_model == "Phi":
    assess_model = Microsoft_Phi4_request
elif eval_model == "mistral7b_instruct_request":
    assess_model = mistral7b_instruct_request
elif eval_model == "gpt35_turbo_0613_request":
    assess_model = gpt35_turbo_0613_request

filter_paragraph = ["No content to", "no content to", "I'm sorry", "I am sorry", "I can not provide", "I can't provide", "Could you clarify", "Sorry, I", "Could you clarify", "?"]

def _run_nli_GPT3(num, docs):
    global eval_model
    prompt = f"""
    You are given a question, a list of possible correct answers, and a document. Extract any key point or evidence from the document that directly answers the question, especially if it matches or supports any of the provided answers.

    - Focus only on information relevant to the question and answers.
    - Use concise language and quote or closely paraphrase the document's wording.
    - Do not add any information that is not present in the document.
    - If the document does not contain relevant information, respond with: "No content to extract."

    Question: <...>
    Possible Answers: <...>
    Document: <...>
    Relevant content:
    """
    while True:
        try:
            text = assess_model(prompt)
            if text is None:
                print("Warning: assess_model returned None. Returning default message.")
                return "No content to extract."
            return text.strip()
        except Exception as e:
            print(f"An error occurred: {e}")
            return "No content to extract."

def extract_numbered_sections(text):

    sections = {}
    lines = text.split("\n")
    current_index = None
    
    for line in lines:
        line = line.strip()
        match = re.match(r'^(\d+)\.\s*(.*)', line)
        if match:
            current_index = int(match.group(1))
            sections[current_index] = match.group(2)
        elif current_index is not None and line:
            sections[current_index] += " " + line
    
    return [sections[i].strip() for i in sorted(sections.keys())]


def process_slice(slice_cases):
    global topk, dataset
    for case in tqdm(slice_cases):
        for passage in case['passages']:
            if 'embedding' in passage:
                del passage['embedding']
        topk = int(topk)
        if dataset == "redundancy":
            docs = case["docs"]
        else:
            docs = [case['passages'][i]['text'].strip() for i in range(min(topk, len(case['passages'])))]
        compressed_docs = []
        times = 0
        
        for i in range(0, len(docs), 20):
            doc_chunk = "\n\n".join([f"{j+1}. {doc}" for j, doc in enumerate(docs[i:i+20])])
            k = 0
            extracted_docs = []
            
            while len(extracted_docs) != len(docs[i:i+20]) and k < 3:
                compressed_text = _run_nli_GPT3(len(docs[i:i+20]), doc_chunk)
                extracted_docs = extract_numbered_sections(compressed_text)
                k += 1
                times += 1
            
            if len(extracted_docs) != len(docs[i:i+20]):
                extracted_docs = docs[i:i+20]
            
            compressed_docs.extend(extracted_docs)
        
        case["summary_docs_baseline"] = compressed_docs
    return slice_cases


def run(topk, noise):
    global eval_model, date, dataset, benchmark
    eval_method = {
        "llama4_request": "llama4",
        "Phi_request": "Phi",
        "ChatGPT_request": "3.5turbo",
        "mistral7b_instruct_request": "mistral7b",
        "gpt35_turbo_0613_request": "3.5turbo"
    }.get(eval_model, eval_model)
    
    res_file = f"{benchmark}/datasets/case_{date}_{dataset}_summary_baseline_compress_{eval_method}_noise{noise}_topk{topk}.json"
    case_file = f"{benchmark}/datasets/{benchmark}_results_random_{dataset}_w_negative_passages_noise{noise}_topk{topk}_embedding.json"
    with open(case_file, "r", encoding="utf-8") as lines:
        cases = json.load(lines)
        num_slices = 20
        slice_length = max(1, len(cases) // num_slices)
        slices = [cases[i:i+slice_length] for i in range(0, len(cases), slice_length)]
        final_result = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(process_slice, slices)
        
        for result in results:
            final_result.extend(result)
        
        with open(res_file, "w", encoding="utf-8") as json_file:
            json.dump(final_result, json_file, ensure_ascii=False, indent=4)

for topk in topkk:
    for noise in noises:
        run(topk, noise)
        print(f"Finished running for topk={topk} and noise={noise} In Summarize Docs")