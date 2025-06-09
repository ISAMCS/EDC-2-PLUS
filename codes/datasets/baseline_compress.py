#python codes/datasets/baseline_compress.py llama3_request 0601 full "[20]" "[40]" triviaq

import torch
import json
from tqdm import tqdm
import os
import sys
import requests
from requests.auth import HTTPBasicAuth
import concurrent.futures
from transformers import AutoModelForCausalLM, AutoTokenizer
from codes.datasets.utils import llama4_maverick_request, Microsoft_Phi4_request, mistral7b_instruct_request, gpt35_turbo_0613_request, local_hf_request
import re
import string
import ast
import numpy as np
np.random.seed(42)
from copy import deepcopy

'''

eval_model, date, dataset, topkk ,noises, benchmark])

'''

eval_model = sys.argv[1] 
date = sys.argv[2]
dataset = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]

if eval_model == "llama4":
    assess_model = llama4_maverick_request
elif eval_model == "Phi":
    assess_model = Microsoft_Phi4_request
elif eval_model == "mistral7b_instruct":
    assess_model = mistral7b_instruct_request
elif eval_model == "gpt35_turbo":
    assess_model = gpt35_turbo_0613_request
elif eval_model == "local_hf":
    assess_model = local_hf_request

filter_paragraph = ["No content to", "no content to", "I'm sorry", "I am sorry", "I can not provide", "I can't provide", "Could you clarify", "Sorry, I", "Could you clarify", "?"]

def _run_nli_GPT3(num, docs, question):
    global eval_model
    prompt = f"""
    # Instruction:
    You are given a question and {num} documents.
    Extract (do NOT answer) only the sentences or bullet points that provide direct evidence for the most canonical, standard, or widely accepted answer to the question.

    # Question:
    "{question}"

    # Extraction Instructions:
    1. Carefully identify exactly what the question is asking, including all constraints (such as entity type, date, location, number, or other attributes).
    2. Determine what would be considered the most canonical or standard answer in a reputable reference source.
    3. Extract only the sentences or bullet points that directly support or provide evidence for this canonical answer.
    4. Prefer facts, entities, or statements that use standard names, codes, or formats (e.g., ISO codes, official names, standard spellings) when relevant.
    5. If a reference is ambiguous, use the surrounding context to decide if it meets the canonical criteria.
    6. Implicit references are acceptable only if the context clearly supports them.
    7. Exclude sentences that are off-topic, speculative, or do not provide direct evidence for the canonical answer.
    8. Do NOT include commentary, explanations, or any content that does not directly support the canonical answer.
    9. Extract evidence in a way that maximizes overlap with the canonical answer in relation to the question (the most accurate answer is most important)

    # Documents:
    {docs}

    # Extracted Documents:
    1. <to be extracted>
    2. <to be extracted>
    ...
    {num}. <to be extracted>
    """
    while True:
        try:
            text = assess_model(prompt)
            return text.strip()
        except Exception as e:
            print(f"An error occurred: {e}")

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
        question = case.get("question")
        
        for i in range(0, len(docs), 20):
            doc_chunk = "\n\n".join([f"{j+1}. {doc}" for j, doc in enumerate(docs[i:i+20])])
            k = 0
            extracted_docs = []
            
            while len(extracted_docs) != len(docs[i:i+20]) and k < 3:
                compressed_text = _run_nli_GPT3(len(docs[i:i+20]), doc_chunk, question=question)
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
    
    res_file = f"{dataset}/datasets/case_{date}_{dataset}_summary_baseline_compress_{eval_model}_noise{noise}_topk{topk}.json"
    case_file = f"{dataset}/datasets/{dataset}_results_random_{benchmark}_w_negative_passages_noise{noise}_topk{topk}_embedding.json"
    with open(case_file, "r", encoding="utf-8") as lines:
        cases = json.load(lines)
        num_slices = 20
        slice_length = max(1, len(cases) // num_slices)
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
        print(f"Finished running for topk={topk} and noise={noise} In Summarize Docs")