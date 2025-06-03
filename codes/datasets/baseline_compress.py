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
elif eval_model == "local_hf_request":
    assess_model = local_hf_request

def normalize_text(text):
    """Lowercase, remove punctuation, articles and extra whitespace."""
    def remove_articles(s):
        return re.sub(r'\b(a|an|the)\b', ' ', s)
    def white_space_fix(s):
        return ' '.join(s.split())
    def remove_punc(s):
        return ''.join(ch for ch in s if ch not in set(string.punctuation))
    def lower(s):
        return s.lower()
    return white_space_fix(remove_articles(remove_punc(lower(text))))

def extract_best_answer(passages, gold_answers):
    # Normalize gold answers
    norm_gold = [normalize_text(ans) for ans in gold_answers]
    best_f1 = 0
    best_ans = ""
    for passage in passages:
        text = passage["text"]
        # Try to find any gold answer alias in the passage
        for gold, norm_gold_ans in zip(gold_answers, norm_gold):
            if normalize_text(gold) in normalize_text(text):
                # Direct match, return the gold alias as answer
                return gold
        # Otherwise, try to extract the most overlapping span
        for gold, norm_gold_ans in zip(gold_answers, norm_gold):
            passage_norm = normalize_text(text)
            overlap = set(passage_norm.split()) & set(norm_gold_ans.split())
            if len(overlap) > best_f1:
                best_f1 = len(overlap)
                best_ans = gold
    # Fallback: return the most common gold alias if nothing matches
    return best_ans if best_ans else gold_answers[0]

filter_paragraph = ["No content to", "no content to", "I'm sorry", "I am sorry", "I can not provide", "I can't provide", "Could you clarify", "Sorry, I", "Could you clarify", "?"]

def _run_nli_GPT3(question, answers, doc):
    prompt = f"""
    You are given a question, a list of possible correct answers, and a document. Extract any key point or evidence from the document that directly answers the question, especially if it matches or supports any of the provided answers.

    - Focus only on information relevant to the question and answers.
    - Use concise language and quote or closely paraphrase the document's wording.
    - Do not add any information that is not present in the document.
    - If the document does not contain relevant information, respond with: "No content to extract."
    - If possible, return only the relevant phrase or sentence, not the whole document.

    Question: {question}
    Possible Answers: {answers}
    Document: {doc}
    Relevant content:
    """
    while True:
        try:
            text = assess_model(prompt)
            if not text or "No content" in text:
                return "No content to extract."
            return text.split("\n")[0].strip()
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

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

        question = case.get("question", "")
        answers = case.get("answers", [""])

        for i in range(0, len(docs), 20):
            doc_chunk = "\n\n".join([f"{j+1}. {doc}" for j, doc in enumerate(docs[i:i+20])])
            k = 0
            extracted_docs = []

            while len(extracted_docs) != len(docs[i:i+20]) and k < 3:
                compressed_text = _run_nli_GPT3(question, answers, doc_chunk)
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
        "gpt35_turbo_0613_request": "3.5turbo",
        "local_hf_request": "local"
    }.get(eval_model, eval_model)
    
    res_file = f"{benchmark}/datasets/case_{date}_{dataset}_summary_baseline_compress_{eval_method}_noise{noise}_topk{topk}.json"
    case_file = f"{benchmark}/datasets/{benchmark}_results_random_{dataset}_w_negative_passages_noise{noise}_topk{topk}_embedding.json"
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