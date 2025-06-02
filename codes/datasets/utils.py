import re
import string
import json
import concurrent.futures
import functools
import time
import os
import requests
from dotenv import load_dotenv

load_dotenv()
LLAMA4_API_KEY = os.getenv("KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MICROSOFT_API_KEY = os.getenv("MICROSOFT_KEY")

def llama4_maverick_request(prompt: str, temperature: float = 0.0) -> str:
    headers = {
        "Authorization": f"Bearer {LLAMA4_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/llama-4-maverick:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 512,
        "stop": ["</s>"]
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def microsoft_phi4_reasoning_plus_request(prompt: str, temperature: float = 0.0) -> str:
    headers = {
        "Authorization": f"Bearer {MICROSOFT_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "microsoft/phi-4-reasoning-plus:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 512,
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 401:
        raise Exception("❌ 401 Unauthorized: Check if your OpenRouter key is correct and if your account has access to this model (paid model).")

# Route other models to the same local model for now
def ChatGPT_request(prompt, temperature=0.0):
    return llama4_maverick_request(prompt, temperature)

def GPT_Instruct_request(prompt, temp=0.0):
    return llama4_maverick_request(prompt, temp)

def GPT4o_request(prompt):
    return llama4_maverick_request(prompt, 0.0)

def qwen_request(prompt, temp=0.0):
    return llama4_maverick_request(prompt, temp)

def count_tokens(text: str) -> int:
    return len(text.split())

def run(topk, res_file, case_file, process_slice):
    topk = int(topk)    
    with open(case_file, "r", encoding="utf-8") as lines:
        cases = json.load(lines)
        num_slices = 10
        slice_length = len(cases) // num_slices
        slices = [cases[i:i+slice_length] for i in range(0, len(cases), slice_length)]
        final_result = []
        # Parallel evaluation of slices
        results = []
        process_slice = functools.partial(process_slice, topk=topk)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(process_slice, slices)
        # Merge results
        for result in results:
            final_result.extend(result)
        with open(res_file, "w", encoding="utf-8") as json_file:
            json.dump(final_result, json_file, ensure_ascii=False, indent=4)