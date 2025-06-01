from llama_cpp import Llama
import re
import string
import json
import concurrent.futures
import functools
import time

GGUF_PATH = "models/llama-2-7b-chat.Q4_0.gguf"

# Instantiate the Llama model once
llm = Llama(
    model_path=GGUF_PATH,
    n_ctx=2048,
    n_threads=8
)

def llama_local_request(prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
    """
    Send 'prompt' to the local llama-2-7b-chat.Q4_0.gguf model.
    Returns the raw text of the first choice.
    """
    out = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["</s>"]
    )
    return out["choices"][0]["text"].strip()

def ChatGPT_request(prompt, temperature=0.0):
    return llama_local_request(prompt, temperature)

def GPT_Instruct_request(prompt, temp=0.0):
    return llama_local_request(prompt, temp)

def GPT4o_request(prompt):
    return llama_local_request(prompt, 0.0)

def llama3_request(prompt, temp=0.0):
    return llama_local_request(prompt, temp)

def qwen_request(prompt, temp=0.0):
    return llama_local_request(prompt, temp)

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