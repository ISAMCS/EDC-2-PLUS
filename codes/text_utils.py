
# Open Router API requests for various models

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
# Get an API Key from OpenRouter and set it as an environment variable named "KEY"
KEY = os.getenv("KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def llama4_maverick_request(prompt: str, temperature: float = 0.0) -> str:
    headers = {
        "Authorization": f"Bearer {KEY}",
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

def Microsoft_Phi4_request(prompt: str, temperature: float = 0.0) -> str:
    headers = {
        "Authorization": f"Bearer {KEY}",
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

def mistral7b_instruct_request(prompt: str, temperature: float = 0.0) -> str:
    headers = {
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 512,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def gpt35_turbo_0613_request(prompt: str, temperature: float = 0.0) -> str:
    headers = {
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-3.5-turbo-0613",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 512,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

    '''

    Instructions: If you want to add a new model, please follow 
    the format of the existing models.

    def NAME_request(prompt: str, temperature: float = 0.0) -> str:
    headers = {
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "MODEL_NAME", 
        # e.g., "meta-llama/llama-4-maverick:free"
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 512,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

    '''

    # Local Hugging Face Models

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

_local_model = None
_local_tokenizer = None

def load_local_model(model_name="meta-llama/Llama-2-7b-hf", device=None):
    global _local_model, _local_tokenizer
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _local_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _local_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def local_hf_request(prompt: str, temperature: float = 0.0, max_new_tokens: int = 512, model_name="meta-llama/Llama-2-7b-hf"):
    global _local_model, _local_tokenizer
    if _local_model is None or _local_tokenizer is None:
        load_local_model(model_name)
    device = next(_local_model.parameters()).device
    inputs = _local_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = _local_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0
        )
    response = _local_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Optionally, strip the prompt from the response
    return response[len(prompt):].strip() if response.startswith(prompt) else response.strip()