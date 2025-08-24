import torch
from transformers import AutoTokenizer, AutoModel
# Use a stronger embedding model
path = "sentence-transformers/all-MiniLM-L6-v2"  # Free model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_model = AutoModel.from_pretrained(path, torch_dtype=torch.float16).to(device)

def get_embedding(text):
    # Simple in-memory cache
    if not hasattr(get_embedding, "_cache"):
        get_embedding._cache = {}
    cache = get_embedding._cache
    if text in cache:
        return cache[text]
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        feature = embed_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    emb = feature.squeeze().cpu().numpy()
    cache[text] = emb
    return emb

import re
import string
import os
import requests
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModel
import torch
load_dotenv()
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

def microsoft_phi4_reasoning_plus_request(prompt: str, temperature: float = 0.0) -> str:
    headers = {
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "microsoft/phi-4-reasoning-plus:free",  # no :free
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 512,
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 401:
        raise Exception("❌ 401 Unauthorized: Check if your OpenRouter key is correct and if your account has access to this model (paid model).")

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

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

def gpt35_turbo_0613_request(prompt: str, temperature: float = 0.0, max_retries: int = 5, backoff_factor: float = 2.0) -> str:
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
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait = backoff_factor ** attempt
                print(f"Request failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                print(f"Request failed after {max_retries} attempts: {e}")
                raise

def Microsoft_Phi4_request(prompt, temperature=0.0):
    return microsoft_phi4_reasoning_plus_request(prompt, temperature)

# Hugging Face API requests for various models

_local_model = None
_local_tokenizer = None

MODEL="microsoft/phi-2"

def load_local_model(model_name=MODEL, device=None):
    global _local_model, _local_tokenizer
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _local_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _local_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def local_hf_request(prompt: str, temperature: float = 0.0, max_new_tokens: int = 512, model_name=MODEL):
    global _local_model, _local_tokenizer
    if _local_model is None or _local_tokenizer is None:
        load_local_model(model_name)
    device = next(_local_model.parameters()).device
    inputs = _local_tokenizer(prompt, return_tensors="pt").to(device)
    do_sample = temperature > 0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": _local_tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        outputs = _local_model.generate(
            **inputs,
            **gen_kwargs
        )
    response = _local_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Optionally, strip the prompt from the response
    return response[len(prompt):].strip() if response.startswith(prompt) else response.strip()