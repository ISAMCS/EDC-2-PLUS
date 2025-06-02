import re
import string
import os
import requests
from dotenv import load_dotenv

# Load API key from .env file
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

def Microsoft_Phi4_request(prompt, temperature=0.0):
    return microsoft_phi4_reasoning_plus_request(prompt, temperature)

# Wrapper functions that re-use llama4_maverick_request
def GPT_Instruct_request(prompt, temp=0.0):
    return llama4_maverick_request(prompt, temp)

def ChatGPT_request(prompt, temperature=0.0):
    return llama4_maverick_request(prompt, temperature)

def GPT4o_request(prompt):
    return llama4_maverick_request(prompt, 0.0)

def qwen_request(prompt):
    return llama4_maverick_request(prompt, 0.0)
