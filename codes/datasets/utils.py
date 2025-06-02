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