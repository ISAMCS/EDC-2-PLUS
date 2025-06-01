# File: codes/utils.py

from llama_cpp import Llama

# ----- Change this path if your gguf is somewhere else -----
GGUF_PATH = "models/llama-2-7b-chat.Q4_0.gguf"

# instantiate one Llama() that will be re-used for every call
llm = Llama(
    model_path=GGUF_PATH,
    n_ctx=2048,
    n_threads=8   # or however many threads you want
)

def llama3_request(prompt: str, temperature: float = 0.0) -> str:
    """
    Send 'prompt' to the local llama-2-7b-chat.Q4_0.gguf model.
    Returns the raw text of the first choice.
    """
    out = llm(
        prompt,
        max_tokens=512,
        temperature=temperature,
        stop=["</s>"]
    )
    return out["choices"][0]["text"].strip()


# If you want to keep the old GPT wrappers around (no change in logic), 
# just leave them as stubs or pass them through to llama3_request:
def GPT_Instruct_request(prompt, temp=0.0):
    return llama3_request(prompt, temp)

def ChatGPT_request(prompt, temperature=0.0):
    return llama3_request(prompt, temperature)

def GPT4o_request(prompt):
    return llama3_request(prompt, 0.0)

def qwen_request(prompt):
    return llama3_request(prompt, 0.0)
