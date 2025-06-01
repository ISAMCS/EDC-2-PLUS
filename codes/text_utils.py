from llama_cpp import Llama
import re
import string

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

def clean_text(text: str) -> str:
    """
    Clean input text by removing extra whitespace and non-printable characters.
    """
    text = re.sub(r'\s+', ' ', text)
    text = ''.join(c for c in text if c.isprintable())
    return text.strip()


def normalize_text(text: str) -> str:
    """
    Lowercase, remove punctuation, articles, and extra whitespace.
    """
    def remove_articles(s):
        return re.sub(r'\b(a|an|the)\b', ' ', s)
    def white_space_fix(s):
        return ' '.join(s.split())
    def remove_punc(s):
        return ''.join(ch for ch in s if ch not in set(string.punctuation))
    def lower(s):
        return s.lower()
    return white_space_fix(remove_articles(remove_punc(lower(text))))

def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score (as in SQuAD).
    """
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    num_same = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common)
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def count_tokens(text: str, tokenizer=None) -> int:
    """
    Returns the number of tokens in the text.
    If a tokenizer is provided, use it; otherwise, split on whitespace.
    """
    if tokenizer:
        return len(tokenizer.encode(text))
    return len(text.split())

def to_lower(text: str) -> str:
    """
    Convert text to lowercase.
    """
    return text.lower()

def GPT_Instruct_request(prompt, temp=0.0):
    return llama3_request(prompt, temp)

def ChatGPT_request(prompt, temperature=0.0):
    return llama3_request(prompt, temperature)

def GPT4o_request(prompt):
    return llama3_request(prompt, 0.0)

def qwen_request(prompt):
    return llama3_request(prompt, 0.0)
