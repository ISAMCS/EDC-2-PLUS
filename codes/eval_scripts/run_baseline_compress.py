import sys
import subprocess

def select_summarization_method(doc_length, relevance_score):
    """
    Adaptive policy for selecting summarization method.
    - If doc is short, use extractive (TextRank).
    - If doc is long and relevance is high, use abstractive (LLaMA-2).
    - If doc is long and relevance is low, use extractive (KeyBERT).
    """
    if doc_length < 500:
        return "textrank"
    elif doc_length >= 500 and relevance_score > 0.7:
        return "llama2"
    else:
        return "keybert"

eval_model = sys.argv[1]  # llama3_request, GPT_Instruct_request, ChatGPT_request
date = sys.argv[2]
dataset = sys.argv[3]  # 400 or 113
topkk = sys.argv[4]  # "[20, 50, 70]"
noises = sys.argv[5]  # "[20 ,60, 80]"
benchmark = sys.argv[6]

doc_length = int(sys.argv[7]) if len(sys.argv) > 7 else 1000
relevance_score = float(sys.argv[8]) if len(sys.argv) > 8 else 0.8

#python run_baseline_compress.py ChatGPT_request 0415 full "[5,20]" "[0,20,40,60,80,100]" hotpotqa 1200 0.85

summarization_method = select_summarization_method(doc_length, relevance_score)
print(f"Selected summarization method: {summarization_method}")

print("start_to_run")
print("start_to_summarize")
subprocess.run([
    "python", "codes/datasets/baseline_compress.py",
    eval_model, date, dataset, topkk, noises, benchmark, summarization_method
])
print("end_summarize")
print("start_to_eval")
subprocess.run([
    "python", "codes/run_methods/eval_baseline_compress.py",
    eval_model, date, dataset, topkk, noises, benchmark, summarization_method
])
print("end_eval")
print("start_to_extract_answer")
if eval_model == "llama3_request":
    eval_method = "eval_llama3"
elif eval_model == "GPT_Instruct_request":
    eval_method = "eval_3.5instruct"
elif eval_model == "ChatGPT_request":
    eval_method = "eval_3.5turbo"
elif eval_model == "GPT4o_request":
    eval_method = "eval_4o"
elif eval_model == "qwen_request":
    eval_method = "eval_qwen"
subprocess.run([
    "python", "codes/eval_metric/extracted_answer_topkk_compress.py",
    date, dataset, eval_method, topkk, noises, benchmark, summarization_method
])
print("end_extracte_answer")
print("start_to_caculate_F1_EM")
subprocess.run([
    "python", "codes/eval_metric/caculate_F1_EM_compress.py",
    date, dataset, eval_method, topkk, noises, benchmark, summarization_method
])
print("end_caculate_F1_EM")