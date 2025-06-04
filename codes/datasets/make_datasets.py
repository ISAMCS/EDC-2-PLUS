import sys
import subprocess
import os 

#  python codes/datasets/make_datasets.py triviaq "[20]" "[0]" gpt35_turbo

dataset = sys.argv[1]
topkk = sys.argv[2]  # e.g. "[20]"
noises = sys.argv[3] # e.g. "[40]"
eval_model = sys.argv[4]  # e.g. "llama4_maverick_request"

# python codes/datasets/get_embedding.py triviaq
print("start_to_run")
subprocess.run(["python", "codes/datasets/get_embedding.py", dataset])
print("end_get_embedding")
print("start_to_classify_docs")
# python codes/datasets/classify_noise_topk.py triviaq "[20]" "[0]" gpt35_turbo
subprocess.run(["python", "codes/datasets/classify_noise_topk.py", dataset, topkk, noises, eval_model])
print("end_classify_docs")