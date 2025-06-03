import sys
import subprocess
import os 

#  python codes/datasets/make_datasets.py triviaq llama4_maverick_request "[20]" "[40]"

dataset = sys.argv[1]
topkk = sys.argv[2]  # e.g. "[20]"
noises = sys.argv[3] # e.g. "[40]"

# python codes/datasets/get_embedding.py triviaq llama4_maverick_request
subprocess.run(["python", "codes/datasets/get_embedding.py", dataset])
print("start_to_classify_docs")
# python codes/datasets/classify_noise_topk.py triviaq llama4_maverick_request "[20]" "[40]"
subprocess.run(["python", "codes/datasets/classify_noise_topk.py", dataset, topkk, noises])