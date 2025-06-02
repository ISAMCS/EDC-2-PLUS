
'''
import sys
import subprocess
import os 

dataset = sys.argv[1]
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "2"  # 只使用第0号GPU

subprocess.run(["python", "codes/datasets/get_embedding.py", dataset], env=env)
print("start_to_classify_docs")
subprocess.run(["python", "codes/datasets/classify_noise_topk.py", dataset])

'''

import sys
import subprocess
import os

DATE = "0601"
BENCHMARK = "triviaq"
SUMMARISER = "llama4_request"
TOPKK = "[20]"
NOISES = "[0,20,40,60,80,100]"
EVAL_METHOD = "eval_llama3"

dataset = BENCHMARK
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "2"  # 只使用第0号GPU

subprocess.run(["python", "codes/datasets/get_embedding.py", dataset], env=env)
print("start_to_classify_docs")
subprocess.run([
    "python", "codes/datasets/classify_noise_topk.py",
    dataset, TOPKK, NOISES
])
