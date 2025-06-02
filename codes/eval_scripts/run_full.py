import sys
import subprocess

date = sys.argv[1]
benchmark = sys.argv[2]

subprocess.run(["python", "codes/datasets/make_datasets.py", benchmark])
subprocess.run(["python", "codes/eval_scripts/run_baseline_wo_retrieve.py", "llama3_request", date, "full", benchmark])
subprocess.run(["python", "codes/eval_scripts/run_baseline_rag.py", "llama3_request", date, "full", "[20]", "[0,20,40,60,80,100]", benchmark])
subprocess.run(["python", "codes/eval_scripts/run_baseline_compress.py", "GPT", date, "full", "[20]", "[0,20,40,60,80,100]", benchmark])
subprocess.run(["python", "codes/eval_scripts/run_baseline_compress.py", date, "full", "GPT", "[20]", "[0,20,40,60,80,100]", "1110", "dynamic", benchmark])