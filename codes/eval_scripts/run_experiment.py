import sys
import subprocess

date = sys.argv[1]
benchmark = sys.argv[2]

subprocess.run(["python", "codes/datasets/make_datasets.py", benchmark])
#subprocess.run(["python", "codes/eval_scripts/run_baseline_wo_retrieve.py", "llama3_request", date, "full", benchmark])
#subprocess.run(["python", "codes/eval_scripts/run_baseline_rag.py", "llama3_request", date, "full", "[20]", "[0,20,40,60,80,100]", benchmark])
subprocess.run(["python", "codes/eval_scripts/run_baseline_compress.py", "llama3_request", date, "full", "[20]", "[0,20,40,60,80,100]", benchmark])
subprocess.run(["python", "codes/eval_scripts/run_baseline_compress.py", date, "full", "llama3_request", "[20]", "[0,20,40,60,80,100]", "1110", "dynamic", benchmark])

'''

python codes/eval_scripts/run_experiment.py 0601 triviaq

python codes/datasets/make_datasets.py triviaq
python codes/eval_scripts/run_baseline_wo_retrieve.py llama3_request 0601 full triviaq
python codes/eval_scripts/run_baseline_rag.py llama3_request 0601 full "[20]" "[0,20,40,60,80,100]" triviaq
python codes/eval_scripts/run_baseline_compress.py llama3_request 0601 full "[20]" "[0,20,40,60,80,100]" triviaq
python codes/eval_scripts/run_baseline_compress.py 0601 full llama3_request "[20]" "[0,20,40,60,80,100]" 1110 dynamic triviaq

'''