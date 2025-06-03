import sys
import subprocess

date = sys.argv[1] # 0602
benchmark = sys.argv[2] # triviaq

# python codes/datasets/make_datasets.py triviaq "[20]" "[40]"

subprocess.run(["python", "codes/datasets/make_datasets.py", benchmark])

# python codes/eval_scripts/run_baseline_wo_retrieve.py gpt35_turbo_0613_request 0602 full triviaq

subprocess.run(["python", "codes/eval_scripts/run_baseline_wo_retrieve.py", "llama3_request", date, "full", benchmark])

# python codes/eval_scripts/run_baseline_rag.py gpt35_turbo_0613_request 0602 full "[20]" "[40]" triviaq

subprocess.run(["python", "codes/eval_scripts/run_baseline_rag.py", "llama3_request", date, "full", "[20]", "[40]", benchmark])

# python codes/eval_scripts/run_baseline_compress.py gpt35_turbo_0613_request 0602 full "[20]" "[40]" triviaq

subprocess.run(["python", "codes/eval_scripts/run_baseline_compress.py", "GPT", date, "full", "[20]", "[40]]", benchmark])

#subprocess.run(["python", "codes/eval_scripts/run_baseline_compress.py", date, "full", "GPT", "[20]", "[40]", "1110", "dynamic", benchmark])