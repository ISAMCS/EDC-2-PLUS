import sys
import subprocess
eval_model = sys.argv[1] #llama4_maverick_request
date = sys.argv[2] # 0602
dataset = sys.argv[3] # full
topkk = sys.argv[4] # "[20, 50, 70]"
noises = sys.argv[5] # "[20 ,60, 80]"
benchmark = sys.argv[6] # triviaq

'''

source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.

eval_model=gpt35_turbo_0613_request
date=0603
dataset=full
topkk="[20]"
noises="[40]"
benchmark=triviaq
eval_method=3.5turbo

gpt35_turbo_0613_request
llama4_maverick_request
local_hf_request

python codes/eval_scripts/run_baseline_compress.py local_hf_request 0602 full "[20]" "[40]" triviaq

python codes/eval_scripts/run_baseline_compress.py gpt35_turbo_0613_request 0603 full "[20]" "[40]" triviaq

python codes/eval_scripts/run_baseline_compress.py llama4_maverick_request 0602 full "[20]" "[40]" triviaq

'''

#    python codes/eval_scripts/run_baseline_compress.py llama4_maverick_request 0602 full "[20]" "[40]" triviaq

# Run dataset preparation

#    python codes/datasets/make_datasets.py "$benchmark" "$eval_model" "$topkk" "$noises"
subprocess.run(["python", "codes/datasets/make_datasets.py", benchmark, topkk, noises])

print("start_to_run")
print("start_to_summarize")
# python codes/datasets/baseline_compress.py "$eval_model" "$date" "$dataset" "$topkk" "$noises" "$benchmark"
subprocess.run(["python", "codes/datasets/baseline_compress.py", eval_model, date, dataset,topkk,noises,benchmark])
print("end_summarize")
print("start_to_eval")
# python codes/run_methods/eval_baseline_compress.py "$eval_model" "$date" "$dataset" "$topkk" "$noises" "$benchmark"
subprocess.run(["python", "codes/run_methods/eval_baseline_compress.py", eval_model, date, dataset,topkk,noises,benchmark])
print("end_eval")
print("start_to_extract_answer")
print("start_to_extract_answer")
eval_method = {
    "llama4_request": "llama4",
    "Phi_request": "Phi",
    "mistral7b_instruct_request": "mistral7b",
    "gpt35_turbo_0613_request": "3.5turbo",
    "local_hf_request": "local"
}.get(eval_model, eval_model)
#    python codes/eval_scripts/run_baseline_compress.py gpt35_turbo_0613_request 0603 full "[20]" "[40]" triviaq
# python codes/eval_metric/extracted_answer_topkk_compress.py "$date" "$dataset" "$eval_method" "$topkk" "$noises" "$benchmark"
subprocess.run(["python", "codes/eval_metric/extracted_answer_topkk_compress.py", date, dataset, eval_method, topkk, noises,benchmark])
print("end_extracte_answer")
print("start_to_caculate_F1_EM")   
# python codes/eval_scripts/run_baseline_compress.py "$eval_method" "$date" "$dataset" "$topkk" "$noises" "$benchmark"
subprocess.run(["python", "codes/eval_metric/caculate_F1_EM_compress.py", date, dataset, eval_method, topkk, noises, benchmark])
print("end_caculate_F1_EM")