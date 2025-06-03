import sys
import subprocess
eval_model = sys.argv[1] #gpt35_turbo_0613_request
date = sys.argv[2] #602
dataset = sys.argv[3] # full
topkk = sys.argv[4] # "[20]"
noises = sys.argv[5] # "[40]"
benchmark = sys.argv[6] # triviaq

#python codes/eval_scripts/run_baseline_rag.py gpt35_turbo_0613_request 0602 full "[20]" "[40]" triviaq

print("start_to_run")
print("start_to_eval")
# python codes/run_methods/eval_baseline_rag.py gpt35_turbo_0613_request 0602 full "[20]" "[40]" triviaq
subprocess.run(["python", "codes/run_methods/eval_baseline_rag.py", eval_model, date, dataset,topkk,noises,benchmark])
print("end_eval")
print("start_to_extract_answer")
eval_method = {
    "llama4_request": "llama4",
    "Phi_request": "Phi",
    "mistral7b_instruct_request": "mistral7b",
    "gpt35_turbo_0613_request": "3.5turbo"
}.get(eval_model, eval_model)
    
etype = "rag"
# python codes/eval_metric/extracted_answer_topkk.py 0602 full eval_3.5turbo rag "[20]" "[40]" triviaq
subprocess.run(["python", "codes/eval_metric/extracted_answer_topkk.py", date, dataset, eval_method, etype,topkk,noises,benchmark])
print("end_extracte_answer")
print("start_to_caculate_F1_EM")   
# python codes/eval_metric/caculate_F1_EM.py 0602 full eval_3.5turbo rag "[20]" "[40]" triviaq
subprocess.run(["python", "codes/eval_metric/caculate_F1_EM.py", date, dataset, eval_method, etype,topkk,noises,benchmark])
print("end_caculate_F1_EM")