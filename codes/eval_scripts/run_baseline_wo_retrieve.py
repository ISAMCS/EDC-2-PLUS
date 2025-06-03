import sys
import subprocess
eval_model = sys.argv[1]#llama3_request, GPT_Instruct_request, ChatGPT_request
date = sys.argv[2] # 0602
dataset = sys.argv[3] # full
benchmark = sys.argv[4] # triviaq

#python run_baseline_wo_retriev.py gpt35_turbo_0613_request 0602 full triviaq

print("start_to_run")
print("start_to_eval")
# python codes/run_methods/eval_baseline_wo_retrieve.py gpt35_turbo_0613_request 0602 full triviaq
subprocess.run(["python", "codes/run_methods/eval_baseline_wo_retrieve.py", eval_model, date, dataset,benchmark])
print("end_eval")
print("start_to_extracte_answer")
eval_method = {
    "llama4_request": "llama4",
    "Phi_request": "Phi",
    "mistral7b_instruct_request": "mistral7b",
    "gpt35_turbo_0613_request": "3.5turbo"
}.get(eval_model, eval_model)
# python codes/eval_metric/extracted_answer_single.py 0602 full 3.5turbo triviaq
subprocess.run(["python", "codes/eval_metric/extracted_answer_single.py", date, dataset, eval_method,benchmark])
print("end_extracte_answer")
print("start_to_caculate_F1_EM")
# python codes/eval_metric/caculate_F1_EM_single.py 0602 full 3.5turbo triviaq
subprocess.run(["python", "codes/eval_metric/caculate_F1_EM_single.py", date, dataset, eval_method, benchmark])
print("end_caculate_F1_EM")