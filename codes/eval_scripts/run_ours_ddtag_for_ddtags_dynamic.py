import sys
import subprocess
import os
import ast
date = sys.argv[1] # 0602
dataset = sys.argv[2] # redundancy
eval_model = sys.argv[3]  # gpt35_turbo_0613_request
topkk = sys.argv[4] # "[20]""
noises = sys.argv[5] # "[40]"
summary_prompt = sys.argv[6] # 1110
clustering_type = sys.argv[7] # dynamic 
benchmark = sys.argv[8] # trviaq

'''
# Cannot run this script as the files as there are files missing 
from this codebase. View this message

"Up to now, we have only released the evaluation code and datasets 
related to the main experiments on TriviaQA and WebQ. We will later
update the code for the ablation studies and hallucination detection
atasets. If you need additional datasets or code, please feel free
to contact us."

Use 0602 as the date
Use redundancy as the dataset
Use gpt35_turbo_0613_request as the eval model
Use [20] for topkk
Use [40] for noises
Use 1110 for summary_prompt
Use dynamic for clustering_type
Use triviaq as the benchmark

'''

#  python codes/eval_scripts/run_ours_ddtag_for_ddtags_dynamic.py 0602 redundancy gpt35_turbo_0613_request "[20]" "[20,40,60,80]" 1110 dynamic trviaq

for length in ["3"]:

    print(f"start_to_run_{length}")
    print("start_to_get_ddtags")
    # python codes/datasets/get_ddtags_for_ddtags_dynamic.py 3 redundancy triviaq "[20]" "[40]" 1110 dynamic
    subprocess.run(["python", "codes/datasets/get_tag_doc_doc_similarity_dynamic.py", topkk, noises, length, dataset,benchmark])
    print("end_get_ddtags")
    print("start_to_summarize")
    subprocess.run(["python", "codes/datasets/using_ddtags_to_summary_for_ddtags_dynamic.py", topkk, noises, length, dataset, eval_model, date, summary_prompt, clustering_type,benchmark])
    print("end_summarize")
    print("start_to_eval")
    subprocess.run(["python", "codes/run_methods/eval_ours_ddtag_summary_for_ddtags_dynamic.py", date, dataset, eval_model, topkk, noises, length, summary_prompt, clustering_type,benchmark])
    print("end_eval")
    print("start_to_extracte_answer")  
    eval_method = {
        "llama4_request": "llama4",
        "Phi_request": "Phi",
        "ChatGPT_request": "3.5turbo",
        "mistral7b_instruct_request": "mistral7b",
        "gpt35_turbo_0613_request": "3.5turbo"
    }.get(eval_model, eval_model)
    subprocess.run(["python", "codes/eval_metric/extracted_answer_topkk_for_ddtags_dynamic.py", date, dataset, eval_method, topkk, noises, length, summary_prompt, clustering_type,benchmark])
    print("end_extracte_answer")
    print("start_to_caculate_F1_EM")
    subprocess.run(["python", "codes/eval_metric/caculate_F1_EM_for_ddtags_dynamic.py", date, dataset, eval_method, topkk, noises, length, summary_prompt, clustering_type,benchmark])
    print("end_caculate_F1_EM")
    print(f"end_run_{length}")