import sys
import subprocess

# python codes/eval_scripts/run_baseline_compress.py gpt35_turbo 0604 triviaq "[20]" "[0]" full

eval_model = sys.argv[1] #llama4_maverick_request
date = sys.argv[2] # 0602
dataset = sys.argv[3] # triviaq
topkk = sys.argv[4] # "[20, 50, 70]"
noises = sys.argv[5] # "[20 ,60, 80]"
benchmark = sys.argv[6] # full

'''

# Create a question split for the dataset

python codes/eval_scripts/make_dev_split.py triviaq 100 -> insert number of questions to split

# Start virtual environment (if on Mac) and install requirements

source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.

# Set up variables for evaluation

eval_model=gpt35_turbo
date=0603
dataset=full
topkk="[20]"
noises="[0]"
benchmark=triviaq
eval_method=3.5turbo

# Possible eval_model values:

gpt35_turbo
llama4
Phi
mistral7b_instruct
local_hf_request

# Run dataset preparation

python codes/datasets/make_datasets.py triviaq "[20]" "[0]" gpt35_turbo

dataset = sys.argv[1]
topkk = sys.argv[2]  # e.g. "[20]"
noises = sys.argv[3] # e.g. "[40]"
eval_model = sys.argv[4]

'''
print("start_to_create_dataset")
#    python codes/datasets/make_datasets.py "$dataset" "$topkk" "$noises" "$eval_model"
subprocess.run(["python", "codes/datasets/make_datasets.py", dataset, topkk, noises, eval_model])

print("start_to_run")
print("start_to_summarize")

'''

# pythoncodes/datasets/baseline_compress.py gpt35_turbo_0613 0602 triviaq "[20]" "[0]" full

eval_model = sys.argv[1] 
date = sys.argv[2]
dataset = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]
'''


# python codes/datasets/baseline_compress.py "$eval_model" "$date" "$dataset" "$topkk" "$noises" "$benchmark"
subprocess.run(["python", "codes/datasets/baseline_compress.py", eval_model, date, dataset, topkk ,noises, benchmark])
print("end_summarize")
print("start_to_eval")

'''
# python codes/datasets/baseline_compress.py gpt35_turbo 0602 triviaq "[20]" "[0]" full

eval_model = sys.argv[1] 
date = sys.argv[2]
dataset = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]
'''

print("start_to_eval")
# python codes/run_methods/eval_baseline_compress.py "$eval_model" "$date" "$dataset" "$topkk" "$noises" "$benchmark"
subprocess.run(["python", "codes/run_methods/eval_baseline_compress.py", eval_model, date, dataset,topkk,noises, benchmark])
print("end_eval")

print("start_to_extract_answer")

'''
python codes/eval_metric/extracted_answer_topkk_compress.py 0604 triviaq gpt35_turbo "[20]" "[0]" full gpt35_turbo
python codes/eval_metric/caculate_F1_EM_compress.py 0604 triviaq gpt35_turbo "[20]" "[0]" full

date = sys.argv[1]
dataset = sys.argv[2]
eval_method = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]
eval_model = sys.argv[7]
'''


# python codes/eval_metric/extracted_answer_topkk_compress.py "$date" "$dataset" "$eval_method" "$topkk" "$noises" "$benchmark" "$eval_model"

subprocess.run(["python", "codes/eval_metric/extracted_answer_topkk_compress.py", date, dataset, eval_model, topkk, noises,benchmark])
print("end_extracte_answer")
print("start_to_caculate_F1_EM")   

# python codes/eval_metric/caculate_F1_EM_compress.py "$date" "$dataset" "$eval_model" "$topkk" "$noises" "$benchmark"

subprocess.run(["python", "codes/eval_metric/caculate_F1_EM_compress.py", date, dataset, eval_model, topkk, noises, benchmark])
print("end_caculate_F1_EM")
