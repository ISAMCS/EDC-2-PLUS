import sys
import subprocess

# Example Run Command: python codes/eval_scripts/run_baseline_compress.py gpt35_turbo 0605 triviaq "[20]" "[0]" full

'''

Insturctions to run the baseline compress evaluation script: 

eval_model = sys.argv[1] #llama4_maverick_request
date = sys.argv[2] # 0602
dataset = sys.argv[3] # triviaq
topkk = sys.argv[4] # "[20, 50, 70]"
noises = sys.argv[5] # "[20 ,60, 80]"
benchmark = sys.argv[6] # full

Example of variables to set:

eval_model=gpt35_turbo
date=0608
dataset=triviaq
topkk="[20]"
noises="[0]"
benchmark=full

'''

# Step 1: (Optional) Start virtual environment (if on Mac) and install requirements

source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.

if you didnt create a virtual environment, you still need to install the requirements

pip install -r requirements.txt

# Step 2: Create dev split for the dataset

python codes/eval_scripts/make_dev_split.py triviaq 25 -> insert number of questions to split

# Step 3: Set up variables for evaluation

eval_model=gpt35_turbo
date=0603
dataset=triviaq
topkk="[20]"
noises="[0]"
benchmark=full

# Possible eval_model values:

gpt35_turbo
llama4
Phi
mistral7b_instruct
local_hf_request


# Step 4: Run dataset preparation

Example: python codes/datasets/make_datasets.py triviaq "[20]" "[0]" gpt35_turbo

dataset = sys.argv[1]
topkk = sys.argv[2]  # e.g. "[20]"
noises = sys.argv[3] # e.g. "[40]"
eval_model = sys.argv[4]

'''
#  Single File Run Command: python codes/datasets/make_datasets.py "$dataset" "$topkk" "$noises" "$eval_model"

print("start_to_create_dataset")
subprocess.run(["python", "codes/datasets/make_datasets.py", dataset, topkk, noises, eval_model])

print("start_to_run")
print("start_to_summarize")

'''

# Step 5: Run summarization

Example: python codes/datasets/baseline_compress.py gpt35_turbo 0608 triviaq "[20]" "[0]" full

eval_model = sys.argv[1] 
date = sys.argv[2]
dataset = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]

'''

# Single File Run Command: python codes/datasets/baseline_compress.py "$eval_model" "$date" "$dataset" "$topkk" "$noises" "$benchmark"

subprocess.run(["python", "codes/datasets/baseline_compress.py", eval_model, date, dataset, topkk ,noises, benchmark])
print("end_summarize")
print("start_to_eval")

'''

# Step 6: Run evaluation

Example: python codes/run_methods/eval_baseline_compress.py gpt35_turbo 0608 triviaq "[20]" "[0]" full

eval_model = sys.argv[1] 
date = sys.argv[2]
dataset = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]

'''

# Single File Run Command: python codes/run_methods/eval_baseline_compress.py "$eval_model" "$date" "$dataset" "$topkk" "$noises" "$benchmark"

print("start_to_eval")
subprocess.run(["python", "codes/run_methods/eval_baseline_compress.py", eval_model, date, dataset,topkk,noises, benchmark])
print("end_eval")

'''

# Step 7: Extract answer

Example: python codes/eval_metric/extracted_answer_topkk_compress.py 0608 triviaq gpt35_turbo "[20]" "[0]" full

date = sys.argv[1]
dataset = sys.argv[2]
eval_method = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]
eval_model = sys.argv[7]

'''

# Single File Run: python codes/eval_metric/extracted_answer_topkk_compress.py "$date" "$dataset" "$eval_model" "$topkk" "$noises" "$benchmark"

print("start_to_extract_answer")
subprocess.run(["python", "codes/eval_metric/extracted_answer_topkk_compress.py", date, dataset, eval_model, topkk, noises,benchmark])
print("end_extracte_answer")  

'''

# Step 8: Calculate F1 and EM metrics

# Example: python codes/eval_metric/caculate_F1_EM_compress.py 0608 triviaq gpt35_turbo "[20]" "[0]" full

date = sys.argv[1]
dataset = sys.argv[2]
eval_model = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]

'''

# Single Run Command: python codes/eval_metric/caculate_F1_EM_compress.py "$date" "$dataset" "$eval_model" "$topkk" "$noises" "$benchmark"

print("start_to_caculate_F1_EM") 
subprocess.run(["python", "codes/eval_metric/caculate_F1_EM_compress.py", date, dataset, eval_model, topkk, noises, benchmark])
print("end_caculate_F1_EM")
