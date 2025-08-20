import sys
import subprocess

eval_model = sys.argv[1]
date = sys.argv[2]
dataset = sys.argv[3]
topkk = sys.argv[4]
noises = sys.argv[5]
benchmark = sys.argv[6]


print("start_to_run")
print("start_to_summarize")

'''

# Step 1: Run Compression

Example: python Codespace/EDC2/Compression/baseline_compress.py gpt35_turbo 0608 triviaq "[20]" "[0]" full

Document Args:

eval_model = sys.argv[1] 
date = sys.argv[2]
dataset = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]

'''

# Single File Run Command: python Codespace/EDC2/Compression/baseline_compress.py "$eval_model" "$date" "$dataset" "$topkk" "$noises" "$benchmark"

#subprocess.run(["python", "Codespace/EDC2/Compression/baseline_compress.py", eval_model, date, dataset, topkk ,noises, benchmark])
print("end_summarize")
print("start_to_eval")

'''

# Step 2: Run evaluation

Example: python Codespace/EDC2/Evaluation/eval_baseline_compress.py gpt35_turbo 0608 triviaq "[20]" "[0]" full

Document Args:

eval_model = sys.argv[1] 
date = sys.argv[2]
dataset = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]

'''

# Single File Run Command: python Codespace/EDC2/Evaluation/eval_baseline_compress.py "$eval_model" "$date" "$dataset" "$topkk" "$noises" "$benchmark"

print("start_to_eval")
subprocess.run(["python", "Codespace/EDC2/Evaluation/eval_baseline_compress.py", eval_model, date, dataset,topkk,noises, benchmark])
print("end_eval")

'''

# Step 3: Extract answer

Example: python Codespace/EDC2/Extraction/extracted_answer_topkk_compress.py 0608 triviaq gpt35_turbo "[20]" "[0]" full

Document Args:

date = sys.argv[1]
dataset = sys.argv[2]
eval_method = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]
eval_model = sys.argv[7]

'''

# Single File Run: python Codespace/EDC2/Extraction/extracted_answer_topkk_compress.py "$date" "$dataset" "$eval_model" "$topkk" "$noises" "$benchmark"

print("start_to_extract_answer")
subprocess.run(["python", "Codespace/EDC2/Extraction/extracted_answer_topkk_compress.py", date, dataset, eval_model, topkk, noises,benchmark])
print("end_extracte_answer")

'''

# Step 4: Calculate F1 and EM metrics

# Example: python Codespace/EDC2/Calculation/caculate_F1_EM_compress.py 0608 triviaq gpt35_turbo "[20]" "[0]" full

Document Args

date = sys.argv[1]
dataset = sys.argv[2]
eval_model = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]

'''

# Single Run Command: python Codespace/EDC2/Evaluation/caculate_F1_EM_compress.py "$date" "$dataset" "$eval_model" "$topkk" "$noises" "$benchmark"

print("start_to_caculate_F1_EM") 
#subprocess.run(["python", "Codespace/EDC2/Evaluation/caculate_F1_EM_compress.py", date, dataset, eval_model, topkk, noises, benchmark])
print("end_caculate_F1_EM")
