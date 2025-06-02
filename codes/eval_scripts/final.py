import subprocess

#  python codes/eval_scripts/final.py

DATE = "0602"
BENCHMARK = "triviaq"
SUMMARISER = "gpt35_turbo_0613_request"
TOPKK = "[20]"
NOISES = "[20]"
#NOISES = "[0,20,40,60,80,100]"
EVAL_METHOD = "eval_3.5turbo"



# 1. Make base dataset
subprocess.run(["python", "codes/datasets/make_datasets.py", BENCHMARK], check=True)

# 2. Baseline Compression
subprocess.run([
    "python", "codes/datasets/baseline_compress.py",
    SUMMARISER, DATE, "full", TOPKK, NOISES, BENCHMARK
], check=True)

# 3. Evaluate Baseline Compression
subprocess.run([
    "python", "codes/run_methods/eval_baseline_compress.py",
    SUMMARISER, DATE, "full", TOPKK, NOISES, BENCHMARK
], check=True)

# 4. Extract Answers
subprocess.run([
    "python", "codes/eval_metric/extracted_answer_topkk_compress.py",
    DATE, "full", EVAL_METHOD, TOPKK, NOISES, BENCHMARK
], check=True)

# 5. Calculate F1/EM
subprocess.run([
    "python", "codes/eval_metric/caculate_F1_EM_compress.py",
    DATE, "full", EVAL_METHOD, TOPKK, NOISES, BENCHMARK
], check=True)