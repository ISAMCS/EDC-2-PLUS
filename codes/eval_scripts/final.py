import subprocess

#  python codes/eval_scripts/final.py

DATE="0602"
BENCHMARK="triviaq"
SUMMARISER="gpt35_turbo_0613_request"
TOPKK="[20]"
NOISES="[0,20,40,60,80,100]"
EVAL_METHOD="eval_3.5turbo"
DATASET="full"

#"[20]", "[0,20,40,60,80,100]"

# python codes/datasets/get_embedding.py triviaq
# 1. Make base dataset
subprocess.run(["python", "codes/datasets/get_embedding.py", BENCHMARK], check=True)
subprocess.run([
    "python", "codes/datasets/classify_noise_topk.py",
    BENCHMARK, TOPKK, NOISES
], check=True)


# 2. Baseline Compression
subprocess.run([
    "python", "codes/datasets/baseline_compress.py",
    SUMMARISER, DATE, DATASET, TOPKK, NOISES, BENCHMARK
], check=True)

# 3. Evaluate Baseline Compression
subprocess.run([
    "python", "codes/run_methods/eval_baseline_compress.py",
    SUMMARISER, DATE, DATASET, TOPKK, NOISES, BENCHMARK
], check=True)

# 4. Extract Answers
subprocess.run([
    "python", "codes/eval_metric/extracted_answer_topkk_compress.py",
    DATE, DATASET, EVAL_METHOD, TOPKK, NOISES, BENCHMARK
], check=True)

# 5. Calculate F1/EM
subprocess.run([
    "python", "codes/eval_metric/caculate_F1_EM_compress.py",
    DATE, DATASET, EVAL_METHOD, TOPKK, NOISES, BENCHMARK
], check=True)

'''

EXMPLE USAGE:

source .venv/bin/activate
pip install -r requirements.txt

export PYTHONPATH=.

# 1. Make base dataset
python codes/datasets/get_embedding.py triviaq
python codes/datasets/classify_noise_topk.py triviaq "[20]" "[40]"
python codes/datasets/make_datasets.py triviaq

# 2. Baseline Compression
python codes/datasets/baseline_compress.py llama4_request 0601 full "[20]" "[40]" triviaq
python codes/datasets/baseline_compress.py Phi 0601 full "[20]" "[40]" triviaq
python codes/datasets/baseline_compress.py mistral7b_instruct_request 0601 full "[20]" "[40]" triviaq
python codes/datasets/baseline_compress.py gpt35_turbo_0613_request 0601 full "[20]" "[40]" triviaq


# 3. Evaluate Baseline Compression
python codes/run_methods/eval_baseline_compress.py llama4_request 0601 full "[20]" "[40]" triviaq
python codes/run_methods/eval_baseline_compress.py Phi 0601 full "[20]" "[40]" triviaq
python codes/run_methods/eval_baseline_compress.py mistral7b_instruct_request 0601 full "[20]" "[40]" triviaq
python codes/run_methods/eval_baseline_compress.py gpt35_turbo_0613_request 0601 full "[20]" "[40]" triviaq


# 4. Extract Answers
python codes/eval_metric/extracted_answer_topkk_compress.py 0601 full eval_llama4 "[20]" "[40]" triviaq
python codes/eval_metric/extracted_answer_topkk_compress.py 0601 full eval_mistral7b "[20]" "[40]" triviaq
python codes/eval_metric/extracted_answer_topkk_compress.py 0601 full eval_3.5turbo "[20]" "[40]" triviaq

# 5. Calculate F1/EM
python codes/eval_metric/caculate_F1_EM_compress.py 0601 full eval_llama4 "[20]" "[40]" triviaq
python codes/eval_metric/caculate_F1_EM_compress.py 0601 full eval_mistral7b "[20]" "[40]" triviaq
python codes/eval_metric/caculate_F1_EM_compress.py 0601 full eval_3.5turbo "[20]" "[40]" triviaq

python codes/eval_scripts/final.py

'''