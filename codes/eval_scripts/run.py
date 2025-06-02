import subprocess
import sys

# USAGE (in bash):
# python experiment.py 0601 triviaq llama3_request adaptive

if len(sys.argv) < 5:
    print("Usage: python experiment.py <DATE> <BENCHMARK> <SUMMARISER> <COMPRESS>")
    sys.exit(1)

DATE = sys.argv[1]         # e.g., "0601"
BENCHMARK = sys.argv[2]    # e.g., "triviaq"
SUMMARISER = sys.argv[3]   # e.g., "llama3_request"
COMPRESS = sys.argv[4]     # e.g., "adaptive"

TOPKK = "[20]"
NOISES = "[0,20,40,60,80,100]"

'''

DATE=0601
BENCHMARK=triviaq
SUMMARISER=llama3_request
COMPRESS=adaptive

TOPKK="[20]"
NOISES="[0,20,40,60,80,100]"

'''

# 1. Make base dataset
subprocess.run(["python", "codes/datasets/make_datasets.py", BENCHMARK], check=True)

# 2. No retrieval baseline
subprocess.run([
    "python", "codes/eval_scripts/run_baseline_wo_retrieve.py",
    SUMMARISER, DATE, "full", BENCHMARK
], check=True)

# 3. Baseline RAG
subprocess.run([
    "python", "codes/eval_scripts/run_baseline_rag.py",
    SUMMARISER, DATE, "full", TOPKK, NOISES, BENCHMARK
], check=True)

# 4. Baseline Compression
subprocess.run([
    "python", "codes/eval_scripts/run_baseline_compress.py",
    SUMMARISER, DATE, "full", TOPKK, NOISES, BENCHMARK
], check=True)

# 5. EDC²-RAG (Dynamic Clustering + Compression)
subprocess.run([
    "python", "codes/eval_scripts/run_baseline_compress.py",
    DATE, "full", SUMMARISER, TOPKK, NOISES, "1110", "dynamic", BENCHMARK
], check=True)

# Optionally, add metric scripts here if needed
# subprocess.run([...])

'''
# 1. Make base dataset
python codes/datasets/make_datasets.py $BENCHMARK

# 2. No retrieval baseline
python codes/eval_scripts/run_baseline_wo_retrieve.py $SUMMARISER $DATE full $BENCHMARK

# 3. Baseline RAG
python codes/eval_scripts/run_baseline_rag.py $SUMMARISER $DATE full "[20]" "[0,20,40,60,80,100]" $BENCHMARK
#codes/run_methods/eval_baseline_rag.py

# 4. Baseline Compression
python codes/eval_scripts/run_baseline_compress.py $SUMMARISER $DATE full "[20]" "[0,20,40,60,80,100]" $BENCHMARK

# 5. EDC²-RAG (Dynamic Clustering + Compression)
python codes/eval_scripts/run_baseline_compress.py $DATE full $SUMMARISER "[20]" "[0,20,40,60,80,100]" 1110 dynamic $BENCHMARK
'''