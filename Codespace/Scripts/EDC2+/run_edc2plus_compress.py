
import sys
import subprocess
from Codespace.EDC2plus import query_guard

# Usage: python codes/edc2plus/run_edc2plus_compress.py <eval_model> <date> <dataset> <topkk> <noises> <benchmark>
# python codes/edc2plus/run_edc2plus_compress.py gpt35_turbo 0608 triviaq "[20]" "[0]" full

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: python codes/edc2plus/run_edc2plus_compress.py <eval_model> <date> <dataset> <topkk> <noises> <benchmark>")
        sys.exit(1)
    eval_model, date, dataset, topkk, noises, benchmark = sys.argv[1:7]

    print("start_to_run")
    print("start_to_summarize")
    # python Codespace/EDC2plus/Compression/baseline_compress_plus.py gpt35_turbo 0608 triviaq "[20]" "[0]" full
    subprocess.run(["python", "Codespace/EDC2plus/Compression/baseline_compress_plus.py", eval_model, date, dataset, topkk, noises, benchmark])
    print("end_summarize")
    print("start_to_eval")
    # python Codespace/EDC2plus/Evaluation/eval_baseline_compress_plus.py gpt35_turbo 0608 triviaq "[20]" "[0]" full
    subprocess.run(["python", "Codespace/EDC2plus/Evaluation/eval_baseline_compress_plus.py", eval_model, date, dataset, topkk, noises, benchmark])
    print("end_eval")
    print("start_to_extract_answer")
    # python Codespace/EDC2plus/Extraction/extracted_answer_topkk_compress.py 0608 triviaq gpt35_turbo "[20]" "[0]" full
    subprocess.run(["python", "Codespace/EDC2plus/Extraction/extracted_answer_topkk_compress.py", date, dataset, eval_model, topkk, noises, benchmark])
    print("end_extracte_answer")
    print("start_to_caculate_F1_EM")
    # python Codespace/EDC2plus/Calculation/caculate_F1_EM_compress_plus.py 0608 triviaq gpt35_turbo "[20]" "[0]" full
    subprocess.run(["python", "Codespace/EDC2plus/Calculation/caculate_F1_EM_compress_plus.py", date, dataset, eval_model, topkk, noises, benchmark])
    print("end_caculate_F1_EM")

#