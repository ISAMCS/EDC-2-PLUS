import os
import json
import pandas as pd

# File paths
edc2_path = "triviaq/tables/0608_triviaq_edc2_compress_gpt35_turbo_noise[0]_topk[20].xlsx"
edc2plus_path = "triviaq/tables/0608_triviaq_edc2plus_compress_gpt35_turbo_noise[0]_topk[20].xlsx"

# Read Excel files
edc2_df = pd.read_excel(edc2_path)
edc2plus_df = pd.read_excel(edc2plus_path)

# Extract EM and F1 scores (assumes columns named 'EM' and 'F1')
edc2_em = edc2_df['EM'].iloc[0] if 'EM' in edc2_df.columns else None
edc2_f1 = edc2_df['F1'].iloc[0] if 'F1' in edc2_df.columns else None
edc2plus_em = edc2plus_df['EM'].iloc[0] if 'EM' in edc2plus_df.columns else None
edc2plus_f1 = edc2plus_df['F1'].iloc[0] if 'F1' in edc2plus_df.columns else None

# Prepare comparison JSON
comparison = {
    "edc2": {"EM": edc2_em, "F1": edc2_f1},
    "edc2plus": {"EM": edc2plus_em, "F1": edc2plus_f1}
}

# Write to triviaq/analysis/edc2_vs_edc2plus_scores.json
output_dir = "triviaq/analysis"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "edc2_vs_edc2plus_scores.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(comparison, f, ensure_ascii=False, indent=2)
print(f"Saved comparison scores to {output_path}")

import sys
import subprocess
import json

# Usage: python codes/run_compare_edc2_vs_edc2plus.py <date> <dataset> <eval_model> <topkk> <noises> <benchmark>
# Example: python codes/run_compare_edc2_vs_edc2plus.py 0608 triviaq gpt35_turbo "[20]" "[0]" full

def run_pipeline(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result

def main():
    if len(sys.argv) < 7:
        print("Usage: python codes/run_compare_edc2_vs_edc2plus.py <date> <dataset> <eval_model> <topkk> <noises> <benchmark>")
        sys.exit(1)
    date, dataset, eval_model, topkk, noises, benchmark = sys.argv[1:7]

    # Run EDC2 pipeline (OG)
    run_pipeline(["python", "Codespace/Scripts/EDC2/run_edc2_compress.py", eval_model, date, dataset, topkk, noises, benchmark])
    # Run EDC2+ pipeline (PLUS)
    run_pipeline(["python", "Codespace/Scripts/EDC2Plus/run_edc2plus_compress.py", eval_model, date, dataset, topkk, noises, benchmark])
    # The plus pipeline internally calls its own baseline, eval, extract, and metric scripts

    print("\nComparison complete. Check output above for EM and F1 scores from both pipelines.")

if __name__ == "__main__":
    main()
