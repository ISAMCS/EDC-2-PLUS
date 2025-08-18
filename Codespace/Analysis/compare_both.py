import os
import json
import pandas as pd

# File paths
edc2_path = "triviaq/tables/0608_triviaq_edc2_compress_gpt35_turbo_noise[0]_topk[20].xlsx"
edc2plus_path = "triviaq/tables/0608_triviaq_edc2plus_compress_gpt35_turbo_noise[0]_topk[20].xlsx"

# Read Excel files
edc2_df = pd.read_excel(edc2_path, header=None)
edc2plus_df = pd.read_excel(edc2plus_path, header=None)

# Extract EM and F1 scores from row 4 (index 3) and row 5 (index 4), column 1 (index 0)
edc2_em = int(edc2_df.iloc[3, 0])
edc2_f1 = int(edc2_df.iloc[4, 0])
edc2plus_em = int(edc2plus_df.iloc[3, 0])
edc2plus_f1 = int(edc2plus_df.iloc[4, 0])

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