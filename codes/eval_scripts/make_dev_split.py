import os
import json
from datasets import load_dataset

def main():
    # Locate repo root (two levels up)
    script_dir = os.path.dirname(__file__)
    repo_root  = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # Build TriviaQA dev.json (single file)
    ds = load_dataset("trivia_qa", "rc")["validation"].select(range(10))

    examples = []
    for entry in ds:
        # Extract all possible gold answers
        answer_obj = entry["answer"]
        answers = []
        # Try to collect all possible aliases and values
        if "aliases" in answer_obj and answer_obj["aliases"]:
            answers.extend(answer_obj["aliases"])
        if "normalized_aliases" in answer_obj and answer_obj["normalized_aliases"]:
            answers.extend(answer_obj["normalized_aliases"])
        if "value" in answer_obj and answer_obj["value"]:
            answers.append(answer_obj["value"])
        # Remove duplicates and empty strings
        answers = list({a.strip() for a in answers if a and a.strip()})

        examples.append({
            "question": entry["question"],
            "answers": answers,
            "extracted_answer": "",  # Placeholder for model prediction
            "positive_passages": entry.get("positive_passages", []),
            "negative_passages": entry.get("negative_passages", [])
        })

    triviaq_dir = os.path.join(repo_root, "triviaq", "datasets")
    os.makedirs(triviaq_dir, exist_ok=True)

    triviaq_dev_path = os.path.join(triviaq_dir, "dev.json")
    with open(triviaq_dev_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"Wrote {triviaq_dev_path} ({len(examples)} examples)")

if __name__ == "__main__":
    main()