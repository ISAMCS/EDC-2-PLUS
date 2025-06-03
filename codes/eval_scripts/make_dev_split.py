import os
import json
import sys

DATASET = sys.argv[1]

def main():
    # Set how many questions you want to keep
    X = 10  # Change this value as needed

    # Locate repo root (two levels up)
    script_dir = os.path.dirname(__file__)
    repo_root  = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # Use the specified input file
    input_path = os.path.join(repo_root, DATASET, "OG", f"{DATASET}_results_w_negative_passages_full.json")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    count = 0
    for entry in data:
        if count >= X:
            break
        answers = entry.get("answers", [])
        answers = list({a.strip() for a in answers if a and a.strip()})

        # Format positive_passages
        pos_passages = []
        for p in entry.get("positive_passages", []):
            pos_passages.append({
                "id": p.get("id", ""),
                "title": p.get("title", ""),
                "text": p.get("text", ""),
                "score": p.get("score", ""),
                "has_answer": True
            })
        # Format negative_passages
        neg_passages = []
        for p in entry.get("negative_passages", []):
            neg_passages.append({
                "id": p.get("id", ""),
                "title": p.get("title", ""),
                "text": p.get("text", ""),
                "score": p.get("score", ""),
                "has_answer": False
            })

        examples.append({
            "question": entry.get("question", ""),
            "answers": answers,
            "positive_passages": pos_passages,
            "negative_passages": neg_passages
        })
        count += 1

    output_path = os.path.join(repo_root, DATASET, "datasets", f"{DATASET}_results_w_negative_passages_full.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"Wrote {output_path} ({len(examples)} examples)")

if __name__ == "__main__":
    main()