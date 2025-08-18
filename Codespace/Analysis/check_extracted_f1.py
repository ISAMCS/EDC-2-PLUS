import json
import sys
from collections import Counter
import os

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    import re
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    import string
    def remove_punc(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return int(prediction_tokens == ground_truth_tokens)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def main(json_path, threshold=0.5):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    unmatched_questions = []
    for idx, entry in enumerate(data):
        extracted = entry.get('extracted_answer', '')
        answers = entry.get('answers', [])
        ems = [int(normalize_answer(extracted) == normalize_answer(ans)) for ans in answers] if answers else [0]
        best_em = max(ems)
        if best_em == 0:
            unmatched_questions.append({
                "source": sys.argv[1],
                "index": idx,
                "question": entry.get('question', ''),
                "extracted_answer": extracted,
                "answers": answers
            })
    debug_dir = "triviaq/Debug"
    os.makedirs(debug_dir, exist_ok=True)
    base_name = os.path.basename(sys.argv[1])
    out_path = os.path.join(debug_dir, f"{base_name}_unmatched_questions.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(unmatched_questions, f, ensure_ascii=False, indent=2)
    print(f"Unmatched questions saved to {out_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python check_extracted_f1.py <path_to_json> [threshold]')
        sys.exit(1)
    json_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    main(json_path, threshold)
