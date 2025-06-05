import json
from sklearn.metrics import precision_score, recall_score
import sys
from nltk.tokenize import word_tokenize

date = sys.argv[1]
dataset = sys.argv[2]
eval_method = sys.argv[3]
# 读取json文件
benchmark = sys.argv[4]

def normalize_text(text):
    text = text.lower()
    return ' '.join(text.split())

def compute_f1(pred, true):
    pred_tokens = word_tokenize(pred)
    true_tokens = word_tokenize(true)
    common = set(pred_tokens) & set(true_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(true_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def compute_metrics(dataset):
    em_total = 0
    f1_total = 0
    for data in dataset:
        pred = data["extracted_answer"].split(":")[-1].strip()
        answers = data["answers"]
        pred = normalize_text(pred)
        answers = [normalize_text(ans) for ans in answers]
        # 计算完全匹配
        em = int(any(pred == ans for ans in answers))
        em_total += em

        # 计算F1指数
        f1 = max(compute_f1(pred, ans) for ans in answers)
        f1_total += f1

    em_score = em_total / len(dataset)
    f1_score = f1_total / len(dataset)
    return round(em_score*100,2), round(f1_score*100,2)
    
for topk in topkk:
    for noise in noises:
        input_file = f"{dataset}/extracted_answer/{date}_{benchmark}_compress_{eval_model}_noise{noise}_topk{topk}.json"
        with open(input_file, "r", encoding="utf-8") as f:
            datasets = json.load(f)
        em_score, f1_score = compute_metrics(datasets)
        results.append([topk, noise, em_score, f1_score])
        print(f"{input_file}: EM: {em_score}, F1: {f1_score}")
df = pd.DataFrame(results, columns=["TopK", "Noise", "EM Score", "F1 Score"]).T
output_file = f"{dataset}/tables/{date}_{benchmark}_compress_{eval_model}_noise{noises}_topk{topkk}.xlsx"
df.to_excel(output_file, index=False)
print(f"Results saved to {output_file}")