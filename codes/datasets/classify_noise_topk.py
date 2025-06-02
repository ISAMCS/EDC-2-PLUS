'''
python codes/datasets/classify_noise_topk.py triviaq
'''
import json
import copy
import sys
import ast

dataset = sys.argv[1]
topk_list = ast.literal_eval(sys.argv[2])  # e.g. "[20]"
noise_list = ast.literal_eval(sys.argv[3]) # e.g. "[0,20,40,60,80,100]"

input_file = f"{dataset}/datasets/{dataset}_results_w_negative_passages_full_embedding.json"

with open(input_file, "r", encoding="utf-8") as lines:
    cases = json.load(lines)

for topk in topk_list:
    for noise in noise_list:
        outs = []
        output_file = f"{dataset}/datasets/{dataset}_results_random_full_w_negative_passages_noise{noise}_topk{topk}_embedding.json"
        for case in cases:
            out = copy.deepcopy(case)
            out["passages"] = []
            n = topk * noise // 100
            p = topk - n
            ii = 0
            for passage in out.get("positive_passages", []):
                if ii < p:
                    out["passages"].append(passage)
                    ii += 1
                else:
                    break
            ii = 0
            for passage in out.get("negative_passages", []):
                if ii < n:
                    out["passages"].append(passage)
                    ii += 1
                else:
                    break
            if "negative_passages" in out:
                del out["negative_passages"]
            if "positive_passages" in out:
                del out["positive_passages"]
            out["passages"] = sorted(out["passages"], key=lambda x: x["score"], reverse=True)
            outs.append(out)
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(outs, json_file, ensure_ascii=False, indent=4)