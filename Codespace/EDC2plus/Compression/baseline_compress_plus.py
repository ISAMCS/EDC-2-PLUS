from Codespace.EDC2plus.Core_Modules.query_guard import check_query_stability
from Codespace.EDC2plus.Core_Modules.query_guard import GuardedRetriever
from Codespace.EDC2plus.Core_Modules.temporal_router import TemporalRouter
from Codespace.EDC2plus.Core_Modules.temporal_router import classify_query_temporal
from Codespace.LLMs import utils
from Codespace.EDC2plus.Core_Modules.grounding import sentence_level_ground
from Codespace.EDC2plus.Core_Modules.retrieval_evaluator import RetrievalEvaluator

def get_llm_response(prompt):
	# Use the same LLM call as in EDC2 OG, via utils
	# Example: utils.llama4_maverick_request, utils.gpt35_turbo_0613_request, etc.
	# You may want to select the model based on eval_model
	if eval_model == "llama4":
		return utils.llama4_maverick_request(prompt)
	elif eval_model == "Phi":
		return utils.Microsoft_Phi4_request(prompt)
	elif eval_model == "mistral7b_instruct":
		return utils.mistral7b_instruct_request(prompt)
	elif eval_model == "gpt35_turbo":
		return utils.gpt35_turbo_0613_request(prompt)
	elif eval_model == "local_hf":
		return utils.local_hf_request(prompt)
	else:
		raise ValueError(f"Unknown model: {eval_model}")

import sys
import json
import ast
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
from transformers import AutoModel, AutoTokenizer

np.random.seed(42)

eval_model = sys.argv[1]
date = sys.argv[2]
dataset = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]
toggle = sys.argv[7] if len(sys.argv) > 7 else None

# Embedding model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_path = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(embed_path)
embed_model = AutoModel.from_pretrained(embed_path, torch_dtype=torch.float16).to(device)

def get_embedding(text):
	with torch.no_grad():
		inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
		feature = embed_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
	return feature.squeeze().cpu().numpy()

def rerank_passages(passages, question):
	# Use cosine similarity between passage and question embeddings for reranking
	q_emb = get_embedding(question)
	scored = []
	for passage in passages:
		p_emb = get_embedding(passage['text'])
		score = np.dot(q_emb, p_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(p_emb) + 1e-8)
		scored.append((score, passage))
	scored.sort(reverse=True, key=lambda x: x[0])
	return [p for _, p in scored]

def llm_compress_cluster(cluster, question, llm_prompt_func):
	# Use LLM to compress cluster into short, sourced bullets
	docs = "\n\n".join([f"{i+1}. {doc['text']}" for i, doc in enumerate(cluster)])
	prompt = f"""
	# Instruction:
	You are given a question and a cluster of documents. Compress the cluster into short, sourced bullets that quote spans and carry explicit citations (source: path#chunkN).
	# Question:
	{question}
	# Documents:
	{docs}
	# Output:
	"""
	return get_llm_response(prompt)

def process_case(case, llm_prompt_func, toggle):
	# Ablation logic: only run relevant module for each toggle
	if toggle == "stability":
		case['stability'] = check_query_stability(case['question'])
	elif toggle == "temporal":
		case['temporal_class'] = classify_query_temporal(case['question'])
	elif toggle == "conflict":
		case['claims'] = sentence_level_ground(case['question'], case['passages'], eval_model=eval_model)
	elif toggle == "calibration":
		case['confidence'] = RetrievalEvaluator(case['question'], case['passages'])
	elif toggle == "all":
		case['stability'] = check_query_stability(case['question'])
		case['temporal_class'] = classify_query_temporal(case['question'])
		case['claims'] = sentence_level_ground(case['question'], case['passages'], eval_model=eval_model)
		case['confidence'] = RetrievalEvaluator(case['question'], case['passages'])
	# Rerank passages
	reranked = rerank_passages(case['passages'], case['question'])
	# Clustering
	clusters = []
	used = set()
	for i, passage in enumerate(reranked):
		if i in used:
			continue
		cluster = [passage]
		p_emb = get_embedding(passage['text'])
		for j, other in enumerate(reranked):
			if j != i and j not in used:
				o_emb = get_embedding(other['text'])
				sim = np.dot(p_emb, o_emb) / (np.linalg.norm(p_emb) * np.linalg.norm(o_emb) + 1e-8)
				if sim > 0.85:
					cluster.append(other)
					used.add(j)
		used.add(i)
		clusters.append(cluster)
	# Compress clusters
	compressed_blocks = [llm_compress_cluster(cluster, case['question'], get_llm_response) for cluster in clusters]
	case['compressed_blocks'] = compressed_blocks
	return case

def run(topk, noise):
	global eval_model, date, dataset, benchmark, toggle
	# Only run with 'full' benchmark and each toggle
	case_file = f"{dataset}/datasets/{dataset}_results_random_full_w_negative_passages_noise{noise}_topk{topk}_embedding.json"
	with open(case_file, "r", encoding="utf-8") as lines:
		cases = json.load(lines)
		final_result = []
		# Choose LLM prompt function (stub)
		def llm_prompt_func(prompt):
			return get_llm_response(prompt)
		for case in tqdm(cases):
			result = process_case(case, llm_prompt_func, toggle)
			final_result.append(result)
		# Save results with toggle in filename
		res_file = f"{dataset}/datasets/case_{date}_{dataset}_summary_edc2plus_compress_{eval_model}_noise{noise}_topk{topk}_full"
		if toggle:
			res_file += f"_{toggle}"
		res_file += ".json"
		with open(res_file, "w", encoding="utf-8") as json_file:
			json.dump(final_result, json_file, ensure_ascii=False, indent=4)

for topk in topkk:
	for noise in noises:
		run(topk, noise)
		print(f"Finished running for topk={topk} and noise={noise} In EDC2+ Summarize Docs")
