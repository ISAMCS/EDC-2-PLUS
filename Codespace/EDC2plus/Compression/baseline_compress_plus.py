from Codespace.EDC2plus.Core_Modules.query_guard import check_query_stability
from Codespace.EDC2plus.Core_Modules.query_guard import GuardedRetriever
from Codespace.EDC2plus.Core_Modules.temporal_router import TemporalRouter
from Codespace.EDC2plus.Core_Modules.temporal_router import classify_query_temporal
from Codespace.LLMs import utils
from Codespace.EDC2plus.Core_Modules.grounding import sentence_level_ground
from Codespace.EDC2plus.Core_Modules.answer_postprocess import postprocess_answers
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
from Codespace.LLMs.utils import get_embedding
import re

np.random.seed(42)

eval_model = sys.argv[1]
date = sys.argv[2]
dataset = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]
toggle = sys.argv[7] if len(sys.argv) > 7 else None



def rerank_passages(passages, question):
	# Use cosine similarity between passage and question embeddings for reranking
	# Improved reranker prompt to reduce numeric bias and prefer entity/country names for person/place questions
	passage_list = "\n".join([p['text'] for p in passages])
	prompt = f"""
You are given a question and a list of passages. Each passage is pre-ranked by embedding similarity.  
Your task: reorder them by how directly they support the best possible answer.

Question:
"{question}"

Passages:
{passage_list}

Guidelines:
1. Identify exactly what the question is asking (entity, number, date, etc.).  
2. Prefer passages that contain the exact entity or country name when the question asks for a person/place; prefer numbers only when the question asks for a number/year.
3. Deprioritize passages that are irrelevant or only loosely related.  
"""
	# You may want to call your LLM here for reranking, but for now, fallback to embedding similarity
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
	response = get_llm_response(prompt)
	if not response or not response.strip():
		# Fallback: use the most relevant passage from the cluster
		response = cluster[0]['text'] if cluster and 'text' in cluster[0] else "[No summary available]"
	return response

def process_case(case, llm_prompt_func, toggle):
	# Guarded retrieval and metadata
	retriever = GuardedRetriever(passages=case['passages'])
	stability_info = retriever.stability(case['question'], k=10)
	temporal_info = TemporalRouter()(case['question'])
	# Use LLM reranker for passage ordering
	ranked = rerank_passages(case['passages'], case['question'])
	# --- Time-aware filtering: filter out old chunks for time-sensitive queries ---
	if temporal_info['policy'] == 'hybrid' and temporal_info['freshness_filter_days']:
		now = time.time()
		filtered = []
		for p in ranked:
			dt = p.get('date')
			if dt:
				try:
					if isinstance(dt, str):
						dt_epoch = time.mktime(time.strptime(dt[:10], "%Y-%m-%d"))
					else:
						dt_epoch = float(dt)
					if now - dt_epoch < temporal_info['freshness_filter_days'] * 86400:
						filtered.append(p)
				except Exception:
					pass
		if filtered:
			ranked = filtered
	# --- Less strict confidence gating ---
	evaluator = RetrievalEvaluator(model_name="sentence-transformers/all-MiniLM-L6-v2", thresholds={"high": 0.45, "med": 0.15})
	confidence_label, confidence_score = evaluator.evaluate(case['question'], [p['text'] for p in ranked])
	# Attach guard metadata to passages
	for p in ranked:
		p['stability'] = stability_info.stability_score
		p['temporal_class'] = temporal_info['policy']
		p['confidence'] = confidence_score
	# Ablation logic: add metadata fields to case
	if toggle == "stability":
		case['stability'] = stability_info.stability_score
	if toggle == "temporal":
		case['temporal_class'] = temporal_info['policy']
	if toggle == "conflict":
		case['claims'] = sentence_level_ground(case['question'], ranked, eval_model=eval_model)
	if toggle == "calibration":
		case['confidence'] = confidence_score
	if toggle == "all":
		case['stability'] = stability_info.stability_score
		case['temporal_class'] = temporal_info['policy']
		case['claims'] = sentence_level_ground(case['question'], ranked, eval_model=eval_model)
	case['confidence'] = confidence_score
	# Clustering
	clusters = []
	used = set()
	for i, passage in enumerate(ranked):
		if i in used:
			continue
		cluster = [passage]
		p_emb = get_embedding(passage['text'])
		for j, other in enumerate(ranked):
			if j != i and j not in used:
				o_emb = get_embedding(other['text'])
				sim = np.dot(p_emb, o_emb) / (np.linalg.norm(p_emb) * np.linalg.norm(o_emb) + 1e-8)
				if sim > 0.85:
					cluster.append(other)
					used.add(j)
		used.add(i)
		clusters.append(cluster)
	# Compress clusters, guarantee every block is filled
	compressed_blocks = []
	for cluster in clusters:
		block = llm_compress_cluster(cluster, case['question'], get_llm_response)
		if not block or not block.strip():
			# Fallback: use the most relevant passage from the cluster
			block = cluster[0]['text'] if cluster and 'text' in cluster[0] else "[No summary available]"
		compressed_blocks.append(block)
	# --- New: Confidence head-based answer selection ---
	from Codespace.EDC2plus.Core_Modules.answer_postprocess import infer_answer_type, canonicalize_answer
	from Codespace.EDC2plus.Core_Modules.confidence_head import ConfidenceHead
	ans_type = infer_answer_type(case['question'])
	def strip_provenance(text):
		text = re.sub(r"\(source:[^)]+\)", "", text)
		text = re.sub(r"\b\d+#chunk[^\s)]+", "", text)
		return text
	# Candidate set: baseline span, LLM span, regex shortlist
	candidates = []
	# Baseline EDC2+ span (if available)
	if 'edc2plus_span' in case:
		candidates.append(case['edc2plus_span'])
	# LLM span from claim→cite prompt (stub: use first compressed block)
	if compressed_blocks:
		candidates.append(strip_provenance(compressed_blocks[0]))
	# Type-aware regex shortlist
	for block in compressed_blocks:
		block_clean = strip_provenance(block)
		if ans_type == "year":
			candidates += re.findall(r"\b(?:19|20)\d{2}\b", block_clean)
		elif ans_type == "number":
			candidates += [m for m in re.findall(r"\b\d+\b", block_clean) if not re.match(r"\d{4}", m)]
		elif ans_type == "person":
			# Prefer multi-token entities
			candidates += re.findall(r"\b[A-Z][a-zA-Z0-9\-']+(?: [A-Z][a-zA-Z0-9\-']+)+\b", block_clean)
			# Fallback to single-token
			candidates += re.findall(r"\b[A-Z][a-zA-Z0-9\-']+\b", block_clean)
		elif ans_type == "place":
			# Prefer multi-token entities
			candidates += re.findall(r"\b[A-Z][a-zA-Z0-9\-']+(?: [A-Z][a-zA-Z0-9\-']+)+\b", block_clean)
			# Fallback to single-token
			candidates += re.findall(r"\b[A-Z][a-zA-Z0-9\-']+\b", block_clean)
	# Deduplicate and canonicalize
	candidates = list(dict.fromkeys([canonicalize_answer(a.strip(), ans_type) for a in candidates if a and a.strip()]))
	# Prepare features for confidence head
	confidence_head = ConfidenceHead()
	# Semantic consensus: sample up to 16 candidates
	sample_answers = candidates[:16] if len(candidates) > 1 else candidates
	# Faithfulness: use claims/quotes if available (stub: use all blocks)
	quotes = [strip_provenance(b) for b in compressed_blocks]
	# Retrieval sufficiency: use confidence_score and coverage
	coverage = evaluator.coverage_score(case['question'], [p['text'] for p in ranked])
	retrieval_score = confidence_score
	# Score each candidate with confidence, semantic similarity, and block subject match
	from sentence_transformers import SentenceTransformer
	embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
	question_emb = embedder.encode([case['question']])[0]
	best_candidate = None
	best_score = -float('inf')
	for block in compressed_blocks:
		block_clean = strip_provenance(block)
		if not block_clean.strip():
			continue  # Skip empty blocks
		block_entities = re.findall(r"\b[A-Z][a-zA-Z0-9\-']+(?: [A-Z][a-zA-Z0-9\-']+)*\b", block_clean)
		block_subject = block_entities[0] if block_entities else (block_clean.split()[0] if block_clean.split() else "")
		if not block_subject:
			continue  # Skip if no subject found
		block_emb = embedder.encode([block_subject])[0]
		for candidate in candidates:
			features = confidence_head.extract_features(candidate, sample_answers, quotes, retrieval_score, coverage)
			prob = confidence_head.predict_proba(features)
			cand_emb = embedder.encode([candidate])[0]
			sem_sim_q = float(np.dot(question_emb, cand_emb) / (np.linalg.norm(question_emb) * np.linalg.norm(cand_emb) + 1e-8))
			sem_sim_block = float(np.dot(block_emb, cand_emb) / (np.linalg.norm(block_emb) * np.linalg.norm(cand_emb) + 1e-8))
			score = 0.6 * prob + 0.2 * sem_sim_q + 0.2 * sem_sim_block
			if score > best_score:
				best_candidate = candidate
				best_score = score
	# Abstain only if conformal lower bound < 0.5
	if confidence_head.conformal_abstain(best_score, error_budget=0.5):
		case['final_answer'] = "ABSTAIN"
	else:
		case['final_answer'] = best_candidate
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
