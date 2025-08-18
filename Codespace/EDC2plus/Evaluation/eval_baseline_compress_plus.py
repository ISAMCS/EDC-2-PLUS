
from tqdm import tqdm

import sys
import json
import ast
from Codespace.LLMs import utils

eval_model = sys.argv[1]
date = sys.argv[2]
dataset = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]

def llm_answer_from_blocks(question, compressed_blocks):
	# Compose answer strictly from compressed blocks using LLM
	prompt = f"""
	# Instruction:
	You are given a question and a set of compressed, sourced blocks. Compose a factual answer using only the information in these blocks. Do not hallucinate or add information not present in the blocks.
	# Question:
	{question}
	# Compressed Blocks:
	{chr(10).join(compressed_blocks)}
	# Output:
	"""
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

def run(topk, noise):
	global eval_model, date, dataset, benchmark
	# Load cases with compressed blocks
	case_file = f"{dataset}/datasets/case_{date}_{dataset}_summary_edc2plus_compress_{eval_model}_noise{noise}_topk{topk}.json"
	with open(case_file, "r", encoding="utf-8") as lines:
		cases = json.load(lines)
		final_result = []
		for case in tqdm(cases):
			answer = llm_answer_from_blocks(case['question'], case['compressed_blocks'])
			case['edc2plus_response'] = answer
			final_result.append(case)
		# Save results for downstream EM/F1 calculation
		res_file = f"{dataset}/results/{date}_{dataset}_edc2plus_compress_{eval_model}_noise{noise}_topk{topk}.json"
		with open(res_file, "w", encoding="utf-8") as json_file:
			json.dump(final_result, json_file, ensure_ascii=False, indent=4)

for topk in topkk:
	for noise in noises:
		run(topk, noise)
		print(f"Finished EDC2+ answer generation for topk={topk} and noise={noise}")
