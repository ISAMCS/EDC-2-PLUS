try:
	from tqdm import tqdm
except Exception:  # fallback when tqdm isn't installed
	def tqdm(x, **kwargs):
		return x

import sys
import json
import ast
import re
import os


# Cache LLM utils import so we warn only once if heavy deps (torch) are missing
_llm_utils = None
_llm_utils_import_attempted = False

def _ensure_llm_utils():
	"""Try to import Codespace.LLMs.utils once and cache the result.
	Returns the utils module or None if import failed.
	"""
	global _llm_utils, _llm_utils_import_attempted
	if _llm_utils is not None:
		return _llm_utils
	if _llm_utils_import_attempted:
		return None
	_llm_utils_import_attempted = True
	try:
		from Codespace.LLMs import utils as _u
		_llm_utils = _u
		return _llm_utils
	except Exception as e:
		# Only print warning if explicit verbosity requested
		if os.getenv('EVAL_VERBOSE', '').lower() in ('1', 'true', 'yes'):
			print(f"Warning: could not import LLM utils ({e}); LLM calls will return ABSTAIN")
		return None


def _load_answer_postprocess():
	"""Dynamically load answer_postprocess.py by file path to avoid importing package __init__.
	Returns (postprocess_answers, infer_answer_type) or (None, None) on failure.
	"""
	try:
		import importlib.util
		base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
		mod_path = os.path.join(base, 'Core_Modules', 'answer_postprocess.py')
		spec = importlib.util.spec_from_file_location('answer_postprocess', mod_path)
		mod = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(mod)
		return getattr(mod, 'postprocess_answers', None), getattr(mod, 'infer_answer_type', None)
	except Exception as e:
		print(f"Warning: could not load answer_postprocess directly ({e}); using simple fallbacks")
		return None, None


_postprocess_answers, _infer_answer_type = _load_answer_postprocess()


def postprocess_wrapper(question, answers_list):
	"""Normalize answers using the loaded postprocess_answers (which expects entries list).
	Returns a single finalized answer string.
	"""
	if _postprocess_answers:
		entry = {"question": question, "answers": answers_list}
		try:
			processed = _postprocess_answers([entry])
			if isinstance(processed, list) and processed:
				out = processed[0]
				# prefer 'final_answer' then first of 'final_answers'
				if out.get('final_answer'):
					return out.get('final_answer')
				fa = out.get('final_answers')
				if isinstance(fa, list) and fa:
					return fa[0]
			# fallback
			return answers_list[0] if answers_list else 'ABSTAIN'
		except Exception:
			return answers_list[0] if answers_list else 'ABSTAIN'
	else:
		return answers_list[0] if answers_list else 'ABSTAIN'


def _strip_provenance(s: str) -> str:
	s = re.sub(r"\(source:[^)]+\)", "", s)
	s = re.sub(r"\b\d+#chunk[^\s)]+", "", s)
	return s


def llm_answer_from_blocks(question, compressed_blocks, eval_model):
	blocks = [_strip_provenance(b) for b in compressed_blocks]
	prompt = f"""
# Instruction:
You are given a question and a set of compressed, sourced blocks. Compose a factual answer using only the information in these blocks. Do not hallucinate or add information not present in the blocks.
# Question:
{question}
# Compressed Blocks:
{chr(10).join(blocks)}
# Output:
"""
	utils = _ensure_llm_utils()
	if utils is None:
		return postprocess_wrapper(question, ["ABSTAIN"])

	if eval_model == "llama4":
		text = utils.llama4_maverick_request(prompt)
	elif eval_model == "Phi":
		text = utils.Microsoft_Phi4_request(prompt)
	elif eval_model == "mistral7b_instruct":
		text = utils.mistral7b_instruct_request(prompt)
	elif eval_model == "gpt35_turbo":
		text = utils.gpt35_turbo_0613_request(prompt)
	elif eval_model == "local_hf":
		text = utils.local_hf_request(prompt)
	else:
		raise ValueError(f"Unknown model: {eval_model}")

	infer_answer_type_fn = _infer_answer_type or (lambda q: "other")
	postprocess_answers_fn = _postprocess_answers or (lambda q, lst: (lst[0] if lst else "ABSTAIN"))
	ans_type = infer_answer_type_fn(question)
	if not text.strip():
		cands = []
		for b in blocks:
			sent_hits = [s for s in re.split(r'(?<=[.!?])\s+', b)
						 if any(t in s.lower() for t in ["which country", "where", "who", "how many", "what year"])]
			for s in sent_hits:
				if ans_type == "year":
					cands += re.findall(r"\b(?:19|20)\d{2}\b", s)
				elif ans_type == "number":
					cands += [m for m in re.findall(r"\b\d+\b", s) if not re.match(r"\d{4}", m)]
				elif ans_type == "person":
					cands += re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b", s)
				elif ans_type == "place":
					cands += re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b", s)
		text = postprocess_wrapper(question, list(dict.fromkeys(cands))) or "ABSTAIN"
	return postprocess_wrapper(question, [text])


def main():
	# Flexible argument parsing to support optional second toggle (e.g. `full all`)
	if len(sys.argv) < 7:
		print("Usage: eval_baseline_compress_plus.py <model> <date> <dataset> <topk_list> <noise_list> <benchmark> [toggle]")
		sys.exit(1)

	eval_model = sys.argv[1]
	date = sys.argv[2]
	dataset = sys.argv[3]
	topkk = ast.literal_eval(sys.argv[4])
	noises = ast.literal_eval(sys.argv[5])
	benchmark = sys.argv[6]
	toggle = sys.argv[7] if len(sys.argv) > 7 else None

	# Build suffix used in filenames (handles cases like `_full_all`)
	suffix = f"_{benchmark}"
	if toggle:
		suffix += f"_{toggle}"

	for topk in topkk:
		for noise in noises:
			# Discover input file by searching common locations and matching files that contain compressed_blocks
			import glob

			pattern_candidates = []
			# datasets folder (may not contain compressed_blocks for EDC2+)
			pattern_candidates.append(f"{dataset}/datasets/case_{date}_{dataset}_summary_edc2plus_compress_{eval_model}_noise{noise}_topk{topk}*{suffix}.json")
			# extracted answers folder (likely contains compressed_blocks)
			pattern_candidates.append(f"{dataset}/extracted_answer/*{date}*{eval_model}*noise{noise}*topk{topk}*.json")
			# results folder (sometimes contains compressed_blocks)
			pattern_candidates.append(f"{dataset}/results/*{date}*{eval_model}*noise{noise}*topk{topk}*.json")

			case_file = None
			for pat in pattern_candidates:
				for candidate in glob.glob(pat):
					try:
						with open(candidate, 'r', encoding='utf-8') as fh:
							data = json.load(fh)
							# look at first few entries for compressed_blocks key
							sample = data[:5] if isinstance(data, list) else [data]
							if any(isinstance(entry, dict) and 'compressed_blocks' in entry for entry in sample):
								case_file = candidate
								break
					except Exception:
						continue
				if case_file:
					break

			# Fall back to constructed filename if discovery failed
			if case_file is None:
				case_file = f"{dataset}/datasets/case_{date}_{dataset}_summary_edc2plus_compress_{eval_model}_noise{noise}_topk{topk}{suffix}.json"

			res_file = f"{dataset}/results/{date}_{dataset}_edc2plus_compress_{eval_model}_noise{noise}_topk{topk}{suffix}.json"

			if not os.path.exists(case_file):
				print(f"Skipping missing case file: {case_file}")
				continue

			print(f"Using case file: {case_file}")

			os.makedirs(os.path.dirname(res_file), exist_ok=True)

			with open(case_file, "r", encoding="utf-8") as fh:
				cases = json.load(fh)

			# Support smoke mode: limit number of cases processed when SMOKE env var is set
			smoke_mode = os.getenv('SMOKE', '').lower() in ('1', 'true', 'yes')
			if smoke_mode:
				smoke_n = int(os.getenv('SMOKE_N', '10'))
				print(f"SMOKE mode active: limiting to first {smoke_n} cases")
				cases = cases[:smoke_n]

			final_result = []
			for case in tqdm(cases, desc=f"{topk}/noise{noise}{suffix}"):
				# Robustly find compressed blocks under several possible keys
				cb_keys = ['compressed_blocks', 'compressed_block', 'compressed', 'blocks', 'sourced_blocks']
				compressed_blocks = None
				for k in cb_keys:
					if k in case:
						compressed_blocks = case[k]
						break
				if compressed_blocks is None:
					print(f"Warning: case missing compressed blocks, skipping question: {case.get('question', '<no question>')}")
					case['edc2plus_response'] = 'ABSTAIN'
					final_result.append(case)
					continue

				# Ensure compressed_blocks is a list of strings
				if isinstance(compressed_blocks, str):
					compressed_blocks = [compressed_blocks]
				elif not isinstance(compressed_blocks, list):
					try:
						compressed_blocks = list(compressed_blocks)
					except Exception:
						compressed_blocks = [str(compressed_blocks)]

				answer = llm_answer_from_blocks(case.get('question', ''), compressed_blocks, eval_model)
				# Normalize and post-process answer (use dynamic function if available)
				case['edc2plus_response'] = postprocess_wrapper(case.get('question', ''), [answer])
				final_result.append(case)

			with open(res_file, "w", encoding="utf-8") as json_file:
				json.dump(final_result, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
	main()
