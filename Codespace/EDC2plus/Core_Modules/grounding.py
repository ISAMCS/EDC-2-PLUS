
"""
Sentence-level grounding that forces claim → cite alternation.

Usage:
    from codes.rag_plus.grounding import sentence_level_ground, build_final_prompt_claim_cite
    claims = sentence_level_ground(question, passages, eval_model="mistral7b_instruct_request")
    prompt = build_final_prompt_claim_cite(question, claims)
"""
from __future__ import annotations
import json, re
from typing import List, Dict, Any

# We import lazily to avoid hard dependency if user only uses offline features.
import importlib

def _get_llm(eval_model: str):
    tu = importlib.import_module("codes.text_utils")
    fn = getattr(tu, eval_model, None)
    if fn is None:
        # fallbacks in likely order
        for name in ["mistral7b_instruct_request", "gpt35_turbo_0613_request"]:
            if hasattr(tu, name):
                return getattr(tu, name)
        raise RuntimeError(f"Cannot find LLM function '{eval_model}' in codes.text_utils")
    return fn

_CLAIM_PROMPT = """You are assisting with precise, quote-backed summarization for a Retrieval-Augmented Generation system.

Given the USER QUESTION and a set of PASSAGES, extract 3-7 concise factual CLAIMS that directly help answer the question.
For each CLAIM, include an exact QUOTED SPAN (<= 30 words) copied verbatim from one passage that supports the claim.
Return STRICT JSON as a list of objects with keys: claim, quote, source_id, source_title.

Rules:
- The quote MUST be a direct substring of the passage text.
- Prefer short, atomic facts per claim.
- If claims disagree across passages, include both claims (different quotes/sources).
- Do not invent sources.
- If no support exists, return an empty list []

JSON ONLY. No prose.
USER QUESTION:
{question}

PASSAGES (id :: title :: text snippet):
{passage_lines}
"""

def sentence_level_ground(question: str, passages: List[Dict[str, Any]], eval_model: str = "mistral7b_instruct_request") -> List[Dict[str, str]]:
    # Build compact passage listing
    lines = []
    for p in passages[:12]:
        pid = p.get("id") or p.get("doc_id") or p.get("title","unk")
        title = p.get("title","")
        text = (p.get("text","") or "")[:500].replace("\n"," ")
        lines.append(f"{pid} :: {title} :: {text}")
    prompt = _CLAIM_PROMPT.format(question=question, passage_lines="\n".join(lines))

    try:
        llm = _get_llm(eval_model)
        raw = llm(prompt, temperature=0.0)
    except Exception as e:
        # On failure, fall back to empty grounding
        raw = "[]"

    # Parse as JSON robustly
    m = re.search(r"\[.*\]", raw, flags=re.S)
    payload = m.group(0) if m else "[]"
    try:
        claims = json.loads(payload)
        # sanity enforcement
        out = []
        for c in claims:
            if isinstance(c, dict) and all(k in c for k in ["claim","quote","source_id","source_title"]):
                out.append({k: c[k] for k in ["claim","quote","source_id","source_title"]})
        return out
    except Exception:
        return []

def build_final_prompt_claim_cite(question: str, claims: List[Dict[str,str]]) -> str:
    if not claims:
        return f"Answer the question concisely and truthfully.\nQuestion: {question}\nIf unsure, say you don't know."
    lines = [f"You must alternate: Cite → Claim. Use only the provided quotes.\nQuestion: {question}\n"]
    for i, c in enumerate(claims, 1):
        src = f'[{c.get("source_id","?")} - {c.get("source_title","")}]'
        lines.append(f'Cite {i}: "{c["quote"]}" {src}')
        lines.append(f'Claim {i}: {c["claim"]}')
    lines.append("\nThen produce a final answer that is fully supported by the cited quotes.")
    return "\n".join(lines)
