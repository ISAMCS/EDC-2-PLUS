import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROV_RE = re.compile(r"\(source:.*?\)|#chunk\d+|path#chunk\d+", flags=re.I)

def sanitize_passage(p):
    # coerce score
    if "score" in p:
        try:
            p["score"] = float(p["score"])
        except Exception:
            p["score"] = 0.0
    # normalize has_answer
    if "has_answer" in p and not isinstance(p["has_answer"], bool):
        p["has_answer"] = bool(p["has_answer"])
    # strip provenance from text/title
    if "text" in p and isinstance(p["text"], str):
        p["text"] = PROV_RE.sub("", p["text"]).strip()
    if "title" in p and isinstance(p["title"], str):
        p["title"] = PROV_RE.sub("", p["title"]).strip()
    return p

def sanitize_file(path_in: str, path_out: str):
    p = Path(path_in)
    data = json.loads(p.read_text(encoding="utf-8"))
    # assume top-level is list of cases
    if isinstance(data, dict):
        # some outputs are one big dict with keys -> list; try to find list
        for k, v in data.items():
            if isinstance(v, list):
                data_list = v
                break
        else:
            data_list = [data]
    else:
        data_list = data
    cleaned = []
    for case in data_list:
        # sanitize passages
        if "passages" in case and isinstance(case["passages"], list):
            case["passages"] = [sanitize_passage(pp) for pp in case["passages"]]
        # coerce top-level score if present
        if "score" in case:
            try:
                case["score"] = float(case["score"])
            except Exception:
                case["score"] = 0.0
        # normalize has_answer
        if "has_answer" in case and not isinstance(case["has_answer"], bool):
            case["has_answer"] = bool(case["has_answer"])
        cleaned.append(case)
    # dedupe
    cleaned = dedupe_entries(cleaned)
    Path(path_out).write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sanitized -> {path_out}")

# New helper: infer a coarse answer type from the question
def infer_answer_type(question: str) -> str:
    """Very small heuristic to infer answer type: 'number','year','person','place','organization','other','ambiguous'"""
    q = (question or "").lower()
    if any(tok in q for tok in ["how many", "number of", "count of", "how much"]):
        return "number"
    if "year" in q or q.startswith("when") or "in what year" in q:
        return "year"
    if q.startswith("who") or "which person" in q:
        return "person"
    if q.startswith("where") or "which country" in q or "city" in q or "location" in q:
        return "place"
    if "company" in q or "organization" in q or "organisation" in q or "which organization" in q:
        return "organization"
    # ambiguous detection: asks for name but could be many things
    if any(tok in q for tok in ["what is the", "what are the", "which"]) and "year" not in q:
        return "ambiguous"
    return "other"

# New helper: canonicalize an answer string given an answer type
def canonicalize_answer(answer: str, ans_type: str) -> str:
    """Return a cleaned, type-normalized answer string (very lightweight)."""
    if answer is None:
        return ""
    a = str(answer).strip()
    # strip provenance tokens inserted into answers
    a = PROV_RE.sub("", a).strip()
    # collapse whitespace
    a = re.sub(r"\s+", " ", a)
    if not a:
        return ""
    if ans_type == "year":
        m = re.search(r"\b(19|20)\d{2}\b", a)
        return m.group(0) if m else a
    if ans_type == "number":
        # prefer plain integers; strip commas and extraneous text
        m = re.search(r"[-+]?\d{1,3}(?:[,.\d]{0,})\b", a)
        if m:
            return m.group(0).replace(",", "")
        return a
    if ans_type in ("person", "place", "organization"):
        # remove trailing punctuation and normalize spacing; keep capitalization
        a = a.strip(" .;,:")
        return a
    # fallback: return trimmed string
    return a

def consensus_gating(candidates: List[str], weights: List[float] = None, min_support: float = 0.5) -> Tuple[str, float]:
    """Pick a consensus answer from candidates using optional weights.
    Returns (chosen_answer, support_fraction). If none meets min_support returns ("", 0.0).
    """
    if not candidates:
        return "", 0.0
    # normalize candidates (strip/empty filter)
    clean = [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
    if not clean:
        return "", 0.0
    weights = weights or [1.0] * len(clean)
    # accumulate weighted counts keyed by normalized candidate
    counts: Dict[str, float] = {}
    for c, w in zip(clean, weights):
        key = c.lower()
        counts[key] = counts.get(key, 0.0) + float(w)
    total = sum(counts.values()) if counts else 0.0
    if total <= 0:
        return "", 0.0
    # pick top key
    top_key, top_weight = max(counts.items(), key=lambda kv: kv[1])
    support = top_weight / total
    if support >= min_support:
        # return candidate in its original (first matching) form if possible
        for c in clean:
            if c.lower() == top_key:
                return c, support
        return top_key, support
    return "", support


def postprocess_answers(
    entries_or_question,
    entries=None,
    top_k: int = 1,
    consensus_min_support: float = 0.5
) -> List[Dict[str, Any]]:
    """Postprocess a list of answer entries.
    Supports two call patterns:
      - postprocess_answers(entries, top_k=..., consensus_min_support=...)
      - postprocess_answers(question_str, entries, top_k=..., consensus_min_support=...)
    The function will normalize entries (strings/dicts/lists) into mappings so downstream code can
    safely call dict(e).
    - Collect candidate answers from 'answers', 'predictions', or 'passages'
    - Canonicalize according to inferred answer type per question
    - Produce 'final_answers' (list) and 'final_answer' (top string) and 'consensus_answer'
    Returns modified entries (shallow-copied).
    """
    # Normalize call signature: allow (question, entries) or (entries)
    question_override = None
    if entries is None:
        raw_entries = entries_or_question
    else:
        # caller provided question as first arg, entries as second
        question_override = entries_or_question if isinstance(entries_or_question, str) else None
        raw_entries = entries

    # Normalize raw_entries into a list of mappings
    norm_entries: List[Dict[str, Any]] = []
    if raw_entries is None:
        raw_entries = []
    if isinstance(raw_entries, dict):
        norm_entries = [raw_entries]
    elif isinstance(raw_entries, (list, tuple)):
        for it in raw_entries:
            if isinstance(it, dict):
                norm_entries.append(dict(it))
            elif isinstance(it, str):
                norm_entries.append({"text": it, "answer": it, "question": question_override or ""})
            else:
                norm_entries.append({"text": str(it), "answer": str(it), "question": question_override or ""})
    else:
        # scalar
        norm_entries = [{"text": str(raw_entries), "answer": str(raw_entries), "question": question_override or ""}]

    out: List[Dict[str, Any]] = []
    for e in norm_entries:
        entry = dict(e)  # shallow copy so we don't mutate input unexpectedly
        # if a question override was supplied and entry lacks a question, set it
        if question_override and not entry.get("question"):
            entry["question"] = question_override
        q = entry.get("question") or entry.get("query") or ""
        ans_type = infer_answer_type(q)
        candidates: List[str] = []

        # Preferred explicit fields
        if isinstance(entry.get("answers"), list) and entry["answers"]:
            for a in entry["answers"]:
                # each answer may be dict or string
                if isinstance(a, dict):
                    # common keys
                    cand = a.get("text") or a.get("answer") or a.get("value")
                else:
                    cand = a
                if cand is not None:
                    candidates.append(str(cand))
        elif isinstance(entry.get("predictions"), list) and entry["predictions"]:
            for p in entry["predictions"]:
                if isinstance(p, dict):
                    cand = p.get("text") or p.get("answer") or p.get("value")
                else:
                    cand = p
                if cand is not None:
                    candidates.append(str(cand))
        # Fallback: extract from passages
        elif isinstance(entry.get("passages"), list):
            for p in entry["passages"]:
                if not isinstance(p, dict):
                    continue
                # try common fields
                for k in ("answer", "text", "title", "summary"):
                    if k in p and p[k]:
                        candidates.append(str(p[k]))
                # also allow short spans in 'bullets' or 'snippet'
                if "snippet" in p and p["snippet"]:
                    candidates.append(str(p["snippet"]))
                if "bullets" in p and isinstance(p["bullets"], list):
                    for b in p["bullets"]:
                        candidates.append(str(b))

        # canonicalize and filter
        canoned: List[str] = []
        for c in candidates:
            ca = canonicalize_answer(c, ans_type)
            if ca and isinstance(ca, str):
                ca = ca.strip()
                if ca:
                    canoned.append(ca)

        # dedupe preserving order
        seen = set()
        uniq: List[str] = []
        for c in canoned:
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)

        # Build final answers: take top_k shortest / highest-confidence heuristically
        # If numeric type prefer numeric answers first
        if ans_type in ("number", "year"):
            # prefer answers that are digits
            digits = [u for u in uniq if re.fullmatch(r"\d+", u)]
            rest = [u for u in uniq if u not in digits]
            ranked = digits + rest
        else:
            # prefer shorter spans (likely concise entities) then original order
            ranked = sorted(uniq, key=lambda s: (len(s.split()), len(s)))  # shorter tokens first

        final_answers = ranked[:max(1, top_k)]
        consensus, consensus_support = consensus_gating(uniq, None, min_support=consensus_min_support)

        entry["final_answers"] = final_answers
        entry["final_answer"] = final_answers[0] if final_answers else ""
        entry["consensus_answer"] = consensus
        entry["consensus_support"] = consensus_support
        out.append(entry)
    return out

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python Scripts/sanitize_extracted.py input.json output.json")
        sys.exit(1)
    sanitize_file(sys.argv[1], sys.argv[2])