from __future__ import annotations

def check_query_stability(question):
    """
    Stub for query stability guard.
    Replace with your actual logic or model.
    """
    # Example: classify by question length
    length = len(question.split())
    if length < 8:
        return "high"
    elif length < 16:
        return "medium"
    else:
        return "low"

"""
Query Guard: retrieval “sanity checks” with negation & counterfactual probes.

Drop-in usage:
    from codes.rag_plus.query_guard import GuardedRetriever
    retr = GuardedRetriever(passages=case["passages"])  # offline reranker
    info = retr.stability(case["question"], k=20)
    contexts = retr.retrieve(case["question"], k=20, tau=0.4)
"""
import os, re, json, math, time, hashlib, random, string
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SK = True
except Exception:
    _HAS_SK = False

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

@dataclass
class GuardResult:
    query: str
    k: int
    stability_score: float            # mean Jaccard overlap across probes
    action: str                       # 'pass' | 'tighten' | 'ask_followup'
    base_ids: List[str]
    overlaps: List[float]
    probes: List[str]
    probe_ids: List[List[str]]

class GuardedRetriever:
    """
    If `passages` is provided: uses an internal TF-IDF retriever over the list of passages.
    Otherwise you can pass a `retriever_fn` that takes (query, k) -> list of {id, title, text}.
    """
    def __init__(self, passages: Optional[List[Dict[str, Any]]] = None,
                 retriever_fn=None):
        self.passages = passages
        self.retriever_fn = retriever_fn

    # ---------------------- PERTURBERS ----------------------
    def _negate(self, q: str) -> str:
        # naive negation injection
        if re.search(r"\bnot\b", q, flags=re.I): 
            return re.sub(r"\bnot\b", "not", q, flags=re.I)
        return re.sub(r"\b(is|are|was|were|do|does|did|has|have|had|should|could|would)\b",
                      r"\1 not", q, flags=re.I, count=1)

    def _swap_entities(self, q: str) -> str:
        # swap two capitalized tokens (very lightweight counterfactual)
        caps = re.findall(r"\b([A-Z][a-zA-Z0-9\-']+)\b", q)
        if len(caps) >= 2:
            a, b = caps[0], caps[1]
            return re.sub(rf"\b{a}\b", "__TMP__", q).replace(b, a).replace("__TMP__", b)
        # fallback: swap a year if present
        years = re.findall(r"\b(19|20)\d{2}\b", q)
        if years:
            y = years[0]
            yy = str(int("".join(years[0])) + 1)
            return q.replace(y, yy)
        return q

    def _except_probe(self, q: str) -> str:
        if " except " in q.lower(): 
            return q
        return q.strip().rstrip("?") + " except"

    def perturb_queries(self, q: str, n: int = 3) -> List[str]:
        # Generate up to 3 probes
        probes = [self._negate(q), self._swap_entities(q), self._except_probe(q)]
        uniq = []
        for p in probes:
            if p and p != q and p not in uniq:
                uniq.append(p)
        return uniq[:n]

    # ---------------------- RETRIEVAL ----------------------
    def _rank_passages(self, query: str, k: int) -> List[Dict[str, Any]]:
        if not self.passages:
            raise ValueError("Offline mode requires `passages` to be provided")
        texts = [(p.get("id") or str(i), p.get("title","") + " " + p.get("text","")) 
                 for i, p in enumerate(self.passages)]
        ids, docs = zip(*texts)

        if _HAS_SK:
            try:
                vec = TfidfVectorizer(stop_words="english", max_features=50000)
                X = vec.fit_transform(docs + (query,))
                qv = X[-1]
                D = X[:-1]
                sims = (D @ qv.T).toarray().ravel()
                order = sims.argsort()[::-1][:k]
                return [self.passages[i] for i in order]
            except Exception:
                pass

        # fallback: simple token overlap
        q_toks = set(re.findall(r"\w+", query.lower()))
        scored = []
        for i, d in enumerate(docs):
            toks = set(re.findall(r"\w+", d.lower()))
            sim = len(q_toks & toks) / (len(q_toks) + 1e-6)
            scored.append((sim, i))
        scored.sort(reverse=True)
        return [self.passages[i] for _, i in scored[:k]]

    def _ids(self, items: List[Dict[str, Any]]) -> List[str]:
        out = []
        for p in items:
            pid = p.get("id") or p.get("doc_id") or p.get("title") or hashlib.md5((p.get("text",""))[:128].encode()).hexdigest()
            out.append(str(pid))
        return out

    def _jaccard(self, a: List[str], b: List[str]) -> float:
        A, B = set(a), set(b)
        if not A and not B: return 1.0
        if not A or not B: return 0.0
        return len(A & B) / float(len(A | B))

    # ---------------------- PUBLIC API ----------------------
    def stability(self, query: str, k: int = 10) -> GuardResult:
        base = self._rank_passages(query, k)
        base_ids = self._ids(base)

        probes = self.perturb_queries(query, n=3)
        overlaps = []
        probe_ids = []
        for p in probes:
            ranked = self._rank_passages(p, k)
            pids = self._ids(ranked)
            probe_ids.append(pids)
            overlaps.append(self._jaccard(base_ids, pids))

        score = float(sum(overlaps)/len(overlaps)) if overlaps else 1.0

        # action policy
        action = "pass"
        if score < 0.25: action = "ask_followup"
        elif score < 0.5: action = "tighten"

        result = GuardResult(
            query=query, k=k, stability_score=round(score,3),
            action=action, base_ids=base_ids,
            overlaps=[round(x,3) for x in overlaps],
            probes=probes, probe_ids=probe_ids
        )
        # log
        with open(os.path.join(LOG_DIR, "query_guard.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        return result

    def retrieve(self, query: str, k: int = 10, tau: float = 0.4) -> List[Dict[str, Any]]:
        info = self.stability(query, k=k)
        ranked = self._rank_passages(query, k)
        if info.stability_score >= tau or info.action == "pass":
            return ranked

        # tighten: reduce k and require higher lexical match
        tightened_k = max(5, int(k * 0.5))
        q_toks = set(re.findall(r"\w+", query.lower()))
        filtered = []
        for p in ranked:
            toks = set(re.findall(r"\w+", (p.get("title","") + " " + p.get("text","")).lower()))
            if len(q_toks & toks) / (len(q_toks) + 1e-6) >= 0.25:
                filtered.append(p)
            if len(filtered) >= tightened_k:
                break

        return filtered or ranked[:tightened_k]
