
"""
Uncertainty-aware retrieval evaluator (low/med/high).
"""
from __future__ import annotations
import re
from typing import List, Dict, Any

def _ngrams(tokens, n):
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def RetrievalEvaluator(question: str, previews: List[Dict[str, Any]]) -> str:
    """Heuristic confidence score based on lexical coverage of question in previews."""
    qtoks = re.findall(r"\w+", question.lower())
    if not qtoks or not previews:
        return "low"
    grams = _ngrams(qtoks, 1) | _ngrams(qtoks, 2)
    cover = 0.0
    total = len(grams) or 1.0
    for p in previews[:8]:
        txt = (p.get("title","") + " " + p.get("text","")).lower()
        ptoks = re.findall(r"\w+", txt)
        g2 = _ngrams(ptoks, 1) | _ngrams(ptoks, 2)
        cover += len(grams & g2) / total
    cover /= min(len(previews[:8]), 8) or 1.0
    if cover >= 0.7: return "high"
    if cover >= 0.4: return "med"
    return "low"
