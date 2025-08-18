from __future__ import annotations

def classify_query_temporal(question):
    """
    Stub classifier for time-sensitive vs static queries.
    Replace with your actual logic or model.
    """
    # Example: classify by presence of time words
    time_words = ["year", "date", "recent", "today", "last", "ago", "month", "week"]
    if any(word in question.lower() for word in time_words):
        return "time-sensitive"
    return "static"
# -*- coding: utf-8 -*-
"""
Time-aware retrieval policy.
"""
import re
from datetime import datetime, timedelta
from typing import Literal

Policy = Literal["local-only","hybrid","web-only"]

RECENCY_WORDS = r"\b(latest|today|yesterday|this week|this month|this year|current|recent|breaking|now|202[3-9]|20[3-9]\d)\b"

def _is_time_sensitive(q: str) -> bool:
    if re.search(RECENCY_WORDS, q.lower()):
        return True
    # asks "who is/are the president/CEO of ..." style
    if re.search(r"\b(who|what)\s+(is|are)\b.*\b(ceo|president|prime minister|price|score|weather)\b", q.lower()):
        return True
    return False

def TemporalRouter(question: str = None):
    """
    Callable class pattern for convenience: route = TemporalRouter()(question)
    """
    class _Router:
        def __call__(self, question: str) -> dict:
            ts = _is_time_sensitive(question)
            policy: Policy = "local-only"
            if ts:
                policy = "hybrid"
            return {
                "policy": policy,
                "freshness_filter_days": 45 if ts else None,
                "reason": "Detected time-sensitive phrasing" if ts else "No recency cues detected"
            }
    return _Router()
