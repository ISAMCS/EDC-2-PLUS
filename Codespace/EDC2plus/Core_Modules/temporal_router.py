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

class TemporalRouter:
    def __init__(self):
        pass
    def __call__(self, question: str) -> dict:
        ts = classify_query_temporal(question)
        policy = "hybrid" if ts == "time-sensitive" else "local-only"
        freshness_filter_days = 45 if policy == "hybrid" else None
        # Advisory only: do not enforce dropping static trivia
        return {
            "policy": policy,
            "freshness_filter_days": freshness_filter_days,
            "reason": "Detected time-sensitive phrasing" if policy == "hybrid" else "No recency cues detected",
            "prefer_recent": policy == "hybrid",
            "advisory_only": True
        }
