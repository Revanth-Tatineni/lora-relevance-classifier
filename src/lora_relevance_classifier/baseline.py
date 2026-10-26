"""A zero-training baseline to compare the LoRA classifier against: word overlap between the
question and the candidate context. This is a deliberately weak proxy for rag-qa-api's real
MIN_RERANK_SCORE gate (a trained cross-encoder) -- getting the real cross-encoder's scores on
this same data needs rag-qa-api's retrieval stack running, which this project doesn't spin up
(see docs/explained). Until that integration exists, this baseline still answers a real,
narrower question: does fine-tuning beat a trivial lexical heuristic?
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def word_overlap_score(question: str, context: str) -> float:
    q_words = _words(question)
    if not q_words:
        return 0.0
    return len(q_words & _words(context)) / len(q_words)


def word_overlap_predict(question: str, context: str, threshold: float = 0.3) -> int:
    return 1 if word_overlap_score(question, context) >= threshold else 0
