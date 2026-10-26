"""Turns rag-qa-api's answerability eval dataset (a question, plus real evidence quotes for an
answerable one, or no evidence for an unanswerable one) into a sentence-pair relevance
classification dataset: given a question and a candidate chunk of text, does that chunk
actually answer it? This is exactly the decision rag-qa-api's MIN_RERANK_SCORE threshold makes
today -- turning it into a supervised task is what a LoRA-tuned classifier could learn to make
instead, per open-source-project-plan.md's own description of project 5's task.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SourceQuestion:
    id: str
    question: str
    evidence: list[str]
    answerable: bool


@dataclass
class PairExample:
    question: str
    context: str
    label: int  # 1 = context answers the question, 0 = it doesn't
    source_id: str
    pair_kind: str  # "positive" | "hard_negative" | "no_context_negative"


def load_source_questions(path: Path) -> list[SourceQuestion]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        questions.append(
            SourceQuestion(
                id=row["id"],
                question=row["question"],
                evidence=row.get("evidence") or [],
                answerable=row.get("answer") is not None,
            )
        )
    return questions


def build_pairs(
    questions: list[SourceQuestion], *, hard_negatives_per_question: int = 2, seed: int = 0
) -> list[PairExample]:
    """One positive pair per real evidence snippet a question has, plus up to
    `hard_negatives_per_question` pairs against OTHER questions' evidence (a chunk a retriever
    might plausibly surface but that doesn't actually answer this question -- the hard case the
    gate has to get right), plus one negative pair per genuinely unanswerable question against
    a randomly chosen piece of evidence (simulating "the retriever found something, but nothing
    relevant to this question exists anywhere in the corpus").
    """
    rng = random.Random(seed)
    answerable = [q for q in questions if q.answerable and q.evidence]
    all_evidence = [(q.id, snippet) for q in answerable for snippet in q.evidence]

    pairs: list[PairExample] = []
    for q in questions:
        if q.answerable and q.evidence:
            for snippet in q.evidence:
                pairs.append(PairExample(q.question, snippet, 1, q.id, "positive"))

            other_evidence = [(oid, s) for oid, s in all_evidence if oid != q.id]
            sample_size = min(hard_negatives_per_question, len(other_evidence))
            for _oid, snippet in rng.sample(other_evidence, sample_size):
                pairs.append(PairExample(q.question, snippet, 0, q.id, "hard_negative"))
        else:
            if all_evidence:
                _, snippet = rng.choice(all_evidence)
                pairs.append(PairExample(q.question, snippet, 0, q.id, "no_context_negative"))

    return pairs


def split_pairs(
    pairs: list[PairExample], *, val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[PairExample], list[PairExample]]:
    """Splits by source_id, not by pair -- so one question's positive and negative pairs never
    end up on both sides of the split. Splitting by pair would leak that question's exact
    phrasing into validation, making the held-out score optimistic."""
    source_ids = sorted({p.source_id for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(source_ids)
    n_val = max(1, round(len(source_ids) * val_fraction))
    val_ids = set(source_ids[:n_val])
    train = [p for p in pairs if p.source_id not in val_ids]
    val = [p for p in pairs if p.source_id in val_ids]
    return train, val


def save_pairs(pairs: list[PairExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(
                json.dumps(
                    {
                        "question": p.question,
                        "context": p.context,
                        "label": p.label,
                        "source_id": p.source_id,
                        "pair_kind": p.pair_kind,
                    }
                )
                + "\n"
            )


def load_pairs(path: Path) -> list[PairExample]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            pairs.append(PairExample(**json.loads(line)))
    return pairs
