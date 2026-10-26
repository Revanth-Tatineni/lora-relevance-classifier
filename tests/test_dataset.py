from __future__ import annotations

from lora_relevance_classifier.dataset import (
    PairExample,
    SourceQuestion,
    build_pairs,
    load_pairs,
    load_source_questions,
    save_pairs,
    split_pairs,
)


def _q(
    source_id: str, question: str, evidence: list[str] | None = None, answerable: bool = True
) -> SourceQuestion:
    return SourceQuestion(
        id=source_id, question=question, evidence=evidence or [], answerable=answerable
    )


def test_load_source_questions_from_jsonl(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        '{"id": "a", "question": "q1", "answer": "x", "evidence": ["e1"]}\n'
        '{"id": "b", "question": "q2", "answer": null, "evidence": []}\n',
        encoding="utf-8",
    )
    questions = load_source_questions(path)
    assert len(questions) == 2
    assert questions[0].answerable is True
    assert questions[1].answerable is False


def test_build_pairs_creates_one_positive_per_evidence_snippet():
    questions = [_q("a", "q1", evidence=["e1", "e2"]), _q("b", "q2", evidence=["e3"])]
    pairs = build_pairs(questions, hard_negatives_per_question=0)
    positives = [p for p in pairs if p.pair_kind == "positive"]
    assert len(positives) == 3
    assert all(p.label == 1 for p in positives)


def test_build_pairs_creates_hard_negatives_from_other_questions_evidence():
    questions = [
        _q("a", "q1", evidence=["e1"]),
        _q("b", "q2", evidence=["e2"]),
        _q("c", "q3", evidence=["e3"]),
    ]
    pairs = build_pairs(questions, hard_negatives_per_question=2, seed=1)
    hard_negs = [p for p in pairs if p.pair_kind == "hard_negative" and p.source_id == "a"]
    assert len(hard_negs) == 2
    assert all(p.label == 0 for p in hard_negs)
    assert all(p.context != "e1" for p in hard_negs)


def test_build_pairs_unanswerable_question_gets_one_negative_pair():
    questions = [
        _q("a", "q1", evidence=["e1"]),
        _q("b", "q2", evidence=[], answerable=False),
    ]
    pairs = build_pairs(questions, hard_negatives_per_question=0)
    unanswerable_pairs = [p for p in pairs if p.source_id == "b"]
    assert len(unanswerable_pairs) == 1
    assert unanswerable_pairs[0].label == 0
    assert unanswerable_pairs[0].pair_kind == "no_context_negative"


def test_split_pairs_keeps_each_source_id_on_one_side_only():
    questions = [_q(f"q{i}", f"question {i}", evidence=[f"e{i}"]) for i in range(10)]
    pairs = build_pairs(questions, hard_negatives_per_question=2, seed=0)
    train, val = split_pairs(pairs, val_fraction=0.2, seed=0)

    train_ids = {p.source_id for p in train}
    val_ids = {p.source_id for p in val}
    assert train_ids.isdisjoint(val_ids)
    assert len(val_ids) >= 1


def test_save_then_load_round_trips(tmp_path):
    pairs = [PairExample("q", "c", 1, "id1", "positive")]
    path = tmp_path / "pairs.jsonl"
    save_pairs(pairs, path)
    assert load_pairs(path) == pairs
