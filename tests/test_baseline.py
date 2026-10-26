from __future__ import annotations

from lora_relevance_classifier.baseline import word_overlap_predict, word_overlap_score


def test_identical_text_scores_one():
    assert word_overlap_score("what is the deduction", "what is the deduction") == 1.0


def test_no_shared_words_scores_zero():
    assert word_overlap_score("tax deduction amount", "banana fruit recipe") == 0.0


def test_empty_question_scores_zero_not_a_crash():
    assert word_overlap_score("", "some context") == 0.0


def test_predict_respects_threshold():
    assert word_overlap_predict("a b c d", "a b", threshold=0.4) == 1
    assert word_overlap_predict("a b c d", "a", threshold=0.4) == 0
