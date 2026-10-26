from __future__ import annotations

import pytest

from lora_relevance_classifier.metrics import compute_metrics


def test_perfect_predictions():
    m = compute_metrics([1, 0, 1, 0], [1, 0, 1, 0])
    assert m.accuracy == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.false_positive_rate == 0.0


def test_all_wrong_predictions():
    m = compute_metrics([1, 0], [0, 1])
    assert m.accuracy == 0.0
    assert m.false_positive_rate == 1.0


def test_empty_inputs_do_not_divide_by_zero():
    m = compute_metrics([], [])
    assert m.accuracy == 0.0


def test_no_predicted_positives_gives_zero_precision_not_a_crash():
    m = compute_metrics([1, 0], [0, 0])
    assert m.precision == 0.0
    assert m.recall == 0.0


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        compute_metrics([1, 0], [1])
