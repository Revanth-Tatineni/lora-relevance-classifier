"""Pure metric computation over (label, prediction) pairs -- no model, no tokenizer, so this
is correct and testable independent of whether the ML stack can even load a model in a given
environment. This is the one piece that has to be right for the before/after comparison to
mean anything.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    # Predicted "relevant" when it wasn't -- the direction that lets an unsupported answer
    # through rag-qa-api's relevance gate. This is the number that matters most for this task,
    # more than accuracy: a gate that's slightly too strict just over-refuses; one that's too
    # loose lets hallucination-risk answers through.
    false_positive_rate: float


def compute_metrics(labels: list[int], predictions: list[int]) -> ClassificationMetrics:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must be the same length")

    paired = list(zip(labels, predictions, strict=True))
    tp = sum(1 for lbl, pred in paired if lbl == 1 and pred == 1)
    tn = sum(1 for lbl, pred in paired if lbl == 0 and pred == 0)
    fp = sum(1 for lbl, pred in paired if lbl == 0 and pred == 1)
    fn = sum(1 for lbl, pred in paired if lbl == 1 and pred == 0)

    accuracy = (tp + tn) / len(labels) if labels else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

    return ClassificationMetrics(accuracy, precision, recall, f1, false_positive_rate)
