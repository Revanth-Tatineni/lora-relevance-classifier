"""Evaluates a model -- the base model zero-shot, or base+LoRA adapter -- on a set of pairs.
Pure inference, no training, so this is the "after" half of the before/after comparison
(baseline.py's word-overlap heuristic is the "before" half that doesn't need a model at all).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lora_relevance_classifier.config import TrainConfig
from lora_relevance_classifier.dataset import PairExample
from lora_relevance_classifier.metrics import ClassificationMetrics, compute_metrics


@torch.no_grad()
def predict(model: Any, tokenizer: Any, pairs: list[PairExample], max_length: int) -> list[int]:
    model.eval()
    predictions = []
    for pair in pairs:
        inputs = tokenizer(
            pair.question,
            pair.context,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        logits = model(**inputs).logits
        predictions.append(int(torch.argmax(logits, dim=-1).item()))
    return predictions


def evaluate_pairs(
    model: Any, tokenizer: Any, pairs: list[PairExample], max_length: int
) -> ClassificationMetrics:
    predictions = predict(model, tokenizer, pairs, max_length)
    labels = [p.label for p in pairs]
    return compute_metrics(labels, predictions)


def load_base_model(cfg: TrainConfig) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=cfg.num_labels
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def load_adapter_model(adapter_path: Path, cfg: TrainConfig) -> tuple[Any, Any]:
    base_model, tokenizer = load_base_model(cfg)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model, tokenizer
