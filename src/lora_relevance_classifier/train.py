"""The actual LoRA fine-tune. For the real target model (config.TrainConfig.base_model,
Qwen2.5-0.5B per open-source-project-plan.md), this needs a GPU to run in a reasonable time --
see docs/explained and the Colab notes. This module is importable and exercised in
tests/test_train_smoke.py with a tiny proxy model on CPU, to prove the code is mechanically
correct; that smoke test is NOT a claim that the real target model has been fine-tuned (see
README's status section).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from lora_relevance_classifier.config import TrainConfig
from lora_relevance_classifier.dataset import PairExample
from lora_relevance_classifier.metrics import compute_metrics


def _to_hf_dataset(pairs: list[PairExample], tokenizer: Any, max_length: int) -> Dataset:
    def _tokenize(batch: dict[str, list[Any]]) -> Any:
        return tokenizer(
            batch["question"],
            batch["context"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    raw = Dataset.from_dict(
        {
            "question": [p.question for p in pairs],
            "context": [p.context for p in pairs],
            "label": [p.label for p in pairs],
        }
    )
    return raw.map(_tokenize, batched=True)


def build_model_and_tokenizer(cfg: TrainConfig) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=cfg.num_labels
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
    )
    model = get_peft_model(base_model, lora_config)
    return model, tokenizer


def _compute_metrics_for_trainer(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics = compute_metrics(list(labels), list(predictions))
    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "false_positive_rate": metrics.false_positive_rate,
    }


def train(cfg: TrainConfig, train_pairs: list[PairExample], val_pairs: list[PairExample]) -> Path:
    model, tokenizer = build_model_and_tokenizer(cfg)
    train_ds = _to_hf_dataset(train_pairs, tokenizer, cfg.max_length)
    val_ds = _to_hf_dataset(val_pairs, tokenizer, cfg.max_length)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        seed=cfg.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics_for_trainer,
    )
    trainer.train()

    adapter_path = Path(cfg.output_dir) / "adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    return adapter_path
