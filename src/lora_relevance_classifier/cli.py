"""`lora-relevance-classifier train`, `evaluate`, and `compare` subcommands."""

from __future__ import annotations

import argparse
from pathlib import Path

from lora_relevance_classifier.baseline import word_overlap_predict
from lora_relevance_classifier.config import TrainConfig
from lora_relevance_classifier.dataset import load_pairs
from lora_relevance_classifier.metrics import compute_metrics


def _cmd_train(args: argparse.Namespace) -> int:
    # Deferred: these imports pull in torch/transformers.
    from lora_relevance_classifier.train import train

    cfg = TrainConfig(
        base_model=args.base_model, output_dir=args.output_dir, num_epochs=args.epochs
    )
    train_pairs = load_pairs(args.train_file)
    val_pairs = load_pairs(args.val_file)
    adapter_path = train(cfg, train_pairs, val_pairs)
    print(f"Adapter saved to {adapter_path}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    pairs = load_pairs(args.val_file)
    labels = [p.label for p in pairs]

    baseline_predictions = [word_overlap_predict(p.question, p.context) for p in pairs]
    baseline_metrics = compute_metrics(labels, baseline_predictions)
    print("word-overlap baseline:", baseline_metrics)

    if args.adapter_dir:
        from lora_relevance_classifier.evaluate import evaluate_pairs, load_adapter_model

        cfg = TrainConfig(base_model=args.base_model)
        model, tokenizer = load_adapter_model(Path(args.adapter_dir), cfg)
        adapter_metrics = evaluate_pairs(model, tokenizer, pairs, cfg.max_length)
        print("LoRA adapter:       ", adapter_metrics)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lora-relevance-classifier")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Fine-tune the LoRA adapter")
    train_p.add_argument("--train-file", type=Path, default=Path("data/train.jsonl"))
    train_p.add_argument("--val-file", type=Path, default=Path("data/val.jsonl"))
    train_p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
    train_p.add_argument("--output-dir", default="runs/lora-adapter")
    train_p.add_argument("--epochs", type=int, default=5)
    train_p.set_defaults(func=_cmd_train)

    compare_p = sub.add_parser("compare", help="Compare the baseline against a trained adapter")
    compare_p.add_argument("--val-file", type=Path, default=Path("data/val.jsonl"))
    compare_p.add_argument("--adapter-dir", default=None)
    compare_p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
    compare_p.set_defaults(func=_cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
