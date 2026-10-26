"""Builds this project's train/val splits from rag-qa-api's answerability eval dataset.

    python -m lora_relevance_classifier.build_dataset \\
        --source D:/personal/rag-qa-api/evals/dataset.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lora_relevance_classifier.dataset import (
    build_pairs,
    load_source_questions,
    save_pairs,
    split_pairs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--hard-negatives-per-question", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    questions = load_source_questions(args.source)
    pairs = build_pairs(
        questions, hard_negatives_per_question=args.hard_negatives_per_question, seed=args.seed
    )
    train, val = split_pairs(pairs, val_fraction=args.val_fraction, seed=args.seed)

    save_pairs(train, args.out_dir / "train.jsonl")
    save_pairs(val, args.out_dir / "val.jsonl")

    print(
        f"{len(questions)} source questions -> {len(pairs)} pairs "
        f"({len(train)} train, {len(val)} val)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
