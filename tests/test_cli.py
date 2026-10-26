from __future__ import annotations

from lora_relevance_classifier.cli import build_parser
from lora_relevance_classifier.dataset import PairExample, save_pairs


def test_train_command_parses_flags():
    parser = build_parser()
    args = parser.parse_args(
        ["train", "--base-model", "x/y", "--output-dir", "runs/t", "--epochs", "2"]
    )
    assert args.base_model == "x/y"
    assert args.output_dir == "runs/t"
    assert args.epochs == 2


def test_compare_command_runs_baseline_only_without_an_adapter(tmp_path, capsys):
    val_file = tmp_path / "val.jsonl"
    save_pairs(
        [
            PairExample("what is x", "x is the deduction", 1, "a", "positive"),
            PairExample("what is x", "unrelated text here", 0, "b", "hard_negative"),
        ],
        val_file,
    )

    parser = build_parser()
    args = parser.parse_args(["compare", "--val-file", str(val_file)])
    exit_code = args.func(args)

    assert exit_code == 0
    assert "word-overlap baseline" in capsys.readouterr().out
