from __future__ import annotations

from lora_relevance_classifier.build_dataset import main as build_dataset_main
from lora_relevance_classifier.dataset import load_pairs


def test_build_dataset_cli_writes_train_and_val_files(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            [
                '{"id": "a", "question": "q1", "answer": "x", "evidence": ["e1"]}',
                '{"id": "b", "question": "q2", "answer": "y", "evidence": ["e2"]}',
                '{"id": "c", "question": "q3", "answer": "z", "evidence": ["e3"]}',
                '{"id": "d", "question": "q4", "answer": "w", "evidence": ["e4"]}',
                '{"id": "e", "question": "q5", "answer": null, "evidence": []}',
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    exit_code = build_dataset_main(["--source", str(source), "--out-dir", str(out_dir)])

    assert exit_code == 0
    train_pairs = load_pairs(out_dir / "train.jsonl")
    val_pairs = load_pairs(out_dir / "val.jsonl")
    assert len(train_pairs) > 0
    assert len(val_pairs) > 0
