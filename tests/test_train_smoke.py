"""The one test that actually runs a real LoRA fine-tune end to end, on CPU, using a tiny
proxy model -- not this project's real target model (Qwen2.5-0.5B, which needs a GPU to
fine-tune in a reasonable time; see docs/explained and notebooks/train_on_colab.md). This
proves the train/evaluate code is mechanically correct: a real peft.get_peft_model(), a real
transformers.Trainer.train() loop, a real adapter saved to disk and reloaded. It is explicitly
NOT a claim that the real target model has been fine-tuned -- see README's status section.
"""

from __future__ import annotations

from lora_relevance_classifier.config import LoraSettings, TrainConfig
from lora_relevance_classifier.dataset import PairExample
from lora_relevance_classifier.evaluate import evaluate_pairs, load_adapter_model
from lora_relevance_classifier.train import train

TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


def _make_pairs(n: int) -> list[PairExample]:
    pairs = []
    for i in range(n):
        pairs.append(
            PairExample(f"question {i}", f"relevant context {i}", 1, f"pos-{i}", "positive")
        )
        pairs.append(
            PairExample(f"question {i}", f"unrelated text {i}", 0, f"neg-{i}", "hard_negative")
        )
    return pairs


def test_lora_fine_tune_runs_end_to_end_on_a_tiny_model(tmp_path):
    cfg = TrainConfig(
        base_model=TINY_MODEL,
        lora=LoraSettings(r=4, alpha=8, dropout=0.0, target_modules=["c_attn"]),
        num_epochs=1,
        batch_size=4,
        max_length=32,
        output_dir=str(tmp_path / "run"),
    )
    train_pairs = _make_pairs(8)
    val_pairs = _make_pairs(2)

    adapter_path = train(cfg, train_pairs, val_pairs)

    assert adapter_path.exists()
    assert (adapter_path / "adapter_config.json").exists()

    model, tokenizer = load_adapter_model(adapter_path, cfg)
    metrics = evaluate_pairs(model, tokenizer, val_pairs, cfg.max_length)

    assert 0.0 <= metrics.accuracy <= 1.0
