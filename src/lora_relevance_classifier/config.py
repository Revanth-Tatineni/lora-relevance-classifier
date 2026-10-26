"""Training configuration for the LoRA fine-tune. A plain dataclass with documented defaults,
not a YAML-schema framework -- this project has exactly one training run shape, so a config
class earns its keep here more than a config *system* would.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoraSettings:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class TrainConfig:
    # The real target model from open-source-project-plan.md. Override to a tiny model (e.g.
    # "hf-internal-testing/tiny-random-gpt2") for a CPU smoke test -- see docs/explained.
    base_model: str = "Qwen/Qwen2.5-0.5B"
    num_labels: int = 2
    lora: LoraSettings = field(default_factory=LoraSettings)
    learning_rate: float = 2e-4
    num_epochs: int = 5
    batch_size: int = 8
    max_length: int = 256
    seed: int = 0
    output_dir: str = "runs/lora-adapter"
