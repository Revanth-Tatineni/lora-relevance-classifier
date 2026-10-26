from __future__ import annotations

from lora_relevance_classifier.config import LoraSettings, TrainConfig


def test_default_config_has_sane_values():
    cfg = TrainConfig()
    assert cfg.lora.r > 0
    assert 0 < cfg.learning_rate < 1
    assert cfg.num_epochs >= 1


def test_lora_target_modules_default_is_attention_projections():
    cfg = TrainConfig()
    assert "q_proj" in cfg.lora.target_modules
    assert "v_proj" in cfg.lora.target_modules


def test_config_is_overridable():
    cfg = TrainConfig(base_model="some/other-model", lora=LoraSettings(r=16))
    assert cfg.base_model == "some/other-model"
    assert cfg.lora.r == 16
