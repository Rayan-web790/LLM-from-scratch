from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polycodegpt.model.config import ModelConfig

from tokenizers import Tokenizer
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "artifacts" / "polycodegpt_tokenizer_v1_1.json"


def test_config_loads_from_yaml():
    cfg = ModelConfig.from_yaml(PROJECT_ROOT / "configs" / "model.yaml")
    assert cfg.vocab_size == 16000
    assert cfg.d_model == 384
    assert cfg.n_layers == 6
    assert cfg.n_heads == 6
    assert cfg.ffn_dim == 1536
    assert cfg.context_length == 512
    assert cfg.dropout == 0.1
    assert cfg.tie_embeddings is True
    assert cfg.head_dim == 64


def test_invalid_dimensions_rejected():
    with pytest.raises(ValueError):
        ModelConfig(d_model=384, n_heads=5)
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=0)
    with pytest.raises(ValueError):
        ModelConfig(context_length=0)
    with pytest.raises(ValueError):
        ModelConfig(dropout=1.0)


def test_tokenizer_vocab_compat():
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    cfg = ModelConfig.from_yaml(PROJECT_ROOT / "configs" / "model.yaml")
    cfg.validate_tokenizer_vocab(tok.get_vocab_size())
    with pytest.raises(ValueError):
        cfg.validate_tokenizer_vocab(cfg.vocab_size + 1)


def test_param_count_calculated():
    cfg = ModelConfig.small()
    n = cfg.estimate_parameters()
    assert isinstance(n, int) and n > 0
    cfg2 = ModelConfig.from_yaml(PROJECT_ROOT / "configs" / "model.yaml")
    assert cfg2.estimate_parameters() > 1_000_000


def test_small_config():
    c = ModelConfig.small()
    assert c.vocab_size == 32 and c.d_model == 16

