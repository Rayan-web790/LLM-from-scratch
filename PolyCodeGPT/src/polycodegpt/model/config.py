"""Model configuration for PolyCodeGPT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"


@dataclass
class ModelConfig:
    """Validated configuration for the decoder-only Transformer.

    All dimensions are explicit; nothing is inferred from data.
    """

    vocab_size: int = 16000
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    ffn_dim: int = 1536
    context_length: int = 512
    dropout: float = 0.1
    tie_embeddings: bool = True
    layer_norm_eps: float = 1e-5
    init_std: float = 0.02
    architecture: str = "decoder_only_transformer"

    def __post_init__(self) -> None:
        if self.architecture != "decoder_only_transformer":
            raise ValueError(f"unsupported architecture {self.architecture!r}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.ffn_dim <= 0:
            raise ValueError(f"ffn_dim must be positive, got {self.ffn_dim}")
        if self.context_length <= 0:
            raise ValueError(f"context_length must be positive, got {self.context_length}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0,1), got {self.dropout}")
        if self.layer_norm_eps <= 0:
            raise ValueError(f"layer_norm_eps must be positive, got {self.layer_norm_eps}")
        if self.init_std <= 0:
            raise ValueError(f"init_std must be positive, got {self.init_std}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def validate_tokenizer_vocab(self, tokenizer_vocab_size: int) -> None:
        """Raise if the tokenizer vocabulary does not match the model."""
        if int(tokenizer_vocab_size) != self.vocab_size:
            raise ValueError(
                f"tokenizer vocab_size {tokenizer_vocab_size} != model vocab_size {self.vocab_size}"
            )

    def estimate_parameters(self) -> int:
        """Analytical parameter count (respects tie_embeddings)."""
        d = self.d_model
        v = self.vocab_size
        h = self.n_heads
        f = self.ffn_dim
        ctx = self.context_length
        n = self.n_layers
        # per layer
        attn = (3 * (d * d + d)) + (d * d + d)  # Q,K,V + out, all with bias
        ffn = (d * f + f) + (f * d + d)
        norms = 4 * d  # two LayerNorms: weight+ bias each
        per_layer = attn + ffn + norms
        total = v * d + ctx * d + n * per_layer + 2 * d  # token + pos + layers + final LN
        if not self.tie_embeddings:
            total += v * d
        return total

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "ModelConfig":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"model config not found: {path}")
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if "model" not in raw or not isinstance(raw["model"], Mapping):
            raise ValueError("model.yaml must contain a 'model' mapping")
        m = raw["model"]
        mapping = {
            "vocab_size": m.get("vocab_size"),
            "d_model": m.get("d_model"),
            "n_layers": m.get("n_layers"),
            "n_heads": m.get("n_heads"),
            "ffn_dim": m.get("ffn_dim"),
            "context_length": m.get("context_length"),
            "dropout": m.get("dropout"),
            "tie_embeddings": m.get("tie_embeddings"),
            "architecture": m.get("architecture", "decoder_only_transformer"),
        }
        mapping = {k: v for k, v in mapping.items() if v is not None}
        if "tokenizer" in raw and isinstance(raw["tokenizer"], Mapping):
            tv = raw["tokenizer"].get("vocab_size")
            if tv is not None and "vocab_size" not in mapping:
                mapping["vocab_size"] = tv
        return cls(**mapping)

    @classmethod
    def small(cls, **overrides: Any) -> "ModelConfig":
        """A tiny config for unit tests."""
        base = dict(vocab_size=32, d_model=16, n_layers=2, n_heads=4, ffn_dim=32,
                    context_length=8, dropout=0.0, tie_embeddings=True)
        base.update(overrides)
        return cls(**base)