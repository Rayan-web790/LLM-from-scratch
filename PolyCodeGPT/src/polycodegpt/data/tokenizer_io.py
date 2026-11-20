"""Tokenizer loading and integrity checks for the data pipeline.

The tokenizer is loaded read-only from its v1.1 artifact. Nothing here fits,
modifies or rewrites the tokenizer: it verifies the artifact's SHA-256, the
vocabulary size and the special-token ID contract before any text is encoded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from tokenizers import Tokenizer

from ..model.config import PROJECT_ROOT

TOKENIZER_VERSION = "1.1.0"
TOKENIZER_SHA256 = "0459f8556a6f79ca1ba1c3dc3f76c26ae2e55653ebbf97beaa9bd23467358a03"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "artifacts" / "polycodegpt_tokenizer_v1_1.json"
TOKENIZER_METADATA_PATH = PROJECT_ROOT / "tokenizer" / "artifacts" / "tokenizer_metadata_v1_1.json"

#: The fixed special-token contract of tokenizer v1.1 (IDs 0-10).
SPECIAL_TOKEN_IDS: Mapping[str, int] = {
    "<|pad|>": 0,
    "<|unk|>": 1,
    "<|eos|>": 2,
    "<|code|>": 3,
    "<|docstring|>": 4,
    "<|text|>": 5,
    "<|lang:python|>": 6,
    "<|lang:javascript|>": 7,
    "<|lang:en|>": 8,
    "<|lang:fr|>": 9,
    "<|lang:ar|>": 10,
}

PAD_ID = SPECIAL_TOKEN_IDS["<|pad|>"]
UNK_ID = SPECIAL_TOKEN_IDS["<|unk|>"]
EOS_ID = SPECIAL_TOKEN_IDS["<|eos|>"]
CODE_ID = SPECIAL_TOKEN_IDS["<|code|>"]
DOCSTRING_ID = SPECIAL_TOKEN_IDS["<|docstring|>"]
TEXT_ID = SPECIAL_TOKEN_IDS["<|text|>"]
LANG_PYTHON_ID = SPECIAL_TOKEN_IDS["<|lang:python|>"]
LANG_JAVASCRIPT_ID = SPECIAL_TOKEN_IDS["<|lang:javascript|>"]
LANG_EN_ID = SPECIAL_TOKEN_IDS["<|lang:en|>"]

#: Token IDs that may begin a serialized example.
LANG_MARKER_IDS: Mapping[int, str] = {
    LANG_PYTHON_ID: "python",
    LANG_JAVASCRIPT_ID: "javascript",
    LANG_EN_ID: "en",
}


class TokenizerContractError(RuntimeError):
    """Raised when the tokenizer artifact violates the expected contract."""


def load_tokenizer(
    path: str | Path = TOKENIZER_PATH,
    expected_sha256: str | None = TOKENIZER_SHA256,
    vocab_size: int | None = None,
) -> Tokenizer:
    """Load the v1.1 tokenizer and verify its file hash and vocabulary contract."""
    from .integrity import sha256_file

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer artifact missing: {path}")

    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise TokenizerContractError(
            f"tokenizer SHA-256 mismatch: {actual} != expected {expected_sha256} ({path})"
        )

    tok = Tokenizer.from_file(str(path))

    if vocab_size is not None and tok.get_vocab_size() != vocab_size:
        raise TokenizerContractError(
            f"tokenizer vocab size {tok.get_vocab_size()} != configured {vocab_size}"
        )

    verify_special_tokens(tok)
    return tok


def verify_special_tokens(tok: Tokenizer) -> None:
    """Assert all 11 special tokens are registered with their recorded IDs."""
    vocab = tok.get_vocab()
    added = tok.get_added_tokens_decoder()

    missing = [t for t in SPECIAL_TOKEN_IDS if t not in vocab]
    if missing:
        raise TokenizerContractError(f"special tokens missing from vocabulary: {missing}")

    for token, expected_id in SPECIAL_TOKEN_IDS.items():
        actual_id = vocab[token]
        if actual_id != expected_id:
            raise TokenizerContractError(
                f"special token {token} has ID {actual_id}, expected {expected_id}"
            )
        if actual_id not in added:
            raise TokenizerContractError(f"special token {token} is not registered as special")
        encoded = tok.encode(token, add_special_tokens=False).ids
        if encoded != [expected_id]:
            raise TokenizerContractError(
                f"special token {token} does not encode to its own ID: {encoded}"
            )

    ids = [vocab[t] for t in SPECIAL_TOKEN_IDS]
    if len(set(ids)) != len(ids):
        raise TokenizerContractError(f"duplicate special token IDs: {ids}")


def load_tokenizer_metadata(path: str | Path = TOKENIZER_METADATA_PATH) -> dict:
    """Read the tokenizer metadata JSON (version, vocab size, special tokens)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer metadata missing: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def encode(tok: Tokenizer, text: str) -> list[int]:
    """Encode text without any post-processing special tokens.

    The serializer is solely responsible for structural markers, so nothing is
    added here. Using ``add_special_tokens=False`` guarantees that even if the
    tokenizer artifact later gains a post-processor, no extra EOS can appear.
    """
    return tok.encode(text, add_special_tokens=False).ids
