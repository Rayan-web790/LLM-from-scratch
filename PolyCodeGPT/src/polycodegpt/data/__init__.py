"""Tokenized data pipeline for PolyCodeGPT.

Public entry points:

* :func:`prepare_dataset` — serialize, tokenize and write the three token streams
* :class:`TokenBlockDataset` — memory-mapped next-token block dataset
* :func:`build_dataloader` — Windows-safe DataLoader factory
* :func:`load_tokenizer` — verified read-only tokenizer loader
"""

from .dataset import TokenBlockDataset, build_dataloader, open_split
from .integrity import sha256_file, verify_outputs
from .prepare import prepare_dataset
from .stream import (
    BYTE_ORDER,
    DTYPE_NAME,
    STREAM_DTYPE,
    TokenStreamWriter,
    block_geometry,
    stream_token_count,
)
from .tokenizer_io import EOS_ID, SPECIAL_TOKEN_IDS, encode, load_tokenizer, verify_special_tokens

__all__ = [
    "BYTE_ORDER",
    "DTYPE_NAME",
    "EOS_ID",
    "SPECIAL_TOKEN_IDS",
    "STREAM_DTYPE",
    "TokenBlockDataset",
    "TokenStreamWriter",
    "block_geometry",
    "build_dataloader",
    "encode",
    "load_tokenizer",
    "open_split",
    "prepare_dataset",
    "sha256_file",
    "stream_token_count",
    "verify_outputs",
    "verify_special_tokens",
]
