"""Buffered writer for binary token streams.

A stream is a flat, little-endian ``uint16`` array of token IDs with no header
or padding. uint16 is safe because the vocabulary is 16,000 (< 65,536), a
property checked for every batch written and re-checked when a stream is read.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

STREAM_DTYPE = np.dtype("<u2")
DTYPE_NAME = "uint16"
BYTE_ORDER = "little"
BITS_PER_TOKEN = 16

#: Tokens buffered in RAM before a flush (1 Mi tokens = 2 MiB).
DEFAULT_BUFFER_TOKENS = 1 << 20


def dtype_is_safe(vocab_size: int) -> bool:
    """uint16 can represent every ID in ``[0, vocab_size)`` only if vocab fits."""
    return 0 < vocab_size <= np.iinfo(STREAM_DTYPE).max + 1


class TokenStreamWriter:
    """Append token IDs to a binary file, hashing and counting as it goes.

    Memory use is bounded by ``buffer_tokens`` regardless of stream length.
    """

    def __init__(
        self,
        path: str | Path,
        vocab_size: int,
        buffer_tokens: int = DEFAULT_BUFFER_TOKENS,
    ) -> None:
        if not dtype_is_safe(vocab_size):
            raise ValueError(
                f"vocab_size {vocab_size} does not fit in {DTYPE_NAME}; "
                "use a wider storage dtype"
            )
        if buffer_tokens <= 0:
            raise ValueError("buffer_tokens must be positive")

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.vocab_size = vocab_size
        self._buffer = np.empty(buffer_tokens, dtype=STREAM_DTYPE)
        self._filled = 0
        self._buffer_tokens = buffer_tokens
        self._hash = hashlib.sha256()
        self._token_count = 0
        self._max_id = -1
        self._closed = False
        self._fh = open(self.path, "wb")

    # ------------------------------------------------------------------ write
    def write(self, ids: Sequence[int] | Iterable[int]) -> int:
        """Append token IDs, validating range; returns the number of tokens written."""
        if self._closed:
            raise RuntimeError("writer is closed")

        array = np.asarray(ids, dtype=np.int64)
        if array.ndim != 1:
            raise ValueError(f"expected a 1-D token sequence, got shape {array.shape}")
        if array.size == 0:
            return 0

        lowest = int(array.min())
        highest = int(array.max())
        if lowest < 0 or highest >= self.vocab_size:
            bad = lowest if lowest < 0 else highest
            raise ValueError(
                f"token ID {bad} outside vocabulary [0, {self.vocab_size}) in {self.path.name}"
            )

        position = 0
        while position < array.size:
            room = self._buffer_tokens - self._filled
            chunk = array[position : position + room]
            self._buffer[self._filled : self._filled + chunk.size] = chunk
            self._filled += chunk.size
            position += chunk.size
            if self._filled == self._buffer_tokens:
                self._flush()

        self._token_count += int(array.size)
        if highest > self._max_id:
            self._max_id = highest
        return int(array.size)

    def _flush(self) -> None:
        if self._filled == 0:
            return
        payload = np.ascontiguousarray(self._buffer[: self._filled], dtype=STREAM_DTYPE).tobytes()
        self._fh.write(payload)
        self._hash.update(payload)
        self._filled = 0

    # ------------------------------------------------------------------ close
    def close(self) -> dict:
        """Flush, close, and return measured statistics for the finished stream."""
        if not self._closed:
            self._flush()
            self._fh.close()
            self._closed = True
        size = self.path.stat().st_size
        expected = self._token_count * (BITS_PER_TOKEN // 8)
        if size != expected:
            raise RuntimeError(f"stream size {size} != {expected} bytes for {self._token_count} tokens")
        return {
            "tokens": self._token_count,
            "file": self.path.name,
            "file_size_bytes": size,
            "sha256": self._hash.hexdigest(),
            "max_token_id": self._max_id,
            "dtype": DTYPE_NAME,
            "byte_order": BYTE_ORDER,
        }

    def __enter__(self) -> "TokenStreamWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        elif not self._closed:
            self._fh.close()
            self._closed = True


def read_stream(path: str | Path, offset: int = 0, count: int = -1) -> np.ndarray:
    """Read a slice of a stream (small, bounded reads; not for whole-file loads)."""
    return np.fromfile(path, dtype=STREAM_DTYPE, count=count, offset=offset * STREAM_DTYPE.itemsize)


def stream_token_count(path: str | Path) -> int:
    """Number of tokens in a stream, derived from its size."""
    size = Path(path).stat().st_size
    if size % STREAM_DTYPE.itemsize != 0:
        raise ValueError(f"stream {path} has {size} bytes, not a multiple of the token size")
    return size // STREAM_DTYPE.itemsize


def block_geometry(
    token_count: int,
    context_length: int,
    stride: int | None = None,
) -> dict[str, int]:
    """Window geometry for ``context_length + 1`` windows stepping by ``stride``.

    The default stride is ``context_length``, so window ``i`` starts at
    ``i * stride`` and covers ``[i*stride, i*stride + context_length]``. Because
    window ``i`` ends exactly where window ``i+1`` starts, consecutive windows
    overlap by one token: the last token of window ``i`` is the first token of
    window ``i+1``.

    That single shared token is the point of the overlap. With a stride equal to
    the window length, the last token of a window could never appear as a target
    (no next token is inside the same window), so one prediction per boundary
    would be lost. Overlapping by one makes the target ranges
    ``[i*stride+1, i*stride+context_length]`` tile ``1 .. blocks*stride`` exactly:
    every next-token target is used exactly once, position 0 is context only, and
    the only unused tokens are the incomplete tail.
    """
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    window_length = context_length + 1
    stride = context_length if stride is None else stride
    if stride <= 0:
        raise ValueError("stride must be positive")
    if stride > window_length:
        raise ValueError(
            f"stride {stride} exceeds window length {window_length}; "
            "that would skip targets at every boundary"
        )

    if token_count < window_length:
        blocks = 0
        tokens_in_blocks = 0
    else:
        blocks = 1 + (token_count - window_length) // stride
        tokens_in_blocks = (blocks - 1) * stride + window_length

    return {
        "window_length": window_length,
        "window_stride": stride,
        "window_overlap": window_length - stride,
        "block_span": window_length,  # alias for readers that predate the stride change
        "blocks": blocks,
        "dropped_trailing_tokens": token_count - tokens_in_blocks,
        "tokens_in_blocks": tokens_in_blocks,
    }
