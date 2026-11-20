"""Memory-mapped block dataset and DataLoader factory.

A token stream is a flat uint16 array. Block ``i`` is the window
``[i * stride, i * stride + L + 1)`` where ``L`` is ``context_length`` and
``stride`` defaults to ``L``; the sample is ``input_ids = block[:-1]`` and
``targets = block[1:]``, so target ``t`` is always the token that follows input
``t``. Because ``stride == L``, consecutive blocks share exactly one token and no
next-token target is skipped at a block boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .stream import STREAM_DTYPE, block_geometry, stream_token_count


class TokenBlockDataset(Dataset):
    """Random-access view over a binary token stream.

    The stream is never loaded into RAM: it is opened as a read-only memory map
    on first access, so worker processes created by ``spawn`` only inherit the
    file path, not an open handle.
    """

    def __init__(
        self,
        bin_path: str | Path,
        context_length: int,
        vocab_size: int | None = None,
        verify_size: bool = True,
        stride: int | None = None,
    ) -> None:
        self.bin_path = Path(bin_path)
        if not self.bin_path.is_file():
            raise FileNotFoundError(f"token stream not found: {self.bin_path}")
        if context_length <= 0:
            raise ValueError("context_length must be positive")

        self.context_length = context_length
        self.vocab_size = vocab_size
        self._mm: np.memmap | None = None

        token_count = stream_token_count(self.bin_path)
        geometry = block_geometry(token_count, context_length, stride)
        self.token_count = token_count
        self.num_blocks = geometry["blocks"]
        self.dropped_trailing_tokens = geometry["dropped_trailing_tokens"]
        self.window_length = geometry["window_length"]
        self.window_stride = geometry["window_stride"]
        self.window_overlap = geometry["window_overlap"]
        self.block_span = geometry["window_length"]

        if verify_size and self.token_count == 0:
            raise ValueError(f"token stream is empty: {self.bin_path}")

    # ------------------------------------------------------------- memmap I/O
    @property
    def mm(self) -> np.memmap:
        """Lazily opened read-only memory map (not pickled to workers)."""
        if self._mm is None:
            self._mm = np.memmap(self.bin_path, dtype=STREAM_DTYPE, mode="r")
        return self._mm

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_mm"] = None
        return state

    def close(self) -> None:
        if self._mm is not None:
            del self._mm
            self._mm = None

    # ---------------------------------------------------------------- samples
    def __len__(self) -> int:
        return self.num_blocks

    def block(self, index: int) -> np.ndarray:
        """Return the raw ``context_length + 1`` token window as a copy."""
        if index < 0:
            index += self.num_blocks
        if not 0 <= index < self.num_blocks:
            raise IndexError(f"block index {index} out of range [0, {self.num_blocks})")
        start = index * self.window_stride
        window = np.array(self.mm[start : start + self.window_length], dtype=np.int64, copy=True)
        if window.size != self.window_length:
            raise RuntimeError(
                f"short read from {self.bin_path.name} at block {index}: {window.size} tokens"
            )
        return window

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.block(index)
        input_ids = torch.from_numpy(window[:-1])  # [L]
        targets = torch.from_numpy(window[1:])  # [L]
        return input_ids, targets


def build_dataloader(
    dataset: TokenBlockDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    seed: int = 42,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader with Windows-safe defaults.

    ``pin_memory`` defaults to "only when CUDA is available". ``num_workers``
    defaults to 0 because on Windows each worker re-imports torch (slow start-up)
    and memory-mapped reads in the main process are already fast.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
        persistent_workers=False if num_workers == 0 else True,
    )


def open_split(
    output_dir: str | Path,
    split: str,
    manifest: dict[str, Any],
    verify: bool = True,
) -> TokenBlockDataset:
    """Open one split from a prepared output directory, checking the manifest."""
    output_dir = Path(output_dir)
    info = manifest["splits"][split]
    path = output_dir / info["file"]
    stride = manifest.get("window_stride")
    dataset = TokenBlockDataset(
        path,
        context_length=int(manifest["context_length"]),
        vocab_size=int(manifest["tokenizer"]["vocab_size"]),
        verify_size=verify,
        stride=None if stride is None else int(stride),
    )
    if verify:
        if stride is not None and dataset.window_stride != int(stride):
            raise ValueError(
                f"{split}: dataset stride {dataset.window_stride}, manifest says {stride}"
            )
        if dataset.token_count != info["tokens"]:
            raise ValueError(
                f"{split}: stream has {dataset.token_count} tokens, manifest says {info['tokens']}"
            )
        if dataset.num_blocks != info["blocks"]:
            raise ValueError(
                f"{split}: {dataset.num_blocks} blocks, manifest says {info['blocks']}"
            )
    return dataset
