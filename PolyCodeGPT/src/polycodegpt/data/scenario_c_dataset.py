import json
from array import array
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class ScenarioCCompletionDataset(Dataset):
    """
    Record-aware Scenario C dataset.

    Inputs contain the serialized record through the <|code|> marker.
    Targets supervise code tokens and EOS only.
    """

    def __init__(
        self,
        jsonl_path,
        tokenizer_path,
        context_length=512,
    ):
        self.path = Path(jsonl_path)
        self.context_length = int(context_length)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        self.pad_id = self.tokenizer.token_to_id("<|pad|>")
        self.code_id = self.tokenizer.token_to_id("<|code|>")
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")

        if None in (self.pad_id, self.code_id, self.eos_id):
            raise ValueError("Required special token is missing")

        self.offsets = array("Q")
        self.skipped_invalid_json = 0
        self.skipped_oversize = 0

        offset = 0

        with self.path.open("rb") as f:
            for line in f:
                line_offset = offset
                offset += len(line)

                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    self.skipped_invalid_json += 1
                    continue

                if not isinstance(record, dict):
                    continue

                if not record.get("fits_512", False):
                    self.skipped_oversize += 1
                    continue

                if int(record.get("token_count", 0)) > self.context_length:
                    self.skipped_oversize += 1
                    continue

                if not isinstance(record.get("text"), str):
                    continue

                self.offsets.append(line_offset)

        if not self.offsets:
            raise RuntimeError(f"No eligible records found in {self.path}")

        self._file = None

    def __len__(self):
        return len(self.offsets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __getitem__(self, index):
        if self._file is None:
            self._file = self.path.open("rb")

        self._file.seek(self.offsets[index])
        record = json.loads(self._file.readline())

        encoded = self.tokenizer.encode(record["text"])
        ids = encoded.ids

        if len(ids) > self.context_length:
            raise ValueError(
                f"Record exceeds context: {record.get('record_id')}"
            )

        if record.get("token_count") != len(ids):
            raise ValueError(
                f"Token count mismatch: {record.get('record_id')}"
            )

        code_positions = [
            i for i, token_id in enumerate(ids)
            if token_id == self.code_id
        ]

        if len(code_positions) != 1:
            raise ValueError(
                f"Expected one code marker: {record.get('record_id')}"
            )

        if not ids or ids[-1] != self.eos_id:
            raise ValueError(
                f"Missing EOS: {record.get('record_id')}"
            )

        code_position = code_positions[0]

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        targets = torch.tensor(ids[1:], dtype=torch.long)

        # Target at index code_position predicts the first code token.
        targets[:code_position] = -100

        if not (targets != -100).any():
            raise ValueError(
                f"No supervised targets: {record.get('record_id')}"
            )

        return input_ids, targets


def scenario_c_collate(batch, pad_id):
    """Right-pad inputs and ignore padded target positions."""

    batch_size = len(batch)
    max_length = max(input_ids.numel() for input_ids, _ in batch)

    input_ids = torch.full(
        (batch_size, max_length),
        int(pad_id),
        dtype=torch.long,
    )

    targets = torch.full(
        (batch_size, max_length),
        -100,
        dtype=torch.long,
    )

    for i, (sample_input, sample_targets) in enumerate(batch):
        length = sample_input.numel()
        input_ids[i, :length] = sample_input
        targets[i, :length] = sample_targets

    return input_ids, targets
