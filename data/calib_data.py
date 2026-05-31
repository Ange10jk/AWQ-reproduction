"""Calibration dataset utilities."""

from __future__ import annotations # lazy compute

from dataclasses import dataclass # generates class template automatically 
from typing import List 

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer

@dataclass
class CalibSample:
    input_ids: torch.Tensor


def get_calib_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer | None = None,
    num_samples: int = 512,
    seq_len: int = 512,
    seed: int = 42,
) -> List[CalibSample]:
    """Build calibration samples for AWQ.

    Supported datasets:
    - ``pileval``: Pile validation backup (official AWQ default), concat then split by ``seq_len``.
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for calibration dataset construction")

    name = dataset_name.lower()
    if name == "pileval":
        return _build_pileval_samples(tokenizer, num_samples, seq_len, seed)

    raise NotImplementedError(
        f"Unsupported calibration dataset: {dataset_name!r}. "
        "Choose from: wikitext, wikitext2, pileval."
    )


def _build_pileval_samples(
    tokenizer: PreTrainedTokenizer,
    num_samples: int,
    seq_len: int,
    seed: int,
) -> List[CalibSample]:
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    dataset = dataset.shuffle(seed=seed)

    encoded_lines: List[torch.Tensor] = []
    for row in dataset:
        line = row["text"].strip()
        if not line:
            continue
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > seq_len:
            continue
        encoded_lines.append(torch.tensor([line_encoded]))
        if len(encoded_lines) == num_samples:
            break

    if not encoded_lines:
        raise RuntimeError("No valid Pileval lines found for calibration.")

    cat_samples = torch.cat(encoded_lines, dim=1)
    n_split = cat_samples.shape[1] // seq_len
    if n_split == 0:
        raise RuntimeError(
            f"Not enough Pileval tokens to form one block of seq_len={seq_len}."
        )

    print(f" * Split Pileval into {n_split} blocks with seq_len={seq_len}...")
    samples = []
    for i in range(n_split):
        chunk = cat_samples[:, i * seq_len : (i + 1) * seq_len]
        samples.append(
            CalibSample(
                input_ids=chunk,
            )
        )

    print(f" * Successfully built {len(samples)} samples.")
    return samples
