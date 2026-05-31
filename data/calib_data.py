"""Calibration dataset utilities."""

from __future__ import annotations # lazy compute

from dataclasses import dataclass # generates class template automatically 
from typing import List 

import torch
from datasets import load_dataset 
from transformers import PreTrainedTokenizer

@dataclass
class CalibSample:
    text: str
    input_ids: torch.Tensor | None = None
 

def get_calib_dataset(
    dataset_name: str,
    tokenizer : PreTrainedTokenizer | None = None,
    num_samples: int = 512,
    seq_len: int = 512,
    seed: int = 42,
) -> List[CalibSample]:
    """Build calibration samples for AWQ.

    - Load dataset (e.g., WikiText-2 subset) with deterministic seed.
    - Tokenize and truncate to `seq_len`.
    - Return representative text samples for activation collection.
    """
    if dataset_name.lower() == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    elif dataset_name.lower() == 'pileval':
        dataset = load_dataset('mit-han-lab/pile-val-backup', split='validation')
    else:
        raise NotImplementedError
    # Please note the way Hugging face datasets are arranged
    dataset = dataset.filter(lambda x: len(x['text'].strip() > 0))
    dataset = dataset.shuffle(seed=seed)
    
    samples = []
    max_samples = min(num_samples, len(dataset))

    print(f" * Building {max_samples} calibration samples with seq_len={seq_len}...") 

    for i in range(max_samples):
        text = dataset[i]['text']

        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=seq_len,
            return_tensors="pt",  # returns PyTorch Tensor
        )

        sample = CalibSample(
            text=text,
            input_ids=encoded['input_ids'],
            attenion_mask=encoded['attention_masks'],
        )
        samples.append(sample)

    print(f' * Successfully built {len(samples)} samples.')
    return samples
