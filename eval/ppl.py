"""Perplexity evaluation entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
import tqdm
from datasets import load_dataset
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer

def evaluate_ppl(
    model: PreTrainedModel,
    enc: PreTrainedTokenizer,
    cfg: Dict[str, Any],
) -> None:
    eval_cfg = cfg.get("evaluation", {})
    dataset_name = eval_cfg.get("dataset", "wikitext2").lower()
    split = eval_cfg.get("split", "test")
    seqlen = int(eval_cfg.get("max_length", 2048))

    if dataset_name not in {"wikitext", "wikitext2"}:
        raise NotImplementedError(
            f"Unsupported eval dataset: {dataset_name!r}. "
            "Only wikitext/wikitext2 is supported now."
        )

    print(f"[myawq] Loading WikiText-2 ({split}) for PPL evaluation...")
    testset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    testenc = enc("\n\n".join(testset["text"]), return_tensors="pt")

    if hasattr(model, "get_input_embeddings"):
        model_device = model.get_input_embeddings().weight.device
    else:
        model_device = next(model.parameters()).device

    input_ids = testenc.input_ids.to(model_device)
    nsamples = input_ids.numel() // seqlen
    if nsamples == 0:
        raise RuntimeError(
            f"Not enough tokens ({input_ids.numel()}) for max_length={seqlen}."
        )

    model = model.eval()
    nlls = []
    loss_fn = nn.CrossEntropyLoss()
    for i in tqdm.tqdm(range(nsamples), desc="evaluating ppl"):
        batch = input_ids[:, (i * seqlen) : ((i + 1) * seqlen)].to(model_device)
        with torch.no_grad():
            lm_logits = model(batch).logits # 未经过 softmax 归一化的 log 概率
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = input_ids[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:]
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        nlls.append(loss.float() * seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()
    results = {
        "dataset": "wikitext-2-raw-v1",
        "split": split,
        "max_length": seqlen,
        "num_segments": nsamples,
        "ppl": ppl,
    }
    print(f"[myawq] PPL: {ppl:.6f}")

    output_cfg = cfg.get("output", {})
    if output_cfg.get("save_metrics", True):
        output_dir = Path(output_cfg.get("dir", "results"))
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_name = output_cfg.get("metrics_name", "ppl_metrics.json")
        metrics_path = output_dir / metrics_name
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[myawq] Saved PPL metrics to: {metrics_path}")
