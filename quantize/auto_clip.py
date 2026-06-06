"""Weight clipping search for decoder layers."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

from quantize.auto_scale import resolve_submodule
from quantize.quantizer import pseudo_quantize_tensor
from utils.device import empty_cache

__all__ = ["ClipRecord", "apply_clip_records", "optimize_decoder_layer_clips"]

_SKIP_LINEAR_KEYS = ("q_proj", "k_proj", "query", "key", "Wqkv")


@dataclass
class ClipRecord:
    linear_path: str
    max_val: torch.Tensor


@torch.no_grad()
def _search_linear_clip(
    weight: torch.Tensor,
    input_feat: torch.Tensor,
    n_bits: int,
    group_size: int,
    symmetric: bool,
    n_grid: int = 20,
    max_shrink: float = 0.5,
    n_sample_token: int = 512,
) -> torch.Tensor:
    assert weight.dim() == 2
    flat_input = input_feat.reshape(-1, input_feat.shape[-1])
    flat_input = flat_input.reshape(1, flat_input.shape[0], -1, group_size)
    stride = max(1, flat_input.shape[1] // n_sample_token)
    flat_input = flat_input[:, ::stride]

    grouped_weight = weight.reshape(weight.shape[0], 1, -1, group_size)
    batch_size = 256 if grouped_weight.shape[0] % 256 == 0 else 64
    if grouped_weight.shape[0] % batch_size != 0:
        raise ValueError(
            f"Output channels {grouped_weight.shape[0]} not divisible by batch size {batch_size}."
        )

    best_vals: List[torch.Tensor] = []
    flat_input = flat_input.to(grouped_weight.device)

    for start in range(0, grouped_weight.shape[0], batch_size):
        chunk = grouped_weight[start : start + batch_size]
        chunk_fp32 = chunk.float()
        input_fp32 = flat_input.float()

        org_max = chunk_fp32.abs().amax(dim=-1, keepdim=True)
        best_max = org_max.to(dtype=chunk.dtype)
        min_err = torch.full(
            org_max.shape, 1e9, device=org_max.device, dtype=torch.float32
        )
        org_out = (input_fp32 * chunk_fp32).sum(dim=-1)

        for step in range(int(max_shrink * n_grid)):
            max_val = org_max * (1 - step / n_grid)
            clipped = torch.clamp(chunk_fp32, -max_val, max_val)
            qweight = pseudo_quantize_tensor(
                clipped.reshape(-1, group_size),
                n_bits=n_bits,
                group_size=group_size,
                symmetric=symmetric,
            ).qweight.reshape_as(clipped)
            cur_out = (input_fp32 * qweight.float()).sum(dim=-1)
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_err.shape)

            improved = err < min_err
            min_err = torch.where(improved, err, min_err)
            best_max = torch.where(
                improved, max_val.to(dtype=best_max.dtype), best_max
            )

        best_vals.append(best_max)

    result = torch.cat(best_vals, dim=0).squeeze(1)
    del flat_input
    gc.collect()
    empty_cache()
    return result


@torch.no_grad()
def apply_clip_records(
    layer: nn.Module,
    records: List[ClipRecord],
    device: torch.device | None = None,
) -> None:
    for record in records:
        linear = resolve_submodule(layer, record.linear_path)
        if device is not None and linear.weight.device != device:
            linear.to(device)
        max_val = record.max_val.to(linear.weight.device, dtype=linear.weight.dtype)
        org_shape = linear.weight.shape
        reshaped = linear.weight.data.reshape(*max_val.shape[:2], -1)
        linear.weight.data = torch.clamp(reshaped, -max_val, max_val).reshape(org_shape)


@torch.no_grad()
def optimize_decoder_layer_clips(
    layer: nn.Module,
    input_feat: Dict[str, torch.Tensor],
    n_bits: int,
    group_size: int,
    symmetric: bool,
    device: torch.device,
) -> List[ClipRecord]:
    records: List[ClipRecord] = []
    for name, module in layer.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(token in name for token in _SKIP_LINEAR_KEYS):
            continue

        max_val = _search_linear_clip(
            module.weight.data,
            input_feat[name],
            n_bits=n_bits,
            group_size=group_size,
            symmetric=symmetric,
        )
        record = ClipRecord(linear_path=name, max_val=max_val.detach().cpu())
        apply_clip_records(layer, [record], device=device)
        records.append(record)

    return records
