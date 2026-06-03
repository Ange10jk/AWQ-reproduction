"""Tensor quantization primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QuantResult:
    qweight: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor | None


def pseudo_quantize_tensor(
    w: torch.Tensor,
    n_bits: int = 4,
    group_size: int = 128,
    symmetric: bool = False,
    inplace: bool = False
) -> QuantResult:
    """Group-wise pseudo quantization.

    - Reshape by group_size.
    - Compute per-group min/max (or absmax for symmetric).
    - Derive scale / zero-point.
    - Quantize + dequantize (fake quant) and return buffers.
    """
    org_w_shape = w.shape
    assert org_w_shape[-1] % group_size == 0
    w = w.reshape(-1, group_size)
    if not symmetric:
        max_val = w.amax(dim=1, keepdim=True) # do not return index like torch.max
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** n_bits - 1
        min_int = 0
        fake_scale = (max_val - min_val).clamp(min=1e-5) / max_int
        # min_int = round(min_val / scale) + zeros
        fake_zero_point = (-torch.round(min_val / fake_scale)).clamp_(min_int, max_int)
    else:
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bits - 1) - 1
        min_int = -(2 ** (n_bits - 1))
        fake_scale = max_val / max_int
        fake_zero_point = 0

    assert torch.isnan(fake_scale).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (
            w.div_(fake_scale).round_().add_(fake_zero_point).clamp_(min_int, max_int)
            ).sub_(fake_zero_point)
        ).mul_(fake_scale)
    else:
        w = (
            torch.clamp(torch.round(w / fake_scale) + fake_zero_point, min_int, max_int) - fake_zero_point
        ) * fake_scale

    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if isinstance(fake_zero_point, torch.Tensor):
        zero_point = fake_zero_point.view(w.shape[0], -1)
    else:
        zero_point = None

    return QuantResult(
        qweight=w,
        scale=fake_scale.view(w.shape[0], -1),
        zero_point=zero_point,
    )


@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    n_bits: int = 4,
    group_size: int = 128,
    symmetric: bool = False,
) -> None:
    """Apply fake int4 quantization to all decoder linear layers."""
    from quantize.pre_quant import get_blocks, get_named_linears

    for layer in get_blocks(model):
        for linear in get_named_linears(layer).values():
            result = pseudo_quantize_tensor(
                linear.weight.data,
                n_bits=n_bits,
                group_size=group_size,
                symmetric=symmetric,
                inplace=True,
            )
            linear.weight.data.copy_(result.qweight)
