"""Activation-aware scale search for decoder layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from quantize.quantizer import pseudo_quantize_tensor

__all__ = [
    "ScaleRecord",
    "apply_scale_records",
    "optimize_decoder_layer_scales",
    "resolve_submodule",
]


@dataclass
class ScaleSpec:
    """One scale-search target inside a decoder layer."""

    transform: str  # "ln_fc" | "fc_fc"
    prev_path: str
    target_paths: List[str]
    input_key: str
    inspect_path: Optional[str] = None
    pass_layer_kwargs: bool = False


@dataclass
class ScaleRecord:
    transform: str
    prev_path: str
    target_paths: List[str]
    scales: torch.Tensor


def resolve_submodule(root: nn.Module, path: str) -> nn.Module:
    node = root
    for part in path.split("."):
        node = getattr(node, part)
    return node

def _activation_scale(x: torch.Tensor) -> torch.Tensor:
    """每个channel平均激活强度 (hidden, )"""
    return x.abs().view(-1, x.shape[-1]).mean(0) 


@torch.no_grad()
def _scale_layernorm_to_linears(
    norm: nn.Module, linears: Iterable[nn.Linear], scales: torch.Tensor
) -> None:
    scales = scales.to(norm.weight.device, dtype=norm.weight.dtype)
    norm.weight.div_(scales)
    if getattr(norm, "bias", None) is not None:
        norm.bias.div_(scales)
    for linear in linears:
        linear.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def _scale_linear_to_linear(
    fc1: nn.Linear, fc2: nn.Linear, scales: torch.Tensor
) -> None:
    scales = scales.to(fc1.weight.device, dtype=fc1.weight.dtype)
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))
    fc2.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def _search_best_scales(
    inspect_module: nn.Module,
    target_linears: List[nn.Linear],
    activations: torch.Tensor,
    forward_kwargs: Dict,
    n_bits: int,
    group_size: int,
    symmetric: bool,
    n_grid: int = 20,
) -> torch.Tensor:
    """Grid-search channel scales by minimizing block output MSE."""

    def fake_quant(weight: torch.Tensor) -> torch.Tensor:
        return pseudo_quantize_tensor(
            weight,
            n_bits=n_bits,
            group_size=group_size,
            symmetric=symmetric,
        ).qweight

    device = next(inspect_module.parameters()).device
    kwargs = dict(forward_kwargs)
    kwargs.pop("use_cache", None)
    for key, value in list(kwargs.items()):
        if torch.is_tensor(value):
            kwargs[key] = value.to(device)

    activations = activations.to(device)

    # output baseline
    with torch.no_grad():
        reference = inspect_module(activations, **kwargs)
        if isinstance(reference, tuple):
            reference = reference[0]

    act_scale = _activation_scale(activations)
    best_error = float("inf")
    best_scales: torch.Tensor | None = None
    # Keep backup on the same device; loading CPU tensors would leave weights on CPU.
    original_state = {k: v.detach().clone() for k, v in inspect_module.state_dict().items()}

    for step in range(n_grid):
        ratio = step / n_grid
        scales = act_scale.pow(ratio).clamp(min=1e-4).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()

        for linear in target_linears:
            linear.weight.mul_(scales.view(1, -1).to(linear.weight.device))
            # W' = Quant(W * s) / s
            linear.weight.data = fake_quant(linear.weight.data) / scales.view(1, -1)

        candidate = inspect_module(activations, **kwargs)
        # ？？？
        if isinstance(candidate, tuple):
            candidate = candidate[0]

        error = (reference - candidate).float().pow(2).mean().item()
        if error < best_error:
            best_error = error
            best_scales = scales.detach().clone()
        # restore original weight
        inspect_module.load_state_dict(original_state)

    if best_scales is None:
        raise RuntimeError("Scale search failed to find a valid candidate.")

    return best_scales


def _llama_scale_specs(layer: LlamaDecoderLayer) -> List[ScaleSpec]:
    specs = [
        ScaleSpec(
            transform="ln_fc",
            prev_path="input_layernorm",
            target_paths=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
            ],
            input_key="self_attn.q_proj",
            inspect_path="self_attn",
            pass_layer_kwargs=True,
        ),
        ScaleSpec(
            transform="ln_fc",
            prev_path="post_attention_layernorm",
            target_paths=["mlp.gate_proj", "mlp.up_proj"],
            input_key="mlp.gate_proj",
            inspect_path="mlp",
        ),
        ScaleSpec(
            transform="fc_fc",
            prev_path="mlp.up_proj",
            target_paths=["mlp.down_proj"],
            input_key="mlp.down_proj",
            inspect_path="mlp.down_proj",
        ),
    ]

    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj
    if v_proj.weight.shape == o_proj.weight.shape:
        specs.insert(
            1,
            ScaleSpec(
                transform="fc_fc",
                prev_path="self_attn.v_proj",
                target_paths=["self_attn.o_proj"],
                input_key="self_attn.o_proj",
                inspect_path="self_attn.o_proj",
            ),
        )
    return specs


@torch.no_grad()
def apply_scale_records(
    layer: nn.Module,
    records: List[ScaleRecord],
    input_feat: Dict[str, torch.Tensor] | None = None,
) -> None:
    for record in records:
        scales = record.scales
        if record.transform == "ln_fc":
            norm = resolve_submodule(layer, record.prev_path)
            linears = [resolve_submodule(layer, path) for path in record.target_paths]
            _scale_layernorm_to_linears(norm, linears, scales)
        elif record.transform == "fc_fc":
            fc1 = resolve_submodule(layer, record.prev_path)
            fc2 = resolve_submodule(layer, record.target_paths[0])
            _scale_linear_to_linear(fc1, fc2, scales)
        else:
            raise NotImplementedError(f"Unknown transform: {record.transform}")
        # 激活统计也要同步除以 s
        if input_feat is not None:
            for path in record.target_paths:
                input_feat[path].div_(
                    scales.view(1, -1).to(input_feat[path].device, input_feat[path].dtype)
                )


@torch.no_grad()
def optimize_decoder_layer_scales(
    layer: nn.Module,
    input_feat: Dict[str, torch.Tensor],
    layer_kwargs: Dict,
    n_bits: int,
    group_size: int,
    symmetric: bool,
) -> List[ScaleRecord]:
    if not isinstance(layer, LlamaDecoderLayer):
        raise NotImplementedError(f"Unsupported decoder layer: {type(layer)}")

    records: List[ScaleRecord] = []
    for spec in _llama_scale_specs(layer):
        inspect_module = resolve_submodule(
            layer, spec.inspect_path or spec.target_paths[0]
        )
        target_linears = [resolve_submodule(layer, path) for path in spec.target_paths]
        activations = input_feat[spec.input_key]
        forward_kwargs = dict(layer_kwargs) if spec.pass_layer_kwargs else {}

        scales = _search_best_scales(
            inspect_module=inspect_module,
            target_linears=target_linears,
            activations=activations,
            forward_kwargs=forward_kwargs,
            n_bits=n_bits,
            group_size=group_size,
            symmetric=symmetric,
        )

        record = ScaleRecord(
            transform=spec.transform,
            prev_path=spec.prev_path,
            target_paths=spec.target_paths,
            scales=scales.detach().cpu(),
        )
        apply_scale_records(layer, [record], input_feat=input_feat)
        records.append(record)

    return records
