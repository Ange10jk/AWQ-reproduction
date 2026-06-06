"""Pre-quantization orchestration for AWQ."""

from __future__ import annotations

import gc
import inspect
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import torch
import torch.nn as nn
import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from data.calib_data import get_calib_dataset
from quantize.auto_clip import ClipRecord, apply_clip_records, optimize_decoder_layer_clips
from quantize.auto_scale import ScaleRecord, apply_scale_records, optimize_decoder_layer_scales
from quantize.quantizer import pseudo_quantize_tensor
from utils.device import empty_cache, require_accelerator, to_accelerator, to_cpu

__all__ = [
    "run_awq",
    "apply_awq",
    "get_blocks",
    "get_named_linears",
    "move_embed",
]


@dataclass
class AwqRunConfig:
    n_bits: int
    group_size: int
    symmetric: bool
    auto_scale: bool
    auto_clip: bool
    dataset: str
    num_samples: int
    seq_len: int
    seed: int
    output_dir: Path
    save_quant_ckpt: bool
    quant_ckpt_name: str
    mixed_precision: str
    topk: int

    @classmethod # 类方法，不是对象方法
    def from_dict(cls, cfg: Dict[str, Any]) -> AwqRunConfig:
        calib = cfg.get("calibration", {})
        awq = cfg.get("awq", {})
        output = cfg.get("output", {})
        mixed_precision = str(awq.get("mixed_precision", "none")).lower()
        if mixed_precision not in {"none", "fp16", "int8"}:
            raise ValueError(
                "awq.mixed_precision must be one of: none | fp16 | int8"
            )
        return cls(
            n_bits=int(awq.get("n_bits", 4)),
            group_size=int(awq.get("group_size", 128)),
            symmetric=bool(awq.get("symmetric", False)),
            auto_scale=bool(awq.get("auto_scale", True)),
            auto_clip=bool(awq.get("auto_clip", True)),
            dataset=calib.get("dataset", "pileval"),
            num_samples=int(calib.get("num_samples", 512)),
            seq_len=int(calib.get("seq_len", 512)),
            seed=int(calib.get("seed", 42)),
            output_dir=Path(output.get("dir", "results")),
            save_quant_ckpt=bool(output.get("save_quant_ckpt", True)),
            quant_ckpt_name=str(output.get("quant_ckpt_name", "fake_quant_model")),
            mixed_precision=mixed_precision,
            topk=max(0, int(awq.get("topk", 0))),
        )


def get_blocks(model: PreTrainedModel) -> nn.ModuleList:
    if isinstance(model, LlamaForCausalLM):
        return model.model.layers
    raise NotImplementedError(f"Unsupported model type: {type(model)}")


def move_embed(model: PreTrainedModel, device: str | torch.device) -> None:
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device) # 旋转位置编码
        return
    raise NotImplementedError(f"Unsupported model type: {type(model)}")


def get_named_linears(module: nn.Module) -> Dict[str, nn.Linear]:
    return {name: layer for name, layer in module.named_modules() if isinstance(layer, nn.Linear)}


def _create_causal_mask_compat(
    config: Any,
    hidden: torch.Tensor,
    cache_position: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor | None:
    """Transformers 4.x uses input_embeds; 5.x renamed it to inputs_embeds."""
    from transformers.masking_utils import create_causal_mask

    params = inspect.signature(create_causal_mask).parameters
    embed_kw = "inputs_embeds" if "inputs_embeds" in params else "input_embeds"
    return create_causal_mask(
        config=config,
        attention_mask=None,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
        **{embed_kw: hidden},
    )


def _refresh_layer_forward_kwargs(
    model: LlamaForCausalLM,
    hidden: torch.Tensor,
    layer_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Rebuild RoPE/mask kwargs for the current hidden states.

    Transformers 4.x/5.x compute `position_embeddings` from the current hidden
    states. Reusing the tensors captured at layer 0 can break from layer 1 onward.
    """
    device = hidden.device
    batch_size, seq_len = hidden.shape[:2]

    position_ids = layer_kwargs.get("position_ids")
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_ids = position_ids.to(device)
    if position_ids.shape[0] == 1 and batch_size > 1:
        position_ids = position_ids.expand(batch_size, -1)

    cache_position = layer_kwargs.get("cache_position")
    if cache_position is None:
        cache_position = torch.arange(seq_len, device=device)
    else:
        cache_position = cache_position.to(device)

    rotary_emb = model.model.rotary_emb.to(device)
    position_embeddings = rotary_emb(hidden, position_ids=position_ids)

    causal_mask = _create_causal_mask_compat(
        model.config, hidden, cache_position, position_ids
    )

    return {
        "attention_mask": causal_mask,
        "position_ids": position_ids,
        "cache_position": cache_position,
        "position_embeddings": position_embeddings,
        "use_cache": False,
    }


def _extract_hidden_states(output: torch.Tensor | tuple) -> torch.Tensor:
    """Decoder layers may return a tensor (HF 5.x) or a tuple (older HF)."""
    if isinstance(output, tuple):
        return output[0]
    return output


def _build_calib_batch(calib_samples, device: torch.device) -> torch.Tensor:
    batch = torch.cat([sample.input_ids for sample in calib_samples], dim=0)
    return batch.to(device)


@contextmanager # support WITH
def capture_layer0_inputs(
    model: PreTrainedModel,
    first_layer: nn.Module,
    calib_batch: torch.Tensor,
) -> Iterator[Tuple[torch.Tensor, Dict[str, Any]]]:
    """Run embedding + layer0 once and capture hidden states and kwargs."""
    captured_inputs: List[torch.Tensor] = []
    captured_kwargs: Dict[str, Any] = {}

    class PrefillProbe(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, **kwargs):
            captured_inputs.append(hidden_states)
            captured_kwargs.update(kwargs)
            raise RuntimeError("prefill_probe_stop") # 第一层即退出

    probe = PrefillProbe(first_layer)
    layers = get_blocks(model)
    layers[0] = probe

    try:
        model(calib_batch)
    except RuntimeError as exc:
        if "prefill_probe_stop" not in str(exc):
            raise

    layers[0] = first_layer
    if not captured_inputs:
        raise RuntimeError("Failed to capture layer-0 inputs during calibration prefill.")

    yield captured_inputs[0], captured_kwargs


class LinearActivationRecorder:
    """Collect linear-layer inputs during one decoder-layer forward."""

    def __init__(self, linears: Dict[str, nn.Linear]):
        self._linears = linears
        self._storage: Dict[str, List[torch.Tensor]] = {name: [] for name in linears}
        # 清理用
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> LinearActivationRecorder:
        for name, module in self._linears.items():

            def _hook(mod, inputs, output, layer_name=name):
                """record input tensor for each linear-layer"""
                self._storage[layer_name].append(inputs[0].detach().cpu())

            self._handles.append(module.register_forward_hook(_hook))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._handles:
            handle.remove()

    def consolidated(self) -> Dict[str, torch.Tensor]:
        return {name: torch.cat(tensors, dim=0) for name, tensors in self._storage.items()}


class AwqPipeline:
    """Layer-wise AWQ driver with an explicit prefill + optimize loop."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cfg: Dict[str, Any],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.run_cfg = AwqRunConfig.from_dict(cfg)
        self.device = require_accelerator(cfg)

    def _free_device_cache(self) -> None:
        empty_cache()

    def _load_calibration_batch(self) -> torch.Tensor:
        samples = get_calib_dataset(
            dataset_name=self.run_cfg.dataset,
            tokenizer=self.tokenizer,
            num_samples=self.run_cfg.num_samples,
            seq_len=self.run_cfg.seq_len,
            seed=self.run_cfg.seed,
        )
        print(f"[myawq] Loaded {len(samples)} calibration blocks.")
        return _build_calib_batch(samples, self.device)

    def _prefill_hidden_states(self, calib_batch: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        layers = get_blocks(self.model)
        layers[0] = to_accelerator(layers[0], self.cfg)
        move_embed(self.model, self.device)

        with capture_layer0_inputs(self.model, layers[0], calib_batch) as (hidden, kwargs):
            pass

        layers[0] = to_cpu(layers[0])
        move_embed(self.model, "cpu")
        self._free_device_cache()
        return hidden, kwargs # layer 0 
    
    def _process_layer(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        layer_kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[ScaleRecord], List[ClipRecord], float]:
        layer = to_accelerator(layer, self.cfg)
        linears = get_named_linears(layer)

        layer_input = hidden.to(next(layer.parameters()).device)
        forward_kwargs = _refresh_layer_forward_kwargs(self.model, layer_input, layer_kwargs)

        with LinearActivationRecorder(linears) as recorder:
            _extract_hidden_states(layer(layer_input, **forward_kwargs))

        input_feat = recorder.consolidated()
        scale_records: List[ScaleRecord] = []
        clip_records: List[ClipRecord] = []

        if self.run_cfg.auto_scale:
            scale_records = optimize_decoder_layer_scales(
                layer=layer,
                input_feat=input_feat,
                layer_kwargs=forward_kwargs,
                n_bits=self.run_cfg.n_bits,
                group_size=self.run_cfg.group_size,
                symmetric=self.run_cfg.symmetric,
            )

        if self.run_cfg.auto_clip:
            clip_records = optimize_decoder_layer_clips(
                layer=layer,
                input_feat=input_feat,
                n_bits=self.run_cfg.n_bits,
                group_size=self.run_cfg.group_size,
                symmetric=self.run_cfg.symmetric,
                device=self.device,
            )

        # scale/clip search may leave submodules on CPU; re-sync before forward.
        layer = to_accelerator(layer, self.cfg)
        layer_input = layer_input.to(self.device)
        forward_kwargs = _refresh_layer_forward_kwargs(self.model, layer_input, layer_kwargs)

        hidden = _extract_hidden_states(layer(layer_input, **forward_kwargs))
        output_mse = self._estimate_layer_output_mse(
            layer,
            linears,
            layer_input,
            forward_kwargs,
            hidden,
        )

        layer = to_cpu(layer)
        self._free_device_cache()
        return hidden.detach().cpu(), input_feat, scale_records, clip_records, output_mse

    @torch.no_grad()
    def _estimate_layer_output_mse(
        self,
        layer: nn.Module,
        linears: Dict[str, nn.Linear],
        layer_input: torch.Tensor,
        forward_kwargs: Dict[str, Any],
        fp_output: torch.Tensor,
    ) -> float:
        backups: Dict[str, torch.Tensor] = {}
        for name, linear in linears.items():
            backups[name] = linear.weight.data.detach().clone()
            qweight = pseudo_quantize_tensor(
                linear.weight.data,
                n_bits=self.run_cfg.n_bits,
                group_size=self.run_cfg.group_size,
                symmetric=self.run_cfg.symmetric,
                inplace=False,
            ).qweight
            linear.weight.data.copy_(qweight)

        quant_output = _extract_hidden_states(layer(layer_input, **forward_kwargs))
        mse = torch.mean((fp_output.float() - quant_output.float()) ** 2).item()

        for name, linear in linears.items():
            linear.weight.data.copy_(backups[name])
        return float(mse)

    @staticmethod
    def _select_topk_layers_by_mse(awq_results: Dict[str, Any], topk: int) -> List[int]:
        if topk <= 0:
            return []
        ranked = sorted(
            (
                (int(entry["index"]), float(entry.get("output_mse", 0.0)))
                for entry in awq_results.get("layers", [])
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        return [idx for idx, _ in ranked[:topk]]

    @torch.no_grad()
    def _apply_mixed_precision_quant(
        self,
        protected_layers: List[int],
    ) -> None:
        protected_set = set(protected_layers)
        mode = self.run_cfg.mixed_precision

        for layer_idx, layer in enumerate(get_blocks(self.model)):
            if layer_idx in protected_set and mode == "fp16":
                continue
            bit_width = 8 if (layer_idx in protected_set and mode == "int8") else self.run_cfg.n_bits
            for linear in get_named_linears(layer).values():
                qweight = pseudo_quantize_tensor(
                    linear.weight.data,
                    n_bits=bit_width,
                    group_size=self.run_cfg.group_size,
                    symmetric=self.run_cfg.symmetric,
                    inplace=False,
                ).qweight
                linear.weight.data.copy_(qweight)

    def _persist_results(
        self,
        awq_results: Dict[str, Any],
        save_tokenizer: bool = True,
    ) -> None:
        self.run_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        result_path = self.run_cfg.output_dir / "awq_results.pt"
        torch.save(awq_results, result_path)
        print(f"[myawq] Saved AWQ records to {result_path}")

        if not self.run_cfg.save_quant_ckpt:
            return

        protected_layers = (
            self._select_topk_layers_by_mse(awq_results, self.run_cfg.topk)
            if self.run_cfg.mixed_precision != "none"
            else []
        )
        self._apply_mixed_precision_quant(protected_layers)

        selected = [
            {
                "index": idx,
                "output_mse": next(
                    float(entry.get("output_mse", 0.0))
                    for entry in awq_results["layers"]
                    if int(entry["index"]) == idx
                ),
            }
            for idx in protected_layers
        ]
        plan = {
            "base_bits": self.run_cfg.n_bits,
            "group_size": self.run_cfg.group_size,
            "symmetric": self.run_cfg.symmetric,
            "mixed_precision": self.run_cfg.mixed_precision,
            "topk": self.run_cfg.topk,
            "protected_layers": selected,
        }
        plan_path = self.run_cfg.output_dir / "mixed_precision_plan.json"
        with plan_path.open("w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)
        if protected_layers:
            print(
                f"[myawq] Mixed precision mode={self.run_cfg.mixed_precision}, "
                f"topk={self.run_cfg.topk}, layers={protected_layers}"
            )
        else:
            print("[myawq] Mixed precision disabled (uniform quantization).")

        ckpt_dir = self.run_cfg.output_dir / self.run_cfg.quant_ckpt_name
        self.model.save_pretrained(ckpt_dir)
        if save_tokenizer:
            self.tokenizer.save_pretrained(ckpt_dir)
        print(f"[myawq] Saved fake-quant model to {ckpt_dir}")

    def run(self) -> Dict[str, Any]:
        calib_batch = self._load_calibration_batch()
        hidden, layer_kwargs = self._prefill_hidden_states(calib_batch)
        del calib_batch

        awq_results: Dict[str, Any] = {"layers": []}
        layers = get_blocks(self.model)

        for layer_idx in tqdm.tqdm(range(len(layers)), desc="Running AWQ"):
            hidden, _, scale_records, clip_records, output_mse = self._process_layer(
                layers[layer_idx],
                hidden,
                layer_kwargs,
            )
            awq_results["layers"].append(
                {
                    "index": layer_idx,
                    "scales": scale_records,
                    "clips": clip_records,
                    "output_mse": output_mse,
                }
            )

        self._persist_results(awq_results)
        return awq_results


def run_awq(
    model: PreTrainedModel,
    enc: PreTrainedTokenizer,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    return AwqPipeline(model, enc, cfg).run()


@torch.no_grad()
def apply_awq(model: PreTrainedModel, awq_results: Dict[str, Any]) -> None:
    """Replay saved scale/clip records layer by layer."""
    layers = get_blocks(model)
    for entry in awq_results.get("layers", []):
        layer = layers[entry["index"]]
        if entry.get("scales"):
            apply_scale_records(layer, entry["scales"])
        if entry.get("clips"):
            apply_clip_records(layer, entry["clips"], device=torch.device("cpu"))
