"""group-wise RTN (round-to-nearest) pseudo quantization baseline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from transformers import PreTrainedModel, PreTrainedTokenizer

from quantize.quantizer import pseudo_quantize_model_weight


__all__ = ["run_rtn", "RtnRunConfig"]


@dataclass
class RtnRunConfig:
    n_bits: int
    group_size: int
    symmetric: bool
    output_dir: Path
    save_quant_ckpt: bool
    quant_ckpt_name: str

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> RtnRunConfig:
        awq = cfg.get("awq", {})
        output = cfg.get("output", {})
        return cls(
            n_bits=int(awq.get("n_bits", 4)),
            group_size=int(awq.get("group_size", 128)),
            symmetric=bool(awq.get("symmetric", False)),
            output_dir=Path(output.get("dir", "results")),
            save_quant_ckpt=bool(output.get("save_quant_ckpt", True)),
            quant_ckpt_name=str(output.get("quant_ckpt_name", "rtn_fake_quant")),
        )


def run_rtn(
    model: PreTrainedModel,
    enc: PreTrainedTokenizer,
    cfg: Dict[str, Any],
) -> None:
    """Apply RTN fake quantization without AWQ scale/clip search."""
    run_cfg = RtnRunConfig.from_dict(cfg)
    print(
        "[myawq] RTN baseline: "
        f"n_bits={run_cfg.n_bits}, group_size={run_cfg.group_size}, "
        f"symmetric={run_cfg.symmetric}"
    )

    t0 = time.perf_counter()
    pseudo_quantize_model_weight(
        model,
        n_bits=run_cfg.n_bits,
        group_size=run_cfg.group_size,
        symmetric=run_cfg.symmetric,
    )
    elapsed = time.perf_counter() - t0
    print(f"[myawq] RTN pseudo-quant finished in {elapsed:.1f}s")

    if not run_cfg.save_quant_ckpt:
        return

    run_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_cfg.output_dir / run_cfg.quant_ckpt_name
    model.save_pretrained(ckpt_dir)
    enc.save_pretrained(ckpt_dir)
    print(f"[myawq] Saved RTN fake-quant model to {ckpt_dir}")
