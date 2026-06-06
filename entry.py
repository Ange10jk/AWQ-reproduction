"""Command line entrypoint for AWQ workflow.

Usage:
    python entry.py quantize --config configs/base_quantize.yaml
    python entry.py quantize_rtn --config configs/quantize_rtn_Llama2_7b.yaml
    # 量化前
    python entry.py eval --config configs/base_eval_fp16.yaml
    # 量化后
    python entry.py eval --config configs/base_eval_awq.yaml

Huawei Ascend NPU (huawei branch):
    python entry.py quantize --config configs/quantize_awq_Llama2_7b.yaml
    python entry.py quantize_rtn --config configs/quantize_rtn_Llama2_7b.yaml
    python entry.py eval --config configs/eval_Llama2_7b_b4q.yaml
    python entry.py eval --config configs/eval_Llama2_7b_RTN.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from utils.config import load_config
from utils.device import device_label, init_npu_backend
from utils.model import build_model_and_enc
from quantize.pre_quant import run_awq
from quantize.rtn import run_rtn
from eval.ppl import evaluate_ppl


def _ensure_load_mode(cfg: Dict[str, Any], command: str) -> None:
    model_cfg = cfg.setdefault("model", {})
    if "load_mode" not in model_cfg:
        model_cfg["load_mode"] = (
            "quantize" if command in {"quantize", "quantize_rtn"} else "eval"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="myawq CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base_quantize.yaml"),
        help="Path to yaml config file",
    )

    subparsers.add_parser(
        "quantize", parents=[shared], help="Run AWQ quantization pipeline"
    )
    subparsers.add_parser(
        "quantize_rtn",
        parents=[shared],
        help="Run naive RTN fake-quant baseline (no AWQ search)",
    )
    subparsers.add_parser("eval", parents=[shared], help="Run PPL evaluation")
    return parser


def main() -> None:
    init_npu_backend()
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    _ensure_load_mode(cfg, args.command)
    print(f"[myawq] Using accelerator: {device_label(cfg)}")
    model, enc = build_model_and_enc(cfg)

    if args.command == "quantize":
        run_awq(model, enc, cfg)
    elif args.command == "quantize_rtn":
        run_rtn(model, enc, cfg)
    elif args.command == "eval":
        evaluate_ppl(model, enc, cfg)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
