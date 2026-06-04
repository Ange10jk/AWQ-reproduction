"""Command line entrypoint for AWQ workflow.

Usage:
    # awq 量化
    python entry.py quantize --config configs/base_quantize.yaml
    # 量化前
    python entry.py eval --config configs/base_eval_fp16.yaml 
    # 量化后
    python entry.py eval --config configs/base_eval_awq.yaml


entry.py
  ├─ load_config(config_path)          # utils/config.py
  ├─ build_model_and_enc(cfg)          # utils/model.py
  └─ 按 command 分发
       ├─ quantize → run_awq(model, enc, cfg)
       └─ eval     → evaluate_ppl(model, enc, cfg)

load config
→ build_model_and_enc()
→ run_awq(model, enc, w_bit, q_config, calib_samples...)
→ apply_awq() + pseudo_quantize_model_weight()  # 在 pre_quant 里完成
→ 保存 awq_results.pt / fake-quant checkpoint 到 results/

load config
→ build_model_and_enc()  # FP16 baseline
→ 或 load 已量化 checkpoint
→ WikiText-2 test 滑窗算 PPL
→ 写入 results/metrics.json
    
    """

from __future__ import annotations

import argparse
from pathlib import Path

from utils.config import load_config
from utils.model import build_model_and_enc
from quantize.pre_quant import run_awq
from eval.ppl import evaluate_ppl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="myawq CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to yaml config file",
    )

    subparsers.add_parser(
        "quantize", parents=[shared], help="Run AWQ quantization pipeline"
    )
    subparsers.add_parser("eval", parents=[shared], help="Run PPL evaluation")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    model, enc = build_model_and_enc(cfg)

    if args.command == "quantize":
        run_awq(model, enc, cfg)
    elif args.command == "eval":
        evaluate_ppl(model, enc, cfg)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
