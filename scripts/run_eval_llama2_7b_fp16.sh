#!/usr/bin/env bash
# FP16 baseline PPL eval for Llama-2-7B (pre-AWQ).
#
# Linux usage:
#   ./scripts/run_eval_llama2_7b_fp16.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"
myawq_prepare

echo "[myawq] Running FP16 eval: configs/eval_Llama2_7b_b4q.yaml"
python entry.py eval --config configs/eval_Llama2_7b_b4q.yaml
