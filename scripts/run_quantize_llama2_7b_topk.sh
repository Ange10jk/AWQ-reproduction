#!/usr/bin/env bash
# AWQ mixed-precision quantize for Llama-2-7B (protect top-K sensitive layers).
#
# Usage:
#   chmod +x scripts/*.sh
#   ./scripts/run_quantize_llama2_7b_topk.sh 0
#   ./scripts/run_quantize_llama2_7b_topk.sh 2
#   ./scripts/run_quantize_llama2_7b_topk.sh 4
#   ./scripts/run_quantize_llama2_7b_topk.sh 8
set -euo pipefail

TOPK="${1:-0}"
case "${TOPK}" in
  0|2|4|8) ;;
  *)
    echo "[myawq] TOPK must be one of: 0, 2, 4, 8" >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"
myawq_prepare

CFG="configs/quantize_awq_Llama2_7b_top${TOPK}_fp16protect.yaml"
echo "[myawq] Running AWQ mixed quantize: ${CFG}"
python entry.py quantize --config "${CFG}"
