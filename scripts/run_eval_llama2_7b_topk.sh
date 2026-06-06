#!/usr/bin/env bash
# AWQ mixed-precision eval for Llama-2-7B (top-K sensitive layers kept in FP16).
#
# Usage:
#   ./scripts/run_eval_llama2_7b_topk.sh 0
#   ./scripts/run_eval_llama2_7b_topk.sh 2
#   ./scripts/run_eval_llama2_7b_topk.sh 4
#   ./scripts/run_eval_llama2_7b_topk.sh 8
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

CKPT="${PROJECT_ROOT}/results/Llama2_7b_awq_top${TOPK}_fp16protect"
if [[ ! -f "${CKPT}/config.json" ]]; then
  echo "[myawq] Missing mixed checkpoint: ${CKPT}" >&2
  echo "[myawq] Run first: ./scripts/run_quantize_llama2_7b_topk.sh ${TOPK}" >&2
  exit 1
fi

CFG="configs/eval_awq_Llama2_7b_top${TOPK}_fp16protect.yaml"
echo "[myawq] Running AWQ mixed eval: ${CFG}"
python entry.py eval --config "${CFG}"
