#!/usr/bin/env bash
# Eval AWQ model quantized with clip-then-scale order.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"
myawq_prepare

CKPT="${PROJECT_ROOT}/results/Llama2_7b_awq_clip_scale"
if [[ ! -f "${CKPT}/config.json" ]]; then
  echo "[myawq] Missing checkpoint: ${CKPT}" >&2
  echo "[myawq] Run first: ./scripts/run_quantize_llama2_7b_clip_scale.sh" >&2
  exit 1
fi

echo "[myawq] Running eval: configs/eval_awq_Llama2_7b_clip_scale.yaml"
python entry.py eval --config configs/eval_awq_Llama2_7b_clip_scale.yaml
