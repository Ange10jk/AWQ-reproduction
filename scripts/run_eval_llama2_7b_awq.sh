#!/usr/bin/env bash
# AWQ fake-quant PPL eval for Llama-2-7B.
# Prerequisite: results/Llama2_7b_fake_quant from run_quantize_llama2_7b.sh
#
# Linux usage:
#   ./scripts/run_eval_llama2_7b_awq.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"
myawq_prepare

CKPT="${PROJECT_ROOT}/results/Llama2_7b_fake_quant"
if [[ ! -f "${CKPT}/config.json" ]]; then
  echo "[myawq] Missing fake-quant checkpoint: ${CKPT}" >&2
  echo "[myawq] Run first: ./scripts/run_quantize_llama2_7b.sh" >&2
  exit 1
fi

echo "[myawq] Running AWQ eval: configs/eval_awq_Llama2_7b.yaml"
python entry.py eval --config configs/eval_awq_Llama2_7b.yaml
