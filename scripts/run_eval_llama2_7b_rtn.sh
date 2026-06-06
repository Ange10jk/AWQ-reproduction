#!/usr/bin/env bash
# RTN fake-quant PPL eval for Llama-2-7B.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"
myawq_prepare

CKPT="${PROJECT_ROOT}/results/Llama2_7b_rtn_fake_quant"
if [[ ! -f "${CKPT}/config.json" ]]; then
  echo "[myawq] Missing RTN checkpoint: ${CKPT}" >&2
  echo "[myawq] Run first: ./scripts/run_quantize_llama2_7b_rtn.sh" >&2
  exit 1
fi

echo "[myawq] Running RTN eval: configs/eval_Llama2_7b_RTN.yaml"
python entry.py eval --config configs/eval_Llama2_7b_RTN.yaml
