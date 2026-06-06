#!/usr/bin/env bash
# RTN fake-quant for Llama-2-7B (no AWQ scale/clip search).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"
myawq_prepare

echo "[myawq] Running RTN quantize: configs/quantize_rtn_Llama2_7b.yaml"
python entry.py quantize_rtn --config configs/quantize_rtn_Llama2_7b.yaml
