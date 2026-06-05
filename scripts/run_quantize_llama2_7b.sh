#!/usr/bin/env bash
# AWQ quantize for Llama-2-7B.
#
# Linux usage:
#   chmod +x scripts/*.sh
#   ./scripts/run_quantize_llama2_7b.sh
#
# Optional env vars:
#   MYAWQ_CONDA_ENV=myenv     default: d2l
#   MYAWQ_SKIP_INSTALL=1      skip pip install
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"
myawq_prepare

echo "[myawq] Running AWQ quantize: configs/quantize_awq_Llama2_7b.yaml"
python entry.py quantize --config configs/quantize_awq_Llama2_7b.yaml
