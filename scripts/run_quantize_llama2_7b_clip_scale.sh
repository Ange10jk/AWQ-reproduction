#!/usr/bin/env bash
# AWQ quantize with clip-then-scale search order (ablation).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"
myawq_prepare

echo "[myawq] Running AWQ (clip_scale order): configs/quantize_awq_Llama2_7b_clip_scale.yaml"
python entry.py quantize --config configs/quantize_awq_Llama2_7b_clip_scale.yaml
