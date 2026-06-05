#!/usr/bin/env bash
# Shared helpers for myawq scripts (Linux / bash).
# Supports CUDA dev (conda) and Huawei Ascend NPU (CANN notebook).

MYAWQ_CONDA_ENV="${MYAWQ_CONDA_ENV:-d2l}"
MYAWQ_SKIP_INSTALL="${MYAWQ_SKIP_INSTALL:-0}"
MYAWQ_BACKEND="${MYAWQ_BACKEND:-auto}"
MYAWQ_NPU_ID="${ASCEND_RT_VISIBLE_DEVICES:-0}"

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"

myawq_detect_backend() {
  if [[ "${MYAWQ_BACKEND}" != "auto" ]]; then
    return 0
  fi
  if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]] \
    || [[ -f /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash ]]; then
    MYAWQ_BACKEND="npu"
  else
    MYAWQ_BACKEND="conda"
  fi
}

myawq_init_conda() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    return 0
  fi

  local candidate
  for candidate in \
    "${CONDA_EXE%/*}/../etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "${candidate}" ]]; then
      # shellcheck source=/dev/null
      source "${candidate}"
      return 0
    fi
  done

  echo "[myawq] conda not found. Install Miniconda or set MYAWQ_BACKEND=npu." >&2
  exit 1
}

myawq_activate_conda() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${MYAWQ_CONDA_ENV}" ]]; then
    echo "[myawq] Using active conda env: ${MYAWQ_CONDA_ENV}"
    return 0
  fi

  myawq_init_conda
  conda activate "${MYAWQ_CONDA_ENV}"
  echo "[myawq] Activated conda env: ${MYAWQ_CONDA_ENV}"
}

myawq_setup_ascend_env() {
  if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    # shellcheck source=/dev/null
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
  elif [[ -f /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash ]]; then
    # shellcheck source=/dev/null
    source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
  fi

  export ASCEND_RT_VISIBLE_DEVICES="${MYAWQ_NPU_ID}"
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_DISABLE_SYMLINKS_WARNING=1
}

myawq_check_npu() {
  python - <<'PY'
import torch
try:
    import torch_npu  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "[myawq] torch_npu not found. Use CANN notebook image or set MYAWQ_BACKEND=conda."
    ) from exc

if not torch.npu.is_available():
    raise SystemExit("[myawq] Ascend NPU is not available.")
print(f"[myawq] NPU ready, count={torch.npu.device_count()}")
PY
}

myawq_install_deps() {
  if [[ "${MYAWQ_SKIP_INSTALL}" == "1" ]]; then
    echo "[myawq] Skip dependency install (MYAWQ_SKIP_INSTALL=1)."
    return 0
  fi

  echo "[myawq] Installing Python dependencies"
  python -m pip install -U pip
  python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
  if [[ "${MYAWQ_BACKEND}" == "conda" ]]; then
    python -m pip install "httpx[socks]"
  fi
}

myawq_prepare() {
  myawq_detect_backend
  echo "[myawq] Backend: ${MYAWQ_BACKEND}"

  if [[ "${MYAWQ_BACKEND}" == "npu" ]]; then
    myawq_setup_ascend_env
    myawq_check_npu
  else
    myawq_activate_conda
  fi

  myawq_install_deps
  cd "${PROJECT_ROOT}"
}
