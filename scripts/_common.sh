#!/usr/bin/env bash
# Shared helpers for myawq scripts (Linux / bash).

MYAWQ_CONDA_ENV="${MYAWQ_CONDA_ENV:-d2l}"
MYAWQ_SKIP_INSTALL="${MYAWQ_SKIP_INSTALL:-0}"

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"

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

  echo "[myawq] conda not found. Install Miniconda or add conda to PATH." >&2
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

myawq_install_deps() {
  if [[ "${MYAWQ_SKIP_INSTALL}" == "1" ]]; then
    echo "[myawq] Skip dependency install (MYAWQ_SKIP_INSTALL=1)."
    return 0
  fi

  echo "[myawq] Installing Python dependencies into: ${MYAWQ_CONDA_ENV}"
  python -m pip install -U pip
  python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
  # Optional: SOCKS proxy support for Hugging Face downloads
  python -m pip install "httpx[socks]"
}

myawq_prepare() {
  myawq_activate_conda
  myawq_install_deps
  cd "${PROJECT_ROOT}"
}
