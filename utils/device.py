"""Device helpers for Ascend NPU / CUDA / CPU."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch

_NPU_INITIALIZED = False


def init_npu_backend() -> None:
    """Import torch_npu once so torch.npu APIs are registered."""
    global _NPU_INITIALIZED
    if _NPU_INITIALIZED:
        return
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return
    _NPU_INITIALIZED = True


def is_npu_available() -> bool:
    init_npu_backend()
    return hasattr(torch, "npu") and torch.npu.is_available()


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def resolve_backend(cfg: Optional[Dict[str, Any]] = None) -> str:
    """Return device backend name: npu | cuda | cpu."""
    requested = "auto"
    if cfg is not None:
        requested = str(cfg.get("model", {}).get("device", "auto")).lower()

    if requested in {"npu", "ascend"}:
        if not is_npu_available():
            raise RuntimeError("Config requests NPU but torch.npu is unavailable.")
        return "npu"
    if requested == "cuda":
        if not is_cuda_available():
            raise RuntimeError("Config requests CUDA but torch.cuda is unavailable.")
        return "cuda"
    if requested == "cpu":
        return "cpu"

    if is_npu_available():
        return "npu"
    if is_cuda_available():
        return "cuda"
    return "cpu"


def get_torch_device(cfg: Optional[Dict[str, Any]] = None) -> torch.device:
    backend = resolve_backend(cfg)
    if backend == "npu":
        return torch.device("npu:0")
    if backend == "cuda":
        return torch.device("cuda:0")
    return torch.device("cpu")


def require_accelerator(cfg: Optional[Dict[str, Any]] = None) -> torch.device:
    device = get_torch_device(cfg)
    if device.type == "cpu":
        raise RuntimeError("AWQ calibration requires NPU or CUDA.")
    return device


def resolve_device_map(cfg: Optional[Dict[str, Any]] = None) -> Union[str, Dict[str, str], None]:
    """HF device_map for model loading."""
    backend = resolve_backend(cfg)
    load_mode = "eval"
    if cfg is not None:
        load_mode = str(cfg.get("model", {}).get("load_mode", "eval")).lower()

    if load_mode == "quantize":
        return "cpu"

    if backend == "npu":
        return {"": "npu:0"}
    if backend == "cuda":
        return "auto"
    return "cpu"


def to_accelerator(module: torch.nn.Module, cfg: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
    return module.to(get_torch_device(cfg))


def to_cpu(module: torch.nn.Module) -> torch.nn.Module:
    return module.to("cpu")


def empty_cache() -> None:
    gc = __import__("gc")
    gc.collect()
    if is_npu_available():
        torch.npu.empty_cache()
    elif is_cuda_available():
        torch.cuda.empty_cache()


def device_label(cfg: Optional[Dict[str, Any]] = None) -> str:
    return get_torch_device(cfg).type
