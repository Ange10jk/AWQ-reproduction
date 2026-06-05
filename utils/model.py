# utils/model.py
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from utils.device import init_npu_backend, resolve_device_map


def resolve_model_path(cfg: Dict[str, Any]) -> str:
  model_cfg = cfg.get("model", {})
  return model_cfg.get("ckpt_path") or model_cfg.get("name", "")


def build_model_and_enc(
  cfg: Dict[str, Any],
) -> Tuple[PreTrainedModel, AutoTokenizer]:
  init_npu_backend()
  model_cfg = cfg.get("model", {})
  model_path = resolve_model_path(cfg)
  dtype = model_cfg.get("dtype", "float16")
  torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
  local_files_only = bool(model_cfg.get("local_files_only", False))
  device_map = resolve_device_map(cfg)

  config = AutoConfig.from_pretrained(
    model_path, trust_remote_code=True, local_files_only=local_files_only
  )
  config.use_cache = model_cfg.get("use_cache", False)

  tokenizer_path = model_cfg.get("tokenizer") or model_path
  enc = AutoTokenizer.from_pretrained(
    tokenizer_path,
    use_fast=False,
    trust_remote_code=True,
    local_files_only=local_files_only,
  )

  model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    device_map=device_map,
    trust_remote_code=True,
    local_files_only=local_files_only,
  )
  model.eval()
  return model, enc
