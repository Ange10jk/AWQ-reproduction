# utils/model.py
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


def resolve_model_path(cfg: Dict[str, Any]) -> str: 
  """
  args : config (集中管理所有参数的 Dict )
    cfg = {
      "model": {
        "name": "Qwen/Qwen2.5-7B-Instruct",      
        "ckpt_path": "./checkpoints/qwen_sft",    # 本地权重路径（优先于 name）
        "dtype": "bfloat16",                      # 精度
        "use_cache": False,                       # 校准/评测阶段关，推理阶段开
        "tokenizer": None                         # 可选：分词器独立路径
      },
      "quantization": {
        "w_bit": 4,
        "q_group_size": 128,
        "backend": "real"
      },
      "training": { ... }
  }
  """
  model_cfg = cfg.get("model", {})
  return model_cfg.get("ckpt_path") or model_cfg.get("name", "")


def build_model_and_enc(
  cfg: Dict[str, Any], 
  device: str = "auto"
) -> Tuple[PreTrainedModel, AutoTokenizer]:
  model_cfg = cfg.get("model", {})
  model_path = resolve_model_path(cfg)
  dtype = model_cfg.get("dtype", "float16")
  torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

  config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
  # 校准/评测阶段关闭 KV Cache 防止历史 Token 显存泄漏；若需推理生成，可在 cfg 中覆盖为 True
  config.use_cache = model_cfg.get("use_cache", False)

  tokenizer_path = model_cfg.get("tokenizer") or model_path
  enc = AutoTokenizer.from_pretrained(
    tokenizer_path, use_fast=False, trust_remote_code=True
  ) # 慢速分词器（纯 python）稳定

  model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    device_map=device,  # "auto" 依赖 accelerate，也可传 "cuda:0" 或 None
    trust_remote_code=True,
  )
  # 锁定推理模式
  model.eval() 
  return model, enc