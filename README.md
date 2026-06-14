# myawq

轻量级 [AWQ（Activation-aware Weight Quantization）](https://arxiv.org/abs/2306.00978) 实现，用于大语言模型的 3/4-bit 伪量化、困惑度（PPL）评估与消融实验。支持 **CUDA**、**华为昇腾 NPU** 。
作为对官方 AWQ 论文的复现，只复现了 AWQ 伪量化的部分、和部分实验结果，关于内核 kernel 优化、TinyChat的部分未复现。

## 功能

- **AWQ 量化**：自动搜索 scale / clip，支持 `scale_clip` 与 `clip_scale` 两种搜索顺序
- **RTN 基线**：不经过 AWQ 搜索的朴素 round-to-nearest 伪量化
- **混合精度**：可对敏感度最高的 Top-K 层保留 FP16（`mixed_precision: fp16`）
- **PPL 评估**：在 WikiText-2 上计算困惑度，结果保存为 JSON

## 项目结构

```
myawq/
├── entry.py          # 命令行入口
├── quantize/         # AWQ / RTN 量化逻辑
├── eval/             # PPL 评估
├── data/             # 校准数据加载
├── utils/            # 模型加载、设备抽象、配置解析
├── configs/          # YAML 配置文件
├── scripts/          # quantize/eval 一键运行脚本（Linux / bash）
└── results/          # 量化 checkpoint 与评估指标
```

## 环境安装

```bash
pip install -r requirements.txt
# PyTorch 需单独安装（CUDA 用 conda，昇腾 NPU 使用 CANN 镜像预装版本）
```

依赖：`transformers`、`accelerate`、`datasets`、`PyYAML` 等，详见 `requirements.txt`。

## 快速开始

### 命令行

```bash
# AWQ 量化
python entry.py quantize --config configs/quantize_awq_Llama2_7b.yaml

# RTN 基线
python entry.py quantize_rtn --config configs/quantize_rtn_Llama2_7b.yaml

# RTN 基线评估
python entry.py eval --config configs/eval_Llama2_7b_RTN.yaml

# FP16 评估
python entry.py eval --config configs/eval_Llama2_7b_b4q.yaml

# 量化模型评估
python entry.py eval --config configs/eval_awq_Llama2_7b.yaml
```

### Shell 脚本（Linux）

```bash
chmod +x scripts/*.sh

./scripts/run_eval_llama2_7b_fp16.sh      # FP16 基线
./scripts/run_quantize_llama2_7b.sh       # AWQ 量化
./scripts/run_eval_llama2_7b_awq.sh       # AWQ 评估
./scripts/run_quantize_llama2_7b_rtn.sh   # RTN 基线
./scripts/run_quantize_llama2_7b_topk.sh 0(default)/2/4/8 # Top-K FP16 保护实验
```

脚本会自动检测运行环境（conda / 昇腾 NPU），也可通过环境变量 `MYAWQ_BACKEND=conda|npu` 手动指定。

## 配置说明

量化与评估均通过 YAML 配置，主要字段如下：


| 区块            | 说明                                                        |
| ------------- | --------------------------------------------------------- |
| `model`       | 模型名称、dtype、设备（`cuda` / `npu` / `cpu` / `auto`）            |
| `calibration` | 校准数据集（`pileval`）、样本数、序列长度                                 |
| `awq`         | 位宽、group size、symmetric、auto_scale / auto_clip、混合精度 Top-K |
| `evaluation`  | 评估数据集（`wikitext2`）、最大序列长度                                 |
| `output`      | 输出目录、是否保存checkpoint、checkpoint 名称                         |


TinyLlama-1.1B 示例配置见 `configs/base_quantize.yaml`；Llama-2-7B 完整实验配置见 `configs/quantize_awq_Llama2_7b*.yaml`。


## 参考

- AWQ 原论文：[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- 官方实现：[mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)

