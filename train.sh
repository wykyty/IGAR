#!/bin/bash
# 完全禁用 huggingface 的所有锁机制
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 使用内存文件系统
export HF_DATASETS_CACHE=/dev/shm/hf_cache_$$
export HF_HOME=/dev/shm/hf_home_$$

# 创建临时目录
mkdir -p $HF_DATASETS_CACHE $HF_HOME

# 运行训练
torchrun --nproc_per_node=2 src/train_gr.py

# 清理
rm -rf $HF_DATASETS_CACHE $HF_HOME