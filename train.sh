#!/bin/bash
# 运行训练
torchrun --nproc_per_node=2 src/train_gr.py
