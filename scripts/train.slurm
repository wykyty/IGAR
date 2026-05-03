#!/bin/bash
#SBATCH --job-name=train_gr
#SBATCH --partition=GPUNorm
#SBATCH --nodes=1
#SBATCH --gres=gpu:2              # 核心修改：申请 6 张 GPU
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32        # 建议调高 CPU 核心数，通常为 GPU 数量的 4-8 倍
#SBATCH --mem=64G                # 6张卡训练建议相应调大内存，防止 OOM
#SBATCH --time=24:00:00           # 根据任务量可能需要延长时限
#SBATCH --output=output/train/train_%j.log

# 激活虚拟环境
source ~/workspace/IGAR/.venv/bin/activate

# 调试：确认 GPU 分配情况
echo "=== Slurm GPU分配信息 ==="
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi -L
echo "========================"

cd ~/workspace/IGAR

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 使用 torchrun 启动六卡训练
# nproc_per_node 必须与 --gres=gpu:N 中的 N 保持一致
torchrun --nproc_per_node=2 src/train_gr.py