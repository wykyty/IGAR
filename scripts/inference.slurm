#!/bin/bash
#SBATCH --job-name=inference_gr      # 任务名称，方便在 squeue 中查看
#SBATCH --partition=GPUNorm            # 指定提交到 GPU 队列/分区 (请根据你所在集群的实际队列名称修改，比如 gpu, gpu-v100 等)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1               # 最核心的一行：申请 1 张 GPU
#SBATCH --ntasks=1                 # 运行 1 个任务 (进程)
#SBATCH --cpus-per-task=16          # 为该任务分配 4 个 CPU 核心 (常用于 DataLoader 的 num_workers)
#SBATCH --mem=64G                  # 申请 16GB 内存 (根据你的模型大小调整)
#SBATCH --time=24:00:00            # 申请的最长运行时间 (格式：小时:分钟:秒)
#SBATCH --output=output/inference/job_%j.out        # 标准输出保存文件 (%j 会被替换为真实的 Job ID)
#SBATCH --error=output/inference/job_%j.err         # 标准错误保存文件


# 打印分配到的节点和 GPU 信息（方便后续排查问题）
echo "Job runs on node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 激活你的虚拟环境 (例如使用 Conda)
WORKSPACE="/labmount/users/202220150143/workspace/IGAR" 
export HF_ENDPOINT="https://hf-mirror.com" 
export PYTHONPATH="${WORKSPACE}:${PYTHONPATH}"
source ~/workspace/IGAR/.venv/bin/activate
cd $WORKSPACE

python src/inference_gr.py