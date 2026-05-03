#!/bin/bash
#SBATCH --job-name=t5_single_train
#SBATCH --partition=GPUNorm
#SBATCH --nodes=1
#SBATCH --nodelist=gn8
#SBATCH --gres=gpu:1              # 只用1张卡！
#SBATCH --ntasks=1               
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00          # 72小时
#SBATCH --output=train_single_%j.log
#SBATCH --error=train_single_%j.err

echo "=========================================="
echo "T5-Large 单卡训练 (优化版)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Submit Time: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# ==================== 环境清理 ====================
echo "清理环境..."
rm -rf /tmp/hf_cache_* 2>/dev/null
rm -rf ~/.cache/huggingface/datasets/*.lock 2>/dev/null
echo "✓ 环境清理完成"

# ==================== GPU信息 ====================
echo ""
echo "GPU 信息:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,temperature.gpu --format=csv,noheader
else
    echo "  nvidia-smi 不可用"
fi

# ==================== 环境变量 ====================
# 设置缓存目录
export JOB_CACHE_DIR="/tmp/hf_cache_${SLURM_JOB_ID}"
mkdir -p $JOB_CACHE_DIR

export HF_HOME=$JOB_CACHE_DIR
export HF_DATASETS_CACHE=$JOB_CACHE_DIR/datasets
export TRANSFORMERS_CACHE=$JOB_CACHE_DIR/transformers

# 性能优化
export TOKENIZERS_PARALLELISM=true
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

# PyTorch配置
export CUDA_VISIBLE_DEVICES=0     # 只使用GPU 0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 禁用不需要的功能
export WANDB_DISABLED=true
export HF_DATASETS_IN_MEMORY=0

echo ""
echo "环境变量:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  JOB_CACHE_DIR: $JOB_CACHE_DIR"
echo "=========================================="

# ==================== 激活环境 ====================
echo "激活虚拟环境..."
source ~/workspace/IGAR/.venv/bin/activate

if [ -n "$VIRTUAL_ENV" ]; then
    python_version=$(python --version)
    echo "✓ 虚拟环境: $VIRTUAL_ENV"
    echo "  Python: $python_version"
    
    # 检查依赖
    echo "  检查依赖包:"
    python -c "import torch; print(f'    PyTorch: {torch.__version__}')"
    python -c "import transformers; print(f'    Transformers: {transformers.__version__}')"
else
    echo "✗ 虚拟环境激活失败"
    exit 1
fi
echo "=========================================="

# ==================== 检查文件 ====================
cd ~/workspace/IGAR || exit 1
echo "工作目录: $(pwd)"
echo ""
echo "数据文件检查:"
if [ -f "./data/t5_train.jsonl" ]; then
    lines=$(wc -l < ./data/t5_train.jsonl)
    size=$(du -sh ./data/t5_train.jsonl | cut -f1)
    echo "  ✓ 训练数据: $size, $lines 行"
else
    echo "  ✗ 训练数据不存在"
    exit 1
fi

if [ -f "./data/t5_dev.jsonl" ]; then
    lines=$(wc -l < ./data/t5_dev.jsonl)
    size=$(du -sh ./data/t5_dev.jsonl | cut -f1)
    echo "  ✓ 验证数据: $size, $lines 行"
else
    echo "  ✗ 验证数据不存在"
    exit 1
fi

# 检查模型
model_path="/labmount/public/models/google/google-t5/t5-large"
if [ -d "$model_path" ]; then
    echo "  ✓ 模型存在: $model_path"
else
    echo "  ✗ 模型路径不存在"
    exit 1
fi
echo "=========================================="

# ==================== 系统资源 ====================
echo ""
echo "系统资源:"
echo "  CPU核心数: $(nproc)"
echo "  可用内存: $(free -h | grep Mem | awk '{print $7}')"
echo "  磁盘空间: $(df -h . | tail -1 | awk '{print $4}')"
echo "=========================================="

# ==================== GPU监控 ====================
# 后台监控GPU使用率
monitor_gpu() {
    while true; do
        if command -v nvidia-smi &> /dev/null; then
            gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
            echo "[$(date +%H:%M:%S)] GPU: $gpu_info" >> gpu_monitor.log
        fi
        sleep 30
    done
}

# 启动监控
monitor_gpu &
MONITOR_PID=$!
echo "✓ GPU监控已启动 (PID: $MONITOR_PID)"
echo "=========================================="

# ==================== 开始训练 ====================
echo ""
echo "开始训练: $(date)"
echo "=========================================="
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 运行训练
python src/train_gr_single.py 2>&1 | tee training_output.log

TRAIN_EXIT_CODE=$?

# 计算总时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
echo "训练结束: $(date)"
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo "退出码: $TRAIN_EXIT_CODE"
echo "=========================================="

# ==================== 监控统计 ====================
if [ -f "gpu_monitor.log" ]; then
    echo ""
    echo "GPU使用统计:"
    if command -v awk &> /dev/null; then
        avg_gpu=$(awk -F',' '{sum+=$1; count++} END {printf "%.1f", sum/count}' gpu_monitor.log 2>/dev/null)
        avg_mem=$(awk -F',' '{sum+=$2; count++} END {printf "%.1f", sum/count}' gpu_monitor.log 2>/dev/null)
        echo "  平均GPU利用率: ${avg_gpu}%"
        echo "  平均显存使用: ${avg_mem} MB"
    else
        echo "  无法统计（缺少awk命令）"
    fi
fi

# ==================== 清理 ====================
echo ""
echo "清理资源..."

# 停止监控进程
kill $MONITOR_PID 2>/dev/null

# 清理缓存（可选）
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "训练成功，清理临时缓存..."
    rm -rf $JOB_CACHE_DIR
    rm -f gpu_monitor.log
else
    echo "训练失败，保留缓存用于调试: $JOB_CACHE_DIR"
fi

echo "✓ 清理完成"
echo "=========================================="

exit $TRAIN_EXIT_CODE