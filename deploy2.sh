MODEL_PATH="/data/wyh/IGAR/model/qwen2.5-1.5b-sft-merged"
GPU_IDS="4"
PORT=8001
LOG_FILE="vllm_server.log"

echo "🚀 正在后台启动 vLLM 服务..."
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 1 \
    --port $PORT \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 > $LOG_FILE 2>&1 &