MODEL_PATH="/home/aizoo/data/workspace/wangyikang/models/Meta-Llama-3-8B-Instruct"
GPU_IDS="0"
TP_SIZE=1
SERVED_MODEL_NAME="llama3-8B"
PORT=8000
LOG_FILE="vllm_server.log"

echo "🚀 正在后台启动 vLLM 服务..."
# CUDA_VISIBLE_DEVICES=$GPU_IDS python -m vllm.entrypoints.openai.api_server \
#     --model $MODEL_PATH \
#     --tensor-parallel-size 2 \
#     --port $PORT \
#     --max-model-len 8192 \
#     --gpu-memory-utilization 0.9 > $LOG_FILE 2>&1 &

CUDA_VISIBLE_DEVICES=$GPU_IDS python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size $TP_SIZE \
    --port $PORT \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.4 \
    --dtype auto \
    --trust-remote-code

