# #!/bin/bash
# # 使用vllm自动部署模型，运行脚本后结束部署。

# MODEL_PATH="/data/wyh/IGAR/model/qwen2.5-1.5b-sft-merged"
# GPU_IDS="4"
# PORT=8001
# LOG_FILE="vllm_server.log"

# # 1. 定义加强版清理函数
# cleanup() {
#     echo ""
#     echo "🛑 正在停止 vLLM 服务..."
    
#     # 1. 查找所有占用该端口的 PID (不管是父进程还是子进程)
#     # lsof -t -i:8001 会列出所有相关 PID
#     PIDS=$(lsof -t -i:$PORT 2>/dev/null)
    
#     if [ -n "$PIDS" ]; then
#         echo "   发现占用端口 $PORT 的进程: $PIDS"
#         # 转换为一行，用空格分隔，传给 kill
#         echo "$PIDS" | xargs kill
        
#         # 等待 5 秒让它们优雅退出
#         echo "   等待进程退出..."
#         sleep 5
        
#         # 2. 二次检查：如果还在，强制杀掉 (kill -9)
#         REMAINING_PIDS=$(lsof -t -i:$PORT 2>/dev/null)
#         if [ -n "$REMAINING_PIDS" ]; then
#             echo "⚠️  进程未响应，执行强制清理 (kill -9)..."
#             echo "$REMAINING_PIDS" | xargs kill -9
#         fi
#     else
#         echo "   端口 $PORT 已经被释放。"
#     fi
    
#     # 3. 兜底：通过关键字清理可能残留的 vllm 僵尸进程
#     # 注意：pkill -f 会匹配命令行参数，防止 Ray 进程残留
#     pkill -f "vllm.entrypoints.openai.api_server"
    
#     echo "✅ 环境清理完毕。"
# }

# # 注册 trap，当脚本退出或被中断时，执行 cleanup
# trap cleanup EXIT SIGINT SIGTERM

# # 2. 启动 vLLM 服务
# echo "🚀 正在后台启动 vLLM 服务..."
# CUDA_VISIBLE_DEVICES=$GPU_IDS python -m vllm.entrypoints.openai.api_server \
#     --model $MODEL_PATH \
#     --tensor-parallel-size 1 \
#     --port $PORT \
#     --max-model-len 8192 \
#     --gpu-memory-utilization 0.9 > $LOG_FILE 2>&1 &

# # 这里不再需要记录 PID，因为 cleanup 会通过端口反查

# # 3. 等待服务就绪 (Health Check)
# echo "⏳ 等待模型加载 (检查端口 $PORT)..."
# MAX_RETRIES=60
# count=0

# while true; do
#     HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/v1/models)
    
#     if [ "$HTTP_CODE" == "200" ]; then
#         echo "✅ vLLM 服务已就绪！"
#         break
#     fi

#     if [ $count -ge $MAX_RETRIES ]; then
#         echo "❌ 等待超时，查看日志:"
#         tail -n 10 $LOG_FILE
#         exit 1
#     fi

#     sleep 5
#     ((count++))
#     echo -ne "   加载中... ($count/$MAX_RETRIES)\r"
# done
# echo ""

# 4. 运行你的测试代码
echo "----------------------------------------"
echo "🏃 开始运行测试脚本..."
echo "----------------------------------------"
export HF_ENDPOINT="https://hf-mirror.com"  # 配置hf镜像

# 01. 获取当前时间，格式为：20260315_0428 (年月日_时分)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 02. 定义文件夹名称
DIR_NAME="output/logs/${TIMESTAMP}"

# 03. 创建文件夹 (-p 确保父目录 logs 存在)
mkdir -p "$DIR_NAME"

echo "任务开始，输出将保存至: ${DIR_NAME}/output.txt"

# 4. 运行 Python 脚本并重定向
# >  会将标准输出 (stdout) 保存到文件
# 2>&1 会将错误信息 (stderr) 也一并保存到同一个文件
# python src/test_gr.py 2>&1 | tee "${DIR_NAME}/output.txt"
export PYTHONPATH="/data/wyh/IGAR:$PYTHONPATH"
python -m src.test_gr 2>&1 | tee "${DIR_NAME}/output.txt"

echo "任务完成。日志已记录。"
EXIT_CODE=$?

echo "----------------------------------------"
if [ $EXIT_CODE -eq 0 ]; then
    echo "🎉 测试脚本运行成功！"
else
    echo "⚠️  测试脚本运行失败 (Exit Code: $EXIT_CODE)"
fi

# 脚本结束，自动触发 cleanup