# export HF_ENDPOINT="https://hf-mirror.com"

# # 1. naive_rag + dense retrieval
# python src/evaluate_gr.py --max-samples 50  

# # 2. simple_searcher + dense retrieval
# python src/evaluate_gr.py --query-mode simple_searcher --retrieval-method dr --max-samples 50

# # 3. simple_searcher + generative retrieval
# CUDA_VISIBLE_DEVICES=2,3 python src/evaluate_gr.py --query-mode simple_searcher --retrieval-method gr --max-samples 200


#!/bin/bash

# 1. 准备全局输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WORKSPACE=$(pwd)
BATCH_OUTPUT_DIR="${WORKSPACE}/output/batch_eval_${TIMESTAMP}"

mkdir -p "${BATCH_OUTPUT_DIR}"

# 定义统一个主日志文件，方便后续排查错误
MASTER_LOG="${BATCH_OUTPUT_DIR}/master_run.log"

echo "=================================================================" | tee -a "$MASTER_LOG"
echo "开始执行 IGAR 毕设多跳检索 4 组核心对比实验" | tee -a "$MASTER_LOG"
echo "批次输出总目录: ${BATCH_OUTPUT_DIR}" | tee -a "$MASTER_LOG"
echo "主日志文件: ${MASTER_LOG}" | tee -a "$MASTER_LOG"
echo "=================================================================" | tee -a "$MASTER_LOG"

# # 实验 1：传统单跳向量检索 (Naive Dense RAG)
# # 注意：加了 --reload-data 确保向量库被初始化。如果你确认 Milvus 库已经建好，可以删掉这个参数。
# echo "" | tee -a "$MASTER_LOG"
# echo "[$(date +'%T')] >>> 正在运行实验 1/4: Naive Dense RAG (dr, max-iter=1)" | tee -a "$MASTER_LOG"
# python src/evaluate_gr.py \
#     --query-mode naive_rag \
#     --retrieval-method dr \
#     --max-iter 1 \
#     --max-samples 2000 \
#     --output-dir "${BATCH_OUTPUT_DIR}" \
#     --reload-data 2>&1 | tee -a "$MASTER_LOG"

# # 实验 2：传统多跳向量检索 (Agentic Dense RAG)
# echo "" | tee -a "$MASTER_LOG"
# echo "[$(date +'%T')] >>> 正在运行实验 2/4: Agentic Dense RAG (dr, max-iter=3)" | tee -a "$MASTER_LOG"
# python src/evaluate_gr.py \
#     --query-mode deep_searcher \
#     --retrieval-method dr \
#     --max-iter 3 \
#     --max-samples 2000 \
#     --output-dir "${BATCH_OUTPUT_DIR}" 2>&1 | tee -a "$MASTER_LOG"

# 实验 3：单跳生成式检索 (Naive GR RAG)
echo "" | tee -a "$MASTER_LOG"
echo "[$(date +'%T')] >>> 正在运行实验 3/4: Naive GR RAG (gr, max-iter=1)" | tee -a "$MASTER_LOG"
CUDA_VISIBLE_DEVICES=2,3 python src/evaluate_gr.py \
    --query-mode naive_rag \
    --retrieval-method gr \
    --max-iter 1 \
    --max-samples 2000 | tee -a "$MASTER_LOG"

# 实验 4：指令化生成式多跳检索 (IGAR 完整版)
echo "" | tee -a "$MASTER_LOG"
echo "[$(date +'%T')] >>> 正在运行实验 4/4: IGAR 完全体 (gr, max-iter=3)" | tee -a "$MASTER_LOG"
CUDA_VISIBLE_DEVICES=2,3 python src/evaluate_gr.py \
    --query-mode deep_searcher \
    --retrieval-method gr \
    --max-iter 3 \
    --max-samples 2000 | tee -a "$MASTER_LOG"

echo "=================================================================" | tee -a "$MASTER_LOG"
echo "[$(date +'%T')] 所有实验执行完毕！" | tee -a "$MASTER_LOG"
echo "请前往 ${BATCH_OUTPUT_DIR} 查看 4 份独立的 json 结果文件。" | tee -a "$MASTER_LOG"