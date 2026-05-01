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

# 04. 运行 Python 脚本并重定向
python src/test.py --rag --question "When did Lothair Ii's mother die?" 2>&1 | tee "${DIR_NAME}/output.txt"

echo "任务完成。日志已记录。"
