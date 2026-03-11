import os
from dotenv import load_dotenv

# 1. 加载 .env 文件
load_dotenv() 

# 2. 从环境变量中获取 key
api_key = os.getenv("OPENAI_API_KEY")

print(f"我的 API Key 是: {api_key}")