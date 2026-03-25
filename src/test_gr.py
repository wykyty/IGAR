import logging
import os
from dotenv import load_dotenv

# 必须在 import deepsearcher 之前或初始化 config 之前设置
load_dotenv()  # 1. 加载 .env 文件
api_key = os.getenv("OPENAI_API_KEY") # 2. 从环境变量中获取key

from deepsearcher.offline_loading import load_from_local_files
from deepsearcher.online_query import query
from deepsearcher.configuration import Configuration, init_config
from deepsearcher.vector_db.generative_milvus import GenerativeRetrievalDB
from .scripts.load_docids import load_valid_docids_from_jsonl

httpx_logger = logging.getLogger("httpx")  # disable openai's logger output
httpx_logger.setLevel(logging.WARNING)

# current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace = os.path.dirname(current_dir)
data_path = os.path.join(workspace, "data/corpus_with_docidsv2.jsonl")
# data_path = "/data/wyh/IGAR/examples/data/2wikimultihopqa_corpus.json"

config = Configuration()  # Customize your config here

# 配置 LLM
config.set_provider_config("llm", "OpenAI", {
    "model": "llama3-8B",           
    "base_url": "http://localhost:8000/v1", 
    "api_key": "LOCAL"
})

# 配置 Embedding
# config.set_provider_config("embedding", "OpenAIEmbedding", {
#     "model": "text-embedding-ada-002",
#     "base_url": "http://123.129.219.111:3000/v1", 
#     "api_key": api_key
# })
# config.set_provider_config("embedding", "MilvusEmbedding", {"model": "BAAI/bge-base-en-v1.5"})
config.set_provider_config("embedding", "FastEmbedEmbedding", {"model": "BAAI/bge-base-en-v1.5"})

# 配置 FileLoader
config.set_provider_config("file_loader", "JsonFileLoader", {
    "text_key": "content", #"content","text"
    "id_key": "semantic_docid"  #"reference"."title"
})

# 配置 VectorDB
valid_ids = load_valid_docids_from_jsonl(data_path)

new_valid_ids = [s.replace(" ", "-") for s in valid_ids]

# print(f"{len(valid_ids)} : len valid_ids")
# for ids in new_valid_ids:
#     print(f"{ids}\n") 

config.set_provider_config("vector_db", "GenerativeRetrievalDB", {
    "api_url": "http://localhost:8001/v1/completions",
    "valid_doc_ids": new_valid_ids,
    "uri": "./milvus.db", 
    "dim": 768,
    "token":"root:Milvus", 
    "default_collection": "deep"
})

# config.set_provider_config("vector_db", "Milvus", {
#     "uri": "./milvus.db",
#     "dim": 768
# })

init_config(config=config)

# Hint: You can also load a single file, please execute it in the root directory of the deep searcher project
# load_from_local_files(
#     paths_or_directory=os.path.join(workspace, "data/pdfs/Lixiang2023AnnualReport.pdf"),
#     collection_name="Lixiang_docs",
#     collection_description="Lixiang2023 Documents",
#     # force_new_collection=True, # If you want to drop origin collection and create a new collection every time, set force_new_collection to True
# )

load_from_local_files(
    paths_or_directory=data_path,
    collection_name="LiLiLi",
    collection_description="Li auto",
)

question = "Write a report comparing Milvus with other vector databases."
# question = "Write a report comparing Li Auto's vehicle technology comparing with other vehicle manufacturers."
# question = "When did Lothair Ii's mother die?"

final_answer, all_retrieved_results, consumed_token = query(question, max_iter=1)
print(f"Consumed tokens: {consumed_token}")

# save to file
import json
import numpy as np
import os

def append_query_results(filepath: str, answer: str, results: list, tokens: int):
    # 1. 构造当前这次查询的数据字典
    current_data = {
        "final_answer": answer,
        "consumed_token": tokens,
        "timestamp": "2023-10-27 10:00:00", # 可选：建议加个时间戳区分不同记录
        "all_retrieved_results": []
    }

    # 处理 RetrievalResult 对象转字典
    for res in results:
        res_dict = {
            "text": res.text,
            "reference": res.reference,
            "metadata": res.metadata,
            "score": res.score,
            # 处理 numpy array
            "embedding": res.embedding.tolist() if isinstance(res.embedding, np.ndarray) else res.embedding
        }
        current_data["all_retrieved_results"].append(res_dict)

    # 2. 读取现有数据（如果文件存在）
    existing_data = []
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # 如果文件不为空，则加载
                if content.strip():
                    loaded_data = json.loads(content)
                    # 确保加载进来的是列表，如果之前存的是单个对象，则转为列表
                    if isinstance(loaded_data, list):
                        existing_data = loaded_data
                    else:
                        existing_data = [loaded_data]
        except json.JSONDecodeError:
            print("⚠️ 文件格式错误或为空，将创建新列表。")
            existing_data = []

    # 3. 追加新数据到列表
    existing_data.append(current_data)

    # 4. 覆盖写入更新后的列表
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 数据已追加，当前共有 {len(existing_data)} 条记录。")

# 使用方法
save_filename = "/data/wyh/IGAR/output/history_records.json"
append_query_results(save_filename, final_answer, all_retrieved_results, consumed_token)


