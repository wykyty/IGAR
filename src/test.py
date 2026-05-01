import logging
import os
import argparse
import json
import numpy as np
from datetime import datetime

from deepsearcher.offline_loading import load_from_local_files
from deepsearcher.online_query import query, naive_rag_query
from deepsearcher.configuration import Configuration, init_config
from deepsearcher import configuration 
from deepsearcher.vector_db.generative_milvus import GenerativeRetrievalDB

from dotenv import load_dotenv

# 设置日志
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

def setup_config():
    """配置 LLM 和向量数据库"""
    config = Configuration()
    
    # 配置 LLM
    config.set_provider_config("llm", "OpenAI", {
        "model": model_name,           
        "base_url": base_url, 
        "api_key": api_key
    })
    
    # 配置 Embedding
    config.set_provider_config("embedding", "FastEmbedEmbedding", {
        "model": "BAAI/bge-base-en-v1.5",
        # "batch_size": 32,  # 增加批次大小（默认可能是1或4）
        # "parallel": 4,      # 并行处理
        # "cache_folder": "./embedding_cache",  # 缓存目录
        # "verbose": True  # 显示详细进度
    })
    
    # 配置 FileLoader
    config.set_provider_config("file_loader", "JsonFileLoader", {
        "text_key": "text", # "content",
        "id_key": "docid"
    })
    
    # 可选：配置向量数据库（如果需要）
    # config.set_provider_config("vector_db", "Milvus", {
    #     "uri": "./milvus.db",
    #     "dim": 768
    # })
    
    init_config(config=config)
    return config

def load_data(data_path):
    """加载数据到向量数据库"""
    print(f"📂 正在加载数据: {data_path}")
    load_from_local_files(
        paths_or_directory=data_path,
        collection_name="wiki",
        collection_description="wiki documents",
    )
    print("✅ 数据加载完成")

def append_query_results(filepath: str, answer: str, results: list, tokens: int, query_mode: str, question: str):
    """保存查询结果到文件"""
    current_data = {
        "query_mode": query_mode,
        "question": question,
        "final_answer": answer,
        "consumed_token": tokens,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "all_retrieved_results": []
    }
    
    # 处理 RetrievalResult 对象转字典
    for res in results:
        res_dict = {
            "text": res.text,
            "reference": res.reference,
            "metadata": res.metadata,
            "score": res.score,
            "embedding": res.embedding.tolist() if isinstance(res.embedding, np.ndarray) else res.embedding
        }
        current_data["all_retrieved_results"].append(res_dict)
    
    # 读取现有数据
    existing_data = []
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    loaded_data = json.loads(content)
                    if isinstance(loaded_data, list):
                        existing_data = loaded_data
                    else:
                        existing_data = [loaded_data]
        except json.JSONDecodeError:
            print("⚠️ 文件格式错误或为空，将创建新列表。")
            existing_data = []
    
    # 追加新数据
    existing_data.append(current_data)
    
    # 写入文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 数据已追加，当前共有 {len(existing_data)} 条记录。")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='DeepSearcher 查询脚本')
    parser.add_argument('--rag', action='store_true', help='使用标准 RAG 查询（支持多轮迭代）')
    parser.add_argument('--naive-rag', action='store_true', help='使用 Naive RAG 查询（单次检索）')
    parser.add_argument('--question', type=str, help='查询问题（如果不提供，使用默认问题）')
    parser.add_argument('--max-iter', type=int, default=1, help='RAG 查询的最大迭代次数（仅用于 --rag，默认 1）')
    parser.add_argument('--reload-data', action='store_true', help='重新加载数据到向量数据库')
    parser.add_argument('--output', type=str, default='history_records.json', help='输出文件名（默认 history_records.json）')
    
    args = parser.parse_args()
    
    # 检查是否指定了查询模式
    if not args.rag and not args.naive_rag:
        parser.error("请指定查询模式：--rag 或 --naive-rag")
    
    # 加载环境变量
    load_dotenv()
    global base_url, api_key, model_name
    base_url = os.getenv("BASE_URL")
    api_key = os.getenv("API_KEY")
    model_name = os.getenv("MODEL_NAME")
    
    if not all([base_url, api_key, model_name]):
        print("❌ 错误：请在 .env 文件中配置 BASE_URL, API_KEY, MODEL_NAME")
        return
    
    # 配置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.dirname(current_dir)
    data_path = os.path.join(workspace, "data/corpus_with_ids.jsonl")
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return
    
    # 配置系统
    print("🔧 初始化配置...")
    setup_config()
    
    # 加载数据（如果需要）
    if args.reload_data:
        load_data(data_path)
    else:
        print("ℹ️ 跳过数据加载，使用已有数据（如需重新加载，请使用 --reload-data）")
    
    # 设置查询问题
    default_question = "What are the key manufacturing challenges, supply chain risks, and consumer perception issues we face in continuing to upgrade and commercialize our EREVs in China?"
    question = args.question if args.question else default_question
    
    # 执行查询
    print(f"\n🔍 开始查询（模式: {'RAG' if args.rag else 'Naive RAG'}）")
    print(f"❓ 问题: {question}")
    print("-" * 80)
    
    try:
        if args.rag:
            print(f"📖 使用标准 RAG 查询，最大迭代次数: {args.max_iter}")
            final_answer, all_retrieved_results, consumed_token = query(question, max_iter=args.max_iter)
            query_mode = "RAG"
        else:  # naive-rag
            print("📖 使用 Naive RAG 查询")
            final_answer, all_retrieved_results, consumed_token = naive_rag_query(question)
            query_mode = "Naive RAG"
        
        # 输出结果
        print("\n" + "=" * 80)
        print("📝 最终答案:")
        print("=" * 80)
        print(final_answer)
        print("\n" + "=" * 80)
        print(f"📊 Token 消耗: {consumed_token}")
        print(f"📚 检索到的文档数: {len(all_retrieved_results)}")
        print("=" * 80)
        
        # 保存结果
        output_path = os.path.join(workspace, "output", args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        append_query_results(
            output_path, 
            final_answer, 
            all_retrieved_results, 
            consumed_token, 
            query_mode, 
            question
        )
        
        print(f"\n💾 结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"\n❌ 查询失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()