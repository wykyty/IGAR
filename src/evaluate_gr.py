import json
import os
import logging
from tqdm import tqdm

from deepsearcher.online_query import query, naive_rag_query
from deepsearcher.configuration import Configuration, init_config
from deepsearcher.offline_loading import load_from_local_files

from src.utils import load_valid_docids
from dotenv import load_dotenv

load_dotenv()
base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
vllm_logger = logging.getLogger("vllm")
vllm_logger.propagate = False 
vllm_logger.handlers.clear()
vllm_file_handler = logging.FileHandler("vllm_server_log.log", encoding="utf-8")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
vllm_file_handler.setFormatter(formatter)
vllm_logger.setLevel(logging.INFO) 
vllm_logger.addHandler(vllm_file_handler)

# ------------------------------------------------------------------
# 初始化 Config (保持你原有的配置)
# ------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace = os.path.dirname(current_dir)
corpus_path = os.path.join(workspace, "data/corpus.jsonl")
dev_json_path = os.path.join(workspace, "data/2wiki/dev.json") # 2Wiki 官方 dev 集
gr_model_path = os.path.join(workspace, "model/t5_base_gr")
save_filename = os.path.join(workspace, "output/evaluation_results.json")

config = Configuration()
# config.set_provider_config("llm", "OpenAI", {
#     "model": "./models/Qwen2.5-1.5B-Instruct",           
#     "base_url": "http://localhost:8000/v1", 
#     "api_key": "LOCAL"
# })
config.set_provider_config("llm", "OpenAI", {
    "model": "LongCat-Flash-Lite",
    "base_url": base_url,
    "api_key": api_key
})
config.set_provider_config("embedding", "SentenceTransformerEmbedding", {
    "model": "BAAI/bge-large-en-v1.5",
    "batch_size": 256
})

config.set_provider_config("file_loader", "JsonFileLoader", {
    "text_key": "text", 
    "id_key": "docid" 
})

# valid_docids = load_valid_docids(corpus_path)
# config.set_provider_config("vector_db", "GenerativeRetrievalDB", {
#     "gr_model_path": gr_model_path,
#     "valid_doc_ids": valid_docids,
#     "uri": "./milvus.db", 
#     "token":"root:Milvus"
# })

init_config(config=config)

load_from_local_files(
    paths_or_directory=corpus_path,
    collection_name="wiki",
    collection_description="2wiki corpus"
)

def load_title_to_docid_mapping(corpus_file):
    mapping = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            mapping[doc['title']] = doc['docid']
    return mapping

def evaluate_retrieval_with_debug(dev_file, corpus_file, output_file, invalid_log_file, max_samples=2400):
    print("1. 正在加载 Title -> DocID 映射表...")
    title_to_docid = load_title_to_docid_mapping(corpus_file)
    print(f"   - Corpus 中共加载了 {len(title_to_docid)} 篇文档的映射。")
    
    print(f"2. 正在加载验证集数据 {dev_file}...")
    with open(dev_file, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
        
    if max_samples:
        dev_data = dev_data[:max_samples]
        
    metrics = {
        "strict_hits@1": 0, "strict_hits@5": 0, "strict_hits@10": 0,
        "partial_hits@1": 0, "partial_hits@5": 0, "partial_hits@10": 0,
        "total_valid_queries": 0,
    }
    
    detailed_results = []
    invalid_queries_log = [] # 🌟 新增：专门记录无效 Query 的原因
    
    print(f"3. 开始评估 {len(dev_data)} 条多跳查询...")
    for item in tqdm(dev_data, desc="Evaluating"):
        question = item['question']
        
        # 提取 Gold Titles
        gold_titles = list(set([fact[0] for fact in item.get('supporting_facts', [])]))
        
        # 🌟 排错核心逻辑 1：检查为什么找不到 DocID
        missing_titles = [t for t in gold_titles if t not in title_to_docid]
        if missing_titles:
            # 如果存在哪怕一个缺失的 title，记录到无效日志并跳过
            invalid_queries_log.append({
                "question": question,
                "gold_titles": gold_titles,
                "missing_titles_in_corpus": missing_titles,
                "reason": "部分或全部支撑文档不在语料库的 title_to_docid 映射中"
            })
            continue # 如果你想放宽要求，可以注释掉这行，但 Strict Hits 会永远是 0
            
        gold_docids = [title_to_docid[t] for t in gold_titles]
        metrics["total_valid_queries"] += 1
        
        try:
            # 调用检索
            _, retrieved_results = naive_rag_query(question)
            
            # 🌟 排错核心逻辑 2：提取预测的 ID
            pred_docids = []
            for res in retrieved_results:
                # 兼容 deepsearcher 的不同返回格式
                docid = res.metadata.get("docid") if res.metadata else getattr(res, "reference", "")
                if docid:
                    pred_docids.append(str(docid)) # 确保转为字符串比较
                    
        except Exception as e:
            invalid_queries_log.append({"question": question, "reason": f"RAG Pipeline 报错: {e}"})
            continue

        # 打印前几个有效 Query 的对比，帮你肉眼 Debug
        if metrics["total_valid_queries"] <= 3:
            print(f"\n[Debug] Question: {question[:50]}...")
            print(f"   -> 真实的 Semantic ID (Gold) : {gold_docids}")
            print(f"   -> Deepsearcher 返回的 ID (Pred): {pred_docids[:3]}...")
            
        # 计算 Hits@K
        gold_set = set(gold_docids)
        for k in [1, 5, 10]:
            top_k_preds = set(pred_docids[:k])
            overlap = gold_set.intersection(top_k_preds)
            if len(overlap) > 0:
                metrics[f"partial_hits@{k}"] += 1
            if len(overlap) == len(gold_set):
                metrics[f"strict_hits@{k}"] += 1
                
        detailed_results.append({
            "question": question,
            "gold_docids": gold_docids,
            "pred_docids": pred_docids
        })

    total = metrics["total_valid_queries"]
    print("\n" + "="*40)
    if total > 0:
        print("📍 严格召回率 (Strict Hits):")
        print(f"  Hits@10: {metrics['strict_hits@10'] / total * 100:.2f}%\n")
        print("📍 宽松召回率 (Partial Hits):")
        print(f"  Hits@10: {metrics['partial_hits@10'] / total * 100:.2f}%")
    print("="*40)

    # 🌟 保存有效结果和无效日志
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"summary": metrics, "details": detailed_results}, f, ensure_ascii=False, indent=4)
        
    with open(invalid_log_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_queries_log, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 有效评估已保存至: {output_file}")
    print(f"🚨 注意：共有 {len(invalid_queries_log)} 条 Query 被判为无效，原因已保存至: {invalid_log_file}")
    print("请务必打开 invalid_log_file 查看具体的 missing_titles！")

if __name__ == "__main__":
    # 配置你的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.dirname(current_dir)
    dev_json_path = os.path.join(workspace, "data/2wiki/dev.json")
    corpus_path = os.path.join(workspace, "data/corpus.jsonl")
    
    save_filename = os.path.join(workspace, "output/evaluation_results.json")
    invalid_log_file = os.path.join(workspace, "output/invalid_queries_log.json") # 新增排错日志路径
    
    evaluate_retrieval_with_debug(dev_json_path, corpus_path, save_filename, invalid_log_file, max_samples=200)