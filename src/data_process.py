import json
# import numpy as np
# from sklearn.cluster import KMeans, MiniBatchKMeans
# from sentence_transformers import SentenceTransformer

# import os
# from dotenv import load_dotenv
# load_dotenv()
# base_url = "http:localhost:8000"
# api_key = "LOCAL"

# def extract_and_build_corpus(input_files, output_file):

#     """
#     从 2WikiMultihopQA 数据集中提取 Corpus 并去重。
    
#     :param input_files: 包含数据集文件路径的列表，例如 ['train.json', 'dev.json', 'test.json']
#     :param output_file: 输出的 Corpus 文件路径，通常保存为 JSONL 格式
#     """
#     # 使用字典来去重，key 为 title，value 为合并后的 text
#     corpus_dict = {}
    
#     for file_path in input_files:
#         print(f"正在处理文件: {file_path} ...")
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
                
#             for item in data:
#                 # 遍历当前样本的 context
#                 for context_item in item.get('context', []):
#                     title = context_item[0]
#                     sentences = context_item[1]
                    
#                     # 如果这篇文档还没有被加入语料库
#                     if title not in corpus_dict:
#                         # 清理句子两端的空格，并用空格拼接成完整的段落文本
#                         full_text = " ".join([sent.strip() for sent in sentences])
#                         corpus_dict[title] = full_text
                        
#         except Exception as e:
#             print(f"读取或解析 {file_path} 时出错: {e}")

#     # 将去重后的 Corpus 写入 JSON Lines (JSONL) 文件
#     print(f"\n提取完成，共获得 {len(corpus_dict)} 篇独立的文档。正在写入 {output_file} ...")
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for title, text in corpus_dict.items():
#             # 构建目标对象
#             doc_obj = {
#                 "title": title,
#                 "text": text
#             }
#             # 将字典转换为 JSON 字符串并写入，ensure_ascii=False 保证原样输出 Unicode（如中文或特殊符号）
#             f.write(json.dumps(doc_obj, ensure_ascii=False) + '\n')
            
#     print("写入完成！")

# from openai import OpenAI

# def get_cluster_label_from_llm(doc_summaries: list, api_key: str, api_base: str) -> str:
#     """让大模型根据Top5标题+摘要生成一个宏观类别标签"""
    
#     # 将列表拼接成易于阅读的文本格式
#     combined_info = "\n\n".join(doc_summaries)
    
#     prompt = prompt = f"""You are a document clustering label generator. Please extract the shared "Macro Category" or "Theme" based on the titles and texts of the provided articles.

# Requirements:
# 1. Must accurately summarize the core commonality (e.g., American Film Directors, Historic Battles, Machine Learning).
# 2. Output strictly 1 to 4 English words. No full sentences.
# 3. Pay special attention to the disambiguation hints in parentheses within the titles (e.g., "(special effects artist)" or "(1970 film)").
# 4. Output the label directly without any explanation.

# --- Example 1 ---
# Input:
# 【Title】Brian Johnson (special effects artist) 【Text】Brian Johnson( born 1939 or 1940) is a British designer and director...
# 【Title】Stuart Rosenberg 【Text】Stuart Rosenberg was an American film and television director whose motion pictures...
# 【Title】Peter Levin 【Text】Peter Levin is an American director of film, television and theatre.
# Output: Film and Television Directors

# --- Example 2 ---
# Input:
# 【Title】Move (1970 film) 【Text】Move is a 1970 American comedy film starring Elliott Gould...
# 【Title】The Amityville Horror (1979 film) 【Text】The Amityville Horror is a 1979 American supernatural horror film...
# Output: American Films

# --- Actual Task ---
# Input:
# {combined_info}
# Output:"""

#     try:
#         # 初始化 OpenAI 客户端
#         client = OpenAI(
#             api_key=api_key,
#             base_url=api_base
#         )
        
#         # 调用大模型
#         response = client.chat.completions.create(
#             model="./models/Qwen2.5-1.5B-Instruct", # 替换为你实际使用的模型名称
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.1,  # 保持极低的温度，确保标签稳定
#             max_tokens=10,
#             timeout=8.0       # 设置超时时间（秒）
#         )
        
#         # 解析返回结果
#         label = response.choices[0].message.content.strip()
#         return label
        
#     except Exception as e:
#         print(f"API 调用异常: {e}")
#         return "综合文档" # 失败兜底

# def build_hierarchical_ids(input_file, output_file, dict_output_file, k=10, max_leaf_size=5):
#     """
#     为 Corpus 生成层次化聚类ID (支持大规模数据与 JSONL 格式)
#         k: 每次 k-means 聚类的簇数量（树的分支数）
#         max_leaf_size: 叶子节点最大包含的文档数，低于此值停止聚类
#     """
#     print("1. 加载数据 (JSONL 格式)...")
#     data = []
#     with open(input_file, "r", encoding="utf-8") as f:
#         for line in f:
#             data.append(json.loads(line))
            
#     texts = [f"{doc.get('title', '')} {doc.get('text', '')}" for doc in data]

#     print("2. 加载向量模型生成 Embeddings...")
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     # 显式转换为 numpy 数组，方便后续进行高效的切片操作
#     embeddings = model.encode(texts, show_progress_bar=True, batch_size=256, convert_to_numpy=True)

#     doc_ids = [""] * len(data)
    
#     # [新增] 初始化前缀语义映射字典
#     prefix_semantics_map = {}

#     def cluster_recursive(indices, current_prefix):
#         n_samples = len(indices)
        
#         # 递归终止条件 1：文档数量足够少，直接分配叶子节点 ID
#         if n_samples <= max_leaf_size:
#             for i, idx in enumerate(indices):
#                 # 最终生成的 ID 类似于 "3.2.1.0"
#                 doc_ids[idx] = f"{current_prefix}{i}"
#             return

#         # 动态调整 K 值（防止剩余文档少于 K）
#         n_clusters = min(k, n_samples)
        
#         if n_samples > 10000:
#             kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2048, n_init=3)
#         else:
#             kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            
#         subset_embeddings = embeddings[indices]
#         labels = kmeans.fit_predict(subset_embeddings)

#         # 极端情况处理：如果所有文档被分到同一个簇（向量极度相似）
#         if len(set(labels)) == 1:
#             for i, idx in enumerate(indices):
#                 doc_ids[idx] = f"{current_prefix}{i}"
#             return

#         indices_arr = np.array(indices)
        
#         # [新增] 获取当前所有簇的质心
#         centroids = kmeans.cluster_centers_
        
#         for cluster_id in range(n_clusters):
#             cluster_mask = (labels == cluster_id)
#             cluster_indices = indices_arr[cluster_mask].tolist()
            
#             if len(cluster_indices) > 0:
#                 next_prefix = f"{current_prefix}{cluster_id}."
                
#                 # ================= 语义展开核心逻辑 =================
#                 current_cluster_embeddings = subset_embeddings[cluster_mask]
#                 dists = np.linalg.norm(current_cluster_embeddings - centroids[cluster_id], axis=1)
                
#                 # 1. 取出距离最近的 Top 5 篇文档的索引
#                 top_k = min(5, len(current_cluster_embeddings))
#                 top_indices_in_subset = np.argsort(dists)[:top_k]
#                 global_top_indices = [cluster_indices[idx] for idx in top_indices_in_subset]
                
#                 # 2. 提取它们的标题和内容摘要 (加入截断机制)
#                 doc_summaries = []
#                 for idx in global_top_indices:
#                     doc = data[idx]
#                     title = doc.get("title", "无标题")
#                     # 截取正文前 150 个字符，并替换掉换行符，保持格式整洁
#                     text_snippet = doc.get("text", "")
                    
#                     # 组装成单篇文档的信息块
#                     doc_summaries.append(f"【标题】{title}\n【摘要】{text_snippet}...")
                
#                 # 3. 调用 LLM 生成宏观标签
#                 semantic_repr = get_cluster_label_from_llm(doc_summaries, api_key, base_url)
                
#                 # 4. 存入字典
#                 prefix_semantics_map[next_prefix] = semantic_repr
#                 # =====================================================
#                 # =====================================================

#                 # 继续递归
#                 cluster_recursive(cluster_indices, next_prefix)

#     print(f"3. 开始执行层次化 K-Means 聚类 (总计 {len(data)} 篇文档)...")
#     all_indices = list(range(len(data)))
#     cluster_recursive(all_indices, "")

#     print("4. 保存带有语义 ID 的 Corpus 到 JSONL 文件...")
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for i, doc in enumerate(data):
#             new_doc = {
#                 "docid": doc_ids[i],
#                 "title": doc.get("title", ""),
#                 "text": doc.get("text", "")
#             }
#             f.write(json.dumps(new_doc, ensure_ascii=False) + '\n')
            
#     # [新增] 保存前缀语义映射字典
#     print(f"5. 保存前缀语义字典到 {dict_output_file} ...")
#     with open(dict_output_file, 'w', encoding='utf-8') as f:
#         json.dump(prefix_semantics_map, f, ensure_ascii=False, indent=4)
            
#     print(f"完成！已生成 {len(data)} 条数据，保存至 {output_file}")
    
#     # 打印前几条作为示例展示
#     print("\n生成的层次化 ID 示例:")
#     for i in range(min(5, len(data))):
#         print(f"ID: {doc_ids[i]:<15} | Title: {data[i].get('title', '')}")
        
#     print("\n生成的前缀语义示例:")
#     for prefix, semantics in list(prefix_semantics_map.items())[:5]:
#         print(f"Prefix: {prefix:<8} | Semantics: {semantics}")

def build_seq2seq_dataset(train_json_file, corpus_id_file, output_indexing_file, output_retrieval_file):
    """
    为生成式检索模型构建 Seq2Seq 格式的训练数据。
    采用 One-to-Many Baseline 策略：一个 Query 对应多个独立的目标 DocID。
    """
    print("1. 正在加载 Corpus 并建立 Title 到 DocID 的映射字典...")
    title_to_docid = {}
    
    # ---------------- 1. 构建 Indexing (记忆) 数据集 ----------------
    print(f"2. 正在生成 Indexing 训练数据并保存至 {output_indexing_file}...")
    with open(corpus_id_file, 'r', encoding='utf-8') as f_in, \
         open(output_indexing_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            doc = json.loads(line)
            title = doc['title']
            text = doc['text']
            docid = doc['docid']
            
            # 保存映射供 Retrieval 阶段使用
            title_to_docid[title] = docid
            
            # Prefix 'Doc: ' 帮助模型区分当前是记忆任务
            input_text = f"Document: {title} {text}"
            
            indexing_sample = {
                "input_text": input_text,
                "target_text": docid,
                "task_type": "indexing"
            }
            f_out.write(json.dumps(indexing_sample, ensure_ascii=False) + '\n')

    # ---------------- 2. 构建 Retrieval (检索) 数据集 ----------------
    print(f"3. 正在生成 Retrieval 训练数据 (One-to-Many 策略) 并保存至 {output_retrieval_file}...")
    with open(train_json_file, 'r', encoding='utf-8') as f_in, \
         open(output_retrieval_file, 'w', encoding='utf-8') as f_out:
        
        train_data = json.load(f_in)
        missing_doc_count = 0
        total_queries = 0
        total_retrieval_samples = 0
        
        for item in train_data:
            question = item['question']
            
            # 提取 supporting_facts 中的黄金段落标题并去重
            gold_titles = []
            for fact in item.get('supporting_facts', []):
                title = fact[0]
                if title not in gold_titles:
                    gold_titles.append(title)
            
            # 将 Title 转换为对应的 Semantic ID
            gold_docids = []
            for title in gold_titles:
                if title in title_to_docid:
                    gold_docids.append(title_to_docid[title])
                else:
                    missing_doc_count += 1
            
            # 核心修改：一对多拆分逻辑
            if gold_docids:
                total_queries += 1
                input_text = f"Query: {question}"
                
                # 为该 Query 依赖的每一个 DocID 生成一条独立的训练数据
                for docid in gold_docids:
                    retrieval_sample = {
                        "input_text": input_text,
                        "target_text": docid,
                        "task_type": "retrieval",
                        "total_gold_docs": len(gold_docids) # 记录该问题原本需要几个文档，方便后续分析
                    }
                    f_out.write(json.dumps(retrieval_sample, ensure_ascii=False) + '\n')
                    total_retrieval_samples += 1

    print("\n构建完成！")
    print(f"统计信息:")
    print(f"- 处理的有效 Query 数量: {total_queries}")
    print(f"- 拆分后生成的 Retrieval 训练样本总数: {total_retrieval_samples}")
    print(f"- 平均每个 Query 被拆分为 {total_retrieval_samples/total_queries:.2f} 条数据")
    print(f"- 未匹配到 ID 的文档数量 (可能因为清洗被过滤): {missing_doc_count}")

import json
import random

def mix_and_shuffle_datasets(indexing_file, retrieval_file, output_file, seed=42):
    """
    将 Indexing 和 Retrieval 数据集合并，并进行全局随机打乱。
    """
    print("1. 正在读取 Indexing 和 Retrieval 数据...")
    combined_data = []
    
    # 读取 Indexing 数据
    with open(indexing_file, 'r', encoding='utf-8') as f:
        for line in f:
            combined_data.append(json.loads(line))
    print(f"   - Indexing 数据量: {len(combined_data)}")
            
    # 读取 Retrieval 数据
    retrieval_start_idx = len(combined_data)
    with open(retrieval_file, 'r', encoding='utf-8') as f:
        for line in f:
            combined_data.append(json.loads(line))
    print(f"   - Retrieval 数据量: {len(combined_data) - retrieval_start_idx}")
    print(f"   - 合并后总数据量: {len(combined_data)}")

    # 设置随机种子以保证可复现性
    print(f"2. 正在进行全局随机打乱 (Random Seed: {seed})...")
    random.seed(seed)
    random.shuffle(combined_data)

    # 写入混合后的新文件
    print(f"3. 正在将打乱后的数据保存至 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in combined_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("完成！混合数据集已准备就绪。")
    
    # 打印前 5 条数据，检查打乱效果
    print("\n打乱后的前 5 条数据示例：")
    for i in range(5):
        task = combined_data[i].get('task_type', 'unknown')
        print(f"[{task}] Input: {combined_data[i]['input_text'][:50]}... -> Target: {combined_data[i]['target_text']}")

import json

def build_dev_dataset(dev_json_file, corpus_id_file, output_dev_file):
    """
    为生成式检索模型构建纯净的 Dev 验证集。
    仅包含 Query -> DocID 的检索任务，不包含 Indexing 记忆任务。
    """
    print("1. 正在加载 Corpus 并建立 Title 到 DocID 的映射字典...")
    title_to_docid = {}
    
    with open(corpus_id_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            doc = json.loads(line)
            title_to_docid[doc['title']] = doc['docid']

    print(f"2. 正在读取 {dev_json_file} 生成 Dev 验证集...")
    with open(dev_json_file, 'r', encoding='utf-8') as f_in, \
         open(output_dev_file, 'w', encoding='utf-8') as f_out:
        
        dev_data = json.load(f_in)
        total_queries = 0
        total_samples = 0
        
        for item in dev_data:
            question = item['question']
            
            # 提取黄金段落标题并去重
            gold_titles = []
            for fact in item.get('supporting_facts', []):
                title = fact[0]
                if title not in gold_titles:
                    gold_titles.append(title)
            
            # 转换为 Semantic ID
            gold_docids = [title_to_docid[t] for t in gold_titles if t in title_to_docid]
            
            # 采用与训练集一致的一对多拆分策略
            if gold_docids:
                total_queries += 1
                input_text = f"Query: {question}"
                
                for docid in gold_docids:
                    dev_sample = {
                        "input_text": input_text,
                        "target_text": docid,
                        "task_type": "retrieval_eval"
                    }
                    f_out.write(json.dumps(dev_sample, ensure_ascii=False) + '\n')
                    total_samples += 1

    print("\n构建完成！")
    print(f"- 处理的有效 Dev Query 数量: {total_queries}")
    print(f"- 生成的验证集样本总数: {total_samples}")


if __name__ == "__main__":
    # 1. 提取corpus
    train_file = './data/2wiki/train.json'
    dev_file = './data/2wiki/dev.json'
    test_file = './data/2wiki/test.json'
    datasets = [train_file, dev_file, test_file] 
      
    corpus = './data/corpus.jsonl' 
    # extract_and_build_corpus(datasets, corpus)
    '''
    提取完成，共获得 398354 篇独立的文档。正在写入 data/corpus.jsonl ...
    写入完成！
    '''

    # 2. 聚类出结构化语义docid
    corpus_ids = './data/corpus_with_ids.jsonl'
    dict_file = './data/prefix_semantics.json'
    # build_hierarchical_ids(corpus, corpus_ids, dict_file, k=10, max_leaf_size=5)
    """
    聚类完成.
    """
    
    # 3. Seq2Seq训练数据构建脚本
    indexing_train = './data/t5_indexing_train.jsonl'
    retrieval_train = './data/t5_retrieval_train.jsonl'
    # build_seq2seq_dataset(train_file, corpus_ids, indexing_train, retrieval_train)
    """
    python .\src\data_process.py
    (igar) PS D:\WorkSpace\IGAR> python .\src\data_process.py                                         
    1. 正在加载 Corpus 并建立 Title 到 DocID 的映射字典...                            
    2. 正在生成 Indexing 训练数据并保存至 ./data/t5_indexing_train.jsonl...
    3. 正在生成 Retrieval 训练数据 (One-to-Many 策略) 并保存至 ./data/t5_retrieval_train.jsonl...

    构建完成！
    统计信息:
    - 处理的有效 Query 数量: 167454
    - 拆分后生成的 Retrieval 训练样本总数: 404170
    - 平均每个 Query 被拆分为 2.41 条数据
    - 未匹配到 ID 的文档数量 (可能因为清洗被过滤): 0
    """

    # 4. 全局随机打乱 Indexing 和 Retrieval 数据 
    t5_train = './data/t5_train.jsonl'
    # mix_and_shuffle_datasets(indexing_train, retrieval_train, t5_train)
    """
    python .\src\data_process.py
    (igar) PS D:\WorkSpace\IGAR> python .\src\data_process.py
    1. 正在读取 Indexing 和 Retrieval 数据...
    - Indexing 数据量: 398354
    - Retrieval 数据量: 404170
    - 合并后总数据量: 802524
    2. 正在进行全局随机打乱 (Random Seed: 42)...
    3. 正在将打乱后的数据保存至 ./data/t5_train.jsonl...
    完成！混合数据集已准备就绪。

    打乱后的前 5 条数据示例：
    [indexing] Input: Document: List of Ojarumaru characters This is a l... -> Target: 7.1.3.6.4.0
    [indexing] Input: Document: The Scent of the Night (film) The Scent ... -> Target: 9.3.9.7.0.0
    [retrieval] Input: Query: Which film has the director died earlier, T... -> Target: 1.4.9.4.4.3.0
    [retrieval] Input: Query: Do both films, Handle with Care (1977 film)... -> Target: 4.6.9.0.5.1.0
    [indexing] Input: Document: Richard Ledes Richard Ledes is an Americ... -> Target: 4.6.5.7.2.8.0
    """

    # 5. Dev 验证集生成    
    # 输出给大模型做 eval 的数据集
    t5_dev = './data/t5_dev.jsonl'
    build_dev_dataset(dev_file, corpus_ids, t5_dev)
    """
    python .\src\data_process.py
    (igar) PS D:\WorkSpace\IGAR> python .\src\data_process.py
    1. 正在加载 Corpus 并建立 Title 到 DocID 的映射字典...
    2. 正在读取 ./data/2wiki/dev.json 生成 Dev 验证集...

    构建完成！
    - 处理的有效 Dev Query 数量: 12576
    - 生成的验证集样本总数: 30654
    """

