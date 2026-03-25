import json
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

def build_hierarchical_ids(input_file, output_file, k = 10, max_leaf_size = 5):
    """
    为 Corpus 生成层次化聚类ID
        k: 每次k-means 聚类的簇数量（树的分支数）
        max_leaf_size: 叶子节点最大包含的文档数，低于此值停止聚类
    """
    print("1. 加载数据...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [f"{doc.get('title', '')} {doc.get('text', '')}" for doc in data]

    print("2. 加载向量模型生成Embeddigns...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)

    doc_ids = [""] * len(data)

    def cluster_recursive(indices, current_prefix):
        # 递归终止条件 1：文档数量足够少，直接分配叶子节点 ID
        if len(indices) <= max_leaf_size:
            for i, idx in enumerate(indices):
                doc_ids[idx] = f"{current_prefix}{i}"
            return

        # 递归终止条件 2：动态调整 K 值（防止剩余文档少于 K）
        n_clusters = min(k, len(indices))
        
        # 执行 K-Means 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        subset_embeddings = embeddings[indices]
        labels = kmeans.fit_predict(subset_embeddings)

        # 极端情况处理：如果所有文档被分到同一个簇（向量极度相似）
        if len(set(labels)) == 1:
            for i, idx in enumerate(indices):
                doc_ids[idx] = f"{current_prefix}{i}"
            return

        # 为每个簇继续递归
        for cluster_id in range(n_clusters):
            # 找出属于当前簇的文档索引
            cluster_indices = [indices[i] for i in range(len(indices)) if labels[i] == cluster_id]
            if len(cluster_indices) > 0:
                # 前缀拼接，例如 "3." -> "3.2."
                next_prefix = f"{current_prefix}{cluster_id}."
                cluster_recursive(cluster_indices, next_prefix)

    print("3. 开始执行层次化 K-Means 聚类...")
    all_indices = list(range(len(data)))
    cluster_recursive(all_indices, "")

    print("4. 保存带有语义 ID 的 Corpus...")
    corpus_with_semantic_ids = []
    for i, doc in enumerate(data):
        corpus_with_semantic_ids.append({
            "docid": doc_ids[i],
            "title": doc.get("title", ""),
            "text": doc.get("text", "")
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(corpus_with_semantic_ids, f, ensure_ascii=False, indent=4)
        
    print(f"完成！已生成 {len(data)} 条数据，保存至 {output_file}")
    
    # 打印前几条作为示例展示
    print("\n生成的 ID 示例:")
    for i in range(min(5, len(corpus_with_semantic_ids))):
        print(f"ID: {corpus_with_semantic_ids[i]['docid']:<15} | Title: {corpus_with_semantic_ids[i]['title']}")

# 运行脚本
# 请将 'raw_data.json' 替换为你的输入文件，'corpus_semantic_ids.json' 为输出文件
build_hierarchical_ids('/data/wyh/IGAR/examples/data/2wikimultihopqa_corpus.json', '/data/wyh/IGAR/data/corpus_semantic_ids.json', k=10, max_leaf_size=5)