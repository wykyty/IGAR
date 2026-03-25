import json

def build_seq2seq_training_data(corpus_file, query_file, output_file):
    print("1. 加载带有 docid 的 Corpus，建立 Title 到 docid 的映射...")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
        
    # 构建映射字典：Key 是 Title, Value 是对应的 docid
    # 注意：2Wiki 的 supporting_facts 使用的是文档的 title
    title_to_docid = {doc['title']: doc['docid'] for doc in corpus_data}
    
    print("2. 加载 2Wiki 格式的 Query 数据...")
    with open(query_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    training_pairs = []
    missing_docs = 0

    print("3. 开始构造 (Input, Target) 训练对...")
    for q_item in queries:
        question = q_item.get("question", "").strip()
        
        # 提取支持文档的 title (去重，因为有时候同一个文档的不同句子会被引用多次)
        support_titles = set([fact[0] for fact in q_item.get("supporting_facts", [])])
        
        for title in support_titles:
            # 查找该 title 对应的 docid
            if title in title_to_docid:
                target_docid = title_to_docid[title]
                
                # 构造 Seq2Seq 格式
                # 可以选择加上 prompt prefix，例如 "Generate docid for query: "
                training_pairs.append({
                    "input_text": f"Retrieve docid for: {question}",
                    "target_text": target_docid,
                    "query_id": q_item.get("_id"), # 保留原始问题ID，方便评测
                    "hop_type": q_item.get("type")
                })
            else:
                missing_docs += 1
                
    print(f"4. 保存训练集，共生成 {len(training_pairs)} 条样本...")
    # 保存为 JSONL 格式（每行一个 JSON），这是大模型微调最标准的格式
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in training_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
            
    if missing_docs > 0:
        print(f"⚠️ 警告: 发现 {missing_docs} 个 supporting_fact 的 title 在 corpus 中找不到对应的 docid。")
    print("处理完成！")

# 运行示例（请替换为你真实的文件名）
# build_seq2seq_training_data('./data/corpus_semantic_ids.json', './data/2wiki/train.json', './data/seq2seq_train_data.jsonl')


import json

def build_indexing_data(corpus_file, output_file):
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
        
    indexing_pairs = []
    
    for doc in corpus_data:
        # 拼接标题和正文
        content = f"Title: {doc.get('title', '')}. Text: {doc.get('text', '')}"
        
        # 构造 Seq2Seq 格式，加上明确的前缀
        indexing_pairs.append({
            "input_text": f"Memorize document: {content}",
            "target_text": doc["docid"],
            "task_type": "indexing"
        })
        
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in indexing_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
            
    print(f"已生成 {len(indexing_pairs)} 条文档记忆（Indexing）数据！")

build_indexing_data('./data/corpus_semantic_ids.json', './data/seq2seq_indexing_data.jsonl')