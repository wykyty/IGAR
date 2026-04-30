import json

# 建议将输入和输出逻辑放在一起，避免大文件占用内存
input_path = "./data/corpus_with_docids.jsonl"
output_path = "./data/corpus_with_docidsv2.jsonl" # 建议加上后缀

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    
    for line in fin:
        # 1. 使用 loads 解析字符串
        data = json.loads(line.strip()) 
        
        # 2. 修改数据
        docid = data.get("semantic_docid", "")
        data["semantic_docid"] = docid.replace(" ", "-")
        
        # 3. 序列化并写入文件
        # ensure_ascii=False 保证中文不被转义成 \uXXXX
        json_line = json.dumps(data, ensure_ascii=False)
        fout.write(json_line + "\n")
