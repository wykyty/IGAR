import json
from typing import List

def load_valid_docids_from_jsonl(file_path: str) -> List[str]:
    """
    从 JSONL 文件中提取所有的 semantic_docid
    """
    valid_docids = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 解析单行 JSON
                data = json.loads(line)
                
                # 提取 semantic_docid
                docid = data.get("semantic_docid")
                if docid:
                    valid_docids.append(str(docid))
                    
        return list(set(valid_docids)) # 去重以防万一
    
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return []