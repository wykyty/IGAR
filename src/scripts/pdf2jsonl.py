import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI 
from tqdm import tqdm

SOURCE_FOLDER = "./data/pdfs"    
OUTPUT_FILE = "./data/corpus_with_docids.jsonl"
API_KEY = "sk-372db93fdb7a45aa8b5adefe01ed92e2"                   
BASE_URL = "https://api.deepseek.com" 

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ZeroGR 核心 Prompt (源自论文 Appendix A)  
ZEROGR_PROMPT = """
Task: Generate a Semantic Document Identifier (DocID) for the provided text.

Requirements:
1. Length: Strictly 6-8 words (terms/words).
2. Term Inclusion: Must include 3-5 core terms directly from the document.
3. Term Positioning: Rank by relevance and importance (highest -> lowest, general -> specific).
4. Formatting: Use lowercase letters, numbers, and spaces only. No punctuation.
5. Content: No articles (a, the), no linking verbs, no auxiliary verbs. Use nouns/adjectives only.
6. Uniqueness: Ensure precise core content representation.

Text:
{text}

DocID:
"""

# 切分文档
def load_and_split_pdfs(folder_path):
    all_chunks = []
    # 使用递归切分器，目标块大小 400 字符，重叠 50 字符保持上下文连贯
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", ".", " ", ""]
    )
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    print(f"检测到 {len(files)} 个PDF文件，开始处理...")

    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            # 切分
            chunks = text_splitter.split_documents(pages)
            
            for i, chunk in enumerate(chunks):
                # 过滤掉太短的垃圾片段
                if len(chunk.page_content) < 50:
                    continue
                    
                chunk_data = {
                    "unique_id": f"{file}_{i}",
                    "content": chunk.page_content.replace("\n", " "), # 清洗换行符
                    "metadata": {
                        "source": file,
                        "page": chunk.metadata.get("page", 0) + 1
                    }
                }
                all_chunks.append(chunk_data)
        except Exception as e:
            print(f"处理文件 {file} 出错: {e}")
            
    print(f"切分完成，共获得 {len(all_chunks)} 个段落片段。")
    return all_chunks

# DocID生成
def generate_docid(text):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",   # 可修改
            messages=[
                {"role": "user", "content": ZEROGR_PROMPT.format(text=text)}
            ],
            temperature=0.1, # 低温度保证生成稳定
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API调用失败: {e}")
        return None


if __name__ == "__main__":
    # 1. 切分文档
    chunks = load_and_split_pdfs(SOURCE_FOLDER)
    
    # 2. 生成 DocID 并保存
    # 只需要跑前500-1000条用于毕设，或者全部跑完
    print("开始生成 Semantic DocIDs...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for chunk in tqdm(chunks):
            doc_id = generate_docid(chunk['content'])  # 调用 LLM 生成 DocID
            

            if doc_id:
                chunk['semantic_docid'] = doc_id
                # 写入一行 JSONL
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                
    print(f"处理完毕！数据已保存至 {OUTPUT_FILE}")