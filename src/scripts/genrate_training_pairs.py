import json
import random
import asyncio
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

# ================= 配置 =================
INPUT_FILE = "./data/corpus_with_docids.jsonl"
OUTPUT_FILE = "./data/train_mini.jsonl"
API_KEY = "sk-372db93fdb7a45aa8b5adefe01ed92e2"
BASE_URL = "https://api.deepseek.com"
CONCURRENT_REQUESTS = 10  # 控制并发数，避免触发 API 频率限制 (Rate Limit)

# 初始化异步客户端
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 1. 定义金融领域指令模板 =================
INSTRUCTION_TEMPLATES = [
    {
        "intent": "financial_metric",
        "instruction": "Identify and retrieve specific financial performance data or key accounting metrics (e.g., revenue, net loss, R&D expenses) for a specific fiscal period."
    },
    {
        "intent": "delivery_status",
        "instruction": "Retrieve documents containing vehicle delivery numbers, market share data, or production capacity information."
    },
    {
        "intent": "risk_factor",
        "instruction": "Find sections describing potential business risks, regulatory challenges, or market uncertainties mentioned in the report."
    },
    {
        "intent": "strategic_goal",
        "instruction": "Locate descriptions of the company's future product roadmap, technology R&D plans (like AD Max/Pro), or long-term strategic visions."
    }
]

# ================= 2. 增强型 Prompt (加入金融专家角色) =================
QUERY_GEN_PROMPT = """
You are an expert Financial Analyst and Synthetic Data Generator.
I will provide you with an excerpt from a corporate annual report (e.g., Li Auto 2022 Annual Report) and a retrieval task instruction.

Your goal is to generate a professional User Query that an analyst or investor would actually search for to find this specific information.

Document Content:
{doc_content}

Task Instruction:
{instruction}

Requirements:
1. The query must be answerable using the provided document.
2. Professional Tone: Use financial terminology (e.g., "YoY growth", "gross margin", "delivery guidance").
3. Language: The query should be in Chinese (Simplified), as the document is a Chinese financial report.
4. Output ONLY the query text.

User Query:
"""

# ================= 核心异步生成逻辑 =================

async def generate_pseudo_query(doc, template, semaphore):
    """
    单个生成任务。使用 semaphore 控制并发。
    """
    async with semaphore:  # 限制同时进行的请求数量
        try:
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "user", "content": QUERY_GEN_PROMPT.format(
                        doc_content=doc['content'], 
                        instruction=template['instruction']
                    )}
                ],
                temperature=0.7,
                max_tokens=100
            )
            query = response.choices[0].message.content.strip()
            
            # 返回组合好的数据，确保不会发生错乱
            return {
                "doc_id": doc['semantic_docid'],
                "instruction": template['instruction'],
                "intent": template['intent'],
                "query": query,
                "source_content": doc['content']
            }
        except Exception as e:
            print(f"Error processing doc {doc.get('semantic_docid')}: {e}")
            return None

async def main():
    # 1. 加载数据
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        corpus = [json.loads(line) for line in f]
    
    # 2. 准备任务列表
    tasks = []
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    print(f"加载了 {len(corpus)} 条文档。正在构建异步任务队列...")
    
    for doc in corpus:
        # 随机选 2 个指令模板
        selected_templates = random.sample(INSTRUCTION_TEMPLATES, 2)
        for temp in selected_templates:
            # 创建协程任务
            tasks.append(generate_pseudo_query(doc, temp, semaphore))
    
    # 3. 异步执行并显示进度条
    # tqdm.gather 会保留原始任务的顺序
    results = await tqdm.gather(*tasks, desc="Generating queries")
    
    # 4. 过滤掉失败的任务并保存
    final_results = [r for r in results if r is not None]
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in final_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\n生成完毕！成功获得 {len(final_results)} 条训练数据。")

if __name__ == "__main__":
    asyncio.run(main())