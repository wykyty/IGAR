import json
import asyncio
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

# ================= 配置 =================
INPUT_FILE = "./data/train_mini.jsonl"
EVAL_LOG_FILE = "./data/eval_results.jsonl"
API_KEY = "sk-tTo3MNJgAsRIvFgyuRCWfUKSVkBpIgBtPZi7yKTGGAmspl5D"                   
BASE_URL = "http://123.129.219.111:3000/v1" 
CONCURRENT_REQUESTS = 10 

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

EVAL_PROMPT = """
你是一位专业的金融数据审计员。请根据提供的[文档片段]，对[生成任务]的质量进行严格审查。

[上下文]
文档来源：汽车行业年度报告（如理想汽车年报）
任务类型：ZeroGR 指令化合成数据

[待评审数据]
1. 指令 (Instruction): {instruction}
2. 查询 (Query): {query}
3. 文档内容 (Content): {content}

[审计标准]
- 准确性 (Accuracy): Query 中的数字、年份、专业名词是否与 Content 严格一致？若 Content 没提到具体数值但 Query 问了，视为不通过。
- 独立性 (Independence): Query 应该是一个独立的问题，不应包含“根据这段话”、“文中提到”等指代词。
- 匹配度 (Alignment): Query 是否体现了 Instruction 的意图（如：如果是财务指标指令，Query 必须涉及具体的科目或数据）。

[输出格式 - 必须为 JSON]
{{
    "thought": "请简述你的思考过程，分析 Query 和 Content 的匹配关系",
    "score": 1-5,
    "pass": true/false,
    "reason": "如果分数低于4分，请说明具体问题；如果通过，请填'N/A'"
}}
"""

async def evaluate_sample(sample, semaphore):
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": EVAL_PROMPT.format(
                    instruction=sample['instruction'],
                    query=sample['query'],
                    content=sample['source_content']
                )}],
                response_format={'type': 'json_object'}, # 强制返回 JSON
                temperature=0.3 # 降低随机性，确保评估稳定
            )
            eval_res = json.loads(response.choices[0].message.content)
            # 合并原始数据和评估结果
            sample['eval'] = eval_res
            return sample
        except Exception as e:
            print(f"Eval Error: {e}")
            return None

async def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = [evaluate_sample(s, semaphore) for s in data]
    
    print(f"开始对 {len(data)} 条数据进行质量检查...")
    results = await tqdm.gather(*tasks, desc="Evaluating")
    
    # 过滤失败任务
    results = [r for r in results if r is not None]
    
    # 统计通过率
    passed_count = sum(1 for r in results if r['eval']['pass'])
    
    with open(EVAL_LOG_FILE, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\n检查完毕！")
    print(f"通过率: {passed_count}/{len(results)} ({passed_count/len(results):.2%})")
    print(f"详细报告已保存至: {EVAL_LOG_FILE}")

if __name__ == "__main__":
    asyncio.run(main())