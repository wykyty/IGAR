from deepsearcher.agent import DeepSearch
from deepsearcher.llm import OpenAI  # 或 DeepSeek
from zerogr_adapter import ZeroGRRetriever

# 1. 初始化组件
llm = OpenAI(model="gpt-4o") # 你的规划器 LLM
retriever = ZeroGRRetriever(
    model_path="./models/zerogr_v1", 
    docid_map_path="./data/corpus_with_docids.jsonl"
)

# 2. 定义指令规划 Prompt
PLANNING_PROMPT = """
你是一个专业的检索规划专家。用户会问一个复杂问题。
你需要根据问题类型，生成一个 JSON 格式的检索指令。

支持的指令类型 (instruction):
- "Find documents that describe the latest timeline..." (用于时效性问题)
- "Retrieve relevant legal articles..." (用于政策法规)
- "Compare factual data..." (用于对比)

输出格式示例:
{
    "instruction": "Find documents that describe the latest timeline...",
    "query": "2024年 梯度下降算法 变体"
}

用户问题: {user_query}
请直接输出 JSON，不要废话。
"""

# 3. 定义冲突检测函数 (鲁棒性模块)
def detect_conflicts(docs, user_query):
    # 这里是一个简单的 LLM 调用
    doc_text = "\n\n".join([f"Doc {i}: {d['text']}" for i, d in enumerate(docs)])
    check_prompt = f"""
    基于以下文档回答用户问题：{user_query}
    文档：
    {doc_text}
    
    请检查这些文档是否存在事实冲突（如日期、数据不一致）。
    如果存在冲突，请指出并说明应该采纳哪一个（基于最新日期或权威性）。
    如果没有冲突，输出 "PASS"。
    """
    return llm.chat(check_prompt)

# 4. 主流程 (缝合 Loop)
def run_agent(user_query):
    print(f"用户提问: {user_query}")
    
    # Step 1: 规划 (Planning)
    plan_response = llm.chat(PLANNING_PROMPT.format(user_query=user_query))
    print(f"Agent 规划指令: {plan_response}")
    
    # Step 2: 检索 (Retrieval via ZeroGR)
    # 注意：我们将 JSON 字符串直接传给 search，在 ZeroGRRetriever 里解析
    docs = retriever.search(query=plan_response, top_k=3)
    print(f"ZeroGR 找回 {len(docs)} 篇文档")
    
    # Step 3: 冲突检测 (Conflict Check) - 你的毕设亮点
    conflict_analysis = detect_conflicts(docs, user_query)
    if "PASS" not in conflict_analysis:
        print(f"⚠️ 发现冲突: {conflict_analysis}")
        # 这里可以加入"拒绝证据"的逻辑，比如过滤掉旧文档
    
    # Step 4: 最终生成 (Generation)
    final_context = "\n".join([d['text'] for d in docs])
    final_answer = llm.chat(f"基于背景信息回答: {final_context}\n问题: {user_query}")
    
    print("-" * 20)
    print(f"最终答案: {final_answer}")
    return final_answer

def run_multihop_agent(user_query):
    history = []
    max_steps = 3
    current_query = user_query
    
    for step in range(max_steps):
        # 1. 规划下一步
        # Prompt 需要修改，让它知道之前的历史信息
        plan = llm.chat(f"历史信息: {history}\n当前目标: {current_query}\n请生成下一步检索指令...")
        
        # 2. 检索
        docs = retriever.search(plan)
        
        # 3. 检查是否足够回答
        judge = llm.chat(f"有了这些文档: {docs}，能否回答: {user_query}? 回答YES或NO")
        
        if "YES" in judge:
            break
        
        # 4. 如果不够，更新查询 (Self-Correction)
        history.append(docs)
        current_query = llm.chat("还需要查什么信息？")
    
    # 最后生成答案...

# 测试
if __name__ == "__main__":
    # run_agent("梯度下降算法在2024年的最新优化变体有哪些？")
    run_agent("理想汽车2023年母公司净利润大幅转正，但其自身几乎没有营业收入，这一利润主要来源是什么？是否依赖与子公司的内部交易或资金安排？")