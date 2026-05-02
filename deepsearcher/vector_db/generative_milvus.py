from typing import List, Optional, Tuple, Union
import numpy as np
import httpx
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import LogitsProcessor, LogitsProcessorList

from deepsearcher.vector_db.base import BaseVectorDB, RetrievalResult
from deepsearcher.vector_db.milvus import Milvus
from deepsearcher.utils import log

# -----------------------------
# [新增] Agent 前瞻价值探查模块 
# -----------------------------
class ConsistencyChecker:
    def __init__(self, api_key: str, api_base: str, prefix_semantics_map: dict):
        self.api_key = api_key
        self.api_base = api_base
        self.client = httpx.Client(timeout=2.0) # 设置极短的超时时间，防止阻塞
        self.prefix_semantics_map = prefix_semantics_map # 例如: {"2": "计算机科学...", "2.4": "深度学习..."}
        
    def _build_prompt(self, query: str, candidate_prefix: str) -> str:
        # 1. 尝试获取前缀的语义描述
        # 假设 candidate_prefix 是 "2.4"
        semantics = self.prefix_semantics_map.get(candidate_prefix, "未分类的一般文档")
        
        prompt = f"""你是一个严格的逻辑校验器。判断以下 Query 是否属于该类别。
Query: {query}
类别特征: {semantics}
只输出一个词: MATCH, NEUTRAL, 或 CONFLICT。"""
        return prompt

    def evaluate(self, query: str, candidate_prefix: str) -> float:
        """
        向高速 API 发送请求，并映射为具体的 \mathcal{V}_{agent} 分数
        """
        # 如果前缀太短或没有对应语义，直接放行，依靠 T5 底层概率
        if candidate_prefix not in self.prefix_semantics_map:
            return 0.0 

        prompt = self._build_prompt(query, candidate_prefix)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "your-fast-llm-model-name", # 建议用小参数量的高速模型
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, # 必须是 0，保证确定的判断
            "max_tokens": 5     # 只需要 1-2 个 token 的输出
        }

        try:
            # 2. 发起 API 请求
            response = self.client.post(f"{self.api_base}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            result_text = response.json()["choices"][0]["message"]["content"].strip().upper()
            
            # 3. 将文本转化为 POMDP 的收益/惩罚分数
            if "CONFLICT" in result_text:
                return -1.0  # 触发动态剪枝，彻底抛弃该分支
            elif "MATCH" in result_text:
                return 1.0   # 强力奖励，拉高该分支在 Beam 中的权重
            else:
                return 0.0   # NEUTRAL，不奖不罚，完全交由 T5 的 P_LM 决定

        except Exception as e:
            # 发生网络错误或超时时，静默降级为 0.0，不影响主流程解码
            print(f"Agent探查超时或失败，降级放行: {e}")
            return 0.0

# -----------------------------
# [核心修改] 自定义对数概率处理器 (LogitsProcessor)
# -----------------------------
class AgenticLookaheadLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, query: str, checker: ConsistencyChecker, alpha: float = 1.0, beta: float = 2.0):
        self.tokenizer = tokenizer
        self.query = query
        self.checker = checker
        self.alpha = alpha
        self.beta = beta
        
        # 缓存机制：Beam Search 会在同一个前缀上反复扩展，必须加缓存避免 Agent 重复调用导致卡死
        self.agent_score_cache = {}

    # def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    #     # scores 形状: (batch_size * num_beams, vocab_size)
    #     # input_ids 形状: (batch_size * num_beams, current_sequence_length)
        
    #     # 注意：为了性能，我们绝不能对词表里 3 万个 token 都跑一遍 Agent！
    #     # 我们只对那些被 Trie 树允许（未被 mask 为 -inf），且原始 LM 概率排在前列的候选 token 进行 Agent 探查
        
    #     for beam_idx in range(input_ids.shape[0]):
    #         # 获取当前 Beam 的前缀 sequence
    #         prefix_tokens = input_ids[beam_idx].tolist()
    #         if prefix_tokens and prefix_tokens[0] == self.tokenizer.pad_token_id:
    #             prefix_tokens = prefix_tokens[1:]
            
    #         prefix_text = self.tokenizer.decode(prefix_tokens, skip_special_tokens=True)
            
    #         # 找到当前还没有被 prefix_allowed_tokens_fn 强行置为 -inf 的合法 token
    #         # 并且为了速度，我们只对 top 5 的高潜 token 做 Agent 探查
    #         valid_indices = torch.where(scores[beam_idx] > -1e4)[0] 
            
    #         if len(valid_indices) == 0:
    #             continue
                
    #         # 获取 top_k 候选
    #         k = min(5, len(valid_indices))
    #         top_k_indices = valid_indices[torch.topk(scores[beam_idx][valid_indices], k).indices]

    #         for token_id in top_k_indices:
    #             token_str = self.tokenizer.decode([token_id])
    #             candidate_y = prefix_text + token_str
                
    #             # 查缓存
    #             if candidate_y in self.agent_score_cache:
    #                 v_agent = self.agent_score_cache[candidate_y]
    #             else:
    #                 # 动态调用 Agent 进行探查
    #                 v_agent = self.checker.evaluate(self.query, candidate_y)
    #                 self.agent_score_cache[candidate_y] = v_agent
                
    #             # [算法核心公式与动态剪枝]
    #             if v_agent < 0:
    #                 # 探测到逻辑冲突，动态剪枝，彻底熔断该分支
    #                 scores[beam_idx, token_id] = -float('inf')
    #             else:
    #                 # 双重驱动打分：Score(y) = \alpha * P_LM + \beta * V_agent
    #                 # 注意：HuggingFace 的 scores 本身就是未归一化的对数概率 (logits)，可以近似代替 log P_LM
    #                 scores[beam_idx, token_id] = self.alpha * scores[beam_idx, token_id] + self.beta * v_agent
                    
    #     return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        
        # 设置探查的最大深度，比如只探查前 3 层
        MAX_AGENT_DEPTH = 3 

        for beam_idx in range(input_ids.shape[0]):
            prefix_tokens = input_ids[beam_idx].tolist()
            if prefix_tokens and prefix_tokens[0] == self.tokenizer.pad_token_id:
                prefix_tokens = prefix_tokens[1:]
            
            prefix_text = self.tokenizer.decode(prefix_tokens, skip_special_tokens=True)
            valid_indices = torch.where(scores[beam_idx] > -1e4)[0] 
            
            if len(valid_indices) == 0:
                continue
                
            k = min(5, len(valid_indices))
            top_k_indices = valid_indices[torch.topk(scores[beam_idx][valid_indices], k).indices]

            for token_id in top_k_indices:
                token_str = self.tokenizer.decode([token_id])
                candidate_y = prefix_text + token_str
                
                # ========================================================
                # [核心新增] 动态探查剪枝逻辑
                # ========================================================
                
                # 计算当前处于第几层 (通过统计 "." 的数量)
                current_depth = candidate_y.count('.')
                
                # 触发条件 1: 是否是一个完整的节点 (以 "." 结尾，或者只有1位数字)
                # 触发条件 2: 是否没有超过最大探查深度
                is_trigger_node = candidate_y.endswith('.') or len(candidate_y) == 1
                
                if is_trigger_node and current_depth < MAX_AGENT_DEPTH:
                    # 满足条件，调 Agent 进行探查
                    if candidate_y in self.agent_score_cache:
                        v_agent = self.agent_score_cache[candidate_y]
                    else:
                        v_agent = self.checker.evaluate(self.query, candidate_y)
                        self.agent_score_cache[candidate_y] = v_agent
                else:
                    # 深度太深，或处于节点中间态，直接免检放行！
                    v_agent = 0.0  # 评分为 0，不奖不罚，完全由 P_LM 决定
                
                # ========================================================

                if v_agent < 0:
                    scores[beam_idx, token_id] = -float('inf')
                else:
                    # 双重驱动打分：Score(y) = \alpha * P_LM + \beta * V_agent
                    scores[beam_idx, token_id] = self.alpha * scores[beam_idx, token_id] + self.beta * v_agent
                    
        return scores


# 定义前缀树结构用于受限解码
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class DocidTrie:
    def __init__(self, tokenizer, valid_doc_ids):
        self.root = TrieNode()
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        for docid in valid_doc_ids:
            tokens = tokenizer.encode(docid, add_special_tokens=False)
            self._insert(tokens)

    def _insert(self, tokens):
        node = self.root 
        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_end = True

    def get_allowed_tokens(self, prefix_tokens):
        node = self.root
        for token in prefix_tokens:
            if token not in node.children:
                return []
            node = node.children[token]

        allowed = list(node.children.keys())
        if node.is_end: 
            allowed.append(self.eos_token_id)  
        return allowed


class GenerativeRetrievalDB(Milvus):
    def __init__(self, gr_model_path: str, valid_doc_ids: List[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = T5Tokenizer.from_pretrained(gr_model_path)
        self.llm = T5ForConditionalGeneration.from_pretrained(
            gr_model_path,
            device_map="auto",
            dtype=torch.float16
        )
        self.valid_doc_ids = valid_doc_ids if valid_doc_ids else []
        self.trie = DocidTrie(self.tokenizer, self.valid_doc_ids)
        
        # [新增] 初始化 Agent Checker
        self.checker = ConsistencyChecker()

    def _prefix_allowed_tokens_fn(self, batch_id, input_ids):
        prefix = input_ids.tolist()
        if prefix and prefix[0] == self.trie.pad_token_id:
            prefix = prefix[1:]
            
        allowed_tokens = self.trie.get_allowed_tokens(prefix)
        
        if not allowed_tokens:
            return [self.trie.eos_token_id]
        return allowed_tokens

    def _generate_docids(self, instruct: str, query: str, top_k: int = 10) -> List[str]: 
        prompt = f"Query: {query}"

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True).to(self.llm.device)
            
            # [新增] 实例化自定义的对数概率处理器，并传入超参数 alpha, beta
            agent_processor = AgenticLookaheadLogitsProcessor(
                tokenizer=self.tokenizer, 
                query=query, 
                checker=self.checker,
                alpha=1.0, 
                beta=2.0
            )
            logits_processor = LogitsProcessorList([agent_processor])

            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs, 
                    max_length=16,
                    num_beams=top_k,
                    num_return_sequences=top_k,
                    prefix_allowed_tokens_fn=self._prefix_allowed_tokens_fn,
                    logits_processor=logits_processor, # [注入] 将处理器放入生成流程
                    early_stopping=True
                )
            
            generated_docids = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            docids = [docid.strip() for docid in generated_docids]

            return list(set(docids))

        except Exception as e:
            print(f"生成docid失败， {e}")
            return []

    def search_data(
        self, 
        collection: Optional[str], 
        vector: Union[np.array, List[float]],  
        query_text: Optional[str] = None,   
        instruct: Optional[str] = None,      
        top_k: int = 5,
        *args, 
        **kwargs
    ) -> List['RetrievalResult']:
        """
        核心检索方法：
        1. 调用 LLM 生成候选 DocIDs
        2. 到 Milvus 中通过 Scalar Query 捞取具体内容
        """
        if not collection:
            collection = self.default_collection
        
        # 1. 生成 DocIDs
        safe_instruct = instruct if instruct else "Retrieve relevant documents for the query."
        log.color_print("[STEP 1] 生成模型推理 (T5)")
        log.color_print(f"Query: {query_text}")
        target_doc_ids = self._generate_docids(safe_instruct, query_text, top_k)
        
        if not target_doc_ids:
            return []
        
        # 2. 使用 Milvus 进行标量查询获取内容
        log.color_print(f"\n[STEP 2] 检索 Milvus 标量数据", color="cyan")
        formatted_ids = [f"'{i}'" for i in target_doc_ids]  
        expr = f"reference in [{', '.join(formatted_ids)}]"    
        log.color_print(f"Filter Expression: {expr}", color="blue")
        
        try:
            # 使用 query 接口而不是 search 接口
            res = self.client.query(
                collection_name=collection,
                filter=expr,
                output_fields=["text", "reference", "metadata"],
                limit=top_k
            )
        except Exception as e:
            log.color_print(f"Milvus query failed: {e}")
            return []

        # 3. 封装结果
        log.color_print(f"\n[STEP 3] 检索结果汇总 (Found {len(res)} docs):", color="cyan")
        results = []
        for i, item in enumerate(res):
            ref = item.get("reference", "N/A")
            text = item.get("text", "").replace('\n', ' ')  
            
            log.color_print(f" Result [{i+1}] | ID: {ref} | Content: {text[:60]}...", color="white")
            results.append(
                RetrievalResult(
                    embedding=[], 
                    text=item.get("text", ""),
                    reference=item.get("reference", ""),
                    score=1.0,    
                    metadata=item.get("metadata", {}),
                )
            )
        if not results:
            log.color_print("Milvus 中未找到对应生成的 DocID。", color="yellow")
        
        return results

# vllm 版本

# from typing import List, Optional, Tuple, Union
# import numpy as np
# from deepsearcher.vector_db.base import BaseVectorDB, RetrievalResult
# from deepsearcher.vector_db.milvus import Milvus

# from vllm import LLM, SamplingParams

# class GenerativeRetrievalDB(Milvus):
#     def __init__(self, gr_model_path: str, *args, **kwargs):
#         super().__init__(*args, **kwargs) 
        
#         # 1. 使用 vLLM 加载模型
#         # GPU 内存占比(gpu_memory_utilization)可根据实际显存调整
#         self.llm = LLM(
#             model=gr_model_path,
#             trust_remote_code=True,
#             tensor_parallel_size=1, # 如果有多张显卡可以增加此值
#             gpu_memory_utilization=0.6 # 留一部分显存给 Milvus 或其他组件
#         )
#         print(f"Loaded Generative Retrieval Model (vLLM) from {gr_model_path}")

#     def _generate_docids(self, instruct: str, query: str, top_k: int = 5) -> List[str]:
#         prompt = f"Instruct: {instruct}\nQuery: {query}\nDocID:"
        
#         # 2. 设置采样参数以模拟 Beam Search 或获取 Top-K 序列
#         # vLLM 的 n 表示返回多少个序列，best_of 配合 n 使用
#         sampling_params = SamplingParams(
#             n=top_k, 
#             best_of=top_k, 
#             use_beam_search=True, # 开启束搜索
#             max_tokens=32,
#             stop=["\n"] # 遇到换行提前停止，提高效率
#         )

#         # 3. 异步推理（在 vLLM 内部是高度优化的）
#         outputs = self.llm.generate([prompt], sampling_params)

#         doc_ids = []
#         for output in outputs:
#             for res in output.outputs:
#                 # 提取生成的文本并清理多余字符
#                 doc_id = res.text.strip()
#                 if doc_id:
#                     doc_ids.append(doc_id)

#         return list(set(doc_ids))  # 去重
    
#     def search_data(
#         self, 
#         collection: Optional[str], 
#         vector: Union[np.array, List[float]],  # 失效了，GR不需要向量
#         query_text: Optional[str] = None,   # 原始query
#         instruct: Optional[str] = None,      # 新增参数：指令
#         top_k: int = 5,
#         *args, 
#         **kwargs
#     ) -> List[RetrievalResult]:
#         if not collection:
#             collection = self.default_collection
        
#         # 1. 调用生成模型获得DocIDs
#         safe_instruct = instruct if instruct else "Retrieve relevant documents for the query."
#         target_doc_ids = self._generate_docids(safe_instruct, query_text, top_k)
        
#         if not target_doc_ids:
#             return []
        
#         # 2. 使用 Milvus 进行标量查询 (Scalar Query) 获取内容
#         expr = f"reference in {str(target_doc_ids)}" 
        
#         try:
#             # 使用 query 接口而不是 search 接口
#             res = self.client.query(
#                 collection_name=collection,
#                 filter=expr,
#                 output_fields=["text", "reference", "metadata"],
#                 limit=top_k
#             )
#         except Exception as e:
#             print(f"Milvus query failed: {e}")
#             return []

#         # 3. 封装结果
#         return [
#             RetrievalResult(
#                 embedding=[], # 生成式检索没有向量，留空
#                 text=item["text"],
#                 reference=item["reference"],
#                 score=1.0, # 生成式检索通常没有距离分数，置为 1.0
#                 metadata=item.get("metadata", {}),
#             )
#             for item in res
#         ]

# api 版本 

# import numpy as np
# import requests
# from typing import List, Optional, Union
# from deepsearcher.vector_db.base import RetrievalResult
# from deepsearcher.vector_db.milvus import Milvus
# from deepsearcher.utils import log

# class GenerativeRetrievalDB(Milvus):
#     def __init__(self, api_url: str, valid_doc_ids: List[str] = None, *args, **kwargs):
#         """
#         初始化生成式检索数据库。
#         :param api_url: 生成式模型部署接口
#         :param valid_doc_ids: 数据库中所有合法DocID的全量列表（用于受限解码）
#         :param args/kwargs: 传递给 Milvus 基类的参数
#         """
#         super().__init__(*args, **kwargs)
#         self.api_url = api_url
#         self.session = requests.Session()
#         self.valid_doc_ids = valid_doc_ids or []
#         log.color_print("初始化 GenerativeRetrievalDB 成功。")

#     def _request_docids_sync(self, instruct: str, query: str, top_k: int) -> List[str]:
#         """使用 requests 发送同步 HTTP 请求获取 DocID"""
#         prompt = f"Instruct: {instruct}\nQuery: {query}\n"
        
#         payload = {
#             "prompt": prompt,
#             "temperature": 0.5,  
#             "max_tokens": 32,
#             "n": top_k,         
#             "stop": ["\n"],  
#         }

#         # 受限解码
#         if self.valid_doc_ids:
#             payload["guided_choice"] = self.valid_doc_ids

#         try:
#             log.color_print(f"正在请求生成模型 (Generative Retrieval)...", color="blue")
#             response = self.session.post(self.api_url, json=payload, timeout=30)
            
#             if response.status_code != 200:
#                 print(f"Error from Inference Server: {response.status_code} - {response.text}")
#                 log.color_print(f"推理服务器错误: {response.status_code}", color="red")
#                 return []
            
#             data = response.json()
#             doc_ids = []
#             choices = data.get("choices", []) 
#             for choice in choices:
#                 text = choice.get("text", "").strip()
#                 if text:
#                     doc_ids.append(text)
            
#             return list(set(doc_ids))
            
#         except Exception as e:
#             # print(f"GR Request failed: {e}")
#             log.color_print(f"请求失败: {e}", color="red")
#             return []

#     def search_data(
#         self, 
#         collection: Optional[str], 
#         vector: Union[np.array, List[float]], 
#         query_text: Optional[str] = None, 
#         instruct: Optional[str] = None, 
#         top_k: int = 5,
#         *args, 
#         **kwargs
#     ) -> List[RetrievalResult]:
#         """
#         核心检索方法：
#         1. 调用 LLM 生成候选 DocIDs
#         2. 到 Milvus 中通过 Scalar Query 捞取具体内容
#         """

#         if not collection:
#             collection = self.default_collection

#         # 1. 生成 DocIDs
#         safe_instruct = instruct if instruct else "Retrieve relevant documents for the query."
#         log.color_print(f"\n[STEP 1] 生成模型推理", color="cyan")
#         log.color_print(f"Query: {query_text}", color="white")

#         target_doc_ids = self._request_docids_sync(safe_instruct, query_text, top_k)
#         # 可视化输出生成的 DocIDs
#         log.color_print(f">>> 模型生成的候选 DocIDs: {target_doc_ids}", color="green")

#         if not target_doc_ids:
#             return []
        
#         # 2. 检索阶段：Milvus 标量查询 (Scalar Query)
#         log.color_print(f"\n[STEP 2] 检索 Milvus 标量数据", color="cyan")
#         formatted_ids = [f"'{i}'" for i in target_doc_ids]  # 确保字符串 ID 被正确包裹在引号内
#         expr = f"reference in [{', '.join(formatted_ids)}]"     # 表达式形如: reference in ['doc_1', 'doc_2']
#         log.color_print(f"Filter Expression: {expr}", color="blue")

#         try:
#             # Milvus 的 Python SDK (pymilvus) 本身就是同步阻塞调用的
#             res = self.client.query(
#                 collection_name=collection,
#                 filter=expr,
#                 output_fields=["text", "reference", "metadata"],
#                 limit=top_k
#             )
#         except Exception as e:
#             # print(f"Milvus query failed: {e}")
#             log.color_print(f"Milvus 查询失败: {e}", color="red")
#             return []

#         # 3. 封装为 RetrievalResult 格式
#         log.color_print(f"\n[STEP 3] 检索结果汇总 (Found {len(res)} docs):", color="cyan")
#         results = []
#         for i, item in enumerate(res):
#             ref = item.get("reference", "N/A")
#             # text_snippet = item.get("text", "")[:60].replace('\n', ' ')
#             text = item.get("text", "").replace('\n', ' ')  # 改为输出text全文
            
#             # 打印每一个检索到的结果
#             log.color_print(f" Result [{i+1}] | ID: {ref} | Content: {text}...", color="white")
#             results.append(
#                 RetrievalResult(
#                     embedding=[], # GR 无向量
#                     text=item.get("text", ""),
#                     reference=item.get("reference", ""),
#                     score=1.0,    # 生成式检索通常不提供距离分，默认 1.0
#                     metadata=item.get("metadata", {}),
#                 )
#             )
#         if not results:
#             log.color_print("Milvus 中未找到对应生成的 DocID。", color="yellow")
        
#         return results