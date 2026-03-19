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


import numpy as np
import requests
from typing import List, Optional, Union
from deepsearcher.vector_db.base import RetrievalResult
from deepsearcher.vector_db.milvus import Milvus
from deepsearcher.utils import log

class GenerativeRetrievalDB(Milvus):
    def __init__(self, api_url: str, *args, **kwargs):
        """
        初始化生成式检索数据库。
        :param api_url: 生成式模型部署接口
        :param args/kwargs: 传递给 Milvus 基类的参数
        """
        super().__init__(*args, **kwargs)
        self.api_url = api_url
        self.session = requests.Session()
        log.color_print("初始化 GenerativeRetrievalDB 成功。")

    def _request_docids_sync(self, instruct: str, query: str, top_k: int) -> List[str]:
        """使用 requests 发送同步 HTTP 请求获取 DocID"""
        prompt = f"Instruct: {instruct}\nQuery: {query}\n"
        
        payload = {
            "prompt": prompt,
            "temperature": 0.3,      # 检索需要确定性
            "max_tokens": 32,
            "n": top_k,         
            "stop": ["\n"],     # 遇到换行停止生成
        }

        try:
            log.color_print(f"正在请求生成模型 (Generative Retrieval)...", color="blue")
            response = self.session.post(self.api_url, json=payload, timeout=30)
            
            if response.status_code != 200:
                print(f"Error from Inference Server: {response.status_code} - {response.text}")
                log.color_print(f"推理服务器错误: {response.status_code}", color="red")
                return []
            
            data = response.json()
            doc_ids = []
            choices = data.get("choices", []) # 解析 OpenAI/vLLM 兼容格式
            for choice in choices:
                text = choice.get("text", "").strip()
                if text:
                    doc_ids.append(text)
            
            return list(set(doc_ids))
            
        except Exception as e:
            # print(f"GR Request failed: {e}")
            log.color_print(f"请求失败: {e}", color="red")
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
    ) -> List[RetrievalResult]:
        """
        核心检索方法：
        1. 调用 LLM 生成候选 DocIDs
        2. 到 Milvus 中通过 Scalar Query 捞取具体内容
        """

        if not collection:
            collection = self.default_collection

        # 1. 生成 DocIDs
        safe_instruct = instruct if instruct else "Retrieve relevant documents for the query."
        log.color_print(f"\n[STEP 1] 生成模型推理", color="cyan")
        log.color_print(f"Query: {query_text}", color="white")

        target_doc_ids = self._request_docids_sync(safe_instruct, query_text, top_k)
        # 可视化输出生成的 DocIDs
        log.color_print(f">>> 模型生成的候选 DocIDs: {target_doc_ids}", color="green")

        if not target_doc_ids:
            return []
        
        # 2. 检索阶段：Milvus 标量查询 (Scalar Query)
        log.color_print(f"\n[STEP 2] 检索 Milvus 标量数据", color="cyan")
        formatted_ids = [f"'{i}'" for i in target_doc_ids]  # 确保字符串 ID 被正确包裹在引号内
        expr = f"reference in [{', '.join(formatted_ids)}]"     # 表达式形如: reference in ['doc_1', 'doc_2']
        log.color_print(f"Filter Expression: {expr}", color="blue")

        try:
            # Milvus 的 Python SDK (pymilvus) 本身就是同步阻塞调用的
            res = self.client.query(
                collection_name=collection,
                filter=expr,
                output_fields=["text", "reference", "metadata"],
                limit=top_k
            )
        except Exception as e:
            # print(f"Milvus query failed: {e}")
            log.color_print(f"Milvus 查询失败: {e}", color="red")
            return []

        # 3. 封装为 RetrievalResult 格式
        log.color_print(f"\n[STEP 3] 检索结果汇总 (Found {len(res)} docs):", color="cyan")
        results = []
        for i, item in enumerate(res):
            ref = item.get("reference", "N/A")
            text_snippet = item.get("text", "")[:60].replace('\n', ' ')
            
            # 打印每一个检索到的结果
            log.color_print(f" Result [{i+1}] | ID: {ref} | Content: {text_snippet}...", color="white")
            results.append(
                RetrievalResult(
                    embedding=[], # GR 无向量
                    text=item.get("text", ""),
                    reference=item.get("reference", ""),
                    score=1.0,    # 生成式检索通常不提供距离分，默认 1.0
                    metadata=item.get("metadata", {}),
                )
            )
        if not results:
            log.color_print("Milvus 中未找到对应生成的 DocID。", color="yellow")
        
        return results