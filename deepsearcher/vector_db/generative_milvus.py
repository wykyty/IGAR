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


import asyncio
import aiohttp
import numpy as np
from typing import List, Optional, Union
from deepsearcher.vector_db.base import RetrievalResult
from deepsearcher.vector_db.milvus import Milvus

class GenerativeRetrievalDB(Milvus):
    def __init__(self, api_url: str, *args, **kwargs):
        # 初始化父类 Milvus 这里的 args/kwargs 包含连接 Milvus 的参数
        super().__init__(*args, **kwargs)
        self.api_url = api_url # 例如 "http://localhost:8001/v1/completions"
        self.concurrency = 5   # 检索时的并发控制
        
    async def _request_docids(self, instruct: str, query: str, top_k: int) -> List[str]:
        """发起异步 HTTP 请求获取 DocID"""
        prompt = f"Instruct: {instruct}\nQuery: {query}\nDocID:"
        
        # 针对生成式检索优化的 Payload
        payload = {
            "prompt": prompt,
            "temperature": 0.0, # 检索通常需要确定性，设为 0
            "max_tokens": 32,
            "n": top_k,         # 让后端一次生成多个候选
            "stop": ["\n", " "],
            # 如果后端支持 beam search，可以加入以下参数
            # "use_beam_search": True,
            # "best_of": top_k
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload, timeout=30) as resp:
                    if resp.status != 200:
                        print(f"Error from Inference Server: {resp.status}")
                        return []
                    
                    data = await resp.json()
                    # 兼容 OpenAI 格式和 vLLM 格式
                    doc_ids = []
                    choices = data.get("choices", [])
                    for choice in choices:
                        text = choice.get("text", "").strip()
                        if text:
                            doc_ids.append(text)
                    return list(set(doc_ids))
        except Exception as e:
            print(f"Request failed: {e}")
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
        
        if not collection:
            collection = self.default_collection
        
        # 1. 运行异步请求获取 DocIDs
        # 注意：由于 search_data 是同步方法，这里需要用 asyncio.run
        safe_instruct = instruct if instruct else "Retrieve relevant documents for the query."
        target_doc_ids = asyncio.run(self._request_docids(safe_instruct, query_text, top_k))
        
        if not target_doc_ids:
            return []
        
        # 2. Milvus 标量查询
        # 将生成的文本 ID 列表转换为 Milvus 的查询表达式
        formatted_ids = [f"'{i}'" for i in target_doc_ids]
        expr = f"reference in [{', '.join(formatted_ids)}]" 
        
        try:
            res = self.client.query(
                collection_name=collection,
                filter=expr,
                output_fields=["text", "reference", "metadata"],
                limit=top_k
            )
        except Exception as e:
            print(f"Milvus query failed: {e}")
            return []

        # 3. 封装结果
        return [
            RetrievalResult(
                embedding=[], 
                text=item["text"],
                reference=item["reference"],
                score=1.0, 
                metadata=item.get("metadata", {}),
            )
            for item in res
        ]