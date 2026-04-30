import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from typing import List, Optional, Union
import numpy as np

from deepsearcher.vector_db.base import RetrievalResult
from deepsearcher.vector_db.milvus import Milvus
from deepsearcher.utils import log

class GenerativeRetrievalDB(Milvus):
    def __init__(self, gr_model_path: str, valid_doc_ids: List[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 1. 获取设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 2. 使用原生 transformers 加载 T5 模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(gr_model_path)
        self.llm = T5ForConditionalGeneration.from_pretrained(
            gr_model_path,
            device_map="auto",           # 自动将模型分配到可用的 GPU
            torch_dtype=torch.float16    # 使用半精度，显著降低显存占用并加速
        )
        
        self.valid_doc_ids = set(valid_doc_ids) if valid_doc_ids else set()

        log.color_print("初始化 GenerativeRetrievalDB 成功")

    def _generate_docids(self, instruct: str, query: str, top_k: int = 5) -> List[str]:
        # 注意：这里的 prompt 格式必须与你微调时（80k数据集）使用的输入格式完全一致
        prompt = f"Query: {query}\n"
        
        try:
            # 将输入转换为 Tensor
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
            
            # 使用模型进行推理
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=32,
                    temperature=0.5,
                    # do_sample=True,             # 对应 vLLM 的 temperature 采样
                    # num_return_sequences=top_k, # 对应 vLLM 中的 n=top_k (生成多条)
                    
                    # 提示: 如果你发现生成的 docID 质量不高，可以注释掉上面两行，
                    # 并取消下面两行的注释，改用 Beam Search (集束搜索)，结果会更稳定：
                    num_beams=top_k,
                    num_return_sequences=top_k,
                )
            
            # 解码生成的 Token IDs
            decoded_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            doc_ids = []
            for text in decoded_texts:
                # 模拟 vLLM 中 stop=["\n"] 的截断行为
                clean_id = text.split("\n")[0].strip()
                
                if clean_id:
                    # 如果提供了 valid_doc_ids 列表，在后处理阶段过滤掉不存在的非法 ID
                    if self.valid_doc_ids and clean_id not in self.valid_doc_ids:
                        continue
                    doc_ids.append(clean_id)

            return list(set(doc_ids))  # 去重后返回
        
        except Exception as e:
            log.color_print(f"生成docid失败：{e}")
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