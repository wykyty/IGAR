import asyncio
from typing import List, Tuple

from deepsearcher.agent.base import RAGAgent, describe_class
from deepsearcher.agent.collection_router import CollectionRouter
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.utils import log
from deepsearcher.vector_db import RetrievalResult
from deepsearcher.vector_db.base import BaseVectorDB

# 摘要 Prompt 也极简化，去掉了对 sub_queries 的依赖
SUMMARY_PROMPT = """You are an AI content analysis expert. Please provide a specific answer based ONLY on the retrieved document chunks.

Original Query: {question}

Related Chunks: 
{chunk_str}
"""

@describe_class(
    "A very simple searcher that directly retrieves the top-k results for the original query without any decomposition or multi-hop reasoning. Best for evaluating base single-hop retrieval performance."
)
class SimpleSearch(RAGAgent):
    """
    Simple Search agent implementation for direct generative retrieval evaluation.

    This agent performs a single-pass search using the exact original query,
    returning the raw Top-K hits without query decomposition, reflection, or reranking.
    """

    def __init__(
        self,
        llm: BaseLLM,
        embedding_model: BaseEmbedding,
        vector_db: BaseVectorDB,
        route_collection: bool = True,
        text_window_splitter: bool = True,
        **kwargs,
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.route_collection = route_collection
        self.collection_router = CollectionRouter(
            llm=self.llm, vector_db=self.vector_db, dim=embedding_model.dimension
        )
        self.text_window_splitter = text_window_splitter

    def retrieve(self, original_query: str, top_k: int = 10, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
        return asyncio.run(self.async_retrieve(original_query, top_k, **kwargs))

    async def async_retrieve(
        self, original_query: str, top_k: int = 10, **kwargs
    ) -> Tuple[List[RetrievalResult], int, dict]:
        log.color_print(f"<query> {original_query} </query>\n")
        
        consume_tokens = 0
        if self.route_collection:
            selected_collections, n_token_route = self.collection_router.invoke(
                query=original_query, dim=self.embedding_model.dimension
            )
            consume_tokens += n_token_route
        else:
            selected_collections = self.collection_router.all_collections

        all_retrieved_results = []
        query_vector = self.embedding_model.embed_query(original_query)
        
        for collection in selected_collections:
            log.color_print(f"<search> Direct search [{original_query}] in [{collection}]... </search>\n")
            retrieved_results = self.vector_db.search_data(
                collection=collection, 
                vector=query_vector,  
                query_text=original_query,
                instruct="Retrieve relevant document for the following query.", 
                top_k=top_k 
            )
            
            if retrieved_results:
                all_retrieved_results.extend(retrieved_results)
                
        # 核心保证：如果查询了多个 Collection，总数可能大于 top_k。
        # 这里进行全局分数排序，并严格截断到 top_k，保证你计算 Hits@K 的严谨性。
        if all_retrieved_results and hasattr(all_retrieved_results[0], 'score'):
            # 假设你的 RetrievalResult 对象包含 score 属性，且越大越好
            all_retrieved_results.sort(key=lambda x: x.score, reverse=True)
            
        all_retrieved_results = all_retrieved_results[:top_k]
        
        log.color_print(
            f"<search> Returned {len(all_retrieved_results)} final document chunk(s). </search>\n"
        )
        
        return all_retrieved_results, consume_tokens, {}

    def query(self, query: str, top_k: int = 10, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        all_retrieved_results, n_token_retrieval, _ = self.retrieve(query, top_k=top_k, **kwargs)
        
        if not all_retrieved_results:
            return f"No relevant information found for query '{query}'.", [], n_token_retrieval
        
        chunk_texts = []
        for chunk in all_retrieved_results:
            if self.text_window_splitter and "wider_text" in chunk.metadata:
                chunk_texts.append(chunk.metadata["wider_text"])
            else:
                chunk_texts.append(chunk.text)
                
        log.color_print(f"<think> Summarize answer from {len(all_retrieved_results)} retrieved chunks... </think>\n")
        
        summary_prompt = SUMMARY_PROMPT.format(
            question=query,
            chunk_str=self._format_chunk_texts(chunk_texts),
        )
        chat_response = self.llm.chat([{"role": "user", "content": summary_prompt}])
        
        log.color_print("\n==== FINAL ANSWER ====\n")
        log.color_print(self.llm.remove_think(chat_response.content))
        
        return (
            self.llm.remove_think(chat_response.content),
            all_retrieved_results,
            n_token_retrieval + chat_response.total_tokens,
        )

    def _format_chunk_texts(self, chunk_texts: List[str]) -> str:
        chunk_str = ""
        for i, chunk in enumerate(chunk_texts):
            chunk_str += f"<chunk_{i}>\n{chunk}\n</chunk_{i}>\n"
        return chunk_str