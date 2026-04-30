# 测试milvus的get_data_by_ids方法
import numpy as np
from deepsearcher.vector_db.milvus import Milvus
from deepsearcher.loader.splitter import Chunk
from deepsearcher.utils import log

def test_get_data_by_ids():
    # 1. 初始化 Milvus 实例
    db = Milvus(
        uri="./milvus.db", 
        token="root:Milvus", 
        default_collection="test_get_ids_collection"
    )
    
    collection_name = "test_get_ids_collection"
    dim = 128
    
    log.color_print(f"开始测试: 集合名称={collection_name}")

    # 2. 初始化集合
    db.init_collection(dim=dim, collection=collection_name, force_new_collection=True)

    # 3. 准备并插入测试数据
    test_chunks = [
        Chunk(
            text="这是第一条测试文档内容", 
            reference="doc_1.pdf", 
            embedding=np.random.rand(dim).tolist(), 
            metadata={"page": 1}
        ),
        Chunk(
            text="这是第二条测试文档内容", 
            reference="doc_2.pdf", 
            embedding=np.random.rand(dim).tolist(), 
            metadata={"page": 2}
        )
    ]
    
    # 插入数据
    db.insert_data(collection=collection_name, chunks=test_chunks)
    
    # 4. 获取刚刚插入的数据的 ID
    # 由于 insert_data 没有返回 ID，我们直接查询数据库获取前两个 ID
    res = db.client.query(
        collection_name=collection_name,
        filter="",
        output_fields=["id", "text"],
        limit=2
    )
    
    if not res:
        log.error("插入数据失败，未能查询到 ID")
        return

    inserted_ids = [item["id"] for item in res]
    expected_texts = [item["text"] for item in res]
    
    log.color_print(f"获取到测试 ID 列表: {inserted_ids}")

    # 5. 调用待测接口: get_data_by_ids
    retrieved_results = db.get_data_by_ids(collection=collection_name, ids=inserted_ids)

    # 6. 验证结果
    log.color_print(f"接口返回结果数量: {len(retrieved_results)}")
    
    assert len(retrieved_results) == len(inserted_ids), "返回结果数量不匹配！"
    
    for i, result in enumerate(retrieved_results):
        log.color_print(f"校验内容 {i}: ID={inserted_ids[i]}, Text='{result.text}'")
        assert result.text in expected_texts, f"文本内容不匹配！期望: {expected_texts}, 实际: {result.text}"
        assert result.reference is not None, "Reference 丢失！"
        assert isinstance(result.metadata, dict), "Metadata 格式错误！"

    log.color_print("恭喜！get_data_by_ids 接口测试通过！")

if __name__ == "__main__":
    test_get_data_by_ids()