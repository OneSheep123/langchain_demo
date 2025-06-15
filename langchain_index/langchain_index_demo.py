# 原文档连接https://python.langchain.com/docs/how_to/indexing/#quickstart
from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client.http import models  # 导入 models 用于创建集合

collection_name = "langchain_demo"

embedding = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434"
)

# 创建 QdrantClient 实例，用于本地文件存储
local_qdrant_client = QdrantClient(path=f"./langchain_index/qdrant_data/{collection_name}") 

# 创建集合（如果不存在）
try:
    local_qdrant_client.get_collection(collection_name=collection_name)
except ValueError:
    print(f"{collection_name} collection不存在")
    # 如果集合不存在，创建它
    local_qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1024,  # 向量维度，需要与 embedding 模型输出维度匹配
            distance=models.Distance.COSINE
        )
    )

# 使用 QdrantVectorStore，并将 QdrantClient 实例作为 client 参数传入
vectorstore = QdrantVectorStore(
    client=local_qdrant_client,  # 将配置好的 QdrantClient 实例传入
    collection_name=collection_name,
    embedding=embedding
)

db_url = f"sqlite:///langchain_index/{collection_name}.sql"
record_manager = SQLRecordManager(
    namespace=collection_name, # 唯一标识符，推荐与 collection_name 对应
    db_url=db_url
)

record_manager.create_schema()

doc1 = Document(page_content="kitty", metadata={"source": "kitty.txt"})
doc2 = Document(page_content="doggy", metadata={"source": "doggy.txt"})

def _clear():
    """Hacky helper method to clear content. See the `full` mode section to to understand why it works."""
    index([], record_manager, vectorstore, cleanup="full", source_id_key="source")

# _clear()
indexing_result = index([doc1, doc2], record_manager, vectorstore, cleanup="full", source_id_key="source")
print(f"文档同步完成。Stats: {indexing_result}")