import os
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量

# 1.Load 导入Document Loaders
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader

base_dir = '/Volumes/D/pyWorkSpace/langchain_demo/02文档QA系统'

# 2.Split 将Documents切分成块以便后续进行嵌入和向量存储
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)


# 3.Store 将分割嵌入并存储在矢量数据库Qdrant中
from langchain.indexes import SQLRecordManager, index # 导入 Indexing API 相关模块 使用 SQLRecordManager 进行持久化记录管理
from langchain_qdrant import QdrantVectorStore # 更新导入为 langchain_qdrant
from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models  # 导入 models 用于创建集合

collection_name = "my_documents" # 定义 Qdrant 集合名称

# 定义 Embedding 模型
embedding_model = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434"
)

# 创建 QdrantClient 实例，用于本地文件存储
# 将数据存储在 OneFlower 目录下的 qdrant_data 子目录中
qdrant_data_path = os.path.join(base_dir, f"qdrant_data/{collection_name}")
local_qdrant_client = QdrantClient(path=qdrant_data_path)

# 创建集合（如果不存在）
try:
    local_qdrant_client.get_collection(collection_name=collection_name)
except ValueError:
    # 如果集合不存在，创建它
    print(f"Qdrant collection '{collection_name}' not found, creating...")
    local_qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1024,  # bge-m3 模型的维度是 1024
            distance=models.Distance.COSINE # 文本相似度通常使用余弦距离
        )
    )

# 使用 QdrantVectorStore，并将 QdrantClient 实例作为 client 参数传入
vectorstore = QdrantVectorStore(
    client=local_qdrant_client,  # 将配置好的 QdrantClient 实例传入
    collection_name=collection_name,
    embedding=embedding_model # 传入 embedding model
)

# 定义记录管理器
# 使用 SQLite 数据库存储索引记录，文件路径在 OneFlower 目录下
record_manager_db_path = os.path.join(base_dir, f"record_manager_{collection_name}.sqlite")
db_url = f"sqlite:///{record_manager_db_path}"
record_manager = SQLRecordManager(
    namespace=collection_name, # 唯一标识符，推荐与 collection_name 对应
    db_url=db_url
)
# 首次运行时需要创建表结构，但如果表已存在，则不会重复创建
record_manager.create_schema() 

# 定义文档加载函数
def load_documents_from_directory(directory):
    loaded_docs = []
    for file in os.listdir(directory): 
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path): # 确保是文件而不是目录
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                loaded_docs.extend(loader.load())
            elif file.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
                loaded_docs.extend(loader.load())
            elif file.endswith('.txt'):
                loader = TextLoader(file_path)
                loaded_docs.extend(loader.load())
    return loaded_docs

# 使用 Indexing API 来同步文档
# 这个函数会在每次运行 Flask 应用时被调用，自动检查并更新向量存储
def sync_documents_to_vectorstore():
    print("开始同步文档到向量存储...")
    # 获取所有文档
    all_documents = load_documents_from_directory(f"{base_dir}/OneFlower")
    chunked_documents = text_splitter.split_documents(all_documents)
    
    # 使用 Indexing API 同步
    # 'cleanup="full"' 表示会清理不再存在于源文档中的旧记录
    # 'source_id_key' 默认为 'source'，用于从文档元数据中获取唯一源 ID
    indexing_result = index(
        docs_source=chunked_documents,
        vector_store=vectorstore,
        record_manager=record_manager,
        cleanup="incremental",
        source_id_key="source"
    )
    
    print(f"文档同步完成。Stats: {indexing_result}")

# 在程序启动时同步文档
# 可以在每次服务启动时运行，它会智能地只处理有变化的文档
sync_documents_to_vectorstore()

# 4. Retrieval 准备模型和Retrieval链
import logging # 导入Logging工具
# 使用本地deepseek 模型
from langchain_ollama import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever # MultiQueryRetriever工具
from langchain.chains import RetrievalQA # RetrievalQA链

# 设置Logging
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

# 实例化一个大模型工具 - OpenAI的GPT-3.5
llm = ChatOllama(
    model="deepseek-r1:8b",          # Ollama 中的模型名称（如 deepseek-r1:7b）
    base_url="http://localhost:11434",  # Ollama 默认端口
    temperature=0,              # 控制生成随机性
    max_tokens=2000               # 最大生成长度
)

# 实例化一个MultiQueryRetriever(多查询检索器)
# - 将用户问题转换为多个相似问题
# - 提高检索的覆盖范围
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)

# 实例化一个RetrievalQA链
qa_chain = RetrievalQA.from_chain_type(llm,retriever=retriever_from_llm)

# 5. Output 问答系统的UI实现
from flask import Flask, request, render_template
app = Flask(__name__) # Flask APP

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # 接收用户输入作为问题
        question = request.form.get('question')        
        
        # RetrievalQA链 - 读入问题，生成答案
        result = qa_chain({"query": question})
        
        # 把大模型的回答结果返回网页进行渲染
        return render_template('index.html', result=result)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True,port=5001, use_reloader=False)