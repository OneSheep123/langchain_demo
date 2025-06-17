import os
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载环境变量，确保敏感信息不硬编码

# 1.Load 导入Document Loaders
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader

base_dir = '/Volumes/D/pyWorkSpace/langchain_demo/02文档QA系统'

# 3. 初始化文本切割器
# `RecursiveCharacterTextSplitter` 根据设定的块大小和重叠度递归地分割文本
# `chunk_size`: 每个文本块的最大字符数，影响检索粒度与上下文完整性
# `chunk_overlap`: 相邻文本块之间的重叠字符数，用于保持上下文连贯性，避免信息丢失
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)


# 4. 初始化 Qdrant 向量数据库和 SQLite 记录管理器
# 导入 LangChain Indexing API 相关模块，用于持久化记录管理和增量同步
from langchain.indexes import SQLRecordManager, index 
# 更新导入为 langchain_qdrant
from langchain_qdrant import QdrantVectorStore 
from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
# 导入 models 用于创建集合
from qdrant_client.http import models  

# 定义 Qdrant 集合的名称，用于存储向量数据
collection_name = "my_documents" 

# 定义 Embedding 模型
embedding_model = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434"
)

# 创建 QdrantClient 实例，配置为本地文件存储模式
# 向量数据将存储在指定 `base_dir` 下的 `qdrant_data/{collection_name}` 目录中
qdrant_data_path = os.path.join(base_dir, f"qdrant_data/{collection_name}")
local_qdrant_client = QdrantClient(path=qdrant_data_path)

# 检查 Qdrant 集合是否存在，如果不存在则创建新集合
try:
    local_qdrant_client.get_collection(collection_name=collection_name)
except ValueError:
    # 集合不存在时，打印提示信息并创建
    print(f"Qdrant collection '{collection_name}' not found, creating...")
    local_qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1024,  # `bge-m3` 模型的向量维度是 1024，必须与 Embedding 模型匹配
            distance=models.Distance.COSINE # 使用余弦距离进行向量相似度计算，适用于文本嵌入
        )
    )

# 使用 `QdrantVectorStore` 封装 QdrantClient，以便与 LangChain 框架集成
vectorstore = QdrantVectorStore(
    client=local_qdrant_client,  # 将配置好的 QdrantClient 实例传入
    collection_name=collection_name,
    embedding=embedding_model # 传入 embedding model
)

# 定义记录管理器 `SQLRecordManager`，用于追踪文档的索引状态
# 使用 SQLite 数据库存储索引记录，文件路径在 `base_dir` 下的指定位置
record_manager_db_path = os.path.join(base_dir, f"record_manager_{collection_name}.sqlite")
db_url = f"sqlite:///{record_manager_db_path}"
record_manager = SQLRecordManager(
    namespace=collection_name, # 命名空间，推荐与 Qdrant 集合名称保持一致，确保记录的唯一性
    db_url=db_url
)
# 首次运行时，创建 SQLite 数据库的表结构。如果表已存在，此操作不会重复创建
record_manager.create_schema() 

# 定义文档加载函数
# 从指定目录加载支持的文档类型（PDF, DOCX, TXT）
def load_documents_from_directory(directory):
    loaded_docs = []
    for file in os.listdir(directory): 
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path): # 确保处理的是文件而不是子目录
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

# 使用 LangChain Indexing API 来同步文档到向量存储
# 该函数在程序启动时运行，智能地检查并更新向量存储中的文档
def sync_documents_to_vectorstore():
    print("开始同步文档到向量存储...")
    # 加载所有源文档
    all_documents = load_documents_from_directory(f"{base_dir}/OneFlower")
    # 将文档分割成适合嵌入的文本块
    chunked_documents = text_splitter.split_documents(all_documents)
    
    # 使用 Indexing API 进行文档同步
    # `cleanup="incremental"` 表示只处理新增或修改的文档，并清理不再存在的旧记录
    # `source_id_key` 默认为 'source'，用于从文档元数据中获取唯一源 ID，便于追踪
    indexing_result = index(
        docs_source=chunked_documents,
        vector_store=vectorstore,
        record_manager=record_manager,
        cleanup="incremental",
        source_id_key="source"
    )
    
    print(f"文档同步完成。Stats: {indexing_result}")

# 在应用程序启动时执行文档同步操作
# 这一步是确保向量存储数据是最新的关键，且只处理有变化的文档
sync_documents_to_vectorstore()

# 5. 准备大型语言模型 (LLM) 和检索链
import logging # 导入 Python 内置的日志工具
# 使用本地deepseek 模型
from langchain_ollama import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever # MultiQueryRetriever工具
from langchain.chains import RetrievalQA # RetrievalQA链

# 设置日志级别，以便观察 `MultiQueryRetriever` 的详细工作情况
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

# 实例化一个 Ollama `ChatOllama` 大模型工具
llm = ChatOllama(
    model="deepseek-r1:8b",          # Ollama 中的模型名称（如 deepseek-r1:7b）
    base_url="http://localhost:11434",  # Ollama 默认端口
    temperature=0,              # 控制生成随机性
    max_tokens=2000               # 最大生成长度
)

# 实例化一个 `MultiQueryRetriever`（多查询检索器）
# `MultiQueryRetriever` 通过将用户原始问题转换为多个相似问题来提高检索覆盖范围和效果
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)

# 实例化一个 `RetrievalQA` 链
# 该链将检索到的文档与大模型结合，用于生成对用户问题的答案
qa_chain = RetrievalQA.from_chain_type(llm,retriever=retriever_from_llm)

# 6. 实现问答系统的 Flask UI 界面
from flask import Flask, request, render_template
app = Flask(__name__) # 初始化 Flask 应用实例

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # 从 POST 请求中获取用户输入的问题
        question = request.form.get('question')        
        
        # 调用 `RetrievalQA` 链，传入用户问题以获取答案
        result = qa_chain({"query": question})
        
        # 将大模型生成的答案渲染到 `index.html` 网页模板中并返回给用户
        return render_template('index.html', result=result)
    
    # 对于 GET 请求，直接渲染 `index.html` 页面（通常用于显示初始表单）
    return render_template('index.html')

if __name__ == "__main__":
    # 启动 Flask 应用，监听所有网络接口（0.0.0.0），启用调试模式，端口设为 5001
    # `use_reloader=False` 禁用自动重载，避免在代码改动时重复加载和处理文档
    app.run(host='0.0.0.0',debug=True,port=5001, use_reloader=False)