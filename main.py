import os
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量

from langchain_openai import ChatOpenAI
#
# 创建 OpenAI 实例
# llm = ChatOpenAI(
#     model="deepseek-chat",
#     temperature=0.1,
#     max_tokens=200,
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url=os.getenv("OPENAI_API_BASE")
# )

from langchain.schema import (
    HumanMessage,
    SystemMessage
)

# 使用本地deepseek 模型
from langchain_ollama import ChatOllama

# 初始化本地 DeepSeek 模型
llm = ChatOllama(
    model="deepseek-r1:8b",          # Ollama 中的模型名称（如 deepseek-r1:7b）
    base_url="http://localhost:11434",  # Ollama 默认端口
    temperature=0.7,              # 控制生成随机性
    max_tokens=2000               # 最大生成长度
)

messages = [
    SystemMessage(content="你是一个AI助手，擅长回答用户的问题。"),
    HumanMessage(content="如何用 LangChain 调用 DeepSeek?")
]

response = llm.invoke(messages)

print(response.content)  # 输出结果