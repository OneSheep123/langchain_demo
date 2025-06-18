import os
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载环境变量，确保敏感信息不硬编码

from langchain.prompts import PromptTemplate

template = """你是一位专业的鲜花店文案撰写员\n
对于售价为 {price} 的 {flower_name}，你需要写一段文案，让顾客觉得物超所值。\n
"""

prompt = PromptTemplate.from_template(template)
print(prompt)

# 使用本地deepseek 模型
from langchain_ollama import ChatOllama
# 实例化一个 Ollama `ChatOllama` 大模型工具
llm = ChatOllama(
    model="deepseek-r1:8b",          # Ollama 中的模型名称（如 deepseek-r1:7b）
    base_url="http://localhost:11434",  # Ollama 默认端口
    temperature=0,              # 控制生成随机性
    max_tokens=2000               # 最大生成长度
)

# from langchain_openai import ChatOpenAI
#
# #创建 OpenAI 实例
# llm = ChatOpenAI(
#     model="deepseek-chat",
#     temperature=1,
#     max_tokens=200,
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url=os.getenv("OPENAI_API_BASE")
# )

input = prompt.format(flower_name=["紫荆花"], price='100')
output = llm.invoke(input)

print(output)