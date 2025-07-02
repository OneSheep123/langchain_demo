# 当前代码存在问题，暂时无法启动
import os
from dotenv import load_dotenv
load_dotenv() 

# 导入必要的库
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# 初始化 HF LLM
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# 创建简单的question-answering提示模板
template = """Question: {question}
              Answer: """

# 创建 Prompt
prompt = PromptTemplate(template=template, input_variables=["question"])

# 新链式调用方式
chain = prompt | llm

# 准备问题
question = "Rose is which type of flower?"

# 调用模型并返回结果
result = chain.invoke({"question": question})
print(result)