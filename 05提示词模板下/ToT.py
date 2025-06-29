# 使用本地deepseek 模型
from langchain_ollama import ChatOllama
# 实例化一个 Ollama `ChatOllama` 大模型工具
llm = ChatOllama(
    model="deepseek-r1:8b",          # Ollama 中的模型名称（如 deepseek-r1:8b）
    base_url="http://localhost:11434",  # Ollama 默认端口
    temperature=0,              # 控制生成随机性
    max_tokens=2000               # 最大生成长度
)


# 设定 AI 的角色和目标
role_template = "你是一个为花店电商公司工作的AI助手, 你的目标是帮助客户根据他们的喜好做出明智的决定"

# CoT 的关键部分，AI 解释推理过程，并加入一些先前的对话示例（Few-Shot Learning）
cot_template = """
假设⼀个顾客在鲜花⽹站上询问：“我想为我的妻⼦购买⼀束鲜花，但我不确定应该选择哪种鲜花。她喜欢淡雅的颜⾊和花⾹。"

AI（使⽤ToT框架）：

思维步骤1：理解顾客的需求。
顾客想为妻⼦购买鲜花。
顾客的妻⼦喜欢淡雅的颜⾊和花⾹。

思维步骤2：考虑可能的鲜花选择。
候选1：百合，因为它有淡雅的颜⾊和花⾹。
候选2：玫瑰，选择淡粉⾊或⽩⾊，它们通常有花⾹。
候选3：紫罗兰，它有淡雅的颜⾊和花⾹。
候选4：桔梗，它的颜⾊淡雅但不⼀定有花⾹。
候选5：康乃馨，选择淡⾊系列，它们有淡雅的花⾹。

思维步骤3：根据顾客的需求筛选最佳选择。
百合和紫罗兰都符合顾客的需求，因为它们都有淡雅的颜⾊和花⾹。
淡粉⾊或⽩⾊的玫瑰也是⼀个不错的选择。
桔梗可能不是昀佳选择，因为它可能没有花⾹。
康乃馨是⼀个可考虑的选择。

思维步骤4：给出建议。
“考虑到您妻⼦喜欢淡雅的颜⾊和花⾹，我建议您可以选择百合或紫罗兰。淡粉⾊或⽩⾊的玫瑰也
是⼀个很好的选择。希望这些建议能帮助您做出决策！”
"""

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

system_prompt_role = SystemMessagePromptTemplate.from_template(role_template)
system_prompt_cot = SystemMessagePromptTemplate.from_template(cot_template)

# 用户的询问
human_template = "{human_input}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 将以上所有信息结合为一个聊天提示
chat_prompt = ChatPromptTemplate.from_messages([system_prompt_role, system_prompt_cot, human_prompt])

prompt = chat_prompt.format_prompt(human_input="我想为我的女朋友购买一些花。她喜欢粉色和紫色。你有什么建议吗?").to_messages()

# 接收用户的询问，返回回答结果
response = llm(prompt)
print(response.content)
