# 使用本地deepseek 模型
from langchain_ollama import ChatOllama
# 实例化一个 Ollama `ChatOllama` 大模型工具
llm = ChatOllama(
    model="deepseek-r1:8b",          # Ollama 中的模型名称（如 deepseek-r1:8b）
    base_url="http://localhost:11434",  # Ollama 默认端口
    temperature=0.3,              # 控制生成随机性
    max_tokens=2000               # 最大生成长度
)


# 设定 AI 的角色和目标
tot_template = """
你是一个专业的花卉顾问，需要为客户的鲜花需求提供最佳解决方案。请按照以下步骤思考：

1. **需求分析**：解析客户的核心需求
2. **方案生成**：生成至少3种不同的鲜花组合方案
3. **方案评估**：分析每个方案的优缺点
4. **最终推荐**：选择最优方案并说明理由

**示例**：
人类：我需要适合婚礼的鲜花，要体现纯洁和永恒
AI：
[思考树]
1. 需求分析：客户需要象征纯洁和永恒的婚礼用花

2. 方案生成：
   - 方案A：白玫瑰+百合
     * 优点：经典组合，白玫瑰象征纯洁，百合代表百年好合
     * 缺点：常见缺乏新意
   - 方案B：铃兰+白色郁金香
     * 优点：清新优雅，铃兰象征幸福回归
     * 缺点：花期较短
   - 方案C：芍药+满天星
     * 优点：芍药富贵吉祥，搭配浪漫
     * 缺点：季节限制性强

3. 方案评估：
   - 婚礼需要稳定性：方案A最可靠
   - 预算考虑：方案C成本较高
   - 季节匹配：当前是芍药花期

4. 最终推荐：推荐方案A，建议增加满天星点缀提升视觉效果
"""
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

system_prompt = SystemMessagePromptTemplate.from_template(tot_template)

# 用户的询问
human_template = "{human_input}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 将以上所有信息结合为一个聊天提示
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

prompt = chat_prompt.format_prompt(human_input="我想为女朋友买花，她喜欢粉色和紫色，预算200元左右").to_messages()

# 接收用户的询问，返回回答结果
response = llm(prompt)
print(response.content)
