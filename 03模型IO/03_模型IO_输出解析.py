from dotenv import load_dotenv  # 用于加载环境变量

load_dotenv()  # 加载 .env 文件中的环境变量

# 导入LangChain中的提示模板
from langchain.prompts import PromptTemplate

# 创建提示模板
prompt_template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
{format_instructions}"""

# 使用本地deepseek 模型
from langchain_ollama import ChatOllama
# 实例化一个 Ollama `ChatOllama` 大模型工具
model = ChatOllama(
    model="deepseek-r1:8b",          # Ollama 中的模型名称（如 deepseek-r1:8b）
    base_url="http://localhost:11434",  # Ollama 默认端口
    temperature=0.5,              # 控制生成随机性
    max_tokens=2000               # 最大生成长度
)


# 导入结构化输出解析器和ResponseSchema
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# 定义我们想要接收的响应模式
response_schemas = [
    ResponseSchema(name="description", description="鲜花的描述文案"),
    ResponseSchema(name="reason", description="问什么要这样写这个文案")
]
# 创建输出解析器
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 获取格式指示
format_instructions = output_parser.get_format_instructions()
# 根据模板创建提示，同时在提示中加入输出解析器的说明
# format_instructions 是一个字符串，用于描述如何解析输出。替换了上面prompt中的占位符
prompt = PromptTemplate.from_template(prompt_template,
                                      partial_variables={"format_instructions": format_instructions})

# 数据准备
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

# 创建一个空的DataFrame用于存储结果
import pandas as pd

df = pd.DataFrame(columns=["flower", "price", "description", "reason"])  # 先声明列名

for flower, price in zip(flowers, prices):
    # 根据提示准备模型的输入
    input = prompt.format(flower_name=flower, price=price)
    # 获取模型的输出
    output = model.invoke(input)

    if hasattr(output, "content"):
        output_text = output.content
    else:
        output_text = output

    # 解析模型的输出（这是一个字典结构）
    parsed_output = output_parser.parse(output_text)

    # 在解析后的输出中添加"flower"和"price"
    parsed_output['flower'] = flower
    parsed_output['price'] = price

    # 将解析后的输出添加到DataFrame中
    df.loc[len(df)] = parsed_output

# 打印字典
print(df.to_dict(orient='records'))

# 保存DataFrame到CSV文件
df.to_csv("flowers_with_descriptions.csv", index=False)