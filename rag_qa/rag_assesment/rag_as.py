# -*-coding:utf-8-*-
# 导入pandas库，用于数据处理和保存CSV文件
import pandas as pd
# 导入ragas库的evaluate函数，用于执行RAG评估
from ragas import evaluate
# 导入ragas的评估指标，包括忠实度、答案相关性、上下文相关性和上下文召回率
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision, # context_relevancy(原来的写法)两者是一回事
    context_recall
)
# 导入datasets库的Dataset类，用于构建RAGAS所需的数据格式
from datasets import Dataset
# 导入 LangChain 的 OpenAI 相关模块（仅用于 LLM）
from langchain_openai import ChatOpenAI
# 导入 LangChain 的嵌入基类，用于自定义嵌入
from langchain_core.embeddings import Embeddings
# 导入json库，用于加载JSON格式的评估数据集
import json, os
# 导入百炼官方SDK
import dashscope
from dashscope import TextEmbedding

# ========== 新增自定义嵌入类（替换OpenAIEmbeddings） ==========
# 我们改用百炼官方SDK重新实现嵌入接口，以保证RAGAS评估能正常计算向量相似度。
class DashScopeEmbeddings(Embeddings):
    """使用百炼官方SDK的文本向量嵌入类，兼容RAGAS接口"""
    def __init__(self, model_name="text-embedding-v2", api_key=None):
        # 初始化嵌入模型名称和API Key
        self.model_name = model_name
        self.api_key = api_key or os.environ["DASHSCOPE_API_KEY"]
        # 设置dashscope全局API Key
        dashscope.api_key = self.api_key

    def embed_documents(self, texts):
        """对多个文本生成向量（RAGAS要求实现的方法）"""
        embeddings = []
        for text in texts:
            # 调用百炼文本向量API，注意参数名为input（而非text）
            resp = TextEmbedding.call(model=self.model_name, input=text)
            if resp.status_code == 200:
                # 提取向量并添加到结果列表
                embeddings.append(resp.output['embeddings'][0]['embedding'])
            else:
                raise ValueError(f"Embedding failed: {resp.message}")
        return embeddings

    def embed_query(self, text):
        """对单个查询文本生成向量（RAGAS要求实现的方法）"""
        # 复用embed_documents方法，取第一个结果
        return self.embed_documents([text])[0]


# 1. 加载生成的数据集
# 使用with语句打开JSON文件，确保文件正确关闭，指定编码为utf-8
with open("rag_evaluate_data.json", "r", encoding="utf-8") as f:
    # 将JSON文件内容加载到data变量中，data为包含多个数据条目的列表
    data = json.load(f)

# print(f'data--》{data}')
print(f'data--》{len(data)}')
# 2. 转换为RAGAS格式
# 创建字典eval_data，将JSON数据转换为RAGAS要求的字段格式
eval_data = {
    # 提取每个数据条目的question字段，组成问题列表
    "question": [item["question"] for item in data],
    # 提取每个数据条目的answer字段，组成答案列表
    "answer": [item["answer"] for item in data],
    # 提取每个数据条目的context字段，组成上下文列表（每个context为列表）
    "contexts": [item["context"] for item in data],
    # 提取每个数据条目的ground_truth字段，组成真实答案列表
    "ground_truth": [item["ground_truth"] for item in data]
}
# print(eval_data)
# 使用Dataset.from_dict将字典转换为RAGAS所需的Dataset对象
dataset = Dataset.from_dict(eval_data)
print(f'dataset--》{dataset}')

# 3. 配置RAGAS评估环境 —— 改用百炼平台API

# 从环境变量获取 API Key
api_key = os.environ["DASHSCOPE_API_KEY"]

# 配置百炼的 Base URL 和模型名称
# 推荐使用 'dashscope.aliyuncs.com/compatible-mode/v1' 这个地址[reference:11]
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 选择一个模型，例如 qwen3-max，具体模型列表请参考官方文档[reference:12]
model_name = "qwen3-max"

# 初始化 ChatOpenAI 模型，指向百炼的服务地址
llm = ChatOpenAI(
    model=model_name,
    openai_api_key=api_key,
    openai_api_base=base_url,
    # 可以添加其他参数，如 temperature=0.1
)

# 初始化 Embeddings 嵌入模型 —— 使用自定义类（替代有问题的 OpenAIEmbeddings）
embeddings = DashScopeEmbeddings(model_name="text-embedding-v2")

# 4. 执行评估
# 调用evaluate函数，传入数据集、评估指标、LLM模型和嵌入模型
result = evaluate(
    # 传入转换好的Dataset对象
    dataset=dataset,
    # 指定使用的评估指标列表
    metrics=[
        faithfulness,  # 忠实度：答案是否基于上下文
        answer_relevancy,  # 答案相关性：答案与问题的匹配度
        context_precision,  # 上下文相关性：上下文是否仅包含相关信息
        context_recall  # 上下文召回率：上下文是否包含所有必要信息
    ],
    # 传入配置好的LLM模型
    llm=llm,
    # 传入配置好的嵌入模型
    embeddings=embeddings,
)

# 5. 输出并保存结果
# 打印评估结果标题
print("RAGAS评估结果：")
# 打印评估结果，包含各指标的分数
print(result)
# 将评估结果转换为pandas DataFrame，便于保存
result_df = pd.DataFrame([result])
# 将DataFrame保存为CSV文件，文件名为ragas_evaluation_results.csv，不保存索引
result_df.to_csv("ragas_evaluation_results.csv", index=False)
