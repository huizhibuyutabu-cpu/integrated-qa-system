import os
import dashscope
from dashscope import TextEmbedding

dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

# 使用 input 参数
resp = TextEmbedding.call(
    model="text-embedding-v2",
    input="测试文本"   # 改为 input
)

if resp.status_code == 200:
    embedding = resp.output['embeddings'][0]['embedding']
    print(f"✅ 成功！向量长度: {len(embedding)}")
else:
    print(f"❌ 失败: {resp.message}")