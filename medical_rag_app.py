# 接入LLM 进行知识增强生成RAG
import os
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import requests  # 用于调用 LLM API
import re

# 基础配置
DB_FILE = "./milvus_medical.db"
COLLECTION_NAME = "medical_knowledge"
MODEL_PATH = "/home/kdyy/project/FastAPI/rag_bge_milvus/bge-large-zh-v1.5"

# LLM 配置 (这里以 Ollama 为例，你也可以换成 OpenAI 兼容接口)
LLM_API_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen2.5:7b" # 或者你下载的其他模型

class MedicalRAG:
    def __init__(self):
        self.milvus_client = MilvusClient(uri=DB_FILE)
        self.embed_model = SentenceTransformer(MODEL_PATH)
        
    def _retrieve(self, query, top_k=3):
        """检索环节：获取相关文档"""
        query_vector = self.embed_model.encode([query], normalize_embeddings=True)[0].tolist()
        results = self.milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            limit=top_k,
            output_fields=["text"]
        )
        # 提取文档内容并拼接
        docs = [res['entity']['text'] for res in results[0]]
        return "\n\n".join([f"资料{i+1}:\n{doc}" for i, doc in enumerate(docs)])

    def _generate(self, query, context):
        """生成环节：调用 LLM"""
        # 1. 构建 Prompt (参考 medical-rag 的专业提示词逻辑)
        prompt = f"""你是一名专业的医学知识助手。请基于提供的参考资料准确回答用户问题。

# 要求:
1. 必须严格基于参考资料回答，不要编造资料中没提到的事实。
2. 如果参考资料不足以回答问题，请如实说明。
3. 语言要专业、严谨且通俗易懂。
4. 回答最后请附带一句：“本建议仅供参考，请以线下医生诊断为准。”

# 参考资料:
{context}

# 用户问题:
{query}

# 你的专业回答:"""

        # 2. 调用 LLM (以 Ollama 为例)
        payload = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(LLM_API_URL, json=payload)
            response.raise_for_status()
            full_text = response.json().get("response", "")
            # 清洗掉可能存在的 <think> 标签
            cleaned_text = re.sub(r"<think>.*?</think>\s*", "", full_text, flags=re.DOTALL)
            return cleaned_text.strip()
        except Exception as e:
            return f"生成回答时出错: {e}"

    def answer(self, query):
        """完整 RAG 流程"""
        # 第一步：搜索资料
        context = self._retrieve(query)
        # 第二步：生成回答
        if not context:
            return "对不起，知识库中没有找到相关资料。"
        
        return self._generate(query, context)

if __name__ == "__main__":
    rag = MedicalRAG()
    while True:
        user_query = input("\n请输入您的医疗咨询: ")
        if user_query.lower() in ['q', 'exit']: break
        
        print("\n正在生成专业回答...\n")
        ans = rag.answer(user_query)
        print(f"【医学助手】: {ans}\n")