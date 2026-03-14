import os
import re
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 1. 基础配置
DB_FILE = "./milvus_medical.db"
COLLECTION_NAME = "medical_knowledge"
MODEL_PATH = "/home/kdyy/project/FastAPI/rag_bge_milvus/bge-large-zh-v1.5"

# 2. DeepSeek API 配置 (请确保你已 export DEEPSEEK_API_KEY)
client_llm = OpenAI(
    api_key="sk-9fdd123439314a459d37a7e0ae6cf7da", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

class MedicalRAGDeepSeek:
    def __init__(self):
        # 初始化 Milvus 和 Embedding 模型
        self.milvus_client = MilvusClient(uri=DB_FILE)
        self.embed_model = SentenceTransformer(MODEL_PATH)

    def _retrieve(self, query, top_k=3):
        """参考项目 IngestionPipeline 对应的检索逻辑"""
        # 生成查询向量并归一化
        query_vector = self.embed_model.encode([query], normalize_embeddings=True)[0].tolist()
        
        # 执行向量搜索
        results = self.milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            limit=top_k,
            output_fields=["text"],
            search_params={"metric_type": "COSINE"}
        )
        return [res['entity']['text'] for res in results[0]]

    def answer(self, query):
        # 第一步：知识检索
        contexts = self._retrieve(query)
        if not contexts:
            return "抱歉，在医学知识库中未找到相关参考信息，建议咨询专业医生。"

        # 第二步：构造上下文
        context_str = "\n\n".join([f"【参考医疗案例/文献 {i+1}】:\n{ctx}" for i, ctx in enumerate(contexts)])

        # 第三步：构建专业 Prompt 
        # 我们使用 deepseek-reasoner，它可以自行思考逻辑
        prompt_messages = [
            {
                "role": "system", 
                "content": (
                    "你是一位严谨的医疗顾问。你的任务是根据提供的参考资料回答用户问题。\n"
                    "要求：\n"
                    "1. 回答必须严格基于【参考资料】，如果资料中没有相关信息，请明确告知。\n"
                    "2. 保持语言的专业性与亲和力，避免机械化地罗列条目。\n"
                    "3. 绝对不要给出具体的药物剂量建议。\n"
                    "4. 在结尾强制包含免责声明：'本建议仅供参考，请以线下医院诊断为准。'"
                )
            },
            {
                "role": "user", 
                "content": f"【参考资料】：\n{context_str}\n\n【用户问题】：\n{query}"
            }
        ]

        # 第四步：调用 DeepSeek (使用 reasoner 模型)
        try:
            response = client_llm.chat.completions.create(
                model="qwen3.5-35b-a3b", # 使用你代码里的思考模型
                messages=prompt_messages,
                stream=False
            )
            
            # 获取最终答案
            final_answer = response.choices[0].message.content
            
            # (可选) 获取思考过程以供调试：reasoning = response.choices[0].message.reasoning_content
            
            return final_answer

        except Exception as e:
            return f"DeepSeek 接入异常: {str(e)}"

# 执行演示
if __name__ == "__main__":
    # 检查 API Key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误：请先设置环境变量 export DEEPSEEK_API_KEY='你的KEY'")
    else:
        rag_sys = MedicalRAGDeepSeek()
        
        print("--- 医疗 RAG 问答系统 (DeepSeek 版) ---")
        while True:
            user_input = input("\n患者咨询 (q退出): ")
            if user_input.lower() == 'q': break
            
            print("\n系统正在调取知识库并思考回答...\n")
            result = rag_sys.answer(user_input)
            print(f"【医生建议】：\n{result}")