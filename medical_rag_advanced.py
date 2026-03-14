# 加了 意图识别  + Rerank 精排
import os
import re
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from FlagEmbedding import FlagReranker # 建议安装 FlagEmbedding 库

# 1. 基础配置
DB_FILE = "./milvus_medical.db"
COLLECTION_NAME = "medical_knowledge"
EMBED_MODEL_PATH = "/home/kdyy/project/FastAPI/rag_bge_milvus/bge-large-zh-v1.5"
# 建议下载并指向 BAAI/bge-reranker-v2-m3 或 bge-reranker-large
RERANK_MODEL_PATH = "/home/kdyy/project/FastAPI/Medical_Rag/bge_reranker_v2_v3" 

client_llm = OpenAI(
    api_key="sk-9fdd123439314a459d37a7e0ae6cf7da", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

class AdvancedMedicalRAG:
    def __init__(self):
        self.milvus_client = MilvusClient(uri=DB_FILE)
        self.embed_model = SentenceTransformer(EMBED_MODEL_PATH)
        # 初始化 BGE Reranker
        self.reranker = FlagReranker(RERANK_MODEL_PATH, use_fp16=True)

    def _query_classification(self, query: str) -> bool:
        """
        步骤 1: 查询分类 (参考 SearchGraph.py 意图识别逻辑)
        """
        prompt = f"判断以下用户输入是否属于医疗、健康、疾病或生理咨询范畴。仅回答'是'或'否'。\n输入: {query}"
        try:
            response = client_llm.chat.completions.create(
                model="qwen3.5-35b-a3b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            result = response.choices[0].message.content.strip()
            return "是" in result
        except:
            return True # 异常情况默认通过

    def _retrieve_and_rerank(self, query, top_k=10, final_k=3):
        """
        步骤 2 & 3: 初筛 + BGE-Reranker 精排
        """
        # A. 初筛 (Recall)
        query_vector = self.embed_model.encode([query], normalize_embeddings=True)[0].tolist()
        recall_results = self.milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            limit=top_k,
            output_fields=["text"],
            search_params={"metric_type": "COSINE"}
        )
        
        if not recall_results[0]:
            return []

        # B. 准备精排数据
        passages = [res['entity']['text'] for res in recall_results[0]]
        query_passage_pairs = [[query, p] for p in passages]

        # C. 执行精排
        # Reranker 返回的分数通常没有固定区间，但相关性越高分数越大
        scores = self.reranker.compute_score(query_passage_pairs)
        
        # D. 组合分数并按阈值过滤 (阈值设定参考项目经验，通常 0 以上代表有一定相关性)
        RERANK_THRESHOLD = 0.0 
        scored_docs = sorted(
            zip(passages, scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 仅保留高于阈值的文档
        valid_docs = [doc for doc, score in scored_docs if score > RERANK_THRESHOLD]
        return valid_docs[:final_k]

    def answer(self, query):
        # 1. 意图拦截
        if not self._query_classification(query):
            return "抱歉，我是一名医疗助手，仅能回答健康医疗相关的问题。您的问题似乎涉及其他领域。"

        # 2. 检索 + 精排
        valid_contexts = self._retrieve_and_rerank(query)
        if not valid_contexts:
            return "抱歉，在医疗知识库中未找到能回答此问题的确切依据，请咨询专业医生。"

        # 3. 构造 Prompt (参考项目模板)
        context_str = "\n\n".join([f"【权威资料 {i+1}】:\n{ctx}" for i, ctx in enumerate(valid_contexts)])
        
        prompt_messages = [
            {
                "role": "system", 
                "content": "你是一位严谨的医疗顾问。请严格基于参考资料回答。如果资料与问题无关，请直接回答无法回答，不得推测或引用站外知识。"
            },
            {
                "role": "user", 
                "content": f"【参考资料】：\n{context_str}\n\n【用户问题】：\n{query}"
            }
        ]

        # 4. 调用大模型 (qwen3.5-35b-a3b)
        try:
            response = client_llm.chat.completions.create(
                model="qwen3.5-35b-a3b", 
                messages=prompt_messages,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"服务响应异常: {str(e)}"

if __name__ == "__main__":
    rag_sys = AdvancedMedicalRAG()
    while True:
        user_input = input("\n咨询 (q退出): ")
        if user_input.lower() == 'q': break
        print("\n正在识别意图并精准检索...\n")
        print(f"【医生回复】：\n{rag_sys.answer(user_input)}")