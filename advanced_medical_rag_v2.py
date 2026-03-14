# 意图识别、双路混合检索、RRF 融合、BGE 精排
import os
import re
import pickle
import gzip
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from openai import OpenAI
from stopwords import filter_stopwords # 导入你之前写的过滤函数

# 1. 基础配置
DB_FILE = "./milvus_medical_v2.db"
COLLECTION_NAME = "medical_knowledge"
MODEL_PATH = "/home/kdyy/project/FastAPI/rag_bge_milvus/bge-large-zh-v1.5"
RERANK_MODEL_PATH = "/home/kdyy/project/FastAPI/Medical_Rag/bge_reranker_v2_v3"
VOCAB_PATH = "/home/kdyy/project/FastAPI/Medical_Rag/bm25_model.pkl.gz" # 你刚才生成的 BM25 模型

client_llm = OpenAI(
    api_key="sk-9fdd123439314a459d37a7e0ae6cf7da", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

class MedicalRAGPipeline:
    def __init__(self):
        self.milvus_client = MilvusClient(uri=DB_FILE)
        self.dense_model = SentenceTransformer(MODEL_PATH)
        self.reranker = FlagReranker(RERANK_MODEL_PATH, use_fp16=True)
        
        # 加载 BM25 相关资源
        print("加载 BM25 词表...")
        with gzip.open(VOCAB_PATH, 'rb') as f:
            # 这里对应你 SimpleBM25 类保存的内容
            self.bm25_data = pickle.load(f)
        
    def _get_sparse_vector(self, query):
        """将 Query 转为稀疏向量 (tid: score)"""
        import pkuseg
        seg = pkuseg.pkuseg(model_name="medicine")
        tokens = [t.strip() for t in seg.cut(query) if t.strip()]
        tokens = filter_stopwords(tokens) # 使用之前的停用词逻辑
        
        # 计算 BM25 得分
        tf = {}
        for t in tokens:
            if t in self.bm25_data['vocab']:
                tid = self.bm25_data['vocab'][t]
                tf[tid] = tf.get(tid, 0) + 1
        
        if not tf: return {}
        
        vec = {}
        k1, b, avgdl = self.bm25_data['k1'], self.bm25_data['b'], self.bm25_data['avgdl']
        dl = sum(tf.values())
        K = k1 * (1 - b + b * dl / avgdl)
        for tid, f in tf.items():
            idf = self.bm25_data['idf'][tid]
            score = idf * (f * (k1 + 1)) / (f + K)
            vec[tid] = float(score)
        return vec

    def _intent_recognition(self, query):
        """意图识别：防止非医疗问题干扰"""
        prompt = f"判断该问题是否为医疗健康相关。只回答'是'或'否'。\n问题: {query}"
        res = client_llm.chat.completions.create(model="qwen3.5-35b-a3b", messages=[{"role": "user", "content": prompt}], temperature=0)
        return "是" in res.choices[0].message.content

    def hybrid_search(self, query, top_k=20):
        # A. 生成稠密向量
        dense_vec = self.dense_model.encode([query], normalize_embeddings=True)[0].tolist()
        # B. 生成稀疏向量
        sparse_vec = self._get_sparse_vector(query)

        # C. 构造多路检索请求
        req_dense = AnnSearchRequest(
            data=[dense_vec], anns_field="dense_vector",
            param={"metric_type": "COSINE"}, limit=top_k
        )
        req_sparse = AnnSearchRequest(
            data=[sparse_vec], anns_field="sparse_vector",
            param={"metric_type": "IP"}, limit=top_k
        )

        # D. RRF 排名融合
        res = self.milvus_client.hybrid_search(
            collection_name=COLLECTION_NAME,
            reqs=[req_dense, req_sparse],
            ranker=RRFRanker(k=60), # 标准 RRF 参数
            limit=top_k,
            output_fields=["text"]
        )
        return [hit['entity']['text'] for hit in res[0]] if res else []

    def answer(self, query):
        # 1. 意图拦截
        if not self._intent_recognition(query):
            return "我是医疗助手，请咨询健康相关问题。"

        # 2. 混合检索
        docs = self.hybrid_search(query)
        if not docs:
            return "知识库未找到匹配内容。"

        # 3. BGE 重排序
        # 混合检索出的 20 条中，选出最强的 3 条
        pairs = [[query, doc] for doc in docs]
        scores = self.reranker.compute_score(pairs)
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        # 过滤相关性过低的内容 (设置一个动态阈值)
        final_docs = [d for d, s in scored_docs if s > -1.0][:3]
        if not final_docs: return "检索结果相关度较低，建议寻求医生帮助。"

        # 4. 最终生成
        context = "\n\n".join([f"【参考{i+1}】: {d}" for i, d in enumerate(final_docs)])
        messages = [
            {"role": "system", "content": "你是一名医学专家。请基于提供的资料给出严谨的回答。"},
            {"role": "user", "content": f"资料：\n{context}\n\n问题：{query}"}
        ]
        
        response = client_llm.chat.completions.create(
            model="qwen3.5-35b-a3b",
            messages=messages,
            temperature=0.1
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    rag = MedicalRAGPipeline()
    while True:
        q = input("\n咨询: ")
        if q == 'q': break
        print(f"\n【医生建议】:\n{rag.answer(q)}")