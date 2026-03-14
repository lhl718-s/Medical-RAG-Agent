### 进行 检索Search  操作   
import logging
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 基础配置
DB_FILE = "./milvus_medical.db"
COLLECTION_NAME = "medical_knowledge"
MODEL_PATH = "/home/kdyy/project/FastAPI/rag_bge_milvus/bge-large-zh-v1.5"

class MedicalSearcher:
    def __init__(self):
        # 1. 初始化 Milvus 客户端 (Lite 模式)
        self.client = MilvusClient(uri=DB_FILE)
        
        # 2. 加载 BGE 模型进行推理
        logger.info(f"正在从 {MODEL_PATH} 加载模型...")
        self.model = SentenceTransformer(MODEL_PATH)
        logger.info("模型加载完成。")

    def search(self, query_text: str, top_k: int = 5):
        """
        执行语义搜索
        :param query_text: 用户输入的问题
        :param top_k: 返回最相关的文档数量
        """
        # 3. 将查询文本转化为向量 (与入库时保持一致，进行归一化)
        # 参考逻辑：BGE 模型在做检索时，建议对 Embedding 进行归一化以匹配 COSINE 指标
        query_vector = self.model.encode(
            [query_text], 
            normalize_embeddings=True
        )[0].tolist()

        # 4. 在 Milvus 中执行搜索
        # 参考 search_data.py 逻辑，指定输出字段
        search_results = self.client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            limit=top_k,
            output_fields=["text"],  # 指定返回我们在入库时存入的 'text' 字段
            search_params={"metric_type": "COSINE", "params": {}}
        )

        return search_results[0]

def main():
    searcher = MedicalSearcher()
    
    while True:
        query = input("\n请输入您的医疗咨询问题 (输入 q 退出): ").strip()
        if query.lower() == 'q':
            break
        
        print(f"正在搜索: '{query}' ...")
        results = searcher.search(query)

        print("\n--- 检索结果 ---")
        if not results:
            print("未找到相关匹配内容。")
        else:
            for i, res in enumerate(results):
                # distance 在 COSINE 下越接近 1 表示越相似
                score = res['distance']
                content = res['entity']['text']
                print(f"[{i+1}] 相关度得分: {score:.4f}")
                # 打印前 200 个字符
                print(f"内容摘要: {content[:200]}...")
                print("-" * 30)

if __name__ == "__main__":
    main()