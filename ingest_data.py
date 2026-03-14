###读取数据， 向量化， 批量写入Milvus 
import json
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 1. 基础配置
DB_FILE = "./milvus_medical.db"
COLLECTION_NAME = "medical_knowledge"
MODEL_PATH = "/home/kdyy/project/FastAPI/rag_bge_milvus/bge-large-zh-v1.5" # 你的模型路径
DATA_PATH = "./data/qa_50000.jsonl"
BATCH_SIZE = 32  # 每次处理32条数据，避免显存/内存溢出

# 2. 初始化资源
client = MilvusClient(uri=DB_FILE)
# 加载 BGE 模型 (确保你已经安装了 sentence-transformers)
model = SentenceTransformer(MODEL_PATH)

def ingest():
    # 3. 读取数据
    print(f"正在从 {DATA_PATH} 加载数据...")
    raw_data = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            raw_data.append(json.loads(line))
    
    # 为了演示，我们先处理前 1000 条，你可以去掉 [:1000] 处理全量
    process_data = raw_data
    total = len(process_data)

    print(f"开始生成向量并入库，共 {total} 条数据...")

    # 4. 分批处理
    for i in tqdm(range(0, total, BATCH_SIZE)):
        batch = process_data[i : i + BATCH_SIZE]
        
        # 拼接文本：参考项目逻辑，将问题和答案组合
        # 这样检索时既能匹配问题，也能匹配答案
        texts = [f"问题: {item['question']}\n答案: {item['answer']}" for item in batch]
        
        # 生成稠密向量 (Embedding)
        # BGE 官方建议在检索任务中给 Query 加上指令，但存入库的文档通常不需要
        vectors = model.encode(texts, normalize_embeddings=True)
        
        # 准备存入 Milvus 的数据格式
        data = []
        for j, text in enumerate(texts):
            data.append({
                "text": text,
                "dense_vector": vectors[j].tolist()
            })
        
        # 批量插入 Milvus
        client.insert(collection_name=COLLECTION_NAME, data=data)

    print(f"✅ 成功插入 {total} 条数据！")

if __name__ == "__main__":
    ingest()