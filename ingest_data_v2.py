import pickle
import json
import gzip
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 配置
DB_FILE = "./milvus_medical_v2.db"
COLLECTION_NAME = "medical_knowledge"
MODEL_PATH = "/home/kdyy/project/FastAPI/rag_bge_milvus/bge-large-zh-v1.5"
DATA_PATH = "./data/qa_50000.jsonl"
SPARSE_VEC_PATH = "/home/kdyy/project/FastAPI/Medical_Rag/corpus_vecs.pkl" # 稀疏向量 

# 1. 加载资源
client = MilvusClient(uri=DB_FILE)
dense_model = SentenceTransformer(MODEL_PATH)

print("加载稀疏向量...")
with open(SPARSE_VEC_PATH, "rb") as f:
    sparse_vectors = pickle.load(f)

print("加载原始数据...")
raw_data = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        raw_data.append(json.loads(line))

# 2. 批量处理并入库
BATCH_SIZE = 32
total = len(raw_data)

for i in tqdm(range(0, total, BATCH_SIZE)):
    batch_json = raw_data[i : i + BATCH_SIZE]
    batch_sparse = sparse_vectors[i : i + BATCH_SIZE]
    
    # 构造文本
    texts = [f"问题: {item['question']}\n答案: {item['answer']}" for item in batch_json]
    
    # 生成稠密向量 (Dense)
    dense_vecs = dense_model.encode(texts, normalize_embeddings=True).tolist()
    
    # 组合数据
    insert_data = []
    for j in range(len(texts)):
        insert_data.append({
            "text": texts[j],
            "dense_vector": dense_vecs[j],
            "sparse_vector": batch_sparse[j] # 这里的格式应该是 {tid: score}
        })
    
    # 插入数据库
    client.insert(collection_name=COLLECTION_NAME, data=insert_data)

print(f"✅ 成功将 {total} 条双向量数据存入 Milvus！")