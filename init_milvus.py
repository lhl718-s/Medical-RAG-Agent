# # 在 Milvus 中创建一个“池子”----Schema  (定义了Collections 的 数据结构)，并规定好里面存哪些东西
# from pymilvus import MilvusClient, DataType

# # 1. 核心配置：使用本地文件作为数据库
# # 这就是 Milvus Lite 的精髓：uri 指向一个本地 .db 文件
# DB_FILE = "./milvus_medical.db"
# COLLECTION_NAME = "medical_knowledge"
# DIMENSION = 1024  # BGE-Large 的输出维度

# # 2. 初始化客户端 (连接到本地文件)
# # 参考官方文档：对于 Milvus Lite，uri 必须以 .db 结尾
# client = MilvusClient(uri=DB_FILE)

# print(f"正在本地初始化数据库文件: {DB_FILE}")

# # 3. 如果集合已存在则删除 (方便重复运行调试)
# if client.has_collection(COLLECTION_NAME):
#     client.drop_collection(COLLECTION_NAME)

# # 4. 定义 Schema (表结构)
# # 参考官方建议使用 create_schema
# schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

# # 添加字段
# # 主键 id
# schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
# # 医疗原始文本 (question + answer)
# schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
# # 稠密向量字段 (存储 BGE-M3 生成的 Embedding)
# schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)

# # 5. 定义索引参数
# # 参考官方文档：Lite 版本同样支持 HNSW 索引
# index_params = client.prepare_index_params()
# index_params.add_index(
#     field_name="dense_vector",
#     index_type="AUTOINDEX",    # 索引类型
#     metric_type="COSINE",  # 度量类型：医疗语义相似度推荐使用余弦相似度
# )

# # 6. 正式创建集合
# client.create_collection(
#     collection_name=COLLECTION_NAME,
#     schema=schema,
#     index_params=index_params
# )

# print(f"✅ Milvus Lite 初始化成功！")
# print(f"数据文件位置: {DB_FILE}")
# print(f"当前集合列表: {client.list_collections()}")



from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="./milvus_medical_v2.db")
COLLECTION_NAME = "medical_knowledge"

if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)

schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

# 1. 基础字段
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

# 2. 稠密向量 (BGE-Large v1.5)
schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)

# 3. 稀疏向量 (BM25 - 模仿项目逻辑)
schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

index_params = client.prepare_index_params()
index_params.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
# 稀疏向量必须使用 IP (内积) 度量
index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

client.create_collection(COLLECTION_NAME, schema=schema, index_params=index_params)
print("✅ 专业的双向量集合初始化成功")