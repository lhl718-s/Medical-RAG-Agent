# 分词 并转化为稀疏向量
import math
import pickle
import gzip
from collections import Counter
import pkuseg
from datasets import load_dataset
from stopwords import filter_stopwords  # 假设你已有此函数

class SimpleBM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.seg = pkuseg.pkuseg(model_name="medicine")           #程序会自动下载所对应的细领域的模型
        
        # 核心数据
        self.vocab = {}      # token -> id
        self.idf = {}        # id -> idf_score
        self.avgdl = 0
        self.N = 0

    def tokenize(self, text):
        """分词并过滤"""
        tokens = [t.strip() for t in self.seg.cut(text) if t.strip()]   # 默认使用细领域模型 进行分词
        return filter_stopwords(tokens)

    def fit(self, texts):
        """一次性构建词表和计算 IDF"""
        self.N = len(texts)
        all_tokens = []
        df = Counter()
        total_len = 0

        print(f"开始分词处理 {self.N} 条文档...")
        for text in texts:
            tokens = self.tokenize(text)
            all_tokens.append(tokens)
            total_len += len(tokens)
            
            # 统计文档频率 (DF)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab)
                df[self.vocab[t]] += 1

        self.avgdl = total_len / self.N
        
        # 计算每个词的 IDF
        print("计算 IDF...")
        for token, tid in self.vocab.items():
            d = df[tid]
            # BM25 标准 IDF 公式
            self.idf[tid] = math.log((self.N - d + 0.5) / (d + 0.5) + 1.0)
        
        return all_tokens

    def transform(self, tokens):
        """将分好词的文档转为稀疏向量: {tid: score}"""
        tf = Counter([self.vocab[t] for t in tokens if t in self.vocab])
        dl = sum(tf.values())
        
        vec = {}
        K = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        for tid, freq in tf.items():
            score = self.idf[tid] * (freq * (self.k1 + 1)) / (freq + K)
            vec[tid] = round(score, 4)
        return vec

    def save(self, path):
        with gzip.open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        instance = cls()
        with gzip.open(path, 'rb') as f:
            instance.__dict__.update(pickle.load(f))
        return instance

# ---------- 使用示例 ----------
if __name__ == "__main__":
    # 1. 加载数据
    ds = load_dataset("json", data_files="data/qa_50000.jsonl", split="train")
    texts = ds['text']

    # 2. 训练
    bm25 = SimpleBM25()
    all_doc_tokens = bm25.fit(texts) # 训练并返回分词结果

    # 3. 向量化（保存为稀疏格式）
    print("生成稀疏向量...")
    corpus_vectors = [bm25.transform(toks) for toks in all_doc_tokens]

    # 4. 保存模型和数据
    bm25.save("bm25_model.pkl.gz")
    with open("corpus_vecs.pkl", "wb") as f:
        pickle.dump(corpus_vectors, f)
        
    print(f"处理完成！词表大小: {len(bm25.vocab)}")