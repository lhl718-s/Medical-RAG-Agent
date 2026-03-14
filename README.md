# 🏥 Med-Agent RAG: 工业级医疗知识图谱问答智能体

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Workflow-orange.svg)
![Milvus](https://img.shields.io/badge/Milvus-Hybrid_Search-00A3E0.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Streaming_API-009688.svg)

## 📖 项目简介 (Overview)

针对医疗咨询场景对严谨性与专业术语（长尾词）匹配的极致要求，本项目构建了一个基于 **LangGraph** 的自反思型 RAG 智能体系统。

项目摒弃了传统的单向线性检索流水线，通过构建 **`Rewrite -> Retrieve -> Generate -> Fact-Check`** 的状态机回路，实现了对模型幻觉的有效拦截。结合 **Milvus** 双路混合检索（Dense + Sparse）与 **FastAPI** 异步流式输出，打造了一个低延迟、低幻觉、支持多轮语境的医疗问答后段引擎。

## ✨ 核心技术亮点 (Key Features)

* **🧠 基于 LangGraph 的自反思纠错回路 (Self-Correction)**
  * 构建了具备“事实一致性核查”能力的 Agent 拓扑图。LLM 生成回答后需经过审计节点交叉比对，检测到潜在幻觉（如忽略高风险用药禁忌）将自动触发重试机制与动态 Prompt 打压，将医疗风险降至最低。
* **🔍 多路混合检索与自定义稀疏引擎 (Hybrid Search)**
  * **Dense**: 使用 BGE-M3 捕获深层语义特征。
  * **Sparse**: 弃用通用分词，引入北京大学 `pkuseg-medicine` 构建领域级 BM25 稀疏倒排索引，精准捕获“布洛芬”、“阿司匹林”等长尾专有名词。
  * **Fusion**: 采用 RRF（倒数秩融合）算法抹平量纲差异，特定医疗名词的 Recall@5 提升约 35%。
* **🔄 多轮对话与上下文独立改写 (Query Rewrite)**
  * 引入滑动窗口记忆机制（Sliding Window Context），通过独立的 Rewrite 节点将包含代词的模糊多轮追问（如“那它能吃吗？”）重构为独立医学查询，彻底解决向量库“无记忆”的痛点。
* **⚡ 工业级异步后端交付 (High-Performance API)**
  * 基于 FastAPI 构建，利用 Pydantic 实现严格的协议校验。通过 Server-Sent Events (SSE) 协议与 LangGraph 的 `astream` 结合，将 Agent 的“思维链路”与最终答案实时流式推送至前端。
* **📊 全链路可观测性 (Observability)**
  * 全面接入 LangSmith 分布式追踪，实现毫秒级节点耗时监控与 Token 消耗统计，为排查检索召回 Bad Case 提供可视化面板。

## 🏗️ 架构概览 (Architecture)

1. **意图网关 & 改写**：判断是否为医疗意图，结合历史对话将 Query 改写为 Standalone Query。
2. **多路并发检索**：Milvus 并发执行 HNSW 稠密检索与倒排稀疏检索。
3. **二阶段精排**：BGE-Reranker (Cross-Encoder) 对 Top-20 粗筛结果进行深度交叉打分，截断至 Top-3。
4. **生成与自省**：LLM 生成建议 -> 审计节点核查 -> (若存在幻觉) -> 警告大模型并重新生成。

## 📁 目录结构 (Project Structure)

```text
Medical_Rag/
├── medical_agent_muti_dialogue_rag.py       # LangGraph 核心智能体逻辑与多轮对话入口
├── medical_rag_v2_Langsmith.py          # 底层 RAG 流水线 (检索、精排、融合)
├── medical_server.py          # FastAPI 异步流式服务端接口
├── models.py                  # Pydantic 数据协议校验模型
├── sparse.py                  # 基于 pkuseg 的 BM25 词频计算与分词逻辑
├── data/                      # 样本数据集目录
├── .env.example               # 环境变量配置模板
└── README.md                  # 项目说明文档
