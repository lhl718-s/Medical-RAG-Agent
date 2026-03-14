####引入 LangGraph   LangSmith  ####
import os
from typing import List, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langsmith import traceable
from dotenv import load_dotenv

# 1. 确保环境变量加载
load_dotenv()

# 2. 导入你之前的稳定 Pipeline 组件
from medical_rag_v2_Langsmith import MedicalRAGPipeline, client_llm

# --- 定义智能体状态 ---
class AgentState(TypedDict):
    query: str          # 用户当前问题
    docs: List[str]     # 检索到的文档
    answer: str         # 模型生成的回答
    retry_count: int    # 内部纠错重试次数

# 初始化底层 RAG
rag_core = MedicalRAGPipeline()

# ===================== 节点定义 =====================

@traceable(run_type="retriever", name="Node_Retrieve")
def retrieve_node(state: AgentState):
    """
    节点 1：检索知识库。
    如果在纠错循环中，retry_count 会增加。
    """
    print(f"--- [智能体思维]：正在检索与 '{state['query']}' 相关的医学资料 ---")
    # 调用你之前写好的混合检索逻辑
    docs = rag_core.hybrid_search(state["query"])
    return {"docs": docs, "retry_count": state.get("retry_count", 0) + 1}

@traceable(run_type="llm", name="Node_Generate")
def generate_node(state: AgentState):
    """
    节点 2：基于检索到的资料生成建议。
    """
    print("--- [智能体思维]：正在整合资料并生成专业回答 ---")
    context = "\n\n".join([f"【参考{i+1}】: {d}" for i, d in enumerate(state["docs"][:3])])
    prompt = f"资料：\n{context}\n\n用户问题：{state['query']}\n请基于上述资料给出严谨回复，若资料不足请直说。"
    
    res = client_llm.chat.completions.create(
        model="qwen3.5-35b-a3b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return {"answer": res.choices[0].message.content}

@traceable(run_type="llm", name="Node_Fact_Check")
def grade_answer_node(state: AgentState) -> Literal["reliable", "unreliable"]:
    """
    节点 3：【核心纠错逻辑】。
    判断 LLM 生成的内容是否背离了参考资料，或者是否包含幻觉。
    """
    print("--- [智能体思维]：正在进行事实一致性核查 (Self-Correction) ---")
    
    check_prompt = f"""
    任务：核查回答是否完全基于资料，严禁幻觉。
    回答内容：{state['answer']}
    参考资料：{state['docs']}
    
    如果回答没有编造资料以外的内容，且严谨地回答了问题，输出 '是'，否则输出 '否'。
    """
    
    res = client_llm.chat.completions.create(
        model="qwen3.5-35b-a3b",
        messages=[{"role": "user", "content": check_prompt}],
        temperature=0
    )
    
    # 逻辑判断：如果被判定为“否”且重试次数不到2次，则触发循环重查
    if "是" in res.choices[0].message.content:
        print("--- [核查通过]：生成内容与事实一致 ---")
        return "reliable"
    else:
        if state["retry_count"] < 2:
            print(f"--- [报警]：检测到幻觉风险，触发第 {state['retry_count']} 次自动重修 ---")
            return "unreliable"
        else:
            print("--- [注意]：已达最大修正次数，输出当前最优版本 ---")
            return "reliable"

# ===================== 构建 LangGraph 工作流 =====================

workflow = StateGraph(AgentState)

# 注册节点
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

# 定义执行流
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")

# 定义条件跳转：根据 grade_answer_node 的结果决定去 END 还是 back to retrieve
workflow.add_conditional_edges(
    "generate",
    grade_answer_node,
    {
        "reliable": END,
        "unreliable": "retrieve"  # 核心！指向回去，形成循环
    }
)

# 编译应用
app = workflow.compile()

# ===================== 持续交互入口 =====================

if __name__ == "__main__":
    print("\n" + "="*30)
    print("🏥 专业医疗 RAG 智能体 (Agent) 已就绪")
    print("输入 'q' 退出对话")
    print("="*30 + "\n")
    
    while True:
        user_query = input("咨询：")
        if user_query.lower() in ['q', 'quit', 'exit']:
            break
            
        inputs = {"query": user_query, "retry_count": 0}
        
        # 实时打印节点运行状态，让你在终端看到大脑的“跳动”
        final_answer = ""
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"  > [系统消息]：完成节点 [{key}]")
                if "answer" in value:
                    final_answer = value["answer"]
        
        print(f"\n【医生建议】:\n{final_answer}\n")
        print("-" * 50)