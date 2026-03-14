####引入 LangGraph   LangSmith   multi_dialogue ####
import os
from typing import List, TypedDict, Literal ,Dict
from langgraph.graph import StateGraph, END
from langsmith import traceable
from dotenv import load_dotenv

# 1. 确保环境变量加载
load_dotenv()

# 2. 导入你之前的稳定 Pipeline 组件
from medical_rag_v2_Langsmith import MedicalRAGPipeline, client_llm

# --- 定义智能体状态 ---
class AgentState(TypedDict):
    query: str                   # 用户原始问题
    chat_history: List[Dict]     # 历史对话 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    standalone_query: str        # 改写后的独立问题 (新加的!)
    docs: List[str]              # 检索到的文档
    answer: str                  # 模型生成的回答
    retry_count: int             # 内部纠错重试次数

# 初始化底层 RAG
rag_core = MedicalRAGPipeline()

# ===================== 节点定义 =====================

@traceable(run_type="llm", name="Node_Rewrite_Query")
def rewrite_node(state: AgentState):
    """
    节点 0：问题改写节点。
    结合历史对话，将当前可能包含指代词的模糊问题，改写为可独立检索的完整问题。
    """
    original_query = state["query"]
    history = state.get("chat_history", [])
    
    # 如果没有历史对话，说明是第一轮，不需要改写
    if not history:
        print(f"--- [智能体思维]：首轮对话，无需改写，检索词: '{original_query}' ---")
        return {"standalone_query": original_query}
    
    print("--- [智能体思维]：检测到历史对话，正在进行问题改写 (Query Rewrite) ---")
    
    # 将历史对话格式化为字符串，方便喂给 LLM
    history_str = ""
    for msg in history:
        role = "用户" if msg["role"] == "user" else "AI医生"
        history_str += f"{role}：{msg['content']}\n"
    
    # 构建改写 Prompt
    rewrite_prompt = f"""
    你是一个医疗领域的语义分析专家。
    请根据以下历史对话，将用户的最新问题改写为一个语义完整、独立、可用于搜索引擎检索的医疗查询。
    注意：
    1. 补全最新问题中的代词（如把“它”替换为具体的疾病或药物名）。
    2. 如果最新问题已经很完整，或者与历史对话无关，请保持原意或稍作润色。
    3. 只输出改写后的句子，不要输出任何其他解释和标点。

    历史对话：
    {history_str}

    用户最新问题：{original_query}
    
    改写后的独立问题：
    """
    
    from medical_rag_v2_Langsmith import client_llm
    res = client_llm.chat.completions.create(
        model="qwen3.5-35b-a3b", # 或者你目前用的模型
        messages=[{"role": "user", "content": rewrite_prompt}],
        temperature=0.1
    )
    
    rewritten_query = res.choices[0].message.content.strip()
    print(f"--- [改写结果]：'{original_query}' -> '{rewritten_query}' ---")
    
    return {"standalone_query": rewritten_query}



@traceable(run_type="retriever", name="Node_Retrieve")
def retrieve_node(state: AgentState):
    """
    节点 1：检索知识库。
    注意：现在我们使用改写后的 standalone_query 进行精确检索！
    """
    # 优先使用改写后的独立问题，如果没有（比如出错了），回退到原始问题
    query_to_search = state.get("standalone_query", state["query"])
    print(f"--- [智能体思维]：正在检索与 '{query_to_search}' 相关的医学资料 ---")
    
    docs = rag_core.hybrid_search(query_to_search)
    # docs= rag_core._rerank_documents(query_to_search, docs)
    return {"docs": docs}

@traceable(run_type="llm", name="Node_Generate")
def generate_node(state: AgentState):
    """
    节点 2：基于检索到的资料和历史对话生成建议。
    """
    print("--- [智能体思维]：正在结合上下文与资料生成专业回答 ---")
    current_retry = state.get("retry_count", 0) +1
    context = "\n\n".join([f"【参考{i+1}】: {d}" for i, d in enumerate(state["docs"][:3])])
    
    # 提取历史对话拼接到 Prompt 中，让 AI 知道之前聊了什么
    history_str = ""
    for msg in state.get("chat_history", []):
        role = "用户" if msg["role"] == "user" else "AI医生"
        history_str += f"{role}：{msg['content']}\n"
    
    prompt = f"""
    资料：
    {context}

    历史对话：
    {history_str if history_str else "无"}

    用户最新问题：{state['query']}
    
    请基于上述资料，结合历史对话的语境，给出严谨回复。若资料不足请直说。
    """
    
    res = client_llm.chat.completions.create(
        model="qwen3.5-35b-a3b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return {"answer": res.choices[0].message.content,"retry_count": current_retry}


@traceable(run_type="llm", name="Node_Fact_Check")
def grade_answer_node(state: AgentState) -> Literal["reliable", "unreliable"]:
    """
    节点 3：【核心纠错逻辑】。
    判断 LLM 生成的内容是否背离了参考资料，或者是否包含幻觉。
    """
    print("--- [智能体思维]：正在进行事实一致性核查 (Self-Correction) ---")
    
    check_prompt = f"""
    你是专业医疗事实核查员。
    任务：严格检查回答是否完全基于参考资料，绝对不能出现资料以外的医疗信息。
    规则：
    1. 回答中所有信息必须来自参考资料
    2. 不能编造疾病、病因、用药、治疗方案
    3. 不能夸大、不能臆测
    4. 若回答完全合规 → 只输出：是
    5. 若存在幻觉、编造、猜测 → 只输出：否

    回答内容：{state['answer']}
    参考资料：{state['docs']}
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
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

# 定义执行流
workflow.set_entry_point("rewrite")
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("retrieve", "generate")

# 定义条件跳转：根据 grade_answer_node 的结果决定去 END 还是 back to retrieve
workflow.add_conditional_edges(
    "generate",
    grade_answer_node,
    {
        "reliable": END,
        "unreliable": "generate"  
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
    
    global_chat_history = []       #初始化一个历史列表  做记录
    while True:
        user_query = input("咨询：")
        if user_query.lower() in ['q', 'quit', 'exit']:
            break
            
        inputs = {"query": user_query, 
                  "chat_history": global_chat_history,
                  "retry_count": 0}
        
        # 实时打印节点运行状态，让你在终端看到大脑的“跳动”
        final_answer = ""
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"  > [系统消息]：完成节点 [{key}]")
                if "answer" in value:
                    final_answer = value["answer"]
        
        print(f"\n【医生建议】:\n{final_answer}\n")
        print("-" * 50)
        # 【核心新增】：一轮对话结束后，把这一问一答存入历史记录
        global_chat_history.append({"role": "user", "content": user_query})
        global_chat_history.append({"role": "assistant", "content": final_answer})
        
        # 为了防止上下文过长导致 LLM 报错，我们只保留最近的 2 轮对话（即 4 条消息）
        global_chat_history = global_chat_history[-6:]