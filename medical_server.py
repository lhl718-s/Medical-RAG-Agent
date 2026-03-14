"""
curl -X POST "http://127.0.0.1:8000/api/v1/chat/stream" \
     -H "Content-Type: application/json" \
     -d '{"query": "我有严重哮喘，牙疼可以吃布洛芬吗？"}'
终端使用这个来测试接口
"""

import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from fastapi.responses import RedirectResponse


# 1. 加载环境变量并导入你的核心组件
load_dotenv()
from medical_agent_langgraph import app as agent_app  # 导入之前定义的 LangGraph 实例

# 初始化 FastAPI 应用
server = FastAPI(
    title="🏥 医疗智能 RAG 问答 API",
    description="基于 LangGraph 医疗咨询接口",
    version="1.0.0"
)

# 2. 定义 Pydantic 数据协议（体现严格校验）
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="用户咨询的问题", example="胃溃疡患者能吃布洛芬吗？")
    user_id: Optional[str] = Field("default_user", description="用户唯一标识")

class AgentEventResponse(BaseModel):
    node: str = Field(..., description="当前运行的智能体节点")
    status: str = Field(..., description="节点状态或摘要信息")
    answer: Optional[str] = Field(None, description="最终生成的回答")

# 3. 核心业务逻辑：异步流式生成器
async def agent_stream_generator(query: str):
    """
    将 LangGraph 的执行过程转化为 SSE (Server-Sent Events) 流
    """
    inputs = {"query": query, "retry_count": 0}
    
    # 使用 astream 异步流式获取节点更新
    async for event in agent_app.astream(inputs):
        for node_name, output in event.items():
            # 构建实时状态包
            data = {
                "node": node_name,
                "status": f"节点 [{node_name}] 处理完成"
            }
            # 如果是生成节点，提取中间答案
            if "answer" in output:
                data["answer"] = output["answer"]
            
            # 以标准的 SSE 格式推送给前端
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            
            # 为了演示流式效果，模拟极短的 I/O 等待
            await asyncio.sleep(0.1)


@server.get("/")
async def root():
    # 访问根目录时自动跳转到 API 文档页，更专业
    return RedirectResponse(url="/docs")

# 4. 定义 API 路由
@server.post("/api/v1/chat/stream")
async def chat_stream(request: QueryRequest):
    """
    流式问答接口：实时推送智能体思维链路及最终答案
    """
    try:
        return StreamingResponse(
            agent_stream_generator(request.query),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@server.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "model": "qwen3.5-35b-agent"}

# 5. 启动服务
if __name__ == "__main__":
    import uvicorn
    # 使用 uvicorn 启动，支持异步高并发
    uvicorn.run(server, host="0.0.0.0", port=8000)