# 导入异步编程库 asyncio
import asyncio
# 导入 JSON 处理库
import json
# 导入日志记录库
import logging
# 导入随机数生成库
import random
# 导入类型提示库
from typing import Dict, List, Union

# 导入 janus 库，用于同步和异步队列
import janus
# 导入 FastAPI 框架
from fastapi import FastAPI
# 导入 CORS 中间件
from fastapi.middleware.cors import CORSMiddleware
# 导入 FastAPI 请求对象
from fastapi.requests import Request
# 导入 Pydantic 模型库
from pydantic import BaseModel, Field
# 导入 SSE (Server-Sent Events) 库
from sse_starlette.sse import EventSourceResponse

# 导入自定义的 agent 初始化函数
from mindsearch.agent import init_agent


# 解析命令行参数的函数
def parse_arguments():
    import argparse

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="MindSearch API")
    # 添加主机地址参数，默认值为 "0.0.0.0"
    parser.add_argument("--host", default="0.0.0.0", type=str, help="Service host")
    # 添加端口号参数，默认值为 8002
    parser.add_argument("--port", default=8002, type=int, help="Service port")
    # 添加语言参数，默认值为 "cn"
    parser.add_argument("--lang", default="cn", type=str, help="Language")
    # 添加模型格式参数，默认值为 "internlm_server"
    parser.add_argument("--model_format", default="internlm_server", type=str, help="Model format")
    # 添加搜索引擎参数，默认值为 "BingSearch"
    parser.add_argument("--search_engine", default="BingSearch", type=str, help="Search engine")
    # 添加异步模式参数，默认值为 False
    parser.add_argument("--asy", default=False, action="store_true", help="Agent mode")
    # 解析命令行参数并返回
    return parser.parse_args()


# 解析命令行参数
args = parse_arguments()

# 创建 FastAPI 应用实例
app = FastAPI(docs_url="/")

# 添加 CORS 中间件，允许所有来源、凭证、方法和头部
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 定义生成参数的 Pydantic 模型
class GenerationParams(BaseModel):
    # 输入数据，可以是字符串或字典列表
    inputs: Union[str, List[Dict]]
    # 会话 ID，默认值为随机生成的整数
    session_id: int = Field(default_factory=lambda: random.randint(0, 999999))
    # 代理配置，默认值为空字典
    agent_cfg: Dict = dict()


# 处理代理消息的后处理函数
def _postprocess_agent_message(message: dict) -> dict:
    # 获取消息内容和格式化内容
    content, fmt = message["content"], message["formatted"]
    # 获取当前节点
    current_node = content["current_node"] if isinstance(content, dict) else None
    if current_node:
        # 如果存在当前节点，清空消息内容
        message["content"] = None
        # 移除 ref2url 字段
        for key in ["ref2url"]:
            fmt.pop(key, None)
        # 获取节点图
        graph = fmt["node"]
        # 移除其他节点
        for key in graph.copy():
            if key != current_node:
                graph.pop(key)
        # 如果当前节点不是 root 或 response，进一步处理
        if current_node not in ["root", "response"]:
            node = graph[current_node]
            # 移除 memory 和 session_id 字段
            for key in ["memory", "session_id"]:
                node.pop(key, None)
            node_fmt = node["response"]["formatted"]
            # 处理 thought 和 action 字段
            if isinstance(node_fmt, dict) and "thought" in node_fmt and "action" in node_fmt:
                node["response"]["content"] = None
                node_fmt["thought"] = (
                    node_fmt["thought"] and node_fmt["thought"].split("<|action_start|>")[0]
                )
                if isinstance(node_fmt["action"], str):
                    node_fmt["action"] = node_fmt["action"].split("<|action_end|>")[0]
    else:
        # 如果不存在当前节点，直接处理消息内容
        if isinstance(fmt, dict) and "thought" in fmt and "action" in fmt:
            message["content"] = None
            fmt["thought"] = fmt["thought"] and fmt["thought"].split("<|action_start|>")[0]
            if isinstance(fmt["action"], str):
                fmt["action"] = fmt["action"].split("<|action_end|>")[0]
        # 移除 node 字段
        for key in ["node"]:
            fmt.pop(key, None)
    # 返回处理后的消息
    return dict(current_node=current_node, response=message)


# 异步处理请求的函数
async def run(request: GenerationParams, _request: Request):
    async def generate():
        try:
            # 创建 janus 队列
            queue = janus.Queue()
            # 创建停止事件
            stop_event = asyncio.Event()

            # 将同步生成器包装为异步生成器
            def sync_generator_wrapper():
                try:
                    # 遍历代理生成的消息
                    for response in agent(inputs, session_id=session_id):
                        # 将消息放入同步队列
                        queue.sync_q.put(response)
                except Exception as e:
                    # 记录异常
                    logging.exception(f"Exception in sync_generator_wrapper: {e}")
                finally:
                    # 通知异步生成器数据生成完成
                    queue.sync_q.put(None)

            async def async_generator_wrapper():
                loop = asyncio.get_event_loop()
                # 在执行器中运行同步生成器
                loop.run_in_executor(None, sync_generator_wrapper)
                while True:
                    # 从异步队列中获取消息
                    response = await queue.async_q.get()
                    if response is None:  # 确保所有元素都被消费
                        break
                    # 生成响应
                    yield response
                # 设置停止事件
                stop_event.set()  # 通知同步生成器停止

            async for message in async_generator_wrapper():
                # 处理消息并生成 JSON 响应
                response_json = json.dumps(
                    _postprocess_agent_message(message.model_dump()),
                    ensure_ascii=False,
                )
                yield {"data": response_json}
                # 检查客户端是否已断开连接
                if await _request.is_disconnected():
                    break
        except Exception as exc:
            # 记录异常并生成错误响应
            msg = "An error occurred while generating the response."
            logging.exception(msg)
            response_json = json.dumps(
                dict(error=dict(msg=msg, details=str(exc))), ensure_ascii=False
            )
            yield {"data": response_json}
        finally:
            # 等待异步生成器停止
            await stop_event.wait()
            # 关闭队列
            queue.close()
            await queue.wait_closed()
            # 清理会话内存
            agent.agent.memory.memory_map.pop(session_id, None)

    # 获取输入数据和会话 ID
    inputs = request.inputs
    session_id = request.session_id
    # 初始化代理，llm及其参数，提示词模板；搜索插件；
    agent = init_agent(
        lang=args.lang,
        model_format=args.model_format,
        search_engine=args.search_engine,
    )
    # 返回 EventSourceResponse
    return EventSourceResponse(generate(), ping=300)


# 异步处理请求的函数（异步模式）
async def run_async(request: GenerationParams, _request: Request):
    async def generate():
        try:
            # 遍历代理生成的消息
            async for message in agent(inputs, session_id=session_id):
                # 处理消息并生成 JSON 响应
                response_json = json.dumps(
                    _postprocess_agent_message(message.model_dump()),
                    ensure_ascii=False,
                )
                yield {"data": response_json}
                # 检查客户端是否已断开连接
                if await _request.is_disconnected():
                    break
        except Exception as exc:
            # 记录异常并生成错误响应
            msg = "An error occurred while generating the response."
            logging.exception(msg)
            response_json = json.dumps(
                dict(error=dict(msg=msg, details=str(exc))), ensure_ascii=False
            )
            yield {"data": response_json}
        finally:
            # 清理会话内存
            agent.agent.memory.memory_map.pop(session_id, None)

    # 获取输入数据和会话 ID
    inputs = request.inputs
    session_id = request.session_id
    # 初始化代理（异步模式）
    agent = init_agent(
        lang=args.lang,
        model_format=args.model_format,
        search_engine=args.search_engine,
        use_async=True,
    )
    # 返回 EventSourceResponse
    return EventSourceResponse(generate(), ping=300)


# 添加 API 路由
app.add_api_route("/solve", run_async if args.asy else run, methods=["POST"])

# 主程序入口
if __name__ == "__main__":
    import uvicorn

    # 使用 uvicorn 运行 FastAPI 应用
    uvicorn.run(app, host=args.host, port=args.port, log_level="debug")

#具体流程
# 启动应用：
# 运行脚本时，if __name__ == "__main__": 判断为真，执行 uvicorn.run。
# uvicorn.run 启动 ASGI 服务器，监听指定的主机和端口。
# 接收请求：
# 客户端发送 HTTP POST 请求到 /solve 路由。
# FastAPI 接收到请求后，根据路由配置调用 run 或 run_async 函数。
# 处理请求：
# run 或 run_async 函数内部定义了一个异步生成器 generate。
# generate 生成器负责处理代理生成的消息，并将结果通过 EventSourceResponse 发送回客户端。
# 生成响应：
# generate 生成器中的 yield 语句会生成多个响应，每个响应都包含处理后的消息。
# EventSourceResponse 会将这些响应持续发送给客户端，直到处理完成或客户端断开连接。