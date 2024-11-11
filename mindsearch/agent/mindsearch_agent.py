import json
import logging
import re
from copy import deepcopy
from typing import Dict, Tuple

from lagent.schema import AgentMessage, AgentStatusCode, ModelStatusCode
from lagent.utils import GeneratorWithReturn

from .graph import ExecutionAction, WebSearchGraph
from .streaming import AsyncStreamingAgentForInternLM, StreamingAgentForInternLM


def _update_ref(ref: str, ref2url: Dict[str, str], ptr: int) -> str:
    """
    更新引用编号，并生成新的引用到 URL 的映射。

    :param ref: 原始引用字符串。
    :param ref2url: 原始引用到 URL 的映射。
    :param ptr: 当前引用编号偏移量。
    :return: 更新后的引用字符串、新的引用到 URL 的映射、新增的引用数量。
    """
    # 提取所有引用编号
    numbers = list({int(n) for n in re.findall(r"\[\[(\d+)\]\]", ref)})
    # 重新编号
    numbers = {n: idx + 1 for idx, n in enumerate(numbers)}
    # 替换引用编号
    updated_ref = re.sub(
        r"\[\[(\d+)\]\]",
        lambda match: f"[[{numbers[int(match.group(1))] + ptr}]]",
        ref,
    )
    # 生成新的引用到 URL 的映射
    updated_ref2url = {}
    if numbers:
        try:
            assert all(elem in ref2url for elem in numbers)
        except Exception as exc:
            logging.info(f"Illegal reference id: {str(exc)}")
        if ref2url:
            updated_ref2url = {
                numbers[idx] + ptr: ref2url[idx] for idx in numbers if idx in ref2url
            }
    return updated_ref, updated_ref2url, len(numbers) + 1


def _generate_references_from_graph(graph: Dict[str, dict]) -> Tuple[str, Dict[int, dict]]:
    """
    从图结构中生成引用文本和引用到 URL 的映射。

    :param graph: 图结构数据。
    :return: 引用文本、引用到 URL 的映射。
    """
    ptr, references, references_url = 0, [], {}
    for name, data_item in graph.items():
        if name in ["root", "response"]:
            continue
        # 只在每个节点搜索一次，结果偏移量为 2
        assert data_item["memory"]["agent.memory"][2]["sender"].endswith("ActionExecutor")
        ref2url = {
            int(k): v
            for k, v in json.loads(data_item["memory"]["agent.memory"][2]["content"]).items()
        }
        updata_ref, ref2url, added_ptr = _update_ref(
            data_item["response"]["content"], ref2url, ptr
        )
        ptr += added_ptr
        references.append(f'## {data_item["content"]}\n\n{updata_ref}')
        references_url.update(ref2url)
    return "\n\n".join(references), references_url


class MindSearchAgent(StreamingAgentForInternLM):
    def __init__(
        self,
        searcher_cfg: dict,
        summary_prompt: str,
        finish_condition=lambda m: "add_response_node" in m.content,
        max_turn: int = 10,
        **kwargs,
    ):
        """
        初始化 MindSearchAgent 实例。

        :param searcher_cfg: 搜索器配置。
        :param summary_prompt: 总结提示。
        :param finish_condition: 结束条件函数。
        :param max_turn: 最大轮数。
        :param kwargs: 其他关键字参数。
        """
        WebSearchGraph.SEARCHER_CONFIG = searcher_cfg  # 设置搜索器配置
        super().__init__(finish_condition=finish_condition, max_turn=max_turn, **kwargs)
        self.summary_prompt = summary_prompt  # 总结提示
        self.action = ExecutionAction()  # 执行动作

    def forward(self, message: AgentMessage, session_id=0, **kwargs):
        """
        处理消息并生成响应。

        :param message: 输入消息。
        :param session_id: 会话 ID。
        :param kwargs: 其他关键字参数。
        :yield: 处理过程中的消息。
        """
        if isinstance(message, str):
            message = AgentMessage(sender="user", content=message)  # 将字符串转换为 AgentMessage
        _graph_state = dict(node={}, adjacency_list={}, ref2url={})  # 初始化图状态
        local_dict, global_dict = {}, globals()  # 初始化局部和全局字典
        for _ in range(self.max_turn):
            last_agent_state = AgentStatusCode.SESSION_READY  # 初始化上一个代理状态
            for message in self.agent(message, session_id=session_id, **kwargs):
                if isinstance(message.formatted, dict) and message.formatted.get("tool_type"):
                    if message.stream_state == ModelStatusCode.END:
                        message.stream_state = last_agent_state + int(
                            last_agent_state
                            in [
                                AgentStatusCode.CODING,
                                AgentStatusCode.PLUGIN_START,
                            ]
                        )
                    else:
                        message.stream_state = (
                            AgentStatusCode.PLUGIN_START
                            if message.formatted["tool_type"] == "plugin"
                            else AgentStatusCode.CODING
                        )
                else:
                    message.stream_state = AgentStatusCode.STREAM_ING
                message.formatted.update(deepcopy(_graph_state))  # 更新消息的图状态
                yield message  # 生成消息
                last_agent_state = message.stream_state  # 更新上一个代理状态
            if not message.formatted["tool_type"]:
                message.stream_state = AgentStatusCode.END  # 结束状态
                yield message  # 生成消息
                return

            gen = GeneratorWithReturn(
                self.action.run(message.content, local_dict, global_dict, True)
            )  # 执行动作
            for graph_exec in gen:
                graph_exec.formatted["ref2url"] = deepcopy(_graph_state["ref2url"])  # 更新图执行的引用到 URL 映射
                yield graph_exec  # 生成图执行的消息

            reference, references_url = _generate_references_from_graph(gen.ret[1])  # 生成引用
            _graph_state.update(node=gen.ret[1], adjacency_list=gen.ret[2], ref2url=references_url)  # 更新图状态
            if self.finish_condition(message):
                message = AgentMessage(
                    sender="ActionExecutor",
                    content=self.summary_prompt,
                    formatted=deepcopy(_graph_state),
                    stream_state=message.stream_state + 1,  # 插件或代码返回
                )
                yield message  # 生成总结消息
                # 生成最终答案
                for message in self.agent(message, session_id=session_id, **kwargs):
                    message.formatted.update(deepcopy(_graph_state))  # 更新消息的图状态
                    yield message  # 生成消息
                return
            message = AgentMessage(
                sender="ActionExecutor",
                content=reference,
                formatted=deepcopy(_graph_state),
                stream_state=message.stream_state + 1,  # 插件或代码返回
            )
            yield message  # 生成引用消息


class AsyncMindSearchAgent(AsyncStreamingAgentForInternLM):
    def __init__(
        self,
        searcher_cfg: dict,
        summary_prompt: str,
        finish_condition=lambda m: "add_response_node" in m.content,
        max_turn: int = 10,
        **kwargs,
    ):
        """
        初始化 AsyncMindSearchAgent 实例。

        :param searcher_cfg: 搜索器配置。
        :param summary_prompt: 总结提示。
        :param finish_condition: 结束条件函数。
        :param max_turn: 最大轮数。
        :param kwargs: 其他关键字参数。
        """
        WebSearchGraph.SEARCHER_CONFIG = searcher_cfg  # 设置搜索器配置
        WebSearchGraph.is_async = True  # 设置为异步模式
        WebSearchGraph.start_loop()  # 启动事件循环
        super().__init__(finish_condition=finish_condition, max_turn=max_turn, **kwargs)
        self.summary_prompt = summary_prompt  # 总结提示
        self.action = ExecutionAction()  # 执行动作

    async def forward(self, message: AgentMessage, session_id=0, **kwargs):
        """
        异步处理消息并生成响应。

        :param message: 输入消息。
        :param session_id: 会话 ID。
        :param kwargs: 其他关键字参数。
        :yield: 处理过程中的消息。
        """
        if isinstance(message, str):
            message = AgentMessage(sender="user", content=message)  # 将字符串转换为 AgentMessage
        _graph_state = dict(node={}, adjacency_list={}, ref2url={})  # 初始化图状态
        local_dict, global_dict = {}, globals()  # 初始化局部和全局字典
        for _ in range(self.max_turn):
            last_agent_state = AgentStatusCode.SESSION_READY  # 初始化上一个代理状态
            async for message in self.agent(message, session_id=session_id, **kwargs):
                if isinstance(message.formatted, dict) and message.formatted.get("tool_type"):
                    if message.stream_state == ModelStatusCode.END:
                        message.stream_state = last_agent_state + int(
                            last_agent_state
                            in [
                                AgentStatusCode.CODING,
                                AgentStatusCode.PLUGIN_START,
                            ]
                        )
                    else:
                        message.stream_state = (
                            AgentStatusCode.PLUGIN_START
                            if message.formatted["tool_type"] == "plugin"
                            else AgentStatusCode.CODING
                        )
                else:
                    message.stream_state = AgentStatusCode.STREAM_ING
                message.formatted.update(deepcopy(_graph_state))  # 更新消息的图状态
                yield message  # 生成消息
                last_agent_state = message.stream_state  # 更新上一个代理状态
            if not message.formatted["tool_type"]:
                message.stream_state = AgentStatusCode.END  # 结束状态
                yield message  # 生成消息
                return

            gen = GeneratorWithReturn(
                self.action.run(message.content, local_dict, global_dict, True)
            )  # 执行动作
            for graph_exec in gen:
                graph_exec.formatted["ref2url"] = deepcopy(_graph_state["ref2url"])  # 更新图执行的引用到 URL 映射
                yield graph_exec  # 生成图执行的消息

            reference, references_url = _generate_references_from_graph(gen.ret[1])  # 生成引用
            _graph_state.update(node=gen.ret[1], adjacency_list=gen.ret[2], ref2url=references_url)  # 更新图状态
            if self.finish_condition(message):
                message = AgentMessage(
                    sender="ActionExecutor",
                    content=self.summary_prompt,
                    formatted=deepcopy(_graph_state),
                    stream_state=message.stream_state + 1,  # 插件或代码返回
                )
                yield message  # 生成总结消息
                # 生成最终答案
                async for message in self.agent(message, session_id=session_id, **kwargs):
                    message.formatted.update(deepcopy(_graph_state))  # 更新消息的图状态
                    yield message  # 生成消息
                return
            message = AgentMessage(
                sender="ActionExecutor",
                content=reference,
                formatted=deepcopy(_graph_state),
                stream_state=message.stream_state + 1,  # 插件或代码返回
            )
            yield message  # 生成引用消息
