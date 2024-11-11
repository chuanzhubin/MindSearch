import copy
from typing import List, Union

from lagent.agents import Agent, AgentForInternLM, AsyncAgent, AsyncAgentForInternLM
from lagent.schema import AgentMessage, AgentStatusCode, ModelStatusCode


class StreamingAgentMixin:
    """使代理调用输出为流式响应的混合类。"""

    def __call__(self, *message: Union[AgentMessage, List[AgentMessage]], session_id=0, **kwargs):
        # 遍历所有钩子，执行前置处理
        for hook in self._hooks.values():
            message = copy.deepcopy(message)
            result = hook.before_agent(self, message, session_id)
            if result:
                message = result

        # 更新内存中的消息
        self.update_memory(message, session_id=session_id)

        # 初始化响应消息
        response_message = AgentMessage(sender=self.name, content="")

        # 调用 forward 方法生成响应消息
        for response_message in self.forward(*message, session_id=session_id, **kwargs):
            if not isinstance(response_message, AgentMessage):
                model_state, response = response_message
                response_message = AgentMessage(
                    sender=self.name,
                    content=response,
                    stream_state=model_state,
                )
            yield response_message.model_copy()

        # 更新内存中的响应消息
        self.update_memory(response_message, session_id=session_id)

        # 遍历所有钩子，执行后置处理
        for hook in self._hooks.values():
            response_message = response_message.model_copy(deep=True)
            result = hook.after_agent(self, response_message, session_id)
            if result:
                response_message = result

        yield response_message


class AsyncStreamingAgentMixin:
    """使异步代理调用输出为流式响应的混合类。"""

    async def __call__(
        self, *message: Union[AgentMessage, List[AgentMessage]], session_id=0, **kwargs
    ):
        # 遍历所有钩子，执行前置处理
        for hook in self._hooks.values():
            message = copy.deepcopy(message)
            result = hook.before_agent(self, message, session_id)
            if result:
                message = result

        # 更新内存中的消息
        self.update_memory(message, session_id=session_id)

        # 初始化响应消息
        response_message = AgentMessage(sender=self.name, content="")

        # 异步调用 forward 方法生成响应消息
        async for response_message in self.forward(*message, session_id=session_id, **kwargs):
            if not isinstance(response_message, AgentMessage):
                model_state, response = response_message
                response_message = AgentMessage(
                    sender=self.name,
                    content=response,
                    stream_state=model_state,
                )
            yield response_message.model_copy()

        # 更新内存中的响应消息
        self.update_memory(response_message, session_id=session_id)

        # 遍历所有钩子，执行后置处理
        for hook in self._hooks.values():
            response_message = response_message.model_copy(deep=True)
            result = hook.after_agent(self, response_message, session_id)
            if result:
                response_message = result

        yield response_message


class StreamingAgent(StreamingAgentMixin, Agent):
    """基础流式代理类。"""

    def forward(self, *message: AgentMessage, session_id=0, **kwargs):
        # 格式化消息
        formatted_messages = self.aggregator.aggregate(
            self.memory.get(session_id),
            self.name,
            self.output_format,
            self.template,
        )

        # 调用 LLM 的流式聊天方法
        for model_state, response, _ in self.llm.stream_chat(
            formatted_messages, session_id=session_id, **kwargs
        ):
            yield AgentMessage(
                sender=self.name,
                content=response,
                formatted=self.output_format.parse_response(response),
                stream_state=model_state,
            ) if self.output_format else (model_state, response)


class AsyncStreamingAgent(AsyncStreamingAgentMixin, AsyncAgent):
    """基础异步流式代理类。"""

    async def forward(self, *message: AgentMessage, session_id=0, **kwargs):
        # 格式化消息
        formatted_messages = self.aggregator.aggregate(
            self.memory.get(session_id),
            self.name,
            self.output_format,
            self.template,
        )

        # 异步调用 LLM 的流式聊天方法
        async for model_state, response, _ in self.llm.stream_chat(
            formatted_messages, session_id=session_id, **kwargs
        ):
            yield AgentMessage(
                sender=self.name,
                content=response,
                formatted=self.output_format.parse_response(response),
                stream_state=model_state,
            ) if self.output_format else (model_state, response)


class StreamingAgentForInternLM(StreamingAgentMixin, AgentForInternLM):
    """`lagent.agents.AgentForInternLM` 的流式实现。"""

    _INTERNAL_AGENT_CLS = StreamingAgent

    def forward(self, message: AgentMessage, session_id=0, **kwargs):
        # 如果消息是字符串，转换为 AgentMessage 对象
        if isinstance(message, str):
            message = AgentMessage(sender="user", content=message)

        # 最大轮数循环
        for _ in range(self.max_turn):
            last_agent_state = AgentStatusCode.SESSION_READY

            # 调用 agent 方法生成响应消息
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

                yield message
                last_agent_state = message.stream_state

            # 检查是否满足结束条件
            if self.finish_condition(message):
                message.stream_state = AgentStatusCode.END
                yield message
                return

            # 如果有工具类型，执行相应的工具
            if message.formatted["tool_type"]:
                tool_type = message.formatted["tool_type"]
                executor = getattr(self, f"{tool_type}_executor", None)
                if not executor:
                    raise RuntimeError(f"No available {tool_type} executor")

                tool_return = executor(message, session_id=session_id)
                tool_return.stream_state = message.stream_state + 1
                message = tool_return
                yield message
            else:
                message.stream_state = AgentStatusCode.STREAM_ING
                yield message


class AsyncStreamingAgentForInternLM(AsyncStreamingAgentMixin, AsyncAgentForInternLM):
    """`lagent.agents.AsyncAgentForInternLM` 的流式实现。"""

    _INTERNAL_AGENT_CLS = AsyncStreamingAgent

    async def forward(self, message: AgentMessage, session_id=0, **kwargs):
        # 如果消息是字符串，转换为 AgentMessage 对象
        if isinstance(message, str):
            message = AgentMessage(sender="user", content=message)

        # 最大轮数循环
        for _ in range(self.max_turn):
            last_agent_state = AgentStatusCode.SESSION_READY

            # 异步调用 agent 方法生成响应消息
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

                yield message
                last_agent_state = message.stream_state

            # 检查是否满足结束条件
            if self.finish_condition(message):
                message.stream_state = AgentStatusCode.END
                yield message
                return

            # 如果有工具类型，执行相应的工具
            if message.formatted["tool_type"]:
                tool_type = message.formatted["tool_type"]
                executor = getattr(self, f"{tool_type}_executor", None)
                if not executor:
                    raise RuntimeError(f"No available {tool_type} executor")

                tool_return = await executor(message, session_id=session_id)
                tool_return.stream_state = message.stream_state + 1
                message = tool_return
                yield message
            else:
                message.stream_state = AgentStatusCode.STREAM_ING
                yield message
