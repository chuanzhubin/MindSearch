import os
from copy import deepcopy
from datetime import datetime

# 导入相关模块和类
from lagent.actions import AsyncWebBrowser, WebBrowser
from lagent.agents.stream import get_plugin_prompt
from lagent.prompts import InterpreterParser, PluginParser
from lagent.utils import create_object

# 导入自定义模块和类
from . import models as llm_factory
from .mindsearch_agent import AsyncMindSearchAgent, MindSearchAgent
from .mindsearch_prompt import (
    FINAL_RESPONSE_CN,
    FINAL_RESPONSE_EN,
    GRAPH_PROMPT_CN,
    GRAPH_PROMPT_EN,
    searcher_context_template_cn,
    searcher_context_template_en,
    searcher_input_template_cn,
    searcher_input_template_en,
    searcher_system_prompt_cn,
    searcher_system_prompt_en,
)

# 初始化一个全局字典来存储语言模型实例
LLM = {}

def init_agent(lang="cn",
               model_format="internlm_server",
               search_engine="BingSearch",
               use_async=False):
    """
    初始化代理（Agent）并返回实例。

    :param lang: 语言，支持 "cn" 和 "en"。
    :param model_format: 语言模型格式。
    :param search_engine: 搜索引擎，支持 "BingSearch" 和 "TencentSearch"。
    :param use_async: 是否使用异步模式。
    :return: 初始化的代理实例。
    """
    mode = "async" if use_async else "sync"  # 根据 use_async 参数确定模式
    llm = LLM.get(model_format, {}).get(mode)  # 从全局字典中获取已存在的语言模型实例
    if llm is None:
        # 如果没有找到已存在的语言模型实例，创建新的实例
        llm_cfg = deepcopy(getattr(llm_factory, model_format))  # 深拷贝模型配置
        if llm_cfg is None:
            raise NotImplementedError # 如果配置不存在，抛出异常
        if use_async:
            # 如果使用异步模式，修改模型类名为异步版本
            cls_name = (
                llm_cfg["type"].split(".")[-1] if isinstance(
                    llm_cfg["type"], str) else llm_cfg["type"].__name__)
            llm_cfg["type"] = f"lagent.llms.Async{cls_name}"
        llm = create_object(llm_cfg)  # 使用配置创建语言模型实例
        LLM.setdefault(model_format, {}).setdefault(mode, llm)  # 将新创建的实例存入全局字典

    # 获取当前日期
    date = datetime.now().strftime("The current date is %Y-%m-%d.")

    # 配置插件（WebBrowser 或 AsyncWebBrowser）
    plugins = [(dict(
            type=AsyncWebBrowser if use_async else WebBrowser,
            searcher_type=search_engine,
            topk=6,
            secret_id=os.getenv("TENCENT_SEARCH_SECRET_ID"),
            secret_key=os.getenv("TENCENT_SEARCH_SECRET_KEY"),
        ) if search_engine == "TencentSearch" else dict(
            type=AsyncWebBrowser if use_async else WebBrowser,
            searcher_type=search_engine,
            topk=6,
            api_key=os.getenv("WEB_SEARCH_API_KEY"),
        ))]

    # 创建代理实例
    agent = (AsyncMindSearchAgent if use_async else MindSearchAgent)(
        llm=llm,  # 语言模型实例
        template=date,  # 当前日期模板
        output_format=InterpreterParser(
            template=GRAPH_PROMPT_CN if lang == "cn" else GRAPH_PROMPT_EN  # 输出格式模板
        ),
        searcher_cfg=dict(
            llm=llm,  # 语言模型实例
            plugins=plugins,  # 插件配置
            template=date,  # 当前日期模板
            output_format=PluginParser(
                template=searcher_system_prompt_cn
                if lang == "cn" else searcher_system_prompt_en,  # 插件系统提示模板
                tool_info=get_plugin_prompt(plugins),  # 插件提示信息
            ),
            user_input_template=(searcher_input_template_cn if lang == "cn"
                                 else searcher_input_template_en),  # 用户输入模板
            user_context_template=(searcher_context_template_cn if lang == "cn"
                                   else searcher_context_template_en),  # 用户上下文模板
        ),
        summary_prompt=FINAL_RESPONSE_CN
        if lang == "cn" else FINAL_RESPONSE_EN,  # 最终响应模板
        max_turn=10,  # 最大轮数
    )
    return agent  # 返回初始化的代理实例
