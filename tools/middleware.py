from typing import Callable
from langchain.agents import AgentState
from langchain.agents.middleware import (
    wrap_tool_call, before_model
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """
    工具调用监控中间件

    功能：
    1. 记录每次工具调用的名称和参数（日志）
    2. 检测到 fill_context_for_report 被调用时，置 context.report = True
       （这会触发 report_prompt_switch 在后续模型调用时切换提示词）
    3. 捕获并记录工具调用中的异常

    :param request: 当前工具调用的请求对象，包含 tool_call 信息和 runtime 上下文
    :param handler: 实际执行工具的下一个处理函数（调用链中的下一环）
    :return: 工具执行的结果消息
    """
    logger.info(f'[tool monitor]执行工具：{request.tool_call["name"]}')
    logger.info(f'[tool monitor]传入参数：{request.tool_call["args"]}')

    try:
        # 执行实际的工具函数
        result = handler(request)
        logger.info(f'[tool monitor]工具{request.tool_call["name"]}调用成功')

        return result

    except Exception as e:
        logger.error(f'工具{request.tool_call["name"]}调用失败，原因：{str(e)}')
        raise e


@before_model
def log_before_model(
    state: AgentState,
    runtime: Runtime
):
    """
    模型调用前日志中间件

    在每次调用大模型之前记录当前对话状态信息，
    用于调试和追踪 Agent 的思考过程。

    :param state: 当前 Agent 状态，包含 messages（消息历史）
    :param runtime: Agent 运行时的上下文环境
    :return: None（不修改请求）
    """
    logger.info(
        f'[log_before_model]即将调用模型，带有{len(state["messages"])}条消息'
    )
    logger.debug(
        f'[log_before_model]{type(state["messages"][-1]).__name__}'
        f'|{state["messages"][-1].content.strip()}'
    )

    return None