"""
LangGraph Agent 工作流
======================
一个三阶段顺序执行的多智能体系统：
  1. 分析阶段（通义千问）— 分析交谈内容，提取问题要点
  2. 处理专家阶段（DeepSeek）— 深度处理问题，给出解决方案
  3. 纠错专家阶段（DeepSeek）— 审查专家输出，纠错优化

增强特性：
  - 时间感知：注入当前时间，AI 知晓对话发生的时间上下文
  - 记忆能力：自动存储/加载历史对话，实现跨会话记忆
  - 文件参考：支持读取文件内容供 AI 分析参考

流程：START → analyze_node → expert_process_node → expert_correction_node → END
"""

from datetime import datetime
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from models import call_qwen, call_deepseek
from memory import get_memory


# ============================================================
# 星期映射（中文）
# ============================================================
WEEKDAY_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def _now_str() -> str:
    """返回形如 "2026-06-03 星期三 14:30:00" 的友好时间字符串"""
    now = datetime.now()
    return now.strftime(f"%Y-%m-%d {WEEKDAY_CN[now.weekday()]} %H:%M:%S")


# ============================================================
# 状态定义
# ============================================================

class AgentState(TypedDict):
    """图的状态类型"""
    conversation: str                     # 用户输入的交谈内容
    analysis: Optional[str]               # 通义千问的分析结果
    analysis_error: Optional[str]         # 分析阶段错误信息
    expert_result: Optional[str]          # DeepSeek 处理专家输出
    expert_error: Optional[str]           # 处理专家阶段错误信息
    correction_result: Optional[str]      # DeepSeek 纠错专家输出
    correction_error: Optional[str]       # 纠错专家阶段错误信息
    final_answer: Optional[str]           # 最终整合答案

    # ---- 增强字段 ----
    current_time: Optional[str]           # 当前时间（注入用）
    memory_context: Optional[str]         # 历史记忆上下文
    file_context: Optional[str]           # 参考文件内容


# ============================================================
# 系统提示词
# ============================================================

ANALYST_PROMPT = """你是一个专业的交谈分析专家。你的任务是：
1. 仔细分析用户提供的交谈内容
2. 提取交谈中的核心问题、关键信息与上下文
3. 识别交谈中的意图、情绪和潜在需求
4. 将分析结果结构化输出
5. 请结合当前时间信息，考虑时间相关的上下文因素

请给出清晰、有条理的分析结果。"""

EXPERT_PROMPT = """你是一个资深领域处理专家。你的任务是：
1. 基于分析结果，深入处理用户提出的问题
2. 提供专业、准确、可落地的解决方案或建议
3. 考虑多种可能性并给出最佳方案
4. 确保输出逻辑严密、论据充分、表达清晰
5. 注意结合当前时间上下文，给出时效性合适的建议

请输出你的专业处理结果。"""

CORRECTOR_PROMPT = """你是一个严谨的质量改善专家。你的任务是审查并优化处理专家的输出，确保最终答案质量最高。

审查要点：
1. 如果处理专家的输出引用了具体数据或文件内容，请与用户提供的参考文件进行核对
2. 纠正事实性错误（如有）
3. 补充遗漏的关键建议或视角
4. 优化表达，使其更清晰、更易执行
5. 合并重复或冗余的内容

重要原则：
- 不要无中生有地指责"幻觉"——处理专家的分析可能基于用户提供的参考文件，这些数据是真实的
- 如果处理专家的输出基本正确，直接做小幅润色即可，不必强行"找错误"
- 如果处理专家的输出完全没问题，直接确认并给出最终版本
- 保持客观、建设性的语气

请输出你优化后的最终答案。"""


# ============================================================
# 节点函数
# ============================================================

def _build_analysis_prompt(state: AgentState) -> tuple[str, str]:
    """
    根据状态组装带有时间 / 记忆 / 文件增强的系统提示词和用户消息
    Returns:
        (enhanced_system_prompt, user_message)
    """
    conversation = state.get("conversation", "")
    current_time = state.get("current_time") or _now_str()
    memory_context = state.get("memory_context", "")
    file_context = state.get("file_context", "")

    # 时间信息
    time_info = f"⏰ 当前时间：{current_time}"

    # 系统增强
    enhanced_system = ANALYST_PROMPT
    if memory_context:
        enhanced_system += f"\n\n{memory_context}"

    # 用户消息
    user_msg = f"{time_info}\n\n请分析以下交谈内容：\n\n{conversation}"
    if file_context:
        user_msg += f"\n\n{file_context}"

    return enhanced_system, user_msg


def analyze_node(state: AgentState) -> dict:
    """
    阶段 1：通义千问 — 分析交谈内容
    """
    conversation = state.get("conversation", "")
    if not conversation.strip():
        return {"analysis_error": "交谈内容为空，请提供有效的交谈内容"}

    try:
        system_prompt, user_message = _build_analysis_prompt(state)
        result = call_qwen(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.5,
        )
        return {"analysis": result, "analysis_error": None}
    except Exception as e:
        return {"analysis_error": f"分析阶段出错：{str(e)}"}


def expert_process_node(state: AgentState) -> dict:
    """
    阶段 2：DeepSeek — 处理专家
    基于分析结果进行专业处理
    """
    analysis = state.get("analysis", "")
    conversation = state.get("conversation", "")
    current_time = state.get("current_time") or _now_str()
    memory_context = state.get("memory_context", "")
    file_context = state.get("file_context", "")

    if state.get("analysis_error"):
        return {"expert_error": f"跳过：分析阶段已有错误 - {state['analysis_error']}"}

    if not analysis:
        return {"expert_error": "无分析结果可供处理"}

    try:
        # 组装增强的系统提示
        system_prompt = EXPERT_PROMPT
        if memory_context:
            system_prompt += f"\n\n{memory_context}"

        user_message_parts = [
            f"⏰ 当前时间：{current_time}",
            f"【原始交谈内容】\n{conversation}",
            f"【分析结果】\n{analysis}",
        ]
        if file_context:
            user_message_parts.append(f"【参考文件内容】\n{file_context}")
        user_message_parts.append(
            "请基于以上信息，作为处理专家给出专业处理结果。"
        )
        user_message = "\n\n".join(user_message_parts)
        result = call_deepseek(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.3,
        )
        return {"expert_result": result, "expert_error": None}
    except Exception as e:
        return {"expert_error": f"处理专家阶段出错：{str(e)}"}


def expert_correction_node(state: AgentState) -> dict:
    """
    阶段 3：DeepSeek — 纠错专家
    审查处理专家的输出并进行纠错优化
    """
    expert_result = state.get("expert_result", "")
    analysis = state.get("analysis", "")
    conversation = state.get("conversation", "")
    current_time = state.get("current_time") or _now_str()
    file_context = state.get("file_context", "")

    if state.get("expert_error"):
        return {"correction_error": f"跳过：处理专家阶段已有错误 - {state['expert_error']}"}

    if not expert_result:
        return {"correction_error": "无处理专家结果可供审查"}

    try:
        user_message_parts = [
            f"⏰ 当前时间：{current_time}",
            f"【原始交谈内容】\n{conversation}",
            f"【分析结果】\n{analysis}",
            f"【处理专家输出】\n{expert_result}",
        ]
        if file_context:
            user_message_parts.append(f"【参考文件内容（供核验）】\n{file_context}")
        user_message_parts.append(
            "请作为质量改善专家，审查以上处理专家的输出，给出优化后的最终答案。"
        )
        user_message = "\n\n".join(user_message_parts)
        result = call_deepseek(
            system_prompt=CORRECTOR_PROMPT,
            user_message=user_message,
            temperature=0.2,           # 纠错用更低的温度
        )
        return {
            "correction_result": result,
            "correction_error": None,
            "final_answer": result,     # 纠错后的结果作为最终答案
        }
    except Exception as e:
        # 即使纠错失败，也可以把专家结果作为最终答案
        return {
            "correction_error": f"纠错审查阶段出错：{str(e)}",
            "final_answer": expert_result,  # 降级使用专家结果
        }


# ============================================================
# 构建 LangGraph
# ============================================================

def build_agent_graph() -> StateGraph:
    """
    构建三阶段顺序执行的 Agent 图：
      START → analyze_node → expert_process_node → expert_correction_node → END
    """
    builder = StateGraph(AgentState)

    # 添加节点
    builder.add_node("analyze", analyze_node)
    builder.add_node("expert_process", expert_process_node)
    builder.add_node("expert_correction", expert_correction_node)

    # 边的逻辑：
    # START → analyze
    builder.add_edge(START, "analyze")
    # analyze → expert_process
    builder.add_edge("analyze", "expert_process")
    # expert_process → expert_correction
    builder.add_edge("expert_process", "expert_correction")
    # expert_correction → END
    builder.add_edge("expert_correction", END)

    return builder.compile()


# ============================================================
# 便捷调用函数
# ============================================================

def run_agent(
    conversation: str,
    files_content: str = "",
    use_memory: bool = True,
) -> AgentState:
    """
    运行整个 Agent 工作流

    Args:
        conversation: 用户输入的交谈内容
        files_content: 参考文件内容（由 file_reader 读取）
        use_memory:   是否启用历史记忆

    Returns:
        完整的执行结果状态
    """
    # ---- 准备增强上下文 ----
    current_time = _now_str()

    memory_context = ""
    if use_memory:
        memory = get_memory()
        memory_context = memory.get_recent_context()

    file_section = ""
    if files_content:
        file_section = f"\n\n附带的参考文件内容：\n{files_content}"

    # ---- 构建初始状态 ----
    graph = build_agent_graph()
    initial_state: AgentState = {
        "conversation": conversation,
        "analysis": None,
        "analysis_error": None,
        "expert_result": None,
        "expert_error": None,
        "correction_result": None,
        "correction_error": None,
        "final_answer": None,
        # 增强字段
        "current_time": current_time,
        "memory_context": memory_context,
        "file_context": file_section,
    }

    # ---- 执行 ----
    result = graph.invoke(initial_state)

    # ---- 保存到记忆 ----
    if use_memory:
        try:
            memory = get_memory()
            memory.add_conversation(
                conversation=conversation,
                analysis=result.get("analysis", ""),
                result=result.get("final_answer")
                        or result.get("correction_result")
                        or result.get("expert_result", ""),
            )
        except Exception:
            pass  # 记忆保存失败不应影响主流程

    return result
