"""
多智能体分析系统 — 对话式 Web UI
=================================
基于 LangGraph 的三阶段 Agent 工作流：
  1. 通义千问（分析）→ 2. DeepSeek（处理专家）→ 3. DeepSeek（纠错专家）

界面风格类似豆包 / DeepSeek 的对话式交互。

增强特性：
  - 对话式界面（聊天气泡）
  - 🕐 时间感知：AI 知晓当前时间
  - 🧠 记忆能力：自动存储 / 召回历史对话
  - 📁 文件支持：上传文件或指定目录批量读取
"""

import os
import gradio as gr
from agents import run_agent
from file_reader import read_files_from_paths
from memory import get_memory

# ============================================================
# CSS 样式（类似豆包 / DeepSeek 风格）
# ============================================================

CUSTOM_CSS = """
/* ---- 全局 ---- */
body, .gradio-container { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
.gradio-container { max-width: 100% !important; padding: 0 !important; }

/* ---- Header ---- */
#header { text-align: center; padding: 16px 24px 8px 24px; border-bottom: 1px solid #e5e7eb; background: white; }
#header h1 { font-size: 1.35rem; font-weight: 700; margin: 0; background: linear-gradient(135deg, #4f46e5, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
#header p { margin: 2px 0 0 0; font-size: 0.8rem; color: #9ca3af; }

/* ---- Chat 容器 ---- */
#chat-display { border: none !important; border-radius: 0 !important; background: #fafafa; padding: 8px 12px; }
#chat-display .message-wrap { max-width: 780px; margin: 0 auto; }

/* 用户气泡 */
#chat-display .user-message { background: #4f46e5 !important; color: #fff !important; border-radius: 18px 18px 4px 18px !important; padding: 10px 16px !important; max-width: 72% !important; margin: 4px 0 4px auto !important; box-shadow: 0 1px 2px rgba(0,0,0,0.08); }
#chat-display .user-message p { color: #fff !important; }

/* AI 气泡 */
#chat-display .assistant-message { background: #fff !important; color: #1f2937 !important; border-radius: 18px 18px 18px 4px !important; padding: 12px 18px !important; max-width: 82% !important; margin: 4px auto 4px 0 !important; box-shadow: 0 1px 3px rgba(0,0,0,0.06); border: 1px solid #f0f0f0; }
#chat-display .assistant-message p { color: #1f2937 !important; }

/* 气泡内代码块 */
#chat-display .assistant-message pre { background: #f6f8fa; border-radius: 8px; padding: 12px; overflow-x: auto; font-size: 0.85rem; border: 1px solid #e5e7eb; margin: 8px 0; }
#chat-display .assistant-message code { background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }
#chat-display .assistant-message pre code { background: transparent; padding: 0; }

/* ---- Input 区域 ---- */
#input-row { background: white; border-top: 1px solid #e5e7eb; padding: 12px 24px 16px 24px; }
#msg-input { border: 1px solid #d1d5db !important; border-radius: 24px !important; padding: 2px 4px 2px 16px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.04); transition: border-color 0.2s; background: #f9fafb; }
#msg-input:focus-within { border-color: #4f46e5 !important; box-shadow: 0 0 0 3px rgba(79,70,229,0.1); }
#msg-input textarea { border: none !important; background: transparent !important; padding: 8px 0 !important; font-size: 0.95rem !important; resize: none; }

/* ---- Settings ---- */
#settings-panel { border-top: 1px solid #e5e7eb; background: #f9fafb; padding: 8px 24px; font-size: 0.85rem; }
#settings-panel .label-text { color: #6b7280; font-size: 0.85rem; }
.status-badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 500; }
.status-ok { background: #dcfce7; color: #166534; }
.status-info { background: #dbeafe; color: #1e40af; }

/* ---- Buttons ---- */
.btn-primary { background: #4f46e5 !important; color: white !important; border-radius: 20px !important; border: none !important; padding: 6px 20px !important; font-weight: 500 !important; }
.btn-primary:hover { background: #4338ca !important; }
.btn-ghost { background: transparent !important; border: 1px solid #d1d5db !important; border-radius: 20px !important; padding: 6px 16px !important; font-size: 0.85rem !important; }
.btn-ghost:hover { background: #f3f4f6 !important; }

/* ---- Footer / watermark ---- */
footer { display: none !important; }

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }
"""

# ============================================================
# 主题
# ============================================================
THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

# ============================================================
# 结果格式化（用于对话气泡渲染）
# ============================================================

def _format_chat_result(state: dict) -> str:
    """
    将 Agent 执行结果格式化为对话友好的 Markdown
    """
    parts = []

    # --- 阶段 1：分析 ---
    if state.get("analysis"):
        analysis = state["analysis"]
        # 如果分析内容较长，只取前几行作为摘要
        lines = analysis.strip().split("\n")
        if len(lines) > 8:
            analysis = "\n".join(lines[:8]) + "\n\n> *... 完整分析已用于后续处理*"
        parts.append(f"## 📝 分析摘要\n\n{analysis}")
    elif state.get("analysis_error"):
        parts.append(f"## ❌ 分析阶段\n\n{state['analysis_error']}")

    # --- 阶段 2：处理专家 ---
    if state.get("expert_result"):
        parts.append(f"---\n\n## 🔧 处理意见\n\n{state['expert_result']}")
    elif state.get("expert_error"):
        err = state["expert_error"]
        if "跳过" not in err:
            parts.append(f"---\n\n## ⚠️ 处理专家\n\n{err}")

    # --- 阶段 3：最终答案 ---
    final = (
        state.get("final_answer")
        or state.get("correction_result")
        or state.get("expert_result")
    )
    if final:
        parts.append(f"---\n\n## ✅ 最终答案\n\n{final}")
    elif state.get("correction_error") and not state.get("correction_result"):
        parts.append(f"---\n\n## ⚠️ 纠错审查\n\n{state['correction_error']}")

    # 纠错阶段降级提示
    if state.get("correction_error") and state.get("correction_result"):
        parts.append(f"\n> ℹ️ *纠错审查备注：{state['correction_error']}*")

    return "\n\n".join(parts) if parts else "（无输出）"


def _format_file_summary(file_paths: list[str]) -> str:
    """将文件路径列表格式化为用户消息中的附件信息"""
    names = []
    for f in file_paths:
        if isinstance(f, str) and os.path.isfile(f):
            names.append(os.path.basename(f))
    if names:
        return f"\n\n📎 **附件：** {', '.join(names)}"
    return ""


# ============================================================
# 核心处理函数（Generator — 逐步更新 UI）
# ============================================================

def chat_respond(
    message: dict,
    history: list,
    dir_content: str,
    memory_toggle: bool,
):
    """
    处理用户发送的聊天消息

    Args:
        message:   MultimodalTextbox 的值 {"text": str, "files": list[str]}
        history:   Chatbot 当前消息列表
        dir_content: 从目录加载的文件内容（由 gr.State 维护）
        memory_toggle: 是否启用记忆
    """
    text = (message.get("text") or "").strip()
    files = message.get("files") or []

    if not text and not files:
        yield history, None
        return

    # ---- 1) 显示用户消息 ----
    user_content = text
    if files and isinstance(files, list):
        file_summary = _format_file_summary(files)
        user_content += file_summary

    history.append({"role": "user", "content": user_content})
    yield history, None

    # ---- 2) 显示思考占位 ----
    history.append({"role": "assistant", "content": "🤔 **正在分析...**"})
    yield history, None

    # ---- 3) 读取上传文件内容 ----
    files_content = ""
    if files and isinstance(files, list):
        valid = [f for f in files if isinstance(f, str) and os.path.isfile(f)]
        if valid:
            try:
                files_content = read_files_from_paths(valid)
            except Exception as e:
                files_content = f"[文件读取出错] {e}"

    # 合并目录文件内容
    all_file_content = ""
    if files_content:
        all_file_content += files_content
    if dir_content:
        if all_file_content:
            all_file_content += "\n" + dir_content
        else:
            all_file_content = dir_content

    # ---- 4) 执行 Agent 工作流 ----
    try:
        result = run_agent(
            conversation=text,
            files_content=all_file_content,
            use_memory=memory_toggle,
        )
        response = _format_chat_result(result)
    except Exception as e:
        response = f"❌ **系统错误**：{str(e)}"

    # ---- 5) 替换占位为最终回答 ----
    history[-1] = {"role": "assistant", "content": response}
    yield history, None


# ============================================================
# 辅助功能
# ============================================================

def load_directory(path: str) -> tuple[str, str]:
    """
    加载指定目录中的文本文件。

    Returns:
        (content, status_message)
    """
    if not path or not path.strip():
        return "", ""
    path = path.strip()
    if not os.path.isdir(path):
        raise gr.Error(f"目录不存在：{path}")
    try:
        content = read_files_from_paths([path])
        count = content.count("=====")
        status = f"📂 已加载 {count} 个文件" if count else "⚠️ 未找到文本文件"
        return content, status
    except Exception as e:
        raise gr.Error(f"读取目录失败：{e}")


def clear_memory_action() -> str:
    """清空记忆"""
    try:
        get_memory().delete_all()
        return "✅ 记忆已清空"
    except Exception as e:
        return f"❌ 清空失败：{e}"


def new_chat_action() -> list:
    """清空当前对话（不涉及记忆）"""
    return []


def update_memory_status(toggle: bool) -> str:
    """更新记忆状态显示"""
    if toggle:
        memory = get_memory()
        count = len(memory.conversations)
        return f"🧠 记忆已启用（{count} 条历史记录）"
    return "🧠 记忆已关闭"


# ============================================================
# 构建界面
# ============================================================

def build_ui():
    """搭建对话式 Web 界面"""
    with gr.Blocks(
        fill_height=True,
        title="🧠 多智能体分析系统",
    ) as demo:
        # ---- 状态变量 ----
        dir_content_state = gr.State("")       # 保存从目录加载的文件内容
        memory_toggle_state = gr.State(True)   # 记忆开关

        # ========== Header ==========
        with gr.Row(elem_id="header"):
            gr.Markdown(
                """
                # 🧠 多智能体分析系统
                <p>基于 LangGraph · 通义千问 + DeepSeek · 带记忆 · 支持文件</p>
                """
            )

        # ========== Chat 区域 ==========
        chatbot = gr.Chatbot(
            elem_id="chat-display",
            scale=1,
            min_width=0,
            render_markdown=True,
            show_label=False,
            height=None,
        )

        # ========== 输入区域 ==========
        with gr.Row(elem_id="input-row", equal_height=True):
            msg = gr.MultimodalTextbox(
                elem_id="msg-input",
                placeholder="输入你想分析的内容… 支持上传 .txt / .md / .py / .json 等文件",
                scale=1,
                file_count="multiple",
                file_types=[".txt", ".md", ".py", ".json", ".csv",
                            ".yaml", ".yml", ".xml", ".html", ".css",
                            ".js", ".ts", ".log", ".sh", ".bat", ".env",
                            ".toml", ".ini", ".cfg", ".rst", ".sql",
                            ".pdf"],
                container=True,
                max_lines=6,
            )

        # ========== 设置面板 ==========
        with gr.Row(elem_id="settings-panel"):
            with gr.Column(scale=2, min_width=200):
                # 记忆开关
                with gr.Row():
                    memory_checkbox = gr.Checkbox(
                        label="🧠 开启记忆",
                        value=True,
                        scale=1,
                        min_width=120,
                    )
                    memory_status = gr.Markdown(
                        value="🧠 记忆已启用（0 条历史记录）",
                    )

            with gr.Column(scale=3, min_width=300):
                # 目录加载
                with gr.Row():
                    dir_input = gr.Textbox(
                        label="📁 目录路径",
                        placeholder="输入目录路径批量加载文件…",
                        scale=3,
                        container=True,
                    )
                    load_dir_btn = gr.Button(
                        "📂 加载",
                        elem_classes="btn-ghost",
                        scale=1,
                        min_width=80,
                    )
                    dir_status = gr.Markdown(
                        value="",
                    )

            with gr.Column(scale=1, min_width=160):
                with gr.Row():
                    clear_memory_btn = gr.Button(
                        "🗑️ 清空记忆",
                        elem_classes="btn-ghost",
                        scale=1,
                    )
                    new_chat_btn = gr.Button(
                        "🔄 新对话",
                        elem_classes="btn-ghost",
                        scale=1,
                    )

        # ========== 事件绑定 ==========

        # 发送消息
        msg.submit(
            fn=chat_respond,
            inputs=[msg, chatbot, dir_content_state, memory_toggle_state],
            outputs=[chatbot, msg],
            queue=True,
            show_progress="full",
            concurrency_limit=1,
        )

        # 记忆开关
        memory_checkbox.change(
            fn=lambda v: (v, update_memory_status(v)),
            inputs=[memory_checkbox],
            outputs=[memory_toggle_state, memory_status],
        )

        # 启动时、切换时更新记忆状态
        demo.load(
            fn=lambda: update_memory_status(True),
            outputs=[memory_status],
        )

        # 目录加载（返回 content → dir_content_state, status → dir_status）
        load_dir_btn.click(
            fn=load_directory,
            inputs=[dir_input],
            outputs=[dir_content_state, dir_status],
        )

        # 清空记忆 → 更新状态显示
        clear_memory_btn.click(
            fn=clear_memory_action,
            outputs=[memory_status],
        ).then(
            fn=lambda: update_memory_status(memory_toggle_state.value),
            outputs=[memory_status],
        )

        # 新对话
        new_chat_btn.click(
            fn=new_chat_action,
            outputs=[chatbot],
        )

    return demo


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    demo = build_ui()
    print("=" * 60)
    print("  🧠 多智能体分析系统启动中...")
    print("  🔗 浏览器访问: http://127.0.0.1:7860")
    print("=" * 60)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=THEME,
        css=CUSTOM_CSS,
    )
