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
import time as time_mod
import base64
import gradio as gr
from agents import run_agent
from file_reader import read_files_from_paths
from memory import get_memory

# ============================================================
# 图片资源 —— base64 内嵌，无需依赖文件路径
# ============================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(ROOT_DIR, "img")

def _img_data_uri(filename: str) -> str:
    """读取 PNG 并返回 data: URI"""
    path = os.path.join(IMG_DIR, filename)
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{b64}"
    except FileNotFoundError:
        return ""

# 预加载所有图标
_MEMO_URI   = _img_data_uri("记忆.png")
_NET_URI    = _img_data_uri("网络.png")
_FOLDER_URI = _img_data_uri("文件夹.png")
_TRASH_URI  = _img_data_uri("垃圾桶.png")
_REFRESH_URI = _img_data_uri("刷新.png")
_LOGO_URI   = _img_data_uri("logo.png")

def _img(uri: str, size: int = 18) -> str:
    """生成 <img> 标签，等比例严格限制尺寸"""
    return f'<img src="{uri}" width="{size}" height="{size}" style="vertical-align:middle;margin-right:4px;object-fit:contain">'

# ============================================================
# CSS 样式
# ============================================================

CUSTOM_CSS = ("""
/* ===============================================================
   波普风格 (Pop Art)
   蓝 #5271FF · 橙 #FF7D54 · 黄 #FFE500
   =============================================================== */

html, body {
    height: 100% !important;
    margin: 0 !important;
    overflow: hidden !important;
}
body {
    font-family: 'Helvetica Neue', Arial, 'Noto Sans SC', sans-serif !important;
    background: #FFE500 !important;
    background-image:
        radial-gradient(circle, rgba(82, 113, 255, 0.15) 2px, transparent 2px),
        radial-gradient(circle, rgba(255, 125, 84, 0.10) 1px, transparent 1px) !important;
    background-size: 28px 28px, 14px 14px !important;
    background-position: 0 0, 14px 14px !important;
}

/* ---- Blocks 容器：全容器滚动 ---- */
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    background: transparent !important;
    border-radius: 0 !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow-y: auto !important;          /* 整个容器滚动，不做 flex 限制 */
    display: block !important;            /* 改成 block 让自然 flow 驱动 */
}

/* ============================================================
   Header —— 波普蓝底 + 粗体白字 + 漫画风格
   ============================================================ */
#header {
    text-align: center;
    padding: 10px 24px 8px 24px;
    border-bottom: 4px solid #FF7D54;
    background: #5271FF !important;
    flex-shrink: 0 !important;
    position: sticky !important;          /* sticky 固定在顶部 */
    top: 0 !important;
    z-index: 100 !important;
    box-shadow: 0 4px 0 #FFE500;
}
#header::after {
    content: '';
    position: absolute;
    bottom: -8px;
    left: 0;
    right: 0;
    height: 4px;
    background: repeating-linear-gradient(
        90deg,
        #FF7D54 0px, #FF7D54 12px,
        #FFE500 12px, #FFE500 16px
    );
}
#header h1 {
    font-size: 1.35rem;
    font-weight: 900;
    margin: 0;
    color: #FFFFFF !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    -webkit-text-fill-color: #FFFFFF !important;
    text-shadow: 3px 3px 0 rgba(0, 0, 0, 0.15);
}
#header p {
    margin: 2px 0 0 0;
    font-size: 0.78rem;
    color: rgba(255, 255, 255, 0.85) !important;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* ============================================================
   PNG 图标 — CSS ::before 替代 Checkbox 标签 Emoji
   ============================================================ */
#memory-checkbox label,
#web-checkbox label {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
}
#memory-checkbox label::before {
    content: '' !important;
    display: inline-block !important;
    width: 17px !important;
    height: 17px !important;
    background: url('__MEMO__') no-repeat center !important;
    background-size: contain !important;
    flex-shrink: 0 !important;
}
#web-checkbox label::before {
    content: '' !important;
    display: inline-block !important;
    width: 17px !important;
    height: 17px !important;
    background: url('__NET__') no-repeat center !important;
    background-size: contain !important;
    flex-shrink: 0 !important;
}
#dir-input label::before {
    content: '' !important;
    display: inline-block !important;
    width: 17px !important;
    height: 17px !important;
    background: url('__FOLDER__') no-repeat center !important;
    background-size: contain !important;
    margin-right: 6px !important;
    vertical-align: middle !important;
}
.btn-with-icon {
    display: inline-flex !important;
    align-items: center !important;
    gap: 5px !important;
}
.btn-with-icon::before {
    content: '' !important;
    display: inline-block !important;
    width: 16px !important;
    height: 16px !important;
    background-size: contain !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    flex-shrink: 0 !important;
}
.btn-folder::before  { background-image: url('__FOLDER__') !important; }
.btn-trash::before   { background-image: url('__TRASH__') !important; }
.btn-refresh::before { background-image: url('__REFRESH__') !important; }

/* ============================================================
   Chat 容器
   ============================================================ */
#chat-display {
    border: none !important;
    border-radius: 0 !important;
    background: transparent !important;
    padding: 12px 16px;
    min-height: 200px !important;
    overflow: visible !important;          /* 不自带滚动，让父容器统一滚动 */
}
/* 内部所有层自然流动 */
#chat-display .wrap,
#chat-display .gr-box,
#chat-display > div {
    overflow: visible !important;
    height: auto !important;
    max-height: none !important;
}
/* 气泡容器 */
#chat-display .message-wrap {
    max-width: 780px;
    margin: 0 auto;
}

/* ---- 用户气泡：波普橙色 ---- */
#chat-display .user-message {
    background: #FF7D54 !important;
    color: #fff !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 12px 20px !important;
    max-width: 72% !important;
    margin: 8px 0 8px auto !important;
    box-shadow: 4px 4px 0 rgba(0, 0, 0, 0.15) !important;
    border: 2px solid #000 !important;
    font-weight: 500 !important;
    position: relative;
}
#chat-display .user-message p { color: #fff !important; }

/* ---- AI 气泡：白底 + 蓝边 + 粗框漫画风 ---- */
#chat-display .assistant-message {
    background: #FFFFFF !important;
    color: #1a1a2e !important;
    border-radius: 16px 16px 16px 4px !important;
    padding: 16px 22px !important;
    max-width: 82% !important;
    margin: 8px auto 8px 0 !important;
    box-shadow: 4px 4px 0 #5271FF !important;
    border: 2px solid #000 !important;
    position: relative;
}
#chat-display .assistant-message::before {
    content: '✦';
    position: absolute;
    top: -10px;
    left: -6px;
    font-size: 14px;
    color: #FFE500;
    text-shadow: 1px 1px 0 #000;
}

/* ---- 气泡内代码块 ---- */
#chat-display .assistant-message pre {
    background: #f0f0f0 !important;
    border-radius: 8px;
    padding: 14px;
    overflow-x: auto;
    font-size: 0.83rem;
    border: 2px solid #000 !important;
    margin: 10px 0;
}
#chat-display .assistant-message code {
    background: #f0f0f0;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.85em;
    color: #5271FF;
}
#chat-display .assistant-message pre code {
    background: transparent;
    padding: 0;
    color: inherit;
}

/* ---- 气泡内表格 ---- */
#chat-display .assistant-message table {
    border-collapse: collapse;
    margin: 8px 0;
    font-size: 0.88rem;
    width: 100%;
    border: 2px solid #000;
}
#chat-display .assistant-message th,
#chat-display .assistant-message td {
    border: 1px solid #000;
    padding: 6px 12px;
    text-align: left;
}
#chat-display .assistant-message th {
    background: #5271FF !important;
    color: #fff !important;
    font-weight: 700;
}

/* ============================================================
   思考占位 —— 波普动画计时指示器
   ============================================================ */
.thinking-indicator {
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-width: 200px;
}
.thinking-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.05rem;
    font-weight: 600;
}
@keyframes pop-bounce {
    0%, 80%, 100% { transform: translateY(0); }
    40% { transform: translateY(-8px); }
}
.thinking-dots {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    margin-left: 4px;
}
.thinking-dots .dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    border: 1.5px solid #000;
    animation: pop-bounce 1.2s ease-in-out infinite;
}
.thinking-dots .dot:nth-child(1) { background: #5271FF; animation-delay: -0.32s; }
.thinking-dots .dot:nth-child(2) { background: #FF7D54; animation-delay: -0.16s; }
.thinking-dots .dot:nth-child(3) { background: #FFE500; animation-delay: 0s; }
.thinking-timer {
    display: inline-block;
    font-variant-numeric: tabular-nums;
    font-size: 0.85rem;
    color: #5271FF;
    font-weight: 700;
    min-width: 3em;
    text-align: center;
}
@keyframes wave-slide {
    0% { background-position: 0 0; }
    100% { background-position: 200px 0; }
}
.thinking-progress {
    height: 4px;
    border-radius: 2px;
    background: repeating-linear-gradient(
        90deg,
        #5271FF 0px, #5271FF 20px,
        #FF7D54 20px, #FF7D54 40px,
        #FFE500 40px, #FFE500 60px
    );
    background-size: 60px 100%;
    animation: wave-slide 0.8s linear infinite;
    border: 1px solid #000;
    overflow: hidden;
}
.response-timer {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.75rem;
    color: #888;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px dashed #ddd;
}

/* ============================================================
   底部固定区域 —— sticky 固定在底部
   ============================================================ */
#bottom-area {
    flex-shrink: 0 !important;
    position: sticky !important;          /* sticky 固定在底部 */
    bottom: 0 !important;
    z-index: 100 !important;
    background: #FFFFFF !important;
    border-top: 4px solid #5271FF !important;
    box-shadow: 0 -2px 0 #FFE500;
}

/* ---- Input 区域 ---- */
#input-row {
    padding: 8px 20px 4px 20px;
    background: transparent !important;
}
#msg-input {
    border: 2px solid #000 !important;
    border-radius: 24px !important;
    padding: 2px 4px 2px 18px !important;
    box-shadow: 4px 4px 0 rgba(0, 0, 0, 0.12) !important;
    transition: box-shadow 0.15s, transform 0.15s;
    background: #fff !important;
}
#msg-input:focus-within {
    border-color: #5271FF !important;
    box-shadow: 4px 4px 0 #5271FF !important;
    transform: translate(-1px, -1px);
}
#msg-input textarea {
    border: none !important;
    background: transparent !important;
    padding: 8px 0 !important;
    font-size: 0.95rem !important;
    resize: none;
    line-height: 1.4 !important;
}
#msg-input textarea::placeholder {
    color: #bbb !important;
    font-style: italic;
}

/* ---- Settings 面板 ---- */
#settings-panel {
    padding: 4px 20px 10px 20px;
    font-size: 0.82rem;
    background: transparent !important;
    border: none !important;
}
#settings-panel label,
#settings-panel .label-text {
    color: #1a1a2e !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}

/* ---- Buttons ---- */
.btn-primary {
    background: #5271FF !important;
    color: white !important;
    border-radius: 20px !important;
    border: 2px solid #000 !important;
    padding: 6px 20px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    transition: transform 0.1s, box-shadow 0.1s !important;
    box-shadow: 3px 3px 0 rgba(0, 0, 0, 0.15) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.btn-primary:hover {
    transform: translate(-1px, -1px) !important;
    box-shadow: 4px 4px 0 rgba(0, 0, 0, 0.2) !important;
}
.btn-ghost {
    background: #FFFFFF !important;
    border: 2px solid #000 !important;
    border-radius: 20px !important;
    padding: 4px 16px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    transition: background 0.1s, transform 0.1s !important;
    box-shadow: 2px 2px 0 rgba(0, 0, 0, 0.1) !important;
}
.btn-ghost:hover {
    background: #FFE500 !important;
    transform: translate(-1px, -1px) !important;
    box-shadow: 3px 3px 0 rgba(0, 0, 0, 0.15) !important;
}
/* ---- 发送按钮 ---- */
#msg-input button {
    border-radius: 50% !important;
    min-width: 38px !important;
    min-height: 38px !important;
    background: #5271FF !important;
    color: white !important;
    border: 2px solid #000 !important;
    transition: transform 0.1s, box-shadow 0.1s !important;
    box-shadow: 2px 2px 0 rgba(0, 0, 0, 0.15) !important;
    font-weight: bold !important;
}
#msg-input button:hover {
    transform: scale(1.08) !important;
    background: #FF7D54 !important;
    box-shadow: 3px 3px 0 rgba(0, 0, 0, 0.2) !important;
}

/* ---- Checkbox ---- */
#settings-panel input[type="checkbox"] {
    accent-color: #5271FF !important;
    width: 16px !important;
    height: 16px !important;
    cursor: pointer !important;
    outline: 1px solid #000 !important;
}

/* ---- Footer / watermark ---- */
footer { display: none !important; }

/* ---- 自定义滚动条 ---- */
.gradio-container::-webkit-scrollbar { width: 8px; }
.gradio-container::-webkit-scrollbar-track { background: transparent; }
.gradio-container::-webkit-scrollbar-thumb {
    background: #5271FF !important;
    border-radius: 4px;
    border: 2px solid transparent;
    background-clip: content-box;
}
.gradio-container::-webkit-scrollbar-thumb:hover {
    background: #FF7D54 !important;
    background-clip: content-box;
}

/* ---- 状态标签 ---- */
.status-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    border: 1px solid #000;
}
.status-ok { background: #FFE500; color: #1a1a2e; }
.status-info { background: #5271FF; color: #fff; }

/* ---- 响应式 ---- */
@media (max-width: 640px) {
    #settings-panel .gr-form { flex-wrap: wrap !important; }
    #header h1 { font-size: 1.1rem; }
    #chat-display .user-message,
    #chat-display .assistant-message {
        max-width: 90% !important;
    }
}
""").replace('__MEMO__', _MEMO_URI).replace('__NET__', _NET_URI).replace('__FOLDER__', _FOLDER_URI).replace('__TRASH__', _TRASH_URI).replace('__REFRESH__', _REFRESH_URI)

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
    web_toggle: bool,
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

    # ---- 2) 显示思考占位（带可视化计时器和动画） ----
    thinking_html = (
        '<div class="thinking-indicator">'
        '<div class="thinking-header">'
        '🤔 正在分析'
        '<span class="thinking-dots">'
        '<span class="dot"></span>'
        '<span class="dot"></span>'
        '<span class="dot"></span>'
        '</span>'
        '<span class="thinking-timer" id="thinking-timer">⏱ 0.0s</span>'
        '</div>'
        '<div class="thinking-progress"></div>'
        '</div>'
    )
    history.append({"role": "assistant", "content": thinking_html})
    yield history, None

    _start = time_mod.time()

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
            use_web_search=web_toggle,
        )
        response = _format_chat_result(result)
    except Exception as e:
        response = f"❌ **系统错误**：{str(e)}"

    # ---- 5) 追加处理耗时 ----
    elapsed = time_mod.time() - _start
    response += f"\n\n<div class=\"response-timer\">⏱️ 处理耗时 {elapsed:.1f} 秒</div>"

    # ---- 6) 替换占位为最终回答 ----
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
        status = f"{_img(_FOLDER_URI, 16)} 已加载 {count} 个文件" if count else "未找到文本文件"
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
        return f"{_img(_MEMO_URI, 17)} 记忆已启用（{count} 条历史记录）"
    return f"{_img(_MEMO_URI, 17)} 记忆已关闭"


# ============================================================
# 构建界面
# ============================================================

def build_ui():
    """搭建对话式 Web 界面"""
    with gr.Blocks(
        fill_height=True,
        title="多智能体分析系统",
        head="""
<script>
// 波普风格思考计时器 —— 定时轮询，自动开始/停止
(function() {
    var startTime = null;
    setInterval(function() {
        var el = document.getElementById('thinking-timer');
        if (el) {
            if (!startTime) startTime = Date.now();
            var s = (Date.now() - startTime) / 1000;
            el.textContent = '⏱ ' + s.toFixed(1) + 's';
        } else {
            startTime = null;
        }
    }, 100);
})();
</script>
""",
    ) as demo:
        # ---- 状态变量 ----
        dir_content_state = gr.State("")       # 保存从目录加载的文件内容
        memory_toggle_state = gr.State(True)   # 记忆开关
        web_toggle_state = gr.State(True)      # 联网搜索开关

        # ========== Header ==========
        with gr.Row(elem_id="header"):
            gr.Markdown(
                f"""
                # {_img(_LOGO_URI, 28)} 多智能体分析系统
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
            height=None,              # 必须为 None，否则内部固定 400px 无法撑满
        )

        # ========== 底部固定区域 ==========
        with gr.Column(elem_id="bottom-area"):
            # ========== 输入区域 ==========
            with gr.Row(elem_id="input-row", equal_height=True):
                msg = gr.MultimodalTextbox(
                    elem_id="msg-input",
                    placeholder="输入你想分析的内容… 支持上传 PDF .txt / .md / .py / .json 等文件",
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
                    # 记忆开关 + 联网搜索
                    with gr.Row():
                        memory_checkbox = gr.Checkbox(
                            label="开启记忆",
                            elem_id="memory-checkbox",
                            value=True,
                            scale=1,
                            min_width=100,
                        )
                        web_checkbox = gr.Checkbox(
                            label="联网搜索",
                            elem_id="web-checkbox",
                            value=True,
                            scale=1,
                            min_width=100,
                        )
                        memory_status = gr.Markdown(
                            value=f"{_img(_MEMO_URI, 17)} 记忆已启用",
                        )

                with gr.Column(scale=3, min_width=300):
                    # 目录加载
                    with gr.Row():
                        dir_input = gr.Textbox(
                            label="目录路径",
                            elem_id="dir-input",
                            placeholder="输入目录路径批量加载文件…",
                            scale=3,
                            container=True,
                        )
                        load_dir_btn = gr.Button(
                            "加载",
                            elem_classes=["btn-ghost", "btn-with-icon", "btn-folder"],
                            scale=1,
                            min_width=80,
                        )
                        dir_status = gr.Markdown(
                            value="",
                        )

                with gr.Column(scale=1, min_width=160):
                    with gr.Row():
                        clear_memory_btn = gr.Button(
                            "清空记忆",
                            elem_classes=["btn-ghost", "btn-with-icon", "btn-trash"],
                            scale=1,
                        )
                        new_chat_btn = gr.Button(
                            "新对话",
                            elem_classes=["btn-ghost", "btn-with-icon", "btn-refresh"],
                            scale=1,
                        )

        # ========== 事件绑定 ==========

        # 发送消息
        msg.submit(
            fn=chat_respond,
            inputs=[msg, chatbot, dir_content_state, memory_toggle_state, web_toggle_state],
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

        # 联网搜索开关
        web_checkbox.change(
            fn=lambda v: v,
            inputs=[web_checkbox],
            outputs=[web_toggle_state],
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
