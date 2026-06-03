"""
模型客户端模块
=============
封装通义千问（DashScope）和 DeepSeek 的 API 调用。
两者均兼容 OpenAI SDK 格式。

API Key 通过环境变量传入，不硬编码在代码中。
"""

import os
from openai import OpenAI


# ============================================================
# 从 key.txt 加载 API Key（如果文件存在且环境变量未设置）
# ============================================================
_key_file = os.path.join(os.path.dirname(__file__), "key.txt")
if os.path.exists(_key_file):
    with open(_key_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#") and not line.startswith("ERror"):
                key, value = line.split("=", 1)
                # 只有环境变量中没设时才从文件读取
                if key.strip() not in os.environ:
                    os.environ[key.strip()] = value.strip()

# ============================================================
# API Key（从环境变量读取）
# ============================================================
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# ============================================================
# 模型配置
# ============================================================
QWEN_MODEL = "qwen-plus"          # 通义千问：分析用
DEEPSEEK_MODEL = "deepseek-chat"  # DeepSeek：专家+纠错用

# ============================================================
# 客户端初始化
# ============================================================
_qwen_client = None
_deepseek_client = None


def get_qwen_client() -> OpenAI:
    """获取通义千问 OpenAI 兼容客户端"""
    global _qwen_client
    if _qwen_client is None:
        if not DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY 未设置，请在环境变量中配置")
        _qwen_client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return _qwen_client


def get_deepseek_client() -> OpenAI:
    """获取 DeepSeek OpenAI 兼容客户端"""
    global _deepseek_client
    if _deepseek_client is None:
        if not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY 未设置，请在环境变量中配置")
        _deepseek_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )
    return _deepseek_client


# ============================================================
# 模型调用函数
# ============================================================

def call_qwen(system_prompt: str, user_message: str, temperature: float = 0.7) -> str:
    """
    调用通义千问模型
    Args:
        system_prompt: 系统提示词
        user_message: 用户消息
        temperature: 温度参数 (0-1)
    Returns:
        模型回复文本
    """
    client = get_qwen_client()
    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        stream=False,
    )
    return response.choices[0].message.content or ""


def call_deepseek(system_prompt: str, user_message: str, temperature: float = 0.3) -> str:
    """
    调用 DeepSeek 模型
    Args:
        system_prompt: 系统提示词
        user_message: 用户消息
        temperature: 温度参数 (0-1)
    Returns:
        模型回复文本
    """
    client = get_deepseek_client()
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        stream=False,
    )
    return response.choices[0].message.content or ""
