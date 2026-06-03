"""
记忆系统模块
=============
提供对话记忆持久化功能，使 AI 能记住历史对话内容。

功能：
- 自动保存每次对话摘要
- 启动时自动加载历史记忆
- 将近期对话作为上下文注入分析阶段
"""

import json
import os
from datetime import datetime
from typing import Optional

# ============================================================
# 存储路径
# ============================================================
MEMORY_DIR = os.path.join(os.path.dirname(__file__), "memory_store")
MEMORY_FILE = os.path.join(MEMORY_DIR, "conversations.json")


class ConversationMemory:
    """对话记忆管理器（单例模式）"""

    def __init__(self):
        self.conversations: list[dict] = []
        self._load()

    # ----------------------------------------------------------
    # 持久化
    # ----------------------------------------------------------
    def _load(self):
        """从磁盘加载记忆"""
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.conversations = data if isinstance(data, list) else []
            except (json.JSONDecodeError, IOError):
                self.conversations = []

    def save(self):
        """保存记忆到磁盘（仅保留最近 50 条）"""
        os.makedirs(MEMORY_DIR, exist_ok=True)
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.conversations[-50:], f, ensure_ascii=False, indent=2)

    # ----------------------------------------------------------
    # 读写
    # ----------------------------------------------------------
    def add_conversation(
        self,
        conversation: str,
        analysis: str = "",
        result: str = "",
    ):
        """添加一条对话记录"""
        now = datetime.now()
        self.conversations.append({
            "id": len(self.conversations) + 1,
            "timestamp": now.isoformat(),
            "date": now.strftime("%Y-%m-%d %H:%M"),
            "conversation": conversation[:300],
            "analysis": analysis[:200],
            "result": result[:300],
        })
        self.save()

    def get_recent_context(self, max_items: int = 5) -> str:
        """获取最近对话摘要，作为注入提示词的上下文"""
        recent = self.conversations[-max_items:]
        if not recent:
            return ""
        lines = ["【历史对话记忆 — 供参考】"]
        for i, conv in enumerate(recent, 1):
            summary = conv["conversation"][:120]
            lines.append(f"  {i}. [{conv['date']}] {summary}…")
        return "\n".join(lines)

    def delete_all(self):
        """清空所有记忆"""
        self.conversations.clear()
        self.save()


# ----------------------------------------------------------
# 全局单例
# ----------------------------------------------------------
_memory: Optional[ConversationMemory] = None


def get_memory() -> ConversationMemory:
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory
