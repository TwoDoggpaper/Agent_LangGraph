"""测试脚本 — 验证模块导入和图构建"""
import sys, os
sys.path.insert(0, r"C:\Users\Think\PycharmProjects\LangGraph框架agent项目")

from models import call_qwen, call_deepseek, get_qwen_client, get_deepseek_client
from agents import build_agent_graph, run_agent

print("✅ 所有模块导入成功")
graph = build_agent_graph()
print(f"✅ LangGraph 图构建成功: {type(graph)}")
print("✅ 项目结构完整，可以启动")
