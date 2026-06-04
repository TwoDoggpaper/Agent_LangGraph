"""
联网搜索模块
============
为 AI Agent 提供互联网搜索能力。
使用 Bing 网页搜索（国内可访问，无需 API Key）。

功能：
  - search_web()     — 通用网页搜索
  返回格式化文本，适合直接注入 LLM 上下文。
"""

import re
import time
from typing import Optional
from urllib.parse import quote_plus


# ============================================================
# Bing 实时搜索（使用网页抓取，无需 API Key）
# ============================================================

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _safe_fetch(url: str, max_retries: int = 2, timeout: int = 15) -> Optional[str]:
    """带重试的安全 HTTP GET"""
    import requests
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.text
        except requests.RequestException:
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
    return None


def _parse_bing_results(html: str, max_results: int = 5) -> list[dict]:
    """
    解析 Bing 搜索结果页 HTML，提取标题、摘要、URL。
    Bing 结构：<li class="b_algo"><h2><a href="url">title</a></h2><div class="b_caption"><p>snippet</p>
    """
    results = []

    # 提取每个 b_algo 块
    items = re.findall(
        r'<li[^>]*class="[^"]*\bb_algo\b[^"]*"[^>]*>(.*?)</li>',
        html,
        re.DOTALL | re.IGNORECASE,
    )

    for item in items:
        if len(results) >= max_results:
            break

        # URL 和标题 — 从 <h2><a href="url">title</a></h2> 中提取
        title_match = re.search(
            r'<h2[^>]*>.*?<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>',
            item,
            re.DOTALL | re.IGNORECASE,
        )
        if not title_match:
            continue
        url = title_match.group(1)
        title = re.sub(r'<[^>]+>', '', title_match.group(2)).strip()
        title = re.sub(r'\s+', ' ', title)

        # 摘要 — 从 <div class="b_caption"><p> 中提取
        snippet_match = re.search(
            r'<div[^>]*class="b_caption"[^>]*>.*?<p[^>]*>(.*?)</p>',
            item,
            re.DOTALL | re.IGNORECASE,
        )
        snippet = ""
        if snippet_match:
            snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip()
            snippet = re.sub(r'\s+', ' ', snippet)
            # 清理 &ensp; &#0183; 等 HTML 实体
            snippet = snippet.replace('&ensp;', ' ').replace('&#0183;', '·')

        if title and url:
            results.append({
                "title": title,
                "body": snippet,
                "href": url,
            })

    return results


def _search_bing(query: str, max_results: int = 5) -> list[dict]:
    """
    通过 Bing 网页搜索获取结果

    Args:
        query:       搜索关键词
        max_results: 返回结果数量

    Returns:
        结果列表，搜索失败时返回空列表
    """
    encoded = quote_plus(query)
    url = f"https://www.bing.com/search?q={encoded}&count={max_results}&setlang=zh-Hans"

    html = _safe_fetch(url)
    if not html:
        return []

    results = _parse_bing_results(html, max_results)
    return results


# ============================================================
# 公开 API
# ============================================================


def search_web(query: str, max_results: int = 5) -> str:
    """
    通用网页搜索（使用 Bing）。

    Args:
        query:       搜索关键词
        max_results: 返回结果数量 (1-10)

    Returns:
        格式化后的搜索结果文本
    """
    if not query or not query.strip():
        return "[联网搜索] 搜索查询为空"

    raw = _search_bing(query.strip(), min(max_results, 10))

    if not raw:
        return (
            f"[联网搜索] 搜索“{query}”时遇到网络问题，"
            f"请检查网络连接或稍后重试。"
        )

    lines = [
        f"🔍 搜索词：{query}",
        f"📄 共找到 {len(raw)} 条结果",
        "",
    ]
    for i, r in enumerate(raw, 1):
        title = (r.get("title") or "").strip()
        snippet = (r.get("body") or "").strip()
        url = (r.get("href") or "").strip()
        lines.append(f"─── 结果 {i} ───")
        lines.append(f"标题：{title}")
        if snippet:
            lines.append(f"摘要：{snippet}")
        lines.append(f"来源：{url}")
        lines.append("")

    return "\n".join(lines)
