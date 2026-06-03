"""
文件读取模块
=============
支持批量读取文件／目录内容，供 AI 分析参考。

支持扩展名：
  文本类 — .txt .md .py .json .csv .yaml .yml .xml .html .css .js .ts .log 等
  PDF    — .pdf（通过 pypdf 库提取文本）

注意：扫描件/图片型 PDF 无法提取文本，将提示"无提取内容"。
"""

import os

# 支持的文本文件扩展名（全小写）
TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".json", ".csv", ".yaml", ".yml",
    ".xml", ".html", ".htm", ".css", ".js", ".ts", ".jsx", ".tsx",
    ".log", ".ini", ".cfg", ".conf", ".toml", ".rst", ".sql",
    ".env", ".sh", ".bat", ".ps1", ".gradle", ".properties",
    ".yaml", ".yml", ".tex", ".vue", ".svelte",
    ".pdf",   # PDF 通过 pypdf 库读取
}

# 每个文件的读入上限（字符数）
MAX_FILE_CHARS = 50_000
# 总内容上限
MAX_TOTAL_CHARS = 100_000


def read_text_file(file_path: str) -> str:
    """
    读取单个文本文件，自动尝试 UTF-8 / GBK。
    返回文件内容；失败时返回错误描述。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="gbk") as f:
                return f.read()
        except Exception as e:
            return f"[编码识别失败] {os.path.basename(file_path)}: {e}"
    except Exception as e:
        return f"[读取失败] {os.path.basename(file_path)}: {e}"


def read_pdf_file(file_path: str) -> str:
    """
    使用 pypdf 读取 PDF 文件的文本内容。

    Args:
        file_path: PDF 文件路径

    Returns:
        提取的文本内容，失败时返回错误描述
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        pages_text = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                pages_text.append(f"--- 第 {i} 页 ---\n{text.strip()}")

        if not pages_text:
            return f"[PDF 无提取内容] {os.path.basename(file_path)}（可能为扫描件/图片型 PDF）"

        header = f"【PDF 文件：{os.path.basename(file_path)}】（共 {len(reader.pages)} 页）\n"
        return header + "\n\n".join(pages_text)
    except ImportError:
        return f"[PDF 读取失败] {os.path.basename(file_path)}: 缺少 pypdf 库，请执行 pip install pypdf"
    except Exception as e:
        return f"[PDF 读取失败] {os.path.basename(file_path)}: {e}"


def read_files_from_paths(paths: list[str]) -> str:
    """
    批量读取文件或目录。

    Args:
        paths: 文件路径或目录路径列表

    Returns:
        合并后的文本内容
    """
    all_content: list[str] = []
    total_chars = 0

    def _append(label: str, text: str):
        nonlocal total_chars
        if total_chars >= MAX_TOTAL_CHARS:
            return
        block = f"===== {label} =====\n{text}\n"
        if total_chars + len(block) > MAX_TOTAL_CHARS:
            block = block[: MAX_TOTAL_CHARS - total_chars]
            block += "\n… (已截断)"
        all_content.append(block)
        total_chars += len(block)

    for path in paths:
        if not os.path.exists(path):
            all_content.append(f"[路径不存在] {path}")
            continue

        # ---- 单个文件 ----
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                content = read_pdf_file(path)
                if len(content) > MAX_FILE_CHARS:
                    content = content[:MAX_FILE_CHARS] + "\n… (文件过长已截断)"
                _append(os.path.basename(path), content)
            elif ext in TEXT_EXTENSIONS:
                content = read_text_file(path)
                if len(content) > MAX_FILE_CHARS:
                    content = content[:MAX_FILE_CHARS] + "\n… (文件过长已截断)"
                _append(os.path.basename(path), content)
            else:
                all_content.append(f"[跳过非文本文件] {os.path.basename(path)}")

        # ---- 目录 ----
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                # 跳过无关目录
                dirs[:] = [
                    d
                    for d in dirs
                    if not d.startswith(".")
                    and d not in ("__pycache__", "node_modules", "venv", ".venv", ".git", "idea")
                ]
                for file in sorted(files):
                    ext = os.path.splitext(file)[1].lower()
                    if ext not in TEXT_EXTENSIONS:
                        continue
                    filepath = os.path.join(root, file)
                    if ext == ".pdf":
                        content = read_pdf_file(filepath)
                    else:
                        content = read_text_file(filepath)
                    if len(content) > MAX_FILE_CHARS:
                        content = content[:MAX_FILE_CHARS] + "\n… (文件过长已截断)"
                    rel = os.path.relpath(filepath, path)
                    _append(rel, content)

    return "\n".join(all_content)


def format_file_context(file_contents: str) -> str:
    """
    将原始文件内容包装成供 AI 阅读的上下文块。
    空内容返回空字符串。
    """
    if not file_contents.strip():
        return ""
    if len(file_contents) > MAX_TOTAL_CHARS:
        file_contents = file_contents[:MAX_TOTAL_CHARS] + "\n… (总内容已截断)"
    return f"\n【参考文件内容】\n{file_contents}\n"
