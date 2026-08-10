# tools/web_search.py
"""把现有 searcher.py 包装成 Tool，便于 Router 统一调度。"""
from __future__ import annotations

from src.search.searcher import web_search as _ws, format_results
from src.pipeline.query_rewriter import shorten_query   # 规则方式的 query 改写（含否定意图处理）


def open_web_search(query: str, top_k: int = 5) -> str:
    """开放网络检索；返回已格式化的可读文本。"""
    if not query:
        return "(空 query)"
    # 对过长的query进行修改
    query = shorten_query(query) if len(query) > 20 else query
    results = _ws(query, top_k=top_k)
    return format_results(results)
