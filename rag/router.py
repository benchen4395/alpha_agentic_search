# rag/router.py
"""Router：根据 query 决定激活哪些层。

策略：
    - offline_only : 只走 L1 + L2 + L3 + L5（不联网）
    - web_only     : 只走 L1 + L4
    - hybrid       : L1 → 若命中直接短路；否则并行 (L2, L3, L5)，
                     若 top1 分数不够高再补 L4（推荐，默认）

未来可以在此接入 LLM Router；当前用规则即可。
"""
from __future__ import annotations

from typing import Iterable

from . import config as rag_config
from .types import LayerName


_TIME_SENSITIVE_HINTS = (
    "今天", "昨天", "最新", "刚刚", "刚才", "今日", "本周", "本月",
    "现在", "最近", "今年", "2024", "2025", "2026", "股价", "汇率",
    "天气", "新闻", "latest", "today", "yesterday", "current",
)


def is_time_sensitive(query: str) -> bool:
    q = (query or "").lower()
    return any(h in q for h in _TIME_SENSITIVE_HINTS)


def route(query: str, strategy: str = "") -> list[LayerName]:
    """返回本次要激活的层名列表（不含 L1，L1 由 Retriever 单独短路）。"""
    strategy = strategy or rag_config.ROUTER_STRATEGY

    if strategy == "offline_only":
        return ["L2_wiki", "L3_history", "L5_kg"]
    if strategy == "web_only":
        return ["L4_web"]

    # hybrid（默认）：时间敏感 query 直接叠加 L4
    layers: list[LayerName] = ["L2_wiki", "L3_history", "L5_kg"]
    if is_time_sensitive(query):
        layers.append("L4_web")
    return layers


def should_fallback_to_web(top_score: float) -> bool:
    """当离线层最高分低于阈值时，追加 L4。"""
    return top_score < rag_config.WEB_FALLBACK_THRESHOLD
