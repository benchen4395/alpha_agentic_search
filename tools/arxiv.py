# tools/arxiv.py
"""arXiv 论文检索（官方 API，无需 key）。

API 文档: https://info.arxiv.org/help/api/index.html
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import requests
from configs import config


_API = "http://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom"}


def _proxies() -> dict | None:
    if config.DEFAULT_USE_PROXY and config.SEARCH_PROXY:
        return {"http": config.SEARCH_PROXY, "https": config.SEARCH_PROXY}
    return None


def search_arxiv(query: str, days: int = 5, max_results: int = 10) -> list[dict]:
    """检索 arXiv 论文。

    参数：
      query       : 关键词（会拼到 all: 字段）
      days        : 仅保留最近 N 天提交的论文（按 published 时间过滤）
      max_results : API 端最多返回多少条（过滤前）

    返回：
      [{"title", "summary", "url", "authors", "primary_category", "published"}, ...]
    """
    if not query:
        return []

    try:
        r = requests.get(
            _API,
            params={
                "search_query": f"all:{query}",
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": max_results,
            },
            timeout=20,
            proxies=_proxies(),
        )
        r.raise_for_status()
    except Exception as e:
        return [{"error": f"arXiv 请求失败: {e}"}]

    try:
        root = ET.fromstring(r.text)
    except Exception as e:
        return [{"error": f"arXiv 返回解析失败: {e}"}]

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out: list[dict] = []
    for e in root.findall("a:entry", _NS):
        pub_text = (e.findtext("a:published", default="", namespaces=_NS) or "").strip()
        try:
            pub = datetime.fromisoformat(pub_text.replace("Z", "+00:00"))
        except Exception:
            continue
        if days and pub < cutoff:
            continue

        authors = [
            (a.findtext("a:name", default="", namespaces=_NS) or "").strip()
            for a in e.findall("a:author", _NS)
        ]
        primary_cat_el = e.find("{http://arxiv.org/schemas/atom}primary_category")
        primary_cat = primary_cat_el.get("term") if primary_cat_el is not None else ""

        out.append({
            "title":            (e.findtext("a:title", default="", namespaces=_NS) or "").strip(),
            "summary":          (e.findtext("a:summary", default="", namespaces=_NS) or "").strip()[:400],
            "url":              (e.findtext("a:id", default="", namespaces=_NS) or "").strip(),
            "authors":          authors,
            "primary_category": primary_cat,
            "published":        pub.isoformat(),
        })
    return out
