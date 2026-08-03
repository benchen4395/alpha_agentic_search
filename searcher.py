# searcher.py
"""联网检索：缓存 → DuckDuckGo → Tavily/Serper → Bing 兜底。

每个 provider 返回原始 list[dict]，统一在 web_search() 末尾标准化为：
  {"title": str, "url": str, "snippet": str}
"""
from __future__ import annotations

import time
import random
import hashlib
import urllib.parse

from ddgs import DDGS  # 新版包名，旧版可改为 from duckduckgo_search import DDGS
from configs import config
from query_rewriter import shorten_query   # 规则方式的 query 改写（含否定意图处理）

try:
    import requests
    from bs4 import BeautifulSoup
except Exception:  # 缺依赖时仅 Bing/HTTP API 不可用
    requests = None
    BeautifulSoup = None

try:
    import diskcache
    _cache = ( diskcache.Cache(config.SEARCH_CACHE_DIR) if config.SEARCH_CACHE_ENABLED else None )
except Exception:
    _cache = None


_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


# ---------------- 工具 ----------------
def _log(msg: str) -> None:
    print(f"[searcher] {msg}")


def _proxy(use_proxy: bool) -> str | None:
    return config.SEARCH_PROXY or None if use_proxy else None


def _proxies(use_proxy: bool) -> dict | None:
    p = _proxy(use_proxy)
    return {"http": p, "https": p} if p else None


def _cache_key(query: str, top_k: int) -> str:
    # 这里增加top_k, 也是为了让模型的检索数量也做到cache；比如新的搜索要求query检索10个，那么就会触发新的检索
    h = hashlib.md5(f"{query}|{top_k}".encode()).hexdigest()
    return f"web_search:{h}"


# ---------------- Providers ----------------
def _ddg(query: str, top_k: int, use_proxy: bool) -> list[dict]:
    """DDG：多 backend/region 轮换重试。

    ⚡ 延迟收敛（本次性能优化）：重试次数与退避间隔改为可配置
    （见 configs/config.py 里 DDG_* 的实测数据与推算）。要点：
      * 最后一次尝试**不再 sleep** —— 原实现在循环末尾无条件退避，
        即使已经没有下一次重试了，白等 1~2.5s；
      * 退避区间收敛到 0.3~1.0s，避免退避本身成为延迟大头。
    """
    proxy = _proxy(use_proxy)
    backends, regions = config.DDG_BACKENDS, config.DDG_REGIONS
    last_err = None
    retries = max(int(config.DDG_MAX_RETRIES), 1)
    for i in range(retries):
        backend = backends[i % len(backends)]
        region = random.choice(regions)
        kwargs = {"timeout": config.DDG_TIMEOUT, **({"proxy": proxy} if proxy else {})}
        try:
            with DDGS(**kwargs) as ddgs:
                raw = list(ddgs.text(
                    query, max_results=top_k,
                    safesearch="moderate", region=region, backend=backend,
                ))
            if raw:
                return raw
            _log(f"DDG 空结果 backend={backend} region={region}，重试...")
        except Exception as e:
            last_err = e
            _log(f"DDG 失败 backend={backend}: {e}")
        # 只有"还会再试一次"时才退避；最后一轮直接返回，别白等。
        if i < retries - 1:
            time.sleep(
                config.DDG_RETRY_BACKOFF_MIN
                + random.random()
                * max(config.DDG_RETRY_BACKOFF_MAX - config.DDG_RETRY_BACKOFF_MIN, 0.0)
            )
    if last_err:
        _log(f"DDG 最终失败: {last_err}")
    return []


def _http_post(url: str, *, json=None, headers=None, use_proxy=False, timeout=20):
    """统一的 POST 调用，失败返回 None。"""
    if requests is None:
        return None
    try:
        r = requests.post(
            url, json=json, headers=headers,
            timeout=timeout, proxies=_proxies(use_proxy),
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        _log(f"POST {url} 失败: {e}")
        return None


def _tavily(query: str, top_k: int, use_proxy: bool) -> list[dict]:
    if not config.TAVILY_API_KEY:
        return []
    data = _http_post(
        "https://api.tavily.com/search",
        json={
            "api_key": config.TAVILY_API_KEY,
            "query": query, "max_results": top_k, "search_depth": "basic",
        },
        use_proxy=use_proxy,
    )
    if not data:
        return []
    return [
        {"title": it.get("title", ""), "href": it.get("url", ""), "body": it.get("content", "")}
        for it in (data.get("results") or [])[:top_k]
    ]


def _serper(query: str, top_k: int, use_proxy: bool) -> list[dict]:
    if not config.SERPER_API_KEY:
        return []
    data = _http_post(
        "https://google.serper.dev/search",
        json={"q": query, "num": top_k},
        headers={"X-API-KEY": config.SERPER_API_KEY, "Content-Type": "application/json"},
        use_proxy=use_proxy,
    )
    if not data:
        return []
    return [
        {"title": it.get("title", ""), "href": it.get("link", ""), "body": it.get("snippet", "")}
        for it in (data.get("organic") or [])[:top_k]
    ]


def _bing(query: str, top_k: int, use_proxy: bool) -> list[dict]:
    """Bing HTML 兜底（无 API key，解析公开搜索页）。"""
    if requests is None or BeautifulSoup is None:
        return []
    try:
        r = requests.get(
            "https://www.bing.com/search?q=" + urllib.parse.quote(query),
            headers={"User-Agent": _UA},
            timeout=15, proxies=_proxies(use_proxy),
        )
        soup = BeautifulSoup(r.text, "html.parser")
        items = []
        for li in soup.select("li.b_algo")[:top_k]:
            a = li.select_one("h2 a")
            p = li.select_one("p")
            if a:
                items.append({
                    "title": a.get_text(strip=True),
                    "href": a.get("href", ""),
                    "body": p.get_text(strip=True) if p else "",
                })
        return items
    except Exception as e:
        _log(f"Bing 兜底失败: {e}")
        return []


# ---------------- 对外入口 ----------------
def web_search(
    query: str,
    top_k: int | None = None,
    use_cache: bool = True,
    use_proxy: bool | None = False,
) -> list[dict]:
    """执行联网搜索，返回 [{title, url, snippet}, ...]。

    检索链路（前一步无结果才走下一步）：
      0) 缓存命中 → 直接返回
      1) DDG
      2) DDG（关键词截断）
      3) Tavily（有 key）
      4) Serper（有 key）
      5) Bing HTML 兜底
    """
    query = (query or "").strip()
    if not query:
        return []
    top_k = top_k or config.SEARCH_DEFAULT_TOP_K
    if use_proxy is None:
        use_proxy = config.DEFAULT_USE_PROXY

    # 0) 缓存
    ck = _cache_key(query, top_k)
    if use_cache and _cache is not None:
        cached = _cache.get(ck)
        if cached:
            _log(f"缓存命中: {query!r}")
            return cached

    # 1~5) 按顺序尝试 provider；每项是 (label, fn, query)
    short = shorten_query(query)
    pipeline = [
        ("DDG",        _ddg,    query),
        ("DDG-short",  _ddg,    short) if short != query else None,
        ("Tavily",     _tavily, query),
        ("Serper",     _serper, query),
        ("Bing",       _bing,   query),
    ]
    raw: list[dict] = []
    for step in pipeline:
        if step is None:
            continue
        label, fn, q = step
        _log(f"尝试 {label}...")
        raw = fn(q, top_k, use_proxy)
        if raw:
            _log(f"{label} 命中 {len(raw)} 条")
            break

    results = [
        {
            "title":   r.get("title", ""),
            "url":     r.get("href") or r.get("url", ""),
            "snippet": r.get("body") or r.get("snippet", ""),
        }
        for r in raw
    ]

    if use_cache and _cache is not None and results:
        _cache.set(ck, results, expire=config.SEARCH_CACHE_TTL)
    return results


def format_results(results: list[dict]) -> str:
    """把检索结果格式化成可读文本，喂给主控 LLM。"""
    if not results:
        return "(无搜索结果)"
    return "\n\n".join(
        f"[{i}] {r['title']}\nURL: {r['url']}\n摘要: {r['snippet']}"
        for i, r in enumerate(results, 1)
    )


def clear_cache() -> int:
    """清空检索缓存，返回删除条数。"""
    if _cache is None:
        return 0
    n = len(_cache)
    _cache.clear()
    return n
