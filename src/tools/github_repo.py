# tools/github_repo.py
"""GitHub 仓库信息查询。

公共仓库无需 token，但匿名调用有 60 次/小时的速率限制。
若 config.GITHUB_TOKEN 存在，会带上 Authorization 头放宽到 5000 次/小时。

HTTP 层（重试 / 代理 / UA / 超时）统一走 `tools/_http.py`。
"""
from __future__ import annotations

from src.configs import config

from ._http import get_json


_API = "https://api.github.com"


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    token = getattr(config, "GITHUB_TOKEN", "")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def get_repo_info(full_name: str) -> dict:
    """查询单个仓库基本信息。

    full_name : 'owner/repo'，例如 'openclaw/openclaw'

    Raises:
        ValueError:    full_name 格式不对。
        ToolHTTPError: 仓库不存在（404）、限流（403/429）或网络故障。
                       抛异常而非返回 {"error": ...}，让 call_tool 归类为
                       exec_error → agent 降级到通用检索。
    """
    full_name = (full_name or "").strip().strip("/")
    if full_name.count("/") != 1 or not all(full_name.split("/")):
        raise ValueError(f"full_name 必须是 'owner/repo' 形式，收到: {full_name!r}")

    j = get_json(
        f"{_API}/repos/{full_name}",
        headers=_headers(),
        timeout=15,
        label=f"GitHub({full_name})",
    )
    return {
        "name":         j.get("full_name"),
        "stars":        j.get("stargazers_count"),
        "forks":        j.get("forks_count"),
        "watchers":     j.get("subscribers_count"),
        "open_issues":  j.get("open_issues_count"),
        "language":     j.get("language"),
        "topics":       j.get("topics") or [],
        "license":      (j.get("license") or {}).get("spdx_id"),
        "description":  j.get("description"),
        "homepage":     j.get("homepage"),
        "url":          j.get("html_url"),
        "archived":     j.get("archived"),
        "created_at":   j.get("created_at"),
        "updated_at":   j.get("updated_at"),
        "pushed_at":    j.get("pushed_at"),
        "source":       "GitHub API",
    }


def search_repo(query: str, top_k: int = 5) -> list[dict]:
    """当用户给的不是精确 owner/repo 时，先模糊搜索仓库（按 star 排序）。

    Raises:
        ValueError:    query 为空。
        ToolHTTPError: 网络故障或 GitHub 限流。
    """
    if not (query or "").strip():
        raise ValueError("query 不能为空")
    try:
        top_k = max(1, min(int(top_k), 20))
    except (TypeError, ValueError):
        top_k = 5

    data = get_json(
        f"{_API}/search/repositories",
        headers=_headers(),
        params={"q": query, "sort": "stars", "order": "desc", "per_page": top_k},
        timeout=15,
        label="GitHub 搜索",
    )
    return [
        {
            "name":        it.get("full_name"),
            "stars":       it.get("stargazers_count"),
            "language":    it.get("language"),
            "description": it.get("description"),
            "url":         it.get("html_url"),
            "pushed_at":   it.get("pushed_at"),
        }
        for it in (data.get("items") or [])
    ]
