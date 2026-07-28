# tools/github_repo.py
"""GitHub 仓库信息查询。

公共仓库无需 token，但匿名调用有 60 次/小时的速率限制。
若 config.GITHUB_TOKEN 存在，会带上 Authorization 头放宽到 5000 次/小时。
"""
from __future__ import annotations

import requests
from configs import config


_API = "https://api.github.com"


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "agentic-search"}
    token = getattr(config, "GITHUB_TOKEN", "")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _proxies() -> dict | None:
    if config.DEFAULT_USE_PROXY and config.SEARCH_PROXY:
        return {"http": config.SEARCH_PROXY, "https": config.SEARCH_PROXY}
    return None


def get_repo_info(full_name: str) -> dict:
    """查询单个仓库基本信息。

    full_name : 'owner/repo'，例如 'openclaw/openclaw'
    """
    if "/" not in (full_name or ""):
        return {"error": "full_name 必须是 'owner/repo' 形式"}

    try:
        r = requests.get(
            f"{_API}/repos/{full_name}",
            headers=_headers(),
            proxies=_proxies(),
            timeout=15,
        )
    except Exception as e:
        return {"error": f"GitHub 请求失败: {e}"}

    if r.status_code == 404:
        return {"error": f"repo not found: {full_name}"}
    if r.status_code != 200:
        return {"error": f"GitHub API 返回 {r.status_code}: {r.text[:200]}"}

    j = r.json()
    return {
        "name":         j.get("full_name"),
        "stars":        j.get("stargazers_count"),
        "forks":        j.get("forks_count"),
        "watchers":     j.get("subscribers_count"),
        "open_issues":  j.get("open_issues_count"),
        "language":     j.get("language"),
        "license":      (j.get("license") or {}).get("spdx_id"),
        "description":  j.get("description"),
        "homepage":     j.get("homepage"),
        "url":          j.get("html_url"),
        "created_at":   j.get("created_at"),
        "updated_at":   j.get("updated_at"),
        "pushed_at":    j.get("pushed_at"),
    }


def search_repo(query: str, top_k: int = 5) -> list[dict]:
    """当用户给的不是精确 owner/repo，先模糊搜索仓库。"""
    try:
        r = requests.get(
            f"{_API}/search/repositories",
            headers=_headers(),
            proxies=_proxies(),
            params={"q": query, "sort": "stars", "order": "desc", "per_page": top_k},
            timeout=15,
        )
        r.raise_for_status()
        items = r.json().get("items", [])
        return [
            {
                "name": it.get("full_name"),
                "stars": it.get("stargazers_count"),
                "description": it.get("description"),
                "url": it.get("html_url"),
            }
            for it in items
        ]
    except Exception as e:
        return [{"error": f"GitHub 搜索失败: {e}"}]
