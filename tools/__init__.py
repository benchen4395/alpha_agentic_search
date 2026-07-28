# tools/__init__.py
"""工具注册表：每个工具一个 module，统一通过 TOOLS 暴露。

新增工具步骤：
  1. 在本目录新建 xxx.py，写一个或多个函数
  2. 在下方 TOOLS 字典里登记 {name, fn, desc, params}
"""
from __future__ import annotations

from . import weather, github_repo, arxiv, web_search, current_time

TOOLS: dict[str, dict] = {
    "get_current_time": {
        "fn": current_time.get_current_time,
        "desc": "查询当前日期、时间、星期；支持指定城市/地区或 IANA 时区。适合回答现在几点、今天几号、今天星期几、拉斯维加斯时间是多少等问题。",
        "params": {
            "location": "可选，城市或地区名，如 '北京'、'拉斯维加斯'、'New York'、'London'；若用户问题包含地点，请填这里。",
            "timezone": "可选，IANA 时区，如 'Asia/Shanghai'、'America/Los_Angeles'；若用户直接给出时区，请填这里。",
        },
    },
    "get_weather": {
        "fn": weather.get_weather,
        "desc": "查询指定城市当前天气（温度、天气状况、风向等）。",
        "params": {
            "city": "城市名（中文/英文/拼音皆可），如 '北京' / 'Beijing'",
        },
    },
    "get_repo_info": {
        "fn": github_repo.get_repo_info,
        "desc": "查询 GitHub 仓库的 star / fork / 描述等信息。",
        "params": {
            "full_name": "owner/repo 形式，如 'openclaw/openclaw'",
        },
    },
    "search_arxiv": {
        "fn": arxiv.search_arxiv,
        "desc": "检索 arXiv 上最近 N 天发布的论文（按提交时间倒序）。",
        "params": {
            "query": "检索关键词，如 'on-policy distillation'",
            "days": "回看天数，默认 5",
            "max_results": "返回条数上限，默认 10",
        },
    }
}

# "open_web_search": {
#         "fn": web_search.open_web_search,
#         "desc": "通用网络检索（新闻、明星动态、不在专用 API 范围内的开放问题）。",
#         "params": {
#             "query": "搜索词",
#             "top_k": "返回条数，默认 5",
#         },
#     },


def list_tools_brief() -> str:
    """生成给 LLM 看的工具清单（紧凑格式）。"""
    lines = []
    for name, t in TOOLS.items():
        params = ", ".join(t["params"].keys())
        lines.append(f"- {name}({params}): {t['desc']}")
        for p, p_desc in t["params"].items():
            lines.append(f"    · {p}: {p_desc}")
    return "\n".join(lines)


def call_tool(name: str, args: dict | None = None):
    """根据工具名调度。未登记 / 异常时返回错误信息。"""
    if name not in TOOLS:
        return {"error": f"unknown tool: {name}"}
    try:
        return TOOLS[name]["fn"](**(args or {}))
    except TypeError as e:
        return {"error": f"参数错误: {e}"}
    except Exception as e:
        return {"error": f"工具执行失败: {e}"}
