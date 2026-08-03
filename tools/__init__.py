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


def call_tool(name: str, args: dict | None = None) -> dict:
    """根据工具名调度，返回**统一的结构化结果**（P0-4）。

    ════════════════════════════════════════════════════════════════════
    为什么要改成结构化返回
    ════════════════════════════════════════════════════════════════════
    改造前本函数失败时返回 `{"error": "..."}`，而 `agent.chat()` 的判定是：

        used_tool = (... and tool_decision.get("result") is not None)

    `{"error": ...}` 显然不是 `None` → `used_tool = True`
    → **agent 直接跳过全部检索**，并把错误信息当作"外部资料"喂给 LLM。

    真实后果：天气 API 挂了 / GitHub 限流 / arXiv 超时，用户得到的是
    "抱歉，信息不足"，而不是自动降级去走搜索兜底。工具越多，这个
    静默失败面越大。

    ════════════════════════════════════════════════════════════════════
    统一返回契约
    ════════════════════════════════════════════════════════════════════
        {
          "ok":    bool,          # 是否成功。agent 只信这个字段
          "data":  Any | None,    # 成功时的工具原始返回
          "error": str | None,    # 失败原因（写日志/事件，不进 prompt）
          "kind":  str,           # 失败类型，便于分类监控与告警：
                                  #   "unknown_tool" 未注册的工具名
                                  #   "bad_args"     参数不匹配（LLM 幻觉参数）
                                  #   "exec_error"   工具内部抛异常
                                  #   "empty"        执行成功但没拿到有效数据
        }

    `kind` 的价值：`bad_args` 高说明 router prompt 需要改；
    `exec_error` 高说明外部 API 不稳定，该加重试/熔断。两者处置完全不同。
    """
    if name not in TOOLS:
        return {"ok": False, "data": None, "kind": "unknown_tool",
                "error": f"unknown tool: {name}"}
    try:
        data = TOOLS[name]["fn"](**(args or {}))
    except TypeError as e:
        # LLM 经常幻觉出工具不支持的参数名 → 归类为 bad_args
        return {"ok": False, "data": None, "kind": "bad_args",
                "error": f"参数错误: {e}"}
    except Exception as e:
        return {"ok": False, "data": None, "kind": "exec_error",
                "error": f"工具执行失败: {e}"}

    # ---- 成功路径的二次校验 ----
    # 有些工具（weather/github）自己会返回 {"error": ...} 而不抛异常，
    # 必须识别出来，否则又变成"把错误当资料"。
    if data is None:
        return {"ok": False, "data": None, "kind": "empty",
                "error": "工具返回空结果"}
    if isinstance(data, dict) and data.get("error"):
        return {"ok": False, "data": None, "kind": "exec_error",
                "error": str(data["error"])}
    if isinstance(data, (list, tuple)) and len(data) == 0:
        return {"ok": False, "data": None, "kind": "empty",
                "error": "工具返回空列表"}

    return {"ok": True, "data": data, "kind": "ok", "error": None}
