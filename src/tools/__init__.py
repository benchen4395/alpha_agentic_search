# tools/__init__.py
"""工具注册表：每个工具一个 module，统一通过 TOOLS 暴露。

新增工具步骤：
  1. 在本目录新建 xxx.py，写一个或多个函数
  2. 在下方 TOOLS 字典里登记 {name, fn, desc, params}

工具函数的约定（重要）：
  * 成功 → 返回 dict 或 list[dict]
  * 失败 → **抛异常**（网络类用 `tools._http.ToolHTTPError`），
           不要返回 `{"error": ...}` 或 `[{"error": ...}]`
           —— 原因见 `call_tool()` 的 docstring。
  * HTTP 请求统一走 `tools/_http.py`（自带重试 / 代理 / UA / 超时）。
"""
from __future__ import annotations

from . import weather, github_repo, arxiv, current_time
# `web_search` 有意不在上面一行里导入使用：它**不注册为工具**，
# 原因见文件下方"关于 web_search.open_web_search"的完整说明。
# 这里仍保留导入以便 `tools.web_search` 可被 scripts / A-B 对照直接引用。
from . import web_search  # noqa: F401  (deliberately unregistered)

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
        "desc": (
            "查询指定城市的当前天气实况（温度、体感、天气状况、湿度、风向、"
            "紫外线等）以及未来几天预报（最高/最低温、降雨概率、日出日落）。"
            "适合回答现在天气怎么样、明天会不会下雨、要不要带伞等问题。"
        ),
        "params": {
            "city": "城市名（中文/英文/拼音皆可），如 '北京' / 'Beijing'；也支持机场码或 '纬度,经度'",
            "forecast_days": "附带几天预报，0~3，默认 2（今天+明天）；只问当前天气可传 0",
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
        "desc": (
            "检索 arXiv 学术论文（标题、摘要、作者、分类、PDF 链接）。"
            "适合回答某个研究方向最近有哪些新论文、某个方法出自哪篇文章等问题。"
        ),
        "params": {
            "query": "检索主题，用英文效果最好，如 'on-policy distillation'",
            "days": "只看最近 N 天提交的论文，默认 5；想搜全部历史论文请传 0",
            "max_results": "返回条数上限，1~50，默认 10",
            "sort_by": "'submittedDate'（默认，最新优先）或 'relevance'（最相关优先）",
        },
    },
    "search_github_repo": {
        "fn": github_repo.search_repo,
        "desc": (
            "按关键词模糊搜索 GitHub 仓库（按 star 数排序）。"
            "当用户只给了项目名或描述、而非精确的 'owner/repo' 时用这个。"
        ),
        "params": {
            "query": "搜索关键词，如 'agentic search framework'",
            "top_k": "返回条数，默认 5",
        },
    },
}

# ---------------------------------------------------------------------------
# 关于 web_search.open_web_search：**有意不注册**
# ---------------------------------------------------------------------------
# `tools/web_search.py` 把 `searcher.web_search` 包装成了工具，但它**不在**
# 上面的 TOOLS 里，因此 router 永远不会选中它。这不是遗漏，是设计决定：
#
# 工具通路只应放「RAG 无法覆盖的确定性数据源」：实时时间、天气实况、
# GitHub 仓库指标、arXiv 结构化元数据 —— 这些要么搜索引擎给不准，
# 要么需要结构化字段而非网页摘要。
#
# 文件保留的价值：`enable_rag=False` 的 A/B 对照、以及 scripts/ 下想
# 直接拿到"格式化好的检索文本"的调试场景。如需临时启用，取消下面的注释
# 并**同时**确认你真的想绕过 RAG。
#
#     "open_web_search": {
#         "fn": web_search.open_web_search,
#         "desc": "通用网络检索（新闻、明星动态、不在专用 API 范围内的开放问题）。",
#         "params": {"query": "搜索词", "top_k": "返回条数，默认 5"},
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
    """根据工具名调度，返回**统一的结构化结果**。

    ════════════════════════════════════════════════════════════════════
    为什么要改成结构化返回
    ════════════════════════════════════════════════════════════════════
    若本函数失败时只返回 `{"error": "..."}`，而 `agent.chat()` 的判定是：

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

    ════════════════════════════════════════════════════════════════════
    工具作者约定：失败请**抛异常**，不要返回错误值
    ════════════════════════════════════════════════════════════════════
    抛出的异常会被下面的 `except Exception` 捕获并归类为 `exec_error`，
    这是唯一在**所有返回类型**下都可靠的失败表达方式。
    用返回值表达失败则依赖本函数逐一识别其形状（dict / list / 其它），
    只要有一种没覆盖到就是静默失败（详见下方 list 分支里的实测案例）。
    网络类失败建议直接用 `tools._http.ToolHTTPError`。
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
    # 有些工具自己会返回 {"error": ...} 而不抛异常，必须识别出来，
    # 否则又变成"把错误当资料"。
    if data is None:
        return {"ok": False, "data": None, "kind": "empty",
                "error": "工具返回空结果"}
    if isinstance(data, dict) and data.get("error"):
        return {"ok": False, "data": None, "kind": "exec_error",
                "error": str(data["error"])}
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return {"ok": False, "data": None, "kind": "empty",
                    "error": "工具返回空列表"}
        # ══════════════════════════════════════════════════════════════
        # 列表里包着的错误 —— 一个会绕过全部防线的静默失败
        # ══════════════════════════════════════════════════════════════
        # 返回 list 的工具（search_arxiv / search_github_repo）无法用
        # `{"error": ...}` 表达失败，只能包成 `[{"error": "..."}]`。
        # 而上面那条 dict 检查对它无效（外层是 list 不是 dict），
        # 空列表检查也无效（长度是 1 不是 0）——
        # 于是 `ok=True`，错误文本被当作"检索到的论文"喂给 LLM。
        #
        # 实测（未修复时）：
        #     >>> call_tool('search_arxiv', {'query': 'test'})   # 网络故障
        #     {'ok': True, 'data': [{'error': 'arXiv 请求失败: ...'}],
        #      'kind': 'ok', 'error': None}
        # agent 据此判定工具成功 → 跳过整个 RAG 检索通路 → 用户拿到的是
        # 基于一句报错文本编出来的答案。这比直接报错危险得多，因为
        # **看起来一切正常**。
        #
        # 本项目的工具已统一改为抛 ToolHTTPError（见 tools/_http.py），
        # 但这条校验仍必须保留：它是**契约层**的兜底，防止将来新增工具
        # 或第三方 SDK 重新引入这种返回风格。
        first = data[0]
        if isinstance(first, dict) and first.get("error"):
            return {"ok": False, "data": None, "kind": "exec_error",
                    "error": str(first["error"])}

    return {"ok": True, "data": data, "kind": "ok", "error": None}
