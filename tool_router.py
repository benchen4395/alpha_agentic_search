# tool_router.py
"""Tool Router：让 LLM 决定是否需要调用工具，以及调哪个、传什么参数。

输出协议（严格 JSON）：
    {"tool": "<tool_name | NO_TOOL>", "args": {...}}

实现说明：
- 不再写死 ollama，统一走 llm_client.chat("router", ...)
- 模型/采样参数 → models_config.STAGES["router"]
- prompt 模板    → prompts.PROMPTS["router"]
"""
from re import Pattern

import json
import re

from llm_client import complete as llm_complete
from configs.prompts import render as render_prompt
from tools import TOOLS, list_tools_brief, call_tool


_JSON_RE: Pattern[str] = re.compile(r"\{.*\}", re.DOTALL)
_LOCAL_TIME_FAST_PATH = {
    "现在几点",
    "现在几点？",
    "当前时间",
    "当前时间？",
    "今天星期几",
    "今天星期几？",
    "今天周几",
    "今天周几？",
    "今天几号",
    "今天几号？",
    "今天日期",
    "今天日期？",
}


def _rule_route(query: str) -> dict | None:
    """极窄 fast path：只处理无地点的本地时间高频问法。"""
    q = (query or "").strip()
    if not q:
        return {"tool": "NO_TOOL", "args": {}}
    if q in _LOCAL_TIME_FAST_PATH:
        return {"tool": "get_current_time", "args": {}}
    return None


def _extract_json(text: str) -> dict | None:
    """从模型输出里抽出第一个 JSON 对象。"""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.strip().strip("`")
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def route(query: str) -> dict:
    """只做路由决策，返回 {'tool': ..., 'args': {...}}。"""
    # 0) 极窄规则前置路由：仅处理无地点的本地时间高频问法
    rule_decision = _rule_route(query)
    if rule_decision is not None:
        return rule_decision

    prompt = render_prompt(
        "router",
        tool_list=list_tools_brief(),
        query=query,
    )
    # /no_think 是给 qwen3 的兜底标记；其它模型会被忽略，无副作用
    text = llm_complete("router", prompt + " /no_think")

    decision = _extract_json(text) or {"tool": "NO_TOOL", "args": {}}

    name = decision.get("tool", "NO_TOOL")
    args = decision.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    if name != "NO_TOOL" and name not in TOOLS:
        name = "NO_TOOL"
    return {"tool": name, "args": args}


def route_and_call(query: str) -> dict:
    """路由 + 执行工具，返回统一结构（P0-4 改造）。

    返回：
        {
          "tool":       str,            # 工具名 或 "NO_TOOL"
          "args":       dict,
          "ok":         bool,           # 工具是否成功（NO_TOOL 时为 False）
          "result":     Any | None,     # 成功时的工具数据（= call_tool 的 data）
          "error":      str | None,     # 失败原因
          "error_kind": str | None,     # 失败分类（unknown_tool/bad_args/exec_error/empty）
        }

    ════════════════════════════════════════════════════════════════════
    P0-4 关键变更：把「工具是否成功」显式化
    ════════════════════════════════════════════════════════════════════
    改造前直接返回 `call_tool()` 的原始值（失败时是 `{"error": ...}`），
    而 agent 只判断 `result is not None` → 失败也被当成成功，
    于是跳过检索、把错误信息塞进 prompt。

    现在：
      * `ok=True`  → agent 使用工具结果，跳过检索（原有快路径不变）
      * `ok=False` → agent **降级到正常检索通路**，并通过 on_event
                     发出 `tool_failed` 事件供观测。
    `result` 字段保留为"成功时的数据"，语义更干净；旧调用方若只读
    `result`，失败时拿到 None，也不会再把错误文本当资料使用。
    """
    decision = route(query)
    if decision["tool"] == "NO_TOOL":
        return {**decision, "ok": False, "result": None,
                "error": None, "error_kind": None}

    print(f"开始执行工具：{decision['tool']}，参数为：{decision['args']}")
    outcome = call_tool(decision["tool"], decision["args"])

    if not outcome.get("ok"):
        # 只打日志，不把 error 文本放进 result —— 避免它流入 prompt
        print(
            f"[tool_router] 工具 {decision['tool']} 失败"
            f"（kind={outcome.get('kind')}）: {outcome.get('error')}"
            f" → 将降级到通用检索通路"
        )
    return {
        **decision,
        "ok": bool(outcome.get("ok")),
        "result": outcome.get("data"),
        "error": outcome.get("error"),
        "error_kind": outcome.get("kind"),
    }


def format_tool_result(decision: dict) -> str:
    """把工具调用结果格式化为可塞进 prompt 的文本。

    P0-4：只有 `ok=True` 才产出内容。失败时返回空串，让 agent 走检索通路，
    绝不把 `{"error": ...}` 当作"外部资料"喂给 LLM。
    """
    if not decision.get("ok") or decision.get("result") is None:
        return ""
    if decision["tool"] == "NO_TOOL":
        return ""
    head = f"[工具调用] {decision['tool']}({json.dumps(decision['args'], ensure_ascii=False)})"
    body = json.dumps(decision["result"], ensure_ascii=False, indent=2)
    return f"{head}\n{body}"
