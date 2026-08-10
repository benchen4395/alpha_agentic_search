# llm_client.py
"""统一的 LLM 调用封装（支持非流式 / 流式）。

对外四个函数：
    chat(stage, messages, **ov)         → str              # 非流式多轮
    complete(stage, prompt, **ov)       → str              # 非流式单轮
    stream_chat(stage, messages, **ov)  → Iterator[str]    # 流式多轮，逐 token yield
    stream_complete(stage, prompt, **ov)→ Iterator[str]    # 流式单轮

另外提供预热入口（消除本地模型冷加载毛刺）：
    warmup_stage(stage)                 → float | None    # 单个 stage
    warmup_all()                        → dict            # 所有本地 stage

════════════════════════════════════════════════════════════════════════
ollama keep_alive 透传 + 启动预热
════════════════════════════════════════════════════════════════════════
【问题】用户实测「首次工具路由 23s，第二次 3.8s」。本机复测同一 query
连续 3 次：5129ms → 607ms → 558ms，且 `/api/ps` 显示无模型驻留。
结论：这**不是** prompt 或网络问题，纯粹是 ollama 的模型冷加载
（4.3GB 权重 mmap + KV cache 分配 + kernel 预热）。

【解法】两件事同时做，缺一不可：
  1. `keep_alive=-1`（models_config.OLLAMA_KEEP_ALIVE）→ 模型常驻，
     不再因空闲 5 分钟被卸载；
  2. `warmup_all()` 在进程 ready 前打一次 dummy 请求，把加载成本
     提前到"用户还没开始提问"的时刻。

这与 RAG 侧已有的 `LayeredRetriever.warmup()` 是完全对称的设计 ——
之前只预热了 embedding/FAISS/KG，**唯独漏了 LLM**，而 LLM 恰恰是
本机上最大的那块冷启动成本。
"""
from __future__ import annotations

import os
import time
from typing import Any, Iterator

from src.configs.models_config import (
    OLLAMA_KEEP_ALIVE, get_stage_config, local_stages,
)


def _coerce_keep_alive(v: Any) -> Any:
    """把 keep_alive 规范成 ollama 能接受的类型。

    ⚠️ 这是实测踩到的坑（端到端验证时才暴露）：
        ollama._types.ResponseError: time: missing unit in duration "-1"

    ollama 的 `keep_alive` 接受两种形式：
      * **整数** → 秒数；`-1` 表示永不卸载（这就是我们要的）
      * **带单位的字符串** → "5m" / "30s" / "1h"
    而配置来自环境变量，天然是**字符串**。直接把 `"-1"` 发过去，
    服务端会当"缺单位的 duration"解析 → HTTP 400，整个 router 阶段全挂。

    所以这里做一次显式转换：纯数字（含负号）的字符串 → int，
    带单位的字符串原样保留。
    """
    if isinstance(v, str):
        s = v.strip()
        # 允许 "-1" / "0" / "300" 这类纯秒数写法
        if s.lstrip("-").isdigit():
            return int(s)
        return s
    return v


def _with_keep_alive(extra: dict[str, Any]) -> dict[str, Any]:
    """给 ollama 请求补上 `keep_alive`，让模型常驻内存。

    为什么放在这里而不是写进 STAGES["extra"]：
      * `keep_alive` 是**部署策略**（跟机器内存有关），不是模型能力参数，
        不该和 temperature / think 混在同一份模型配置里；
      * 集中在一处注入，新增 stage 时不会漏配；
      * 调用方仍可通过 `extra={"keep_alive": "10m"}` 显式覆盖 ——
        这里用 setdefault，尊重显式传入的值。
    """
    out = dict(extra or {})
    out.setdefault("keep_alive", OLLAMA_KEEP_ALIVE)
    # 无论值来自配置还是调用方，都统一走一次类型规范化
    out["keep_alive"] = _coerce_keep_alive(out["keep_alive"])
    return out


def _ollama_kwargs(
    model: str,
    messages: list[dict],
    temperature: float,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """组装 ollama `client.chat()` 的关键字参数。

    ⚠️ 这里必须**深合并 `options`**，不能用 `kwargs.update(extra)` 直接覆盖。
    踩过的坑：`options` 里既有本函数注入的 `temperature`，也可能有调用方
    通过 `extra={"options": {"num_predict": 1}}` 传进来的采样参数
    （预热就是这么用的）。浅 update 会让**整个 options 字典被替换掉**，
    temperature 静默丢失 —— 路由 stage 的 temperature=0.0（要确定性）
    会被打回模型默认值 0.8，路由结果开始随机抖动，而且没有任何报错。
    """
    ex = _with_keep_alive(extra)
    # 先取出调用方的 options（如果有），与我们的 temperature 合并
    caller_options = dict(ex.pop("options", None) or {})
    options: dict[str, Any] = {"temperature": temperature}
    options.update(caller_options)      # 调用方显式传的优先
    return {
        "model": model,
        "messages": messages,
        "options": options,
        **ex,
    }


def _openai_safe_extra(extra: dict[str, Any]) -> dict[str, Any]:
    """剔除 ollama 私有参数，只保留 OpenAI 兼容端点认得的字段。

    ⚠️ 这是实测踩到的坑（把某个 stage 从 ollama 切到远端时暴露）：
        TypeError: Completions.create() got an unexpected keyword argument 'think'

    成因是配置与 provider 的**耦合**：`extra={"think": False}` 是给
    ollama/qwen3 关思考模式用的，只要把该 stage 的 provider 改成
    openai，这个字段就会被 `**extra` 原样展开进 `create()`，直接 TypeError。

    为什么必须在这里做「白名单过滤」，而不是让使用者改配置：
      * 切 provider 时**只应该改 provider/model/base_url**，不该被迫
        记住"顺手把 extra 里的 ollama 专属字段删掉"——这种隐式约定
        一定会被忘掉，而报错信息（unexpected keyword argument）离
        真正的原因（配置里有个 ollama 专属字段）非常远，很难排查；
      * 反向已经处理过了：`_call_openai` 特意不注入 keep_alive（见下），
        这里只是把同一个原则补齐到"配置里带过来的字段"。

    采用**黑名单**而不是白名单：OpenAI 生态字段很多且在演进
    （max_tokens / top_p / response_format / tools / reasoning_effort …），
    白名单会把合法字段误杀。而 ollama 私有字段是可枚举的小集合。
    """
    dropped = {k: v for k, v in (extra or {}).items() if k in _OLLAMA_ONLY_KEYS}
    if dropped:
        print(f"[llm_client] ℹ️ openai provider 已忽略 ollama 专属参数 "
              f"{sorted(dropped)}（若需生效请改用 ollama provider）")
    return {k: v for k, v in (extra or {}).items() if k not in _OLLAMA_ONLY_KEYS}


# ollama 私有的顶层/专属字段：出现在 OpenAI 兼容请求里会 TypeError 或 400。
#   think      → qwen3 思考模式开关（ollama 扩展）
#   keep_alive → 模型驻留时长（ollama 部署参数）
#   options    → ollama 的采样参数容器（OpenAI 用平铺字段）
_OLLAMA_ONLY_KEYS: frozenset[str] = frozenset({"think", "keep_alive", "options"})


# ============================================================
# 非流式 Provider 适配层
# ============================================================
def _call_ollama(
    model: str,
    messages: list[dict],
    temperature: float,
    extra: dict[str, Any],
    base_url: str | None = None,
) -> str:
    try:
        import ollama
    except ImportError as e:
        raise RuntimeError("使用 ollama provider 需要先 `pip install ollama`") from e

    client = ollama.Client(host=base_url) if base_url else ollama
    # keep_alive 是 ollama /api/chat 的顶层参数（与 model/messages 同级），
    # 不能放进 options —— 放错位置会被静默忽略，模型照旧 5 分钟后卸载。
    kwargs = _ollama_kwargs(model, messages, temperature, extra)
    resp = client.chat(**kwargs)
    return (resp.get("message", {}) or {}).get("content", "") or ""


def _call_openai(
    model: str,
    messages: list[dict],
    temperature: float,
    extra: dict[str, Any],
    base_url: str,
    api_key_env: str,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("使用 openai provider 需要先 `pip install openai`") from e

    api_key = os.environ.get(api_key_env, "") if api_key_env else ""
    if api_key_env and not api_key:
        raise RuntimeError(
            f"未找到 API key：环境变量 {api_key_env} 未设置。"
            f"请 export {api_key_env}=sk-xxx 后重试。"
        )

    client = OpenAI(api_key=api_key or "dummy", base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        # ⚠️ 注意这里**不注入 keep_alive**：它是 ollama 私有参数，
        # OpenAI 兼容端点收到未知字段会直接 400。远端服务的模型常驻
        # 由服务方保证，客户端无需也无权干预。
        # 同理，配置里可能残留的 think/options 也必须先剔除，
        # 否则切 provider 时会报 unexpected keyword argument。
        **_openai_safe_extra(extra),
    )
    return resp.choices[0].message.content or ""


# ============================================================
# 流式 Provider 适配层
# ============================================================
def _stream_ollama(
    model: str,
    messages: list[dict],
    temperature: float,
    extra: dict[str, Any],
    base_url: str | None = None,
) -> Iterator[str]:
    """ollama 流式：逐 chunk yield 增量 token。"""
    try:
        import ollama
    except ImportError as e:
        raise RuntimeError("使用 ollama provider 需要先 `pip install ollama`") from e

    client = ollama.Client(host=base_url) if base_url else ollama
    kwargs = _ollama_kwargs(model, messages, temperature, extra)
    kwargs["stream"] = True
    for chunk in client.chat(**kwargs):
        piece = (chunk.get("message", {}) or {}).get("content", "")
        if piece:
            yield piece


def _stream_openai(
    model: str,
    messages: list[dict],
    temperature: float,
    extra: dict[str, Any],
    base_url: str,
    api_key_env: str,
) -> Iterator[str]:
    """openai 兼容流式：使用 SSE chunk.delta.content。"""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("使用 openai provider 需要先 `pip install openai`") from e

    api_key = os.environ.get(api_key_env, "") if api_key_env else ""
    if api_key_env and not api_key:
        raise RuntimeError(
            f"未找到 API key：环境变量 {api_key_env} 未设置。"
            f"请 export {api_key_env}=sk-xxx 后重试。"
        )

    client = OpenAI(api_key=api_key or "dummy", base_url=base_url)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
        # 同 _call_openai：剔除 ollama 专属参数（think/keep_alive/options）
        **_openai_safe_extra(extra),
    )
    for chunk in stream:
        # chunk.choices 偶尔为空（首/末帧）；安全访问
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        piece = getattr(delta, "content", None) or ""
        if piece:
            yield piece


# ============================================================
# 内部统一调度
# ============================================================
def _resolve_cfg(stage: str, overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = get_stage_config(stage)
    cfg.update(overrides)
    return cfg


# ============================================================
# 对外入口：非流式
# ============================================================
def chat(stage: str, messages: list[dict], **overrides: Any) -> str:
    cfg = _resolve_cfg(stage, overrides)
    provider = cfg["provider"]
    common = dict(
        model=cfg["model"],
        messages=messages,
        temperature=float(cfg.get("temperature", 0.7)),
        extra=dict(cfg.get("extra") or {}),
    )
    if provider == "ollama":
        return _call_ollama(**common, base_url=cfg.get("base_url") or None)
    if provider == "openai":
        return _call_openai(
            **common,
            base_url=cfg.get("base_url") or "https://api.openai.com/v1",
            api_key_env=cfg.get("api_key_env") or "",
        )
    raise ValueError(f"未知 provider: {provider}（当前仅支持 ollama / openai）")


def complete(stage: str, prompt: str, **overrides: Any) -> str:
    return chat(stage, [{"role": "user", "content": prompt}], **overrides)


# ============================================================
# 对外入口：流式
# ============================================================
def stream_chat(stage: str, messages: list[dict], **overrides: Any) -> Iterator[str]:
    """返回逐 token 的生成器。调用方 ``for piece in stream_chat(...): ...``。"""
    cfg = _resolve_cfg(stage, overrides)
    provider = cfg["provider"]
    common = dict(
        model=cfg["model"],
        messages=messages,
        temperature=float(cfg.get("temperature", 0.7)),
        extra=dict(cfg.get("extra") or {}),
    )
    if provider == "ollama":
        return _stream_ollama(**common, base_url=cfg.get("base_url") or None)
    if provider == "openai":
        return _stream_openai(
            **common,
            base_url=cfg.get("base_url") or "https://api.openai.com/v1",
            api_key_env=cfg.get("api_key_env") or "",
        )
    raise ValueError(f"未知 provider: {provider}（当前仅支持 ollama / openai）")


def stream_complete(stage: str, prompt: str, **overrides: Any) -> Iterator[str]:
    return stream_chat(stage, [{"role": "user", "content": prompt}], **overrides)


# ============================================================
# 对外入口：预热（warmup）
# ============================================================
# 预热探针：只需让请求量级接近真实 prompt（约 300 token）。
#
# 关于“为什么预热之后首调还是慢”，实测推翻过三个错误假设，记在
# 这里以免后人（包括未来的自己）重蹈：
#   ✗ prompt 太短？  → 改成 900 字符同量级，毛刺照旧（1672ms）
#   ✗ num_predict=1？→ 改成完整解码，毛刺照旧（2526ms）
#   ✗ 差 `/no_think`？→ 补上后缀，毛刺照旧（1591ms）
#   ✓ 真实原因：**warmup_all 逐个预热时，后一个 stage 会顶掉前面
#      stage 的预热成果** —— 详见 `warmup_all` 的说明。
#
# 定位方法（而不是继续猜）：受控对比“只预热 router” vs
# “warmup_all”，两者唯一差异就是多预热了一个 rewriter：
#     仅预热 router : route  799 → 769 → 769 → 760 ms（毛刺 39ms）
#     warmup_all     : route 2353 → 770 → 761 → 776 ms（毛刺 1591ms）
_WARMUP_PROMPT_CHARS = 900


def _warmup_prompt(stage: str) -> str:
    """构造预热 prompt：优先复用真实模板，使量级与线上一致。

    渲染失败时退化成等长填充串 —— 预热不应因模板变动而崩掉。
    """
    try:
        if stage == "router":
            from src.configs.prompts import render as _render
            from src.tools import list_tools_brief as _brief
            body = _render("router", tool_list=_brief(), query="预热")
            if body:
                return body
    except Exception:
        pass
    return "预热" * (_WARMUP_PROMPT_CHARS // 2)


def warmup_stage(stage: str, verbose: bool = True) -> float | None:
    """预热单个 stage 的模型，返回耗时（秒）；失败返回 None。

    做法：发一条与真实请求同量级的请求，触发权重加载 + kernel 编译：
      * prompt 复用真实模板（约 300 token）；
      * `num_predict=1` → 只解码 1 个 token 就停，不浪费解码算力
        （实测：改成完整解码对消除毛刺没有任何帮助）。

    ⚠️ **单独调用本函数只能保证该 stage 本身的首调变快**。多个
    stage 共用同一模型但采样参数不同时，后预热的会顶掉先预热的，
    所以多 stage 场景请用 `warmup_all()`（它做了回扫补偿）。

    Args:
        stage:   models_config.STAGES 里的 stage 名。
        verbose: 是否打印每个 stage 的预热耗时。
    Returns:
        耗时（秒）；该 stage 不存在 / 模型拉不起来时返回 None
        （**不抛异常** —— 预热失败不应阻止服务启动，用户仍可正常提问，
         只是首条会慢一点）。
    """
    t0 = time.perf_counter()
    try:
        cfg = get_stage_config(stage)
        # 远端 openai 兼容服务无"客户端可控的模型加载"，预热没有意义，
        # 且 `options` 是 ollama 私有字段，传过去会 400。直接跳过。
        if cfg.get("provider") != "ollama":
            if verbose:
                print(f"[llm warmup]   {stage:<10s}: 跳过（远端 provider="
                      f"{cfg.get('provider')}，模型常驻由服务方保证）")
            return None
        # num_predict=1：把解码成本压到最低（实测不影响预热效果）
        chat(
            stage,
            [{"role": "user", "content": _warmup_prompt(stage)}],
            extra={"options": {"num_predict": 1}},
        )
    except Exception as e:
        if verbose:
            print(f"[llm warmup]   {stage:<10s}: 跳过（{type(e).__name__}: {e}）")
        return None
    dt = time.perf_counter() - t0
    if verbose:
        print(f"[llm warmup]   {stage:<10s}: {dt:.2f} s")
    return dt


def warmup_all(verbose: bool = True) -> dict[str, float | None]:
    """预热所有跑在本地 ollama 上的 stage。

    只预热本地 stage（见 `models_config.local_stages()`）：
      * 远端 openai 兼容服务（DeepSeek 等）没有"客户端可控的模型加载"，
        预热只会白花一次 token 和 API 配额；
      * 而本地 ollama 的加载成本是实打实的 5~23s，必须提前付掉。

    ══════════════════════════════════════════════════════════
    ⚠️ 为什么末尾要「回扫」重新预热第一个 stage
    ══════════════════════════════════════════════════════════
    这是实测出来的一个反直觉行为：多个 stage 共用同一模型但采样参数
    不同时（本项目 router temperature=0.0、rewriter=0.2，同为
    qwen3:4b-instruct），**后预热的 stage 会顶掉先预热的**：

        仅预热 router               : route  799 → 769 → 769 → 760 ms
                                          ↑ 首调即稳态（毛刺 39ms）
        预热 router 再预热 rewriter : route 2353 → 770 → 761 → 776 ms
                                          ↑ router 的预热被顶掉了（1591ms）

    而 router 是每条 query 的**第一个**环节，所以这个毛刺 100%
    落在用户感知最强的位置。回扫一次把它拉回稳态。

    ⚠️ 这个互顶**无法靠预热策略消除**，是 ollama 的固有约束：
    实测「交替预热 router/rewriter 各 3 轮」不但没让两者都热，反而
    让 router 稳态从 ~770ms 退化到 ~2200ms（反复切换采样参数导致
    运行态持续重建）。所以问题不是"能否都热"，而是"毛刺留给谁"。

    用真实链路的**首条合计延迟**裁决（router → rewriter 两环节，
    各取 3 次中位数），而不是凭直觉：

        不回扫: router 2179 + rewriter 1610 = 3789 ms
        回扫  : router  616 + rewriter 1678 = 2294 ms   ← 净省约 1.5s

    注意这不是"把毛刺从 router 转移给 rewriter"的零和交换 ——
    rewriter 无论如何都要付一次参数切换成本（1610 → 1678ms 几乎
    没变），回扫是让 router 额外**免费**拿回了热态。

    之前漏掉这个问题的原因：单测 `warmup_stage("router")` 看起来完全
    正常，只有走完整的 `warmup_all()` 才会复现 —— 而两个入口
    （main.py / main_web.py）用的都是后者。

    它不做 stage 去重：日志里每个 stage 的真实首调耗时有诊断价值
    （能看出是否真的复用上了已驻留的模型）。

    Returns:
        {stage: 耗时秒 | None}，便于记录到启动日志 / 监控。
        回扫那一次不计入返回值（它不代表新 stage）。
    """
    timing: dict[str, float | None] = {}
    stages = local_stages()
    if not stages:
        return timing
    if verbose:
        print(f"[llm warmup] start (本地 stage: {stages}, "
              f"keep_alive={OLLAMA_KEEP_ALIVE}) ...")
    t_all = time.perf_counter()
    for s in stages:
        timing[s] = warmup_stage(s, verbose=verbose)

    # 回扫：把被后续 stage 顶掉的第一个 stage 重新拉回稳态。
    # 只在真的有多个成功预热的 stage 时才需要（单 stage 不存在互顶）。
    warmed = [s for s in stages if timing.get(s) is not None]
    if len(warmed) > 1:
        first = warmed[0]
        t_re = warmup_stage(first, verbose=False)
        if verbose:
            extra = f"{t_re:.2f} s" if t_re is not None else "失败"
            print(f"[llm warmup]   回扫 {first:<5s}: {extra}"
                  f"（被后续 stage 顶掉，重新拉回稳态）")

    if verbose:
        print(f"[llm warmup] done in {time.perf_counter() - t_all:.2f} s")
    return timing
