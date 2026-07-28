# llm_client.py
"""统一的 LLM 调用封装（支持非流式 / 流式）。

对外四个函数：
    chat(stage, messages, **ov)         → str              # 非流式多轮
    complete(stage, prompt, **ov)       → str              # 非流式单轮
    stream_chat(stage, messages, **ov)  → Iterator[str]    # 流式多轮，逐 token yield
    stream_complete(stage, prompt, **ov)→ Iterator[str]    # 流式单轮
"""
from __future__ import annotations

import os
from typing import Any, Iterator

from configs.models_config import get_stage_config


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
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature},
    }
    kwargs.update(extra or {})
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
        **(extra or {}),
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
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature},
        "stream": True,
    }
    kwargs.update(extra or {})
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
        **(extra or {}),
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
