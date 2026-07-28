# configs/models_config.py
"""每个"阶段"用什么模型 / 什么参数，全部集中在这里。

设计目标（参考 opencode 的多 provider 思路）：
- 每个阶段（router / rewriter / summary / ...）独立配置
- 同一阶段内可切换 provider（ollama 本地 / deepseek 远端 / openai 兼容服务）
- 仅改这一个文件，就能完成"换模型 / 换 provider / 改采样参数"

字段说明（每个 stage 是一个 dict）：
  provider     : "ollama" | "openai"      —— "openai" 泛指所有 OpenAI 兼容 API
  model        : 模型名（例：qwen3:4b / deepseek-chat / gpt-4o-mini）
  base_url     : OpenAI 兼容 endpoint；ollama 用本地 host 即可
  api_key_env  : 从哪个环境变量读 API key（ollama 留空）
  temperature  : 采样温度
  extra        : 透传到 provider 的额外参数，例如 {"think": False}

只需要修改本文件里 STAGES 字段，就能切换：
  - router 用 ollama 还是 deepseek
  - rewriter 用 qwen3 还是 gpt
  - summary 用 deepseek-v4-flash 还是 deepseek-v4-pro
"""
from __future__ import annotations

import os
from typing import Any


# ---------- 共用 endpoint / key 配置 ----------
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY_ENV: str = "DEEPSEEK_API_KEY"

OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY_ENV: str = "OPENAI_API_KEY"


# ---------- 各阶段配置 ----------
# key = stage 名称；约定的 stage：
#   "router"          : 工具路由（tool_router.py）
#   "rewriter"        : Query 改写（query_rewriter.py）
#   "summary"         : 最终回答（agent.py）
#   "fallback_rewriter": 当 rewriter 模型不可用时的兜底（可选）
#
# 想新增阶段直接加 key，业务里用 get_stage_config("xxx") 读取。
STAGES: dict[str, dict[str, Any]] = {
    "router": {
        "provider": "ollama",         # 决策框架
        "model": "qwen3:4b-instruct-2507-q8_0",
        "base_url": OLLAMA_HOST,
        "api_key_env": "",
        "temperature": 0.0,           # 路由要确定性
        "extra": {"think": False},
    },
    "rewriter": {
        "provider": "ollama",
        "model": "qwen3:4b-instruct-2507-q8_0",
        "base_url": OLLAMA_HOST,
        "api_key_env": "",
        "temperature": 0.2,
        "extra": {"think": False},
    },
    "summary": {
        "provider": "openai",         # DeepSeek 走 openai 兼容协议
        "model": "deepseek-v4-flash",
        "base_url": DEEPSEEK_BASE_URL,
        "api_key_env": DEEPSEEK_API_KEY_ENV,
        "temperature": 0.7,
        "extra": {},
    },
    # 示例：想给 summary 阶段挂一个备份 OpenAI 模型，可以这样写：
    # "summary_backup": {
    #     "provider": "openai",
    #     "model": "gpt-4o-mini",
    #     "base_url": OPENAI_BASE_URL,
    #     "api_key_env": OPENAI_API_KEY_ENV,
    #     "temperature": 0.7,
    #     "extra": {},
    # },
}


# ---------- 工具函数 ----------
def get_stage_config(stage: str) -> dict[str, Any]:
    """读取某个 stage 的配置；缺失则报错。"""
    if stage not in STAGES:
        raise KeyError(
            f"未配置的 stage: {stage}。可选: {list(STAGES.keys())}。"
            f"请在 configs.models_config.STAGES 中新增。"
        )
    # 返回浅拷贝，避免外部误改全局配置
    return dict(STAGES[stage])


def override_stage(stage: str, **kwargs: Any) -> None:
    """运行期临时覆盖某个 stage 的字段（一般用于调试/单测）。"""
    if stage not in STAGES:
        STAGES[stage] = {}
    STAGES[stage].update(kwargs)


def list_stages() -> list[str]:
    return list(STAGES.keys())
