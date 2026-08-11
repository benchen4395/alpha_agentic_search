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

════════════════════════════════════════════════════════════════════════
性能优化（本次）：ollama 的模型驻留（keep_alive）
════════════════════════════════════════════════════════════════════════
【实测现象】用户反馈「第一次工具路由 23s，第二次 3.8s」。
本机复测（`tool_router.route` 连续 3 次，同一 query）：

    第 1 次: 5129 ms      ← 冷启动
    第 2 次:  607 ms
    第 3 次:  558 ms

差了约 8~9 倍，且 `curl http://127.0.0.1:11434/api/ps` 返回
`{"models":[]}` —— **ollama 里一个模型都没驻留**。

【根因】ollama 的模型生命周期是「按需加载 + 空闲卸载」：
  1. 收到请求时若模型不在显存/内存 → 从磁盘 mmap 权重、建 KV cache、
     跑一遍 prefill 预热 kernel。qwen3:4b-q8_0 约 4.3GB，在 Mac 上
     这一步就是 5~20s（用户机器上是 23s，取决于磁盘与内存压力）。
  2. 默认 `keep_alive="5m"`：**空闲 5 分钟后自动卸载**。
     所以哪怕已经预热过，只要用户思考/离开超过 5 分钟，
     下一条问题又会重新付一次冷加载。

这解释了用户观测的全部现象：首次慢是①，第二次快是模型已驻留。

【解法（业界通用两件套）】
  * **常驻**：显式把 `keep_alive` 设为 `-1`（永不卸载，直到 ollama 退出）。
    这是 ollama 官方推荐的「服务化部署」配置，等价于 vLLM/TGI 的
    常驻 worker 模型 —— 推理服务的模型加载本就该是**启动期一次性成本**，
    而不是分摊到每个请求头上。
  * **预热**：进程启动时对每个 stage 打一次极短的 dummy 请求
    （见 `llm_client.warmup_stage`），把①提前到用户输入之前。

  两者必须**同时**做，缺一不可：
    - 只预热不常驻 → 5 分钟后打回原形；
    - 只常驻不预热 → 首条问题仍要等冷加载。

可用 `LLM_KEEP_ALIVE` 覆盖（如内存紧张的机器可设 "30m"）：
    export LLM_KEEP_ALIVE=30m
"""
from __future__ import annotations

import os
from typing import Any


# ---------- 共用 endpoint / key 配置 ----------
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# ---------- ollama 模型驻留时长 ----------
# "-1"  → 永不卸载（推荐：把模型加载变成启动期一次性成本）
# "30m" → 空闲 30 分钟后卸载（内存紧张时的折中）
# "5m"  → ollama 默认值（不推荐：交互式使用几乎必然反复冷加载）
#
# ⚠️ 只对 provider="ollama" 生效；openai 兼容的远端服务由服务方管理驻留，
#    传这个参数会被拒（见 llm_client._call_openai 的注释）。
OLLAMA_KEEP_ALIVE: str = os.getenv("LLM_KEEP_ALIVE", "-1")

# ---------- 单次 LLM 请求的超时上限（秒） ----------
# ⚠️ 这个值不是"调优参数"，是**故障隔离**参数。
#
# 【实测故障】跑 BrowseComp-ZH 评测时，summary 阶段单次调用卡了
# **444 秒**才返回（长题干 + 6 段网页证据 → 模型写超长带引用回答）。
# 在此之前 openai / ollama 两个 provider 都**没有传 timeout**，
# 而 openai SDK 的默认值是 600s、ollama 更是不限 —— 等价于"永不超时"。
# 用户侧的体验就是整个问答卡死，且没有任何日志能解释原因。
#
# 【为什么 600s 默认值等于没有】交互式问答的可接受上限是十几秒量级，
# 一个 10 分钟才失败的请求，在用户放弃之前根本不会触发。超时的意义
# 是"尽早把控制权交回调用方"，而不是"兜住理论最坏情况"。
#
# 【定值依据】实测 LLM 延迟由**输出**长度主导，与输入长度几乎无关：
#     prompt≈200 字 → 3.8s ｜ ≈2000 字 → 1.9s ｜ ≈6000 字 → 1.7s
# 正常回答（含引用 + 追问推荐）在 3~30s，联网证据多时可到 60s。
# 取 90s 给出约 3 倍余量：既能兜住正常的长回答，又能在真出问题时
# 及时失败，而不是让用户干等 7 分钟。
#
# 流式（stream_chat）另设更宽的上限：流式是逐 token 返回，用户第一个
# 字出来得很快，总时长长一些不影响体感，卡死风险也低得多。
LLM_TIMEOUT_SEC: float = float(os.getenv("LLM_TIMEOUT_SEC", "90"))
LLM_STREAM_TIMEOUT_SEC: float = float(os.getenv("LLM_STREAM_TIMEOUT_SEC", "180"))

# ---------- 单次请求的重试次数（仅 openai 兼容 provider） ----------
# openai SDK 默认 max_retries=2，会对连接错误/5xx/429 自动重试。
# ⚠️ 重试与超时是**乘法关系**：默认值下最坏耗时是 3×timeout，
# 90s 的超时会变成 270s，等于把刚加的超时又废掉一半。
# 取 1 是折中：既保留一次机会兜住偶发网络抖动（本项目实测遇到过
# APIConnectionError: nodename nor servname provided），
# 又把最坏耗时限制在 2×timeout。
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "1"))

DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY_ENV: str = "DEEPSEEK_API_KEY"

OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY_ENV: str = "OPENAI_API_KEY"

TOKENVERSE_BASE_URL: str = os.getenv("TOKENVERSE_BASE_URL", "https://tokenverse.corp.kuaishou.com/v1")
TOKENVERSE_API_KEY_ENV: str = "TOKENVERSE_API_KEY"


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
    # "rewriter": {
    #     "provider": "ollama",
    #     "model": "qwen3:4b-instruct-2507-q8_0",
    #     "base_url": OLLAMA_HOST,
    #     "api_key_env": "",
    #     "temperature": 0.2,
    #     "extra": {"think": False},
    # },
    "rewriter": {                   # 可切换成tokenverse方式
        "provider": "openai",
        "model": "deepseek-v4-pro",
        "base_url": TOKENVERSE_BASE_URL,
        "api_key_env": TOKENVERSE_API_KEY_ENV,
        "temperature": 0.2,
        # 这里**故意留空**：`think` 是 ollama/qwen3 专属参数。
        # 虽然 llm_client._openai_safe_extra() 已经会兜底剔除它，
        # 但配置层就不该带上与当前 provider 无关的字段。
        "extra": {},
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


def local_stages() -> list[str]:
    """列出所有跑在**本地 ollama** 上的 stage（即需要预热的那些）。

    远端 openai 兼容服务（DeepSeek / OpenAI）的模型常驻由服务方保证，
    客户端预热只会白花一次 token，所以这里只筛 provider == "ollama"。
    """
    return [s for s, cfg in STAGES.items() if cfg.get("provider") == "ollama"]
