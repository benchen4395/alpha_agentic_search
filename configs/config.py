# configs/config.py
"""项目集中配置。

所有可调字段集中在此文件，便于修改/版本管理。
环境变量优先级高于此处默认值（保留向后兼容）。
"""
from __future__ import annotations

import os


# ============== 统一数据目录（data/） ==============
# 本项目所有本地落盘数据（缓存 / RAG 知识库 / 历史归档）统一收纳到
# 项目根的 data/ 目录下，结构一目了然：
#
#   data/
#   ├── rag_data/      ── RAG 知识库：L2 Wiki 索引、L5 KG 图谱、L3 历史归档（详见 rag/config.py）
#   ├── qa_cache/      ── L1 Q&A 缓存（diskcache）：高频问答精确/模糊命中，毫秒级返回
#   └── search_cache/  ── 联网搜索结果缓存（diskcache）：避免重复请求搜索引擎
#
# 设计要点：
#   - 以项目根为锚点（本文件位于 configs/，上一级即项目根），保证无论从哪个 CWD
#     启动（main / main_web / scripts）都能定位到同一份 data/。
#   - 支持环境变量 DATA_DIR 整体覆盖（例如挂载到外置 SSD / 数据盘）。
#   - 各子项也各自保留独立环境变量（QA_CACHE_DIR / SEARCH_CACHE_DIR / RAG_DATA_DIR），
#     优先级高于 DATA_DIR，便于单独重定向某一块。
_CONFIGS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT: str = os.path.dirname(_CONFIGS_DIR)          # configs/ 的上一级 = 项目根
DATA_DIR: str = os.getenv("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))


# ============== 搜索引擎 API Keys ==============
# Tavily: https://tavily.com  （免费 1000 次/月）
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# Serper.dev: https://serper.dev （免费 2500 次）
SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")


# ============== 代理 ==============
# 当 web_search(use_proxy=True) 时使用此代理
# 例如："socks5://127.0.0.1:1080" 或 "http://127.0.0.1:7890"
SEARCH_PROXY: str = os.getenv("SEARCH_PROXY", "http://127.0.0.1:7890")

# web_search 默认是否启用代理（也可在调用时通过 use_proxy 参数覆盖）
DEFAULT_USE_PROXY: bool = False


# ============== 缓存：联网搜索结果（searcher.py） ==============
# 作用：缓存 web_search 的原始返回结果（DDG/Tavily/Serper 等），避免短时间内
#       重复请求同一 query 浪费配额与延迟。后端为 diskcache。
# 位置：data/search_cache/（继承自旧的 .search_cache）
SEARCH_CACHE_DIR: str = os.getenv("SEARCH_CACHE_DIR", os.path.join(DATA_DIR, "search_cache"))
SEARCH_CACHE_TTL: int = int(os.getenv("SEARCH_CACHE_TTL", str(10 * 24 * 3600)))  # 10 天
SEARCH_CACHE_ENABLED: bool = True   # 是否开启缓存


# ============== DuckDuckGo ==============
DDG_BACKENDS: list[str] = ["api", "html", "lite"]
# wt-wt = Worldwide（无地域偏好）
DDG_REGIONS: list[str] = ["wt-wt", "us-en", "cn-zh"]
DDG_MAX_RETRIES: int = 3
DDG_TIMEOUT: int = 15


# ============== 通用搜索 ==============
SEARCH_DEFAULT_TOP_K: int = 5   # 选取top-k的检索结果
SEARCH_LONG_QUERY_THRESHOLD: int = 8  # 超过此词数将触发短词降级


# ============== LLM 相关说明 ==============
# 各阶段（router / rewriter / summary）用什么 provider / model / 采样参数，
# 统一由 models_config.STAGES 管理，这里不再重复配置模型名，避免"两处配置"打架。
# 当模型判定无需联网时返回的哨兵值
NO_SEARCH_SENTINEL: str = "NO_SEARCH"

# ============== Query 改写策略 ==============
# 0 = 仅规则 (shorten_query)
# 1 = 仅 LLM  (rewrite_query)
# 2 = 混合：LLM 优先，失败回退到规则 (query_rewrite_route) —— 生产推荐
QUERY_REWRITE_TYPE: int = 2

# ============== Tools ==============
# GitHub Token（可选）：匿名 60 次/小时，带 token 5000 次/小时
# 仅需 public_repo 权限
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")


# ============== Q&A 预设缓存 (qa_cache.py) ==============
# Q&A 缓存用于"通用 / 自我介绍 / 高频问题"的精准回答；命中后绕过
# 工具路由 / Query 改写 / 联网检索 / LLM 调用，毫秒级返回。
#
# 支持三种后端：
#   "memory"    : 进程内 dict（最快，不持久化，重启丢失）
#   "diskcache" : 本地磁盘（持久化、跨进程共享，无需服务）—— 推荐单机部署
#   "redis"     : Redis 服务（多实例共享 + 支持 TTL）        —— 推荐生产部署

# AgenticSearchAgent 默认使用的缓存后端
QA_CACHE_BACKEND: str = os.getenv("QA_CACHE_BACKEND", "diskcache")

# DiskCache 数据目录（仅当 backend="diskcache" 时生效）
# 作用：L1 Q&A 缓存落盘位置（精确 + BGE-M3 模糊命中）。
# 位置：data/qa_cache/（继承自旧的 .qa_cache）
QA_CACHE_DIR: str = os.getenv("QA_CACHE_DIR", os.path.join(DATA_DIR, "qa_cache"))

# Redis 连接 URL（仅当 backend="redis" 时生效）
# 格式：redis://[username:password@]host:port/db
QA_REDIS_URL: str = os.getenv("QA_REDIS_URL", "redis://localhost:6379/0")

# Q&A 条目过期时间（秒）；None = 永不过期。memory 后端会忽略此项
# 默认 30 天，可按业务需求调整（如新闻类用 1 天，常识类设 None）
QA_CACHE_TTL: int | None = (
    int(os.getenv("QA_CACHE_TTL", str(30 * 24 * 3600)))
    if os.getenv("QA_CACHE_TTL", "30").lower() not in ("none", "0", "")
    else None
)

