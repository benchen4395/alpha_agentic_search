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
# src/configs/ → src/ → 项目根：搬进 src/ 后是**两级**，少一级会把
# data/ 解析成 src/data/，导致缓存与索引全部找不到。
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(_CONFIGS_DIR))
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

# 单条 value 超过此字节数时，diskcache 不再内联进 cache.db，而是**外溢成独立文件**，
# 路径形如 data/search_cache/58/6b/bb62f9....val（32 位随机 hex 切成 2/2/28 三段，
# 目的是避免单目录堆积过多文件；目录名与 query 无关，纯随机）。
#
# 【为什么要把默认 32KB 抬到 1MB】
# 这套外溢机制对本项目是纯负担，有三个具体问题：
#   ① 一致性风险：cache.db 是被 Git 追踪的（见 .gitignore 说明，为了让 clone
#      即带一批预热缓存），但随机生成的外溢目录不在追踪列表里。只提交 cache.db
#      时，库里那行 filename 指针指向的 .val 在别人机器上并不存在 —— 命中该 key
#      不是缓存 miss，而是读文件直接抛异常。
#   ② 目录污染：每条超阈值的 value 都会随机出现一个新的两级目录，持续制造
#      untracked 噪声，且无法用固定路径写进 .gitignore。
#   ③ 收益为零：搜索结果实测 avg ≈ 11KB、max ≈ 26KB，仅个别长表格页面
#      （实测 54800 字节）会越过 32KB。为这种个例引入上述两个问题不值得。
#
# 抬到 1MB 后，所有搜索结果都内联进单个 cache.db，目录结构恒定干净。
# SQLite 存储 1MB 以内的 BLOB 性能良好，不构成瓶颈。
SEARCH_CACHE_MIN_FILE_SIZE: int = int(os.getenv("SEARCH_CACHE_MIN_FILE_SIZE", str(1 << 20)))  # 1MB


# ============== DuckDuckGo ==============
DDG_BACKENDS: list[str] = ["api", "html", "lite"]
# wt-wt = Worldwide（无地域偏好）
DDG_REGIONS: list[str] = ["wt-wt", "us-en", "cn-zh"]
# ⚡ 重试与超时的收敛（本次性能优化）
#
# 【实测数据】同一 query 绕过缓存实测两次：37957ms / 16412ms。
# 而离线三层（L2+L3+L5）合计只要约 0.5s、聚合置信度已达 0.99 ——
# 也就是说这十几到几十秒的联网等待，对答案质量的贡献是 0。
#
# 【最坏情况推算（未收敛超时与重试时）】
#   DDG_MAX_RETRIES=3 × DDG_TIMEOUT=15s + 2 次 sleep(1.0~2.5s)
#   ≈ 45 + 5 = 50s，**仅 DDG 一路**；之后还要继续尝试
#   Tavily → Serper → Bing。整个 web_search 理论最坏值超过一分钟。
#
# 【调整依据】搜索引擎的响应时间分布是重尾的：正常响应在 1~3s 内，
#   超过 8s 的基本是被限流或网络异常 —— 这种情况**继续等下去的期望
#   收益极低**，换个 backend 重试的成功率反而更高。
#   所以：单次超时 15s → 8s（快速失败），重试 3 → 2 次（换 backend 更划算），
#   重试间隔也相应收敛。
#   最坏 DDG 耗时降到 ≈ 8×2 + 1 = 17s，再叠加 rag/config.L4_TIMEOUT_SEC
#   的硬预算（8s），前台延迟被彻底封顶。
DDG_MAX_RETRIES: int = int(os.getenv("DDG_MAX_RETRIES", "2"))
DDG_TIMEOUT: int = int(os.getenv("DDG_TIMEOUT", "8"))
# 重试之间的退避区间（秒）。原来固定 1.0 + random()*1.5（最多 2.5s），
# 在只重试 2 次的新配置下收敛到最多 1.0s，避免退避本身成为延迟大头。
DDG_RETRY_BACKOFF_MIN: float = float(os.getenv("DDG_RETRY_BACKOFF_MIN", "0.3"))
DDG_RETRY_BACKOFF_MAX: float = float(os.getenv("DDG_RETRY_BACKOFF_MAX", "1.0"))


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

