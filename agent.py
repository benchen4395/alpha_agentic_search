# agent.py
"""主控 Agent：统一通过 llm_client + models_config + prompts 调用各阶段 LLM。

链路：
    Step 0  Q&A 缓存命中 → 直接返回
    Step 1  工具路由 (天气 / GitHub / arXiv …)      ← stage="router"
    Step 2  Query 改写 + 分层 RAG 检索               ← stage="rewriter"
    Step 3  生成最终回答                             ← stage="summary"

任何阶段想换模型 / 换 provider / 改 prompt，都不需要动这个文件，分别去：
    - models_config.STAGES["<stage>"]
    - prompts.PROMPTS["<key>"]

════════════════════════════════════════════════════════════════════════
P0 / P0.5 改造总览（本次）
════════════════════════════════════════════════════════════════════════
* **P0-1 L1 缓存准入策略**（`cache_policy.py`）
  `_archive_if_enabled()` 不再无条件写 L1。先过
  `decide_cacheability()`：时效类拒收、拒答类拒收、太短拒收，
  其余按 web兜底/易变槽位/常识 给 6h / 24h / 30d 分级 TTL。
  同时 fuzzy 阈值 0.8 → 0.93，并在 QACache 内加槽位一致性门禁。

* **P0-2 跨层分数校准**（`rag/calibration.py`）
  修掉 L5 的 `or 0.9` bug，各层原始分统一映射到 P(relevant)，
  L4 兜底与 abstention 都基于校准后的聚合置信度判定。

* **P0-3 session/user 隔离**
  `chat(..., session_id=..., user_id=...)`：
    - memory  → 按 session_id 分桶（`_memories` dict）
    - L1/L3   → 按 namespace 隔离（user_id 优先，其次 session_id）
  不传时行为与改造前**完全一致**（单例 memory + 全局 namespace）。

* **P0-4 工具失败降级 + 证据 <doc> 隔离**
  工具返回 `ok=False` 时**降级到检索通路**（而不是把 error 当资料）；
  所有外部资料经 `evidence.build_evidence_block()` 清洗 + `<doc>` 定界。

* **P0.5 AnswerResult（sources + citations）**
  `chat(..., return_result=True)` 返回 `AnswerResult`，携带
  sources / citations / confidence / trace。默认仍返回 `str`，
  既有调用方（scripts/search.py 等）零改动。
"""
from configs import config

from llm_client import chat as llm_chat, stream_chat as llm_stream_chat
# LLM 预热：把本地 ollama 的模型加载成本移到启动期（见 warmup() 的说明）
from llm_client import warmup_all as llm_warmup_all
from memory import ConversationMemory
from configs.prompts import PROMPTS
from qa_cache import QACache
from query_rewriter import query_rewrite_route
from searcher import web_search, format_results
from tool_router import route_and_call, format_tool_result

# ---- P0-1：L1 缓存准入策略（时效判定 / 分级 TTL / 低质答案识别）----
from cache_policy import decide_cacheability

# ---- P0-4：证据清洗与 <doc> 结构化定界（Prompt Injection 防护）----
from evidence import build_evidence_block, build_user_message

# ---- P0.5：结构化答案契约（sources / citations / confidence）----
from answer_types import (
    AnswerResult, Source, StreamingAnswer, parse_citations,
)

# ---- 分层记忆 RAG（rag/）----
# 走 LayeredRetriever：L1 QACache → L2 Wiki → L3 History → L5 KG (并行)
# → 离线不达标才补 L4 Web。相比原来的裸 web_search，可命中常识/历史，越用越强。
from rag import LayeredRetriever

from typing import Iterator, Union, Callable, Optional


# 默认 session：不传 session_id 时所有请求共用这一个桶，
# 行为与改造前（单个 self.memory）完全一致。
_DEFAULT_SESSION = "__default__"


def _stream_text(text: str, chunk_size: int = 2, delay: float = 0.018) -> Iterator[str]:
    """把一段已就绪的完整文本切成小片，模拟流式逐字吐出。

    用于 L1 缓存命中 / 确定性工具命中这类"答案已经拿到、无需再调 LLM"的短路
    路径：这些路径本没有 token 流，若直接 ``iter([text])`` 一次性返回整段，CLI/Web
    在流式模式下就会瞬间刷出全部内容，看起来"不是流式"。这里按 ``chunk_size``
    个字符切片逐个 yield，视觉上与真实 LLM 流式一致（打字机效果）。

    Args:
        text:       完整答案。
        chunk_size: 每次吐出的字符数（默认 2；中文按字符切，1~3 都自然）。
        delay:      每chunk之间的 sleep 秒数（默认 0.018s ≈ 55 字/秒的打字机节奏）。
                    ⚠️ 必须 > 0，否则所有切片会瞬间 yield 完、终端一次性刷出，
                    视觉上仍是"立即输出"。设 0 可用于单测（只验证切片数、不拖慢）。
    """
    if not text:
        return
    if delay > 0:
        import time as _t
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
            _t.sleep(delay)
    else:
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]


class AgenticSearchAgent:
    def __init__(
        self,
        max_memory_turns: int = 8,            # 最多保留8轮记忆内容
        top_k: int = 5,                       # rag-search，每次最多保留5个检索结果
        enable_tools: bool = True,            # 默认开启工具筛选
        rewrite_type: int | None = None,      # 0: 规则，1：LLM，2：混合（LLM->规则）
        # ---- Q&A 缓存配置（None = 走 config.py 默认值） ----
        qa_cache: QACache | None = None,
        qa_cache_backend: str | None = None,   # memory | diskcache | redis
        qa_cache_dir: str | None = None,
        qa_redis_url: str | None = None,
        qa_cache_ttl: int | None = None,
        # ---- 分层 RAG 配置 ----
        enable_rag: bool = True,           # True: 走 LayeredRetriever; False: 退回裸 web_search
        rag_strategy: str | None = None,   # hybrid / offline_only / web_only
        # ---- P0-1：L1 缓存准入策略 ----
        enable_cache_policy: bool = True,  # False 则退回"无条件写 L1"的旧行为（仅用于 A/B）
        # ---- P0-4：证据结构化封装 ----
        enable_evidence_guard: bool = True,  # False 则退回裸文本拼接（仅用于 A/B）
    ):
        # ⚠️ 模型选择由 models_config.STAGES["summary"] 决定
        # 调用方需要换模型/provider 直接改 models_config.py 即可
        self.top_k = top_k
        self.enable_tools = enable_tools
        self.rewrite_type = (
            rewrite_type if rewrite_type is not None else config.QUERY_REWRITE_TYPE
        )
        self.max_memory_turns = max_memory_turns
        self.enable_cache_policy = enable_cache_policy
        self.enable_evidence_guard = enable_evidence_guard

        # ══════════════════════════════════════════════════════════════════
        # P0-3：按 session 隔离的会话记忆
        # ══════════════════════════════════════════════════════════════════
        # 【改造前的问题】
        #   `self.memory` 是**实例级**的单个 ConversationMemory，而
        #   `main_web.py` 全进程只创建一个 agent 供所有浏览器会话共享：
        #       if agent is None: agent = AgenticSearchAgent()
        #   于是 A 用户的对话历史会进入 B 用户的 rewriter 上下文和
        #   summary messages —— 多用户直接串味，且属于隐私泄漏。
        #
        # 【改造后】
        #   `_memories: dict[session_id, ConversationMemory]` 按需创建。
        #   不传 session_id → 落到 `_DEFAULT_SESSION` 桶，
        #   行为与改造前完全一致（CLI 单用户场景无感）。
        #
        #   ⚠️ 内存增长：长期运行的 Web 服务会累积 session。
        #   当前用 `max_sessions` 做 FIFO 淘汰（够用且零依赖）；
        #   生产环境建议换成带 TTL 的 LRU 或外部存储（Redis）。
        self._memories: dict[str, ConversationMemory] = {}
        self.max_sessions: int = 512

        # 兼容属性：保留 `agent.memory` 让既有代码（如 reset()、测试）继续可用，
        # 它指向默认 session 的 memory。
        self.memory = self._get_memory(_DEFAULT_SESSION)

        # Q&A 缓存：支持外部注入或按参数自建
        # P0-1：fuzzy_threshold 不再硬编码 0.8——传 None 让 QACache 采用
        # cache_policy.FUZZY_THRESHOLD（0.93），并默认开启槽位门禁。
        self.qa_cache: QACache = qa_cache or QACache(
            backend=qa_cache_backend,   # 如果启用多级缓存，可使用: layers=["diskcache", "redis"]
            cache_dir=qa_cache_dir,
            redis_url=qa_redis_url,
            ttl=qa_cache_ttl,
            enable_fuzzy=True,          # 先精准匹配，再模糊匹配
            fuzzy_threshold=None,       # None → 采用 cache_policy.FUZZY_THRESHOLD (0.93)
            enable_slot_gate=True,      # P0-1：槽位一致性门禁（核心防线）
        )

        # 分层 RAG：把 qa_cache 作为 L1 复用（enable_rag=False 可退回裸 web_search）
        self.retriever = None
        if enable_rag:
            self.retriever = LayeredRetriever(
                qa_cache=self.qa_cache,
                strategy=rag_strategy,
            )

    # --------------------------------------------------------------------- #
    # P0-3：session / namespace 管理
    # --------------------------------------------------------------------- #
    def _get_memory(self, session_id: str) -> ConversationMemory:
        """取（或创建）某个 session 的会话记忆。

        FIFO 淘汰：超过 `max_sessions` 时丢弃最早创建的 session。
        Python 3.7+ 的 dict 保序，所以 `next(iter(...))` 就是最老的 key。
        选 FIFO 而非 LRU 是因为实现零依赖且对聊天场景足够
        （真正活跃的 session 通常远少于 512）。
        """
        mem = self._memories.get(session_id)
        if mem is None:
            if len(self._memories) >= self.max_sessions:
                oldest = next(iter(self._memories))
                self._memories.pop(oldest, None)
            mem = ConversationMemory(max_turns=self.max_memory_turns)
            self._memories[session_id] = mem
        return mem

    @staticmethod
    def _resolve_namespace(
        user_id: Optional[str], session_id: Optional[str],
    ) -> Optional[str]:
        """决定 L1/L3 的隔离命名空间。

        优先级与语义：
          1. `user_id` 存在 → `"u:<user_id>"`
             用户级隔离：跨 session 复用同一份缓存/历史（登录用户的最佳选择，
             既隔离他人，又保留"越用越强"的个人积累）。
          2. 否则 `session_id` 存在且非默认 → `"s:<session_id>"`
             会话级隔离：匿名访客场景，会话结束即自然失效。
          3. 都没有 → `None`
             全局共享，与改造前**完全一致**（CLI 单用户 / 单测场景）。

        为什么加 `u:` / `s:` 前缀：避免 user_id="42" 与 session_id="42"
        撞到同一个 namespace。
        """
        if user_id:
            return f"u:{user_id}"
        if session_id and session_id != _DEFAULT_SESSION:
            return f"s:{session_id}"
        return None

    # --------------------------------------------------------------------- #
    # 公开方法
    # --------------------------------------------------------------------- #
    def chat(
        self,
        user_input: str,
        verbose: bool = True,
        is_stream: bool = False,
        save_on_interrupt: bool = True,
        on_event: Optional[Callable[[dict], None]] = None,
        # ---- P0-3：多租户隔离 ----
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        # ---- P0.5：结构化返回 ----
        return_result: bool = False,
    ) -> Union[str, Iterator[str], AnswerResult, "StreamingAnswer"]:
        """完整一轮：Q&A 缓存 → 工具路由 → (改写 → 检索) → 回答 → 记忆。

        参数：
            is_stream = False        → 返回完整字符串（默认）
            is_stream = True         → 返回生成器
            save_on_interrupt = True → 流式中途被打断时，已收到的部分仍入记忆（默认）
            save_on_interrupt = False→ 仅在流式完整结束时写记忆；中途打断/报错不入库

            session_id (P0-3): 会话标识。不同 session 的对话记忆互相隔离。
                               不传 → 落到默认桶（与改造前一致）。
            user_id (P0-3):    用户标识。用于 L1/L3 的 namespace 隔离，
                               优先级高于 session_id（跨 session 复用个人积累）。

            return_result (P0.5): 
                False（默认）→ 返回 str / Iterator[str]，**与改造前完全一致**
                True         → 非流式时返回 `AnswerResult`（含 sources /
                               citations / confidence / trace）；
                               流式时返回 `StreamingAnswer` —— 它在迭代上
                               等价于生成器（`for piece in it`），但**迭代结束后**
                               `.result` 会被填成完整的 AnswerResult，
                               供前端渲染来源面板。
                               （不能直接给生成器挂属性：CPython 的 generator
                                没有 __dict__，详见 answer_types.StreamingAnswer）

        说明：
            "打断" 的判定：成功走完 for-loop 后会设 completed=True；
            若 finally 运行时 completed 仍为 False，表示 generator 被提前关闭
            （调用方 break / gen.close() / 被 GC 回收 / LLM 内部报错）。
            非流式路径调用 LLM 报错会直接 raise，不会写记忆，
            save_on_interrupt 仅对流式路径生效。

            on_event: 可选回调，每个流水线步骤会发射一个事件 dict：
                {"type": "step", "stage": "cache|router|tool|rewrite|retrieve|answer|sources",
                 "title": str, "detail": str, "elapsed_ms": int}
                不传则行为与原来完全一致（CLI 无感）。Web 端可据此渲染可折叠步骤块。
                elapsed_ms 为「该步骤自身的耗时」（毫秒）。
                P0-4 新增 stage="tool_failed"；P0.5 新增 stage="sources"。

                ⚠️ 耗时归属的修正（本次）：
                改造前 `elapsed_ms` 是「距上一个事件」的差值，而
                `_emit("answer")` 是在**调用 LLM 之前**发射的，于是：
                    answer  事件 → 只统计了"拼 messages"的时间（~0ms）
                    sources 事件 → 把**整个 LLM 生成时间**算成了"来源归因"
                用户看到的「🔖 来源归因 6.1s / 💬 生成回答 0ms」正是这个
                错位 —— 来源归因本身只是一次正则解析，实测 1000 次仅
                324ms（**单次 0.32ms**），根本不可能要 6 秒。
                现在 answer 事件在 LLM 返回后发射并携带真实生成耗时，
                sources 事件只计自己的解析耗时。
        """
        import time as _time
        _t_start = _time.perf_counter()
        _t_prev = [_t_start]               # 用 list 包裹以便闭包内可变更
        trace: list[dict] = []             # P0.5：把事件也收集到 AnswerResult.trace

        def _emit(stage: str, title: str, detail: str = "",
                  elapsed_ms: Optional[int] = None) -> None:
            """发射一个流水线步骤事件。

            elapsed_ms 语义：**该步骤自身的耗时**。
            默认取「距上一个事件」的墙钟差值，这在"事件紧跟在对应工作之后
            发射"时是准确的。个别步骤（如流式回答，见下方 TTFT 说明）需要
            显式传入自己测出来的值，此时仍会同步推进 `_t_prev` 基线，
            保证后续步骤的差值不会被重复计算。
            """
            now = _time.perf_counter()
            auto_ms = int((now - _t_prev[0]) * 1000)
            _t_prev[0] = now
            ev = {"type": "step", "stage": stage, "title": title,
                  "detail": detail,
                  "elapsed_ms": auto_ms if elapsed_ms is None else int(elapsed_ms)}
            trace.append(ev)               # P0.5：无条件收集，便于结构化返回
            if on_event is not None:
                try:
                    on_event(ev)
                except Exception:
                    pass

        def _total_ms() -> int:
            return int((_time.perf_counter() - _t_start) * 1000)

        # ---- P0-3：解析 session / namespace ----
        sid = session_id or _DEFAULT_SESSION
        memory = self._get_memory(sid)
        namespace = self._resolve_namespace(user_id, session_id)

        # ══════════════════════════════════════════════════════════════════
        # 0) L1 QACache 短路
        # ══════════════════════════════════════════════════════════════════
        # 说明：L1 精准/模糊命中很轻量，先单独走 qa_cache.get 短路，
        # 避免工具路由能命中时白算一次向量。真正的 L2/L3/L5 检索留到 Step 2。
        #
        # P0-1/P0-3 加固：
        #   - 传入 namespace → 精确命中与 fuzzy 命中都严格按租户过滤；
        #   - QACache 内部对 fuzzy 候选额外做「0.93 阈值 + 槽位一致性门禁」，
        #     所以这里能命中的都是真正等价的问题（不会再出现
        #     "问 CFO 返回 CEO 答案" 这类毫秒级错误答案）。
        cached = self.qa_cache.get(user_input, namespace=namespace)
        if cached is not None:
            if verbose:
                print("[0/3] Q&A 缓存命中 ✓ → 直接返回预设答案")
            _emit("cache", "L1 Q&A 缓存命中", "直接返回预设答案（毫秒级）")
            memory.add_user(user_input)
            memory.add_assistant(cached)
            if is_stream:
                # 缓存答案已就绪，切片逐字吐出，保持与真实流式一致的打字机观感
                gen = _stream_text(cached)
                if return_result:
                    # 短路路径的答案立即可得，直接给出完整 AnswerResult
                    return StreamingAnswer(gen, AnswerResult(
                        text=cached, cache_hit=True, confidence=1.0,
                        layer_hits={"L1_qa": 1}, trace=trace,
                        elapsed_ms=_total_ms(),
                    ))
                return gen
            if return_result:
                return AnswerResult(
                    text=cached, cache_hit=True, confidence=1.0,
                    layer_hits={"L1_qa": 1}, trace=trace, elapsed_ms=_total_ms(),
                )
            return cached

        # 本轮上下文累积变量
        evidence_block = ""                # P0-4：<doc> 结构化证据块
        sources: list[Source] = []         # P0.5：来源列表
        rag_result = None                  # 记录本轮 RAG 结果，供成功后 archive
        confidence = 0.0                   # P0-2：整体证据置信度
        low_evidence = False               # P0-2：abstention 信号
        rewritten = ""
        used_tool_name: Optional[str] = None
        tool_failed = False                # P0-4：工具失败并降级

        # ══════════════════════════════════════════════════════════════════
        # 1) 工具路由（stage="router"）
        # ══════════════════════════════════════════════════════════════════
        tool_decision = None
        if self.enable_tools:
            tool_decision = route_and_call(user_input)
            if verbose:
                print(f"[1/3] 工具路由 → {tool_decision['tool']}, args: {tool_decision['args']}")
            _emit("router", "工具路由",
                  f"tool={tool_decision['tool']}, args={tool_decision['args']}")

        # ══════════════════════════════════════════════════════════════════
        # P0-4：工具成功/失败的判定与降级
        # ══════════════════════════════════════════════════════════════════
        # 【改造前的 bug】
        #     used_tool = (... and tool_decision.get("result") is not None)
        #   `call_tool()` 失败时返回 `{"error": "..."}`，它不是 None，
        #   于是 used_tool=True → **跳过全部检索** → 把错误信息当"外部资料"
        #   喂给 LLM。天气 API 挂了，用户看到的是"抱歉信息不足"，
        #   而不是自动降级去搜索。
        #
        # 【改造后】
        #   `route_and_call()` 返回显式的 `ok` 字段（见 tool_router.py）。
        #   只信 `ok`：
        #     ok=True  → 用工具结果，跳过检索（原快路径不变）
        #     ok=False → 降级到通用检索通路，并发 tool_failed 事件供观测
        used_tool = bool(
            tool_decision is not None
            and tool_decision.get("tool") not in (None, "NO_TOOL")
            and tool_decision.get("ok")
        )

        # 工具被路由到了但执行失败 → 记录并降级
        if (
            tool_decision is not None
            and tool_decision.get("tool") not in (None, "NO_TOOL")
            and not tool_decision.get("ok")
        ):
            tool_failed = True
            err = tool_decision.get("error") or "unknown error"
            kind = tool_decision.get("error_kind") or "unknown"
            if verbose:
                print(f"[1/3] 工具 {tool_decision['tool']} 失败({kind}) → 降级到通用检索")
            _emit("tool_failed", "工具调用失败 → 降级检索",
                  f"{tool_decision['tool']} ({kind}): {err}")

        if used_tool and tool_decision is not None:
            used_tool_name = tool_decision["tool"]

            # 确定性工具直接短路返回（时间类问题答案唯一，无需再过 LLM）
            if (
                tool_decision["tool"] == "get_current_time"
                and isinstance(tool_decision.get("result"), dict)
                and tool_decision["result"].get("answer")
            ):
                answer = tool_decision["result"]["answer"]
                if verbose:
                    print("[2/3] 当前时间工具命中 ✓ → 直接返回工具答案")
                _emit("tool", "命中确定性工具", "get_current_time → 直接返回工具答案")
                memory.add_user(user_input)
                memory.add_assistant(answer)
                if is_stream:
                    gen = _stream_text(answer)
                    if return_result:
                        return StreamingAnswer(gen, AnswerResult(
                            text=answer, used_tool=used_tool_name,
                            confidence=1.0, trace=trace,
                            elapsed_ms=_total_ms(),
                        ))
                    return gen
                if return_result:
                    return AnswerResult(
                        text=answer, used_tool=used_tool_name, confidence=1.0,
                        trace=trace, elapsed_ms=_total_ms(),
                    )
                return answer

            # 其余工具：结果作为资料交给 LLM 组织语言
            tool_text = format_tool_result(tool_decision)
            if tool_text:
                # P0-4：工具结果同样走 <doc> 封装。
                # 虽然工具返回来自可信 API（不像网页那样可被投毒），
                # 但统一封装有三个好处：
                #   ① prompt 结构一致，模型不需要适应两种资料格式；
                #   ② 工具结果也能被 [n] 引用，来源面板会显示"工具调用"；
                #   ③ 万一某个工具代理了第三方内容（如 arXiv 摘要、
                #      GitHub README），清洗逻辑依然覆盖到它。
                evidence_block, src_dicts = self._wrap_tool_evidence(
                    tool_decision["tool"], tool_text
                )
                sources = [Source.from_dict(d) for d in src_dicts]
                confidence = 0.95   # 工具是确定性数据源，给高置信度
            if verbose:
                print("[2/3] 已使用专用工具，跳过通用检索")
            _emit("tool", "使用专用工具",
                  f"{tool_decision['tool']} → 已获取结果，跳过通用检索")
        else:
            # ══════════════════════════════════════════════════════════════
            # 2) 通用检索通路：rewriter → RAG 分层检索（含 L4 web 兜底）
            # ══════════════════════════════════════════════════════════════
            history_snippet = memory.summarize_recent(n=3)
            rewritten = query_rewrite_route(
                user_input,
                history=history_snippet,
                rewrite_type=self.rewrite_type,
            )
            if verbose:
                print(f"[2/3] query 改写 (mode={self.rewrite_type}) → {rewritten}")
            _emit("rewrite", "Query 改写",
                  f"mode={self.rewrite_type} → {rewritten}")

            if rewritten and rewritten.upper() != config.NO_SEARCH_SENTINEL:
                if self.retriever is not None:
                    # ---- 走分层 RAG：L2 wiki + L3 history + L5 KG，必要时补 L4 web ----
                    # P0-3：namespace 透传 → L1 精确/模糊命中与 L3 检索都按租户隔离
                    #
                    # ⚡ 性能：`route_query=user_input` 传**原始** query 做层激活决策。
                    # 原因（实测定位到的首要瓶颈）：rewriter 会给
                    #   "美国一共多少位副总统 历史上"
                    # 改写成
                    #   "美国历史上共有多少位副总统 2026年"   ← 凭空多了年份
                    # 而 `is_time_sensitive()` 把「当前年份」当强时效信号，
                    # 于是 L4_web 被**强制激活** → DDG 未命中缓存时要 16~38s
                    # → 整个检索被拖到 21s。而离线三层实测只要 ~0.5s、
                    # 聚合置信度 0.99（完全够用，本不需要联网）。
                    #
                    # 时效性是"用户意图"的属性，不该由改写结果决定 ——
                    # 详见 `LayeredRetriever.retrieve()` 里 route_query 的完整说明。
                    rag_result = self.retriever.retrieve(
                        rewritten, namespace=namespace, route_query=user_input,
                    )
                    confidence = rag_result.confidence          # P0-2
                    low_evidence = rag_result.low_evidence      # P0-2

                    # ---- P0-4 + P0.5：证据封装 + 来源抽取 ----
                    if self.enable_evidence_guard:
                        evidence_block, src_dicts = build_evidence_block(
                            rag_result.passages
                        )
                        sources = [Source.from_dict(d) for d in src_dicts]
                    else:
                        # A/B 对照组：退回改造前的裸文本拼接（不推荐用于生产）
                        evidence_block = rag_result.as_context_block()
                        sources = []

                    hits = ", ".join(
                        f"{k}:{v}" for k, v in rag_result.layer_hits.items()
                    )
                    if verbose:
                        print(f"      RAG 检索完成 [{hits}], 融合 "
                              f"{len(rag_result.passages)} 段, conf={confidence:.3f}")
                    detail = (
                        f"[{hits}], 融合 {len(rag_result.passages)} 段, "
                        f"置信度 {confidence:.2f}"
                    )
                    if rag_result.web_fallback:
                        # 可观测：这个标记突然变多通常意味着离线索引覆盖度下降
                        detail += "（离线证据不足 → 已补 L4 web）"
                    if low_evidence:
                        detail += " ⚠️ 证据不足"
                    _emit("retrieve", "分层 RAG 检索", detail)
                else:
                    # ---- 无 RAG 时的兼容路径：裸 web_search ----
                    results = web_search(rewritten, top_k=self.top_k)
                    if self.enable_evidence_guard:
                        # 把 web 结果也包成 <doc>，保持 prompt 结构与归因能力一致
                        evidence_block, src_dicts = self._wrap_web_evidence(results)
                        sources = [Source.from_dict(d) for d in src_dicts]
                    else:
                        evidence_block = format_results(results)
                    # 裸 web 路径没有校准分，用一个保守的经验值：
                    # 有结果 → 0.6（约等于 web top-2 的校准概率）；无结果 → 0
                    confidence = 0.6 if results else 0.0
                    low_evidence = not results
                    if verbose:
                        print(f"      web_search 完成，共 {len(results)} 条结果")
                    _emit("retrieve", "Web 检索", f"共 {len(results)} 条结果")
            else:
                if verbose:
                    print("      无需检索（NO_SEARCH）")
                _emit("rewrite", "无需检索", "NO_SEARCH（闲聊/追问）")

        # ══════════════════════════════════════════════════════════════════
        # 3) 拼装 messages 发给 summary 阶段模型
        # ══════════════════════════════════════════════════════════════════
        messages = [{"role": "system", "content": PROMPTS["summary_system"]}]
        messages.extend(memory.get_messages())
        if evidence_block:
            # P0-4：用 build_user_message 而不是裸 f-string 拼接。
            # 它会把证据包在 <evidence>、把提问包在 <question>，
            # 双向结构化 —— 即使证据里伪造了定界符（已被 sanitize 转义），
            # 模型看到的仍然是清晰的"数据区 / 指令区"边界。
            # low_evidence=True 时还会附一条 <retrieval_note>，
            # 引导模型明确说"资料不足"而不是硬编（抑制幻觉）。
            user_msg = build_user_message(
                user_input, evidence_block, low_evidence=low_evidence
            )
        else:
            user_msg = user_input
        messages.append({"role": "user", "content": user_msg})

        if verbose:
            print(f"[3/3] 调用 summary 模型生成回答 (is_stream={is_stream}) ...")

        # ══════════════════════════════════════════════════════════════════
        # 耗时归属修正：answer 事件必须在 LLM **返回之后**发射
        # ══════════════════════════════════════════════════════════════════
        # 【改造前的错位】
        #   这里原本直接 `_emit("answer", ...)`，然后才去调 llm_chat()。
        #   由于 elapsed_ms = 距上一个事件的差值：
        #     answer  事件 → 只量到"拼 messages"这几微秒        → 显示 0ms
        #     sources 事件 → 把 LLM 的整个生成时间算进去了        → 显示 6.1s
        #   于是终端出现了自相矛盾的一幕：
        #       💬 生成回答 → 调用 summary 模型     ⏱️ 0ms
        #       🔖 来源归因 → 6 条来源…             ⏱️ 6.1s
        #   而"来源归因"实际只是一次正则解析：实测 parse_citations
        #   跑 1000 次共 324ms（**单次 0.32ms**），比显示值小了 4 个数量级。
        #   用户据此怀疑"归因慢"，但真正慢的是远端 DeepSeek 的生成 ——
        #   **观测口径的 bug 会把优化引向完全错误的方向**，这比慢本身更危险。
        #
        # 【改造后】
        #   记录 LLM 调用前的时间戳，调用返回后用显式 elapsed_ms 发射 answer，
        #   sources 则只计它自己那 0.3ms。这样每个步骤的数字都对应自己的工作。
        #
        # 【流式的特殊处理】
        #   流式下"生成耗时"没有单一定义，业界（Perplexity / vLLM / TGI）
        #   统一用两个指标：
        #     TTFT (Time To First Token) —— 用户感知的响应速度
        #     TPOT (Time Per Output Token) —— 后续吐字速度
        #   所以流式路径在**首个 token 到达时**发射 answer 事件并标注 TTFT，
        #   这才是"用户等了多久才看到东西"的正确度量。
        _t_llm_start = _time.perf_counter()

        def _emit_answer_done(detail: str) -> None:
            """在 LLM 真正产出内容后发射 answer 事件，携带真实耗时。"""
            _emit("answer", "生成回答", detail,
                  elapsed_ms=int((_time.perf_counter() - _t_llm_start) * 1000))

        # ---- 构造本轮的结果骨架（文本稍后填充）----
        def _finalize(answer_text: str) -> AnswerResult:
            """P0.5：把答案 + 来源 + 引用组装成 AnswerResult。

            这里做的关键一步是 `parse_citations()`：
            解析答案里所有 [n]，**校验编号是否真实存在**，
            并回写 `Source.cited` 标志。于是：
              - 前端能只展示"真正被引用"的来源（更干净）；
              - 系统能检测"引用幻觉"（编造不存在的编号）；
              - 能算出引用覆盖率，作为检索精度的在线指标。
            """
            # 显式计时：sources 事件只应包含**归因本身**的耗时。
            # 不能再依赖"距上一个事件"的隐式差值 —— 那样任何插在中间的
            # 工作（LLM 生成、L1 归档写盘…）都会被错算成"来源归因"，
            # 这正是用户看到 6.1s 的根源。
            _t_attr = _time.perf_counter()
            cits, srcs = parse_citations(answer_text, sources)
            res = AnswerResult(
                text=answer_text,
                sources=srcs,
                citations=cits,
                confidence=round(float(confidence), 4),
                low_evidence=low_evidence,
                cache_hit=False,
                used_tool=used_tool_name,
                tool_failed=tool_failed,
                layer_hits=(
                    dict(rag_result.layer_hits) if rag_result is not None else {}
                ),
                rewritten=rewritten,
                trace=trace,
                elapsed_ms=_total_ms(),
            )
            # 发一个 sources 事件，让前端（Web/CLI）能渲染来源面板。
            # 放在 answer 之后，语义上是"回答完成后给出出处"。
            if srcs:
                cited_ids = [s.id for s in res.cited_sources]
                detail = (
                    f"{len(srcs)} 条来源，其中被引用 {len(cited_ids)} 条 "
                    f"{cited_ids}"
                )
                if res.invalid_citation_count:
                    detail += f"；⚠️ {res.invalid_citation_count} 处无效引用编号"
                if res.has_risky_source:
                    detail += "；⚠️ 含可疑指令来源（已中和）"
                _emit("sources", "来源归因", detail,
                      elapsed_ms=int((_time.perf_counter() - _t_attr) * 1000))
            return res

        def _archive_traced(answer_text: str) -> None:
            """归档到 L1/L3，并把耗时单独记成一个事件。

            ⚠️ 为什么必须单独计时：`_archive_if_enabled()` 里的
            `qa_cache.add()` 会**同步**做一次 BGE-M3 编码（用于 fuzzy 命中
            的向量），实测冷启动时含模型加载可达 4 秒、常态约 60ms。
            改造前它夹在 LLM 调用与 sources 事件之间，于是这段时间又被
            算进了"来源归因" —— 本次写完测试才暴露出这第二处错位
            （测试报 sources=4133ms，而归因本身只要 0.3ms）。

            现在它有自己的 stage，观测上一目了然：如果哪天 L1 写入变慢，
            能直接看到是归档慢，而不是去怀疑归因或检索。
            """
            _t_ar = _time.perf_counter()
            self._archive_if_enabled(
                user_input, answer_text, rag_result, namespace=namespace,
            )
            _emit("archive", "归档到 L1/L3", "写入缓存与历史层（越用越强）",
                  elapsed_ms=int((_time.perf_counter() - _t_ar) * 1000))

        # --------- 流式 与 非流式 分路——写什么记忆是一样的 ---------
        if not is_stream:
            answer = llm_chat("summary", messages)
            # LLM 已返回 → 此刻发射 answer 事件，elapsed_ms 就是真实生成耗时
            _emit_answer_done(f"summary 模型返回 {len(answer)} 字")
            memory.add_user(user_input)
            memory.add_assistant(answer)
            # 先组装结果（含来源归因），再归档：
            # 归档是"为了下一次更快"的副作用，不该挡在本次结果之前。
            result = _finalize(answer)
            _archive_traced(answer)
            return result if return_result else answer

        # 流式：返回一个生成器。记忆在 generator 内部完成。
        # P0.5：`_holder` 是生成器与外层 StreamingAnswer 之间的桥。
        # 生成器内部无法直接引用还没创建出来的 StreamingAnswer 实例，
        # 所以用一个 dict 做后期绑定（late binding）。
        _holder: dict = {}

        def _gen() -> Iterator[str]:
            buf: list[str] = []
            completed = False     # 标记是否"正常走完"
            first = True          # 用于在首个 token 到达时发 TTFT
            try:
                for piece in llm_stream_chat("summary", messages):
                    if first:
                        # TTFT：用户从"提交问题"到"看到第一个字"的等待时间。
                        # 这是流式场景下唯一有意义的"生成延迟"指标 ——
                        # 总生成时长受答案长度影响，不能反映响应速度。
                        first = False
                        _emit_answer_done("首个 token 已到达（TTFT）")
                    buf.append(piece)
                    yield piece
                completed = True   # 正常迭代结束（未被报错 / close / GeneratorExit 打断）
            finally:
                full_answer = "".join(buf)
                # 决策是否写记忆：
                #   - 正常走完       → 写
                #   - 未走完（被打断） → 看 save_on_interrupt
                should_save = bool(full_answer) and (completed or save_on_interrupt)
                if should_save:
                    memory.add_user(user_input)
                    memory.add_assistant(full_answer)
                elif verbose and full_answer and not completed:
                    print(
                        f"\n[stream] 中途打断（save_on_interrupt=False），"
                        f"本轮 {len(full_answer)} 字不入记忆"
                    )
                # P0.5：迭代结束（含被打断）后回填 AnswerResult，
                # 前端消费完 token 即可读 `stream.result` 渲染来源面板。
                # 注意放在 finally 里：即使被 Ctrl-C 打断，
                # 也能拿到"已生成部分"的引用解析结果，不会是空对象。
                #
                # ⚡ 顺序：先 _finalize（用户要的来源面板）再归档。
                # 归档要同步跑一次 BGE-M3 编码（常态 ~60ms，冷启动可达数秒），
                # 让它挡在来源面板前面纯属没必要。
                wrapper = _holder.get("wrapper")
                if wrapper is not None:
                    try:
                        wrapper.result = _finalize(full_answer)
                    except Exception as e:
                        print(f"[agent] 组装 AnswerResult 失败: {e}")
                # 只有真正完整回答才归档到 L3/L1，避免入库半截答案污染
                if should_save and completed:
                    _archive_traced(full_answer)

        gen = _gen()
        if not return_result:
            return gen

        # 占位结果：sources / confidence 在生成前就已确定，先填上；
        # text / citations 等生成结束后由 _gen 的 finally 回填。
        placeholder = AnswerResult(
            text="", sources=sources, confidence=round(float(confidence), 4),
            low_evidence=low_evidence, used_tool=used_tool_name,
            tool_failed=tool_failed, rewritten=rewritten, trace=trace,
            layer_hits=(dict(rag_result.layer_hits) if rag_result is not None else {}),
        )
        wrapper = StreamingAnswer(gen, placeholder)
        _holder["wrapper"] = wrapper
        return wrapper

    # --------------------------------------------------------------------- #
    # 内部辅助
    # --------------------------------------------------------------------- #

    @staticmethod
    def _wrap_tool_evidence(tool_name: str, tool_text: str) -> tuple[str, list[dict]]:
        """把工具调用结果包成 <doc> 证据块（P0-4）。

        复用 `evidence.build_evidence_block()`，需要构造一个最小的
        Passage-like 对象。用轻量匿名类而不 import rag.types.Passage，
        是为了让 `enable_rag=False` 时本函数依然可用（不依赖 rag 包）。
        """
        class _ToolPassage:
            text = tool_text
            title = f"工具调用: {tool_name}"
            url = ""
            layer = "tool"
            score = 0.95
            metadata = {"calibrated": 0.95}

        block, srcs = build_evidence_block([_ToolPassage()])
        for s in srcs:
            s["layer_label"] = "工具调用"
        return block, srcs

    @staticmethod
    def _wrap_web_evidence(results: list[dict]) -> tuple[str, list[dict]]:
        """把裸 web_search 结果包成 <doc> 证据块（P0-4，enable_rag=False 路径）。

        为什么这条兼容路径也要封装：它同样是**不可信的网页内容**，
        风险与走 L4 完全一样。如果只给 RAG 路径加防护，攻击者只要
        让系统落到 enable_rag=False 分支就能绕过（虽然不容易，
        但"防护有缺口"本身就是问题）。
        """
        class _WebPassage:
            def __init__(self, r: dict, rank: int):
                self.text = r.get("snippet", "") or ""
                self.title = r.get("title", "") or ""
                self.url = r.get("url", "") or ""
                self.layer = "L4_web"
                self.score = 1.0 - rank * 0.05
                # 复用 L4 的校准参数，与走 RAG 路径的置信度口径保持一致
                try:
                    from rag.calibration import calibrate
                    cal = calibrate("L4_web", self.score)
                except Exception:
                    cal = self.score
                self.metadata = {"rank": rank, "calibrated": round(cal, 4)}

        passages = [_WebPassage(r, i) for i, r in enumerate(results or [])]
        return build_evidence_block(passages)

    def _archive_if_enabled(
        self,
        query: str,
        answer: str,
        rag_result,
        namespace: Optional[str] = None,
    ) -> None:
        """成功回答后归档，让系统"越用越强"、热点问题二次命中即毫秒返回。

        写两个地方，各司其职：
          1. **L1 QACache**（同步、精确+模糊命中）：把 (原始 query, answer) 写进 L1。
             下次问**同样/近似**的问题时，chat() 的 Step0 `qa_cache.get()`
             会精确/模糊命中直接短路，跳过工具路由 / Query 改写 / 分层检索 / LLM，
             把 18s 压到毫秒级。
          2. **L3 History**（异步、向量库）：把 query+answer+sources 存进历史层，
             作为**融合检索的一路**丰富后续**不同**问题的召回，但它不会短路整条链路。

        注意：L1 用原始 user_input 作 key（Step0 也用原始 query 查），二者对齐才能命中；
        若用改写后的 query 作 key 会与查询侧不一致，导致命中不了。

        ══════════════════════════════════════════════════════════════════
        P0-1：这里是「越用越强」变成「越用越错」的关键分岔点
        ══════════════════════════════════════════════════════════════════
        改造前本方法**无条件** `self.qa_cache.add(query, answer)`，配合
        全局 30 天 TTL + 0.8 的 fuzzy 阈值，产生三类静默错误：

          ① 时效污染：「今天上海天气」被冻结 30 天。
             注意 `rag/router.py` 里虽有 `is_time_sensitive()`，但它只决定
             "要不要召 L4"，**完全不影响是否写 L1**；而 Step0 的 L1 短路
             发生在 router 之前，时效判断根本没机会跑。
          ② 语义碰撞：见 cache_policy.py 顶部的实测数据
             （CEO/CFO 余弦 0.878、2024/2025 GDP 0.853）。
          ③ 幻觉固化：一次拒答/幻觉同时写进 L1 和 L3，
             后续同类问题被这条脏数据不断加固。

        改造后：先过 `decide_cacheability()` 拿到 (cacheable, ttl)，
        拒收的直接跳过 L1（但**仍然写 L3**——见下方注释说明为什么）。
        """
        if self.retriever is None or not answer:
            return

        layer_hits = (
            dict(rag_result.layer_hits) if rag_result is not None else {}
        )

        # 准入决策只算一次，L1 与 L3 共用（原先算了两遍，浪费且可能不一致）
        decision = (
            decide_cacheability(query, answer, layer_hits)
            if self.enable_cache_policy else None
        )

        # ---- 1) 写 L1 QACache（受准入策略控制）----
        if decision is not None:
            if decision.cacheable:
                try:
                    self.qa_cache.add(
                        query, answer, ttl=decision.ttl, namespace=namespace,
                    )
                except Exception as e:
                    print(f"[agent] 写入 L1 QACache 失败: {e}")
            else:
                # 不写 L1，但记录原因。这行日志很有价值：
                # 上线后统计各 tier 的分布，能看出流量里时效类问题的占比。
                print(
                    f"[agent] 跳过 L1 缓存 (tier={decision.tier}): "
                    f"{decision.reason}"
                )
        else:
            # A/B 对照组：退回改造前的无条件写入（不推荐用于生产）
            try:
                self.qa_cache.add(query, answer, namespace=namespace)
            except Exception as e:
                print(f"[agent] 写入 L1 QACache 失败: {e}")

        # ---- 2) 归档到 L3 History（异步）----
        # 为什么被 L1 拒收的**大多仍要**写 L3：两者作用完全不同。
        #   L1 是「短路」：命中即直接返回，答案陈旧 = 直接给错。
        #   L3 是「召回的一路」：只作为参考资料参与 RRF 融合，
        #      最终答案仍由 LLM 结合其它层（尤其 L4 实时）重新生成。
        # 所以时效类问答进 L3 是有价值的（记录用户关注点、可复用推理），
        # 风险远低于进 L1。
        #
        # ⚠️ 例外：拒答类 / 过短答案连 L3 也不该进——它们没有信息量，
        # 只会污染召回并稀释真正有用的历史条目。
        #
        # ══════════════════════════════════════════════════════════════
        # Stage-1 修复②：`reject_partial_refusal` 必须也在这个名单里
        # ══════════════════════════════════════════════════════════════
        # 「部分拒答」（开头正常、结尾承认核心信息缺失）在改造前被判为
        # `tier=stable`，于是**同时写进 L1 和 L3**。写进 L3 的后果最严重，
        # 因为它会形成一个自我强化的失败循环（实测已观测到）：
        #
        #     拒答 → 存进 L3 → 下次召回到自己的拒答（实测 calib=0.578）
        #          → 置信度虚高 → 不触发 L4 → 再次拒答 → 循环加固
        #
        # 实测证据：查「茅盾文学奖 历届 获奖名单」时 L3 返回的 top-1 就是
        # 系统**自己上一轮的拒答**。这比单纯答不出来更糟 —— 它会持续污染
        # 历史层，而且越用越严重，与"越用越强"的设计目标完全相反。
        #
        # 所以这类答案的处理必须和纯拒答一致：L1 不写、**L3 也不写**。
        if decision is not None and decision.tier in (
            "reject_low_quality", "reject_too_short", "reject_partial_refusal",
        ):
            return

        sources = None
        if rag_result is not None and getattr(rag_result, "passages", None):
            sources = [p.to_dict() for p in rag_result.passages]
        try:
            self.retriever.archive(
                query, answer, sources=sources, namespace=namespace,
            )
        except Exception as e:
            print(f"[agent] archive 到 L3 失败: {e}")

    def reset(self, session_id: Optional[str] = None) -> None:
        """清空会话记忆。

        Args:
            session_id: 指定则只清该 session；不传则清**全部** session
                        （与改造前 `reset()` 清空单个 memory 的语义等价，
                        因为改造前本来就只有一个 memory）。
        """
        if session_id:
            mem = self._memories.get(session_id)
            if mem is not None:
                mem.clear()
            return
        for mem in self._memories.values():
            mem.clear()

    def stats(self) -> dict:
        """运维快照：L1 命中统计 + 活跃 session 数。

        P0-1 重点关注 `slot_gate_rejects`：它近似等于
        "如果没有槽位门禁，本进程会返回多少次错误答案"。
        """
        out: dict = {"active_sessions": len(self._memories)}
        try:
            out["qa_cache"] = self.qa_cache.stats()
        except Exception:
            pass
        return out

    def warmup(self, probe_query: str = "预热", verbose: bool = True) -> None:
        """预热整条链路：LLM 模型驻留 + L2/L5 索引 + BGE-M3 权重 + reranker。

        目的：消除**首条查询**的冷启动延迟毛刺，让首查询延迟与稳态一致。
        建议在 CLI/服务进入交互循环前调用一次。

        ════════════════════════════════════════════════════════════════
        本次新增：LLM 预热（这是之前最大的遗漏）
        ════════════════════════════════════════════════════════════════
        改造前本方法只预热了 RAG 侧（embedding / FAISS / KG / reranker），
        **完全没有预热 LLM**。而实测数据显示 LLM 才是本机上最贵的那块
        冷启动成本：

            tool_router.route 连续 3 次（同一 query）：
                第 1 次 5129 ms → 第 2 次 607 ms → 第 3 次 558 ms
            且 `curl /api/ps` 返回 {"models":[]} —— 无模型驻留

        用户观测到的「首次工具路由 23s，第二次 3.8s」就是这个：
        ollama 要把 qwen3:4b-q8_0 的 4.3GB 权重 mmap 进来、分配 KV cache、
        跑首次 kernel 预热。这个成本与 prompt 无关，纯粹是模型加载。

        配合 `models_config.OLLAMA_KEEP_ALIVE=-1`（模型常驻），
        这块成本被彻底移到启动期，**且不会因空闲 5 分钟而重新付一次**
        （ollama 默认 keep_alive=5m，交互式使用几乎必然反复冷加载）。

        ════════════════════════════════════════════════════════════════
        ⚠️ 预热顺序：必须 **RAG 先、LLM 后**（这一点与直觉相反）
        ════════════════════════════════════════════════════════════════
        第一版实现按"用户先等到 router"的直觉把 LLM 放在最前面，
        结果实测发现**首条 router 反而更慢了**：

            LLM 先、RAG 后：  router 首调 12220 ms  ← 反而退化！
            RAG 先、LLM 后：  router 首调  1647 ms  ← 快 7 倍

        且两种顺序下 `/api/ps` 都显示模型仍然驻留（expires_at 是 2318 年），
        **说明慢的原因不是模型被卸载**，而是资源竞争：

          * BGE-M3（fp16，约 2.3GB）加载到 MPS 时会大量申请统一内存，
            同时读 2.6GB FAISS 索引 + 10GB SQLite 造成密集页缓存压力；
          * 这会把已经驻留的 ollama 模型权重页**挤出物理内存**
            （macOS 统一内存架构下 CPU/GPU 共享同一池子，
             ollama 与 PyTorch 互相看不见对方的占用）；
          * 于是下一次推理要重新从磁盘缺页换入 —— 表现为"模型明明驻留
            却依然很慢"，这是最容易误判的一类性能问题。

        把 LLM 放最后，它的权重页就是"最后被 touch 的"，在 LRU 回收
        顺序里最安全。这是内存受限环境下的通用原则：
        **让延迟最敏感的组件最后预热**。
        """
        # ---- 1) RAG：L2/L5 索引 + BGE-M3 + reranker ----
        # 必须放在 LLM 之前：它的内存/页缓存压力会把 ollama 的权重页挤出去
        # （见上方实测数据）。enable_rag=False 时 retriever 为 None，跳过。
        if self.retriever is not None:
            try:
                self.retriever.warmup(probe_query=probe_query, verbose=verbose)
            except Exception as e:
                print(f"[agent] RAG warmup 异常（不影响后续查询）: {e}")

        # ---- 2) LLM：最后把本地 ollama 模型拉起来并常驻 ----
        # 放最后 = 它的权重页最"新"，最不容易被后续内存压力回收。
        try:
            llm_warmup_all(verbose=verbose)
        except Exception as e:
            print(f"[agent] LLM warmup 异常（不影响后续查询）: {e}")

    def close(self) -> None:
        """优雅关闭：把 L3 增量 worker 队列 flush 后再退出。"""
        if self.retriever is not None:
            try:
                self.retriever.close()
            except Exception as e:
                print(f"[agent] retriever.close 异常: {e}")
