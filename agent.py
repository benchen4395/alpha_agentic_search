# agent.py
"""主控 Agent：统一通过 llm_client + models_config + prompts 调用各阶段 LLM。

链路：
    Step 0  Q&A 缓存命中 → 直接返回
    Step 1  工具路由 (天气 / GitHub / arXiv …)      ← stage="router"
    Step 2  Query 改写 + Web 检索                   ← stage="rewriter"
    Step 3  生成最终回答                             ← stage="summary"

任何阶段想换模型 / 换 provider / 改 prompt，都不需要动这个文件，分别去：
    - models_config.STAGES["<stage>"]
    - prompts.PROMPTS["<key>"]
"""
from configs import config

from llm_client import chat as llm_chat, stream_chat as llm_stream_chat
from memory import ConversationMemory
from configs.prompts import PROMPTS
from qa_cache import QACache
from query_rewriter import query_rewrite_route
from searcher import web_search, format_results
from tool_router import route_and_call, format_tool_result

# ---- 方案 C：分层记忆 RAG（rag/）----
# 走 LayeredRetriever：L1 QACache → L2 Wiki → L3 History → L5 KG (并行)
# → 离线不达标才补 L4 Web。相比原来的裸 web_search，可命中常识/历史，越用越强。
from rag import LayeredRetriever

from typing import Iterator, Union, Callable, Optional


def _stream_text(text: str, chunk_size: int = 2, delay: float = 0.018) -> Iterator[str]:
    """把一段已就绪的完整文本切成小片，模拟流式逐字吐出。

    用于 L1 缓存命中 / 确定性工具命中这类"答案已经拿到、无需再调 LLM"的短路
    路径：这些路径本没有 token 流，若直接 ``iter([text])`` 一次性返回整段，CLI/Web
    在流式模式下就会瞬间刷出全部内容，看起来"不是流式"。这里按 ``chunk_size``
    个字符切片逐个 yield，视觉上与真实 LLM 流式一致（打字机效果）。

    Args:
        text:       完整答案。
        chunk_size: 每次吐出的字符数（默认 2；中文按字符切，1~3 都自然）。
        delay:      每片之间的 sleep 秒数（默认 0.018s ≈ 55 字/秒的打字机节奏）。
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
    ):
        # ⚠️ 模型选择由 models_config.STAGES["summary"] 决定
        # 调用方需要换模型/provider 直接改 models_config.py 即可
        self.top_k = top_k
        self.enable_tools = enable_tools
        self.rewrite_type = (
            rewrite_type if rewrite_type is not None else config.QUERY_REWRITE_TYPE
        )
        self.memory = ConversationMemory(max_turns=max_memory_turns)

        # Q&A 缓存：支持外部注入或按参数自建
        self.qa_cache: QACache = qa_cache or QACache(
            backend=qa_cache_backend,   # 如果启用多级缓存，可使用: layers=["diskcache", "redis"]
            cache_dir=qa_cache_dir,
            redis_url=qa_redis_url,
            ttl=qa_cache_ttl,
            enable_fuzzy=True,          # 先精准匹配，再模糊匹配
            fuzzy_threshold=0.8
        )

        # 分层 RAG：把 qa_cache 作为 L1 复用（enable_rag=False 可退回裸 web_search）
        self.retriever = None
        if enable_rag:
            self.retriever = LayeredRetriever(
                qa_cache=self.qa_cache,
                strategy=rag_strategy,
            )

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
    ) -> Union[str, Iterator[str]]:
        """完整一轮：Q&A 缓存 → 工具路由 → (改写 → 检索 fallback) → 回答 → 记忆。

        参数：
            is_stream = False        → 返回完整字符串（默认）
            is_stream = True         → 返回生成器
            save_on_interrupt = True → 流式中途被打断时，已收到的部分仍入记忆（默认）
            save_on_interrupt = False→ 仅在流式完整结束时写记忆；中途打断/报错不入库

        说明：
            "打断" 的判定：成功走完 for-loop 后会设 completed=True；
            若 finally 运行时 completed 仍为 False，表示 generator 被提前关闭
            （调用方 break / gen.close() / 被 GC 回收 / LLM 内部报错）。
            非流式路径调用 LLM 报错会直接 raise，不会写记忆，
            save_on_interrupt 仅对流式路径生效。

            on_event: 可选回调，每个流水线步骤会发射一个事件 dict：
                {"type": "step", "stage": "cache|router|tool|rewrite|retrieve|answer",
                 "title": str, "detail": str, "elapsed_ms": int}
                不传则行为与原来完全一致（CLI 无感）。Web 端可据此渲染可折叠步骤块。
                elapsed_ms 为「距上一个步骤事件」的耗时（毫秒），首个步骤为距 chat() 起点。
        """
        import time as _time
        _t_prev = [_time.perf_counter()]   # 用 list 包裹以便闭包内可变更

        def _emit(stage: str, title: str, detail: str = "") -> None:
            now = _time.perf_counter()
            elapsed_ms = int((now - _t_prev[0]) * 1000)
            _t_prev[0] = now
            if on_event is not None:
                try:
                    on_event({"type": "step", "stage": stage,
                              "title": title, "detail": detail,
                              "elapsed_ms": elapsed_ms})
                except Exception:
                    pass

        # 0) L1 QACache 短路
        # 说明：L1 精准/模糊命中很轻量，先单独走 qa_cache.get 短路，
        # 避免工具路由能命中时白算一次向量。真正的 L2/L3/L5 检索留到 Step 2。
        cached = self.qa_cache.get(user_input)
        if cached is not None:
            if verbose:
                print("[0/3] Q&A 缓存命中 ✓ → 直接返回预设答案")
            _emit("cache", "L1 Q&A 缓存命中", "直接返回预设答案")
            self.memory.add_user(user_input)
            self.memory.add_assistant(cached)
            if is_stream:
                # 缓存答案已就绪，切片逐字吐出，保持与真实流式一致的打字机观感
                return _stream_text(cached)
            return cached

        context_block = ""
        rag_result = None                     # 记录本轮 RAG 结果，供成功后 archive

        # 1) 工具路由（stage="router"）
        tool_decision = None
        if self.enable_tools:
            tool_decision = route_and_call(user_input)
            if verbose:
                print(f"[1/3] 工具路由 → {tool_decision['tool']}, args: {tool_decision['args']}")
            _emit("router", "工具路由",
                  f"tool={tool_decision['tool']}, args={tool_decision['args']}")

        used_tool = (
            tool_decision is not None
            and tool_decision["tool"] not in (None, "NO_TOOL")
            and tool_decision.get("result") is not None
        )

        if used_tool:
            # 确定性工具直接短路返回
            if (
                tool_decision["tool"] == "get_current_time"
                and isinstance(tool_decision.get("result"), dict)
                and tool_decision["result"].get("answer")
            ):
                answer = tool_decision["result"]["answer"]
                if verbose:
                    print("[2/3] 当前时间工具命中 ✓ → 直接返回工具答案")
                _emit("tool", "命中确定性工具", "get_current_time → 直接返回工具答案")
                self.memory.add_user(user_input)
                self.memory.add_assistant(answer)
                if is_stream:
                    # 工具答案已就绪，同样切片逐字吐出以保持流式观感
                    return _stream_text(answer)
                return answer

            context_block = format_tool_result(tool_decision)
            if verbose:
                print("[2/3] 已使用专用工具，跳过通用检索")
            _emit("tool", "使用专用工具",
                  f"{tool_decision['tool']} → 已获取结果，跳过通用检索")
        else:
            # 2) 通用检索通路：rewriter → RAG 分层检索（含 L4 web 兜底）
            history_snippet = self.memory.summarize_recent(n=3)
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
                    rag_result = self.retriever.retrieve(rewritten)
                    context_block = rag_result.as_context_block()
                    if verbose:
                        hits = ", ".join(f"{k}:{v}" for k, v in rag_result.layer_hits.items())
                        print(f"      RAG 检索完成 [{hits}], 融合 {len(rag_result.passages)} 段")
                    _emit("retrieve", "分层 RAG 检索",
                          "[" + ", ".join(f"{k}:{v}" for k, v in rag_result.layer_hits.items())
                          + f"], 融合 {len(rag_result.passages)} 段")
                else:
                    # ---- 无 RAG 时的兼容路径：裸 web_search ----
                    results = web_search(rewritten, top_k=self.top_k)
                    context_block = format_results(results)
                    if verbose:
                        print(f"      web_search 完成，共 {len(results)} 条结果")
                    _emit("retrieve", "Web 检索", f"共 {len(results)} 条结果")
            else:
                if verbose:
                    print("      无需检索（NO_SEARCH）")
                _emit("rewrite", "无需检索", "NO_SEARCH")

        # 3) 拼装 messages 发给 summary 阶段模型
        messages = [{"role": "system", "content": PROMPTS["summary_system"]}]
        messages.extend(self.memory.get_messages())
        if context_block:
            user_msg = (
                f"[外部资料]\n{context_block}\n\n"
                f"[用户问题]\n{user_input}"
            )
        else:
            user_msg = user_input
        messages.append({"role": "user", "content": user_msg})

        if verbose:
            print(f"[3/3] 调用 summary 模型生成回答 (is_stream={is_stream}) ...")
        _emit("answer", "生成回答", "调用 summary 模型" + ("（流式）" if is_stream else ""))

        # --------- 流式 与 非流式 分路——备什么记忆是一样的 ---------
        if not is_stream:
            answer = llm_chat("summary", messages)
            self.memory.add_user(user_input)
            self.memory.add_assistant(answer)
            self._archive_if_enabled(user_input, answer, rag_result)
            return answer

        # 流式：返回一个生成器。记忆在 generator 内部完成。
        def _gen() -> Iterator[str]:
            buf: list[str] = []
            completed = False     # 标记是否“正常走完”
            try:
                for piece in llm_stream_chat("summary", messages):
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
                    self.memory.add_user(user_input)
                    self.memory.add_assistant(full_answer)
                    # 只有真正完整回答才归档到 L3，避免入库半截答案污染
                    if completed:
                        self._archive_if_enabled(user_input, full_answer, rag_result)
                elif verbose and full_answer and not completed:
                    print(
                        f"\n[stream] 中途打断（save_on_interrupt=False），"
                        f"本轮 {len(full_answer)} 字不入记忆"
                    )
        return _gen()

    # --------------------------------------------------------------------- #
    # 内部辅助
    # --------------------------------------------------------------------- #
    def _archive_if_enabled(self, query: str, answer: str, rag_result) -> None:
        """成功回答后归档，让系统"越用越强"、热点问题二次命中即毫秒返回。

        写两个地方，各司其职：
          1. **L1 QACache**（同步、精确+模糊命中）：把 (原始 query, answer) 写进 L1。
             这是关键——下次问**同样/近似**的问题时，chat() 的 Step0 `qa_cache.get()`
             会精确/模糊命中直接短路，跳过工具路由 / Query 改写 / 分层检索 / LLM，
             把 18s 压到毫秒级。（此前只写 L3 导致第二次重复提问仍走完整 18s 检索。）
          2. **L3 History**（异步、向量库）：把 query+answer+sources 存进历史层，
             作为**融合检索的一路**丰富后续**不同**问题的召回，但它不会短路整条链路。

        注意：L1 用原始 user_input 作 key（Step0 也用原始 query 查），二者对齐才能命中；
        若用改写后的 query 作 key 会与查询侧不一致，导致命中不了。
        """
        if self.retriever is None or not answer:
            return

        # 1) 写 L1 QACache —— 让重复/近似提问下次毫秒短路
        try:
            self.qa_cache.add(query, answer)
        except Exception as e:
            print(f"[agent] 写入 L1 QACache 失败: {e}")

        # 2) 归档到 L3 History（异步）—— 丰富后续不同问题的召回
        sources = None
        if rag_result is not None and getattr(rag_result, "passages", None):
            sources = [p.to_dict() for p in rag_result.passages]
        try:
            self.retriever.archive(query, answer, sources=sources)
        except Exception as e:
            print(f"[agent] archive 到 L3 失败: {e}")

    def reset(self) -> None:
        self.memory.clear()

    def warmup(self, probe_query: str = "预热", verbose: bool = True) -> None:
        """预热分层 RAG：提前加载 L2/L5 索引、BGE-M3 权重、二阶 reranker。

        目的：消除**首条查询**的冷启动延迟毛刺（懒加载的 GB 级 FAISS/SQLite +
        模型权重首次 touch 会多花 3-8s）。建议在 CLI/服务进入交互循环前调用一次，
        让首查询延迟与稳态延迟基本一致。

        无 RAG（enable_rag=False）时为空操作。预热内部各组件独立容错，
        某层未构建索引也不会影响启动。
        """
        if self.retriever is None:
            return
        try:
            self.retriever.warmup(probe_query=probe_query, verbose=verbose)
        except Exception as e:
            print(f"[agent] warmup 异常（不影响后续查询）: {e}")

    def close(self) -> None:
        """优雅关闭：把 L3 增量 worker 队列 flush 后再退出。"""
        if self.retriever is not None:
            try:
                self.retriever.close()
            except Exception as e:
                print(f"[agent] retriever.close 异常: {e}")
