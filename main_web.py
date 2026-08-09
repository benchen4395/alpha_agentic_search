# main_web.py
"""Web 入口：基于 Gradio 的 Claude Code 风格图形化聊天界面。

特性：
    - Agent 每个流水线步骤实时渲染为「可折叠步骤块」（带图标 + ⏱️ 耗时）
    - 最终回答在步骤块下方流式输出
    - 宽扁单屏布局：左侧聊天、右侧配置栏，无需下滑

启动方式：
    python main_web.py                        # 默认 127.0.0.1:7860
    python main_web.py --port 7861            # 指定端口
    python main_web.py --host 0.0.0.0 --share # 对外 + 生成分享链接

CLI 模式请用 main.py：python main.py
"""
import sys
import re
import uuid
import queue
import threading
import argparse

from agent import AgenticSearchAgent
from configs.models_config import STAGES
from main import STAGE_ICON, _fmt_elapsed   # 复用 CLI/Web 共享的图标与耗时格式化


# --------------------------------------------------------------------------- #
# Web 交互 (Gradio)
# --------------------------------------------------------------------------- #
def run_web(host: str = "127.0.0.1", port: int = 7860, share: bool = False,
            agent=None) -> None:
    try:
        import gradio as gr
    except ImportError:
        print("[error] 需要安装 gradio: pip install gradio", file=sys.stderr)
        sys.exit(1)

    if agent is None:
        agent = AgenticSearchAgent()

    # 启动预热：提前拉起分层 RAG 的懒加载重资源（L2/L5 索引 + BGE-M3 + reranker），
    # 消除用户第一次提问时的 3-8s 冷启动毛刺。未构建索引的层会自动跳过。
    agent.warmup()

    def _model_info() -> str:
        rows = [f"- **{stage}**: `{cfg['provider']}` / `{cfg['model']}`"
                for stage, cfg in STAGES.items()]
        return "### 当前模型配置\n" + "\n".join(rows)

    def _to_text(x) -> str:
        """把 Gradio 6 里可能是 list[dict]/dict/str 的 message content 归一为纯文本。"""
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            return x.get("text") or x.get("content") or ""
        if isinstance(x, list):
            return "".join(_to_text(item) for item in x)
        return str(x)

    # ══════════════════════════════════════════════════════════════════════
    # 来源面板渲染（Web 端）
    # ══════════════════════════════════════════════════════════════════════
    def _render_sources_md(result) -> str:
        """把 AnswerResult 的来源渲染成 Markdown（Perplexity 风格来源卡片）。

        若不接住结构化结果，Web UI **一条来源都显示不出来** —— `chat()` 默认只返回 `str`，
        `rag_result.passages`（含 title/url/layer/score）在函数结束时随栈销毁，
        前端根本拿不到。现在通过 `AnswerResult.sources` 拿到，并且每条 source 的
        `id` 与 prompt 里 `<doc id="n">` 严格一一对应，所以模型写的 `[n]`
        可以精确映射到 URL。

        渲染策略：
          * 优先展示**被引用**的来源（`cited=True`），它们是答案的真正依据；
          * 未被引用的折叠成一行统计（既保持界面干净，又暴露检索精度指标）；
          * 含注入风险的来源打 ⚠️，让用户知道该页面试图操纵模型；
          * 无效引用编号单独告警 —— 这是「引用幻觉」的直接证据。
        """
        if result is None or not getattr(result, "sources", None):
            return ""

        lines: list[str] = []
        for s in result.cited_sources:
            risk = " `⚠️ 含可疑指令`" if s.risks else ""
            label = s.title[:60] or s.display_name
            if s.is_clickable:
                head = f"**[{s.id}]** [{label}]({s.url})  \n`{s.domain}`"
            else:
                head = f"**[{s.id}]** {label}"
            lines.append(
                f"{head} · {s.layer_label} · 置信度 `{s.confidence:.2f}`{risk}"
            )

        unused = len(result.uncited_sources)
        if unused:
            lines.append(
                f"<sub>另有 {unused} 条检索到但未被引用"
                f"（引用覆盖率 {result.citation_coverage:.0%}）</sub>"
            )

        flags = [f"整体证据置信度 `{result.confidence:.2f}`"]
        if result.low_evidence:
            flags.append("⚠️ 证据不足，回答可能不完整")
        if result.invalid_citation_count:
            flags.append(
                f"⚠️ {result.invalid_citation_count} 处引用编号不存在（模型幻觉）"
            )
        if result.tool_failed:
            flags.append("⚠️ 工具调用失败，已降级为检索")
        lines.append("<sub>" + " · ".join(flags) + "</sub>")
        return "\n\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════
    # 追问推荐（Web 端）
    # ═══════════════════════════════════════════════════════════════════════
    def _render_followups_md(result) -> str:
        """把追问推荐渲染成 Markdown。

        与来源面板分开渲染（而不是拼在 `_render_sources_md` 里），
        因为两者的**触发条件不同**：来源面板依赖 `sources`（闲聊 /
        NO_SEARCH 时为空），而追问只依赖 `followups` —— 即使本轮没有任何
        外部资料，模型仍可能给出有价值的追问。合并会让它被同一个
        `if sources` 卡住，闲聊场景下永远不展示。

        展示形式用有序列表而不是 gr.Button：做成真按钮需要动态创建
        组件并绑定事件，而 Gradio 的组件数量必须在 Blocks 构造时确定，
        动态追加需要预先占位 + visible 切换，复杂度远高于收益。
        列表已经能让用户看到并复制，是成本最低的方案。
        """
        if result is None:
            return ""
        fups = getattr(result, "followups", None)
        if not fups:
            return ""
        lines = [f"{i}. {q}" for i, q in enumerate(fups, 1)]
        return "\n".join(lines)

    def bot_reply(history, is_stream, save_on_interrupt, session_id):
        """核心回调：后台线程跑 agent，事件+token 经队列流回前端。

        Claude Code 风格：
            - 每个流水线步骤 → 一条带 metadata 的可折叠"思考块"消息（标题含 ⏱️ 耗时）
            - 最终回答 → 一条普通 assistant 消息（流式追加）
            - 回答之后再追加一个可折叠的"来源"块
            - 最后追加"你可能还想问"（不折叠，保证可发现性）

        Args:
            session_id: 由 `demo.load` 为**每个浏览器会话**独立生成的 UUID。

            为什么必须有它：全进程只有一个 agent，而 agent 内部
            `self.memory` 是**单个** ConversationMemory ——
            A 用户的对话历史会直接进入 B 用户的 rewriter 上下文和 summary
            messages，多用户串味且属于隐私泄漏。现在把 session_id 传下去，
            agent 按 session 分桶存记忆、按 namespace 隔离 L1/L3。
        """
        user_msg = re.sub(r'[\x00-\x1f\x7f]', '', _to_text(history[-1]["content"]).strip())
        if not user_msg:
            history.append({"role": "assistant", "content": ""})
            yield history
            return

        # 兜底：若 demo.load 尚未写入（或未来改动弄丢了绑定），当场补一个，
        # 结果退化为“本轮独立”而不会变成全局共享同一个 namespace。
        if not isinstance(session_id, str) or not session_id:
            session_id = _new_session_id()

        ev_queue: "queue.Queue" = queue.Queue()
        _DONE = object()
        # 用 dict 而非局部变量：worker 在另一个线程里跑，需要可变容器回传结果
        holder: dict = {"result": None}

        def _worker():
            """后台线程：把事件与 token 统一塞进队列。"""
            try:
                result = agent.chat(
                    user_msg,
                    is_stream=is_stream,
                    save_on_interrupt=save_on_interrupt,
                    verbose=False,
                    on_event=lambda ev: ev_queue.put(("event", ev)),
                    # ---- 会话隔离 ----
                    session_id=session_id,
                    # ⚠️ 刻意**不传** user_id。
                    #
                    # 本项目没有登录体系，session_id 是每个浏览器会话随机
                    # 生成的 UUID。若把它兼作 user_id，L1/L3 的 namespace
                    # 就变成 `u:<随机值>`，每次刷新页面都换一个，后果是：
                    #   ① 仓库自带的全局预热问答（无 namespace）永远命不中；
                    #   ② 每个新会话从零积累，"越用越强"退化成"每次重来"；
                    #   ③ 缓存里持续堆积永不再被访问的死条目。
                    #
                    # 只传 session_id 时 namespace = `s:<uuid>`，语义是
                    # 「会话级隔离」—— 同一会话内复用、跨会话互不可见，
                    # 会话结束即自然失效，这正是匿名访客场景应有的行为。
                    # 而 `agent._l1_get()` 会在租户未命中时回退查一次全局
                    # 公共池，于是预热问答也能命中（写入仍只进租户空间，
                    # 用户产生的答案不会进全局池，隐私不受影响）。
                    #
                    # 接入真实登录体系后，把登录态的稳定用户 ID 传给
                    # user_id 即可获得「跨会话复用个人积累」的用户级隔离。
                    # ---- 拿结构化结果以渲染来源面板 ----
                    return_result=True,
                )
                if is_stream:
                    # result 是 StreamingAnswer：迭代等价于生成器，
                    # 耗尽后 .result 才被填成完整 AnswerResult（含 citations）
                    for piece in result:
                        ev_queue.put(("token", piece))
                    holder["result"] = getattr(result, "result", None)
                else:
                    # AnswerResult.__str__ 返回 .text，前端按文本消费
                    ev_queue.put(("token", result.text))
                    holder["result"] = result
            except Exception as e:
                ev_queue.put(("error", f"⚠️ **Error**: {e}"))
            finally:
                ev_queue.put(("done", _DONE))

        threading.Thread(target=_worker, daemon=True).start()

        answer_buf = ""
        answer_started = False
        while True:
            kind, payload = ev_queue.get()
            if kind == "done":
                break
            if kind == "event":
                # sources 事件不渲染成步骤块（下面用独立的来源面板呈现），
                # 否则同一份信息会重复出现两次。
                # followup 事件同理 —— 下面有独立的追问推荐区块。
                if payload.get("stage") in ("sources", "followup"):
                    continue
                icon = STAGE_ICON.get(payload.get("stage", ""), "•")
                elapsed = _fmt_elapsed(payload.get("elapsed_ms"))
                tail = f"  ⏱️ {elapsed}" if elapsed else ""
                title = f"{icon} {payload.get('title', '')}{tail}"
                # 带 metadata.title 的消息 → Gradio 渲染成可折叠块
                history.append({
                    "role": "assistant",
                    "content": payload.get("detail", "") or " ",
                    "metadata": {"title": title},
                })
                yield history
            elif kind == "token":
                if not answer_started:
                    history.append({"role": "assistant", "content": ""})
                    answer_started = True
                answer_buf += payload
                history[-1]["content"] = answer_buf
                yield history
            elif kind == "error":
                history.append({"role": "assistant", "content": payload})
                yield history

        # ---- 回答结束后追加来源面板 ----
        # 必须放在 while 循环之后：引用 [n] 只能在**完整答案文本**就绪后解析，
        # 流式过程中拿不到 citations。
        src_md = _render_sources_md(holder["result"])
        if src_md:
            res = holder["result"]
            n_cited = len(res.cited_sources)
            badge = "⚠️ " if (res.low_evidence or res.has_risky_source) else ""
            history.append({
                "role": "assistant",
                "content": src_md,
                "metadata": {
                    "title": f"{badge}🔖 来源 · {n_cited}/{len(res.sources)} 条被引用"
                },
            })
            yield history

        # ---- 追问推荐 ----
        # 同样必须在 while 之后：追问是从完整输出里剥离出来的，
        # 流式过程中 `holder["result"]` 还是 None。
        #
        # 用 metadata.title 做成**默认展开**以外的可折叠块会降低可发现性，
        # 而追问的价值就在"被看到"，所以这里**不加** metadata ——
        # 作为普通消息直接展示在答案下方，与 Perplexity 的交互一致。
        fup_md = _render_followups_md(holder["result"])
        if fup_md:
            history.append({
                "role": "assistant",
                "content": "**🔮 你可能还想问**\n\n" + fup_md,
            })
            yield history

    def clear_memory(session_id):
        """清空**当前会话**的记忆（不再影响其它用户）。"""
        agent.reset(session_id=session_id)
        return "🧹 本会话记忆已清空"

    def _new_session_id() -> str:
        """为每个浏览器会话生成独立标识。

        `gr.State` 的值是**每个前端会话独立**的（不是全局共享），
        所以这里返回的 UUID 天然做到一人一份。
        """
        return uuid.uuid4().hex[:16]

    # 宽扁布局：撑满宽度、压缩纵向间距，让整页一屏可见无需下滑
    _CSS = """
    /* 容器：铺满整个视口的宽与高，去掉 Gradio 默认的居中与最大宽度限制 */
    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
        min-height: 100vh !important;
        padding: 4px 12px 0 12px !important;
        margin: 0 !important;
    }
    #app-header {margin: 0 0 4px 0 !important;}
    #app-header h1 {font-size: 20px !important; margin: 0 !important;}
    #app-header p {font-size: 12px !important; margin: 2px 0 0 0 !important; opacity: .7;}
    .gr-accordion {margin: 0 !important;}
    footer {display: none !important;}
    /* 聊天窗口：高度跟随视口自适应（视口高度 - 顶部标题与输入框占用），
       让对话区尽可能撑满整屏纵向空间；用户手动缩放窗口也会同步伸缩。 */
    #chatbox {
        height: calc(100vh - 180px) !important;
        min-height: 360px !important;
    }
    #chatbox .bubble-wrap {height: 100% !important;}
    """

    with gr.Blocks(title="Agentic Search (BenBot)", fill_height=True, fill_width=True) as demo:
        gr.Markdown(
            "# 🔎 Agentic Search — BenBot\n"
            "Claude Code 风格：Agent 每步执行过程实时透明、可折叠、带耗时。",
            elem_id="app-header",
        )

        # 主区：左侧聊天占大头，右侧窄栏放配置/开关
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    elem_id="chatbox",
                    show_label=False,
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="请输入问题，Enter 发送…",
                        show_label=False,
                        lines=1,
                        scale=6,
                        autofocus=True,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1, min_width=80)

            with gr.Column(scale=1, min_width=220):
                with gr.Row():
                    clear_btn = gr.Button("清空记忆 🧹", size="sm")
                    reset_ui = gr.Button("清空对话 🗑️", size="sm")
                stream_toggle = gr.Checkbox(value=True, label="流式输出")
                save_toggle = gr.Checkbox(value=True, label="打断时保留入库")
                status = gr.Markdown("")
                with gr.Accordion("⚙️ 当前模型配置", open=False):
                    gr.Markdown(_model_info())

        # 每个浏览器会话独立的 session_id。
        #
        # ⚠️ 不能写 `gr.State(value=_new_session_id)`：
        # Gradio **不会**调用这个 callable，而是把**函数对象本身**当成初始值
        # 存进 State（gradio 6.19 实测：`gr.State(value=f).value` 就是 `f`）。
        # 后果极其严重且完全静默：
        #   · 所有浏览器会话拿到的是**同一个**函数对象 → session_id 全局相同
        #     → 会话隔离形同虚设，A 用户的记忆/L1/L3 直接串给 B 用户；
        #   · namespace 变成 `s:<function _new_session_id at 0x...>`，
        #     内存地址还会随进程重启变化 → 缓存条目永久失效并持续堆积。
        # 实测证据：仓库自带的 data/qa_cache 里真的存在这类 key：
        #     orig::s:<function run_web.<locals>._new_session_id at 0x40880eb90>::11等于几
        #
        # 正确做法：State 初值留空，用 `demo.load`（每个会话建立时触发一次）
        # 把真正的 UUID 灌进去。
        session_state = gr.State(value="")

        def user_submit(user_msg, history):
            history = history or []
            history.append({"role": "user", "content": user_msg})
            return "", history

        msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_reply,
            [chatbot, stream_toggle, save_toggle, session_state],
            chatbot,
        )
        send_btn.click(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_reply,
            [chatbot, stream_toggle, save_toggle, session_state],
            chatbot,
        )
        clear_btn.click(clear_memory, inputs=session_state, outputs=status)
        reset_ui.click(lambda: [], outputs=chatbot)

        # 每个前端会话建立时生成一次 UUID 并写入 State（见上方说明）。
        demo.load(_new_session_id, inputs=None, outputs=session_state)

    try:
        demo.queue().launch(server_name=host, server_port=port, share=share,
                            theme=gr.themes.Monochrome(), css=_CSS)
    finally:
        # 退出前 flush L3 归档队列。
        #
        # `launch()` 是阻塞调用，Ctrl-C 会以 KeyboardInterrupt 冒出来；
        # 若不在这里 close，IncrementalWorker 是 daemon 线程，会随主进程
        # 直接被杀，队列里**尚未落盘的归档事件全部丢失** —— 表现为
        # "刚问过的问题重启后 L3 检索不到"，而且没有任何报错。
        try:
            agent.close()
        except Exception as e:
            print(f"[web] agent.close() 异常（不影响退出）: {e}")


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic Search Web 图形界面 (Gradio)")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--share", action="store_true", help="生成 Gradio 分享链接")
    args = parser.parse_args()
    run_web(host=args.host, port=args.port, share=args.share)


if __name__ == "__main__":
    main()
