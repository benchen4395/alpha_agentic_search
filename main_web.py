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

    def bot_reply(history, is_stream, save_on_interrupt):
        """核心回调：后台线程跑 agent，事件+token 经队列流回前端。

        Claude Code 风格：
            - 每个流水线步骤 → 一条带 metadata 的可折叠"思考块"消息（标题含 ⏱️ 耗时）
            - 最终回答 → 一条普通 assistant 消息（流式追加）
        """
        user_msg = re.sub(r'[\x00-\x1f\x7f]', '', _to_text(history[-1]["content"]).strip())
        if not user_msg:
            history.append({"role": "assistant", "content": ""})
            yield history
            return

        ev_queue: "queue.Queue" = queue.Queue()
        _DONE = object()

        def _worker():
            """后台线程：把事件与 token 统一塞进队列。"""
            try:
                result = agent.chat(
                    user_msg,
                    is_stream=is_stream,
                    save_on_interrupt=save_on_interrupt,
                    verbose=False,
                    on_event=lambda ev: ev_queue.put(("event", ev)),
                )
                if is_stream:
                    for piece in result:
                        ev_queue.put(("token", piece))
                else:
                    ev_queue.put(("token", result))
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

    def clear_memory():
        agent.reset()
        return "🧹 记忆已清空"

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

        def user_submit(user_msg, history):
            history = history or []
            history.append({"role": "user", "content": user_msg})
            return "", history

        msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_reply, [chatbot, stream_toggle, save_toggle], chatbot
        )
        send_btn.click(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_reply, [chatbot, stream_toggle, save_toggle], chatbot
        )
        clear_btn.click(clear_memory, outputs=status)
        reset_ui.click(lambda: [], outputs=chatbot)

    demo.queue().launch(server_name=host, server_port=port, share=share,
                        theme=gr.themes.Monochrome(), css=_CSS)


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
