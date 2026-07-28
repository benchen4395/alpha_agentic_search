# main.py
"""命令行入口：进入交互式 Alpha Agentic Search 对话（CLI 模式）。

Claude Code 风格：Agent 每个流水线步骤（路由/改写/检索/回答）
会实时打印为一行 trace，并标注 ⏱️ 耗时。

启动方式：
    python main.py                  # CLI 模式（终端交互）

CLI 开关：
    :stream on|off            切换流式 / 非流式
    :save-interrupt on|off    流式中途打断时，是否把已收到部分写入记忆

Web 图形界面请用 main_web.py：
    python main_web.py --port 7860
"""
import sys
import re
import argparse
from agent import AgenticSearchAgent
from configs.models_config import STAGES


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #
def _banner(is_stream: bool, save_on_interrupt: bool) -> str:
    lines = [
        "=" * 60,
        " Agentic Search CLI (BenBot)",
        "-" * 60,
        " 各阶段模型配置 (修改 models_config.py 即可切换)",
    ]
    for stage, cfg in STAGES.items():
        lines.append(
            f"   - {stage:<10s}: {cfg['provider']:<8s} | {cfg['model']}"
        )
    lines += [
        "-" * 60,
        f" 输出模式      : {'STREAM (流式)' if is_stream else 'NON-STREAM (一次性)'}",
        f" 打断入库策略  : {'SAVE_ON_INTERRUPT (保留已收到部分)' if save_on_interrupt else 'DISCARD (本轮废弃)'}",
        "-" * 60,
        " 命令:",
        "   exit                       退出",
        "   clear                      清空记忆",
        "   config                     显示当前各阶段模型 / 模式",
        "   :stream on|off             切换流式输出",
        "   :save-interrupt on|off     流式中途打断时是否入库",
        "=" * 60,
    ]
    return "\n".join(lines)


def _print_stream(gen) -> str:
    """逐 token 打印生成器，支持 Ctrl-C 优雅打断。"""
    buf: list[str] = []
    print("\nBot > ", end="", flush=True)
    try:
        for piece in gen:
            print(piece, end="", flush=True)
            buf.append(piece)
    except KeyboardInterrupt:
        print("\n[stream] Ctrl-C 被打断")
        gen.close()
    print("\n")
    return "".join(buf)


def _parse_bool(s: str) -> bool | None:
    if s in {"on", "true", "1", "yes", "y"}:
        return True
    if s in {"off", "false", "0", "no", "n"}:
        return False
    return None


# ---- Claude Code 风格：各步骤图标（CLI / Web 共用） ----
STAGE_ICON = {
    "cache":    "⚡",
    "router":   "🔀",
    "tool":     "🛠️",
    "rewrite":  "✏️",
    "retrieve": "📚",
    "answer":   "💬",
}


def _fmt_elapsed(ms) -> str:
    """把毫秒格式化成人类友好的耗时字符串。"""
    if ms is None:
        return ""
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms / 1000:.1f}s"


def _cli_event_printer(ev: dict) -> None:
    """CLI 端事件渲染：统一步骤 trace（含 ⏱️ 耗时）。"""
    icon = STAGE_ICON.get(ev.get("stage", ""), "•")
    title = ev.get("title", "")
    detail = ev.get("detail", "")
    elapsed = _fmt_elapsed(ev.get("elapsed_ms"))
    tail = f"  ⏱️ {elapsed}" if elapsed else ""
    line = f"  {icon} {title}"
    if detail:
        line += f" → {detail}"
    print(f"{line}{tail}")


# --------------------------------------------------------------------------- #
# CLI 交互
# --------------------------------------------------------------------------- #
def run_cli() -> None:
    is_stream = False
    save_on_interrupt = True

    print(_banner(is_stream, save_on_interrupt))
    agent = AgenticSearchAgent()

    # ---- 启动预热（warmup）----
    # 分层 RAG 的 L2/L5 索引、BGE-M3 权重、二阶 reranker 都是懒加载：不预热的话，
    # 你输入的【第一条问题】会额外多等 3-8s（加载 GB 级 FAISS/SQLite + 模型权重
    # 首次 touch + kernel 编译）。这里在进入交互循环前主动拉起一遍，让首查询和
    # 后续查询延迟基本一致（对应 rag/scripts/07_demo_retrieve.py 的 warmup_all）。
    # 未构建离线索引时各层预热会自动跳过，不影响启动。
    agent.warmup()

    while True:
        try:
            user_input = input("You > ").strip()
            user_input = re.sub(r'[\x00-\x1f\x7f]', '', user_input) # 替换，删除字符串中控制字符，只保留正常可见、可打印字符。
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            break

        if not user_input:
            continue

        low = user_input.lower()
        if low in {"exit", "quit"}:
            print("[bye]")
            break
        if low == "clear":
            agent.reset()
            print("[memory cleared]\n")
            continue
        if low == "config":
            print(_banner(is_stream, save_on_interrupt))
            continue

        if low.startswith(":stream"):
            parts = low.split()
            if len(parts) == 1:
                print(f"[stream] 当前: {'on' if is_stream else 'off'}\n")
            else:
                v = _parse_bool(parts[1])
                if v is None:
                    print("[stream] 用法: :stream on | off\n")
                else:
                    is_stream = v
                    print(f"[stream] {'已开启流式输出' if v else '已切换为非流式输出'}\n")
            continue

        if low.startswith(":save-interrupt") or low.startswith(":save_interrupt"):
            parts = low.split()
            if len(parts) == 1:
                print(f"[save-interrupt] 当前: {'on' if save_on_interrupt else 'off'}\n")
            else:
                v = _parse_bool(parts[1])
                if v is None:
                    print("[save-interrupt] 用法: :save-interrupt on | off\n")
                else:
                    save_on_interrupt = v
                    msg = "中途打断时仍入库" if v else "中途打断时丢弃本轮"
                    print(f"[save-interrupt] {msg}\n")
            continue

        try:
            print()  # trace 与输入之间空一行
            result = agent.chat(
                user_input,
                is_stream=is_stream,
                save_on_interrupt=save_on_interrupt,
                verbose=False,                 # 关闭旧 print，改用统一事件 trace
                on_event=_cli_event_printer,
            )
            if is_stream:
                _print_stream(result)
            else:
                print(f"\nBot > {result}\n")
        except Exception as e:
            print(f"[error] {e}\n", file=sys.stderr)

    # 优雅关闭：flush L3 增量归档队列
    try:
        agent.close()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alpha Agentic Search — CLI 交互入口（Web 界面请用 main_web.py）"
    )
    parser.parse_args()   # 目前 CLI 无额外参数，保留 argparse 以便 -h / 后续扩展
    run_cli()

if __name__ == "__main__":
    main()
