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
    """逐 token 打印生成器，支持 Ctrl-C 优雅打断。

    P0.5：`gen` 现在可能是 `StreamingAnswer`（当 return_result=True）。
    它实现了完整的迭代 + close 协议，所以这里**无需改动**即可兼容；
    来源面板由调用方在迭代结束后读 `gen.result` 渲染。
    """
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
    # P0-4：工具调用失败并降级到检索时发出（用 ⚠️ 区别于成功的 🛠️）
    "tool_failed": "⚠️",
    "rewrite":  "✏️",
    "retrieve": "📚",
    "answer":   "💬",
    # P0.5：回答完成后的来源归因步骤
    "sources":  "🔖",
    # P2-3：追问推荐（"你可能还想问"）。耗时恒为 0 —— 它是主答案
    # 那次 LLM 调用的副产品（指令写在 summary prompt 里），无额外调用。
    "followup": "🔮",
    # 归档到 L1/L3（"越用越强"的写入侧）。单独成步是为了让它的耗时
    # 不再被错算进「来源归因」—— 它内部要同步跑一次 BGE-M3 编码。
    "archive":  "📥",
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


def _print_sources(result) -> None:
    """CLI 端来源面板（P0.5）。

    这是 Perplexity 最核心的体验在终端里的等价物：让用户能核实每一句话的出处。
    改造前 `chat()` 只返回 `str`，`rag_result.passages` 随栈销毁，
    调用方**根本拿不到来源**，所以 CLI/Web 都一条来源都显示不出来。

    渲染规则：
      * 只列**被答案真正引用**的来源（`cited=True`），检索到但没用上的
        折叠为一行统计 —— 既清爽，又能让用户感知检索精度。
      * 无效引用编号（模型编造的 [n]）单独告警：这是引用幻觉的直接证据。
      * 含注入风险的来源打 ⚠️，提示用户该来源可能被投毒。
    """
    if result is None or not getattr(result, "sources", None):
        return

    cited = result.cited_sources
    if cited:
        print("  📎 来源")
        for s in cited:
            risk = "  ⚠️含可疑指令" if s.risks else ""
            loc = s.url if s.is_clickable else f"（{s.layer_label}·本地）"
            print(f"     [{s.id}] {s.title[:44] or s.display_name}")
            print(f"         {s.layer_label} · 置信度 {s.confidence:.2f} · {loc}{risk}")

    # 未被引用的来源只报数量：它们占了 token 却没贡献答案，
    # 覆盖率长期偏低说明检索精度需要优化（P2 的 MMR/精排要解决的问题）。
    unused = len(result.uncited_sources)
    if unused:
        print(f"     （另有 {unused} 条检索到但未被引用，引用覆盖率 "
              f"{result.citation_coverage:.0%}）")

    # 置信度与 abstention 提示
    flags = [f"整体置信度 {result.confidence:.2f}"]
    if result.low_evidence:
        flags.append("⚠️ 证据不足，回答可能不完整")
    if result.invalid_citation_count:
        flags.append(f"⚠️ {result.invalid_citation_count} 处引用编号不存在（模型幻觉）")
    if result.tool_failed:
        flags.append("⚠️ 工具调用失败，已降级为检索")
    print(f"     {' · '.join(flags)}")


def _print_followups(result) -> None:
    """CLI 端追问推荐面板（P2-3）。

    为什么与 `_print_sources` 拆开而不合并：
      两者的**触发条件不同**。来源面板依赖 `sources`（闲聊/NO_SEARCH
      时为空），而追问推荐只依赖 `followups`—— 即使本轮没有任何
      外部资料，模型仍可能给出有价值的追问。合并会让两个独立的
      展示区块被同一个 `if` 卡住。

    空列表时**静默返回**：追问是锦上添花，没有就不该占屏，
    更不能打一行"无追问推荐"之类的无用提示。
    """
    if result is None:
        return
    fups = getattr(result, "followups", None)
    if not fups:
        return
    print("  🔮 你可能还想问")
    for i, q in enumerate(fups, 1):
        print(f"     {i}. {q}")


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
            # P0.5：return_result=True 拿到 AnswerResult（含 sources/citations）。
            # 非流式 → 直接是 AnswerResult；流式 → StreamingAnswer（迭代等价于
            # 生成器，耗尽后 .result 可用）。两者的 str()/迭代行为都与改造前一致，
            # 所以下面的打印逻辑改动极小。
            result = agent.chat(
                user_input,
                is_stream=is_stream,
                save_on_interrupt=save_on_interrupt,
                verbose=False,                 # 关闭旧 print，改用统一事件 trace
                on_event=_cli_event_printer,
                return_result=True,
            )
            if is_stream:
                _print_stream(result)
                # 流式的来源面板必须等 token 全部消费完才有 citations
                # （引用只能在完整答案文本就绪后解析），所以放在 _print_stream 之后。
                _print_sources(getattr(result, "result", None))
                # P2-3：追问同理 —— 它是从完整输出里剥离出来的，
                # 流式下只有迭代结束后 `.result` 里才有值。
                _print_followups(getattr(result, "result", None))
                print()
            else:
                # AnswerResult.__str__ 返回 .text，所以 f-string 插值与改造前等价
                print(f"\nBot > {result}\n")
                _print_sources(result)
                _print_followups(result)
                print()
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
