# -*- coding: utf-8 -*-
"""三臂对照评测：现状 / 建议1(并列并发) / 建议3(loop agent)。

在 GAIA + BrowseComp-ZH 真实数据上测准确率与耗时。

⚠️ 三个 arm 全部实现在本文件内，**不修改任何生产代码**。
目的是先拿数据决策，避免为了一个还没验证的方案去动主链路。

arm 定义
────────────────────────────────────────────────────────────────────
baseline : 现状。retrieve(query) 一次，直接作答。
suggest1 : 并列实体并发子检索。复用已有的
           `entities.extract_parallel_entities`（误报率 0、0.024ms），
           抽出 ≥2 个并列实体时对每个实体拼子 query **并发**检索，
           合并去重后一起作答。关键：并发（耗时是 max 而非 sum）、
           且**不需要 LLM**。
suggest3 : loop agent。planner 判断证据够不够并给出下一跳 query，
           最多 MAX_HOP 跳。每多一跳要一次 LLM 调用（实测 ~1.9s）。

用法
────────────────────────────────────────────────────────────────────
    python evals/run_multihop_eval.py --dataset gaia --limit 20
    python evals/run_multihop_eval.py --dataset bcz  --limit 20
    python evals/run_multihop_eval.py --dataset both --limit 15 \
        --arms baseline,suggest1,suggest3
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.datasets import Case, judge, load_bcz, load_gaia  # noqa: E402
from src.configs.models_config import override_stage  # noqa: E402
from src.configs.prompts import build_summary_system  # noqa: E402
from src.core.llm_client import chat as llm_chat  # noqa: E402
from src.pipeline.evidence import build_evidence_block, build_user_message  # noqa: E402
from src.rag import answerability as rag_ans  # noqa: E402
from src.rag import entities as rag_entities  # noqa: E402
from src.rag.retriever import LayeredRetriever  # noqa: E402

MAX_HOP = 3          # loop agent 最大跳数（含第一跳）

_PLANNER_SYS = (
    "你是检索规划器。给定用户问题和已检索到的证据，判断证据是否足以完整回答问题。\n"
    "若不足，给出**下一步应该检索的具体 query**——必须是可独立检索的完整问句，"
    "要把已经从证据中确认的中间实体名代入进去，不要输出无主语的词组。\n"
    "严格只输出一行 JSON：{\"enough\": true/false, \"next_query\": \"...\"}"
)


# 单次 LLM 调用的墙钟上限（秒）。可用 --llm-timeout 覆盖。
LLM_TIMEOUT = 120.0
_LLM_POOL = ThreadPoolExecutor(max_workers=4)


def _llm(messages: list[dict], *, timeout: float | None = None) -> str:
    """带墙钟超时的 LLM 调用。

    ⚠️ 生产的 `llm_client.chat()` **没有超时**。实测 BrowseComp-ZH 的
    长题干 + 6 段网页证据会让单次 summary 卡到 444s 甚至 >600s，
    一题就把评测预算烧光。这里只在评测侧加护栏，不动生产代码 ——
    但"生产链路缺 LLM 超时"本身是个值得单独修的隐患。

    注意 ThreadPoolExecutor 无法中断已启动的任务，超时后线程仍在跑，
    所以这里只保证**评测主流程**不被拖死，池子会自然泄漏几个线程。
    """
    tmo = LLM_TIMEOUT if timeout is None else timeout
    fut = _LLM_POOL.submit(llm_chat, "summary", messages)
    try:
        return fut.result(timeout=tmo)
    except FuturesTimeout:
        raise TimeoutError(f"LLM 调用超过 {tmo:.0f}s") from None


def _answer(passages, question: str) -> str:
    """用给定证据调 summary 模型作答（与生产同一套 prompt）。"""
    block, _ = build_evidence_block(passages)
    return _llm([
        {"role": "system", "content": build_summary_system()},
        {"role": "user", "content": build_user_message(question, block)},
    ])


def _dedup(passages):
    seen, uniq = set(), []
    for p in passages:
        k = (getattr(p, "text", "") or "")[:120]
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


# ════════════════════════════════════════════════════════════════════
#                               arm 实现
# ════════════════════════════════════════════════════════════════════
def arm_baseline(r: LayeredRetriever, q: str) -> tuple[str, dict]:
    res = r.retrieve(q)
    return _answer(res.passages, q), {
        "hops": 1, "llm_calls": 1, "cov": res.term_coverage, "subs": [],
    }


def arm_suggest1(r: LayeredRetriever, q: str) -> tuple[str, dict]:
    """建议1：并列实体 → 并发子检索。

    子 query 用「实体 + 原问题」而不是裸实体：裸实体（"日本"）会召回
    该实体的泛泛介绍，而我们要的是它在**这个问题维度**上的事实，
    把原问题拼进去检索的语义中心才对。
    """
    res = r.retrieve(q)
    ents = rag_entities.extract_parallel_entities(q)
    passages, subs = list(res.passages), []

    if len(ents) >= 2:
        subs = [f"{e} {q}" for e in ents]
        # 并发：耗时是 max 而非 sum，这是本方案相对 loop 的核心优势
        with ThreadPoolExecutor(max_workers=min(len(subs), 5)) as ex:
            for sr in ex.map(lambda s: r.retrieve(s), subs):
                passages.extend(sr.passages)
        passages = _dedup(passages)

    return _answer(passages, q), {
        "hops": 1, "llm_calls": 1, "cov": res.term_coverage, "subs": subs,
    }


def arm_suggest3(r: LayeredRetriever, q: str) -> tuple[str, dict]:
    """建议3：loop agent（gate 触发 + planner 分解 + MAX_HOP 上限）。"""
    res = r.retrieve(q)
    passages = list(res.passages)
    subs: list[str] = []
    llm_calls, hops = 1, 1          # llm_calls 含最后那次 summary

    for _ in range(MAX_HOP - 1):
        insuf, _cov, _missing = rag_ans.is_evidence_insufficient(q, passages)
        if not insuf:
            break                   # gate 未触发 → 简单题零额外成本
        block, _ = build_evidence_block(passages)
        llm_calls += 1
        try:
            raw = _llm([
                {"role": "system", "content": _PLANNER_SYS},
                {"role": "user", "content": f"{block}\n\n问题：{q}"},
            ])
            m = re.search(r"\{.*\}", raw, re.S)
            plan = json.loads(m.group(0)) if m else {}
        except Exception as e:
            print(f"      [planner] 失败，停止迭代: {type(e).__name__}")
            break
        if plan.get("enough") or not plan.get("next_query"):
            break
        nq = str(plan["next_query"]).strip()
        if nq in subs:              # planner 原地打转，别白烧一跳
            break
        subs.append(nq)
        hops += 1
        passages = _dedup(passages + list(r.retrieve(nq).passages))

    return _answer(passages, q), {
        "hops": hops, "llm_calls": llm_calls, "cov": res.term_coverage,
        "subs": subs,
    }


ARMS = {"baseline": arm_baseline, "suggest1": arm_suggest1,
        "suggest3": arm_suggest3}


def main() -> int:
    global LLM_TIMEOUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="both", choices=["gaia", "bcz", "both"])
    ap.add_argument("--arms", default="baseline,suggest1,suggest3")
    ap.add_argument("--limit", type=int, default=15, help="每个数据集取几题")
    ap.add_argument("--levels", default="1,2,3", help="GAIA level 过滤")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="evals/out/eval_result.json")
    ap.add_argument("--llm-timeout", type=float, default=LLM_TIMEOUT,
                    help="单次 LLM 调用墙钟上限（秒）")
    ap.add_argument("--max-tokens", type=int, default=600,
                    help="summary 输出上限（仅评测用，控制单题耗时）")
    args = ap.parse_args()

    LLM_TIMEOUT = args.llm_timeout

    # 限制 summary 的输出长度。
    # ⚠️ 这是**评测吞吐**的需要，不是在调优生产参数：实测 LLM 延迟由
    # **输出**长度主导（输入 200 / 2000 / 6000 字时分别 3.8s / 1.9s / 1.7s，
    # 几乎无关），而生产 prompt 会让模型写出带引用标注 + 追问推荐的长
    # 段落，BrowseComp-ZH 的长题干下实测单次可以卡到 444s，一题就把
    # 整晚预算烧光。判分只看标准答案是否出现，短答案完全够用。
    # 三个 arm 用的是同一套限制，不影响相对比较。
    override_stage("summary", extra={"max_tokens": args.max_tokens})
    print(f"[eval] summary max_tokens={args.max_tokens}（仅评测用，"
          f"生产配置不变）")

    rnd = random.Random(args.seed)
    cases: list[Case] = []
    if args.dataset in ("gaia", "both"):
        g = load_gaia(levels=tuple(args.levels.split(",")))
        rnd.shuffle(g)
        cases += g[:args.limit]
    if args.dataset in ("bcz", "both"):
        b = load_bcz()
        rnd.shuffle(b)
        cases += b[:args.limit]

    # 关掉 L1/L3 读取：否则前一个 arm 的答案会经缓存/历史喂给后一个 arm，
    # 三臂就不再是独立对照（实测 L1 命中会直接 0ms 白拿上一 arm 的答案）。
    r = LayeredRetriever(qa_cache=None, enable_l3=False)
    print(f"[eval] {len(cases)} 题 | arms={args.arms} | MAX_HOP={MAX_HOP}")
    print("[eval] 已禁用 L1/L3（避免 arm 之间通过缓存互相污染）")
    r.warmup(verbose=False)

    arm_names = [a for a in args.arms.split(",") if a in ARMS]
    rows: list[dict] = []

    for n, c in enumerate(cases, 1):
        print(f"\n{'='*88}\n[{n}/{len(cases)}] {c.cid} (L{c.level}) "
              f"{c.question[:70]}…\n  gold: {c.answer}")
        for a in arm_names:
            t0 = time.perf_counter()
            try:
                text, meta = ARMS[a](r, c.question)
            except Exception as e:
                # 超时单独标记：它不是"答错"，而是"没算出来"。
                # 把两者混在一起会让准确率失真。
                kind = "⏱ 超时" if isinstance(e, TimeoutError) else "💥"
                print(f"  {a:<9} {kind} {type(e).__name__}: {str(e)[:60]}")
                rows.append({"cid": c.cid, "src": c.source, "level": c.level,
                             "arm": a, "ok": False, "ms": 0.0,
                             "timeout": isinstance(e, TimeoutError),
                             "err": str(e)[:200]})
                continue
            dt = (time.perf_counter() - t0) * 1000
            ok = judge(text, c.answer)
            rows.append({"cid": c.cid, "src": c.source, "level": c.level,
                         "arm": a, "ok": ok, "ms": dt,
                         "hops": meta["hops"], "llm": meta["llm_calls"],
                         "cov": meta["cov"], "subs": meta["subs"],
                         "answer": text[:400]})
            extra = f" subs={meta['subs'][:2]}" if meta["subs"] else ""
            print(f"  {a:<9} {'✅' if ok else '❌'}  {dt:7.0f}ms  "
                  f"cov={meta['cov']:.2f} hops={meta['hops']} "
                  f"llm={meta['llm_calls']}{extra}")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    _report(rows, arm_names)
    print(f"\n明细已存 {outp}")
    return 0


def _report(rows: list[dict], arm_names: list[str]) -> None:
    def agg(sel: list[dict]) -> str:
        if not sel:
            return f"{'—':>16}"
        ok = sum(1 for x in sel if x["ok"])
        return f"{ok}/{len(sel)} ({ok/len(sel)*100:.0f}%)".rjust(16)

    print(f"\n\n{'='*88}\n准确率\n{'='*88}")
    print(f"{'arm':<10}{'总体':>16}{'GAIA':>16}{'BrowseComp-ZH':>18}")
    for a in arm_names:
        sel = [x for x in rows if x["arm"] == a]
        print(f"{a:<10}{agg(sel)}"
              f"{agg([x for x in sel if x['src']=='gaia'])}"
              f"{agg([x for x in sel if x['src']=='bcz'])[:18].rjust(18)}")

    print(f"\n{'='*88}\nGAIA 按 Level\n{'='*88}")
    print(f"{'arm':<10}" + "".join(f"{'L'+l:>16}" for l in ("1", "2", "3")))
    for a in arm_names:
        sel = [x for x in rows if x["arm"] == a and x["src"] == "gaia"]
        print(f"{a:<10}" + "".join(
            agg([x for x in sel if x["level"] == l]) for l in ("1", "2", "3")))

    print(f"\n{'='*88}\n耗时 / 成本（仅统计正常完成的题）\n{'='*88}")
    print(f"{'arm':<10}{'中位':>11}{'均值':>11}{'P90':>11}"
          f"{'最差':>11}{'LLM/题':>10}{'跳/题':>9}{'超时':>7}")
    for a in arm_names:
        allrows = [x for x in rows if x["arm"] == a]
        sel = [x for x in allrows if x.get("ms")]
        n_to = sum(1 for x in allrows if x.get("timeout"))
        if not sel:
            continue
        ts = sorted(x["ms"] for x in sel)
        p90 = ts[min(int(len(ts) * 0.9), len(ts) - 1)]
        print(f"{a:<10}{statistics.median(ts):>9.0f}ms"
              f"{statistics.mean(ts):>9.0f}ms{p90:>9.0f}ms{max(ts):>9.0f}ms"
              f"{statistics.mean([x.get('llm',1) for x in sel]):>10.2f}"
              f"{statistics.mean([x.get('hops',1) for x in sel]):>9.2f}"
              f"{n_to:>7}")

    base = [x for x in rows if x["arm"] == "baseline"]
    if base and len(arm_names) > 1:
        print(f"\n{'='*88}\n相对 baseline 的逐题变化（谁修好了、谁弄坏了）\n{'='*88}")
        bmap = {x["cid"]: x["ok"] for x in base}
        for a in arm_names:
            if a == "baseline":
                continue
            sel = [x for x in rows if x["arm"] == a]
            fixed = [x["cid"] for x in sel if x["ok"] and not bmap.get(x["cid"])]
            broke = [x["cid"] for x in sel if not x["ok"] and bmap.get(x["cid"])]
            print(f"{a:<10} 修好 {len(fixed)} {fixed[:6]}  "
                  f"弄坏 {len(broke)} {broke[:6]}")


if __name__ == "__main__":
    raise SystemExit(main())
