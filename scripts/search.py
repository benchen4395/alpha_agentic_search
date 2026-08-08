# scripts/search.py
"""可被外部 Skill 调用的命令行检索脚本。

用法:
    python scripts/search.py "你的问题"
    python scripts/search.py "你的问题" --rewrite-type 0   # 仅规则
    python scripts/search.py "你的问题" --rewrite-type 1   # 仅 LLM
    python scripts/search.py "你的问题" --rewrite-type 2   # 混合（默认）

输出 JSON:
    {
      "rewritten":    "...",
      "rewrite_type": 2,
      "results":      [{title,url,snippet}, ...],
      "answer":       "...",

      # ---- P0.5 新增（旧字段全部保留，只做增量）----
      "sources":      [{id,title,url,domain,layer,layer_label,
                        confidence,risks,snippet,cited}, ...],
      "citations":    [{source_id,start,end,raw,valid,sentence}, ...],
      "confidence":   0.93,      # 整体证据置信度（跨层校准后，见 rag/calibration.py）
      "low_evidence": false,     # 证据不足信号（abstention）
      "metrics":      {invalid_citations, citation_coverage, ...},

      # ---- P2-3 新增 ----
      "followups":    ["...", "..."]   # 追问推荐（可直接当新 query 发起）
    }

为什么要输出 sources/citations：这个脚本是本项目对外的**准 API 接口**
（被外部 Skill 调用）。改造前它只回一个 `answer` 字符串，调用方拿不到出处，
无法做二次校验、也无法在自己的 UI 里渲染来源。现在 `answer` 语义完全不变
（仍是纯文本），新增字段是**纯增量**，老调用方零改动。
"""
import json
import sys
import os
import argparse

# 让 scripts/ 能 import 上层包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import config  # noqa: E402
from agent import AgenticSearchAgent  # noqa: E402
from query_rewriter import query_rewrite_route  # noqa: E402
from searcher import web_search  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="命令行检索 + 回答")
    parser.add_argument("query", help="用户原始问题")
    parser.add_argument(
        "--rewrite-type",
        type=int,
        choices=[0, 1, 2],
        default=config.QUERY_REWRITE_TYPE,
        help=(
            "Query 改写策略：0=仅规则(shorten_query), "
            "1=仅LLM(rewrite_query), 2=混合（默认）"
        ),
    )
    parser.add_argument(
        "--top-k", type=int, default=config.SEARCH_DEFAULT_TOP_K,
        help="检索结果条数",
    )
    parser.add_argument(
        "--no-answer",
        action="store_true",
        help="只输出检索结果，不走 LLM 回答（节省 token）",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_query = args.query.strip()
    if not raw_query:
        print(json.dumps({"error": "请传入 query"}, ensure_ascii=False))
        sys.exit(1)

    # 1) 统一走路由器（混合策略 / 规则 / LLM 三档可选）
    rewritten = query_rewrite_route(
        raw_query,
        rewrite_type=args.rewrite_type,
    )

    # 2) 执行搜索（NO_SEARCH 时跳过）
    results = (
        web_search(rewritten, top_k=args.top_k)
        if rewritten and rewritten.upper() != config.NO_SEARCH_SENTINEL
        else []
    )

    # 3) 是否再走一次 Agent 拿最终答案
    answer = ""
    extra: dict = {}
    if not args.no_answer:
        agent = AgenticSearchAgent(
            top_k=args.top_k,
            rewrite_type=args.rewrite_type,
        )
        # P0.5：return_result=True 拿到结构化结果。
        # AnswerResult.__str__ 返回 .text，所以即使这里当字符串用也不会出错；
        # 但显式取 .text 语义更清楚，也便于把 sources/citations 一并输出。
        result = agent.chat(raw_query, verbose=False, return_result=True)
        answer = result.text
        payload = result.to_dict()
        # 只挑对外有意义的字段（trace 太长、elapsed_ms 每次都变，不放进稳定契约）
        extra = {
            "sources":      payload["sources"],
            "citations":    payload["citations"],
            "confidence":   payload["confidence"],
            "low_evidence": payload["low_evidence"],
            "cache_hit":    payload["cache_hit"],
            "used_tool":    payload["used_tool"],
            "tool_failed":  payload["tool_failed"],
            "layer_hits":   payload["layer_hits"],
            "metrics":      payload["metrics"],
            # P2-3：追问推荐。对外很有用 —— 调用方（外部 Skill）可以直接
            # 把这些问题作为下一轮的 query 发起，不需要自己再调一次 LLM
            # 去想"还能问什么"。`parse_followups` 已保证每条都是
            # 能独立看懂的问句（无指代词），可直接喷给检索。
            "followups":    payload["followups"],
        }
        # 归档队列 flush，避免脚本退出时丢掉本轮的 L3 写入
        try:
            agent.close()
        except Exception:
            pass

    out = {
        "rewritten":    rewritten,
        "rewrite_type": args.rewrite_type,
        "results":      results,
        "answer":       answer,
        **extra,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
