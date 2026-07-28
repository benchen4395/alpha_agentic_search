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
      "answer":       "..."
    }
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
    if not args.no_answer:
        agent = AgenticSearchAgent(
            top_k=args.top_k,
            rewrite_type=args.rewrite_type,
        )
        answer = agent.chat(raw_query, verbose=False)

    out = {
        "rewritten":    rewritten,
        "rewrite_type": args.rewrite_type,
        "results":      results,
        "answer":       answer,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
