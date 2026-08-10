"""
12_demo_kg_query.py
===================

KG 端到端查询 demo（薄封装，实现都在 :mod:`wiki_rag.kg_retriever` 和 :mod:`wiki_rag.hybrid` 里）。

三种运行模式
------------

1. **老 demo**：只查单个 mention 或 qid，用于快速验证 link 层::

       python scripts/12_demo_kg_query.py --mention 苹果 --query "苹果公司的CEO是谁"
       python scripts/12_demo_kg_query.py --qid Q148

2. **端到端 KG 查询**：从自然语言 query 走完整 pipeline::

       python scripts/12_demo_kg_query.py --query "苹果公司的CEO是谁"
       python scripts/12_demo_kg_query.py --query "库克领导的公司总部在哪个城市" --multi-hop
       python scripts/12_demo_kg_query.py --query "..." --method hybrid   # ngram+jieba

3. **混合检索 (KG + Wiki + Web + Rerank)**：接入生产 RAG 的完整链路::

       python scripts/12_demo_kg_query.py --query "量子纠缠是什么" --hybrid
       python scripts/12_demo_kg_query.py --query "..." --hybrid --multi-hop --no-rerank

注：等价的最小混合检索 demo 在 :file:`scripts/07_demo_retrieve.py`，那里已默认加了 KG 通路。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wiki_rag.config import load_config
from wiki_rag.kg_store import KGStore
from wiki_rag.linker import Linker
from wiki_rag.kg_retriever import KGRetriever
from wiki_rag.retriever import WikiRetriever
from wiki_rag.hybrid import hybrid_retrieve, build_default_reranker


# =============================================================================
# 老 demo：mention / qid 层测试
# =============================================================================

def demo_link(linker: Linker, mention: str, query: str | None):
    print(f"\n=== link(mention={mention!r}, query={query!r}) ===")
    results = linker.link_and_expand(mention, query_context=query, top_k=3)
    if not results:
        print("  no candidates.")
        return
    for i, r in enumerate(results):
        print(f"\n[{i+1}] qid={r['qid']}  label={r['label_zh']}  "
              f"score={r['score']:.4f}  source={r['source']}")
        if r.get("description"):
            print(f"      desc: {r['description']}")
        if r.get("context"):
            print("      --- context ---")
            for ln in r["context"].splitlines():
                print(f"      {ln}")


def demo_qid(kg: KGStore, qid: str):
    print(f"\n=== triples of {qid} ===")
    print(kg.to_context(qid))


# =============================================================================
# 端到端 KG demo：直接调用 KGRetriever
# =============================================================================

def demo_end_to_end(kg_retr: KGRetriever,
                    query: str,
                    multi_hop: bool = False,
                    max_hops: int = 2,
                    top_k: int = 5):
    """演示 KGRetriever 的端到端流程。"""
    print(f"\n=== KG end-to-end (query={query!r}, multi_hop={multi_hop}) ===")
    t0 = time.perf_counter()
    docs = kg_retr.retrieve(query, top_k=top_k,
                            multi_hop=multi_hop, max_hops=max_hops)
    dt = time.perf_counter() - t0
    print(f"  total latency: {dt*1000:.1f} ms   hits: {len(docs)}")
    if not docs:
        print("  (no entities found)")
        return
    for i, d in enumerate(docs, 1):
        via = f"  (via {d['via']} --{d['predicate']}--)" if d.get("via") else ""
        print(f"\n[{i}] {d['title']} ({d['qid']})  "
              f"score={d['score']:.3f}  mention={d.get('mention')}{via}")
        preview = d["text"][:200].replace("\n", " | ")
        print(f"    {preview} ...")


# =============================================================================
# 混合检索 demo：本地 Wiki + Web + KG + Rerank
# =============================================================================

def fake_web_search(query: str, k: int):
    """占位：换成真实 web 搜索 API（Bing / SerpAPI / Tavily 等）。"""
    return []


def demo_hybrid(query: str,
                config_path: str,
                *,
                top_k: int = 5,
                use_rerank: bool = True,
                multi_hop: bool = False,
                max_hops: int = 2,
                mention_method: str = "hybrid"):
    """完整的混合检索：KG 事实 + 本地稠密 + Web + 重排。

    这就是接入生产 Agent RAG 的推荐姿势：所有源都通过 hybrid_retrieve 一次融合。
    """
    print(f"\n=== hybrid_retrieve (query={query!r}, multi_hop={multi_hop}) ===")

    # 三个组件都是"只需初始化一次"的重型对象，生产中做成全局单例
    wiki_retr = WikiRetriever(config_path)
    kg_retr = KGRetriever(config_path=config_path,
                          mention_method=mention_method)
    rerank_fn = build_default_reranker() if use_rerank else None

    t0 = time.perf_counter()
    results = hybrid_retrieve(
        query,
        wiki_retriever=wiki_retr,
        kg_retriever=kg_retr,           # ⭐ 新接口：一路直通 KG
        web_search_fn=fake_web_search,
        rerank_fn=rerank_fn,
        multi_hop=multi_hop,             # ⭐ 多跳开关，透传到 KGRetriever
        max_hops=max_hops,
        top_k_wiki=top_k,
        top_k_web=0,
        top_k_kg=3,
        final_k=top_k,
    )
    dt = time.perf_counter() - t0
    print(f"  total latency: {dt*1000:.1f} ms   hits: {len(results)}\n")

    for i, r in enumerate(results, 1):
        score = r.get("rerank_score", r.get("score", 0.0))
        print(f"[{i}] ({r['source']}) {r.get('title', '')}  score={score:.3f}")
        preview = r.get("text", "")[:180].replace("\n", " ")
        print(f"    {preview} ...\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")

    # 老参数
    ap.add_argument("--mention", type=str, default=None,
                    help="只测试 link 层（老 demo）")
    ap.add_argument("--qid", type=str, default=None,
                    help="直接展示某个 qid 的三元组（老 demo）")

    # 新参数
    ap.add_argument("--query", type=str, default=None,
                    help="端到端 query：自动抽 mention → link → context")
    ap.add_argument("--method", type=str, default="hybrid",
                    choices=["ngram", "jieba", "hybrid"],
                    help="mention 抽取方案：ngram / jieba / hybrid(默认)")

    # 多跳
    ap.add_argument("--multi-hop", action="store_true",
                    help="打开多跳图遍历（默认关，避免额外延迟）")
    ap.add_argument("--max-hops", type=int, default=2)

    # 混合检索
    ap.add_argument("--hybrid", action="store_true",
                    help="演示混合检索：KG + Wiki 向量 + Web + 重排")
    ap.add_argument("--no-rerank", action="store_true",
                    help="混合检索时跳过重排")
    ap.add_argument("--top-k", type=int, default=5)

    args = ap.parse_args()

    load_config(args.config)   # 触发 path 检查，报错提示清晰
    kg = KGStore(args.config)

    # ---- 老用法：--qid ----
    if args.qid:
        demo_qid(kg, args.qid)

    # ---- 老用法：--mention ----
    if args.mention:
        linker = Linker(kg_store=kg, config_path=args.config)
        demo_link(linker, args.mention, args.query)

    # ---- 新用法：--query ----
    if args.query and not args.mention:
        if args.hybrid:
            # 混合检索：KG + Wiki + Web + Rerank
            demo_hybrid(
                args.query,
                config_path=args.config,
                top_k=args.top_k,
                use_rerank=not args.no_rerank,
                multi_hop=args.multi_hop,
                max_hops=args.max_hops,
                mention_method=args.method,
            )
        else:
            # 纯 KG 端到端
            kg_retr = KGRetriever(config_path=args.config, kg=kg,
                                  mention_method=args.method)
            demo_end_to_end(kg_retr, args.query,
                            multi_hop=args.multi_hop,
                            max_hops=args.max_hops,
                            top_k=args.top_k)

    # ---- 什么都不传：跑默认样例 ----
    if not args.qid and not args.mention and not args.query:
        linker = Linker(kg_store=kg, config_path=args.config)
        for m, q in [
            ("苹果", "苹果公司的 CEO 是谁"),
            ("苹果", "苹果这种水果的产地"),
            ("屈原", None),
            ("量子纠缠", None),
        ]:
            demo_link(linker, m, q)

        # 端到端 demo
        print("\n" + "=" * 60)
        print("  端到端 KGRetriever demo")
        print("=" * 60)
        kg_retr = KGRetriever(config_path=args.config, kg=kg, linker=linker)
        for q, mh in [
            ("苹果公司的 CEO 是谁", False),
            ("量子纠缠是什么？", False),
            ("库克领导的公司总部在哪个城市", True),
        ]:
            demo_end_to_end(kg_retr, q, multi_hop=mh, max_hops=2)


if __name__ == "__main__":
    main()
