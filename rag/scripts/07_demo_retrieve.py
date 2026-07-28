"""
07_demo_retrieve.py
===================

混合检索的命令行最小示例：**本地 Wiki 向量库 + Web 搜索 + KG 三元组 + 重排**。

四路召回全部在 :func:`wiki_rag.hybrid.hybrid_retrieve` 内部融合。想接自己的
Web 搜索后端，把 :func:`fake_web_search` 替换成真实实现即可。

使用文件总结:
1. 本地Wiki向量库:
-- "data/wiki_zh_emb_hnsw.faiss": chunk_emb -- 596390条
-- "data/wiki_zh_chunks.jsonl": (chunk_id, doc_id, title, text)  -- 596390条

2. 知识图谱Wikidata
2.1 link阶段: 
-- "data/wikidata_zh_kg_hot_emb.npy": 与wiki_zh_top_titles.txt的标题命中的top30w实体向量， (N, dim) 归一化，方便模糊查询
-- "data/wiki_zh_kg_hot_qids.txt": 与 hot_emb 逐行对齐的 QID 列表，一行一个


用法::

    # 最简：本地 Wiki + KG + Rerank
    python scripts/07_demo_retrieve.py --query "量子纠缠是什么？"

    # 打开多跳 KG（组合关系问题必需）
    python scripts/07_demo_retrieve.py --query "库克领导的公司总部在哪" --multi-hop

    # 关闭 KG 通路（只保留 wiki+web+rerank，与旧版行为一致）
    python scripts/07_demo_retrieve.py --query "..." --no-kg

    # 关闭重排（快速调试）
    python scripts/07_demo_retrieve.py --query "..." --rerank none
    python scripts/07_demo_retrieve.py --query "..." --rerank rrf         # 零延迟融合
    python scripts/07_demo_retrieve.py --query "..." --rerank cascade     # RRF 粗 → BGE 精
    python scripts/07_demo_retrieve.py --query "..." --rerank bge         # cross-encoder（默认）
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wiki_rag.retriever import WikiRetriever
from wiki_rag.kg_retriever import KGRetriever
from wiki_rag.hybrid import hybrid_retrieve, build_reranker
from wiki_rag.warmup import warmup_all


def fake_web_search(query: str, k: int):
    """占位：换成你项目里真正的 web 搜索（Bing / Google / Tavily / SerpAPI 等）。

    真实实现返回列表元素结构建议为::

        {"source": "web", "title": "...", "text": "...", "url": "...", "score": 0.x}
    """
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=5)

    # KG 通路开关（默认开）
    ap.add_argument("--no-kg", action="store_true", help="关闭 KG 通路（等价于旧版 07_demo_retrieve 行为）")
    ap.add_argument("--multi-hop", action="store_true", help="打开多跳 KG 图遍历（默认关，会增加 10-数百 ms 延迟）")
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--mention-method", type=str, default="hybrid", choices=["ngram", "jieba", "hybrid"])
    ap.add_argument("--top-k-kg", type=int, default=3)

    # 重排开关（默认开）
    ap.add_argument("--rerank", type=str, default="bge", choices=["bge", "rrf", "cascade", "none"], help="重排策略：bge(默认) / rrf / cascade(RRF→BGE) / none")
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF 平滑常数（rrf / cascade 时生效）")
    ap.add_argument("--cascade-coarse-k", type=int, default=20, help="cascade 策略：RRF 粗排后送入 BGE 的候选数上限")
    ap.add_argument("--no-rerank", action="store_true", help="兼容旧开关，等价于 --rerank none")
    ap.add_argument("--repeat", type=int, default=3, help="重复跑几次同一 query 观察稳态延迟（默认 3）")

    # Web 搜索
    ap.add_argument("--top-k-web", type=int, default=0, help="从 web_search_fn 取几条；默认 0 表示禁用")

    args = ap.parse_args()

    # ---- 初始化各路组件（生产中做成全局单例）----
    print(f"[07] loading components ...")
    t0 = time.perf_counter()
    wiki_retr = WikiRetriever(args.config) 
    kg_retr = None if args.no_kg else KGRetriever(
        config_path=args.config,
        mention_method=args.mention_method,     # default: hybrid
    )
    # 处理新旧开关：--no-rerank 优先级高于 --rerank
    rerank_strategy = "none" if args.no_rerank else args.rerank
    if rerank_strategy == "rrf":
        rerank_fn = build_reranker("rrf", k=args.rrf_k)
    elif rerank_strategy == "cascade":
        rerank_fn = build_reranker("cascade",
                                   coarse_k=args.cascade_coarse_k,
                                   rrf_k=args.rrf_k)
    elif rerank_strategy == "bge":
        rerank_fn = build_reranker("bge")
    else:
        rerank_fn = None
    print(f"[07] rerank strategy: {rerank_strategy}")

    # ---- Warmup：一次性把所有懒加载的重资源拉起来 ----
    # 关键：不 warmup 的话，模型 / mmap 页在首查询时才被 touch，会多 3-8 s 延迟毛刺。
    # 生产服务里请把 warmup_all 放在 ready 探针之前。
    warmup_all(
        wiki_retriever=wiki_retr,
        kg_retriever=kg_retr,
        rerank_fn=rerank_fn,
        probe_query=args.query,   # 用真实 query 触发 CUDA kernel 编译走真实 shape
    )

    print(f"[07] components ready in {time.perf_counter() - t0:.1f} s")

    # ---- 一次混合检索 ----
    t0 = time.perf_counter()
    results = hybrid_retrieve(
        args.query,
        wiki_retriever=wiki_retr,
        kg_retriever=kg_retr,        # ⭐ 新增：KG 通路，None 时自动跳过
        web_search_fn=fake_web_search,
        rerank_fn=rerank_fn,
        multi_hop=args.multi_hop,     # ⭐ 新增：多跳开关
        max_hops=args.max_hops,       # default: 2
        top_k_wiki=args.top_k,        # default: 3
        top_k_web=args.top_k_web,     # default: 0
        top_k_kg=args.top_k_kg,       # default: 3
        final_k=args.top_k,           # default: 5
    )
    dt = time.perf_counter() - t0

    # ---- 打印每阶段耗时（便于排查瓶颈） ----
    t = getattr(hybrid_retrieve, "last_timing", {})
    print(f"\n[timing] kg={t.get('kg_ms', 0):.1f} ms  |  "
          f"wiki={t.get('wiki_total_ms', 0):.1f} ms "
          f"(encode={t.get('wiki_encode_ms', 0):.1f} + "
          f"faiss={t.get('wiki_faiss_ms', 0):.1f})  |  "
          f"web={t.get('web_ms', 0):.1f} ms  |  "
          f"dedup={t.get('dedup_ms', 0):.1f} ms  |  "
          f"rerank={t.get('rerank_ms', 0):.1f} ms  |  "
          f"sum={t.get('total_ms', 0):.1f} ms")

    # ---- 再跑一次同 query 观察稳态延迟（排除冷缓存影响） ----
    if args.repeat > 1:
        for i in range(args.repeat - 1):
            _ = hybrid_retrieve(
                args.query,
                wiki_retriever=wiki_retr,
                kg_retriever=kg_retr,
                web_search_fn=fake_web_search,
                rerank_fn=rerank_fn,
                multi_hop=args.multi_hop,
                max_hops=args.max_hops,
                top_k_wiki=args.top_k,
                top_k_web=args.top_k_web,
                top_k_kg=args.top_k_kg,
                final_k=args.top_k,
            )
            t2 = getattr(hybrid_retrieve, "last_timing", {})
            print(f"[timing #{i+2}] kg={t2.get('kg_ms',0):.1f}  "
                  f"wiki={t2.get('wiki_total_ms',0):.1f} "
                  f"(enc={t2.get('wiki_encode_ms',0):.1f}+faiss={t2.get('wiki_faiss_ms',0):.1f})  "
                  f"rerank={t2.get('rerank_ms',0):.1f}  "
                  f"sum={t2.get('total_ms',0):.1f} ms")

    # ---- 打印结果 ----
    print(f"\n=== Query: {args.query} ===")
    print(f"    (latency {dt*1000:.1f} ms, {len(results)} hits, "
          f"kg={'off' if args.no_kg else 'on'}, "
          f"multi_hop={args.multi_hop}, "
          f"rerank={'off' if args.no_rerank else 'on'})\n")

    for i, r in enumerate(results, 1):
        score = r.get("rerank_score", r.get("score", 0.0))
        # KG 通路会带 mention/via/predicate 溯源字段
        via = f"  (via {r['via']} --{r['predicate']}--)" if r.get("via") else ""
        mention = f"  [mention={r['mention']}]" if r.get("mention") else ""
        print(f"[{i}] ({r['source']}) {r.get('title', '')}  "
              f"score={score:.3f}{mention}{via}")
        print("    " + r["text"][:200].replace("\n", " ") + " ...\n")


if __name__ == "__main__":
    main()
