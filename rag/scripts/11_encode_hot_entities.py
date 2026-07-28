"""
11_encode_hot_entities.py
=========================

**方案 C**：只对"热门实体"（能 join 上 L2 pageviews rank 的实体）算 BGE-M3
description 向量，用于在线阶段 mention → qid 消歧的重排辅助。

为什么只编码热门实体？
    - 全部 zh 实体约 250 万，编码一遍要 6~10 小时（单卡 4090），并额外占 ~10GB 磁盘；
    - 但 RAG 场景 query 中出现的 mention 95%+ 都指向"热门实体"（长尾极少被问到）；
    - 只对 article_rank 非空（即能对齐维基百科热门条目的）那批做编码：
      默认 30 万条 → 编码 ~30 分钟，磁盘 ~1.2 GB，性价比最高。

编码内容：
    "{label_zh} —— {description}"
    比如："中国 —— 位于东亚的主权国家"
    这样单条向量同时代表了"名字"和"描述"，query 侧对整个短语做相似度即可。

产物三件套：
    - kg_hot_qids_file: "data/wiki_zh_kg_hot_qids.txt"          # 每行一个 qid（顺序 = article_rank 升序）
    - kg_hot_emb_file: "data/wikidata_zh_kg_hot_emb.npy"        # (N, dim) 的 float32 numpy，L2 归一化
    - kg_hot_index_file: "data/wiki_zh_kg_hot.faiss"            # FAISS 索引（默认 Flat 精确检索）
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import faiss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wiki_rag.config import load_config
from wiki_rag.embedder import encode


def _select_hot_entities(conn: sqlite3.Connection, hot_top_n: int | None):
    """选出"热门实体"：article_rank IS NOT NULL 且 label_zh IS NOT NULL。

    按 article_rank 升序（越小越热）；截断到 hot_top_n。
    每行返回 (qid, label_zh, description)。
    """
    limit_sql = f"LIMIT {int(hot_top_n)}" if hot_top_n else ""
    sql = f"""
        SELECT qid, label_zh, COALESCE(description, '')
        FROM entities
        WHERE article_rank IS NOT NULL
          AND label_zh IS NOT NULL
        ORDER BY article_rank ASC
        {limit_sql}
    """
    return list(conn.execute(sql))


def _compose_text(label: str, desc: str) -> str:
    """把实体拼成一句短语。description 为空则只用 label。"""
    if desc:
        return f"{label} —— {desc}"
    return label


def build(cfg: dict) -> None:
    paths = cfg["paths"]
    kg_cfg = cfg["kg"]
    emb_cfg = cfg["embedder"]

    db_path: Path = paths["kg_db_file"]             # "data/wikidata_zh_kg.db"
    qids_out: Path = paths["kg_hot_qids_file"]      # "data/wiki_zh_kg_hot_qids.txt"
    emb_out: Path = paths["kg_hot_emb_file"]        # "data/wikidata_zh_kg_hot_emb.npy"
    idx_out: Path = paths["kg_hot_index_file"]      # "data/wiki_zh_kg_hot.faiss"

    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} not found; run 10_build_kg_sqlite.py first."
        )

    print(f"[11] opening {db_path}")
    conn = sqlite3.connect(str(db_path))

    # rows: qid, label_zh, COALESCE(description, '')
    rows = _select_hot_entities(conn, kg_cfg.get("hot_top_n"))  # 300000
    conn.close()
    if not rows:
        print("[11] no hot entities (article_rank all NULL). "
              "make sure top_titles.txt was available at step 10.")
        return
    print(f"[11] hot entities to encode: {len(rows):,}")

    qids = [r[0] for r in rows]
    texts = [_compose_text(r[1], r[2]) for r in rows]

    # 落盘 qids（先落，编码断了也知道对齐关系）
    qids_out.parent.mkdir(parents=True, exist_ok=True)
    with open(qids_out, "w", encoding="utf-8") as f:
        for q in qids:
            f.write(q + "\n")
    print(f"[11] saved qids -> {qids_out}")

    # 编码。BGE-M3 的 encode() 会自动 batch + 归一化
    print(f"[11] encoding with BGE-M3 (batch={kg_cfg['hot_batch_size']}) ...")  # 编码时的批大小
    embs = encode(
        texts,
        model_name=emb_cfg["model_name"],
        batch_size=kg_cfg["hot_batch_size"],
        max_length=emb_cfg["max_length"],
        device=emb_cfg["device"],
        normalize=True,
    )
    # 确保 float32；下游 FAISS 只接受 float32
    embs = embs.astype("float32", copy=False)
    np.save(emb_out, embs)      # "data/wikidata_zh_kg_hot_emb.npy"
    print(f"[11] saved emb -> {emb_out}  shape={embs.shape}")

    # 建 FAISS 索引
    dim = embs.shape[1]
    index_type = kg_cfg.get("hot_index_type", "flat").lower()
    print(f"[11] building FAISS index (type={index_type}, dim={dim}) ...")
    if index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, cfg["faiss"]["hnsw_m"],
                                    faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = cfg["faiss"]["hnsw_ef_construction"]
        index.hnsw.efSearch = cfg["faiss"]["hnsw_ef_search"]
    else:
        # 精确检索：小规模够快，且没有召回损失
        index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, str(idx_out))      # "data/wiki_zh_kg_hot.faiss"
    print(f"[11] saved index -> {idx_out}  ntotal={index.ntotal}")
    print("[11] done. now KGStore + Linker can use embedding-assisted disambiguation.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    build(cfg)


if __name__ == "__main__":
    main()
