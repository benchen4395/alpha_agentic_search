"""KGStore：对 L5 SQLite 知识图谱库的只读封装。

设计目标
--------
- 单例 / 线程安全（SQLite check_same_thread=False + query_only 模式）
- 一次打开，反复查询
- 只暴露 RAG 需要的三类操作：

  1) ``link(mention)``     mention 字面串 → [(qid, label, weight, source)]
                            精确 label > 别名 > FTS5 模糊查询
  2) ``triples_of(qid)``   拉一个实体的 1 跳三元组，object 若是实体自动 JOIN 出 label
  3) ``to_context(qid)``   把 triples 拼成一段人类可读的中文事实文本，可直接注入 LLM

FTS5 用来兜底那些"label 表里没有但拼写接近"的 mention（比如 query 里带了错字/繁体/口语化后缀）。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from .config import load_config


class KGStore:
    """SQLite KG 只读客户端。"""

    def __init__(self, config_path: str | Path | None = None):
        cfg = load_config(config_path)
        self.cfg = cfg
        self.kg_cfg = cfg["kg"]
        db_path: Path = cfg["paths"]["kg_db_file"]
        if not db_path.exists():
            raise FileNotFoundError(
                f"KG db not found: {db_path}. run scripts/10_build_kg_sqlite.py first.")

        # check_same_thread=False → 允许 web server / 多线程复用同一连接
        # 加 URI mode=ro 打开可以强制只读，但对某些优化 PRAGMA 会失效；
        # 这里退而求其次：连接后立刻 PRAGMA query_only=ON
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA query_only=ON")
        # 查询时打开 mmap 可以显著减少 I/O
        self.conn.execute("PRAGMA mmap_size=30000000000")
        # sqlite3 默认返回 tuple；用 Row 让下游可以按列名访问
        self.conn.row_factory = sqlite3.Row
        # 预取一次基本信息，同时验证连接可用
        n = self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        print(f"[kg] connected: {db_path}  entities={n:,}")

    # -------------------------------------------------- link
    def link(self, mention: str, top_k: Optional[int] = None) -> List[Dict]:
        """``mention → 候选实体列表``（未做 embedding 重排的原始候选）。

        三级 fallback：
          1. 精确匹配 ``mentions.mention == mention``（label 权重 > alias）
          2. 若不足 ``top_k_exact``，再走 FTS5 前缀 / 模糊查询
          3. 全部失败则返回空列表

        Returns:
            [{qid, label_zh, description, weight, source}, ...]
            按 weight 降序；具体最终 topk 由调用方（Linker）截断。
        """
        top_k = top_k or self.kg_cfg["link_top_k_exact"]
        mention = mention.strip()
        if not mention:
            return []

        # ---- 1) 精确 ----
        exact_sql = """
            SELECT m.qid, m.weight, m.source, e.label_zh, e.description
            FROM mentions m
            JOIN entities e ON m.qid = e.qid
            WHERE m.mention = ?
            ORDER BY m.weight DESC, e.article_rank IS NULL, e.article_rank ASC
            LIMIT ?
        """
        rows = list(self.conn.execute(exact_sql, (mention, top_k)))
        cands = [dict(r) for r in rows]

        # ---- 2) FTS5 兜底（前缀 or 分词匹配）----
        if len(cands) < top_k:
            # FTS5 MATCH 语法：'苹果*' 是前缀，'苹果 OR 蘋果' 是分词
            # 转义单引号 + 附加 '*' 做前缀匹配
            safe = mention.replace('"', ' ').replace("'", " ").strip()
            if safe:
                fuzzy_k = self.kg_cfg["link_top_k_fuzzy"]
                fuzzy_sql = f"""
                    SELECT f.qid,
                           m.weight,
                           m.source,
                           e.label_zh,
                           e.description
                    FROM mentions_fts f
                    JOIN mentions m ON m.mention = f.mention AND m.qid = f.qid
                    JOIN entities e ON e.qid = f.qid
                    WHERE mentions_fts MATCH ?
                    LIMIT ?
                """
                # 加通配符做前缀匹配（BM25 排序默认按相关性）
                pattern = f'"{safe}"*'
                try:
                    seen_qids = {c["qid"] for c in cands}
                    for r in self.conn.execute(fuzzy_sql, (pattern, fuzzy_k)):
                        if r["qid"] in seen_qids:
                            continue
                        d = dict(r)
                        # FTS 命中的分数打折，避免碾压精确命中
                        d["weight"] = float(d["weight"]) * 0.5
                        d["source"] = f"fts:{d['source']}"
                        cands.append(d)
                        seen_qids.add(r["qid"])
                except sqlite3.OperationalError as e:
                    # FTS 语法错误（比如 mention 里有特殊字符）时静默忽略
                    print(f"[kg] fts5 skipped: {e}")

        return cands

    # -------------------------------------------------- triples
    def triples_of(self, qid: str, limit: Optional[int] = None) -> List[Dict]:
        """拉一个实体的 1 跳三元组，object 若是实体则 JOIN 出 label。

        Returns:
            [{predicate_pid, predicate_label, object_qid, object_label, object_type}, ...]
        """
        limit = limit or self.kg_cfg["triples_per_entity"]
        sql = """
            SELECT t.predicate_pid,
                   p.label_zh                       AS predicate_label,
                   t.object_qid,
                   COALESCE(e2.label_zh, t.object_value) AS object_label,
                   t.object_type
            FROM triples t
            LEFT JOIN properties p ON p.pid = t.predicate_pid
            LEFT JOIN entities e2 ON e2.qid = t.object_qid
            WHERE t.subject_qid = ?
            LIMIT ?
        """
        return [dict(r) for r in self.conn.execute(sql, (qid, limit))]

    # -------------------------------------------------- context
    def to_context(self, qid: str, max_chars: int = 1000) -> str:
        """把三元组拼成一段"事实条列"文本，可直接注入 LLM 的 system prompt 或 context。

        输出示例：
            [中国] 位于东亚的主权国家
            · 是一个 国家
            · 首都 北京
            · 官方语言 汉语
            · ...
        """
        row = self.conn.execute(
            "SELECT label_zh, description FROM entities WHERE qid=?", (qid,)
        ).fetchone()
        if row is None:
            return ""
        label, desc = row["label_zh"] or qid, row["description"] or ""
        parts = [f"[{label}]" + (f" {desc}" if desc else "")]
        for t in self.triples_of(qid):
            pred = t["predicate_label"] or t["predicate_pid"]
            obj = t["object_label"] or ""
            if not obj:
                continue
            parts.append(f"· {pred} {obj}")
        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars].rsplit("\n", 1)[0] + "\n..."
        return text

    # -------------------------------------------------- misc
    def qid_to_label(self, qid: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT label_zh FROM entities WHERE qid=?", (qid,)
        ).fetchone()
        return row["label_zh"] if row else None

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
