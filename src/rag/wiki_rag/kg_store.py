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

import math
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional

from .config import load_config

# ══════════════════════════════════════════════════════════════════════════
# 繁简归一
# ══════════════════════════════════════════════════════════════════════════
# Wikidata 的 zh label **简繁混存**，同一实体到底登记成哪种没有规律：
#     Q148  → 中華人民共和國（繁）      Q956  → 北京市（简繁同形）
#     Q312  → 蘋果公司（繁）            Q22686 → 唐納·川普（繁）
# 用户输入几乎都是简体，直接精确查表会**整条查不到**：
#     SELECT ... WHERE mention='苹果公司'  → 只能靠 alias 侥幸命中
# 所以查表前把 mention 归一到简体再补一次查询。
#
# 【为什么不在建库时把 label 统一转简】
#   ① 会**破坏原始数据**：繁体是 Wikidata 的真实内容，转换不可逆，
#      港台用户搜繁体反而查不到；
#   ② t2s 不是双射（「著/着」「乾/干」等多对一），全库转换必然引入
#      新的错误合并。
# 只在**查询侧**做归一、把两种写法都查一遍，是无损的做法。
try:
    import opencc as _opencc
    _T2S = _opencc.OpenCC("t2s")
except Exception:                                    # pragma: no cover
    # opencc 缺失时降级为恒等变换：功能退化回"只查原串"，不影响正确性
    _T2S = None


def _to_simplified(s: str) -> str:
    """繁体 → 简体。opencc 不可用时原样返回。"""
    if not s or _T2S is None:
        return s
    try:
        return _T2S.convert(s)
    except Exception:
        return s


# 维基媒体内部页面的 description 特征词（简繁都要覆盖）。
# 这些"实体"的 label 与真实体同名，但本身不承载任何事实。
_WIKI_META_MARKERS = (
    "消歧义", "消歧義", "维基媒体", "維基媒體", "维基百科", "維基百科",
    "模板", "Wikimedia", "Wikipedia", "Category:", "Template:",
)


# "候选质量过低"的入度阈值。低于它就认为精确匹配召回的全是冷门实体，
# 需要走 FTS 前缀兜底把主实体捞回来。
# 取 1000：实测省级行政区/知名大学/主要国家的入度都在千级以上
# （浙江省 1162、清华大学 1740、北京市 5314、中国 106 万），
# 而同名的虚构地名/小镇/消歧义页普遍在 100 以下（美国小镇北京 11）。
# 这个档位能把两类清楚分开，又不至于让绝大多数正常查询都触发兜底。
_WEAK_POP_THRESHOLD = 1000


def _safe_log(x) -> float:
    """自然对数，供 SQL 侧算消歧组合分用。

    入参可能是 None（LEFT JOIN 未命中）或 0，都必须返回有限值而不是
    抛异常/返回 -inf —— 排序里出现 NaN/None 会让顺序变得不可预测。
    """
    try:
        v = float(x or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return math.log(v) if v > 0 else 0.0


def _is_wiki_meta(desc: Optional[str]) -> bool:
    """判断是否是维基媒体内部页面（消歧义/分类/模板）。

    返回 bool 而不是分数：它只用来做**稳定排序的分组键**（False 组在前），
    组内原有的 popularity 顺序完全保留。
    """
    if not desc:
        return False
    return any(k in desc for k in _WIKI_META_MARKERS)


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

        self._db_path = db_path
        # ══════════════════════════════════════════════════════════════════
        # 每线程一条连接（thread-local），而不是全局共享一条
        # ══════════════════════════════════════════════════════════════════
        # 【实测故障】跑 BrowseComp-ZH 评测时每一题都刷这行日志：
        #     [retriever] L5_kg search 异常: bad parameter or other API misuse
        # 即 sqlite3.InterfaceError（SQLITE_MISUSE）。后果是 **L5 整层
        # 静默返回空列表** —— 异常被 `_safe_search` 吞掉，检索照常继续，
        # 所以功能上看不出坏，只是知识图谱这一路的召回一直是 0。
        # 这类"降级成功但收益归零"的故障最难发现。
        #
        # 【根因】原实现是**单连接 + check_same_thread=False**。
        # ⚠️ `check_same_thread=False` 只是**关掉 Python 层的线程检查**，
        # 它并不让连接变成并发安全 —— 底层 sqlite3_stmt 是有状态的，
        # 两个线程同时在同一连接上 execute，游标状态会互相踩踏。
        # 而 `retriever._parallel_search` 恰恰用线程池**并行**调各层，
        # 建议1 的并发子检索更是让同一层被多路同时调用，必然踩中。
        # 已用 8 线程 × 单连接稳定复现出同一条 InterfaceError。
        #
        # 【为什么用 thread-local 而不是加锁】
        # 加锁能修正确性，但会把并行检索**串行化** —— L5 实测 434ms，
        # 是最慢的离线层，串行化直接吃掉并发收益。SQLite 在只读场景下
        # 多连接读同一文件是完全安全的（且各连接共享 OS page cache 与
        # mmap，内存开销很小），所以每线程一条连接既正确又不牺牲并发。
        #
        # 保留 `check_same_thread=False`：线程池的 worker 会被复用，
        # 连接的创建线程与使用线程虽然一致，但不加这个参数在某些
        # 回收/复用场景下仍会误报。
        self._local = threading.local()
        # 预取一次基本信息，同时验证连接可用（也顺便建好主线程那条连接）
        n = self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        print(f"[kg] connected: {db_path}  entities={n:,}")

    @property
    def conn(self) -> sqlite3.Connection:
        """当前线程专属的只读连接（首次访问时惰性创建）。

        做成 property 而不是改所有调用点：`kg_retriever.py` 里有多处
        直接用 `kg.conn.execute(...)`，保持这个属性名可以让全部调用点
        零改动地获得线程安全，也不破坏任何外部约定。
        """
        c = getattr(self._local, "conn", None)
        if c is None:
            # 加 URI mode=ro 打开可以强制只读，但对某些优化 PRAGMA 会失效；
            # 这里退而求其次：连接后立刻 PRAGMA query_only=ON
            c = sqlite3.connect(str(self._db_path), check_same_thread=False)
            # ⚠️ 自己注册 LOG：SQLite 的数学函数（LOG/POW/...）是**编译期可选**
            # 的（-DSQLITE_ENABLE_MATH_FUNCTIONS），3.35 以前根本没有，且不同
            # 平台预编译的 sqlite3 是否带它并不一致。消歧排序依赖 LOG 算组合分，
            # 若运行环境的 sqlite 恰好没编进去，link() 会直接抛
            # OperationalError 让 L5 整层挂掉。用 create_function 注册一个
            # 同名实现，可以覆盖/补齐，行为在所有环境下都一致。
            # deterministic=True 允许 SQLite 在索引/查询计划里做常量折叠。
            try:
                c.create_function("LOG", 1, _safe_log, deterministic=True)
            except TypeError:                        # pragma: no cover
                # 老版本 Python 不支持 deterministic 参数
                c.create_function("LOG", 1, _safe_log)
            c.execute("PRAGMA query_only=ON")
            # 查询时打开 mmap 可以显著减少 I/O（多连接共享同一份映射）
            c.execute("PRAGMA mmap_size=30000000000")
            # sqlite3 默认返回 tuple；用 Row 让下游可以按列名访问
            c.row_factory = sqlite3.Row
            self._local.conn = c
        return c

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
        # ══════════════════════════════════════════════════════════════════
        # 排序用 weight 与 popularity 的**组合分**，不是字典序
        # ══════════════════════════════════════════════════════════════════
        # 【实测故障】原排序是 `weight DESC, article_rank IS NULL, article_rank`。
        # 但 article_rank 只有 2,869/10,385,628 = **0.03%** 的实体有值，
        # popularity 当时又恒为 0 —— 于是同名候选（全是 weight=1.0、
        # rank=NULL）之间**完全没有排序依据**，谁在前只取决于物理存储顺序：
        #     "北京" → 第一候选 Q578328「美國伊利諾伊州塔茲韋爾縣的縣城」
        #     "中国" → 第一候选 Q2736887「1972年安東尼奧尼的電影」
        # 主实体 Q956(北京市)、Q148(中国) 连前 5 都进不去。
        # 这比"召回为空"更危险：它会把**张冠李戴的事实**喂给 LLM，
        # 而且看起来一切正常。
        #
        # popularity 由 `10/14` 脚本回填为实体入度，区分度达 5 个数量级
        # （中国 1,069,530 vs 美国小镇 11），是这里唯一可靠的重要度信号。
        #
        # ⚠️ 但**只把 popularity 当 tie-breaker 是不够的**：
        # 若排序写成 `ORDER BY weight DESC, popularity DESC`，weight 就成了
        # 第一优先级，于是「冷门实体的 label(w=1.0)」永远压过
        # 「主实体的 alias(w=0.6)」——
        #     长江 → Q137163822「中国歌手与男演员」(w=1.0,pop=0)
        #            压过 Q5413「世界第三长河流」(w=0.6,pop=28)
        # 而 Wikidata 里主实体的 label 常是繁体，简体串只能作为 alias 命中，
        # 恰恰落在 w=0.6 这一档。所以必须做成**组合分**让两个信号可换算：
        #
        #     score = weight × log1p(popularity)
        #
        # 用 log 而不是线性：入度分布跨 6 个数量级（0 ~ 106 万），线性会让
        # 超高频实体碾压一切，log 压缩后 weight 的 1.0/0.6 之差仍有话语权。
        # 在 SQL 里算完直接 ORDER BY，避免把全部候选拉回 Python 再排。
        exact_sql = """
            SELECT m.qid, m.weight, m.source, e.label_zh, e.description,
                   e.popularity,
                   m.weight * LOG(1 + e.popularity) AS score
            FROM mentions m
            JOIN entities e ON m.qid = e.qid
            WHERE m.mention = ?
            ORDER BY score DESC,
                     m.weight DESC,
                     e.article_rank IS NULL, e.article_rank ASC
            LIMIT ?
        """
        rows = list(self.conn.execute(exact_sql, (mention, top_k)))
        cands = [dict(r) for r in rows]

        # ---- 1b) 繁简归一后再查一次 ----
        # Wikidata 的 zh label 简繁混存（Q312 存的是「蘋果公司」），
        # 用户输入的简体串精确查表会漏掉主实体。把 mention 转简体后
        # 补查一遍，两边结果按 qid 去重合并。
        # 只在**转换后确实变了**时才多查一次，避免纯简体输入白跑一次 SQL。
        simplified = _to_simplified(mention)
        if simplified != mention:
            seen = {c["qid"] for c in cands}
            for r in self.conn.execute(exact_sql, (simplified, top_k)):
                if r["qid"] not in seen:
                    d = dict(r)
                    # 归一命中略降权：原串精确匹配的证据强度更高
                    d["weight"] = float(d["weight"]) * 0.9
                    d["score"] = float(d["score"] or 0.0) * 0.9
                    d["source"] = f"t2s:{d['source']}"
                    cands.append(d)
                    seen.add(r["qid"])
            cands.sort(key=lambda c: -float(c.get("score") or 0.0))

        # ---- 2) FTS5 兜底（前缀 or 分词匹配）----
        # ══════════════════════════════════════════════════════════════════
        # 触发条件不能只看"候选够不够"，还要看"候选好不好"
        # ══════════════════════════════════════════════════════════════════
        # 原条件是 `len(cands) < top_k`，即**凑够 5 条就不再兜底**。
        # 但 Wikidata 里冷门同名实体极多，凑够 5 条太容易了，而这 5 条
        # 可能全是垃圾：
        #     "北京" 精确命中 6 条 —— 全是水滸虛構地名/美國小鎮/消歧義頁，
        #            最高 pop 只有 15，主实体 Q956(北京市, pop=5314) 因为
        #            没登记"北京"这个 alias 而根本不在里面；
        #     "中国" 同理，Q148(pop=106万) 的 mention 里没有"中国"二字。
        # 这是 Wikidata 的 **alias 覆盖缺失**，不是排序能解决的问题 ——
        # 排序只能在候选集内部调整顺序，救不回压根没被召回的实体。
        #
        # 所以增加一条**质量闸门**：即使候选数够，只要最高分低于阈值，
        # 说明召回的全是冷门实体，仍然走 FTS 前缀把高入度实体捞进来。
        # 阈值取 log(1000)≈6.9（对应入度千级的"知名实体"档），
        # 用 popularity 而不是 score 判断，避免受 weight 折扣干扰。
        best_pop = max((int(c.get("popularity") or 0) for c in cands),
                       default=0)
        need_fuzzy = len(cands) < top_k or best_pop < _WEAK_POP_THRESHOLD
        if need_fuzzy:
            # FTS5 MATCH 语法：'苹果*' 是前缀，'苹果 OR 蘋果' 是分词
            # 转义单引号 + 附加 '*' 做前缀匹配
            safe = mention.replace('"', ' ').replace("'", " ").strip()
            if safe:
                fuzzy_k = self.kg_cfg["link_top_k_fuzzy"]
                # ⚠️ 两个坑必须同时处理：
                #
                # ① FTS5 的 MATCH **本身不排序**。原实现是裸 `LIMIT ?`，
                #    取到的是索引里最先扫到的若干条，与相关性无关 ——
                #    实测 '"北京"*' 取 300 条里既没有 Q956 也没有 Q148，
                #    全是冷门同前缀实体。必须按 popularity 排。
                #
                # ② 但直接 `ORDER BY popularity LIMIT 20` 会**慢到不可用**：
                #    '"中国"*' 前缀命中 156,306 行，SQLite 得把这 15 万行
                #    全部 JOIN + 建临时 B-Tree 排序，实测 **9.1 秒**
                #    （北京 2.3s）。L5 整层预算才几百毫秒，这是灾难性的。
                #
                # 解法：先用子查询把 FTS 命中**截断**成一个小候选池，
                # 只在池内做 JOIN 和排序。前缀匹配的语义下，取多少条都是
                # 任意子集（FTS 无序），所以截断不损失"排序质量" ——
                # 真正的排序发生在池内，而池已经足够大（fuzzy_k 的 50 倍）
                # 能覆盖到高入度实体。实测 9.1s → 数十毫秒。
                pool = max(fuzzy_k * 50, 1000)
                fuzzy_sql = """
                    SELECT f.qid,
                           m.weight,
                           m.source,
                           e.label_zh,
                           e.description,
                           e.popularity,
                           m.weight * LOG(1 + e.popularity) AS score
                    FROM (SELECT mention, qid FROM mentions_fts
                          WHERE mentions_fts MATCH ? LIMIT ?) f
                    JOIN mentions m ON m.mention = f.mention AND m.qid = f.qid
                    JOIN entities e ON e.qid = f.qid
                    ORDER BY score DESC
                    LIMIT ?
                """
                # 加通配符做前缀匹配
                pattern = f'"{safe}"*'
                try:
                    seen_qids = {c["qid"] for c in cands}
                    for r in self.conn.execute(
                            fuzzy_sql, (pattern, pool, fuzzy_k)):
                        if r["qid"] in seen_qids:
                            continue
                        d = dict(r)
                        # FTS 命中的分数打折，避免碾压精确命中
                        d["weight"] = float(d["weight"]) * 0.5
                        d["score"] = float(d["score"] or 0.0) * 0.5
                        d["source"] = f"fts:{d['source']}"
                        cands.append(d)
                        seen_qids.add(r["qid"])
                except sqlite3.OperationalError as e:
                    # FTS 语法错误（比如 mention 里有特殊字符）时静默忽略
                    print(f"[kg] fts5 skipped: {e}")

        # ---- 3) 统一排序：噪声沉底 + 组合分降序 ----
        # ⚠️ 这一步必须做**全局重排**，不能只做稳定排序。
        # FTS 兜底的候选是 append 到列表尾部的，若只按 is_wiki_meta 做稳定
        # 排序，它们会永远排在精确候选之后 —— 那么"质量闸门捞回主实体"
        # 这件事就白做了（实测 北京 触发了兜底、Q956 也进了候选，但仍排
        # 在 pop=11 的美国小镇后面）。必须把两路候选放在同一把尺子下比。
        #
        # 排序键含义：
        #   ① _is_wiki_meta：Wikidata 里有大量**非实体**的内部页面
        #      （消歧义页 69,377 / 项目分类 361,223 / 模板 205,640），
        #      它们的 label 和真实体完全一样，但 to_context() 只能拼出
        #      「· 是一个 维基媒体消歧义页」这种纯噪声。False 组排前面。
        #      用分组键而不是直接删除：万一某 mention 只能匹配到这类页面，
        #      至少保留候选不至于全空。
        #   ② -score：组内按 weight × log1p(popularity) 降序。
        cands.sort(key=lambda c: (_is_wiki_meta(c.get("description")),
                                  -float(c.get("score") or 0.0)))

        return cands[:top_k]

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
        """关闭**当前线程**的连接。

        只能关自己这条：其它线程的连接对象存在各自的 thread-local 里，
        跨线程 close 本身就是一种 SQLITE_MISUSE。进程退出时未显式关闭的
        连接由 GC / OS 回收，对只读连接没有数据丢失风险。
        """
        c = getattr(self._local, "conn", None)
        if c is not None:
            try:
                c.close()
            except Exception:
                pass
            self._local.conn = None
