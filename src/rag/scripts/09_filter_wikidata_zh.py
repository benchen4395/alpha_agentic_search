"""
09_filter_wikidata_zh.py
========================

从 Wikidata truthy dump 中过滤出**含中文标签**的实体及其**白名单属性**上的三元组。

背景
----
`latest-truthy.nt.bz2` 是一个约 15GB 压缩、~250GB 解压的巨型 N-Triples 文件，
每行是一个三元组，形如：

    <http://www.wikidata.org/entity/Q148>       # subject（实体 URI）
    <http://www.wikidata.org/prop/direct/P36>   # predicate（属性 URI）
    <http://www.wikidata.org/entity/Q956> .     # object（可能是实体 URI 或字面量）

字面量形式：
    "中国"@zh                    ← 语言标签的字符串（label / alias / description）
    "2024-01-01T00:00:00Z"^^<...#dateTime>
    "1.5"^^<...#decimal>

# QID = Q-Identifier，即 Wikidata 中每个"实体（Entity/Item）"的唯一编号, 相当于知识图谱里的"实体主键";
# PID = P-Identifier，即 Wikidata 中每个"属性（Property）"的唯一编号, PID 描述"实体之间/实体与值之间的关系"，相当于知识图谱里的**"边（谓词/关系）"**
# QID = OID = object id, 宾语实体的 QID，其实也是QID
# 三元组 (Q148, P36, Q956) 读作："中国（Q148）的首都（P36）是北京（Q956）"

策略：两遍扫描（Two-Pass）
--------------------------
* **Pass 1**：全量扫一遍，只收集"某个 Q<id> 有 zh label（rdfs:label 或 schema:name @zh）"
  这个信息，写到 ``kg_zh_qids_file``。得到一个 ~200-300 万的 QID 集合。
* **Pass 2**：再扫一遍：
    - 若行是 zh 的 label / alias / description → 累加到 entities_jsonl（实体主表）
    - 若行是白名单里的 P<xxx> 属性，且 subject 在 zh_qids 里
        - object 是实体 URI 且也在 zh_qids 里，或者
        - object 是字面量，
      则写入 kg_triples_tsv

为什么两遍？
    Pass 1 只关心"哪些实体有 zh label"，把 QID 集合缩小到几百万；
    Pass 2 就能用 O(1) 集合查表快速判断，避免把英文/日文-only 的实体也带进来。
    如果一遍扫的话，object 可能先于 subject 的 label 出现，无法即时判断。

进程占用
--------
* Pass 1 主要瓶颈是 bzip2 解压（~150-200 MB/s，单核）；
* Pass 2 同上；两遍加起来约 2-4 小时（i9 / SSD），主要卡在 IO + 解压。
* 内存：QID 集合最多约 300 万 × ~15 字节 = ~45 MB，完全可控。

产物
----
* ``data/wikidata_zh_qids.txt``    —— 每行一个 QID，比如 "Q148"
* ``data/wikidata_zh_entities.jsonl``  —— 每行一个 JSON： {qid, label_zh, label_en, description, aliases:[...]}
* ``data/wikidata_zh_triples.tsv``     —— 每行 5 列（\\t 分隔）：
                             subject_qid \\t predicate_pid \\t object_qid \\t object_value \\t object_type
                            （object_qid 与 object_value 二选一，另一列为空串）
"""
from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # wiki_rag
from wiki_rag.config import load_config


# ---------------------------------------------------------------- 正则 & 常量
# N-Triples 每行末尾是 " ." 加换行；三个 token 之间是空白分隔。
# 用严格正则拆分虽然稳，但慢。我们采用"手工切分"以拉满速度：
#   - 找到第一个 "> " → subject 结束
#   - 找到第二个 "> " → predicate 结束
#   - 剩下的（去掉末尾 " .\n"）就是 object 原样文本

# 匹配 <http://.../entity/QXXX> 或 <http://.../entity/PXXX>
_ENTITY_URI_RE = re.compile(r"^<http://www\.wikidata\.org/entity/(Q\d+)>$")
# Wikidata direct-property URI：<http://www.wikidata.org/prop/direct/PXXX>
_DIRECT_PROP_RE = re.compile(r"^<http://www\.wikidata\.org/prop/direct/(P\d+)>$")
# 字面量 "xxx"@lang
_LITERAL_LANG_RE = re.compile(r'^"(.*)"@([a-zA-Z\-]+)$')
# 字面量 "xxx"^^<datatype>
_LITERAL_TYPED_RE = re.compile(r'^"(.*)"\^\^<([^>]+)>$')
# 纯字符串字面量 "xxx"
_LITERAL_PLAIN_RE = re.compile(r'^"(.*)"$')

# ---------------------------------------------------------------- 反转义
# N-Triples 规范（RDF 1.1 §7）允许把非 ASCII 字符写成 \uXXXX / \UXXXXXXXX，
# Wikidata 的 truthy dump **确实**大量使用这种写法：
#
#     <.../Q148> <...#label> "\u4E2D\u83EF\u4EBA\u6C11\u5171\u548C\u570B"@zh .
#
# ⚠️ 上面那些 _LITERAL_*_RE 只把引号里的内容原样切出来（m.group(1)），
# 拿到的是 '\','u','4','E','2','D',… 这一串 **ASCII 字面字符**，不是中文。
# 少了这一步反转义，转义串会一路原样流进 JSONL → TSV → SQLite。
# 实测污染面（10G 库）：label_zh 96.7%、mention 70.1%、description 44.1%。
#
# 后果是 L5 知识图谱层**几乎完全失效**：mention 抽取靠拿 query 的切词去
# `mentions` 表精确查表，表里存的既然是转义串，用户输入的真中文一条都对
# 不上 —— L5 每次检索都被激活、付 ~434ms，命中率却接近 0，是纯亏损。
# 而且它不报任何错，属于"静默失效"，只有专门去查表才发现得了。
#
# 【为什么不用 codecs.decode(s, "unicode_escape")】
#   ① 它走 latin-1 语义：字符串里若混有**真**中文（dump 里确实有一部分行
#      不转义）会被逐字节拆坏，产生 mojibake；
#   ② 它会把 \n \t \\ 也一并解释掉，而我们只想处理 \u/\U ——
#      实体名里出现反斜杠是合法的，不该被吃掉。
# 用正则只替换 \uXXXX / \UXXXXXXXX，对混合内容安全，且**幂等**
#（已经是真中文的字符串匹配不到，重复跑不会二次损坏）。
_NT_ESCAPE_RE = re.compile(r"\\U([0-9A-Fa-f]{8})|\\u([0-9A-Fa-f]{4})")


def _nt_unescape(s: str) -> str:
    """把 N-Triples 的 ``\\u4E2D\\u83EF`` 还原成 ``中華``。

    畸形/非法码点保持原样而不抛异常：这是在 250GB dump 的流式解析里逐行
    调用的，为一条脏数据中断几小时的扫描完全不划算。
    """
    if not s or "\\" not in s:
        return s

    def _sub(m):
        try:
            cp = int(m.group(1) or m.group(2), 16)
            # 代理区 D800-DFFF 单独出现是非法的：chr() 不报错，但写 SQLite
            # 时会炸（要求合法 UTF-8）→ 保持原样
            if 0xD800 <= cp <= 0xDFFF:
                return m.group(0)
            return chr(cp)
        except (ValueError, OverflowError):
            return m.group(0)

    return _NT_ESCAPE_RE.sub(_sub, s)

# Wikidata 里表示 "label / alias / description" 的三种 predicate URI 常量
_PRED_LABEL       = "<http://www.w3.org/2000/01/rdf-schema#label>"
_PRED_LABEL_SCHEMA = "<http://schema.org/name>"                     # 有时会用这个
_PRED_ALIAS       = "<http://www.w3.org/2004/02/skos/core#altLabel>"
_PRED_DESCRIPTION = "<http://schema.org/description>"


def _parse_line(line: bytes) -> tuple[str, str, str] | None:
    """把一行 N-Triples 切成 (subject, predicate, object_str) 三个 token。

    这里刻意不做 URL 解析或字面量类型识别，纯字符串切分为了极致速度。
    调用方按需再解析 object_str。
    """
    # 编码：Wikidata 的 dump 是 UTF-8，直接 decode，出错走宽松模式
    s = line.decode("utf-8", errors="replace")
    if not s.endswith(" .\n") and not s.endswith(" .\r\n"):
        return None
    # 去掉尾巴 " .\n"
    s = s.rstrip("\r\n")
    if s.endswith(" ."):
        s = s[:-2]
    # 一般三元组：subject predicate object，都以 '<' 开头（或 object 以 '"' 开头）
    # subject 一定是 <...>，找第一个 '> '
    i1 = s.find("> ")
    if i1 < 0:
        return None
    subj = s[:i1 + 1]
    rest = s[i1 + 2:]
    # predicate 也一定是 <...>
    i2 = rest.find("> ")
    if i2 < 0:
        return None
    pred = rest[:i2 + 1]
    obj = rest[i2 + 2:]
    return subj, pred, obj


def _extract_qid(uri: str) -> str | None:
    """<http://www.wikidata.org/entity/Q148> → 'Q148'。不匹配则返回 None。"""
    m = _ENTITY_URI_RE.match(uri)
    return m.group(1) if m else None


def _extract_pid(uri: str) -> str | None:
    """<http://www.wikidata.org/prop/direct/P36> → 'P36'。"""
    m = _DIRECT_PROP_RE.match(uri)
    return m.group(1) if m else None


def _parse_object(obj: str) -> tuple[str, str] | None:
    """把 object 字符串解析成 (value, type) 形式。

    Returns:
        ("Q956",             "entity")   若是实体 URI
        ("中国",              "string")   若是 "xxx" 或 "xxx"@lang
        ("2024-01-01T...",   "time")     若是 dateTime 字面量
        ("42",               "quantity") 若是数值字面量
        None                              若无法识别
    """
    qid = _extract_qid(obj)
    if qid:
        return qid, "entity"

    m = _LITERAL_LANG_RE.match(obj)
    if m:
        return _nt_unescape(m.group(1)), "string"

    m = _LITERAL_TYPED_RE.match(obj)
    if m:
        val, dtype = _nt_unescape(m.group(1)), m.group(2)
        if "dateTime" in dtype or "date" in dtype:
            return val, "time"
        if "decimal" in dtype or "double" in dtype or "integer" in dtype:
            return val, "quantity"
        return val, "string"

    m = _LITERAL_PLAIN_RE.match(obj)
    if m:
        return _nt_unescape(m.group(1)), "string"
    return None


def _open_bz2(path: Path):
    """打开 bz2 文件，返回二进制迭代器。用二进制模式加快解压 IO。"""
    return bz2.open(str(path), "rb")


# ---------------------------------------------------------------- Pass 1
def pass1_collect_zh_qids(dump_path: Path, out_qids_file: Path, progress_every: int) -> set[str]:
    """扫全文件，收集所有"有 zh label"的 QID。

    识别口径：predicate 是 rdfs:label 或 schema:name，object 是 "xxx"@zh 或 "xxx"@zh-*
    """
    print(f"[09] Pass 1: scanning {dump_path} for zh-labeled entities ...")
    qids: set[str] = set()
    n = 0
    n_hit = 0
    with _open_bz2(dump_path) as f:
        for line in f:
            n += 1
            if n % progress_every == 0:
                print(f"[09/P1]   lines scanned: {n:,}  zh-qids: {len(qids):,}")
            # 把一行 N-Triples 切成 (subject, predicate, object_str) 三个 token。
            parsed = _parse_line(line)
            if parsed is None:
                continue
            subj, pred, obj = parsed
            # 只关心 label / schema:name 且 object 带 @zh 语言标签
            if pred != _PRED_LABEL and pred != _PRED_LABEL_SCHEMA:
                continue
            # object 形如 "中国"@zh 或 "中國"@zh-hant
            if not obj.startswith('"'):
                continue
            # 快速判断 @zh 前缀（"...\"@zh" 或 "...\"@zh-*"）
            at = obj.rfind("@")
            if at < 0:
                continue
            lang = obj[at + 1:]
            if not (lang == "zh" or lang.startswith("zh-") or lang.startswith("zh_")):
                continue
            qid = _extract_qid(subj)    # 只返回qid字段，去除网页信息： 'Q148'
            if qid is None:
                continue
            qids.add(qid)
            n_hit += 1

    print(f"[09/P1] done. total lines: {n:,}  zh-qids: {len(qids):,}  label-hits: {n_hit:,}")
    # 落盘一份，方便断点续跑；也便于人肉抽查
    out_qids_file.parent.mkdir(parents=True, exist_ok=True)     # "data/wikidata_zh_qids.txt"
    with open(out_qids_file, "w", encoding="utf-8") as fout:
        for q in qids:
            fout.write(q + "\n")
    print(f"[09/P1] saved -> {out_qids_file}")
    return qids


def _load_qids(path: Path) -> set[str]:
    """从上一次 Pass 1 的产物加载 QID 集合。用于跳过 Pass 1 直接跑 Pass 2。"""
    print(f"[09] loading zh-qids from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        s = {ln.strip() for ln in f if ln.strip()}
    print(f"[09]   loaded {len(s):,} qids")
    return s


# ---------------------------------------------------------------- Pass 2
def pass2_extract(dump_path: Path,
                  zh_qids: set[str],
                  keep_predicates: set[str],
                  entities_out: Path,
                  triples_out: Path,
                  progress_every: int) -> None:
    """再扫一遍，产出：
    * ``entities_out``  ：每个 zh 实体的 label_zh / label_en / description / aliases 汇总（JSONL）
    * ``triples_out``   ：白名单属性上的三元组（TSV）

    为了内存友好，我们不构造 dict[qid, {...}] 在内存里累加所有 label，而是
    先把"每条 label/alias/description 行"以 TSV 追加落地一个 tmp 文件，
    最后再做一次 "按 qid 聚合" 的 external sort 或者简单的分块聚合。

    但实际上 zh 实体只有 ~300 万，每个实体的 label/alias 也就 <10 条，
    直接在内存里累积一个 dict[qid -> {..., "aliases": set()}] 峰值内存
    大约 ~2-4 GB，普通开发机也够用。这里选后者，代码更清爽。
    """
    print(f"[09] Pass 2: extracting entities & triples ...")
    entities_out.parent.mkdir(parents=True, exist_ok=True)          # "data/wikidata_zh_entities.jsonl"
    triples_out.parent.mkdir(parents=True, exist_ok=True)           # "data/wikidata_zh_triples.tsv"

    # 实体主表：qid -> {label_zh, label_en, description, aliases:set()}
    ent: dict[str, dict] = {}

    n = 0
    n_triples = 0
    # triples 直接流式写盘，避免全部堆内存
    ftri = open(triples_out, "w", encoding="utf-8")

    def _ensure(qid: str) -> dict:
        d = ent.get(qid)
        if d is None:
            d = {"qid": qid, "label_zh": None, "label_en": None,
                 "description": None, "aliases": set()}
            ent[qid] = d
        return d

    with _open_bz2(dump_path) as f:
        for line in f:
            n += 1
            if n % progress_every == 0:
                print(f"[09/P2]   lines: {n:,}  entities-so-far: {len(ent):,}  triples: {n_triples:,}")

            # 把一行 N-Triples 切成 (subject, predicate, object_str) 三个 token。不做任何预处理！！！
            parsed = _parse_line(line)
            if parsed is None:
                continue
            subj, pred, obj = parsed
            # 提取qid: <http://www.wikidata.org/entity/Q148> → 'Q148'。不匹配则返回 None。
            subj_qid = _extract_qid(subj)
            if subj_qid is None or subj_qid not in zh_qids:
                continue

            # ---- 分支 A：label / alias / description ----
            # 数据格式如下：
            """
            subj || pred || obj
            # label —— 首选名（用 rdfs:label 或 schema:name，两条其实一样）
            <http://www.wikidata.org/entity/Q148> <http://www.w3.org/2000/01/rdf-schema#label> "中华人民共和国"@zh .
            <http://www.wikidata.org/entity/Q148> <http://schema.org/name>                     "中华人民共和国"@zh .

            # alias —— 别名（用 skos:altLabel，可以有多行）
            <http://www.wikidata.org/entity/Q148> <http://www.w3.org/2004/02/skos/core#altLabel> "中国"@zh .
            <http://www.wikidata.org/entity/Q148> <http://www.w3.org/2004/02/skos/core#altLabel> "PRC"@zh .
            <http://www.wikidata.org/entity/Q148> <http://www.w3.org/2004/02/skos/core#altLabel> "神州"@zh .

            # description —— 简短描述
            <http://www.wikidata.org/entity/Q148> <http://schema.org/description> "东亚国家"@zh .
            """
            if pred == _PRED_LABEL or pred == _PRED_LABEL_SCHEMA:       # 表示是label
                # "xxx"@lang
                m = _LITERAL_LANG_RE.match(obj)
                if not m:
                    continue
                # ⚠️ 必须 _nt_unescape：dump 里中文是 \\uXXXX 转义的，
                # 直接用 m.group(1) 会把 '\\u4E2D\\u83EF...' 当成 label 存进库。
                val, lang = _nt_unescape(m.group(1)), m.group(2)
                if lang == "zh" or lang.startswith("zh"):
                    d = _ensure(subj_qid)
                    # 优先保留最短的 zh 主 label（zh-cn / zh-hans 优先度更高，但先到先得也可）
                    if d["label_zh"] is None:
                        d["label_zh"] = val
                # 英文 label 是"救急兜底"：某些冷门实体可能没登记中文 label（比如某种化学物质、生物学种、外国小地名），这时候有英文 label 兜底不至于变成空白
                elif lang == "en":
                    d = _ensure(subj_qid)
                    if d["label_en"] is None:
                        d["label_en"] = val
                continue

            if pred == _PRED_ALIAS:             # 表示是alias
                m = _LITERAL_LANG_RE.match(obj)
                if not m:
                    continue
                val, lang = _nt_unescape(m.group(1)), m.group(2)
                if lang == "zh" or lang.startswith("zh"):
                    d = _ensure(subj_qid)
                    d["aliases"].add(val)
                continue

            if pred == _PRED_DESCRIPTION:       # 表示是 description
                m = _LITERAL_LANG_RE.match(obj)
                if not m:
                    continue
                val, lang = _nt_unescape(m.group(1)), m.group(2)
                if lang == "zh" or lang.startswith("zh"):
                    d = _ensure(subj_qid)
                    if d["description"] is None:
                        d["description"] = val
                continue

            # ---- 分支 B：白名单里的 direct property ----
            # 数据格式如下: 谓词是 direct property URI （PID）
            """
            格式一：宾语是另一个实体（Entity）
            <http://www.wikidata.org/entity/Q148>
                <http://www.wikidata.org/prop/direct/P36>
                <http://www.wikidata.org/entity/Q956> .
            # 语义：中国 的首都 是 北京
            格式二：宾语是带语言标签的字符串（LangString）
            <http://www.wikidata.org/entity/Q148>
                <http://www.wikidata.org/prop/direct/P1448>
                "中华人民共和国"@zh .
            # 语义：中国 的官方名称 是 "中华人民共和国"（中文）
            格式三: 宾语是带 datatype 的字面量（TypedLiteral）—— 日期
            <http://www.wikidata.org/entity/Q148>
                <http://www.wikidata.org/prop/direct/P571>
                "1949-10-01T00:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime> .
            # 语义：中国 的成立日期 是 1949-10-01
            格式四: 宾语是带 datatype 的字面量 —— 数值
            <http://www.wikidata.org/entity/Q148>
                <http://www.wikidata.org/prop/direct/P1082>
                "1411780000"^^<http://www.w3.org/2001/XMLSchema#decimal> .
            # 语义：中国 的人口 约为 14 亿
            格式五: 宾语是纯字符串（无语言无类型）
            <http://www.wikidata.org/entity/Q148>
                <http://www.wikidata.org/prop/direct/P2853>
                "AC 220V/50Hz" .
            # 语义：中国 的插座标准 描述为 "AC 220V/50Hz"
            """
            pid = _extract_pid(pred)
            if pid is None or pid not in keep_predicates:
                continue

            parsed_obj = _parse_object(obj)
            if parsed_obj is None:
                continue
            obj_val, obj_type = parsed_obj

            # 若 object 是实体，要求 object 也是 zh 实体，避免拉出一堆纯外文孤儿点
            if obj_type == "entity":
                if obj_val not in zh_qids:
                    continue
                # 三元组 TSV：subject_qid \t pid \t object_qid \t "" \t entity
                ftri.write(f"{subj_qid}\t{pid}\t{obj_val}\t\tentity\n")
            else:
                # 字面量：object_qid 留空，object_value 存值
                # 对 value 做保守转义：\t 和 \n 都替成空格，避免破坏 TSV
                v = obj_val.replace("\t", " ").replace("\n", " ").replace("\r", " ")
                ftri.write(f"{subj_qid}\t{pid}\t\t{v}\t{obj_type}\n")
            n_triples += 1

    ftri.close()
    print(f"[09/P2] triples written: {n_triples:,}  -> {triples_out}")

    # 写实体主表
    print(f"[09/P2] writing entities to {entities_out} ...")
    n_ent = 0
    with open(entities_out, "w", encoding="utf-8") as fout:     # "data/wikidata_zh_entities.jsonl"
        for qid, d in ent.items():
            # set 不能直接 json，转 list
            d["aliases"] = sorted(d["aliases"])
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            n_ent += 1
    print(f"[09/P2] entities written: {n_ent:,}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--skip-pass1", action="store_true",
                    help="跳过 Pass 1，直接从已有 kg_zh_qids_file 加载 QID 集合")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dump = cfg["paths"]["wikidata_dump"]                # "data/wikidata-truthy-latest.nt.bz2"
    qids_file = cfg["paths"]["kg_zh_qids_file"]         # "data/wikidata_zh_qids.txt"
    triples_file = cfg["paths"]["kg_triples_tsv"]       # "data/wikidata_zh_triples.tsv"
    entities_file = cfg["paths"]["kg_entities_jsonl"]   # "data/wikidata_zh_entities.jsonl"

    keep_pred = set(cfg["kg"]["keep_predicates"])   # 这份 ~50 个PID 覆盖了"是什么 / 属于哪 / 谁做的 / 何时 / 在哪 / 关联到"这几类高频问答。
    progress_every = cfg["kg"]["progress_every"]        # 出于加速目的，每 N 行打印一次进度

    # 扫全文件，收集所有"有 zh label"的 QID。并保存在 "data/wikidata_zh_qids.txt"
    if args.skip_pass1 and qids_file.exists():
        zh_qids = _load_qids(qids_file)
    else:
        zh_qids = pass1_collect_zh_qids(dump, qids_file, progress_every)

    print(f"[09] keep predicates: {len(keep_pred)}  (whitelist)")
    pass2_extract(dump, zh_qids, keep_pred, entities_file, triples_file, progress_every)

    print("[09] done. next: run 10_build_kg_sqlite.py")


if __name__ == "__main__":
    main()
