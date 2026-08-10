# RAG 分层记忆系统

> Perplexity 风格的 **“越用越强”** 检索栈。`src/cache/qa_cache.py` 是 L1；本目录把它扩展成
> 5 层，一起走过 Router → 并行召回 → 融合去重 → 校准判定 → 回答。
>
> L2 常识层用 **Wikipedia dump** 构建 FAISS 向量库，L5 知识图谱层用
> **Wikidata "truthy" dump** 构建 SQLite 图谱。全库 embedding
> **统一为 FlagEmbedding BGE-M3**，数据统一落在项目根的 `data/rag_data/`。
>
> 本文件聚焦 **rag/ 目录内部**的分层设计与离线索引构建；
> 跨层校准、可答性判据、实体配额、去重等**判据层**的设计动机见
> [根目录 README](../README.md) §4–§5。

## 1. 五层结构

| 层  | 名称           | 存储                       | 更新方式               | 主要用途 |
|-----|----------------|----------------------------|------------------------|----------|
| L1  | QA Cache       | diskcache/redis            | agent 命中即写         | 精准/模糊高频问答，毫秒级 |
| L2  | Commonsense    | FAISS（**仅 Wikipedia**）  | 离线批处理             | 教科书常识（“量子计算”、“秦始皇”）|
| L3  | History        | FAISS（增量）              | **每次成功回答异步写** | 用户偏好 / 领域术语 / 复用推理 |
| L4  | Web            | 无（实时）                 | 实时联网               | 时事、时间敏感 |
| L5  | Knowledge Graph| SQLite（**Wikidata truthy**）| 离线批处理           | 结构化事实（“X 的 CEO 是谁”），支持多跳 |

## 2. 目录

```
rag/
├── __init__.py
├── config.py               # rag 编排器配置 + 桥接 wiki_rag YAML
├── configs/
│   └── default.yaml         # wiki_rag 全部可调参数（L2/L5 数据路径等）
├── types.py                # Passage / RetrievalResult / LayerName（对外契约）
├── embedder.py             # 统一 BGE-M3 编码器（底层 = FlagEmbedding）
├── vector_store.py         # faiss + numpy fallback 的可插拔向量存储（L3 用）
├── layers.py               # L1~L5 五个层（L2→WikiRetriever，L5→KGRetriever）
├── router.py               # 层激活策略（rule-based，可换 LLM）
├── fusion.py               # RRF 融合 + 并列实体配额 + 可选 BGE/cascade rerank
├── entities.py             # 并列实体识别（jieba 词性 + 连接词，供配额融合用）
├── dedup.py                # 近重去重（余弦硬删）+ MMR 多样性重排
├── calibration.py          # 跨层分数校准（Platt scaling + 噪声-OR 聚合）
├── answerability.py        # 证据可答性判据（实词覆盖率，与置信度正交）
├── textclean.py            # L4 snippet 噪声清洗（纯正则，零依赖）
├── incremental_worker.py   # 后台增量写 L3 的 worker
├── retriever.py            # LayeredRetriever（对外主入口）
├── wiki_rag/               # vendored 检索库（L2/L5 内核）
│   ├── retriever.py         #   WikiRetriever（L2：FAISS + BGE-M3）
│   ├── kg_retriever.py      #   KGRetriever（L5：mention→link→triples→多跳）
│   ├── kg_store.py          #   KGStore（SQLite 只读封装）
│   ├── linker.py            #   Linker（mention→qid 消歧 + 热门实体向量重排）
│   ├── embedder.py          #   BGE-M3 单例编码器（跨版本兼容 + 设备自适应）
│   ├── hybrid.py            #   rerank 工厂（bge/rrf/cascade）
│   ├── warmup.py            #   统一预热
│   ├── chunker.py           #   文本切分（构建期）
│   └── config.py            #   YAML 加载（路径锚定到项目根）
├── README.md               # 本文件
└── scripts/                # 离线构建流水线（01~12，见 §3.3）
```

## 3. 快速开始

### 3.1 安装依赖

```bash
# 统一 embedding 栈（BGE-M3）：版本耦合较紧，建议成组安装
pip install "FlagEmbedding>=1.3" "transformers>=4.50" "tokenizers>=0.20" "torch>=2.6"
# 向量库 & 配置
pip install faiss-cpu numpy PyYAML        # 有 GPU 可换 faiss-gpu
# 可选：L5 mention 抽取增强（不装自动降级为 n-gram）
pip install jieba
```

> 依赖组合说明见 `requirements.txt` 顶部注释。无 GPU 时 BGE-M3 亦可跑 CPU，
> 首查较慢，稳态可接受。

若想启用更强的二阶重排（默认是零依赖 RRF）：
```bash
export RAG_RERANK_STRATEGY=bge       # 或 cascade（先 RRF 粗排再 BGE 精排）
```

### 3.2 一分钟接入 agent

```python
from src.cache.qa_cache import QACache
from src.rag import LayeredRetriever

qa = QACache(backend="diskcache", enable_fuzzy=True)   # 向量后端已统一 BGE-M3
retriever = LayeredRetriever(qa_cache=qa)

# 检索
result = retriever.retrieve("量子计算是什么")
if result.cache_hit:
    answer = result.cache_answer                 # L1 短路
else:
    # 生产链路请用 evidence.build_evidence_block(result.passages)：
    # 它会做证据清洗 + <doc> 结构化定界（Prompt Injection 防护）。
    # as_context_block() 是无防护的纯文本降级路径，仅作调试/兜底用。
    from src.pipeline.evidence import build_evidence_block
    context, _ = build_evidence_block(result.passages)
    answer = call_llm(query, context)

# 归档（异步写 L3，越用越强）
retriever.archive("量子计算是什么", answer,
                  sources=[p.to_dict() for p in result.passages])
```

### 3.3 构建离线索引（L2 Wikipedia + L5 Wikidata）

全部脚本在 `rag/scripts/`，读取同一份 `rag/configs/default.yaml`；
产物统一落到项目根 `data/rag_data/`。从 `rag/` 目录下运行：

```bash
cd rag

# ===== L2：Wikipedia 常识向量库 =====
bash   scripts/01_download.sh                     # 下载 zhwiki dump
bash   scripts/02_extract.sh                      # wikiextractor 解析
python scripts/03_fetch_pageviews.py              # 拉取 pageviews（定热度）
python scripts/04_filter_top_articles.py          # 取 Top-N 热门条目
python scripts/05_build_chunks_and_embed.py       # 切 chunk + BGE-M3 编码
python scripts/06_build_faiss.py                  # 建 FAISS 索引

# ===== L5：Wikidata truthy 知识图谱 =====
bash   scripts/08_download_wikidata.sh            # 下载 truthy dump
python scripts/09_filter_wikidata_zh.py           # 过滤中文实体 + 裁剪 predicate
python scripts/10_build_kg_sqlite.py              # 导入 SQLite（含 FTS5）
python scripts/11_encode_hot_entities.py          # 热门实体 description 向量

# ===== 验证检索（可选）=====
python scripts/07_demo_retrieve.py --query "苹果公司的CEO是谁"
python scripts/12_demo_kg_query.py  --query "库克领导的公司在哪个城市"
```

## 4. 检索流程

```
                            ┌──────────────┐
              user query ─► │ L1 QACache   │ ── hit ─► cache_answer  (毫秒级)
                            └──────┬───────┘
                              miss │
                                   ▼
                            ┌──────────────┐
                            │  Router      │  规则/时间敏感检测
                            └──────┬───────┘
                                   │
                ┌──────────────┬───┴────────────┬──────────────┐
                ▼              ▼                ▼              ▼
          ┌─────────┐    ┌──────────┐     ┌─────────┐    ┌──────────┐
          │ L2 wiki │    │L3 history│     │  L5 KG  │    │(L4 web ?)│
          │WikiRetr.│    │ (FAISS)  │     │KGRetr.  │    │(searcher)│
          └────┬────┘    └────┬─────┘     └────┬────┘    └────┬─────┘
               └──────┬──────┴──────────┬──────┘              ▲
                      ▼                 ▼                     │
                      └───► RRF + 实体配额融合                 │
                                  │                           │
                                  ▼                    离线证据不足则补 L4：
                       近重去重 + MMR 多样性            校准置信度 < 0.55
                                  │                       ─ 或 ─
                                  ▼                    实词覆盖率 < 0.6
                     (可选 BGE/cascade rerank)  ────────────┘
                                  │
                                  ▼
                          跨层分数校准（Platt）
                       → confidence / low_evidence
                                  │
                                  ▼
                         top-K Passages → summary LLM
```

L4 兜底的两个触发条件取 **OR**：置信度衡量「语义像不像」，实词覆盖率衡量
「够不够回答」，二者正交、各自捕捉一类失败模式（详见根 README §4.2）。

## 5. 关键设计

### 5.1 统一 Embedding（D1）
L1~L5 全部使用 **FlagEmbedding BGE-M3**（1024 维、L2 归一化、内积=cosine）：
- L2 的 FAISS 索引、L5 的热门实体向量都由 BGE-M3 离线构建，查询侧同源编码，
  召回不掉点；
- L1（qa_cache 模糊命中）、L3（历史归档）复用同一向量空间，跨层 RRF 可比。

### 5.2 RRF 融合（默认 rerank 策略）
`score(d) = Σ_layer 1 / (k + rank_layer(d))`，`k=60`。零依赖、零延迟，天然适配
“每层打分口径不同”。BGE/cascade 精排通过 `RAG_RERANK_STRATEGY` 开启。

### 5.3 惰性加载
- L2/L5 首次检索时才加载 GB 级 FAISS/SQLite，构造后进程内复用；避免 agent 启动
  即加载大文件。可在服务就绪阶段显式调用各层 `warmup()` 预热。

### 5.4 增量索引 worker（L3“越用越强”）
- 主流程调 `retriever.archive(...)` 立即返回（`queue.put_nowait`）；
- 后台 daemon 线程按 batch/interval 刷盘；队列满**丢弃并告警**，不阻塞用户。
- 归档前会经 `cache_policy` 过滤：拒答类答案**不写 L3**，否则会形成
  「拒答 → 入库 → 下次召回到自己的拒答 → 置信度虚高 → 不触发 L4 → 再次拒答」
  的自我强化失败循环。已污染的历史可用 `scripts/clean_l3_refusals.py` 清理。

### 5.5 路由用原始 query、检索用改写后 query
`retrieve(query, route_query=...)` 把两者分开：改写器可能给历史累计型问题
（「历届获奖名单」）凭空补上当前年份，使其被误判为时效敏感而强制联网。
时效性是**用户意图**的属性，应由原始 query 判定。

## 6. 与 agent.py 的集成契约

```python
from src.rag import LayeredRetriever

retriever = LayeredRetriever(qa_cache=self.qa_cache, strategy=rag_strategy)
result = retriever.retrieve(user_input)      # -> RetrievalResult
if result.cache_hit:
    return result.cache_answer
context_block = result.as_context_block()    # 喂给 summary LLM
...
retriever.archive(user_input, answer,
                  sources=[p.to_dict() for p in result.passages])
retriever.close()
```

`RetrievalResult` 的字段契约（详见 `types.py`）：

| 字段 | 含义 |
|---|---|
| `cache_hit` / `cache_answer` | L1 是否命中及其答案（命中即可短路返回） |
| `passages` | 融合去重后的 top-K 证据（`Passage.score` 为层内原始分） |
| `layer_hits` | 各层召回条数，如 `{"L2_commonsense": 3, "L5_kg": 2}` |
| `confidence` | 跨层校准 + 噪声-OR 聚合后的整体置信度 ∈ [0,1] |
| `low_evidence` | abstention 信号：证据不足，应引导模型明说「资料不足」 |
| `web_fallback` | 本轮是否触发了 L4 联网兜底 |
| `term_coverage` / `missing_terms` | 实词覆盖率与缺失词（排查用） |

## 7. 配置速查

编排器可调项在 `rag/config.py`（环境变量优先）；L2/L5 数据路径与参数在
`rag/configs/default.yaml`。

| 变量 | 默认 | 含义 |
|---|---|---|
| `RAG_EMBED_MODEL` | `BAAI/bge-m3` | FlagEmbedding 模型名（HF 仓库） |
| `RAG_EMBED_DIM`   | `1024`   | 维度 |
| `RAG_EMBED_DEVICE`| `auto`   | `auto`/`cuda`/`cuda:0`/`mps`/`cpu` |
| `RAG_DATA_DIR`    | `<项目根>/data/rag_data` | 数据落盘根（L3 等） |
| `RAG_ROUTER_STRATEGY` | `hybrid` | `hybrid` / `offline_only` / `web_only` |
| `RAG_WEB_FALLBACK`| `0.55` | 校准后的聚合置信度低于此值时补 L4 |
| `RAG_ABSTAIN_CONF` | `0.30` | 低于此值标记 `low_evidence`（引导模型明说资料不足） |
| `RAG_MIN_TERM_COVERAGE` | `0.6` | 实词覆盖率低于此值也补 L4（与置信度取 OR） |
| `RAG_CALIBRATION_FILE` | 空 | Platt 参数 JSON 热加载路径（`fit_platt()` 产物） |
| `RAG_RERANK_STRATEGY` | `rrf` | `rrf`（默认）/ `bge` / `cascade` / `none` |
| `RAG_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | BGE reranker 权重 |
| `RAG_ENABLE_NEAR_DUP` | `true` | 近重去重（余弦 ≥ 0.95 硬删转载稿） |
| `RAG_ENABLE_MMR` | `true` | MMR 多样性重排（λ=0.7） |
| `RAG_DEDUP_LAYERS` | `L4_web` | 只对哪些层做语义去重（留空 = 全量，更彻底但更慢） |
| `RAG_FUSION_CANDIDATE_MULT` | `3` | 候选池放大倍数（去重后要有料可补位） |
| `RAG_FUSION_MIN_PER_ENTITY` | `2` | 并列实体每个保底席位数 |
| `RAG_ENABLE_SNIPPET_CLEAN` | `true` | L4 snippet 噪声清洗 |
| `RAG_KG_MULTI_HOP` | `false` | L5 是否开启多跳 BFS |
| `RAG_KG_MAX_HOPS`  | `2` | 多跳最大跳数 |

各参数的**标定依据**（为什么是这个值）写在 `rag/config.py` 对应常量处的注释里。

## 8. 与已有模块的复用关系

- **L1** 复用 `qa_cache.QACache`（向量后端已统一 BGE-M3，ollama 仅兜底）
- **L4** 直接调 `searcher.web_search`
- **L2/L5** 复用 vendored `wiki_rag`（FAISS + Wikidata SQLite）
- **持久化** 与项目风格一致：diskcache + faiss + sqlite，无强依赖服务

## 9. 扩展点

**新增一层**需要三处登记，缺一层就不会生效：

1. `layers.py` 实现 `search(query, top_k, namespace=None) -> list[Passage]`
2. `router.py` 在层激活策略里登记（决定什么 query 会激活它）
3. `calibration.py` 补一组 Platt 参数（否则该层分数无法与其他层比较）

**路线图**

- [ ] LLM Router：把 rule-based 换成小模型分类
- [ ] L3 老化：给历史归档加 TTL 或 LFU，防止越攒越乱
- [ ] L2/L5 索引热更新：不停服替换 FAISS/SQLite
- [ ] KG Tool：把 L5 包装成 Function Calling / MCP tool 让 LLM 自主调用
- [ ] 多 query 并发检索：在实体配额之上的可选增强（`entities.py` 已就绪）
