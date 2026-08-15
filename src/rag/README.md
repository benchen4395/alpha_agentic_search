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
└── scripts/                # 离线构建流水线（01~15，见 §3.3）
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
# 从**项目根目录**运行（脚本内部把路径锚定到项目根）

# ===== L2：Wikipedia 常识向量库 =====
bash   src/rag/scripts/01_download.sh               # 下载 zhwiki dump
bash   src/rag/scripts/02_extract.sh                # wikiextractor 解析
python src/rag/scripts/03_fetch_pageviews.py        # 拉取 pageviews（定热度）
python src/rag/scripts/04_filter_top_articles.py    # 取 Top-N 热门条目
python src/rag/scripts/05_build_chunks_and_embed.py # 切 chunk + BGE-M3 编码
python src/rag/scripts/06_build_faiss.py            # 建 FAISS 索引

# ===== L5：Wikidata truthy 知识图谱 =====
bash   src/rag/scripts/08_download_wikidata.sh      # 下载 truthy dump
python src/rag/scripts/09_filter_wikidata_zh.py     # 过滤中文实体 + 裁剪 predicate
python src/rag/scripts/10_build_kg_sqlite.py        # 导入 SQLite（FTS5 + 入度 + 别名）
python src/rag/scripts/11_encode_hot_entities.py    # 热门实体 label/description 向量

# ===== 验证检索（可选）=====
python src/rag/scripts/07_demo_retrieve.py --query "苹果公司的CEO是谁"
python src/rag/scripts/12_demo_kg_query.py  --query "库克领导的公司在哪个城市"
```

`10_build_kg_sqlite.py` 内部已串好三件事，**从零重建时不需要额外操作**：
导入三元组 → 回填 `popularity`（实体入度）→ 补后缀别名 → 建索引。

#### 脚本清单

| 脚本 | 层 | 作用 | 幂等 |
|---|---|---|---|
| `01`~`02` | L2 | 下载 / 解析 zhwiki dump | — |
| `03`~`04` | L2 | pageviews 热度 → Top-N 条目 | ✅ |
| `05`~`06` | L2 | 切 chunk + BGE-M3 编码 → FAISS | ✅ |
| `07` | L2 | 检索 demo | ✅ |
| `08`~`09` | L5 | 下载 / 过滤 Wikidata truthy dump | ✅ |
| `10` | L5 | 建 SQLite（含入度回填 + 后缀别名） | ✅ |
| `11` | L5 | 热门实体向量（**依赖 label，label 变了必须重跑**） | ✅ |
| `12` | L5 | KG 查询 demo | ✅ |
| `13` | L5 | **修复** `\uXXXX` 转义污染（存量库补丁） | ✅ |
| `14` | L5 | **回填** `popularity` 入度（存量库补丁） | ✅ |
| `15` | L5 | **补** 后缀剥离别名（存量库补丁） | ✅ |

`13`/`14`/`15` 是给**已经建好的库**打的增量补丁，等价逻辑都已内联进
`09`/`10`，所以全量重建不必跑它们。三个脚本均**幂等**，可重复执行。

### 3.4 每周增量更新

Wikidata/Wikipedia 每周出新 dump。全量重建 L5 需 2~4 小时，
若只想刷新数据而不动索引结构，按下面顺序跑：

```bash
# ① 重新拉取并过滤 dump（09 已内置 \uXXXX 反转义）
bash   src/rag/scripts/08_download_wikidata.sh
python src/rag/scripts/09_filter_wikidata_zh.py

# ② 重建 SQLite（自动完成入度回填 + 后缀别名）
python src/rag/scripts/10_build_kg_sqlite.py

# ③ ⚠️ 必须重新编码热门实体向量
RAG_EMBED_DEVICE=cpu python src/rag/scripts/11_encode_hot_entities.py \
    --config src/rag/configs/default.yaml

# ④ 冒烟验证
python src/rag/scripts/12_demo_kg_query.py --query "北京是哪个国家的首都"
```

**第 ③ 步不可跳过**。热门实体向量是拿 `label`/`description` 编出来的，
label 一变，旧向量就与新库对不上。实测一次 label 修复后新旧向量的
cosine 中位数只有 **0.685** —— 不重编等于让 L5 拿着一份错误的语义索引。

若只是给**存量库**打补丁（不重新下载 dump），跑增量脚本即可：

```bash
python src/rag/scripts/13_fix_kg_unicode_escape.py   # 先干跑
python src/rag/scripts/13_fix_kg_unicode_escape.py --apply
python src/rag/scripts/14_backfill_popularity.py --apply
python src/rag/scripts/15_gen_suffix_aliases.py --apply   # 依赖 14 的 popularity
```

> 顺序不能颠倒：`15` 用 `popularity` 筛"值得补别名的高知名度实体"，
> 在 `14` 之前跑会因为入度全是 0 而一条都补不出来（脚本会直接报错退出）。
> 三个脚本都支持**先干跑再 `--apply`**，建议保持这个习惯。
> 改完库需要**重启服务**：`KGStore` 是进程内单例，不会感知文件变化。

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

### 5.4 L5 泛实体重排（mention 抽取后、消歧前）

抽出来的 mention 不都是实体。实测 BrowseComp-ZH 那道混淆式多跳题：

> 「哪个地方拥有AAAAA级景区、被称为成熟周期很长的水果之乡，并且有一位科学家
> 曾在欧洲知名大学学习后回国奠定了一个学科基础？」

hybrid 抽出 7 个 mention：`['AAA', '科学家', '地方', '景区', '水果', '欧洲', '学科']`，
**只有「欧洲」是真实体**，其余 6 个都是类/泛实体。

**真正的杀手是预算挤占，不是排序噪声。** `query_kg_end_to_end` 按 mention
顺序消耗 `max_entities`（默认 5）额度：

```
AAA(+2) → 科学家(+2) → 地方(+1) → 预算耗尽，硬 break
```

于是 `['景区','水果','欧洲','学科']` **完全没有机会被链接**——唯一的真实体
被泛实体挤掉了。这意味着**mention 的顺序就是优先级**，排序层再好也救不回来
一个从未进入候选池的实体。

#### 判别信号选型（全部实测过）

| 信号 | 结论 |
|---|---|
| 候选实体数（歧义度） | ✗ 爱因斯坦=4 vs 城市=10，真实体并不更少 |
| jieba 词性 | ✗ 城市→`ns`(专名)、苹果公司→`n`(普通词)，不可靠 |
| 候选自身是否有 P279 | ✗ **一定不能用**：概念天然是某类的子类，量子纠缠/相对论自身都有 P279，会误杀真实体 |
| **P31/P279 入度** | ✓ 泛实体 城市=5853、公司=2391、地方=1030；真实体 爱因斯坦/北京/苹果公司/相对论 全部 = 0 |

入度的语义是"有多少实体声明**我是它的实例/子类**"——这是数据驱动的，
不需要维护停用词表，也不依赖语言。

#### 两个工程细节

**封顶计数。** 朴素 `COUNT(*)` 在「城市」上要 349ms（扫 5853 行），在线不可接受。
但我们只需要知道"是否 ≥ 阈值"，不需要精确值：

```sql
SELECT COUNT(*) FROM (SELECT 1 FROM triples
                      WHERE predicate_pid=? AND object_qid=? LIMIT ?)
```

实测 349ms → 0.16ms（**2209×**）。

**必须探测多个候选。** 不能只看 popularity top-1：实测「大学」的 top-1 是
`Q1069886「著作」`（一本书），入度=0，于是漏判。但也不能探太多——每个候选要
2 次索引查询，5 个候选时单 query 达 660ms。取 3 个，配合进程内缓存
（判定是纯函数且泛实体天然高频复现）后单 query 降到 ~0ms，
整体 639ms → 29.7ms（**21.5×**）。

#### 策略：重排优先于过滤

默认只做**稳定重排**（`specific + generic`），不丢弃任何 mention。
这样误判的最坏后果是顺序变化，而不是信息丢失。`drop_generic=True` 时
若全部被判为泛实体则回退原列表——宁可带噪声，也不能一个都不剩。

> **修完之后准确率仍是 0/12。** 因为 BrowseComp-ZH **故意混淆实体名**，
> 题干里客观上没有具名实体可抽（12/12 题实词覆盖率 0.07~0.35，全部走 L4 联网）。
> 修复确实生效——9/12 题的 mention 预算分配发生了变化，唯一真实体「欧洲」
> 成功进入了候选——但这个数据集的瓶颈不在 L5。如实记录以免误导后续判断。

### 5.5 L5 实体消歧排序

`mention → qid` 的候选排序用 **组合分**，而不是字典序：

```
score = weight × log1p(popularity)
```

- `weight`：来源先验（label=1.0 / label_en=0.8 / alias=0.6 / 推导别名=0.5）
- `popularity`：实体**入度**（被多少三元组指向），由 `10` 脚本回填

两个信号必须**相乘**而非分主次。若写成 `ORDER BY weight DESC, popularity DESC`，
weight 就成了第一优先级，「冷门实体的 label(1.0)」会永远压过
「主实体的 alias(0.6)」——而 Wikidata 主实体的 label 常是繁体，
简体串恰恰只能作为 alias 命中，正好落在低权重档。

用 `log` 而非线性：入度分布跨 6 个数量级（0 ~ 106 万），
线性会让超高频实体碾压一切，取对数压缩后 weight 的档位差仍有话语权。

排序前还会把**维基媒体内部页面**（消歧义页 / 项目分类 / 模板，共约 63 万条）
整体沉底：它们的 label 与真实体完全同名，但只能拼出
「· 是一个 维基媒体消歧义页」这类纯噪声。

> `linker.py` 的向量重排会在此基础上融合语义相似度
> （`0.6×cosine + 0.4×先验分`）。这一层**必须把先验分带上**，
> 否则 KGStore 算好的入度信号会被整个覆盖掉。

**为什么不用 `article_rank`**：它依赖 `04` 步的 pageviews 数据，
覆盖率只有 2,869/10,385,628 ≈ **0.03%**，无法作为主排序键，
现仅保留为次级 tie-breaker。

#### 5.5.1 上下文相关的两层信号（`linker.py`）

上面的组合分只看实体自身，**与 query 无关**。而向量重排那层的 cosine
只在候选命中热门实体向量库时才算——同样是 2871 条、0.03% 覆盖率。
两者叠加的结果是：绝大多数候选走 `cold_pos` 分支拿 `0.5 × prior`，
`query_context` 参数**传了却完全没起作用**，消歧退化成"按热度返回固定答案"：

```
"特斯拉发明了交流电"      → 特斯拉公司(0.07)   ✗ 应为 尼古拉·特斯拉
"特斯拉发布了新款电动车"  → 特斯拉公司(0.07)   ✓
```

注意两行的**分数完全相同**——这是判定"上下文没生效"的直接证据。

**① description 词面重叠**（`context_lexical_score`）

取 description 与 query 的字符 bigram 交集。两个实现细节都是标定出来的：

- **用 bigram 而非单字**：单字在中文里太容易偶然命中（「的」「国」「人」），
  会让所有候选都拿到分，从而失去判别力。
- **用 description 侧作分母**（而非并集或 query 侧）：要问的是"这条 description
  有多少内容被 query 提到了"，属于**精确率**语义。若用 query 侧作分母，
  长 query 会把所有候选的分数一起压低，候选间的相对差异被抹平。

反直觉的地方是**这里不该上语义向量**。实测三方对比（9 个消歧 case，
正确答案均在候选内）：

| 方案 | 准确率 | 延迟 |
|---|---|---|
| 现状（纯先验） | 5/9 | 0ms |
| 给冷门候选现算 description embedding | **4/9** | **673ms/题** |
| description 词面重叠 | **8/9** | ~0ms |

embedding 方案**比什么都不做还差**，原因很具体：description 极短
（「磁感應單位強度」7 字、「车型」2 字），BGE-M3 在这个长度上向量不稳定，
反而把「特斯拉工廠」「苹果产品发布会」这类**字面接近但语义错误**的候选
拉了上来，还弄坏了本来正确的 3 个 case。

**② 类型-意图匹配**（`query_person_intent` + `_is_human`）

词面分救不了 description 与 query 零重叠的情况：

```
"牛顿提出了万有引力定律" → 牛顿(Q49196, desc="美国城市", pop=135)  ✗
                            应为 艾薩克·牛頓(Q935, pop=60)
```

两个候选的 description 与 query 都零重叠，ctx 都是 0.0，只能比先验热度，
而同名城市的入度反而更高，于是必错。

突破口是实体侧的类型信号异常干净——`P31 → Q5(人類)` 是**二值无歧义**的：

| 实体 | P31 | P106（职业） |
|---|---|---|
| 艾薩克·牛頓 | `人類(Q5)` | 数学家, 哲學家 |
| 牛顿（城市） | 美国城市, 城市, 郊区 | — |
| 尼古拉·特斯拉 | `人類(Q5)` | 发明家, 电气工程师… |
| 特斯拉公司 | 法人團體, 企業 | — |

query 侧则利用中文主谓约束：「提出/发明/创立/担任/毕业于」的主语必须是人，
「位于/成立于/人口/面积/单位/等于」的主语必须不是人。

**关键设计是无线索时完全不介入**：`query_person_intent` 返回 0 时类型分恒为 0，
排序与不启用本特性时逐位相同。实测 11 个 case 有 5 个属于此类
（「水星是太阳系最内侧的行星」「长城是中国古代军事防御工程」等），
它们本来就排对了——这从结构上保证**只可能修好、不可能弄坏**，实测弄坏 0 个。

权重 `_TYPE_W = 0.30` 的标定依据：需能翻盘「牛顿」那组约 0.18 的先验分差
（±0.30 给出 0.60 的相对差，够），同时必须 `< _CTX_W(0.5)`，
保证 description 明确命中时词面分仍占主导。类型分在冷门分支**不乘 0.5**——
它是二值判定，与候选冷热无关，而人物实体恰恰大量是冷门的
（艾薩克·牛頓 pop=60 < 同名城市 135），跟着缩放会把要修的场景信号削弱一半。

**已否决的思路：FTS 简称回退。** 部分实体只挂全名不挂简称
（`蒂姆·库克` 有、`库克` 没有）。验证过用 `mentions_fts` 做简称回退
（`LIKE '%·库克'` 是 1684 万行全表 SCAN，不可用；FTS 的 `unicode61` 分词器
会按 `·` 切开，简称能命中全名，4~23ms），5 个 case 里 4 个能 rank=1。
但两点硬阻塞使其不可采用：

1. **触发条件不成立**：用"原本零人物候选"作条件，「苹果」「长城」「欧洲」
   「水星」会全部误触发——它们本就该链到非人物实体，兜底只会注入噪声。
2. **`popularity` 口径不对**：它是 KG 内部入度而非真实世界知名度。
   Tim Cook 入度=1，探险家詹姆斯·库克=15，**任何排序算法都解不了**。

正解是离线引入外部知名度信号（如中文维基 pageview）重算 popularity。

### 5.6 增量索引 worker（L3“越用越强”）
- 主流程调 `retriever.archive(...)` 立即返回（`queue.put_nowait`）；
- 后台 daemon 线程按 batch/interval 刷盘；队列满**丢弃并告警**，不阻塞用户。
- 归档前会经 `cache_policy` 过滤：拒答类答案**不写 L3**，否则会形成
  「拒答 → 入库 → 下次召回到自己的拒答 → 置信度虚高 → 不触发 L4 → 再次拒答」
  的自我强化失败循环。已污染的历史可用 `scripts/clean_l3_refusals.py` 清理。

### 5.7 路由用原始 query、检索用改写后 query
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
