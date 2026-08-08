# Alpha Agentic Search 

Multi-Agent，分层记忆 Agentic搜索系统

## 说在前面的话:

自agent被提出以来，学术界有太多的search agent框架：从最开始的search r1，到后面各种用于刷榜的web search。各位同行大佬们的奇思妙想都给我们无尽的思考；但对工业界、AI搜初学者、以及想搭建个人搜索助手极客们，**层次化、稳定可控、耗时短**，永远是最基础的需求。

业界那么多AI搜索应用，真正火出圈的也就perplexity、秘塔等；一个很重要的点：如何让你的搜索越用越好用，搭建一套**完整多级检索、会自动更新、支持不同问题分层处理，快速响应**的harness search agent框架，可能远比单个模型优化、或者高耗时无法容忍的multi-agent自进化更重要。先比后者的纯粹预研，前者更考虑真实落地遇到的long horizon & 边界纠偏的问题，某种程度上，这个可能更是未来研究要关注的地方。

我们这里开源 Alpha Agentic Search (AAS) 框架，它主打的就是“稳定、可控、低耗时”！并对前面提到的问题进行了初步探索；该框架的思想内核在快手AI搜上面也有相应的落地。无论你是一名初学者，一位刚入行的AI工程师，还是一位资深AI搜索架构师，相信这套系统对您都有所启发。

- 作者个人主页：https://benchen4395.github.io/
- 如有问题或建议，欢迎沟通：benchen4395@gmail.com

---
## 简介:

> 一个可运行、可教学、也可作生产原型的 **Agentic Search** 项目：
> 用 `Route → Rewrite → Retrieve → Verify → Summary` 的经典链路做联网问答，
> 并在检索侧接入了一套 **5 层记忆 RAG 栈（L1–L5）**，实现 Perplexity 式的 **“越用越强”**。
>
> 两大设计原则：
> 1. **多 Provider / 多 Stage**：每个阶段（路由 / 改写 / 回答）都能独立选模型、改 prompt，业务代码零硬编码。
> 2. **统一向量空间**：L1–L5 全部使用 FlagEmbedding **BGE-M3** 编码，跨层可比、可融合。

> 二期优化点：（未来一个月内）
> 1. **多跳搜索的精准实现**：当前依然倾向于单轮检索 （+模糊搜索实现的多跳问答）；后续会增加 Controllableloop agent；
> 2. **多模态 / 富媒体搜索**：未来考虑”返回图片、表格、代码块”多模态搜索；并支持答案的图片来源和Markdown 表格渲染。
---

## 迭代记录:
- 2026.08.03 -- 更新（可靠性 + 来源归因 + 延迟治理）
- 2026.08.04 -- Stage-1（证据可答性 + 拒答污染 + 超时软放弃）
- 2026.08.06 -- 并列实体配额融合（修复多实体检索时部分检索缺失）+ 时效性校验
---

## 1. 快速开始

```bash
# 1) 本地 LLM（router / rewriter 阶段默认走 ollama）
ollama pull qwen3:4b-instruct-2507-q8_0
ollama serve

# 2) 依赖（建议在 conda 环境 search-agent 中安装）
pip install -r requirements.txt

# 3) summary 阶段默认走 DeepSeek（OpenAI 兼容协议）
export DEEPSEEK_API_KEY="sk-xxx"

# 4) 启动（CLI）；启动会打印当前各 stage 使用的模型
python main.py

# 或启动 Web 图形界面（Gradio）
python main_web.py --port 7860
```

> 说明：RAG 的 L2/L5 需要离线索引才会真正召回；**未构建索引时这两层自然返回空，
> 系统仍可正常用 L1/L3/L4 工作**。索引构建见 [`rag/README.md`](rag/README.md) §3.3。

CLI 内置命令：`config`（查看模型配置）、`clear`（清空记忆）、`:stream on|off`（切换流式）、`exit`。

> **Claude Code 风格的执行透明化**：CLI 与 Web 都会把 Agent 的每个流水线步骤
> （路由 / 改写 / 检索 / 回答）实时展示出来并标注 ⏱️ 耗时。
> - **CLI**：每步以 `🔀 工具路由 → ... ⏱️ 120ms` 的形式打印为一行 trace。
> - **Web**：每步渲染为一个**可折叠的步骤块**，最终回答在其下方流式输出；
>   界面为**宽扁单屏布局**（左侧聊天、右侧配置栏），无需下滑即可看到输入框与全部控件。
>
> 详见 §2.1「执行事件机制」。

---

## 2. 系统架构

```
                         ┌────────────────────────────────┐
   用户输入  ──────────►  │ main.py (CLI) / main_web.py(Web)│
                         └───────────────┬────────────────┘
                                         ▼
                         ┌────────────────────────────────┐
                         │  agent.py  AgenticSearchAgent  │  主控编排
                         └───────────────┬────────────────┘
                                         │
      ┌──────────────┬───────────────────┼───────────────────┬──────────────┐
      ▼              ▼                   ▼                   ▼              ▼
    Step0 L1       Step1 路由           Step2 改写          Step2 检索      Step3 回答
    qa_cache.py    tool_router.py       query_rewriter.py   rag/           llm_client.py
    (精确+模糊)     └► tools/            └► context_provider  LayeredRetriever  └► configs/
                    time/weather        (时间/位置)         (L1–L5)            (models_config / prompts)
                    github/arxiv
```

**一轮 `chat()` 的执行链**（见 `agent.py`）：

```
0) L1 QACache 短路      —— 精确/模糊命中直接返回，毫秒级
1) 工具路由 (router)     —— 命中专用工具（时间/天气/GitHub/arXiv）则短路
2) Query 改写 (rewriter) —— 注入时间/位置上下文，规则/LLM/混合三档
   └► 分层 RAG 检索       —— L2 Wiki ∥ L3 History ∥ L5 KG，必要时补 L4 Web
3) 生成回答 (summary)    —— 拼接外部资料 → LLM 生成 → 异步归档到 L3
```

### 2.1 执行事件机制（Claude Code 风格透明化）

`agent.chat()` 支持一个可选回调参数 `on_event`，在每个流水线步骤发射一个结构化事件：

```python
{
  "type": "step",
  # 步骤类型。本轮新增 tool_failed（工具降级）/ sources（来源归因）/ archive（归档）
  "stage": "cache|router|tool|tool_failed|rewrite|retrieve|answer|sources|archive",
  "title": "分层 RAG 检索",                                # 人类可读标题
  "detail": "[L2:3, L3:1, L5:2], 融合 5 段, 置信度 0.98",   # 步骤明细
  "elapsed_ms": 128,                                       # 该步骤**自身**的耗时（毫秒）
}
```

> ⚠️ **`elapsed_ms` 语义已修正**：改造前是「距上一个事件」的差值，而 `answer`
> 事件在调 LLM **之前**发射，导致 LLM 的生成耗时被算进下一个事件（`sources`）——
> 于是终端出现「💬 生成回答 0ms / 🔖 来源归因 6.1s」这种自相矛盾的显示。
> 现在每个事件携带的都是**它自己那一步**的耗时；归档（同步跑一次 BGE-M3 编码）
> 也独立成 `archive` 步骤，不再混进归因。

- **向后兼容**：不传 `on_event` 时行为与原来完全一致（纯 `verbose` 打印）。
- **一套事件，两种渲染**（这正是 Claude Code 的内部设计思路）：
  - `main.py` 的 `_cli_event_printer` 把事件渲染成终端 trace 行；
  - Web 端 `main_web.py` 的 `bot_reply` 把事件渲染成 Gradio 可折叠步骤块（`metadata.title` 携带图标 + ⏱️ 耗时）。
- **线程模型（Web）**：后台线程跑 `agent.chat`，事件与回答 token 统一经 `queue.Queue`
  流回前端，主线程消费并逐帧 `yield` 更新界面，实现步骤块与流式回答的实时渲染。

```python
# 自定义一个事件消费者（例如接入你自己的前端 / 日志）
def my_sink(ev): print(ev["stage"], ev["title"], ev["elapsed_ms"])
agent.chat("量子计算是什么", on_event=my_sink)
```

---

## 3. 三个配置入口（改这里就够了，均位于 `configs/`）

| 文件 | 作用 | 你想改什么时改这里 |
|---|---|---|
| `configs/models_config.py` | 每个 stage 用什么 provider / model / 采样参数 | 换模型、换 provider、改温度 |
| `configs/prompts.py` | 所有 prompt 模板集中注册 | 改提示词、A/B 试 prompt |
| `configs/config.py` | 非模型类配置（缓存 / 代理 / 搜索 / QA 缓存后端） | 改缓存策略、代理、检索条数 |

业务代码（`tool_router.py` / `query_rewriter.py` / `agent.py`）**不硬编码**模型名或 prompt，全部通过上面三个入口。

> 引用方式：`from configs import config`、`from configs.models_config import STAGES`、`from configs.prompts import PROMPTS, render`。

### Stage 配置示例（`configs/models_config.py`）
```python
"summary": {
    "provider": "openai",                 # ollama（本地） / openai（含所有 OpenAI 兼容 API）
    "model":    "deepseek-v4-flash",
    "base_url": "https://api.deepseek.com",
    "api_key_env": "DEEPSEEK_API_KEY",
    "temperature": 0.7,
    "extra": {},                          # 透传给 provider，如 {"think": False}
}
```
把 summary 换成 `gpt-4o-mini`？只改 `STAGES["summary"]`（位于 `configs/models_config.py`）并 `export OPENAI_API_KEY=...`，无需动任何业务文件。

---

## 4. 分层记忆 RAG 栈（rag/）

普通检索不再走裸 `web_search`，而是走完整的 **L1 → L5 分层链路**，由 Router 决定激活哪些层、
并行召回后用 **RRF 融合**（默认零依赖），可选 BGE / cascade 二阶精排。

| 层 | 名称 | 存储 / 来源 | 更新方式 | 用途 |
|----|------|-------------|----------|------|
| **L1** | QA Cache | diskcache / redis（`qa_cache.py`） | 命中即写 | 高频问答，毫秒级 |
| **L2** | Commonsense | FAISS + **Wikipedia dump** | 离线批处理 | 教科书常识 |
| **L3** | History | FAISS（增量） | **每次成功回答异步写** | 用户偏好 / 复用推理 |
| **L4** | Web | 实时联网（`searcher.py`） | 实时 | 时事、时间敏感 |
| **L5** | Knowledge Graph | SQLite + **Wikidata truthy** | 离线批处理 | 结构化事实，支持多跳 |

> **本轮关键变更**：各层原始分不再直接比大小。L2 是 BGE 余弦、L4 是位次衰减、
> L5 是人工混合分 —— 拿单一阈值裁决四种量纲在统计上没有意义。现在统一经
> `rag/calibration.py` 映射到 **P(relevant)**，再用**噪声-OR** 聚合 top-3
> 得到整体置信度，供 L4 兜底（`< 0.55`）与 abstention（`< 0.30`）判定。
> 原始分仍保留在 `Passage.score`（层内排序 / debug），校准值写入
> `metadata["calibrated"]`，二者并存、各司其职。

### 4.1 并列实体配额（winner-takes-most 修复）

RRF 只有「全局相关度」一个维度，当用户**同时问几个对象**时，资料更丰富的那个
实体会独占 `FUSION_TOP_K`（默认 6）个席位，其余实体被饿死。实测故障：

```
提问：国庆期间，俄罗斯、希腊、巴厘岛的气候和景色分别如何？
融合后 6 段的实体分布  {'俄罗斯': 6, '希腊': 0, '巴厘岛': 0}
模型回答「希腊完全没有相关资料」
```

希腊的证据**其实检索到了**，是在融合阶段被挤掉的。

> **为什么不是「检索次数不够」**：实测只把 query 拆成 3 个子 query 并发搜、
> 不改融合，饥饿只是**换了个实体**（俄罗斯 0 / 希腊 1 / 巴厘岛 3）。
> 6 个席位分给 3 个对象，没有配额约束照样有人归零 —— 所以配额是**前置**修复，
> 多 query 并发检索是它之上的可选增强。

`rag/fusion.py` 的 `quota_fuse()` 在 RRF **结果之上**做一次配额重排：

```
① 先按原样跑 RRF，但取一个放大的候选池（弱实体的证据本就排在 top_k 之外）
② 每个实体保底 FUSION_MIN_PER_ENTITY 段（默认 2；席位不够时自动降到 ≥1）
③ 剩余席位仍按全局相关度回填 —— 强实体依然拿较多份额，只是不能把别人饿死
④ 最后按 RRF 分数重排：配额决定"谁进来"，分数决定"排第几"
```

修复后 `{'俄罗斯': 2, '希腊': 2, '巴厘岛': 1}`，希腊有了实质内容。

**对现有链路的影响：零。** 实体识别由 `rag/entities.py` 提供，要求**同时**
满足「有并列连接词」+「≥2 个多字专名」，因此单一意图 query 返回 `[]`，
`quota_fuse` 内部直接转调 `rrf_fuse` ——

- 实测 900 组随机多层输入（含 `score` / `metadata.calibrated` / `layers` 全字段）
  **逐段完全一致**，不是"结果差不多"；
- 额外开销 0.024ms（实体识别）+ 0.007ms（配额重排）；
- 实体识别刻意偏保守：实测多实体召回 83%、单实体**误报 0%**。
  错误代价不对称 —— 漏检只是退化成今天的行为，误报却会给不存在的实体预留
  席位、挤掉真正相关的证据。


**默认已接入 Agent**，无需额外配置：

```python
from agent import AgenticSearchAgent
agent = AgenticSearchAgent()      # 内部自动挂载 LayeredRetriever
agent.warmup()                    # ★ 建议：启动时预热一次（RAG 先、LLM 后）
agent.chat("量子计算是什么")        # 走 L1→L5，成功后自动归档到 L3
agent.close()                     # 退出前 flush 归档队列
```

**多租户隔离**（P0-3）与**结构化返回**（P0.5）：

```python
# session_id → 会话记忆隔离；user_id → L1/L3 namespace 隔离（优先级更高）
r = agent.chat("我的项目代号是什么", session_id="s1", user_id="42",
               return_result=True)          # 默认 False 时仍返回 str，既有调用方零改动

print(r.text)                     # 答案正文（str(r) 等价，向后兼容）
print(r.confidence)               # 校准后的整体证据置信度
print(r.low_evidence)             # abstention 信号：证据不足，答案可能不完整
for s in r.cited_sources:         # 只列**被答案真正引用**的来源
    print(s.id, s.title, s.layer_label, s.confidence, s.url)
print(r.invalid_citation_count)   # 模型编造的 [n] 数量 —— 引用幻觉的直接证据
print(r.citation_coverage)        # 引用覆盖率 = 检索精度的在线代理指标
```

> `return_result=True` 时，流式返回的是 `StreamingAnswer`：迭代行为等价于生成器，
> **耗尽后** `.result` 才被填成完整 `AnswerResult`（引用只能在完整答案就绪后解析）。
> 之所以需要这个包装类：CPython 的 generator 是 C 层实现、没有 `__dict__`，挂不上属性。

切换策略 / 关闭：
```python
AgenticSearchAgent(enable_rag=False)                # 回退到裸 web_search
AgenticSearchAgent(rag_strategy="offline_only")     # 完全离线（禁用 L4）
AgenticSearchAgent(rag_strategy="web_only")         # 只走 L1 + L4
```

统一编码（BGE-M3）与离线索引构建等细节，见 **[`rag/README.md`](rag/README.md)**。

---

## 5. 目录结构

```
agentic_search/
├── main.py               入口①：CLI 终端交互（python main.py）
├── main_web.py           入口②：Gradio Web 图形界面（python main_web.py）
├── agent.py              ★ 主控 AgenticSearchAgent（编排 0→3 全链路）
│
│   ── 配置层（configs/ 包）──
├── configs/              ★ 三个配置集中收纳于此包
│   ├── __init__.py           聚合导出 STAGES / PROMPTS / render / config
│   ├── models_config.py      各 stage 的 provider / model / 参数
│   ├── prompts.py            所有 prompt 模板集中注册
│   └── config.py             非模型类配置（缓存 / 代理 / 搜索 / QA 后端）
│
│   ── LLM 调用层 ──
├── llm_client.py         ★ 统一 LLM 调用（ollama / OpenAI 兼容，流式+非流式）
│
│   ── 业务 stage ──
├── tool_router.py        stage=router：是否调工具、调哪个
├── query_rewriter.py     stage=rewriter：query 改写（规则/LLM/混合）
├── context_provider.py   环境信息注入（当前时间 / 位置）
├── searcher.py           联网检索（DDG → Tavily → Serper → Bing 兜底）
├── memory.py             会话记忆（滑动窗口，按 session 分桶）
├── qa_cache.py           Q&A 缓存（= RAG L1；精确 + BGE-M3 模糊匹配 + 槽位门禁）
│
│   ── 本轮新增：可靠性 / 安全 / 结构化返回 ──
├── cache_policy.py       ★ P0-1 L1 准入策略（时效判定 / 分级 TTL / 槽位一致性门禁）
├── evidence.py           ★ P0-4 证据清洗 + <doc> 结构化定界（Prompt Injection 防护）
├── answer_types.py       ★ P0.5 AnswerResult / Source / Citation（来源归因契约）
├── conftest.py           pytest 夹具：把缓存目录重定向到 tmp，杜绝测试污染生产数据
├── test_p0.py            P0/P0.5 + 延迟观测 + Stage-1 + 配额融合回归（170 项）
├── test_qa_cache.py      L1 缓存回归（24 项）
│
│   ── 数据层（data/）──
├── data/                 ★ 所有本地落盘数据统一收纳于此
│   ├── rag_data/             RAG 知识库：L2 Wiki 索引 / L5 KG / L3 历史归档（已 gitignore，13GB）
│   ├── qa_cache/             L1 Q&A 缓存（diskcache）★ 被 git 追踪，仓库自带一批预热问答
│   │   ├── cache.db              问答正文
│   │   ├── _embeddings/          BGE-M3 向量（fuzzy 命中用，1024 维）
│   │   └── _meta/                原始 query 原文（槽位门禁用）
│   └── search_cache/         联网搜索结果缓存（diskcache）★ 被 git 追踪
│
│   注：cache.db-wal / cache.db-shm 是 SQLite 运行时文件，已 gitignore。
│       提交前需 `PRAGMA wal_checkpoint(TRUNCATE)` 把 WAL 落进主库，
│       否则只提交 cache.db 会丢掉最新数据（详见 .gitignore 注释）。
│
│   ── 工具与脚本 ──
├── tools/                专用工具（current_time / weather / github_repo / arxiv / web_search）
├── scripts/search.py     供外部 Skill 调用的命令行检索入口
├── scripts/clean_l3_refusals.py  ★ 清理 L3 里的拒答污染（Stage-1 配套）
├── SKILL.md              Skill 触发说明
│
│   ── 分层 RAG 栈 ──
└── rag/                  ★ 5 层记忆 + Router + RRF 融合 + 增量索引
    ├── README.md              分层设计与索引构建文档
    ├── retriever.py           对外主入口 LayeredRetriever
    ├── layers.py              L1–L5 五层实现
    ├── router.py / fusion.py  层激活策略 / RRF 融合 + ★ 并列实体配额 + 可选 rerank
    ├── entities.py            ★ 并列实体识别（jieba 词性 + 连接词；配额融合与未来多 query 拆解共用）
    ├── calibration.py         ★ P0-2 跨层分数校准（Platt scaling + 噪声-OR 聚合）
    ├── answerability.py       ★ Stage-1 证据可答性判据（实词覆盖率，与置信度正交）
    ├── embedder.py            统一 BGE-M3 编码适配器
    ├── vector_store.py        faiss + numpy 可插拔向量存储（L3）
    ├── incremental_worker.py  L3 后台增量写 worker
    ├── config.py / types.py   编排器配置 / Passage & RetrievalResult 契约
    ├── configs/default.yaml    wiki_rag 全部可调参数（L2/L5 路径等）
    ├── wiki_rag/              vendored 检索内核（WikiRetriever / KGRetriever）
    └── scripts/               离线构建流水线（01–12：wiki 索引 + Wikidata KG）
```

---

## 6. 常见操作

**A. 换 summary 模型为 OpenAI**
```python
# configs/models_config.py → STAGES["summary"]
"provider": "openai", "model": "gpt-4o-mini",
"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY",
```
`export OPENAI_API_KEY=sk-...` 后直接运行，无需改业务代码。

**B. 改某个阶段的 prompt**
```python
# configs/prompts.py
PROMPTS["rewriter"] = """你的新模板...
{context}
[对话历史] {history}
[用户提问] {query}
[输出]"""
```

**C. 单次调用临时覆盖模型**
```python
from llm_client import complete
complete("rewriter", "改写：xxx",
         provider="openai", model="deepseek-v4-flash",
         base_url="https://api.deepseek.com", api_key_env="DEEPSEEK_API_KEY")
```

**D. 命令行检索（供 Skill 调用）**
```bash
python scripts/search.py "2025 年 RAG 最新进展"          # 输出 rewritten/results/answer 的 JSON
python scripts/search.py "xxx" --rewrite-type 0 --no-answer
```

---

## 7. 扩展点

- **新增 stage**：`configs/models_config.py` 的 `STAGES` 加 key + `configs/prompts.py` 的 `PROMPTS` 注册同名模板。
- **新增 provider**：`llm_client.py` 加一个 `_call_xxx` / `_stream_xxx`，在 `chat()`/`stream_chat()` 里分支。
- **新增工具**：`tools/` 下新建模块，在 `tools/__init__.py` 的 `TOOLS` 里登记。
- **新增 RAG 层 / 换 reranker**：见 `rag/README.md`（RRF / BGE / cascade 通过环境变量切换）。

---

## 8. 测试

```bash
# 全量回归（194 项）
python -m pytest -q test_p0.py test_qa_cache.py

python -m pytest -q test_qa_cache.py     # L1 QA 缓存（精确/模糊/多级/异步）24 项
python -m pytest -q test_p0.py           # P0/P0.5 + 延迟 + Stage-1 + 配额  170 项

# 按主题跑（排查时更快）
python -m pytest -q test_p0.py -k "SlotGate or FocusSlot"        # L1 误命中
python -m pytest -q test_p0.py -k "Calibration"                  # 跨层校准 / L4 兜底
python -m pytest -q test_p0.py -k "LatencyObservability"         # 延迟与耗时归属
python -m pytest -q test_p0.py -k "Answerability"                # 证据可答性（Stage-1）
python -m pytest -q test_p0.py -k "PartialRefusal"               # 部分拒答（Stage-1）
python -m pytest -q test_p0.py -k "TimeoutSoftAbandon"           # 超时软放弃（Stage-1）
python -m pytest -q test_p0.py -k "EntityQuotaFusion"            # 并列实体配额融合
```

> `conftest.py` 的 `autouse` 夹具会把 `QA_CACHE_DIR` 重定向到每个测试独有的
> tmp 目录 —— 这道隔离很重要：在它加入之前，测试里的假编码器（3/4/8 维）
> 会把脏向量写进**仓库里被追踪的** `data/qa_cache/`，既污染生产数据，
> 又造成「单独跑通过、连着跑失败」的顺序依赖，极难排查。

## 9. License

MIT — 学习与交流、及商业用途 均可以。
