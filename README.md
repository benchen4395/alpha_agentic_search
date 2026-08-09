# Alpha Agentic Search (AAS)

> **稳定、可控、低耗时**的 Multi-Agent 分层记忆 Agentic 搜索框架。

> 用 `Route → Rewrite → Retrieve → Verify → Summary` 的经典链路做联网问答，

> 并在检索侧接入一套 **5 层记忆 RAG 栈（L1–L5）**，实现 Perplexity 式的 **"越用越强"**。


---

## 缘起

自 Agent 概念被提出以来，学术界涌现了大量 search agent 框架：从最早的 Search-R1，到后来各种用于刷榜的 web search。这些工作给了我们无尽的启发；但对工业界、AI 搜索初学者、以及想搭建个人搜索助手的极客们来说，**层次化、稳定可控、耗时短**永远是最基础的需求。

业界那么多 AI 搜索应用，真正火出圈的也就 Perplexity、秘塔等。一个很重要的原因是：**如何让你的搜索越用越好用**。搭建一套完整多级检索、会自动更新、支持不同问题分层处理、快速响应的 harness search agent 框架，可能远比单个模型优化、或高耗时无法容忍的 multi-agent 自进化更重要。

相比后者的纯粹预研，前者更多考虑真实落地会遇到的 long-horizon 与边界纠偏问题 —— 某种程度上，这可能是未来研究更值得关注的方向。

Alpha Agentic Search 对上述问题做了初步探索，其思想内核在快手 AI 搜上也有相应落地。更进一步，作为一个逐步迭代的项目，我们把每次优化的缺陷&改进也都进行了详细的记录，这对于了解Agentic Search的设计理念有着很大帮助。无论您是初学者、刚入行的 AI 工程师，还是资深 AI 搜索架构师，希望这套系统对您都有所启发。

- 作者主页：https://benchen4395.github.io
- 交流与建议：benchen4395@gmail.com

---

## 目录

- [核心特性](#核心特性)
- [1. 快速开始](#1-快速开始)
- [2. 系统架构](#2-系统架构)
- [3. 配置入口](#3-配置入口)
- [4. 分层记忆 RAG 栈](#4-分层记忆-rag-栈)
- [5. 可靠性与安全设计](#5-可靠性与安全设计)
- [6. 目录结构](#6-目录结构)
- [7. 常见操作](#7-常见操作)
- [8. 扩展点](#8-扩展点)
- [9. 测试](#9-测试)
- [10. 路线图](#10-路线图)
- [11. License](#11-license)

---

## 核心特性

| 特性 | 说明 |
|---|---|
| **5 层记忆检索** | L1 缓存 / L2 Wikipedia / L3 历史 / L4 实时联网 / L5 知识图谱，Router 决定激活哪些层，并行召回后 RRF 融合 |
| **统一向量空间** | L1–L5 全部使用 FlagEmbedding **BGE-M3** 编码，跨层可比、可融合 |
| **多 Provider / 多 Stage** | 每个阶段（路由 / 改写 / 回答）独立选模型、改 prompt，业务代码零硬编码 |
| **跨层分数校准** | 各层原始分（余弦 / 位次 / 混合分）统一映射到 `P(relevant)`，再用噪声-OR 聚合出整体置信度 |
| **延迟治理** | 分层延迟预算 + 超时软放弃（主预算 + 短宽限）+ 启动预热，前台延迟有上界 |
| **来源归因** | 每条 `[n]` 引用都被系统解析、校验、映射到 URL；能检测"引用幻觉" |
| **Prompt Injection 防护** | 三层防护：内容清洗 → `<doc>` 结构化定界 → system prompt 守卫声明 |
| **执行透明化** | Claude Code 风格：每个流水线步骤实时展示并标注耗时（CLI trace / Web 可折叠步骤块） |
| **越用越强** | 每次成功回答异步归档到 L1/L3，热点问题二次命中即毫秒返回 |

---

## 1. 快速开始

### 1.1 安装

```bash
# 1) 本地 LLM（router / rewriter 阶段默认走 ollama）
ollama pull qwen3:4b-instruct-2507-q8_0
ollama serve

# 2) Python 依赖（建议使用独立 conda 环境）
conda create -n search-agent python=3.11 && conda activate search-agent
pip install -r requirements.txt

# 3) summary 阶段默认走 DeepSeek（OpenAI 兼容协议）
export DEEPSEEK_API_KEY="sk-xxx"
```

### 1.2 运行

```bash
python main.py                       # CLI 终端交互
python main_web.py --port 7860       # Gradio Web 图形界面
```

启动时会打印当前各 stage 使用的模型。

> **关于离线索引**：RAG 的 L2（Wikipedia）/ L5（知识图谱）需要离线构建才会真正召回；
> **未构建索引时这两层自然返回空，系统仍可正常用 L1/L3/L4 工作**。
> 索引构建见 [`rag/README.md`](rag/README.md) §3.3。

### 1.3 CLI 命令

| 命令 | 作用 |
|---|---|
| `config` | 查看当前各阶段模型 / 模式 |
| `clear` | 清空会话记忆 |
| `:stream on\|off` | 切换流式 / 非流式输出 |
| `:save-interrupt on\|off` | 流式中途打断时，已收到部分是否入库 |
| `exit` | 退出 |

### 1.4 作为库调用

```python
from agent import AgenticSearchAgent

agent = AgenticSearchAgent()
agent.warmup()                       # 建议：启动时预热一次（RAG 先、LLM 后）

answer = agent.chat("量子计算是什么")   # 走 L1→L5，成功后自动归档到 L3
print(answer)

agent.close()                        # 退出前 flush 归档队列
```

---

## 2. 系统架构

```
                         ┌─────────────────────────────────┐
   用户输入  ──────────►  │ main.py (CLI) / main_web.py(Web) │
                         └───────────────┬─────────────────┘
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
                    time/weather        (时间/位置)         (L1–L5)        (models_config / prompts)
                    github/arxiv
```

**一轮 `chat()` 的执行链**（见 `agent.py`）：

```
0) L1 QACache 短路      —— 精确/模糊命中直接返回，毫秒级
1) 工具路由 (router)     —— 命中专用工具（时间/天气/GitHub/arXiv）则短路
                          工具失败则降级到检索通路
2) Query 改写 (rewriter) —— 注入时间/位置上下文，规则/LLM/混合三档
   └► 分层 RAG 检索       —— L2 Wiki ∥ L3 History ∥ L5 KG，必要时补 L4 Web
3) 生成回答 (summary)    —— 拼接外部资料 → LLM 生成 → 来源归因 → 异步归档到 L3
```

### 2.1 执行事件机制

`agent.chat()` 支持可选回调 `on_event`，在每个流水线步骤发射一个结构化事件：

```python
{
  "type": "step",
  "stage": "cache|router|tool|tool_failed|rewrite|retrieve|answer|sources|followup|archive",
  "title": "分层 RAG 检索",                                # 人类可读标题
  "detail": "[L2:3, L3:1, L5:2], 融合 5 段, 置信度 0.98",   # 步骤明细
  "elapsed_ms": 128,                                       # 该步骤**自身**的耗时
}
```

设计要点：

- **`elapsed_ms` 是该步骤自身的耗时**，不是"距上一个事件"的差值。流式路径的
  `answer` 事件在首个 token 到达时发射，携带 **TTFT**（Time To First Token）——
  这才是"用户等了多久才看到东西"的正确度量。归档（要同步跑一次 BGE-M3 编码）
  独立成 `archive` 步骤，不会混进归因耗时。
- **向后兼容**：不传 `on_event` 时行为与不带该参数完全一致（纯 `verbose` 打印）。
- **一套事件，两种渲染**：`main.py` 的 `_cli_event_printer` 渲染成终端 trace 行；
  `main_web.py` 的 `bot_reply` 渲染成 Gradio 可折叠步骤块。
- **线程模型（Web）**：后台线程跑 `agent.chat`，事件与回答 token 统一经
  `queue.Queue` 流回前端，主线程消费并逐帧 `yield` 更新界面。

```python
# 自定义事件消费者（接入你自己的前端 / 日志 / 监控）
def my_sink(ev): print(ev["stage"], ev["title"], ev["elapsed_ms"])
agent.chat("量子计算是什么", on_event=my_sink)
```

---

## 3. 配置入口

三个文件，改这里就够了，均位于 `configs/`：

| 文件 | 作用 | 你想改什么时改这里 |
|---|---|---|
| `configs/models_config.py` | 每个 stage 用什么 provider / model / 采样参数 | 换模型、换 provider、改温度 |
| `configs/prompts.py` | 所有 prompt 模板集中注册 | 改提示词、A/B 试 prompt |
| `configs/config.py` | 非模型类配置（缓存 / 代理 / 搜索 / QA 缓存后端） | 改缓存策略、代理、检索条数 |

业务代码（`agent.py` / `tool_router.py` / `query_rewriter.py`）**不硬编码**模型名或 prompt，全部通过上面三个入口。

```python
from configs import config
from configs.models_config import STAGES
from configs.prompts import PROMPTS, render
```

### Stage 配置示例

```python
# configs/models_config.py
"summary": {
    "provider": "openai",                 # ollama（本地） / openai（含所有 OpenAI 兼容 API）
    "model":    "deepseek-v4-flash",
    "base_url": "https://api.deepseek.com",
    "api_key_env": "DEEPSEEK_API_KEY",
    "temperature": 0.7,
    "extra": {},                          # 透传给 provider，如 {"think": False}
}
```

把 summary 换成 `gpt-4o-mini`？只改 `STAGES["summary"]` 并 `export OPENAI_API_KEY=...`，无需动任何业务文件。

---

## 4. 分层记忆 RAG 栈

普通检索不再走裸 `web_search`，而是走完整的 **L1 → L5 分层链路**：由 Router 决定激活哪些层，并行召回后用 **RRF 融合**（默认零依赖），可选 BGE / cascade 二阶精排。

| 层 | 名称 | 存储 / 来源 | 更新方式 | 用途 |
|----|------|-------------|----------|------|
| **L1** | QA Cache | diskcache / redis（`qa_cache.py`） | 命中即写 | 高频问答，毫秒级 |
| **L2** | Commonsense | FAISS + **Wikipedia dump** | 离线批处理 | 教科书常识 |
| **L3** | History | FAISS（增量） | **每次成功回答异步写** | 用户偏好 / 复用推理 |
| **L4** | Web | 实时联网（`searcher.py`） | 实时 | 时事、时间敏感 |
| **L5** | Knowledge Graph | SQLite + **Wikidata truthy** | 离线批处理 | 结构化事实，支持多跳 |

统一编码（BGE-M3）、离线索引构建、各层参数速查等细节，见 **[`rag/README.md`](rag/README.md)**。

### 4.1 跨层分数校准

各层原始分**不能直接比大小**：L2 是 BGE 余弦、L4 是位次衰减、L5 是人工混合分 —— 拿单一阈值裁决三种量纲在统计上没有意义。

`rag/calibration.py` 用**分层 Platt scaling** 把它们统一映射到 `P(relevant)`：

```
p = sigmoid(a_layer * (s - b_layer))
```

再用**噪声-OR** 聚合 top-3 得到整体置信度，供两个决策使用：

- `conf < WEB_FALLBACK_CONFIDENCE`（0.55）→ 补 L4 联网兜底
- `conf < ABSTAIN_CONFIDENCE`（0.30）→ 标记证据不足，引导模型明确说"资料不足"

原始分仍保留在 `Passage.score`（层内排序 / debug），校准值写入 `metadata["calibrated"]`，二者并存、各司其职。默认参数是基于各层分数分布的先验估计，生产环境可用 `fit_platt()` 在自建标注集上重新拟合，通过 `RAG_CALIBRATION_FILE` 热加载。

### 4.2 证据可答性判据

置信度衡量的是**语义相似度**（"召回的东西像不像这个话题"），但 L4 兜底真正要判断的是**充分性**（"够不够回答问题"）。两者在简单事实题上一致，在多跳 / 聚合题上完全背离：

```
「茅盾文学奖 历届 获奖名单」→ conf 0.93，但证据只有"1982 年首届"
```

`rag/answerability.py` 加了一个与相似度**正交**的信号：**query 实词的单篇最佳覆盖率**。若"获奖名单""次数"在所有证据里一次都没出现，那无论语义多相似，答案都不可能在里面。

两个信号取 **OR**（而非 AND），因为它们各自捕捉一类失败模式：

| conf | 覆盖率 | 失败模式 |
|---|---|---|
| 低 | 高 | 沾了词但语义弱（同名异义） |
| 高 | 低 | 主题相关但不含答案 |

### 4.3 并列实体配额

RRF 只有"全局相关度"一个维度。当用户**同时问几个对象**时，资料更丰富的那个实体会独占 `FUSION_TOP_K`（默认 6）个席位，其余实体被饿死：

```
提问：国庆期间，俄罗斯、希腊、巴厘岛的气候和景色分别如何？
融合后 6 段的实体分布  {'俄罗斯': 4, '希腊': 1, '巴厘岛': 1}
模型回答「希腊完全没有相关资料」
```

希腊的证据**其实检索到了**，是在融合阶段被挤掉的。

> **为什么不是"检索次数不够"**：实测只把 query 拆成 3 个子 query 并发搜、不改融合，
> 饥饿只是**换了个实体**（俄罗斯 0 / 希腊 1 / 巴厘岛 3）。6 个席位分给 3 个对象，
> 没有配额约束照样有人归零 —— 所以配额是**前置**修复，多 query 并发检索是它之上的可选增强。

`rag/fusion.py` 的 `quota_fuse()` 在 RRF **结果之上**做一次配额重排：

```
① 先按原样跑 RRF，但取一个放大的候选池（弱实体的证据本就排在 top_k 之外）
② 每个实体保底 FUSION_MIN_PER_ENTITY 段（默认 2；席位不够时自动降到 ≥1）
③ 剩余席位仍按全局相关度回填 —— 强实体依然拿较多份额，只是不能把别人饿死
④ 最后按 RRF 分数重排：配额决定"谁进来"，分数决定"排第几"
```

修复后分布为 `{'俄罗斯': 2, '希腊': 2, '巴厘岛': 1}`。

**对单一意图 query 零影响**：实体识别（`rag/entities.py`）要求**同时**满足「有并列连接词」+「≥2 个多字专名」，因此单一意图 query 返回 `[]`，`quota_fuse` 内部直接转调 `rrf_fuse`。实测 900 组随机多层输入**逐段完全一致**；额外开销 0.024 ms（实体识别）+ 0.007 ms（配额重排）。实体识别刻意偏保守（多实体召回 83%、单实体误报 0%）—— 漏检只是退化成不配额的行为，误报却会给不存在的实体预留席位、挤掉真正相关的证据。

### 4.4 近重去重 + MMR 多样性

`fusion._passage_key` 只做**精确**去重（URL 全等，或标题+正文前缀全等），挡不住真实世界最常见的重复 —— **同一条新闻被 N 家转载**。4 条转载吃掉 6 个席位里的 4 个，带来三个后果：信息量坍缩、**虚假共识**（模型以为"多个独立信源一致"）、来源面板冗余。

`rag/dedup.py` 拆成两个阶段，**顺序不可颠倒**：

| 阶段 | 函数 | 判据类型 | 阈值 | 作用 |
|---|---|---|---|---|
| A | `drop_near_duplicates()` | **事实判断**（是不是同一份文本） | 余弦 ≥ 0.95 硬删 | 删掉转载 |
| B | `mmr_rerank()` | **偏好判断**（在有差异的候选里偏好互补） | λ = 0.7 软排 | 提升信息覆盖 |

必须先 A 再 B：MMR 的惩罚是**连续**的，对余弦 0.98 的转载只会施加"较大惩罚"，若其相关性也高仍可能入选。硬删必须在前面，否则转载稿会参与多样性计算、污染整个选择过程。

配套地，融合阶段要**多召回**（`FUSION_CANDIDATE_MULTIPLIER=3`）：否则去重删掉 3 条后席位是"凭空消失"而不是"腾给更好的证据"。

**延迟优化**：唯一的真实开销是 BGE-M3 编码（约 80 ms/段），而算法本身只要 1~2 ms。所以默认只对 `DEDUP_LAYERS`（默认 `{"L4_web"}`）做语义去重 —— 转载重复几乎只发生在 web 检索。纯离线命中时**零编码**，L4 触发时只编码 5 段。

### 4.5 结构化返回

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
print(r.followups)                # 追问推荐（"你可能还想问"）
```

> `return_result=True` 且流式时，返回的是 `StreamingAnswer`：迭代行为等价于生成器，
> **耗尽后** `.result` 才被填成完整 `AnswerResult`（引用只能在完整答案就绪后解析）。
> 之所以需要这个包装类：CPython 的 generator 是 C 层实现、没有 `__dict__`，挂不上属性。

### 4.6 切换策略

```python
AgenticSearchAgent(enable_rag=False)                # 回退到裸 web_search
AgenticSearchAgent(rag_strategy="offline_only")     # 完全离线（禁用 L4）
AgenticSearchAgent(rag_strategy="web_only")         # 只走 L1 + L4
```

---

## 5. 可靠性与安全设计

### 5.1 L1 缓存准入策略（`cache_policy.py`）

L1 命中发生在 `chat()` 的 Step 0，**毫秒级直接短路返回**，既不检索也不调 LLM。这意味着一条错误的 L1 条目会被**静默**返回给用户，且没有任何日志能看出问题。因此写入侧与读取侧各有一道防线：

**防线 A｜写入侧准入 + 分级 TTL**

TTL 应与"答案的期望半衰期"匹配，而不是一个全局常量：

| 判定 | 结果 |
|---|---|
| 答案过短（< 10 字） | 拒收 |
| 拒答类（开头 60 字含"抱歉/无法回答/没有找到"） | 拒收 —— 避免固化失败，阻止未来的成功重试 |
| **部分拒答**（结尾承认核心信息缺失） | 拒收 —— 见下方说明 |
| query 强时效（今天 / 最新 / 股价 / 天气） | 拒收 —— 任何 TTL 都是错的 |
| 依赖了 L4 实时 web | 允许，TTL = 6h |
| 含易变槽位（年份 / 价格 / 版本 / 排名） | 允许，TTL = 24h |
| 常识类 | 允许，TTL = 30d |

**"部分拒答"为什么单独判**：开头像个好答案、结尾才承认"没给出完整名单"的类型会绕过只看前 60 字的检查，被判为 `stable` 写进 L1 + L3。写进 L3 的后果最严重 —— 它会形成一个**自我强化的失败循环**：

```
拒答 → 存进 L3 → 下次召回到自己的拒答（实测 calib=0.578）
     → 置信度虚高 → 不触发 L4 → 再次拒答 → 循环加固
```

这与"越用越强"的设计目标完全相反，所以这类答案 L1 不写、**L3 也不写**。
配套清理脚本：`scripts/clean_l3_refusals.py`。

**防线 B｜读取侧槽位一致性门禁**

仅靠调高相似度阈值无法解决语义碰撞 —— BGE-M3 实测：

```
「苹果公司的CEO是谁」  vs 「苹果公司的CFO是谁」   → 0.8781
「美国现任总统是谁」    vs 「美国现任副总统是谁」  → 0.8575
「2024年中国GDP是多少」 vs 「2025年中国GDP是多少」→ 0.8531
```

这些 pair 在任何阈值下都可能过线，而答案完全不同。所以额外加一道**判别式门禁**：抽出双方 query 的关键槽位做比对。向量相似度是"软"的、连续的，槽位比对是"硬"的、离散的，二者互补。

| 槽位 | 比对方式 | 拦截的碰撞 |
|---|---|---|
| 数字 / 年份 / 否定词 / 比较级 / 限定词 | **精确相等** | 2024↔2025、要↔不要、最大↔最小、总统↔副总统 |
| 疑问焦点 | **兼容判定** | 是谁↔在哪 拒绝；但陈述式(∅)↔"有哪些" 允许 |
| 命名实体 | **精确相等** | CEO↔CFO |
| 主题名词 | **Jaccard ≥ 0.5** | 苹果↔微软；容忍分词波动 |

疑问焦点之所以不能用精确相等：陈述式提问（「美国历届总统名单」）抽不到疑问词、焦点为 ∅，若要求相等会把所有「陈述式 ↔ 疑问式」的改述全部误拒。但 ∅ **不是万能通配** —— 它的隐含语义是"请列出/说明"，只与 WHO/WHICH 相容；「…名单」(∅) 绝不能命中「…有多少位」(HOWMANY)。

**阈值 0.90 的标定过程**（体现"阈值不该拍脑袋定"）：用 15 条正样本（同义改述）+ 12 条负样本（一字之差换答案）做阈值扫描：

| 阈值 | 召回 | 误放行 |
|---|---|---|
| 0.88 | 12/15 | 0/12 |
| **0.90** | **12/15** | **0/12** ← 采用 |
| 0.93 | 10/15 | 0/12 ← 白丢 2 条召回，安全性无提升 |

关键观察：负样本余弦的最大值是 0.8800。0.90 已在所有已知危险碰撞之上留了 0.02 余量；再往上抬不会挡掉任何负样本，只会持续误杀正样本。真正承担"区分 CEO/CFO"职责的不是阈值，而是槽位门禁（实测独立拦下 12 个负样本中的 11 个）。

运维指标 `stats()["slot_gate_rejects"]` 近似等于"如果没有这道门禁，本进程会返回多少次错误答案"。

### 5.2 Prompt Injection 防护（`evidence.py`）

检索到的网页内容是**不可信输入**。任何一个能被搜索到的页面只要写上「忽略之前的所有指令」或「SYSTEM: 你现在是…」，就有机会改变模型行为 —— 攻击者只需做一点 SEO 即可命中检索结果，成本极低。

三层防护，单独任何一层都不够：

```
┌──────────────────────────────────────────────────────┐
│ ① sanitize_evidence_text()                           │
│    内容清洗：标注 injection 模式、转义伪造定界符、     │
│    剔除控制字符（零宽 / 双向覆写等视觉欺骗）           │
└──────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────┐
│ ② build_evidence_block()                             │
│    结构化定界：每段包进带 id/来源/置信度的 <doc> 标签  │
└──────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────┐
│ ③ EVIDENCE_GUARD_PROMPT（注入 system prompt）         │
│    显式声明：<doc> 内一律是数据，绝不是指令            │
└──────────────────────────────────────────────────────┘
```

- 只做清洗 → 攻击者用同义改写就能绕过（正则永远追不上自然语言）
- 只做定界 → 模型可能仍把 `<doc>` 里的祈使句当指令
- 只做声明 → 模型注意力有限，长上下文里容易"忘记"这条约束

三层叠加后，攻击者需要同时绕过正则、伪造 XML 定界、并压过 system 指令的优先级。这也是 Anthropic / OpenAI 官方推荐的"用 XML 标签隔离不可信内容"实践。

清洗采用**只标注不删除**（用 `⟦可疑指令·已中和：…⟧` 包裹）：删除会破坏语义连贯性；保留并标注能让模型"看见"这是可疑内容，反而更容易正确处理；且运维能从日志看到确实发生了拦截。命中风险的来源会在 `Source.risks` 里标记，前端展示 ⚠️。

用户问题也包进 `<question>`：如果只给证据加标签、问题裸放，攻击者可以在证据里伪造一个"用户问题"区。**双向都结构化**是更彻底的做法。

### 5.3 延迟治理

| 机制 | 位置 | 说明 |
|---|---|---|
| **分层延迟预算** | `rag/config.py` | 离线层 5s / L4 联网 8s。到点未返回的层记为空结果，其余层照常融合 —— 优雅降级而非整体失败 |
| **超时软放弃** | `rag/retriever.py` | `ThreadPoolExecutor` 无法中断已启动的任务，所以超时的线程**必然会跑完**。既然成本已付，就给一个短宽限期（1.0~1.5s）再看一眼：已完成的收下，仍未完成的才真正放弃 |
| **DDG 快速失败** | `configs/config.py` | 单次超时 15s→8s、重试 3→2 次；搜索引擎响应分布重尾，超过 8s 基本是被限流，换 backend 重试的成功率更高 |
| **模型常驻 + 预热** | `models_config.py` / `llm_client.py` | `keep_alive=-1` 让 ollama 模型不因空闲 5 分钟被卸载；`warmup_all()` 把加载成本提前到启动期 |
| **路由用原始 query** | `rag/retriever.py` | rewriter 可能给历史累计型问题凭空补上当前年份，导致被误判为时效敏感而强制联网。时效性是**用户意图**的属性，不该由改写结果决定 |
| **地理位置预取** | `agent.py` / `context_provider.py` | IP 定位实测 500~1580ms，在 `__init__` 用后台线程预取，并用 single-flight 去重避免并发重复请求 |

**为什么预热顺序必须"RAG 先、LLM 后"**（与直觉相反）：

```
LLM 先、RAG 后：  router 首调 12220 ms   ← 反而退化
RAG 先、LLM 后：  router 首调  1647 ms   ← 快 7 倍
```

两种顺序下模型都仍然驻留，说明慢的原因不是被卸载，而是**资源竞争**：BGE-M3（fp16 约 2.3GB）加载到 MPS 时会大量申请统一内存，同时读 FAISS 索引 + SQLite 造成密集页缓存压力，把已驻留的 ollama 权重页挤出物理内存。把 LLM 放最后，它的权重页在 LRU 回收顺序里最安全。这是内存受限环境下的通用原则：**让延迟最敏感的组件最后预热**。

### 5.4 多租户隔离

```python
agent.chat(q, session_id="s1", user_id="42")
```

- `session_id` → 会话记忆按桶隔离（`_memories` dict，FIFO 淘汰，上限 512）
- `user_id` → L1/L3 的 namespace 隔离（优先级高于 session_id，跨 session 复用个人积累）

namespace 前缀 `u:` / `s:` 避免 `user_id="42"` 与 `session_id="42"` 撞车。不传时行为等价于单例 memory + 全局 namespace。L3 用 metadata 后过滤实现隔离（FAISS `IndexFlatIP` 不支持元数据过滤），并超取 4 倍候选以抵消过滤损耗。

### 5.5 追问推荐（`followup.py`）

答完后给 2~4 条"你可能还想问"。实现上**复用主答案的那次 LLM 调用** —— 指令写进 summary system prompt，**零额外调用、零额外延迟、零额外成本**，而且模型刚写完答案，最清楚"还能往哪问"。

代价是模型输出末尾带一段 `###FOLLOWUP###` 分隔的文本，必须在展示前剥离：

- **非流式**：`parse_followups()` 拿到 `(正文, 追问)`
- **流式**：`StreamFilter` 边过滤边收集。难点是 LLM 的 chunk 边界任意，分隔符很可能被切成 `##` / `#FOLLO` / `WUP##` 几段，逐 chunk 做 `in` 判断必然漏检。用**滞后输出**（hold-back `len(marker)-1` 个字符）解决，代价是视觉上落后 14 个字符（打字机效果下约 0.25s，完全无感）。

> ⚠️ **剥离必须在写 memory / 归档之前**，否则追问区会被存进 L1/L3，
> 下次缓存命中时直接返回给用户，问题被永久固化。

`parse_followups` 有严格的降级保证：分隔符没出现（模型没遵守 / `FOLLOWUP_MODE != "prompt"`）就原样返回全文，绝不会吃掉正文。

### 5.6 Snippet 噪声清洗（`rag/textclean.py`）

L4 的 snippet 并不是干净正文。实测 Tavily 结果中位数 1339 字，但混着大量页面模板噪声：空表格骨架、`TWD 210 起立即預訂` 这类 CTA、`4.7/51358 reviews` 评分块、图片文件名、页脚版权、月份选择器控件等。

三重代价：① 挤占 `evidence.py` 的 8000 字硬预算（噪声占一半 = 证据条数腰斩）；② 干扰模型注意力；③ 污染 `rag/dedup.py` 的语义去重 —— 它只取前 512 字算余弦，若前半是导航栏，算出的是"页面模板像不像"而非"内容像不像"。

纯正则、零联网、零依赖、微秒级，实测去噪 12%（最脏站点 46~54%）。设计原则是**宁可漏删，不可错删**：清洗后不足原文 30% 就放弃清洗、原样返回，所以不可能把证据洗空。

> **为什么在 L4 层洗而不是在 `searcher` 里**：searcher 的结果会进 diskcache（TTL 数小时~数天）。
> 在那里洗等于缓存里存的是洗过的文本 —— 规则一改老缓存不会重洗，新旧策略混在一起，
> 而且再也拿不到原文做对照。这是"**尽量晚地做有损变换**"的一般原则。

---

## 6. 目录结构

```
alpha_agentic_search/
├── main.py               入口①：CLI 终端交互
├── main_web.py           入口②：Gradio Web 图形界面
├── agent.py              ★ 主控 AgenticSearchAgent（编排 Step0→3 全链路）
│
│   ── 配置层 ──
├── configs/
│   ├── __init__.py           聚合导出 STAGES / PROMPTS / render / config
│   ├── models_config.py      各 stage 的 provider / model / 参数
│   ├── prompts.py            所有 prompt 模板集中注册
│   └── config.py             非模型类配置（缓存 / 代理 / 搜索 / QA 后端）
│
│   ── LLM 调用层 ──
├── llm_client.py         ★ 统一 LLM 调用（ollama / OpenAI 兼容，流式+非流式+预热）
│
│   ── 业务 stage ──
├── tool_router.py        stage=router：是否调工具、调哪个（含失败降级契约）
├── query_rewriter.py     stage=rewriter：query 改写（规则/LLM/混合 + 历史污染检测）
├── context_provider.py   环境信息注入（当前时间 / 位置，含缓存与 single-flight）
├── searcher.py           联网检索（Tavily → DDG → Serper → Bing 兜底）
├── memory.py             会话记忆（滑动窗口，按 session 分桶）
├── qa_cache.py           Q&A 缓存（= RAG L1；精确 + BGE-M3 模糊匹配 + 槽位门禁）
│
│   ── 可靠性 / 安全 / 结构化返回 ──
├── cache_policy.py       ★ L1 准入策略（时效判定 / 分级 TTL / 槽位一致性门禁）
├── evidence.py           ★ 证据清洗 + <doc> 结构化定界（Prompt Injection 防护）
├── answer_types.py       ★ AnswerResult / Source / Citation（来源归因契约）
├── followup.py           ★ 追问推荐 + 流式分隔符抑制（澄清提问为预留特性）
│
│   ── 测试 ──
├── conftest.py           pytest 夹具：把缓存目录重定向到 tmp，杜绝测试污染生产数据
├── test_p0.py            可靠性 / 安全 / 归因 / 延迟 / 配额融合回归（170 项）
├── test_p2.py            去重+MMR / 追问推荐 / snippet 清洗回归（116 项）
├── test_qa_cache.py      L1 缓存回归（24 项）
│
│   ── 数据层 ──
├── data/                 ★ 所有本地落盘数据统一收纳于此（详见 data/README.md）
│   ├── rag_data/             RAG 知识库：L2 Wiki 索引 / L5 KG / L3 历史归档（已 gitignore）
│   ├── qa_cache/             L1 Q&A 缓存（diskcache）★ 被 git 追踪，自带一批预热问答
│   │   ├── cache.db              问答正文
│   │   ├── _embeddings/          BGE-M3 向量（fuzzy 命中用，1024 维）
│   │   └── _meta/                原始 query 原文（槽位门禁用）
│   └── search_cache/         联网搜索结果缓存（diskcache）★ 被 git 追踪
│
│   ── 工具与脚本 ──
├── tools/                专用工具（current_time / weather / github_repo / arxiv / web_search）
├── scripts/
│   ├── search.py             供外部 Skill 调用的命令行检索入口（输出 JSON）
│   └── clean_l3_refusals.py  清理 L3 里的拒答污染
├── SKILL.md              Skill 触发说明
│
│   ── 分层 RAG 栈 ──
└── rag/                  ★ 5 层记忆 + Router + 融合 + 去重 + 增量索引
    ├── README.md              分层设计与离线索引构建文档
    ├── retriever.py           对外主入口 LayeredRetriever
    ├── layers.py              L1–L5 五层实现
    ├── router.py              层激活策略（规则，可换 LLM）
    ├── fusion.py              RRF 融合 + 并列实体配额 + 可选二阶 rerank
    ├── entities.py            并列实体识别（jieba 词性 + 连接词）
    ├── dedup.py               近重去重 + MMR 多样性重排
    ├── calibration.py         跨层分数校准（Platt scaling + 噪声-OR 聚合）
    ├── answerability.py       证据可答性判据（实词覆盖率，与置信度正交）
    ├── textclean.py           L4 snippet 噪声清洗
    ├── embedder.py            统一 BGE-M3 编码适配器
    ├── vector_store.py        faiss + numpy 可插拔向量存储（L3）
    ├── incremental_worker.py  L3 后台增量写 worker
    ├── config.py / types.py   编排器配置 / Passage & RetrievalResult 契约
    ├── configs/default.yaml   wiki_rag 全部可调参数（L2/L5 路径等）
    ├── wiki_rag/              vendored 检索内核（WikiRetriever / KGRetriever）
    └── scripts/               离线构建流水线（01–12：wiki 索引 + Wikidata KG）
```

---

## 7. 常见操作

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
python scripts/search.py "2025 年 RAG 最新进展"          # 输出含 answer/sources/citations 的 JSON
python scripts/search.py "xxx" --rewrite-type 0 --no-answer
```

**E. 常用环境变量开关**

```bash
export RAG_ENABLE_NEAR_DUP=false        # 关近重去重
export RAG_ENABLE_MMR=false             # 关 MMR（省约 90ms 向量耗时）
export RAG_DEDUP_LAYERS=""              # 去重改为全量模式（更彻底但更慢）
export RAG_FUSION_CANDIDATE_MULT=2      # 候选池放大倍数 3 → 2
export RAG_ENABLE_SNIPPET_CLEAN=false   # 关 snippet 噪声清洗
export RAG_RERANK_STRATEGY=bge          # 启用二阶精排（rrf / bge / cascade / none）
export FOLLOWUP_MODE=off                # 关追问推荐
export LLM_KEEP_ALIVE=30m               # ollama 模型驻留时长（默认 -1 永不卸载）
export USER_CITY="北京"                  # 显式指定城市，跳过 IP 定位（零延迟）
```

完整列表见 `rag/config.py`、`configs/config.py` 与 [`rag/README.md`](rag/README.md) §7。

---

## 8. 扩展点

| 想做什么 | 改哪里 |
|---|---|
| 新增 stage | `configs/models_config.py` 的 `STAGES` 加 key + `configs/prompts.py` 的 `PROMPTS` 注册同名模板 |
| 新增 provider | `llm_client.py` 加 `_call_xxx` / `_stream_xxx`，在 `chat()` / `stream_chat()` 里分支 |
| 新增工具 | `tools/` 下新建模块，在 `tools/__init__.py` 的 `TOOLS` 里登记 |
| 新增 RAG 层 | `rag/layers.py` 实现 `search(query, top_k) -> list[Passage]`，在 `router.py` 登记，`calibration.py` 补一组校准参数 |
| 换 reranker | `export RAG_RERANK_STRATEGY=bge\|cascade`，见 `rag/README.md` |
| 接自己的前端 | 传 `on_event` 回调消费流水线事件，`return_result=True` 拿结构化结果 |
| 用真实数据重新校准 | `rag/calibration.fit_platt()` + `export RAG_CALIBRATION_FILE=calib.json` |

---

## 9. 测试

全部测试**不依赖外网、不依赖 GB 级离线索引、不调真实 LLM**（外部边界均被 mock），可在 CI 里稳定运行。

```bash
# 全量回归（310 项）
python -m pytest -q test_p0.py test_p2.py test_qa_cache.py

# 分文件
python -m pytest -q test_p0.py           # 可靠性/安全/归因/延迟/配额   170 项
python -m pytest -q test_p2.py           # 去重+MMR/追问/snippet 清洗   116 项
python -m pytest -q test_qa_cache.py     # L1 缓存（精确/模糊/多级/异步） 24 项

# 按主题跑（排查时更快）
python -m pytest -q test_p0.py -k "SlotGate or FocusSlot"    # L1 误命中
python -m pytest -q test_p0.py -k "Calibration"              # 跨层校准 / L4 兜底
python -m pytest -q test_p0.py -k "Answerability"            # 证据可答性
python -m pytest -q test_p0.py -k "PartialRefusal"           # 部分拒答
python -m pytest -q test_p0.py -k "TimeoutSoftAbandon"       # 超时软放弃
python -m pytest -q test_p0.py -k "LatencyObservability"     # 延迟与耗时归属
python -m pytest -q test_p0.py -k "EntityQuotaFusion"        # 并列实体配额融合
python -m pytest -q test_p2.py -k "NearDup or MMR"           # 近重去重 / 多样性
python -m pytest -q test_p2.py -k "StreamFilter"             # 流式分隔符抑制
```

> `conftest.py` 的 `autouse` 夹具会把 `QA_CACHE_DIR` 重定向到每个测试独有的 tmp 目录。
> 这道隔离很重要：在它加入之前，测试里的假编码器（3/4/8 维）会把脏向量写进
> **仓库里被追踪的** `data/qa_cache/`，既污染生产数据，又造成"单独跑通过、
> 连着跑失败"的顺序依赖，极难排查。

部分模块还带有 `python -m xxx` 可直接运行的自检 / 演示（`cache_policy.py`、`evidence.py`、`answer_types.py`、`followup.py`、`rag/calibration.py`），可用来直观查看各判据的行为。

---

## 10. 路线图

- [ ] **多跳搜索的精准实现**：当前仍倾向于单轮检索（+模糊搜索实现的多跳问答），计划增加 Controllable loop agent
- [ ] **多模态 / 富媒体搜索**：返回图片、表格、代码块；支持答案的图片来源与 Markdown 表格渲染
- [ ] **多 query 并发检索**：在实体配额之上的可选增强（实体识别能力已在 `rag/entities.py` 就绪）
- [ ] **LLM Router**：把 rule-based 层激活换成小模型分类
- [ ] **L3 老化**：给历史归档加 TTL 或 LFU，防止越攒越乱
- [ ] **Citation Binder**：span 级蕴含校验（当前已做编号有效性校验）
- [ ] **澄清提问上线**：`followup.should_clarify()` 已实现（基于证据分裂度而非歧义词表），默认关闭，待日志观测精度后开启

---

## 11. License

MIT — 学习交流与商业用途均可。
