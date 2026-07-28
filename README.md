# Alpha Agentic Search — 分层记忆 RAG 检索问答系统

## 说在前面的话:

自agent被提出以来，学术界有太多的search agent框架：从最开始的search r1，到后面各种用于刷榜的web search。各位同行大佬们的奇思妙想都给我们无尽的思考；但对工业界、AI搜初学者、以及想搭建个人搜索助手极客们，**层次化、稳定可控、耗时短**，永远是最基础的需求。

业界那么多AI搜索应用，真正火出圈的也就perplexity、秘塔等；一个很重要的点：如何让你的搜索越用越好用，搭建一套完整的多级检索和自动更新的harness框架，可能远比单个模型优化、或者高耗时容忍的multi-agent自进化更重要。

我们这里开源 Alpha Agentic Search (AAS) 框架，它主打的就是“稳定、可控、低耗时”！并在快手AI搜上面有了相应的落地。无论你是一名初学者，一位刚入行的AI工程师，还是一位资深AI搜索架构师，相信这套系统对您都有所启发。

- 作者个人主页：https://benchen4395.github.io/
- 如有问题或建议，欢迎沟通：benchen4395@gmail.com

---
## 简介:

> 一个可运行、可教学、也可作生产原型的 **Agentic Search** 项目：
> 用 `Route → Rewrite → Retrieve → Read` 的经典链路做联网问答，
> 并在检索侧接入了一套 **5 层记忆 RAG 栈（L1–L5）**，实现 Perplexity 式的 **“越用越强”**。
>
> 两大设计原则：
> 1. **多 Provider / 多 Stage**：每个阶段（路由 / 改写 / 回答）都能独立选模型、改 prompt，业务代码零硬编码。
> 2. **统一向量空间**：L1–L5 全部使用 FlagEmbedding **BGE-M3** 编码，跨层可比、可融合。

> 二期优化点：（未来一个月内）
> 1. **多跳搜索的精准实现**：当前依然倾向于单轮检索 （+模糊搜索实现的多跳问答）；后续会增加 Controllableloop agent；
> 2. **多模态 / 富媒体搜索**：未来考虑”返回图片、表格、代码块”多模态搜索；并支持答案的图片来源和Markdown 表格渲染。
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
                         ┌──────────────────────────────┐
   用户输入  ──────────► │ main.py (CLI) / main_web.py(Web)│
                         └───────────────┬──────────────┘
                                         ▼
                         ┌──────────────────────────────┐
                         │  agent.py  AgenticSearchAgent  │  主控编排
                         └───────────────┬──────────────┘
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
  "stage": "cache|router|tool|rewrite|retrieve|answer",  # 步骤类型
  "title": "分层 RAG 检索",                                # 人类可读标题
  "detail": "[L2:3, L3:1, L5:2], 融合 5 段",               # 步骤明细
  "elapsed_ms": 128,                                       # 距上一步的耗时（毫秒）
}
```

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

**默认已接入 Agent**，无需额外配置：

```python
from agent import AgenticSearchAgent
agent = AgenticSearchAgent()      # 内部自动挂载 LayeredRetriever
agent.chat("量子计算是什么")        # 走 L1→L5，成功后自动归档到 L3
agent.close()                     # 退出前 flush 归档队列
```

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
├── memory.py             会话记忆（滑动窗口）
├── qa_cache.py           Q&A 缓存（= RAG L1；精确 + BGE-M3 模糊匹配）
│
│   ── 数据层（data/，统一收纳，已 gitignore）──
├── data/                 ★ 所有本地落盘数据统一收纳于此
│   ├── rag_data/             RAG 知识库：L2 Wiki 索引 / L5 KG / L3 历史归档
│   ├── qa_cache/             L1 Q&A 缓存（diskcache）
│   └── search_cache/         联网搜索结果缓存（diskcache）
│
│   ── 工具与脚本 ──
├── tools/                专用工具（current_time / weather / github_repo / arxiv / web_search）
├── scripts/search.py     供外部 Skill 调用的命令行检索入口
├── SKILL.md              Skill 触发说明
│
│   ── 分层 RAG 栈 ──
└── rag/                  ★ 5 层记忆 + Router + RRF 融合 + 增量索引
    ├── README.md              分层设计与索引构建文档
    ├── retriever.py           对外主入口 LayeredRetriever
    ├── layers.py              L1–L5 五层实现
    ├── router.py / fusion.py  层激活策略 / RRF 融合 + 可选 rerank
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
python -m pytest -q test_qa_cache.py     # L1 QA 缓存（精确/模糊/多级/异步）24 项
```

## 9. License

MIT — 学习与交流、及商业用途 均可以。
