# data/ — 本地数据统一收纳目录

本目录集中存放 Alpha Agentic Search 运行期产生的**所有本地落盘数据**：
搜索缓存、L1 Q&A 缓存，以及 RAG 知识库（L2 Wiki / L5 KG / L3 历史归档）。

> **版本控制**：整个 `data/` 已在 `.gitignore` 中忽略，不纳入 git。
> 这些都是可再生成 / 可下载的运行期数据，不应提交到仓库。

---

## 目录结构

```
data/
├── search_cache/   联网搜索结果缓存（diskcache）
├── qa_cache/       L1 Q&A 缓存（diskcache）
└── rag_data/       RAG 知识库（L2 Wiki 索引 / L5 KG / L3 历史归档）
```

---

## 各子目录的作用

### 1. `search_cache/` — 联网搜索结果缓存
- **作用**：缓存 `web_search()` 的原始返回结果（DuckDuckGo / Tavily / Serper 等），
  在 TTL 内对相同 query 直接命中，避免重复请求搜索引擎、节省配额与延迟。
- **后端**：`diskcache`（`cache.db`）。
- **配置**：`configs/config.py` → `SEARCH_CACHE_DIR` / `SEARCH_CACHE_TTL`（默认 10 天）/ `SEARCH_CACHE_ENABLED`。
- **消费者**：`searcher.py`。
- **可否删除**：可随时删除，下次搜索会自动重建。

### 2. `qa_cache/` — L1 Q&A 缓存
- **作用**：RAG 第一层（L1）。缓存"高频问答 / 自我介绍 / 通用问题"的答案，
  命中后**绕过**工具路由 / Query 改写 / 联网检索 / LLM 调用，毫秒级返回。
  支持精确命中与 BGE-M3 向量**模糊命中**（`_embeddings/` 子目录存放向量）。
- **后端**：`diskcache`（`cache.db`）；也可切 `memory` / `redis`。
- **配置**：`configs/config.py` → `QA_CACHE_BACKEND` / `QA_CACHE_DIR` / `QA_CACHE_TTL`（默认 30 天）/ `QA_REDIS_URL`。
- **消费者**：`qa_cache.py`、`agent.py`。
- **可否删除**：可删除，仅丢失缓存；预设 Q&A 会在下次运行时重新写入。

### 3. `rag_data/` — RAG 知识库（大文件）
- **作用**：分层 RAG 栈 L2/L3/L5 的所有落盘数据根目录。
  - **L2 Commonsense**：Wikipedia dump 处理产物（chunks、BGE-M3 向量 `.npy`、FAISS 索引 `.faiss`）。
  - **L5 Knowledge Graph**：Wikidata truthy 处理产物（三元组 TSV、实体 JSONL、SQLite 库 `wikidata_zh_kg.db`、热门实体向量与索引）。
  - **L3 History**：`l3_history/` —— 每次成功回答后**异步增量归档**的会话向量索引（"越用越强"）。
- **配置**：
  - `rag/config.py` → `RAG_DATA_DIR`（默认 `<项目根>/data/rag_data`，可用环境变量 `RAG_DATA_DIR` 覆盖）。
  - L2/L5 各文件的具体路径在 `rag/configs/default.yaml` 的 `paths:` 段（均以 `data/rag_data/` 为前缀）。
- **消费者**：`rag/` 包（`layers.py` / `retriever.py` / `wiki_rag/`）、`rag/scripts/` 离线构建流水线。
- **构建方式**：见 [`rag/README.md`](../rag/README.md) §3.3。**未构建索引时 L2/L5 自然返回空，
  系统仍可正常用 L1/L3/L4 工作**。
- **可否删除**：L2/L5 删除后需重新跑离线构建（耗时较长）；`l3_history/` 删除仅丢失历史归档。

---

## 统一路径锚定 & 环境变量覆盖

所有路径均以**项目根**为锚点（无论从哪个 CWD 启动 `main.py` / `main_web.py` / `scripts/` 都能定位同一份数据）：

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `DATA_DIR` | `<项目根>/data` | 数据总根目录（一次性重定向到外置数据盘） |
| `SEARCH_CACHE_DIR` | `<DATA_DIR>/search_cache` | 单独重定向搜索缓存 |
| `QA_CACHE_DIR` | `<DATA_DIR>/qa_cache` | 单独重定向 L1 缓存 |
| `RAG_DATA_DIR` | `<DATA_DIR>/rag_data` | 单独重定向 RAG 知识库 |

> 优先级：各子项专用环境变量 > `DATA_DIR` > 代码默认值。
> 例如把所有数据挂到外置盘：`export DATA_DIR=/mnt/ssd/aas_data`。
