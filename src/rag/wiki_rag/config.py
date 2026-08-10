"""配置加载器（已 vendored 进 agentic_search/rag）。

所有可调参数都集中在一份 YAML（``rag/configs/default.yaml``）里；wiki_rag
库模块和 rag/scripts 下的构建脚本都通过 :func:`load_config` 读取同一份配置，
行为在整条流水线上保持一致。

与原始 wiki_rag 的差异（集成到 agentic_search 后）
------------------------------------------------------
1. **默认配置路径**：指向 vendored 位置 ``rag/configs/default.yaml``。
2. **相对路径锚点**：YAML 里的 ``paths`` 全部写成以项目根为基准的相对路径
   （如 ``data/rag_data/wiki_zh_chunks.jsonl``）。这里加载时会把相对路径统一
   拼接到 **agentic_search 项目根目录**，保证无论从哪个 CWD 启动（agent、
   main、脚本）都能定位到同一份数据文件，满足 D3「路径全部指向
   agentic_search/data/rag_data」的要求。
3. **绝对路径原样保留**：若某项在 YAML 中写成绝对路径（或通过环境变量注入），
   则不做拼接，方便生产环境把大文件放到独立数据盘。
"""
from __future__ import annotations
from pathlib import Path
import yaml

# vendored 后的目录结构：
#   <PROJECT_ROOT>/src/rag/wiki_rag/config.py   <- 本文件
#   <PROJECT_ROOT>/src/rag/configs/default.yaml <- 默认配置
#   <PROJECT_ROOT>/data/rag_data/...            <- 数据文件（D3）
#
# ⚠️ 注意 `src/` 这一层：源码搬进 src/ 后，本文件到仓库根是**三级**，
#    而 YAML 里的 paths 是以**仓库根**为基准的相对路径（data/rag_data/...）。
#    少退一级会把它们解析成 <ROOT>/src/data/rag_data/，于是 L2 Wiki 索引与
#    L5 KG 数据库全部找不到 —— 而且这类失败会被 retriever 当成"索引未构建"
#    的正常降级吞掉（只打一行日志、该层返回空），不会报错，极难发现。
#
# __file__.parents:
#   parents[0] = src/rag/wiki_rag
#   parents[1] = src/rag
#   parents[2] = src
#   parents[3] = <PROJECT_ROOT>（仓库根）
_RAG_DIR = Path(__file__).resolve().parent.parent          # .../src/rag
PROJECT_ROOT = _RAG_DIR.parent.parent                      # .../alpha_agentic_search
DEFAULT_CFG = _RAG_DIR / "configs" / "default.yaml"        # 默认 YAML


def _anchor_path(value: str | Path) -> Path:
    """把 YAML 中的路径值解析成绝对 Path。

    规则：
        - 绝对路径 → 原样返回（生产环境可把大文件放数据盘）。
        - 相对路径 → 统一拼到 agentic_search 项目根，保证任意 CWD 一致。
    """
    p = Path(value)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def load_config(path: str | Path | None = None) -> dict:
    """加载 YAML 配置并做轻量后处理。

    Args:
        path: 可选覆盖。传 ``None`` 时使用包内默认配置
              （``rag/configs/default.yaml``）。

    Returns:
        与 YAML 结构一致的 ``dict``；``paths`` 下的每个值都会被转成
        锚定到项目根的绝对 :class:`pathlib.Path`。
    """
    path = Path(path) if path else DEFAULT_CFG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 把所有 paths 锚定到项目根（D3：统一指向 agentic_search/data/rag_data）
    cfg["paths"] = {k: _anchor_path(v) for k, v in cfg.get("paths", {}).items()}
    return cfg
