# configs/__init__.py
"""集中配置包：把三个配置模块聚合到一个清晰的命名空间下。

包含：
    - config          非模型类配置（缓存 / 代理 / 搜索 / QA 缓存后端 / 哨兵值等）
    - models_config   各 stage 的 provider / model / 采样参数
    - prompts         所有 prompt 模板集中注册

推荐引用方式（保持与拆分前一致，只是多一层包名）：
    from src.configs import config
    from src.configs.models_config import STAGES, get_stage_config
    from src.configs.prompts import PROMPTS, render
"""
from . import config
from . import models_config
from . import prompts

# 便捷再导出：允许 `from configs import STAGES, PROMPTS` 之类的用法
from .models_config import STAGES, get_stage_config
from .prompts import PROMPTS, render

__all__ = [
    "config",
    "models_config",
    "prompts",
    "STAGES",
    "get_stage_config",
    "PROMPTS",
    "render",
]
