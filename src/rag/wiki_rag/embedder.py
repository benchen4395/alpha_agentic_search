"""BGE-M3 稠密编码器。

模块只暴露两个函数：

- :func:`get_model`  按设备做进程内单例，懒加载 :class:`FlagEmbedding.BGEM3FlagModel`。
- :func:`encode`     薄薄一层批量封装，返回已 L2 归一化的 ``float32`` 向量，
  可直接喂给 ``METRIC_INNER_PRODUCT`` 的 FAISS 索引（归一化后内积 = cosine）。

设计要点
--------

* **按设备做单例。** BGE-M3 加载一次要好几秒和 2GB 以上显存/内存，缓存
  实例只在请求的 ``device`` 变化时才重新构造。
* **跨版本兼容。** FlagEmbedding 1.2.x 用 ``device=<str>``，1.3.x 用
  ``devices=[<str>, ...]``；这里在运行时反射 ``__init__`` 签名挑对应的
  关键字，兜底再走 ``CUDA_VISIBLE_DEVICES``。这样同一份代码在任意版本
  的 FlagEmbedding 上都能跑起来。
* **输出显式声明。** 始终传 ``return_dense=True`` 并关掉稀疏 / ColBERT 头，
  一是避免不同版本默认值不一致踩坑，二是只算稠密向量本身也更省算力。
"""
from __future__ import annotations
import os
import inspect
import numpy as np
from FlagEmbedding import BGEM3FlagModel

_MODEL: BGEM3FlagModel | None = None
_MODEL_DEVICE: str | None = None


def _maybe_enable_offline(model_name: str) -> str:
    """本地已缓存该模型时，返回**本地快照目录绝对路径**并切到 HF 离线模式。

    背景（为什么"检测到缓存"却仍在下载）
    --------------------------------------
    1. **环境变量设置时机太晚**：``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``
       会在 ``import huggingface_hub / transformers`` 的那一刻被读进模块常量并
       **冻结**；等到运行时（get_model 里）再设置已经无效。而 FlagEmbedding
       在本模块顶部 import 时就已把这些库拉起来了。
    2. **默认优先 safetensors**：BGE-M3 官方权重同时有 ``pytorch_model.bin`` 和
       ``model.safetensors``；若只把 model 名（``BAAI/bge-m3``）交给库去解析，
       在线状态下它会尝试补齐缺失的 ``model.safetensors``（2.11GB），于是又开始下载。

    解决办法
    --------
    探测到本地缓存完整时，**直接把本地 snapshot 目录的绝对路径**返回给调用方，
    交给 ``BGEM3FlagModel(<本地目录>)`` 加载。传入一个已存在的本地目录时，
    transformers 不会发起任何联网请求，并会自动 fallback 到 ``pytorch_model.bin``
    （无需 safetensors）。这样彻底绕开"环境变量设置太晚"的时机问题。

    同时仍顺手设置 ``HF_HUB_OFFLINE`` 作为双保险（对运行时才触发的少量检查有效）。

    Args:
        model_name: HF repo id（如 ``BAAI/bge-m3``）。
    Returns:
        本地 snapshot 目录绝对路径（缓存完整时）；否则原样返回 ``model_name``
        （首次运行需联网下载）。可用 ``RAG_EMBED_FORCE_ONLINE=1`` 强制在线。
    """
    if os.getenv("RAG_EMBED_FORCE_ONLINE") == "1":
        return model_name
    # 已是本地路径（用户直接传目录）就别折腾了
    if os.path.isdir(model_name):
        return model_name

    local_dir: str | None = None
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(model_name, "config.json")
        if isinstance(cached, str) and os.path.isfile(cached):
            # cached 是 config.json 的真实路径，其所在目录即该快照根目录
            local_dir = os.path.dirname(cached)
    except Exception:
        # huggingface_hub 版本差异导致探测失败时，退回目录名启发式
        hub = (os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface")) + "/hub"
        safe = "models--" + model_name.replace("/", "--")
        snap_root = os.path.join(hub, safe, "snapshots")
        if os.path.isdir(snap_root):
            subs = [os.path.join(snap_root, d) for d in os.listdir(snap_root)]
            subs = [d for d in subs if os.path.isfile(os.path.join(d, "config.json"))]
            if subs:
                local_dir = subs[0]

    if local_dir:
        # 双保险：设离线环境变量（对运行时才读取的检查仍有效）
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        print(f"[embedder] 检测到本地缓存，直接从本地目录离线加载 {model_name}"
              "（如需联网更新模型：export RAG_EMBED_FORCE_ONLINE=1）")
        return local_dir

    # 未命中缓存：保持在线，走正常下载流程
    return model_name


def _resolve_device(device: str = "auto") -> str:
    """把 ``"auto"`` 解析成具体设备字符串。

    优先级：CUDA → Apple MPS → CPU。显式传入的 ``"cuda"``、``"cuda:0"``、
    ``"mps"``、``"cpu"`` 原样返回。
    """
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_model(model_name: str = "BAAI/bge-m3",
              use_fp16: bool = True,
              device: str = "auto") -> BGEM3FlagModel:
    """返回进程内单例 BGE-M3。

    多次调用会复用同一个实例；只有 ``device`` 变了才会重新加载。
    """
    global _MODEL, _MODEL_DEVICE
    dev = _resolve_device(device)

    if _MODEL is not None and _MODEL_DEVICE == dev:
        return _MODEL

    # 本地已缓存则拿到本地快照目录绝对路径，直接离线加载，彻底规避联网下载
    load_target = _maybe_enable_offline(model_name)

    print(f"[embedder] loading {model_name} (fp16={use_fp16}, device={dev}) ...")

    # 按当前安装的 FlagEmbedding 版本挑关键字：devices=[...] / device=... / 走环境变量。
    init_params = inspect.signature(BGEM3FlagModel.__init__).parameters
    kwargs = {"use_fp16": use_fp16}
    if dev != "cpu":
        if "devices" in init_params:            # FlagEmbedding 1.3.x
            kwargs["devices"] = [dev]
        elif "device" in init_params:           # FlagEmbedding 1.2.x
            kwargs["device"] = dev
        else:
            # 两个关键字都没有，只能通过环境变量限制 GPU 可见性。
            if dev.startswith("cuda"):
                idx = dev.split(":", 1)[1] if ":" in dev else "0"
                os.environ["CUDA_VISIBLE_DEVICES"] = idx

    _MODEL = BGEM3FlagModel(load_target, **kwargs)
    _MODEL_DEVICE = dev
    return _MODEL


def encode(texts: list[str],
           model_name: str = "BAAI/bge-m3",
           batch_size: int = 64,
           max_length: int = 512,
           device: str = "auto",
           normalize: bool = True) -> np.ndarray:
    """把一批文本编码成 ``(N, dim)`` 的 ``float32`` 矩阵。

    默认做 L2 归一化，这样 FAISS 用内积搜索就是精确的 cosine 相似度。
    """
    model = get_model(model_name, device=device)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )["dense_vecs"]
    vecs = np.asarray(vecs, dtype="float32")
    if normalize:
        vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    return vecs
