"""段落切分与滑窗 chunk。

模块只暴露两个函数，职责很窄：

- :func:`split_paragraphs`  把原始正文按换行切成段落列表，丢掉过短的行。
- :func:`chunk_paragraphs`  把段落用换行拼回去，再用固定大小的滑动窗口切 chunk。

两条设计原则：

1. **保留段落边界。** 在切 chunk 前用 ``"\\n"`` 拼接，保证 chunk 内部仍
   能看到段落分隔，下游模型对结构感知更好。
2. **chunk 长度可控。** 滑窗保证每个 chunk 不超过 ``chunk_size`` 个字符，
   编码 batch 的形状可预测，显存好估算。
"""
from typing import List


def split_paragraphs(text: str, min_para_len: int = 10) -> List[str]:
    """把正文切成段落，只保留长度达标的那些。

    Args:
        text: 文章正文。
        min_para_len: 段落最少字符数；低于此阈值的短行通常是导航/元信息噪声，直接丢弃。
    """
    return [p.strip() for p in text.split("\n") if len(p.strip()) >= min_para_len]


def chunk_paragraphs(paras: List[str],
                     chunk_size: int = 512,
                     overlap: int = 64) -> List[str]:
    """把段落列表切成带 overlap 的定长 chunk。

    Args:
        paras: :func:`split_paragraphs` 的输出。
        chunk_size: 每个 chunk 的最大字符数。
        overlap: 相邻 chunk 之间共享的字符数，用来给边界处保留上下文冗余。

    Returns:
        chunk 字符串列表；输入为空时返回 ``[]``。
    """
    full = "\n".join(paras)
    if not full:
        return []
    if len(full) <= chunk_size:
        return [full]

    chunks, start = [], 0
    step = chunk_size - overlap
    while start < len(full):
        end = min(start + chunk_size, len(full))
        chunks.append(full[start:end])
        if end == len(full):
            break
        start += step
    return chunks
