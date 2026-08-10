# memory.py
"""极简会话记忆模块：滑动窗口保留最近 N 轮对话。"""
from collections import deque


class ConversationMemory:
    """滑动窗口记忆。

    每一轮包含 user + assistant 两条消息，因此 deque 容量 = max_turns * 2。
    适合短期对话；如需长期记忆请替换为向量库（FAISS / Chroma）。
    """

    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        self.buffer: deque = deque(maxlen=max_turns * 2)

    def add_user(self, content: str) -> None:
        self.buffer.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.buffer.append({"role": "assistant", "content": content})

    def get_messages(self) -> list[dict]:
        """返回 OpenAI 兼容格式的消息列表。"""
        return list(self.buffer)

    def summarize_recent(self, n: int = 3) -> str:
        """把最近 n 轮对话拼成纯文本，供 query 改写器作为上下文。"""
        recent = list(self.buffer)[-n * 2:]
        if not recent:
            return ""
        return "\n".join(f"{m['role']}: {m['content']}" for m in recent)

    def clear(self) -> None:
        self.buffer.clear()
