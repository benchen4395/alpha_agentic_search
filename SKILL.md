---
name: agentic-search
description: 当用户提问需要联网检索（最新新闻、技术资料、人物事件等）或明确说"搜一下/查一下/帮我找…"时触发。本技能会自动改写 query 并通过 DuckDuckGo 检索，再用 DeepSeek 综合回答，引用来源序号。
---

# Agentic Search Skill

## 触发条件
- 用户问题涉及外部最新信息（"今年/最近/2025"等时间词）
- 用户明确说："搜一下"、"查一下"、"帮我找资料"、"研究一下…"
- 问题涉及具体人物、产品、论文等外部知识

## 不触发
- 闲聊、问候、追问当前对话内容
- 纯代码/算法推导（无需联网）

## 执行步骤
1. 调用 `python scripts/search.py "<原始 query>"`
2. 解析返回 JSON，把 `answer` 给用户，并附上 `results` 中的 URL 作为引用

## 示例
> 用户：帮我查一下 2025 年 RAG 的最新进展
>
> Agent → `python scripts/search.py "2025 年 RAG 最新进展"`
>
> 输出含 rewritten/results/answer 的 JSON
