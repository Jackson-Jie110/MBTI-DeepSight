from __future__ import annotations

import html
import os
from typing import AsyncIterator


def _sse_data(text: str) -> str:
    safe = html.escape(text)
    safe = safe.replace("\n", "<br/>")
    return f"data: <span>{safe}</span>\n\n"


def _missing_env_message() -> str | None:
    api_key = os.getenv("MBTI_AI_API_KEY")
    if not api_key:
        return "AI 服务未配置：缺少环境变量 MBTI_AI_API_KEY。"
    return None


async def generate_report_stream(user_type: str, insights: list[str]) -> AsyncIterator[str]:
    missing = _missing_env_message()
    if missing:
        print(missing)
        yield _sse_data(missing)
        return

    base_url = os.getenv("MBTI_AI_BASE_URL") or None
    api_key = os.getenv("MBTI_AI_API_KEY")
    model = os.getenv("MBTI_AI_MODEL", "gpt-3.5-turbo")

    try:
        from openai import AsyncOpenAI
    except Exception as e:  # pragma: no cover
        msg = f"AI 依赖不可用：{type(e).__name__}"
        print(msg)
        yield _sse_data(msg)
        return

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=30.0,
        max_retries=1,
    )

    insights_text = "\n".join(f"- {x}" for x in (insights or [])) or "-（暂无）"
    prompt = (
        "你是一位温暖、专业、有洞察力的心理咨询师。\n"
        f"来访者的 MBTI 类型是：{user_type}。\n"
        "以下是基于答题行为生成的动态洞察（可能不完整）：\n"
        f"{insights_text}\n\n"
        "请输出一段约 300 字的性格整合建议，要求：\n"
        "1) 先共情，再给出可执行建议；\n"
        "2) 避免贴标签式断言，强调情境与可成长；\n"
        "3) 用中文输出。\n"
    )

    yield _sse_data("正在生成 AI 深度分析…\n")

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一位专业心理咨询师。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            stream=True,
        )
        async for chunk in stream:
            delta = ""
            try:
                delta = chunk.choices[0].delta.content or ""
            except Exception:
                delta = ""
            if delta:
                yield _sse_data(delta)
    except Exception as e:
        msg = f"\n（AI 暂时不可用：{type(e).__name__}）"
        print(msg)
        yield _sse_data(msg)

