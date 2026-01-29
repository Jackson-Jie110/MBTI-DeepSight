from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.models import Answer


_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "type_reports.json"


@lru_cache(maxsize=1)
def _load_type_reports() -> dict[str, Any]:
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def _format_dimension_line(dim: str, info: dict[str, Any]) -> str:
    first = info.get("first_pole")
    second = info.get("second_pole")
    fp = info.get("first_percent")
    sp = info.get("second_percent")
    if first and second and fp is not None and sp is not None:
        stronger = second if int(sp) > int(fp) else first
        stronger_p = max(int(fp), int(sp))
        return f"- {dim}：更偏向 {stronger}（{stronger_p}%）"
    return f"- {dim}：数据不足"


def build_report(type_code: str, dimensions: dict[str, Any], *, boundary_notes: list[str]) -> str:
    reports = _load_type_reports()
    info = reports.get(type_code, {})

    title = info.get("title", "类型分析")
    summary = info.get("summary", "这是一个基于题目作答倾向的性格偏好分析结果。")
    strengths = info.get("strengths", [])
    blind_spots = info.get("blind_spots", [])
    advice = info.get("advice", [])
    suitable = info.get("suitable", [])

    lines: list[str] = []
    lines.append(f"# 你的类型：{type_code}（{title}）")
    lines.append("")
    lines.append("## 概览")
    lines.append(summary)
    lines.append("")
    lines.append("## 四维倾向")
    for dim in ["EI", "SN", "TF", "JP"]:
        if dim in dimensions:
            lines.append(_format_dimension_line(dim, dimensions[dim]))
    lines.append("")

    if boundary_notes:
        lines.append("## 边界提示")
        for note in boundary_notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## 优势")
    for s in strengths:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("## 盲点")
    for s in blind_spots:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("## 建议")
    for s in advice:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("## 适合方向")
    for s in suitable:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("## 说明")
    lines.append("本测试仅用于自我了解与娱乐参考，不构成专业心理评估或诊断。")
    lines.append("")
    return "\n".join(lines)


def generate_dynamic_insights(dimensions: dict, answers: list[Answer]) -> list[str]:
    insights: list[str] = []

    # 维度强度检测
    for dim, info in (dimensions or {}).items():
        try:
            gap = int(info.get("gap_percent"))
        except Exception:
            continue

        if gap < 20:
            insights.append(f"你在 {dim} 上表现得非常均衡灵活：在不同情境下能切换策略，不容易被单一偏好束缚。")
        elif gap > 60:
            insights.append(f"你在 {dim} 上表现出极致的倾向：你的决策与能量分配更偏向固定路径，优势突出也更需要留意盲点。")

    # 核心价值观提取：从 5 分回答里抽取 2-3 条
    fives = [
        a
        for a in (answers or [])
        if getattr(a, "value", None) == 5 and getattr(getattr(a, "question", None), "text", None)
    ]
    if fives:
        rng = random.Random(0)
        k = 3 if len(fives) >= 3 else len(fives)
        picked = rng.sample(fives, k=k)
        texts = [str(a.question.text).strip() for a in picked if a.question and a.question.text]
        if texts:
            insights.append("你对以下观点持有坚定立场：" + "；".join(texts))

    # 反差探测：整体倾向 vs 反向高分
    final_pole: dict[str, str] = {}
    for dim, info in (dimensions or {}).items():
        first = info.get("first_pole")
        second = info.get("second_pole")
        fp = info.get("first_percent")
        sp = info.get("second_percent")
        if not (first and second and fp is not None and sp is not None):
            continue
        try:
            fp_i = int(fp)
            sp_i = int(sp)
        except Exception:
            continue
        final_pole[str(dim)] = str(first) if fp_i >= sp_i else str(second)

    contrast = None
    for a in (answers or []):
        q = getattr(a, "question", None)
        if not q or getattr(a, "value", None) != 5:
            continue
        dim = getattr(q, "dimension", None)
        agree = getattr(q, "agree_pole", None)
        if not dim or not agree:
            continue
        overall = final_pole.get(str(dim))
        if overall and str(agree) != str(overall):
            contrast = (str(dim), str(overall), str(agree), str(getattr(q, "text", "")).strip())
            break

    if contrast:
        dim, overall, opposite, text = contrast
        snippet = text[:36] + ("…" if len(text) > 36 else "")
        insights.append(
            f"尽管你整体偏向 {overall}（{dim}），但在「{snippet}」上也展现了 {opposite} 的一面：这意味着你会在特定场景下反向发力。"
        )

    return insights


def build_report_context(
    type_code: str,
    dimensions: dict[str, Any],
    *,
    boundary_notes: list[str],
    answers: list[Answer] | None = None,
) -> dict[str, Any]:
    reports = _load_type_reports()
    info = reports.get(type_code, {})

    title = info.get("title", "类型分析")
    summary = info.get("summary", "这是一个基于题目作答倾向的性格偏好分析结果。")
    strengths = list(info.get("strengths", []))
    blind_spots = list(info.get("blind_spots", []))
    advice = list(info.get("advice", []))
    suitable = list(info.get("suitable", []))

    dim_items: list[dict[str, Any]] = []
    for dim in ["EI", "SN", "TF", "JP"]:
        d = dimensions.get(dim)
        if not d:
            continue
        first = d.get("first_pole")
        second = d.get("second_pole")
        fp = d.get("first_percent")
        sp = d.get("second_percent")
        if first and second and fp is not None and sp is not None:
            dim_items.append(
                {
                    "dimension": dim,
                    "first_pole": first,
                    "second_pole": second,
                    "first_percent": int(fp),
                    "second_percent": int(sp),
                }
            )

    return {
        "type_code": type_code,
        "title": title,
        "summary": summary,
        "dimensions": dim_items,
        "boundary_notes": list(boundary_notes),
        "insights": generate_dynamic_insights(dimensions, list(answers or [])),
        "strengths": strengths,
        "blind_spots": blind_spots,
        "advice": advice,
        "suitable": suitable,
        "disclaimer": "本测试仅用于自我了解与娱乐参考，不构成专业心理评估或诊断。",
    }
